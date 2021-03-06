from typing import List
from typing import Tuple

import numpy as np
import torch
from mmdet.apis.inference import LoadImage
from mmdet.core import bbox2result
from mmdet.datasets.pipelines import Compose

import topi.testing
import tvm
from custom_ops.load_custom_ops import load_custom_ops
from tvm import relay
from tvm import te

_multibox_detection_implement = {
  "generic": (topi.vision.ssd.multibox_detection, topi.generic.schedule_multibox_detection),
}


def convert_multibox_detect():
  def _impl(inputs, input_types):
    ml_cls_score = inputs[0]
    ml_bbox_pred = inputs[1]
    ml_anchors = inputs[2]
    inter_out = relay.op.vision.multibox_transform_loc(ml_cls_score, ml_bbox_pred, ml_anchors, threshold=0.02,
                                                       clip=False)
    out = relay.op.vision.non_max_suppression(inter_out[0], inter_out[1], inter_out[1], top_k=200, iou_threshold=0.45,
                                              return_indices=False)
    return out

  return _impl


SSD_CUSTOM_MAP = {'custom_ops::dummy_multibox_detect': convert_multibox_detect()}


class TraceableSsdModule(torch.nn.Module):
  SSD_WIDTH = 300.0
  SSD_HEIGHT = 300.0
  NUM_LEVELS = 6
  NUM_CLASSES = 80

  def __init__(self, backbone, bbox_head, cfg):
    super(TraceableSsdModule, self).__init__()
    load_custom_ops()
    self.backbone = backbone
    self.bbox_head = bbox_head
    self.anchors = None  # to be created
    self.cfg = cfg
    self.use_sigmoid_cls = False

  def get_scale_factor(self, width: int, height: int) -> List[float]:
    """Compute scale factor.

    :param width: Original image width
    :param height: Original image height
    :return: A list representing scale_factor in mmdetection
    """
    return [self.SSD_WIDTH / width, self.SSD_HEIGHT / height, self.SSD_WIDTH / width, self.SSD_HEIGHT / height]

  def forward(self, x) -> Tuple[torch.Tensor]:
    """Used for tracing.

    :param x: A torch tensor from a single image, expected dimention: [1, 3, 300, 300]
    :return: A tuple of size 2*N, where the first N represents cls_scores and the second N represents bbox_preds.
    """
    x = self.backbone(x)
    cls_scores, bbox_preds = self.bbox_head(x)
    anchors = self.anchors
    if anchors is None:
      raise RuntimeError("Run create_anchors before tracing.")
    ml_cls_score, ml_bbox_pred, ml_anchors = self.convert_multi_level_output_to_tvm_format(cls_scores,
                                                                                           bbox_preds,
                                                                                           anchors)
    output = torch.ops.custom_ops.dummy_multibox_detect(ml_cls_score, ml_bbox_pred, ml_anchors)
    return output

  def postprocess(self, output_tuple, ori_img_shape):
    """Convert tvm output tuple to bounding box predictions.

    :param output_tuple: Output tuple from a "forward" run.
    :param ori_img_shape: Original image size for scaling.
    :return: Bounding box return.
    """
    cls_scores = output_tuple[0:6]
    bbox_preds = output_tuple[6:]
    img_metas = [{'img_shape': (ori_img_shape[0], ori_img_shape[1], 3),
                  'scale_factor': self.get_scale_factor(ori_img_shape[0], ori_img_shape[1]),
                  'flip': False}]

    bbox_list = self.bbox_head.get_bboxes(
      cls_scores, bbox_preds, img_metas, rescale=True)
    bbox_results = [
      bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
      for det_bboxes, det_labels in bbox_list
    ]
    return bbox_results[0]

  def preprocess(self, img):
    """Use mmdetection utilities to preprocess image.

    :param img: A numpy array for test image
    :return: Preprocessed image
    """
    test_pipeline = [LoadImage()] + self.cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)

    return data['img'][0]

  def create_anchors(self, x):
    """Create anchors by analyzing the shape of bbox_head outputs.

    :param x: input tensor
    :return:
    """
    x = self.backbone(x)
    cls_scores, _ = self.bbox_head(x)
    feat_map_size = [cls_scores[i].shape[-2:] for i in range(6)]
    anchors = self.bbox_head.anchor_generator.grid_anchors(feat_map_size)

    return anchors

  def convert_multi_level_output_to_tvm_format(self, cls_scores, bbox_preds, anchors):
    multi_level_cls_probs = []
    multi_level_loc_preds = []
    multi_level_anchors = []

    for level in range(6):
      cls_score = cls_scores[level]
      bbox_pred = bbox_preds[level]
      anchor = anchors[level]
      np_cls_prob, np_loc_preds, np_anchors = self.convert_output_to_tvm_format(cls_score, bbox_pred, anchor)

      multi_level_cls_probs.append(np_cls_prob)
      multi_level_loc_preds.append(np_loc_preds)
      multi_level_anchors.append(np_anchors)

    multi_level_cls_probs = torch.cat(multi_level_cls_probs, axis=2)
    multi_level_loc_preds = torch.cat(multi_level_loc_preds, axis=1)
    multi_level_anchors = torch.cat(multi_level_anchors, axis=1)

    return multi_level_cls_probs, multi_level_loc_preds, multi_level_anchors

  def convert_output_to_tvm_format(self, cls_score, bbox_pred, anchors):
    """Convert cls_score, bbox_pred and anchor to TVM compatible format.

    :param cls_score: cls_score generated by the bbox_head
    :param bbox_pred: bbox_pred generated by the bbox_head
    :param anchor: generated by bbox_head.anchor_generator.grid_anchors
    :return: Transformed cls_score, bbox_pred and anchor that matches with TVM's SSD implementation.
    """
    # ------------------------------------
    # mmdetection-specific transformation
    # ------------------------------------
    # cls_score: [Fw*Fh, num_classes+1] then followed by softmax
    # bbox_pred: [Fw*Fh, 4], where the coordinates are [x0, y0, w, h]
    # Fw and Fh are the width and height of feature map
    cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.NUM_CLASSES + 1)
    cls_score = cls_score.softmax(-1)
    bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

    # ------------------------------------
    # transform to TVM format
    # ------------------------------------
    # mmdet assumes background class to be the rightmost index while tvm assume it is in the first index.
    # We will need to roll the cls_score in dimension 1. However, TVM does not support torch.roll yet, will use
    # split to get the same result.
    cls_score_main, cls_score_bg = torch.split(cls_score, [self.NUM_CLASSES, 1], dim=1)
    cls_score = torch.cat((cls_score_bg, cls_score_main), dim=1)

    # expected format in TVM:
    # cls_score: [batch, num_classes, num_anchors]
    # bbox_pred: [batch, num_anchors * 4]
    # anchor: [batch, num_anchors, 4]
    cls_score = cls_score.unsqueeze(0)
    cls_score = cls_score.transpose(1, 2)

    bbox_pred = bbox_pred.reshape(-1)
    bbox_pred = bbox_pred.unsqueeze(0)
    anchors = anchors.unsqueeze(0)

    return cls_score, bbox_pred, anchors

  def run_tvm_multibox_detection(self, np_cls_probs, np_loc_preds, np_anchors):
    num_anchors = np_anchors.shape[1]

    # create placeholder
    cls_probs_placeholder = te.placeholder((1, self.NUM_CLASSES + 1, num_anchors), name="cls_prob")
    loc_preds_placeholder = te.placeholder((1, num_anchors * 4), name="loc_preds")
    anchors_placeholder = te.placeholder((1, num_anchors, 4), name="anchors")

    # schedule computations
    ctx = tvm.context('llvm', 0)
    fcompute, fschedule = topi.testing.dispatch("llvm", _multibox_detection_implement)
    with tvm.target.create("llvm"):
      # hard coded SSD parameters for now.
      out = fcompute(cls_probs_placeholder, loc_preds_placeholder, anchors_placeholder, threshold=0.02,
                     nms_threshold=0.45, clip=False, nms_topk=200)
      s = fschedule(out)

    # generate computation graph
    tvm_cls_probs = tvm.nd.array(np_cls_probs.astype(cls_probs_placeholder.dtype), ctx)
    tvm_loc_preds = tvm.nd.array(np_loc_preds.astype(loc_preds_placeholder.dtype), ctx)
    tvm_anchors = tvm.nd.array(np_anchors.astype(anchors_placeholder.dtype), ctx)
    f = tvm.build(s, [cls_probs_placeholder, loc_preds_placeholder, anchors_placeholder, out], "llvm")

    # execute and output
    tvm_out = tvm.nd.array(np.zeros((1, num_anchors, 6)).astype(out.dtype), ctx)
    f(tvm_cls_probs, tvm_loc_preds, tvm_anchors, tvm_out)

    return tvm_out.asnumpy()
