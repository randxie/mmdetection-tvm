from typing import List
from typing import Tuple

import torch
from mmdet.apis.inference import LoadImage
from mmdet.core import bbox2result
from mmdet.datasets.pipelines import Compose
from mmdet.models.detectors import SingleStageDetector


class TraceableSsdModule(torch.nn.Module):
  SSD_WIDTH = 300.0
  SSD_HEIGHT = 300.0

  def __init__(self, ssd_model: SingleStageDetector):
    super(TraceableSsdModule, self).__init__()
    self.backbone = ssd_model.backbone
    self.bbox_head = ssd_model.bbox_head
    self.cfg = ssd_model.cfg

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
    return tuple(cls_scores + bbox_preds)

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
