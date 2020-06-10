import os

import torch
from mmdet.apis import inference_detector
from mmdet.apis import init_detector

from modules.traceable_ssd_module import TraceableSsdModule
from utils import ROOT_DIR

CONFIG_FILE = os.path.join(ROOT_DIR, 'configs/ssd300_coco.py')

# download the checkpoint from model zoo and put it in `checkpoints/`
CHECKPOINT_FILE = os.path.join(ROOT_DIR, 'checkpoints/ssd300_coco_20200307-a92d2092.pth')

# test image
TEST_IMAGE_FILE = os.path.join(ROOT_DIR, 'test_images/demo.jpg')

# ---------------------------------------------
# Load mmdetection model
# ---------------------------------------------
mmdet_model = init_detector(CONFIG_FILE, CHECKPOINT_FILE).cpu()
mmdet_model.eval()

result = inference_detector(mmdet_model, TEST_IMAGE_FILE)

# ---------------------------------------------
# Create traceable model from mmdetection model
# ---------------------------------------------
traceable_ssd_model = TraceableSsdModule(mmdet_model)
input_shape = [1, 3, 300, 300]
random_input = torch.randn(input_shape, dtype=torch.float32)
scripted_ssd_model = torch.jit.trace_module(traceable_ssd_model, {"forward": random_input})

# load preprocessed image
preprocessed_demo_img = traceable_ssd_model.preprocess(TEST_IMAGE_FILE).unsqueeze(0)
output_tuple = scripted_ssd_model.forward(preprocessed_demo_img)

# create multi-level predictions
ml_cls_probs, ml_loc_preds, ml_anchors = traceable_ssd_model.convert_multi_level_output_to_tvm_format(output_tuple)

# use TVM to do postprocessing
outputs = traceable_ssd_model.run_tvm_multibox_detection(ml_cls_probs, ml_loc_preds, ml_anchors)

# extract valid detections
outputs = outputs[:, outputs[0, :, 1] > 0, :]

# [label, probability, x0, y0, x1, y1]
print(outputs)
