import os

import numpy as np
import torch
from mmdet.apis import inference_detector
from mmdet.apis import init_detector
from mmdet.apis import show_result_pyplot

import tvm
from modules.traceable_ssd_module import SSD_CUSTOM_MAP
from modules.traceable_ssd_module import TraceableSsdModule
from tvm import relay
from tvm.contrib import graph_runtime
from utils import ROOT_DIR

CONFIG_FILE = os.path.join(ROOT_DIR, 'configs/ssd300_coco.py')

# download the checkpoint from model zoo and put it in `checkpoints/`
CHECKPOINT_FILE = os.path.join(ROOT_DIR, 'checkpoints/ssd300_coco_20200307-a92d2092.pth')

# test image
TEST_IMAGE_FILE = os.path.join(ROOT_DIR, 'test_images/demo.jpg')

visualize = True

# ---------------------------------------------
# Load mmdetection model
# ---------------------------------------------
mmdet_model = init_detector(CONFIG_FILE, CHECKPOINT_FILE).cpu()
mmdet_model.eval()

if visualize:
  mmdet_result = inference_detector(mmdet_model, TEST_IMAGE_FILE)
  show_result_pyplot(mmdet_model, TEST_IMAGE_FILE, mmdet_result)

# ---------------------------------------------
# Create traceable model from mmdetection model
# ---------------------------------------------
traceable_ssd_model = TraceableSsdModule(mmdet_model.backbone, mmdet_model.bbox_head, mmdet_model.cfg)

# trace pytorch modules.
input_shape = [1, 3, 300, 300]
random_input = torch.randn(input_shape, dtype=torch.float32)
traceable_ssd_model.anchors = traceable_ssd_model.create_anchors(random_input)
scripted_ssd_model = torch.jit.trace_module(traceable_ssd_model, {"forward": random_input})

# ---------------------------------------------
# Convert traced SSD model to TVM relay IR
# ---------------------------------------------
input_name = 'input0'
shape_list = [(input_name, (1, 3, 300, 300))]
ssd_module, params = relay.frontend.from_pytorch(scripted_ssd_model,
                                                 shape_list, custom_convert_map=SSD_CUSTOM_MAP)

with torch.no_grad():
  torch_output = scripted_ssd_model(random_input)

# ---------------------------------------------
# Build relay graph
# ---------------------------------------------
target = 'llvm'
target_host = 'llvm'
ctx = tvm.cpu(0)
with relay.build_config(opt_level=3):
  graph, lib, params = relay.build(ssd_module,
                                   target=target,
                                   target_host=target_host,
                                   params=params)

# ------------------------------------------------------
# Build relay graph and compare result with traced model
# ------------------------------------------------------
# execute graph using tvm runtime.
m = graph_runtime.create(graph, lib, ctx)
m.set_input(input_name, tvm.nd.array(random_input.numpy()))
m.set_input(**params)
m.run()
tvm_output = m.get_output(0)

"""
assert m.get_num_outputs() == len(torch_output), "Torch output should have the same shape as tvm output."
assert np.all(np.isclose(torch_output[0].detach().numpy(), tvm_output.asnumpy(),
                         atol=1e-5)), "Torch output should be numerically close to tvm output."
"""

# ------------------------------------------------------
# Test on real images
# ------------------------------------------------------
preprocessed_demo_img = traceable_ssd_model.preprocess(TEST_IMAGE_FILE)

m = graph_runtime.create(graph, lib, ctx)
m.set_input(input_name, tvm.nd.array(preprocessed_demo_img.numpy()))
m.set_input(**params)
m.run()
tvm_outputs = []
for i in range(m.get_num_outputs()):
  tvm_outputs.append(torch.from_numpy(m.get_output(i).asnumpy()))

outputs = tvm_outputs[0]

# extract valid detections
# [label, probability, x0, y0, x1, y1]
outputs = outputs[:, outputs[0, :, 1] > 0, :]

# use mmdetection utils to validate plots.
results = [np.empty(shape=(0, 5), dtype=np.float32) for _ in range(80)]
outputs = outputs[0, :, :]

# rescale to the original image size
scale_factor = traceable_ssd_model.get_scale_factor(640, 427)
outputs[:, 2] = outputs[:, 2] / scale_factor[0]
outputs[:, 3] = outputs[:, 3] / scale_factor[1]
outputs[:, 4] = outputs[:, 4] / scale_factor[2]
outputs[:, 5] = outputs[:, 5] / scale_factor[3]

for idx in range(outputs.shape[0]):
  tmp = results[int(outputs[idx, 0])]
  # mmdetection assumes the probability is the last dimension
  results[int(outputs[idx, 0])] = np.vstack((tmp, torch.roll(outputs[[idx], 1:], -1, 1)))

show_result_pyplot(mmdet_model, TEST_IMAGE_FILE, results)
