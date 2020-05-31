import os

import numpy as np
import torch
from mmdet.apis import init_detector

import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from utils import ROOT_DIR

config_file = os.path.join(ROOT_DIR, 'configs/ssd300_coco.py')

# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = os.path.join(ROOT_DIR, 'checkpoints/ssd300_coco_20200307-a92d2092.pth')

# firstly try to convert the backbone.
model = init_detector(config_file, checkpoint_file).cpu()
model.eval()
print(model)


class TraceableSsdModule(torch.nn.Module):
  def __init__(self, backbone, bbox_head):
    super(TraceableSsdModule, self).__init__()
    self.backbone = backbone
    self.bbox_head = bbox_head

  def forward(self, x):
    x = self.backbone(x)
    cls_scores, bbox_preds = self.bbox_head(x)
    return tuple(cls_scores + bbox_preds)


# trace pytorch modules.
input_shape = [1, 3, 300, 300]
input_data = torch.randn(input_shape, dtype=torch.float32)
scripted_model = torch.jit.trace_module(TraceableSsdModule(model.backbone, model.bbox_head),
                                        {"forward": input_data})

with torch.no_grad():
  torch_output = scripted_model(input_data)

# convert from pytorch to relay IR.
input_name = 'input0'
shape_list = [(input_name, (1, 3, 300, 300))]
mod, params = relay.frontend.from_pytorch(scripted_model,
                                          shape_list)

# build relay graph.
target = 'llvm'
target_host = 'llvm'
ctx = tvm.cpu(0)
with relay.build_config(opt_level=3):
  graph, lib, params = relay.build(mod,
                                   target=target,
                                   target_host=target_host,
                                   params=params)

# execute graph using tvm runtime.
m = graph_runtime.create(graph, lib, ctx)
m.set_input(input_name, tvm.nd.array(input_data.numpy()))
m.set_input(**params)
m.run()
tvm_output = m.get_output(0)

assert m.get_num_outputs() == len(torch_output), "Torch output should have the same shape as tvm output."
assert np.all(np.isclose(torch_output[0].detach().numpy(), tvm_output.asnumpy(),
                         atol=1e-6)), "Torch output should be numerically close to tvm output."
