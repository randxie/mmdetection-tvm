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
backbone = model.backbone

# trace pytorch modules.
input_shape = [1, 3, 300, 300]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace_module(backbone, {"forward": input_data})
print(scripted_model)

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
m.set_input(input_name, tvm.nd.array(np.random.randn(1, 3, 300, 300).astype(np.float32)))
m.set_input(**params)
m.run()
tvm_output = m.get_output(0)

print(tvm_output)
