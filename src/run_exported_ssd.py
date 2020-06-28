import os

import numpy as np

import tvm
from tvm.contrib import graph_runtime
from utils import DEPLOY_WEIGHT_DIR

# load exported parameters, graph def and library
export_lib = os.path.join(DEPLOY_WEIGHT_DIR, "ssd_lib.so")
export_graph = os.path.join(DEPLOY_WEIGHT_DIR, "ssd_graph.json")
export_params = os.path.join(DEPLOY_WEIGHT_DIR, "ssd_param.params")

ctx = tvm.cpu(0)
graph = open(export_graph).read()
lib = tvm.runtime.load_module(export_lib)
params = bytearray(open(export_params, "rb").read())

# create graph runtime
module = graph_runtime.create(graph, lib, ctx)

# define input
input_shape = tuple([1, 3, 300, 300])
random_input = np.random.randn(*input_shape).astype(np.float32)

# load parameters and execute the graph in tvm runtime
module.load_params(params)
module.set_input('input0', tvm.nd.array(random_input))
module.run()

tvm_output = module.get_output(0)
print(tvm_output)
