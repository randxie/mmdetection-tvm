import os
import tvm
import onnx
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm.contrib.graph_runtime as runtime
from utils import ROOT_DIR

# Preprocess image
# TODO: not sure normalizing the image is necessary or not for ssd, to be
# verified
img_path = os.path.join(ROOT_DIR, 'test_images/demo.jpg')
img = Image.open(img_path).resize((300, 300))
x = np.array(img).reshape(1, -1, 300, 300).astype(np.float32)  # cryptic error msg when type is to set, see issue https://github.com/randxie/mmdetection-tvm/issues/2

# Load model
# Sample command to create the onnx model from mmdetection (NOTE: --shape 300
# is an important flag although it is said optional in mmdetection doc):
# python ~/mmedetection/tools/pytorch2onnx.py configs/ssd/ssd300_coco.py 
# checkpoints/ssd300_coco_20200307-a92d2092.pth 
# --out ~/mmdetection-tvm/onnx/ssd300_coco.onnx --shape 300
model_path = os.path.join(ROOT_DIR, 'onnx/ssd300_coco.onnx')
onnx_model = onnx.load(model_path)

# Compile
input_blob = onnx_model.graph.input[0]
input_dim = input_blob.type.tensor_type.shape.dim
shape_dict = {input_blob.name: tuple(d.dim_value for d in input_dim)}

target = 'llvm'
ctx = tvm.context(target)
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with relay.build_config(opt_level=1):
    graph, lib, params = relay.build(mod, target, params=params)

module = runtime.create(graph, lib, ctx)

# run
module.set_input(input_blob.name, tvm.nd.array(x))
module.set_input(**params)
module.run()

# get output
cls_score_list = [module.get_output(i).asnumpy()[0] for i in range(6)]
bbox_pred_list = [module.get_output(i + 6).asnumpy()[0] for i in range(6)]
print(cls_score_list)
print(bbox_pred_list)

# TODO: use the postprocessing module in mmdetection to process the output.
# Installing pytorch in Jetson Nano is quite easy, so wouldn't bother using the
# postprocess scripts in this post https://zhuanlan.zhihu.com/p/136442019
