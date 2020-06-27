import os

from torch.utils import cpp_extension

from utils import ROOT_DIR


def load_custom_ops():
  cpp_extension.load(
    name="custom_ops",
    sources=[os.path.join(ROOT_DIR, "src", "custom_ops", "custom_ops.cc")],
    is_python_module=False,
    verbose=True)
