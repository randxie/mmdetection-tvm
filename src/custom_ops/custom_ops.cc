#include <torch/script.h>

torch::Tensor dummy_multibox_detect(torch::Tensor cls_probs, torch::Tensor bbox_preds, torch::Tensor anchors) {
  return cls_probs.clone();
}

static auto registry = torch::RegisterOperators("custom_ops::dummy_multibox_detect", &dummy_multibox_detect);
