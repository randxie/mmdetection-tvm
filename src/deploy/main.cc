#include <iostream>
#include "ssd_model.h"

int main() {
  const SsdModel model = SsdModel("/home/randxie/mmdetection-tvm/deploy_weight/");
  std::cout << "Successfully load model." << std::endl;
}
