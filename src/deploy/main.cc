#include "ssd_model.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

DEFINE_string(model_path, "", "The ssd model path to use");
DEFINE_string(image_path, "", "The image path to use");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_path.empty()) {
    LOG(FATAL) << "Model path should not be emtpy";
  }
  SsdModel model = SsdModel(FLAGS_model_path);
  std::cout << "Successfully load model." << std::endl;

  cv::Mat image = cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
  if (!image.data) {
    LOG(FATAL) << "Can not load image data";
  }
  cv::Mat result = model.Forward(image);

  // print first 10 rows
  for (int i = 0; i < 10; ++i) {
    std::cout << "Getting detection result: " << result.row(i) << std::endl;
  }

  return 0;
}
