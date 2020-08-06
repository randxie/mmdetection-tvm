# mmdetection-tvm

This project contains early effort on converting mmdetection models into TVM Relay IR, which then can be deployed to different hardware, e.g. Nvidia Jetson.

## Road Map

We will mainly focus on two aspects to bring mmdetection models into production:

1. Contribute to mmdetection to make all the models traceable. Currently, mmdetection is coded in a very Pythonic way that's hard to trace.
2. Contribute to TVM for any missing operators / fix bugs for pytorch frontend. We also target at optimizing deployment performance of mmdetection models using Auto TVM. 

As first step, we will focus on a few set of commonly used object detection models, as listed below:

- [x] SSD
- [ ] Faster R-CNN
- [ ] Cascade R-CNN
- [ ] RetinaNet


## Prerequisite

* Set up a new conda environment: ```conda create -n tvm python=3.7```
* [Install TVM from source](https://docs.tvm.ai/install/from_source.html)
* [Install mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md)
* Hardware (optional)
    * [NVIDIA Jetson Nano Developer Kit](https://www.amazon.com/gp/product/B07PZHBDKT/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1)
    * [Camera](https://www.amazon.com/gp/product/B07SL9P729/ref=ppx_yo_dt_b_asin_title_o01_s00?ie=UTF8&psc=1)
    * [SD card](https://www.amazon.com/gp/product/B06XWZWYVP/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1)

## Weights

### SSD 300

VGG 16 backbone can be downloaded from [here](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd)
