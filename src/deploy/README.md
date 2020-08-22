## Environment

1. Set up conda environment: ```conda create -n tvm python=3.7```
2. Make sure opencv and gtk2 (for plotting) are installed: ```conda install -c conda-forge gtk2```


## How to run the code

1. Update the "TVM_ROOT" path in CMakeLists.txt
2. Update path for the "tvm_runtime_pack.cc". Make sure it points to the actual TVM's src folder.
3. ```mkdir -p build```
4. ```cd build && cmake .. && make```
5. Example run command: ```./run_ssd --model_path="/home/randxie/mmdetection-tvm/deploy_weight/" --image_path="/home/randxie/mmdetection-tvm/test_images/demo.jpg"```