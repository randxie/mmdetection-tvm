## How to run the code

1. Update the "TVM_ROOT" path in CMakeLists.txt
2. Update path for the "tvm_runtime_pack.cc". Make sure it points to the actual TVM's src folder.
3. ```mkdir -p build```
4. ```cd build && cmake .. && make```