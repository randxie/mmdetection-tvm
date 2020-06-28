cmake_minimum_required(VERSION 3.6)
project(mmdetection_tvm)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -ldl -pthread")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR})
SET(TVM_ROOT "/home/randxie/tvm-dev/tvm")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${TVM_ROOT}/include)
INCLUDE_DIRECTORIES(${TVM_ROOT}/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${TVM_ROOT}/3rdparty/dlpack/include)
add_executable(mmdetection_tvm tvm_runtime_pack.cc main.cc)
target_link_libraries(mmdetection_tvm ${OpenCV_LIBS})