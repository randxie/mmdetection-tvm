/*!
 * \brief This is an all in one TVM runtime file.
 *
 *   You only have to use this file to compile libtvm_runtime to
 *   include in your project.
 *
 *  - Copy this file into your project which depends on tvm runtime.
 *  - Compile with -std=c++11
 *  - Add the following include path
 *     - /path/to/tvm/include/
 *     - /path/to/tvm/dmlc-core/include/
 *     - /path/to/tvm/dlpack/include/
 *   - Add -lpthread -ldl to the linked library.
 *   - You are good to go.
 *   - See the Makefile in the same folder for example.
 *
 *  The include files here are presented with relative path
 *  You need to remember to change it to point to the right file.
 *
 */
#include "../../../../tvm-dev/tvm/src/runtime/c_runtime_api.cc"
#include "../../../../tvm-dev/tvm/src/runtime/cpu_device_api.cc"
#include "../../../../tvm-dev/tvm/src/runtime/file_util.cc"
#include "../../../../tvm-dev/tvm/src/runtime/library_module.cc"
#include "../../../../tvm-dev/tvm/src/runtime/module.cc"
#include "../../../../tvm-dev/tvm/src/runtime/ndarray.cc"
#include "../../../../tvm-dev/tvm/src/runtime/object.cc"
#include "../../../../tvm-dev/tvm/src/runtime/registry.cc"
#include "../../../../tvm-dev/tvm/src/runtime/thread_pool.cc"
#include "../../../../tvm-dev/tvm/src/runtime/threading_backend.cc"
#include "../../../../tvm-dev/tvm/src/runtime/workspace_pool.cc"

// NOTE: all the files after this are optional modules
// that you can include remove, depending on how much feature you use.

// Likely we only need to enable one of the following
// If you use Module::Load, use dso_module
// For system packed library, use system_lib_module
#include "../../../../tvm-dev/tvm/src/runtime/dso_library.cc"
#include "../../../../tvm-dev/tvm/src/runtime/system_library.cc"

// Graph runtime
#include "../../../../tvm-dev/tvm/src/runtime/graph/graph_runtime.cc"

// Uncomment the following lines to enable RPC
// #include "../../src/runtime/rpc/rpc_session.cc"
// #include "../../src/runtime/rpc/rpc_event_impl.cc"
// #include "../../src/runtime/rpc/rpc_server_env.cc"

// These macros enables the device API when uncommented.
#define TVM_CUDA_RUNTIME 0  
#define TVM_METAL_RUNTIME 0  
#define TVM_OPENCL_RUNTIME 0

// Uncomment the following lines to enable Metal
// #include "../../src/runtime/metal/metal_device_api.mm"
// #include "../../src/runtime/metal/metal_module.mm"

// Uncomment the following lines to enable CUDA
// #include "../../src/runtime/cuda/cuda_device_api.cc"
// #include "../../src/runtime/cuda/cuda_module.cc"

// Uncomment the following lines to enable OpenCL
// #include "../../src/runtime/opencl/opencl_device_api.cc"
// #include "../../src/runtime/opencl/opencl_module.cc"