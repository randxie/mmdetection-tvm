#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <cstdio>

constexpr char kLibSoFileName[] = "ssd_lib.so";
constexpr char kGraphFileName[] = "ssd_graph.json";
constexpr char kParamsFileName[] = "ssd_param.params";
constexpr int kSsdWidth = 300;
constexpr int kSsdHeight = 300;

// TVM array constants
constexpr int kDTypeCode = kDLFloat;
constexpr int kDTypeBits = 32;
constexpr int kDTypeLanes = 1;
constexpr int kDeviceType = kDLCPU;
constexpr int kDeviceId = 0;
constexpr int kInDim = 4;

class SsdModel
{
public:
    SsdModel(const std::string& weight_folder)
    {
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(weight_folder + kLibSoFileName);

        // load graph
        std::ifstream json_in(weight_folder + kGraphFileName);
        std::string graph_json((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();

        // Define device
        int device_type = kDLCPU;
        int device_id = 0;

        // get global function module for graph runtime
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(graph_json, mod_syslib, device_type, device_id);
        module_ = std::unique_ptr<tvm::runtime::Module>(new tvm::runtime::Module(mod));

        //load param
        std::ifstream params_in(weight_folder + kParamsFileName, std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();

        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();

        tvm::runtime::PackedFunc load_params = module_->GetFunction("load_params");
        load_params(params_arr);
    }

    cv::Mat Forward(cv::Mat input_image)
    {
        // TODO: update the code in forward function to generate bounding box.
        
        cv::Mat tensor = cv::dnn::blobFromImage(input_image, 1.0, cv::Size(kSsdWidth, kSsdHeight), cv::Scalar(0, 0, 0), true);

        //convert uint8 to float32 and convert to RGB via opencv dnn function
        TVMArrayHandle input;
        constexpr int dtype_code = kDLFloat;
        constexpr int dtype_bits = 32;
        constexpr int dtype_lanes = 1;
        constexpr int device_type = kDLCPU;
        constexpr int device_id = 0;
        const int64_t input_shape[kInDim] = {1, 3, kSsdWidth, kSsdHeight};

        TVMArrayAlloc(input_shape, kInDim, kDTypeCode, kDTypeBits, kDTypeLanes, kDeviceType, kDeviceId, &input);
        TVMArrayCopyFromBytes(input, tensor.data, kSsdWidth * 3 * kSsdHeight * 4);

        // set input
        module_->GetFunction("set_input")("input0", input);

        // execute graph
        module_->GetFunction("run")();

        // get output
        tvm::runtime::PackedFunc get_output = module_->GetFunction("get_output");
        const tvm::runtime::NDArray res = get_output(0);

        cv::Mat vector(8732, 6, CV_32F);
        memcpy(vector.data, res->data, 8732 * 6 * 4);

        // vector = vector.reshape(6, 8732);

        TVMArrayFree(input);
        return vector;
    }

private:
    std::unique_ptr<tvm::runtime::Module> module_;
};