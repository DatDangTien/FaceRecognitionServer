#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test_onnx");
    Ort::SessionOptions session_options;
    
    // Configure CUDA provider
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    Ort::Session pnet_session(env, "../../models/cuda_pnet.onnx", session_options);

    cv::Mat img = cv::imread("../../data/anh-son-tung-mtp-thumb.jpg", cv::IMREAD_COLOR);
    
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Convert image to float and normalize to [0,1]
    cv::Mat img_float;
    img.convertTo(img_float, CV_32F, 1.0/255.0);

    // Get image dimensions
    int height = img_float.rows;
    int width = img_float.cols;
    int channels = img_float.channels();
    
    std::cout << "Image dimensions - Height: " << height << ", Width: " << width << ", Channels: " << channels << std::endl;

    // Create tensor with shape [1, channels, height, width]
    std::vector<float> input_tensor_values(1 * channels * height * width);
    float* input_ptr = input_tensor_values.data();

    // Permute from HWC to NCHW format
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                input_ptr[c * height * width + h * width + w] = 
                    img_float.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    // int height = 256;
    // int width = 256;
    // int channels = 3;
    // std::vector<float> input_tensor_values(1 * channels * height * width);
    // // Generate random values from normal distribution (like torch.randn())
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::normal_distribution<float> dist(0.0f, 1.0f);  // mean=0, std=1

    // // Fill input tensor with random values
    // for (size_t i = 0; i < input_tensor_values.size(); i++) {
    //     input_tensor_values[i] = dist(gen);
    // }

    std::vector<int64_t> input_shape = {1, channels, height, width};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    Ort::AllocatorWithDefaultOptions allocator;
    
    // Get input names - store AllocatedStringPtr to keep memory alive
    size_t num_input_nodes = pnet_session.GetInputCount();
    std::vector<Ort::AllocatedStringPtr> input_names_ptrs;
    std::vector<const char*> input_names;
    for (size_t i = 0; i < num_input_nodes; i++) {
        input_names_ptrs.push_back(pnet_session.GetInputNameAllocated(i, allocator));
        input_names.push_back(input_names_ptrs.back().get());
        std::cout << input_names.back() << std::endl;
    }

    // Get output names - store AllocatedStringPtr to keep memory alive
    size_t num_output_nodes = pnet_session.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> output_names_ptrs;
    std::vector<const char*> output_names;
    for (size_t i = 0; i < num_output_nodes; i++) {
        output_names_ptrs.push_back(pnet_session.GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_names_ptrs.back().get());
    }

    std::vector<Ort::Value> output_tensors = pnet_session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        num_input_nodes,
        output_names.data(),
        num_output_nodes
    );

    float* bbox_regression = output_tensors[0].GetTensorMutableData<float>();
    float* face_probability = output_tensors[1].GetTensorMutableData<float>();

    // std::cout << "Bbox regression: " << output_tensors[0].GetTensorTypeAndShapeInfo().GetShape() << std::endl;
    // std::cout << "Face probability: " << output_tensors[1].GetTensorTypeAndShapeInfo().GetShape() << std::endl;

    auto bbox_regression_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    auto face_probability_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();


    std::cout << "Bbox regression: " << std::endl;
    for (int i = 0; i < bbox_regression_shape.size(); i++) {
        std::cout << bbox_regression_shape[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Face probability: " << std::endl;
    for (int i = 0; i < face_probability_shape.size(); i++) {
        std::cout << face_probability_shape[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
