#pragma once

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "mtcnn_utils.h"

class SubNet {
    public:
        SubNet();
        SubNet(Ort::Env& env, Ort::SessionOptions& session_options, const std::string& model_path);
        ~SubNet();
        std::vector<Ort::Value> forward(Ort::Value& input_tensor);
        bool isInitialized() const { return initialized; }

    private:
        Ort::Session* session;
        bool initialized;
        size_t num_input;
        size_t num_output;
        std::vector<Ort::AllocatedStringPtr> input_names_ptrs;
        std::vector<Ort::AllocatedStringPtr> output_names_ptrs;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
};

class MTCNN {
    public:
        MTCNN(
            int image_size = 160,
            int margin = 0,
            int min_face_size = 20,
            const std::vector<float>& thresholds = {0.6f, 0.7f, 0.7f},
            float factor = 0.709f,
            bool post_process = true,
            bool select_largest = true,
            const std::string& selection_method = "largest",
            bool keep_all = false,
            const std::string& device = "cpu");
        ~MTCNN();
        
        // Main API methods
        mtcnn_utils::DetectionResult detect(cv::Mat& img, bool landmarks = true);
        cv::Mat forward(cv::Mat& img);
        
    private:
        // Member variables
        int image_size;
        int margin;
        int min_face_size;
        std::vector<float> thresholds;
        float factor;
        bool post_process;
        bool select_largest;
        std::string selection_method;
        bool keep_all;
        std::string device;
        
        // ONNX Runtime members
        Ort::Env env;
        Ort::SessionOptions session_options;
        Ort::MemoryInfo memory_info;
        SubNet* pnet;
        SubNet* rnet;
        SubNet* onet;
        
        // Torch device
        torch::Device torch_device;
        
        // Helper methods
        mtcnn_utils::DetectionResult detect_face(const torch::Tensor& img);
        std::tuple<torch::Tensor, torch::Tensor> stage_one(const torch::Tensor& img);
        std::tuple<torch::Tensor, torch::Tensor> stage_two(const torch::Tensor& img, const torch::Tensor& boxes, const torch::Tensor& image_inds);
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> stage_three(
            const torch::Tensor& img,
            const torch::Tensor& boxes,
            const torch::Tensor& image_inds
        );
        
        Ort::Value createTensor(const torch::Tensor& tensor);
};