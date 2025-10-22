#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <algorithm>
#include <cmath>

namespace mtcnn_utils {

// Detection result structure
struct DetectionResult {
    torch::Tensor boxes;      // [N, 4] bounding boxes (x1, y1, x2, y2)
    torch::Tensor probs;      // [N] probabilities  
    torch::Tensor landmarks;  // [N, 10] landmark points (x1,x2,x3,x4,x5, y1,y2,y3,y4,y5)
};

// Conversion functions
torch::Tensor cv_to_tensor(const cv::Mat& img, torch::Device device = torch::kCPU);
cv::Mat tensor_to_cv(const torch::Tensor& tensor);
torch::Tensor onnx_to_torch(const float* data, const std::vector<int64_t>& shape, torch::Device device = torch::kCPU);
std::vector<float> torch_to_onnx(const torch::Tensor& tensor);

// Image operations  
torch::Tensor imresample(const torch::Tensor& img, int target_h, int target_w);
torch::Tensor norm(const torch::Tensor& img, float mean = 127.5f, float std = 0.0078125f);

// Bounding box operations
std::tuple<torch::Tensor, torch::Tensor> generateBoundingBox(
    const torch::Tensor& reg,       // [B, 4, H, W] - regression offsets
    const torch::Tensor& face_prob, // [B, H, W] - face probability only (channel 1)
    float scale,
    float threshold
);
// Returns: (boxes [N, 9], image_inds [N]) where boxes = [x1, y1, x2, y2, score, reg0-3]

torch::Tensor bbreg(const torch::Tensor& boxes, const torch::Tensor& reg);
torch::Tensor rerec(const torch::Tensor& boxes);
torch::Tensor pad(const torch::Tensor& boxes, int img_w, int img_h);

// NMS operations
torch::Tensor nms(
    const torch::Tensor& boxes,    // [N, 4]
    const torch::Tensor& scores,   // [N]
    float threshold,
    const std::string& method = "Union"
);

torch::Tensor batched_nms(
    const torch::Tensor& boxes,    // [N, 4]
    const torch::Tensor& scores,   // [N]
    const torch::Tensor& idxs,     // [N]
    float threshold
);

// Utility functions
std::vector<float> calculateScales(int width, int height, int min_face_size, float factor);
torch::Tensor extract_patches(const torch::Tensor& img, const torch::Tensor& boxes, const torch::Tensor& image_inds, int target_size);

} // namespace mtcnn_utils
