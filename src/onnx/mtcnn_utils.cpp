#include "mtcnn_utils.h"
#include <iostream>

namespace mtcnn_utils {

// Convert OpenCV Mat to torch::Tensor
torch::Tensor cv_to_tensor(const cv::Mat& img, torch::Device device) {
    cv::Mat img_float;
    if (img.type() != CV_32FC3) {
        img.convertTo(img_float, CV_32FC3);
    } else {
        img_float = img.clone();
    }
    
    // Convert HWC to CHW
    auto tensor = torch::from_blob(img_float.data, {img_float.rows, img_float.cols, 3}, torch::kFloat32).clone();
    tensor = tensor.permute({2, 0, 1}); // HWC -> CHW
    tensor = tensor.unsqueeze(0)
    
    return tensor.to(device);
}

// Convert torch::Tensor to OpenCV Mat
cv::Mat tensor_to_cv(const torch::Tensor& tensor) {
    auto cpu_tensor = tensor.to(torch::kCPU);
    
    // CHW -> HWC
    if (cpu_tensor.dim() == 3) {
        cpu_tensor = cpu_tensor.permute({1, 2, 0});
    }
    
    cpu_tensor = cpu_tensor.contiguous();
    
    int height = cpu_tensor.size(0);
    int width = cpu_tensor.size(1);
    int channels = cpu_tensor.dim() == 3 ? cpu_tensor.size(2) : 1;
    
    cv::Mat mat(height, width, channels == 3 ? CV_32FC3 : CV_32FC1);
    std::memcpy(mat.data, cpu_tensor.data_ptr<float>(), cpu_tensor.numel() * sizeof(float));
    
    return mat;
}

// Convert ONNX output to torch::Tensor
torch::Tensor onnx_to_torch(const float* data, const std::vector<int64_t>& shape, torch::Device device) {
    std::vector<int64_t> torch_shape(shape.begin(), shape.end());
    auto tensor = torch::from_blob(const_cast<float*>(data), torch_shape, torch::kFloat32).clone();
    return tensor.to(device);
}

// Convert torch::Tensor to ONNX-compatible vector
std::vector<float> torch_to_onnx(const torch::Tensor& tensor) {
    auto cpu_tensor = tensor.to(torch::kCPU).contiguous();
    float* data_ptr = cpu_tensor.data_ptr<float>();
    return std::vector<float>(data_ptr, data_ptr + cpu_tensor.numel());
}

// Resize image tensor using bilinear interpolation
torch::Tensor imresample(const torch::Tensor& img, int target_h, int target_w) {
    // img: [C, H, W] or [B, C, H, W]
    auto input = img;
    bool added_batch = false;
    
    if (input.dim() == 3) {
        input = input.unsqueeze(0); // Add batch dimension
        added_batch = true;
    }
    
    // Use PyTorch's interpolate (area mode for downsampling)
    auto resized = torch::nn::functional::interpolate(input,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{target_h, target_w})
            .mode(torch::kArea));
    
    if (added_batch) {
        resized = resized.squeeze(0);
    }
    
    return resized;
}

// Preprocess image: normalize
torch::Tensor norm(const torch::Tensor& img, float mean, float std) {
    return (img - mean) * std;
}

// Generate bounding boxes from PNet output
std::tuple<torch::Tensor, torch::Tensor> generateBoundingBox(
    const torch::Tensor& reg,        // [1, 4, H, W] or [B, 4, H, W]
    const torch::Tensor& face_prob,  // [1, H, W] or [B, H, W] - already extracted channel 1
    float scale,
    float threshold
) {
    const int stride = 2;
    const int cellsize = 12;
    
    // mask: [B, H, W]
    auto mask = face_prob >= threshold;
    
    // mask_inds: [N, 3] where columns are [batch_idx, h, w]
    auto mask_inds = torch::nonzero(mask);
    
    if (mask_inds.size(0) == 0) {
        // No faces detected
        return {torch::zeros({0, 9}), torch::zeros({0}, torch::kLong)};
    }
    
    auto image_inds = mask_inds.index({torch::indexing::Slice(), 0});    
    auto h_indices = mask_inds.index({torch::indexing::Slice(), 1}).to(torch::kFloat32);
    auto w_indices = mask_inds.index({torch::indexing::Slice(), 2}).to(torch::kFloat32);
    auto scores = face_prob.masked_select(mask);
    auto x1 = torch::floor((stride * w_indices + 1) / scale);
    auto y1 = torch::floor((stride * h_indices + 1) / scale);
    auto x2 = torch::floor((stride * w_indices + cellsize) / scale);
    auto y2 = torch::floor((stride * h_indices + cellsize) / scale);    
    auto reg_permuted = reg.permute({1, 0, 2, 3});  // [4, B, H, W]
    auto reg_values = reg_permuted.index({torch::indexing::Slice(), mask}).permute({1, 0});  // [N, 4]
    
    auto q1 = torch::stack({x1, y1}, 1);  // [N, 2]
    auto q2 = torch::stack({x2, y2}, 1);  // [N, 2]
    auto boxes = torch::cat({q1, q2, scores.unsqueeze(1), reg_values}, 1);  // [N, 9]
    return {boxes, image_inds};
}

// Bounding box regression
torch::Tensor bbreg(const torch::Tensor& boxes, const torch::Tensor& reg) {
    // boxes: [N, 4] or [N, 5+]
    // reg: [N, 4]
    auto result = boxes.clone();
    
    auto w = boxes.index({torch::indexing::Slice(), 2}) - boxes.index({torch::indexing::Slice(), 0}) + 1;
    auto h = boxes.index({torch::indexing::Slice(), 3}) - boxes.index({torch::indexing::Slice(), 1}) + 1;
    
    result.index_put_({torch::indexing::Slice(), 0}, 
        boxes.index({torch::indexing::Slice(), 0}) + reg.index({torch::indexing::Slice(), 0}) * w);
    result.index_put_({torch::indexing::Slice(), 1},
        boxes.index({torch::indexing::Slice(), 1}) + reg.index({torch::indexing::Slice(), 1}) * h);
    result.index_put_({torch::indexing::Slice(), 2},
        boxes.index({torch::indexing::Slice(), 2}) + reg.index({torch::indexing::Slice(), 2}) * w);
    result.index_put_({torch::indexing::Slice(), 3},
        boxes.index({torch::indexing::Slice(), 3}) + reg.index({torch::indexing::Slice(), 3}) * h);
    
    return result;
}

// Convert boxes to square
torch::Tensor rerec(const torch::Tensor& boxes) {
    // boxes: [N, 4+]
    auto result = boxes.clone();
    
    auto w = boxes.index({torch::indexing::Slice(), 2}) - boxes.index({torch::indexing::Slice(), 0});
    auto h = boxes.index({torch::indexing::Slice(), 3}) - boxes.index({torch::indexing::Slice(), 1});
    auto l = torch::max(w, h);
    
    result.index_put_({torch::indexing::Slice(), 0},
        boxes.index({torch::indexing::Slice(), 0}) + w * 0.5 - l * 0.5);
    result.index_put_({torch::indexing::Slice(), 1},
        boxes.index({torch::indexing::Slice(), 1}) + h * 0.5 - l * 0.5);
    result.index_put_({torch::indexing::Slice(), 2},
        result.index({torch::indexing::Slice(), 0}) + l);
    result.index_put_({torch::indexing::Slice(), 3},
        result.index({torch::indexing::Slice(), 1}) + l);
    
    return result;
}

// Pad boxes to image boundaries
torch::Tensor pad(const torch::Tensor& boxes, int img_w, int img_h) {
    auto result = boxes.clone();
    
    result.index_put_({torch::indexing::Slice(), 0}, 
        torch::clamp(result.index({torch::indexing::Slice(), 0}), 1, img_w));
    result.index_put_({torch::indexing::Slice(), 1},
        torch::clamp(result.index({torch::indexing::Slice(), 1}), 1, img_h));
    result.index_put_({torch::indexing::Slice(), 2},
        torch::clamp(result.index({torch::indexing::Slice(), 2}), 1, img_w));
    result.index_put_({torch::indexing::Slice(), 3},
        torch::clamp(result.index({torch::indexing::Slice(), 3}), 1, img_h));
    
    return result;
}

// Non-Maximum Suppression
torch::Tensor nms(
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    float threshold,
    const std::string& method
) {
    if (boxes.size(0) == 0) {
        return torch::empty({0}, torch::kLong);
    }
    
    auto x1 = boxes.index({torch::indexing::Slice(), 0});
    auto y1 = boxes.index({torch::indexing::Slice(), 1});
    auto x2 = boxes.index({torch::indexing::Slice(), 2});
    auto y2 = boxes.index({torch::indexing::Slice(), 3});
    
    auto areas = (x2 - x1 + 1) * (y2 - y1 + 1);
    
    auto order = torch::argsort(scores, /*dim=*/0, /*descending=*/true);
    
    std::vector<int64_t> keep;
    
    while (order.numel() > 0) {
        auto i = order[0].item<int64_t>();
        keep.push_back(i);
        
        if (order.numel() == 1) break;
        
        auto rest = order.index({torch::indexing::Slice(1, torch::indexing::None)});
        
        auto xx1 = torch::maximum(x1[i], x1.index({rest}));
        auto yy1 = torch::maximum(y1[i], y1.index({rest}));
        auto xx2 = torch::minimum(x2[i], x2.index({rest}));
        auto yy2 = torch::minimum(y2[i], y2.index({rest}));
        
        auto w = torch::clamp(xx2 - xx1 + 1, 0);
        auto h = torch::clamp(yy2 - yy1 + 1, 0);
        auto inter = w * h;
        
        torch::Tensor ovr;
        if (method == "Min") {
            ovr = inter / torch::minimum(areas[i], areas.index({rest}));
        } else {
            ovr = inter / (areas[i] + areas.index({rest}) - inter);
        }
        
        auto inds = torch::nonzero(ovr <= threshold).squeeze(1);
        order = rest.index({inds});
    }
    
    return torch::tensor(keep, torch::kLong);
}

// Batched NMS
torch::Tensor batched_nms(
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    const torch::Tensor& idxs,
    float threshold
) {
    if (boxes.numel() == 0) {
        return torch::empty({0}, torch::kLong);
    }
    
    auto max_coordinate = boxes.max();
    auto offsets = idxs.to(boxes.dtype()) * (max_coordinate + 1);
    auto boxes_for_nms = boxes + offsets.unsqueeze(1);
    
    return nms(boxes_for_nms, scores, threshold, "Union");
}

// Calculate scale pyramid
std::vector<float> calculateScales(int width, int height, int min_face_size, float factor) {
    float m = 12.0f / min_face_size;
    float minl = std::min(width, height) * m;
    
    std::vector<float> scales;
    while (minl >= 12) {
        scales.push_back(m);
        m = m * factor;
        minl = minl * factor;
    }
    
    return scales;
}

// Extract patches from image
torch::Tensor extract_patches(const torch::Tensor& img, const torch::Tensor& boxes, const torch::Tensor& image_inds, int target_size) {
    // img: [B, C, H, W]
    // boxes: [N, 4+] where columns are [x1, y1, x2, y2, ...]
    // image_inds: [N] - batch index for each box
    int num_boxes = boxes.size(0);
    int img_h = img.size(2);
    int img_w = img.size(3);
    int img_c = img.size(1);
    
    if (num_boxes == 0) {
        return torch::zeros({0, img_c, target_size, target_size});
    }
    
    std::vector<torch::Tensor> patches;
    
    for (int i = 0; i < num_boxes; i++) {
        auto box = boxes[i];
        
        // Get box coordinates (1-indexed in Python, so subtract 1)
        int x1 = static_cast<int>(box[0].item<float>());
        int y1 = static_cast<int>(box[1].item<float>());
        int x2 = static_cast<int>(box[2].item<float>());
        int y2 = static_cast<int>(box[3].item<float>());
        
        if (y2 <= (y1 - 1) || x2 <= (x1 - 1)) continue;
        
        int y_start = std::max(0, y1 - 1);
        int y_end = std::min(img_h, y2);
        int x_start = std::max(0, x1 - 1);
        int x_end = std::min(img_w, x2);

        // Extract patch: [C, H_patch, W_patch]
        // Matches Python indexing order: [batch, channels, height_range, width_range]
        auto patch = img.index({
            image_inds[i].item<int64_t>(),
            torch::indexing::Slice(),
            torch::indexing::Slice(y_start, y_end),
            torch::indexing::Slice(x_start, x_end)
        });
        
        // Add batch dimension for imresample: [1, C, H_patch, W_patch]
        patch = patch.unsqueeze(0);
        
        // Resize to target size: [1, C, target_size, target_size]
        patch = imresample(patch, target_size, target_size);
        
        patches.push_back(patch);
    }
    
    if (patches.empty()) {
        return torch::zeros({0, img_c, target_size, target_size});
    }
    
    // Stack to [N, C, target_size, target_size]
    return torch::cat(patches, 0);
}

} // namespace mtcnn_utils
