#include "mtcnn.h"
#include <algorithm>
#include <cmath>

// Helper function to get input/output names
void get_node_names(
    Ort::Session& session,
    bool is_input,
    std::vector<Ort::AllocatedStringPtr>& names_ptrs,
    std::vector<const char*>& names
) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_nodes = is_input ? session.GetInputCount() : session.GetOutputCount();
    
    for (size_t i = 0; i < num_nodes; i++) {
        auto name_ptr = is_input ? 
            session.GetInputNameAllocated(i, allocator) :
            session.GetOutputNameAllocated(i, allocator);
        names_ptrs.push_back(std::move(name_ptr));
        names.push_back(names_ptrs.back().get());
    }
}

// ==================== SubNet Implementation ====================

SubNet::SubNet() : session(nullptr), initialized(false), num_input(0), num_output(0) {}

SubNet::SubNet(Ort::Env& env, Ort::SessionOptions& session_options, const std::string& model_path)
    : initialized(false) {
    try {
        session = new Ort::Session(env, model_path.c_str(), session_options);
        
        get_node_names(*session, true, input_names_ptrs, input_names);
        get_node_names(*session, false, output_names_ptrs, output_names);
        
        num_input = input_names.size();
        num_output = output_names.size();
        initialized = true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model " << model_path << ": " << e.what() << std::endl;
        session = nullptr;
        initialized = false;
    }
}

SubNet::~SubNet() {
    if (session) {
        delete session;
        session = nullptr;
    }
}

std::vector<Ort::Value> SubNet::forward(Ort::Value& input_tensor) {
    if (!initialized || !session) {
        throw std::runtime_error("SubNet not properly initialized");
    }
    
    return session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        num_input,
        output_names.data(),
        num_output
    );
}

// ==================== MTCNN Implementation ====================

MTCNN::MTCNN(
    int image_size,
    int margin,
    int min_face_size,
    const std::vector<float>& thresholds,
    float factor,
    bool post_process,
    bool select_largest,
    const std::string& selection_method,
    bool keep_all,
    const std::string& device)
    : image_size(image_size)
    , margin(margin)
    , min_face_size(min_face_size)
    , thresholds(thresholds)
    , factor(factor)
    , post_process(post_process)
    , select_largest(select_largest)
    , selection_method(selection_method)
    , keep_all(keep_all)
    , device(device)
    , env(ORT_LOGGING_LEVEL_ERROR, "MTCNN")
    , memory_info(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault))
    , torch_device(device == "cuda" ? torch::kCUDA : torch::kCPU)
    {
    // Setup execution provider
        if (device == "cuda") {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }
    
    // Load models
    pnet = new SubNet(env, session_options, "../../models/cuda_pnet.onnx");
    rnet = new SubNet(env, session_options, "../../models/cuda_rnet.onnx");
    onet = new SubNet(env, session_options, "../../models/cuda_onet.onnx");
    
    if (!pnet->isInitialized() || !rnet->isInitialized() || !onet->isInitialized()) {
        std::cerr << "Warning: Some MTCNN models failed to load" << std::endl;
    }
    }

MTCNN::~MTCNN() {
    delete pnet;
    delete rnet;
    delete onet;
}

// Create ONNX tensor from torch::Tensor
Ort::Value MTCNN::createTensor(const torch::Tensor& tensor) {
    auto data = mtcnn_utils::torch_to_onnx(tensor);
    auto sizes = tensor.sizes();
    std::vector<int64_t> shape(sizes.begin(), sizes.end());
    
    return Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(data.data()),
        data.size(),
        shape.data(),
        shape.size()
    );
}

// Stage 1: PNet - Image pyramid detection
std::tuple<torch::Tensor, torch::Tensor> MTCNN::stage_one(const torch::Tensor& img) {
    // img: [C, H, W]
    int height = img.size(1);
    int width = img.size(2);
    
    // Calculate scale pyramid
    std::vector<float> scales = mtcnn_utils::calculateScales(width, height, min_face_size, factor);
    
    std::vector<torch::Tensor> all_boxes;
    std::vector<torch::Tensor> all_image_inds;
    std::vector<torch::Tensor> scale_picks;
    int64_t offset = 0;
    
    // Process each scale
    for (float scale : scales) {
        int hs = static_cast<int>(std::ceil(height * scale));
        int ws = static_cast<int>(std::ceil(width * scale));
        
        // Resize image
        auto img_resized = mtcnn_utils::imresample(img, hs, ws);
        
        // Normalize
        auto img_norm = mtcnn_utils::norm(img_resized, 127.5, 0.0078125);
        
        
        // Create ONNX tensor and run PNet
        Ort::Value input_tensor = createTensor(img_norm);
        std::vector<Ort::Value> outputs = pnet->forward(input_tensor);
        
        // Convert outputs to torch tensors
        auto reg_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        auto prob_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        
        auto reg = mtcnn_utils::onnx_to_torch(
            outputs[0].GetTensorMutableData<float>(), reg_shape, torch_device);
        auto prob = mtcnn_utils::onnx_to_torch(
            outputs[1].GetTensorMutableData<float>(), prob_shape, torch_device);
        
        // Extract face probability (channel 1) - matches Python: probs[:, 1]
        auto face_prob = prob.index({torch::indexing::Slice(), 1});  // [1, H, W]
        
        // Generate bounding boxes
        auto [boxes_scale, image_inds_scale] = mtcnn_utils::generateBoundingBox(reg, face_prob, scale, thresholds[0]);
        
        // Store boxes and image_inds BEFORE filtering
        all_boxes.push_back(boxes_scale);
        all_image_inds.push_back(image_inds_scale);
        
        if (boxes_scale.size(0) > 0) {
            // Apply batched NMS within this scale
            auto scores = boxes_scale.index({torch::indexing::Slice(), 4});
            auto pick = mtcnn_utils::batched_nms(
                boxes_scale.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)}),
                scores, 
                image_inds_scale, 
                0.5f
            );
            
            // Add offset to indices - matches Python: pick + offset
            scale_picks.push_back(pick + offset);
            offset += boxes_scale.size(0);
        }
    }
    
    if (all_boxes.empty()) {
        return {torch::zeros({0, 5}), torch::zeros({0}, torch::kLong)};
    }
    
    // Combine all boxes from all scales (BEFORE filtering with scale_picks)
    auto boxes = torch::cat(all_boxes, 0);
    auto image_inds = torch::cat(all_image_inds, 0);
    
    // Filter using scale_picks - matches Python: boxes[scale_picks]
    if (!scale_picks.empty()) {
        auto scale_picks_cat = torch::cat(scale_picks, 0);
        boxes = boxes.index({scale_picks_cat});
        image_inds = image_inds.index({scale_picks_cat});
    }
    
    // Apply batched NMS across all scales
    auto pick = mtcnn_utils::batched_nms(
        boxes.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)}),
        boxes.index({torch::indexing::Slice(), 4}),
        image_inds,
        0.7f
    );
    boxes = boxes.index({pick});
    image_inds = image_inds.index({pick});
    
    // Apply bounding box regression - matches Python lines 197-203
    auto regw = boxes.index({torch::indexing::Slice(), 2}) - boxes.index({torch::indexing::Slice(), 0});
    auto regh = boxes.index({torch::indexing::Slice(), 3}) - boxes.index({torch::indexing::Slice(), 1});
    auto qq1 = boxes.index({torch::indexing::Slice(), 0}) + boxes.index({torch::indexing::Slice(), 5}) * regw;
    auto qq2 = boxes.index({torch::indexing::Slice(), 1}) + boxes.index({torch::indexing::Slice(), 6}) * regh;
    auto qq3 = boxes.index({torch::indexing::Slice(), 2}) + boxes.index({torch::indexing::Slice(), 7}) * regw;
    auto qq4 = boxes.index({torch::indexing::Slice(), 3}) + boxes.index({torch::indexing::Slice(), 8}) * regh;
    
    // Stack and permute - matches Python: torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = torch::stack({qq1, qq2, qq3, qq4, boxes.index({torch::indexing::Slice(), 4})}).permute({1, 0});
    
    // Convert to square
    boxes = mtcnn_utils::rerec(boxes);
    
    // Pad to image boundaries - returns (y, ey, x, ex) but we only need boxes updated
    boxes = mtcnn_utils::pad(boxes, width, height);
    
    return {boxes, image_inds};
}

// Stage 2: RNet - Refinement
std::tuple<torch::Tensor, torch::Tensor> MTCNN::stage_two(const torch::Tensor& img, const torch::Tensor& boxes, const torch::Tensor& image_inds) {
    // img: [B, C, H, W]
    // boxes: [N, 5]
    if (boxes.size(0) == 0) {
        return {torch::zeros({0, 5}), torch::zeros({0}, torch::kLong)};
    }
    
    const int target_size = 24;
    int img_h = img.size(2);
    int img_w = img.size(3);
    
    // Extract patches - extract_patches now handles [B, C, H, W] directly
    auto patches = mtcnn_utils::extract_patches(img, boxes, image_inds, target_size);  // [N, C, 24, 24]
    
    if (patches.size(0) == 0) {
        return {torch::zeros({0, 5}), torch::zeros({0}, torch::kLong)};
    }
    
    // Normalize patches
    auto patches_norm = mtcnn_utils::norm(patches, 127.5f, 0.0078125f);
    
    // Run RNet
    Ort::Value input_tensor = createTensor(patches_norm);
    std::vector<Ort::Value> outputs = rnet->forward(input_tensor);
    
    // Convert outputs to torch tensors
    auto reg_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto prob_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    
    auto reg = mtcnn_utils::onnx_to_torch(
        outputs[0].GetTensorMutableData<float>(), reg_shape, torch_device);
    auto prob = mtcnn_utils::onnx_to_torch(
        outputs[1].GetTensorMutableData<float>(), prob_shape, torch_device);
    
    // Permute to match Python: out0.permute(1, 0) and out1.permute(1, 0)
    reg = reg.permute({1, 0});   // [N, 4] → [4, N]
    prob = prob.permute({1, 0}); // [N, 2] → [2, N]
    
    // Get face probabilities (class 1) - matches Python: score = out1[1, :]
    auto score = prob.index({1, torch::indexing::Slice()}); // [N]
    
    // Filter by threshold - matches Python: ipass = score > threshold[1]
    auto ipass = score > thresholds[1];
    
    // Create boxes with scores - matches Python: boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
    auto boxes_filtered = torch::cat({
        boxes.index({ipass, torch::indexing::Slice(torch::indexing::None, 4)}),
        score.index({ipass}).unsqueeze(1)
    }, 1); // [M, 5]
    
    // Filter image_inds - matches Python: image_inds = image_inds[ipass]
    auto image_inds_filtered = image_inds.index({ipass});
    
    // Filter regression - matches Python: mv = out0[:, ipass].permute(1, 0)
    auto mv = reg.index({torch::indexing::Slice(), ipass}).permute({1, 0}); // [M, 4]
    
    if (boxes_filtered.size(0) == 0) {
        return {torch::zeros({0, 5}), torch::zeros({0}, torch::kLong)};
    }
    
    // NMS within each image - matches Python: pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    auto pick = mtcnn_utils::batched_nms(
        boxes_filtered.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 4)}),
        boxes_filtered.index({torch::indexing::Slice(), 4}),
        image_inds_filtered,
        0.7f
    );
    
    // Apply pick - matches Python: boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
    boxes_filtered = boxes_filtered.index({pick});
    image_inds_filtered = image_inds_filtered.index({pick});
    mv = mv.index({pick});
    
    // Apply bounding box regression - matches Python: boxes = bbreg(boxes, mv)
    boxes_filtered = mtcnn_utils::bbreg(boxes_filtered, mv);
    
    // Convert to square - matches Python: boxes = rerec(boxes)
    boxes_filtered = mtcnn_utils::rerec(boxes_filtered);
    
    return {boxes_filtered, image_inds_filtered};
}

// Stage 3: ONet - Final detection with landmarks
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MTCNN::stage_three(
    const torch::Tensor& img,
    const torch::Tensor& boxes,
    const torch::Tensor& image_inds
) {
    // img: [B, C, H, W]
    // boxes: [N, 5]
    if (boxes.size(0) == 0) {
        return {torch::zeros({0, 5}), torch::zeros({0}, torch::kLong), torch::zeros({0, 10})};
    }
    
    const int target_size = 48;
    int img_h = img.size(2);
    int img_w = img.size(3);
    
    // Extract patches - extract_patches now handles [B, C, H, W] directly
    auto patches = mtcnn_utils::extract_patches(img, boxes, image_inds, target_size);  // [N, C, 48, 48]
    
    if (patches.size(0) == 0) {
        return {torch::zeros({0, 5}), torch::zeros({0}, torch::kLong), torch::zeros({0, 10})};
    }
    
    // Normalize patches
    auto patches_norm = mtcnn_utils::norm(patches, 127.5f, 0.0078125f);
    
    // Run ONet
    Ort::Value input_tensor = createTensor(patches_norm);
    std::vector<Ort::Value> outputs = onet->forward(input_tensor);
    
    // Convert outputs to torch tensors
    // ONet outputs: [0]=bbox_regression, [1]=landmarks, [2]=face_probability
    auto reg_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto landmark_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    auto prob_shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    
    auto reg = mtcnn_utils::onnx_to_torch(
        outputs[0].GetTensorMutableData<float>(), reg_shape, torch_device);
    auto landmarks = mtcnn_utils::onnx_to_torch(
        outputs[1].GetTensorMutableData<float>(), landmark_shape, torch_device);
    auto prob = mtcnn_utils::onnx_to_torch(
        outputs[2].GetTensorMutableData<float>(), prob_shape, torch_device);
    
    // Permute to match Python: out0.permute(1, 0), out1.permute(1, 0), out2.permute(1, 0)
    reg = reg.permute({1, 0});         // [N, 4] → [4, N]
    landmarks = landmarks.permute({1, 0}); // [N, 10] → [10, N]
    prob = prob.permute({1, 0});       // [N, 2] → [2, N]
    
    // Get face probabilities (class 1) - matches Python: score = out2[1, :]
    auto score = prob.index({1, torch::indexing::Slice()}); // [N]
    
    // points = out1 - matches Python
    auto points = landmarks; // [10, N]
    
    // Filter by threshold - matches Python: ipass = score > threshold[2]
    auto ipass = score > thresholds[2];
    
    // Filter points - matches Python: points = points[:, ipass]
    points = points.index({torch::indexing::Slice(), ipass}); // [10, M]
    
    // Create boxes with scores - matches Python: boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
    auto boxes_filtered = torch::cat({
        boxes.index({ipass, torch::indexing::Slice(torch::indexing::None, 4)}),
        score.index({ipass}).unsqueeze(1)
    }, 1); // [M, 5]
    
    // Filter image_inds - matches Python: image_inds = image_inds[ipass]
    auto image_inds_filtered = image_inds.index({ipass});
    
    // Filter regression - matches Python: mv = out0[:, ipass].permute(1, 0)
    auto mv = reg.index({torch::indexing::Slice(), ipass}).permute({1, 0}); // [M, 4]
    
    if (boxes_filtered.size(0) == 0) {
        return {torch::zeros({0, 5}), torch::zeros({0}, torch::kLong), torch::zeros({0, 10})};
    }
    
    // Calculate landmark coordinates - matches Python lines 264-268
    auto w_i = boxes_filtered.index({torch::indexing::Slice(), 2}) - boxes_filtered.index({torch::indexing::Slice(), 0}) + 1;
    auto h_i = boxes_filtered.index({torch::indexing::Slice(), 3}) - boxes_filtered.index({torch::indexing::Slice(), 1}) + 1;
    
    // points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
    auto points_x = w_i.repeat({5, 1}) * points.index({torch::indexing::Slice(torch::indexing::None, 5), torch::indexing::Slice()})
                    + boxes_filtered.index({torch::indexing::Slice(), 0}).repeat({5, 1}) - 1;
    
    // points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
    auto points_y = h_i.repeat({5, 1}) * points.index({torch::indexing::Slice(5, 10), torch::indexing::Slice()})
                    + boxes_filtered.index({torch::indexing::Slice(), 1}).repeat({5, 1}) - 1;
    
    // points = torch.stack((points_x, points_y)).permute(2, 1, 0)
    auto landmarks_out = torch::stack({points_x, points_y}).permute({2, 1, 0}); // [M, 5, 2]
    
    // Flatten landmarks to [M, 10]
    landmarks_out = landmarks_out.reshape({landmarks_out.size(0), 10});
    
    // Apply bounding box regression - matches Python: boxes = bbreg(boxes, mv)
    boxes_filtered = mtcnn_utils::bbreg(boxes_filtered, mv);
    
    // NMS within each image using "Min" strategy - matches Python: pick = batched_nms_numpy(..., 'Min')
    auto pick = mtcnn_utils::batched_nms(
        boxes_filtered.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 4)}),
        boxes_filtered.index({torch::indexing::Slice(), 4}),
        image_inds_filtered,
        0.7f,
        "Min"
    );
    
    // Apply pick - matches Python: boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]
    boxes_filtered = boxes_filtered.index({pick});
    image_inds_filtered = image_inds_filtered.index({pick});
    landmarks_out = landmarks_out.index({pick});
    
    return {boxes_filtered, image_inds_filtered, landmarks_out};
}

// Main detection pipeline
mtcnn_utils::DetectionResult MTCNN::detect_face(const torch::Tensor& img) {
    mtcnn_utils::DetectionResult result;
    
    // Stage 1: PNet - Process image pyramid
    auto [boxes1, image_inds1] = stage_one(img);
    
    if (boxes1.size(0) == 0) {
        result.boxes = torch::zeros({0, 4});
        result.probs = torch::zeros({0});
        result.landmarks = torch::zeros({0, 10});
        return result;
    }
    
    // Stage 2: RNet - Refine detections
    auto [boxes2, image_inds2] = stage_two(img, boxes1, image_inds1);
    
    if (boxes2.size(0) == 0) {
        result.boxes = torch::zeros({0, 4});
        result.probs = torch::zeros({0});
        result.landmarks = torch::zeros({0, 10});
        return result;
    }
    
    // Stage 3: ONet - Final detection with landmarks
    auto [boxes3, image_inds3, landmarks] = stage_three(img, boxes2, image_inds2);
    
    // Fill result
    result.boxes = boxes3.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)});
    result.probs = boxes3.index({torch::indexing::Slice(), 4});
    result.landmarks = landmarks;
    
    return result;
}

// Public detect API
mtcnn_utils::DetectionResult MTCNN::detect(cv::Mat& img, bool landmarks) {
    // Convert OpenCV Mat to torch::Tensor
    auto img_tensor = mtcnn_utils::cv_to_tensor(img, torch_device);
    
    return detect_face(img_tensor);
}

// Forward method (placeholder for face extraction)
cv::Mat MTCNN::forward(cv::Mat& img) {
    auto result = detect(img, true);
    
    // For now, just return the original image
    // In a full implementation, this would extract and return face crops
    return img;
}
