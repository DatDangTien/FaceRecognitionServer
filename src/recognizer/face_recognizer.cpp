#include "face_recognizer.hpp"
#include "../dnn/mtcnn/face.h"

FaceRecognizer::FaceRecognizer(const Config& config) 
    : env_(ORT_LOGGING_LEVEL_ERROR, "face_embedding"),
        memory_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)),
        config_(config) {
    
    // Initialize MTCNN
    ProposalNetwork::Config pConfig;
    pConfig.caffeModel = config_.models_path + "/det1.caffemodel";
    pConfig.protoText = config_.models_path + "/det1.prototxt";
    pConfig.threshold = 0.6f;
    
    RefineNetwork::Config rConfig;
    rConfig.caffeModel = config_.models_path + "/det2.caffemodel";
    rConfig.protoText = config_.models_path + "/det2.prototxt";
    rConfig.threshold = 0.7f;
    
    OutputNetwork::Config oConfig;
    oConfig.caffeModel = config_.models_path + "/det3.caffemodel";
    oConfig.protoText = config_.models_path + "/det3.prototxt";
    oConfig.threshold = 0.7f;
    
    detector_ = std::make_unique<MTCNNDetector>(pConfig, rConfig, oConfig);
    
    // Initialize ONNX Runtime
    inception_net_ = std::make_unique<SubNet>(env_, session_options_, config_.inception_model_path);
    input_shape_ = {1, 3, 160, 160};
    
    // Initialize database
    db_ = std::make_unique<Postgres>(
        config_.db_host,
        config_.db_port,
        config_.db_name,
        config_.db_user,
        config_.db_password
    );
    
    quality_check_ = config_.quality_check;
    // Initialize quality checker
    if (quality_check_) {
        quality_checker_ = std::make_unique<FaceQuality>(
            config_.blur_threshold,
            config_.min_face_size,
            config_.dark_ratio_threshold,
            config_.bright_ratio_threshold,
            config_.pose_threshold,
            config_.quality_threshold
        );
        
        std::cout << "Face recognizer initialized successfully" << std::endl;
    }
    else {
        std::cout << "Quality check disabled" << std::endl;
    }

    if (config_.save_output) {
        if (!std::filesystem::exists(config_.save_dir)) {
            std::filesystem::create_directories(config_.save_dir);
        }
        std::cout << "Save directory: " << config_.save_dir << std::endl;
    }
}

FaceRecognizer::~FaceRecognizer() {
    // Destructor - unique_ptr members will clean themselves up automatically
}
        
DBPerson FaceRecognizer::processFace(const cv::Mat& faceRoi) {
    if (quality_check_) {
    QualityResult quality = quality_checker_->validate(faceRoi);
        if (!quality.is_good_quality) {
            return DBPerson("", "Poor Quality", quality.quality_score, 0.0);
        }
    }
    std::vector<float> faceVector = detector_->preprocessFace(faceRoi);
    if (faceVector.size() != (160 * 160 * 3)) {
        std::cerr << "Invalid face tensor: " << faceVector.size() << std::endl;
        return DBPerson("", "Invalid", 0.0, 0.0);
    }

    Ort::Value face_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(faceVector.data()),
        faceVector.size(),
        input_shape_.data(),
        input_shape_.size()
    );
    std::vector<Ort::Value> outputs = inception_net_->forward(face_tensor);
    std::vector<int64_t> output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    float* embedding = outputs[0].GetTensorMutableData<float>();
    return db_->get_recognition(
        std::vector<float>(embedding, embedding + output_shape[1])
    );
}

std::vector<RecognitionResult> FaceRecognizer::processFrame(const cv::Mat& frame, int tracker_id) {
    std::vector<RecognitionResult> results;
    
    if (frame.empty()) {
        std::cerr << "Empty frame received" << std::endl;
        return results;
    }
    
    // Detect faces
    std::vector<Face> faces;
    try {
        faces = detector_->detect(frame, 20.f, 0.709f);
    } catch (const std::exception& e) {
        std::cerr << "Face detection error: " << e.what() << std::endl;
        return results;
    }
        
    // Process each detected face
    for (size_t i = 0; i < faces.size(); ++i) {
        RecognitionResult result;
        cv::Rect face_bbox = faces[i].bbox.getRect();
        result.tracker_id = tracker_id;
        
        // Clamp bounding box to image bounds
        int x = std::max(0, std::min(face_bbox.x, frame.cols - 1));
        int y = std::max(0, std::min(face_bbox.y, frame.rows - 1));
        int w = std::max(1, std::min(face_bbox.width, frame.cols - x));
        int h = std::max(1, std::min(face_bbox.height, frame.rows - y));
        
        // Set bbox coordinates as floats
        result.xmin = static_cast<float>(x);
        result.ymin = static_cast<float>(y);
        result.xmax = static_cast<float>(x + w);
        result.ymax = static_cast<float>(y + h);
        
        // Extract face landmarks
        result.landmarks.clear();
        for (int p = 0; p < NUM_PTS; ++p) {
            int px = static_cast<int>(faces[i].ptsCoords[2 * p]);
            int py = static_cast<int>(faces[i].ptsCoords[2 * p + 1]);
            // Clamp landmark coordinates to image bounds
            px = std::max(0, std::min(px, frame.cols - 1));
            py = std::max(0, std::min(py, frame.rows - 1));
            result.landmarks.push_back(cv::Point(px, py));
        }
        
        
        try {
            // Extract face ROI using clamped coordinates
            cv::Rect roi_rect(x, y, w, h);
            cv::Mat faceRoi = frame(roi_rect);
            cv::Mat faceRoiRGB;
            cv::cvtColor(faceRoi, faceRoiRGB, cv::COLOR_BGR2RGB);
            DBPerson person = processFace(faceRoiRGB);
            if (person.name == "Poor Quality") {
                result.name = "Poor Quality";
                result.confidence = person.confidence;
                result.status = "poor_quality";
                result.person_id = "";
            }
            else if (person.name == "Invalid") {
                result.name = "";
                result.confidence = 0.0f;
                result.status = "invalid";
                result.person_id = "";
            }
            else if (person.confidence != 0.0 && person.confidence > config_.recognition_threshold) {
                result.name = person.name;
                result.confidence = person.confidence;
                result.status = "recognized";
                result.person_id = person.id;
            }
            else {
                result.name = "Unknown";
                result.confidence = person.confidence;
                result.status = "unknown";
                result.person_id = "";
            }
            results.push_back(result);
        } catch (const std::exception& e) {
            std::cerr << "Face processing error: " << e.what() << std::endl;
        }
    }
    
    return results;
}

 std::pair<bool, std::string> FaceRecognizer::registerFace(const cv::Mat& frame, const std::string& name, const std::string& id) {
    try {
        if (frame.empty()) {
            std::cerr << "Empty frame received" << std::endl;
            return {false, "Empty frame received"};
        }
        std::vector<Face> faces = detector_->detect(frame, 20.f, 0.709f);
        if (faces.empty()) {
            std::cerr << "No faces found in the frame" << std::endl;
            return {false, "No faces found in the frame"};
        }
        if (faces.size() > 1) {
            std::cerr << "Multiple faces found in the frame" << std::endl;
            return {false, "Multiple faces found in the frame"};
        }
        cv::Rect face_bbox = faces[0].bbox.getRect();
        cv::Mat faceRoi = frame(face_bbox);
        if (quality_check_) {
            QualityResult quality = quality_checker_->validate(faceRoi);
            if (!quality.is_good_quality) {
                std::cerr << "Poor quality face detected" << std::endl;
                return {false, "Poor quality face detected"};
            }
        }
        cv::Mat faceRoiRGB;
        cv::cvtColor(faceRoi, faceRoiRGB, cv::COLOR_BGR2RGB);
        std::vector<float> faceVector = detector_->preprocessFace(faceRoiRGB);
        if (faceVector.size() != (160 * 160 * 3)) {
            std::cerr << "Invalid face tensor" << std::endl;
            return {false, "Invalid face tensor"};
        }
        Ort::Value face_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(faceVector.data()),
            faceVector.size(),
            input_shape_.data(),
            input_shape_.size()
        );
        std::vector<Ort::Value> outputs = inception_net_->forward(face_tensor);
        std::vector<int64_t> output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        float* embedding = outputs[0].GetTensorMutableData<float>();
        if (name.empty() && !id.empty()) {
            try {
                db_->update_embedding(id, std::vector<float>(embedding, embedding + output_shape[1]));
            } catch (const std::exception& e) {
                std::cerr << "Update embedding id = " << id << ", error: " << e.what() << std::endl;
                return {false, "Update error: " + std::string(e.what())};
            }
        }
        else if (!name.empty()) {
            try {
                db_->insert_embedding(name, std::vector<float>(embedding, embedding + output_shape[1]));
            } catch (const std::exception& e) {
                std::cerr << "Insert embedding name = " << name << ", error: " << e.what() << std::endl;
                return {false, "Insert error: " + std::string(e.what())};
            }
        }

        // Validate the insert
        DBPerson person = db_->get_recognition(
            std::vector<float>(embedding, embedding + output_shape[1]),
            config_.recognition_threshold
        );
        if (person.confidence > 0.95) {
            std::vector<cv::Point> pts;
            for (int p = 0; p < NUM_PTS; ++p) {
                pts.push_back(cv::Point(faces[0].ptsCoords[2 * p], faces[0].ptsCoords[2 * p + 1]));
            }
            auto rect = faces[0].bbox.getRect();
            std::string anno =  person.id + " " + person.name + ": " + std::to_string(person.confidence).substr(0, 4);
            std::vector<std::tuple<cv::Rect, std::vector<cv::Point>, std::string>> data;
            data.push_back(std::make_tuple(rect, pts, anno));
            auto resultImg = drawRectsAndPoints(frame, data);
            if (config_.visualize) {
                cv::imshow("face-registration", resultImg);
                cv::waitKey(0);
            }
            if (config_.save_output) {
                cv::imwrite(config_.save_dir + "/" + person.name + ".jpg", resultImg);
            }
            std::cout << "Face registered id: " << person.id << " name: " << person.name << " confidence: " << person.confidence << std::endl;
            return {true, name};
        }
        std::cout << "Face similar to " << person.name << " with confidence " << person.confidence << std::endl;
        return {false, "Update verification failed"};

    }
    catch (const std::exception& e) {
        std::cerr << "Face registration error: " << e.what() << std::endl;
        return {false, "Registration error: " + std::string(e.what())};
    }
}