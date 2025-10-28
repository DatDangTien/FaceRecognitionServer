#pragma once

#include <opencv2/opencv.hpp>
#include <../dnn/mtcnn/detector.h>
#include <../dnn/mtcnn/onnx_module.h>
#include <../postgres/postgres.hpp>
#include <../utils/face_quality.hpp>
#include <../utils/config.hpp>
#include <../dnn/draw.hpp>

struct RecognitionResult {
    std::string name;
    float confidence;
    std::string status;
    float xmin, ymin, xmax, ymax;  // Bbox coordinates as floats
    int tracker_id;
    int person_id;
    std::vector<cv::Point> landmarks;  // Face landmarks for drawing
    
    std::string toJSON() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "{"
            << "\"name\": \"" << name << "\","
            << "\"confidence\": " << confidence << ","
            << "\"status\": \"" << status << "\","
            << "\"bbox\": {"
            << "\"xmin\": " << xmin << ","
            << "\"ymin\": " << ymin << ","
            << "\"xmax\": " << xmax << ","
            << "\"ymax\": " << ymax
            << "},"
            << "\"tracker_id\": " << tracker_id
            << "}";
        return oss.str();
    }
};

class FaceRecognizer {
    private:
        std::unique_ptr<MTCNNDetector> detector_;
        std::unique_ptr<SubNet> inception_net_;
        std::unique_ptr<Postgres> db_;
        bool quality_check_;
        std::unique_ptr<FaceQuality> quality_checker_;
        Ort::Env env_;
        Ort::SessionOptions session_options_;
        Ort::MemoryInfo memory_info_;
        std::vector<int64_t> input_shape_;
        Config config_;
    
    public:
        FaceRecognizer(const Config& config);
        ~FaceRecognizer();
        DBPerson processFace(const cv::Mat& faceRoi);
        std::vector<RecognitionResult> processFrame(const cv::Mat& frame, int tracker_id = 0);
        bool registerFace(const cv::Mat& frame, const std::string& name);
};