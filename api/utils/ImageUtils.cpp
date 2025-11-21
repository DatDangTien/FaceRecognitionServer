#include "ImageUtils.hpp"
#include "../../src/utils/base64.hpp"
#include <opencv2/opencv.hpp>

namespace ImageUtils {
    cv::Mat decodeBase64Image(const std::string& base64_str) {
        return Base64::decodeToMat(base64_str);
    }
    
    cv::Mat decodeBinaryImage(const std::vector<uint8_t>& data) {
        if (data.empty()) {
            return cv::Mat();
        }
        return cv::imdecode(data, cv::IMREAD_COLOR);
    }
    
    cv::Mat ensureCorrectFormat(const cv::Mat& img) {
        if (img.empty()) {
            return img;
        }
        
        cv::Mat result = img;
        
        // Ensure uint8
        if ((result.type() & CV_MAT_DEPTH_MASK) != CV_8U) {
            result.convertTo(result, CV_8U);
        }
        
        // Ensure BGR format (3 channels)
        if (result.channels() == 1) {
            cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
        } else if (result.channels() == 4) {
            cv::cvtColor(result, result, cv::COLOR_RGBA2BGR);
        } 
        // else if (result.channels() == 3) {
        //     // Assume RGB, convert to BGR
        //     cv::cvtColor(result, result, cv::COLOR_RGB2BGR);
        // }
        
        return result;
    }
}

