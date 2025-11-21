#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Forward declaration
class Base64;

namespace ImageUtils {
    /**
     * Decode image from base64 string
     */
    cv::Mat decodeBase64Image(const std::string& base64_str);
    
    /**
     * Decode image from binary data
     */
    cv::Mat decodeBinaryImage(const std::vector<uint8_t>& data);
    
    /**
     * Ensure image is in correct format (BGR, uint8)
     */
    cv::Mat ensureCorrectFormat(const cv::Mat& img);
}

#endif // IMAGE_UTILS_HPP
