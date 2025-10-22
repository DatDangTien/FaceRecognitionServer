#ifndef BASE64_HPP
#define BASE64_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class Base64 {
public:
    static std::vector<uint8_t> decode(const std::string& encoded_string) {
        static const std::string base64_chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";
        
        std::string encoded = encoded_string;
        
        // Remove data URL prefix if present
        size_t comma_pos = encoded.find(',');
        if (comma_pos != std::string::npos && encoded.substr(0, 5) == "data:") {
            encoded = encoded.substr(comma_pos + 1);
        }
        
        // Remove whitespace
        encoded.erase(std::remove_if(encoded.begin(), encoded.end(), ::isspace), encoded.end());
        
        std::vector<uint8_t> decoded;
        std::vector<int> T(256, -1);
        for (int i = 0; i < 64; i++) T[base64_chars[i]] = i;
        
        int val = 0, valb = -8;
        for (unsigned char c : encoded) {
            if (T[c] == -1) break;
            val = (val << 6) + T[c];
            valb += 6;
            if (valb >= 0) {
                decoded.push_back(char((val >> valb) & 0xFF));
                valb -= 8;
            }
        }
        
        return decoded;
    }
    
    static cv::Mat decodeToMat(const std::string& encoded_string) {
        std::vector<uint8_t> decoded = decode(encoded_string);
        if (decoded.empty()) {
            return cv::Mat();
        }
        
        // Decode JPEG image
        cv::Mat img = cv::imdecode(decoded, cv::IMREAD_COLOR);
        return img;
    }
};

#endif // BASE64_HPP

