#ifndef FACE_QUALITY_HPP
#define FACE_QUALITY_HPP

#include <opencv2/opencv.hpp>

struct QualityResult {
    bool is_good_quality;
    float quality_score;
    float blur_score;
    bool blur_pass;
    bool size_pass;
    bool lighting_pass;
    bool pose_pass;
    
    QualityResult() 
        : is_good_quality(false), quality_score(0.0f), blur_score(0.0f),
          blur_pass(false), size_pass(false), lighting_pass(false), pose_pass(false) {}
};

class FaceQuality {
public:
    FaceQuality(float blur_threshold = 100.0f,
                float min_face_size = 60.0f,
                float dark_ratio_threshold = 0.4f,
                float bright_ratio_threshold = 0.3f,
                float pose_threshold = 1000.0f,
                float quality_threshold = 0.5f);
    
    QualityResult validate(const cv::Mat& face_roi);
    
private:
    float blur_threshold_;
    float min_face_size_;
    float dark_ratio_threshold_;
    float bright_ratio_threshold_;
    float pose_threshold_;
    float quality_threshold_;
    
    float checkBlur(const cv::Mat& gray);
    bool checkSize(const cv::Mat& face);
    bool checkLighting(const cv::Mat& gray);
    bool checkPose(const cv::Mat& gray);
};

#endif // FACE_QUALITY_HPP

