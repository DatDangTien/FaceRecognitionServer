#include "face_quality.hpp"
#include <iostream>

FaceQuality::FaceQuality(float blur_threshold,
                         float min_face_size,
                         float dark_ratio_threshold,
                         float bright_ratio_threshold,
                         float pose_threshold,
                         float quality_threshold)
    : blur_threshold_(blur_threshold),
      min_face_size_(min_face_size),
      dark_ratio_threshold_(dark_ratio_threshold),
      bright_ratio_threshold_(bright_ratio_threshold),
      pose_threshold_(pose_threshold),
      quality_threshold_(quality_threshold) {
}

float FaceQuality::checkBlur(const cv::Mat& gray) {
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    double variance = stddev.val[0] * stddev.val[0];
    return static_cast<float>(variance);
}

bool FaceQuality::checkSize(const cv::Mat& face) {
    return face.rows >= min_face_size_ && face.cols >= min_face_size_;
}

bool FaceQuality::checkLighting(const cv::Mat& gray) {
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    float total_pixels = gray.rows * gray.cols;
    
    // Count dark pixels (0-49)
    float dark_pixels = 0;
    for (int i = 0; i < 50; i++) {
        dark_pixels += hist.at<float>(i);
    }
    
    // Count bright pixels (200-255)
    float bright_pixels = 0;
    for (int i = 200; i < 256; i++) {
        bright_pixels += hist.at<float>(i);
    }
    
    float dark_ratio = dark_pixels / total_pixels;
    float bright_ratio = bright_pixels / total_pixels;
    
    return dark_ratio < dark_ratio_threshold_ && bright_ratio < bright_ratio_threshold_;
}

bool FaceQuality::checkPose(const cv::Mat& gray) {
    int mid_point = gray.cols / 2;
    
    cv::Mat left_half = gray(cv::Rect(0, 0, mid_point, gray.rows));
    cv::Mat right_half = gray(cv::Rect(mid_point, 0, gray.cols - mid_point, gray.rows));
    
    // Flip right half for comparison
    cv::Mat right_half_flipped;
    cv::flip(right_half, right_half_flipped, 1);
    
    // Resize to same dimensions
    int min_width = std::min(left_half.cols, right_half_flipped.cols);
    cv::Mat left_resized, right_resized;
    cv::resize(left_half, left_resized, cv::Size(min_width, gray.rows));
    cv::resize(right_half_flipped, right_resized, cv::Size(min_width, gray.rows));
    
    // Calculate MSE
    cv::Mat diff;
    cv::absdiff(left_resized, right_resized, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    
    cv::Scalar mse_scalar = cv::mean(diff);
    float mse = mse_scalar[0];
    
    return mse < pose_threshold_;
}

QualityResult FaceQuality::validate(const cv::Mat& face_roi) {
    QualityResult result;
    
    if (face_roi.empty()) {
        return result;
    }
    
    // Convert to grayscale if needed
    cv::Mat gray;
    if (face_roi.channels() == 3) {
        cv::cvtColor(face_roi, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = face_roi.clone();
    }
    
    float quality_score = 0.0f;
    int checks_passed = 0;
    int total_checks = 4;
    
    // 1. Blur Detection
    result.blur_score = checkBlur(gray);
    result.blur_pass = result.blur_score > blur_threshold_;
    if (result.blur_pass) {
        quality_score += 0.25f;
        checks_passed++;
    }
    
    // 2. Face Size Validation
    result.size_pass = checkSize(gray);
    if (result.size_pass) {
        quality_score += 0.25f;
        checks_passed++;
    }
    
    // 3. Lighting Condition
    result.lighting_pass = checkLighting(gray);
    if (result.lighting_pass) {
        quality_score += 0.25f;
        checks_passed++;
    }
    
    // 4. Pose Estimation
    result.pose_pass = checkPose(gray);
    if (result.pose_pass) {
        quality_score += 0.25f;
        checks_passed++;
    }
    
    result.quality_score = quality_score;
    result.is_good_quality = quality_score >= quality_threshold_;
    
    return result;
}

