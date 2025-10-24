#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/recognizer/face_recognizer.hpp"
#include "src/websocket/config.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    
    std::string image_path = argv[1];
    
    // Load image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }
    
    std::cout << "Image loaded successfully: " << image_path << std::endl;
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    
    // Load configuration
    Config config;
    config.load("config.ini");
    
    // Initialize face recognizer
    FaceRecognizer recognizer(config);
    
    // Process frame to detect and recognize faces
    std::vector<RecognitionResult> results = recognizer.processFrame(img, 0);
    
    std::cout << "Detected " << results.size() << " faces" << std::endl;
    
    if (results.empty()) {
        std::cout << "No faces detected in the image" << std::endl;
        return 0;
    }
    
    // Prepare data for drawing
    std::vector<rectPoints> data;
    
    for (const auto& result : results) {
        // Create bounding box rectangle
        cv::Rect rect(
            static_cast<int>(result.xmin),
            static_cast<int>(result.ymin),
            static_cast<int>(result.xmax - result.xmin),
            static_cast<int>(result.ymax - result.ymin)
        );
        
        // Use actual face landmarks from detection
        std::vector<cv::Point> pts = result.landmarks;
        
        // Create annotation text
        std::string annotation;
        if (result.status == "recognized") {
            annotation = "ID:" + std::to_string(result.person_id) + 
                        " " + result.name + 
                        " (" + std::to_string(static_cast<int>(result.confidence * 100)) + "%)";
        } else if (result.status == "poor_quality") {
            annotation = "Poor Quality (" + std::to_string(static_cast<int>(result.confidence * 100)) + "%)";
        } else if (result.status == "unknown") {
            annotation = "Unknown";
        } else {
            annotation = result.status;
        }
        
        data.push_back(std::make_tuple(rect, pts, annotation));
        
        // Print recognition results
        std::cout << "Face " << result.tracker_id << ": " 
                  << result.name << " (confidence: " << result.confidence 
                  << ", status: " << result.status << ")" << std::endl;
    }
    
    // Draw results
    cv::Mat resultImg = drawRectsAndPoints(img, data);
    
    // Display the result
    cv::imshow("Face Recognition Results", resultImg);
    std::cout << "Press any key to close the window..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}
