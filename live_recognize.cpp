#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "src/recognizer/face_recognizer.hpp"
#include "src/websocket/config.hpp"

int main(int argc, char** argv) {
    // Open video capture (default to webcam 0, or use video file if provided)
    cv::VideoCapture cap;
    
    if (argc > 1) {
        // If a file path is provided, try to open it as a video file
        std::string video_path = argv[1];
        cap.open(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video file: " << video_path << std::endl;
            return 1;
        }
        std::cout << "Opening video file: " << video_path << std::endl;
    } else {
        // Otherwise, open webcam
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open webcam" << std::endl;
            return 1;
        }
        std::cout << "Opening webcam..." << std::endl;
    }
    
    // Load configuration
    Config config;
    config.load("config.ini");
    
    // Initialize face recognizer
    FaceRecognizer recognizer(config);
    
    std::cout << "Face recognition initialized. Press 'q' or ESC to quit." << std::endl;
    
    cv::Mat frame;
    int frame_count = 0;
    double total_time = 0.0;
    
    while (true) {
        // Read frame from video capture
        cap >> frame;
        
        if (frame.empty()) {
            std::cout << "End of video or failed to capture frame" << std::endl;
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();
        // Process frame to detect and recognize faces
        std::vector<RecognitionResult> results = recognizer.processFrame(frame, frame_count);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        total_time += duration.count();
        
        // Prepare data for drawing
        std::vector<rectPoints> data;

        if (config.visualize) {
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
            }
            
            double fps = (frame_count + 1) / (total_time / 1000.0);
            // Draw results
            cv::Mat resultImg = drawRectsAndPoints(frame, data, fps);
            
            // Display the result
            cv::imshow("Face Recognition - Live", resultImg);
        }
        
        // Print recognition results (only print when faces are detected)
        if (!results.empty()) {
            std::cout << "Frame " << frame_count << ": Detected " << results.size() << " faces" << std::endl;
            for (const auto& result : results) {
                std::cout << "  Face " << result.tracker_id << ": " 
                          << result.name << " (confidence: " << result.confidence 
                          << ", status: " << result.status << ")" << std::endl;
            }
        }


        // Check for exit key
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 'Q' || key == 27) {  // 'q' or ESC key
            std::cout << "Exiting..." << std::endl;
            break;
        }
        
        frame_count++;
        if (config.benchmark) {
            if (frame_count >= 1000) {
                std::cout << "Reached frame limit of 1000" << std::endl;
                break;
            }
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "Processed " << frame_count << " frames" << std::endl;
    std::cout << "Average latency: " << total_time / frame_count << " ms" << std::endl;
    std::cout << "Average FPS: " << frame_count / (total_time / 1000.0) << std::endl;
    
    return 0;
}

