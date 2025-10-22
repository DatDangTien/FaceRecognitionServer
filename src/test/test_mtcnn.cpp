#include "../onnx/mtcnn.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    // Default image path
    std::string image_path = "../../data/anh-son-tung-mtp-thumb.jpg";
    
    if (argc > 1) {
        image_path = argv[1];
    }
    
    // Load image
    cv::Mat img = cv::imread(image_path);
    
    if (img.empty()) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Image loaded: " << img.cols << "x" << img.rows << std::endl;
    
    // Initialize MTCNN
    std::cout << "Initializing MTCNN..." << std::endl;
    
    std::vector<float> thresholds = {0.6f, 0.7f, 0.7f};
    MTCNN mtcnn(
        160,           // image_size
        0,             // margin
        20,            // min_face_size
        thresholds,    // thresholds
        0.709f,        // factor
        true,          // post_process
        true,          // select_largest
        "largest",     // selection_method
        false,         // keep_all
        "cuda"         // device (use "cpu" if CUDA not available)
    );
    
    std::cout << "MTCNN initialized successfully" << std::endl;
    
    // Detect faces
    std::cout << "Running face detection..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    mtcnn_utils::DetectionResult result = mtcnn.detect(img, true);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Detection completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Number of faces detected: " << result.boxes.size(0) << std::endl;
    
    // Draw results
    cv::Mat img_display = img.clone();
    
    // Convert tensors to CPU for display
    auto boxes_cpu = result.boxes.to(torch::kCPU);
    auto probs_cpu = result.probs.to(torch::kCPU);
    auto landmarks_cpu = result.landmarks.to(torch::kCPU);
    
    int num_faces = boxes_cpu.size(0);
    
    for (int i = 0; i < num_faces; i++) {
        auto box = boxes_cpu[i];
        float x1 = box[0].item<float>();
        float y1 = box[1].item<float>();
        float x2 = box[2].item<float>();
        float y2 = box[3].item<float>();
        float prob = probs_cpu[i].item<float>();
        
        std::cout << "\nFace " << i + 1 << ":" << std::endl;
        std::cout << "  Box: [" << x1 << ", " << y1 << ", " 
                  << x2 << ", " << y2 << "]" << std::endl;
        std::cout << "  Confidence: " << prob << std::endl;
        
        // Draw bounding box
        cv::rectangle(
            img_display,
            cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
            cv::Point(static_cast<int>(x2), static_cast<int>(y2)),
            cv::Scalar(0, 255, 0),
            2
        );
        
        // Draw confidence score
        std::string text = "Face: " + std::to_string(prob).substr(0, 4);
        cv::putText(
            img_display,
            text,
            cv::Point(static_cast<int>(x1), static_cast<int>(y1) - 10),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            2
        );
        
        // Draw landmarks if available
        if (landmarks_cpu.size(0) > 0 && i < landmarks_cpu.size(0)) {
            auto landmark = landmarks_cpu[i];  // [10]
            std::cout << "  Landmarks: ";
            
            for (int j = 0; j < 5; j++) {
                float x = landmark[j].item<float>();
                float y = landmark[j + 5].item<float>();
                
                std::cout << "(" << x << "," << y << ") ";
                
                // Draw landmark points
                cv::circle(
                    img_display,
                    cv::Point(static_cast<int>(x), static_cast<int>(y)),
                    3,
                    cv::Scalar(0, 0, 255),
                    -1
                );
            }
            std::cout << std::endl;
        }
    }
    
    // Save result
    std::string output_path = "mtcnn_detection_result.jpg";
    cv::imwrite(output_path, img_display);
    std::cout << "\nResult saved to: " << output_path << std::endl;
    
    // Display result (if X11 available)
    cv::imshow("MTCNN Face Detection", img_display);
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    
    return 0;
}


