#include <iostream>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "src/recognizer/face_recognizer.hpp"
#include "src/utils/config.hpp"

int main(int argc, char** argv) {
    // Open video capture (default to webcam 0, or use video file if provided)
    cv::VideoCapture cap;
    
    if (argc > 1) {
        // If argument is a number, try to open it as a webcam index
        if (isdigit(argv[1][0])) {
            cap.open(std::stoi(argv[1]), cv::CAP_V4L2);
            if (!cap.isOpened()) {
                std::cerr << "Failed to open webcam: " << argv[1] << std::endl;
                return 1;
            }
            std::cout << "Opening webcam: " << argv[1] << std::endl;
        }
        else {
            // If a file path is provided, try to open it as a video file
            std::string video_path = argv[1];
            cap.open(video_path);
            if (!cap.isOpened()) {
                std::cerr << "Failed to open video file: " << video_path << std::endl;
                return 1;
            }
            std::cout << "Opening video file: " << video_path << std::endl;
        }
    } else {
        // Otherwise, open webcam
        cap.open(0, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open webcam" << std::endl;
            return 1;
        }
        std::cout << "Opening webcam..." << std::endl;
    }
    
    // Set additional properties for better stability
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    // Load configuration
    Config config;
    config.load("config.ini");

    // Create save directory if it doesn't exist
    if (config.save_output) {
        if (!std::filesystem::exists(config.save_dir)) {
            std::filesystem::create_directories(config.save_dir);
        }
    }

    // Prepare video writer (initialize lazily after first frame to get correct size)
    cv::VideoWriter video;
    bool writer_initialized = false;
    
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

        // Build annotations when either visualizing or saving output
        cv::Mat resultImg;
        if (config.visualize || config.save_output) {
            for (const auto& result : results) {
                cv::Rect rect(
                    static_cast<int>(result.xmin),
                    static_cast<int>(result.ymin),
                    static_cast<int>(result.xmax - result.xmin),
                    static_cast<int>(result.ymax - result.ymin)
                );

                std::vector<cv::Point> pts = result.landmarks;

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
            resultImg = drawRectsAndPoints(frame, data, fps);

            // Initialize writer when first needed with correct size and FPS
            if (config.save_output && !writer_initialized) {
                std::string output_path = config.save_dir;
                if (!output_path.empty() && output_path.back() != '/' && output_path.back() != '\\') {
                    output_path += "/";
                }
                output_path += "output.mp4";

                double cap_fps = cap.get(cv::CAP_PROP_FPS);
                if (cap_fps <= 0 || std::isnan(cap_fps) || std::isinf(cap_fps)) {
                    cap_fps = 30.0; // fallback
                }

                // Use MJPG codec for better compatibility (change extension to .avi)
                output_path.replace(output_path.length() - 4, 4, ".avi");
                int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
                if (!video.open(output_path, fourcc, cap_fps, resultImg.size())) {
                    std::cerr << "Failed to open video writer at " << output_path << std::endl;
                    return 1;
                }
                std::cout << "Video writer opened: " << output_path << std::endl;
                writer_initialized = true;
            }

            if (config.visualize) {
                cv::imshow("Face Recognition - Live", resultImg);
            }
            if (config.save_output && writer_initialized) {
                video.write(resultImg);
            }
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
    if (config.save_output) {
        video.release();
    }
    cv::destroyAllWindows();
    
    std::cout << "Processed " << frame_count << " frames" << std::endl;
    std::cout << "Average latency: " << total_time / frame_count << " ms" << std::endl;
    std::cout << "Average FPS: " << frame_count / (total_time / 1000.0) << std::endl;
    
    return 0;
}

