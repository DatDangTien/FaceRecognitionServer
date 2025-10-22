#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <nlohmann/json.hpp>

#include "src/dnn/mtcnn/detector.h"
#include "src/dnn/mtcnn/onnx_module.h"
#include "src/postgres/postgres.hpp"
#include "src/websocket/config.hpp"
#include "src/websocket/face_quality.hpp"
#include "src/websocket/base64.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

// Statistics structure
struct ServerStats {
    std::atomic<uint64_t> total_frames_processed{0};
    std::atomic<uint64_t> total_inferences_run{0};
    std::atomic<uint64_t> successful_recognitions{0};
    std::atomic<uint64_t> quality_rejections{0};
    std::atomic<uint64_t> similarity_rejections{0};
    
    std::string toJSON() const {
        std::ostringstream oss;
        oss << "\"stats\": {"
            << "\"total_frames_processed\": " << total_frames_processed.load() << ","
            << "\"total_inferences_run\": " << total_inferences_run.load() << ","
            << "\"successful_recognitions\": " << successful_recognitions.load() << ","
            << "\"quality_rejections\": " << quality_rejections.load() << ","
            << "\"similarity_rejections\": " << similarity_rejections.load()
            << "}";
        return oss.str();
    }
};

// Face recognition result
struct RecognitionResult {
    std::string name;
    float confidence;
    std::string status;
    float xmin, ymin, xmax, ymax;  // Bbox coordinates as floats
    int tracker_id;
    
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

// JSON parsing using nlohmann/json
using json = nlohmann::json;

// Face Recognizer class
class FaceRecognizer {
private:
    std::unique_ptr<MTCNNDetector> detector_;
    std::unique_ptr<SubNet> inception_net_;
    std::unique_ptr<Postgres> db_;
    std::unique_ptr<FaceQuality> quality_checker_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_;
    std::vector<int64_t> input_shape_;
    Config config_;
    std::mutex mutex_;
    
public:
    FaceRecognizer(const Config& config) 
        : env_(ORT_LOGGING_LEVEL_ERROR, "face_embedding"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)),
          config_(config) {
        
        // Initialize MTCNN
        ProposalNetwork::Config pConfig;
        pConfig.caffeModel = config_.models_path + "/det1.caffemodel";
        pConfig.protoText = config_.models_path + "/det1.prototxt";
        pConfig.threshold = 0.6f;
        
        RefineNetwork::Config rConfig;
        rConfig.caffeModel = config_.models_path + "/det2.caffemodel";
        rConfig.protoText = config_.models_path + "/det2.prototxt";
        rConfig.threshold = 0.7f;
        
        OutputNetwork::Config oConfig;
        oConfig.caffeModel = config_.models_path + "/det3.caffemodel";
        oConfig.protoText = config_.models_path + "/det3.prototxt";
        oConfig.threshold = 0.7f;
        
        detector_ = std::make_unique<MTCNNDetector>(pConfig, rConfig, oConfig);
        
        // Initialize ONNX Runtime
        inception_net_ = std::make_unique<SubNet>(env_, session_options_, config_.inception_model_path);
        input_shape_ = {1, 3, 160, 160};
        
        // Initialize database
        db_ = std::make_unique<Postgres>(
            config_.db_host,
            config_.db_port,
            config_.db_name,
            config_.db_user,
            config_.db_password
        );
        
        // Initialize quality checker
        quality_checker_ = std::make_unique<FaceQuality>(
            config_.blur_threshold,
            config_.min_face_size,
            config_.dark_ratio_threshold,
            config_.bright_ratio_threshold,
            config_.pose_threshold,
            config_.quality_threshold
        );
        
        std::cout << "Face recognizer initialized successfully" << std::endl;
    }
    
    std::vector<RecognitionResult> processFrame(const cv::Mat& frame, ServerStats& stats, int tracker_id = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<RecognitionResult> results;
        
        if (frame.empty()) {
            std::cerr << "Empty frame received" << std::endl;
            return results;
        }
        
        // Detect faces
        std::vector<Face> faces;
        try {
            faces = detector_->detect(frame, 20.f, 0.709f);
        } catch (const std::exception& e) {
            std::cerr << "Face detection error: " << e.what() << std::endl;
            return results;
        }
        
        stats.total_frames_processed++;
        
        // Process each detected face
        for (size_t i = 0; i < faces.size(); ++i) {
            RecognitionResult result;
            cv::Rect face_bbox = faces[i].bbox.getRect();
            result.tracker_id = tracker_id;
            
            // Clamp bounding box to image bounds
            int x = std::max(0, std::min(face_bbox.x, frame.cols - 1));
            int y = std::max(0, std::min(face_bbox.y, frame.rows - 1));
            int w = std::max(1, std::min(face_bbox.width, frame.cols - x));
            int h = std::max(1, std::min(face_bbox.height, frame.rows - y));
            
            // Set bbox coordinates as floats
            result.xmin = static_cast<float>(x);
            result.ymin = static_cast<float>(y);
            result.xmax = static_cast<float>(x + w);
            result.ymax = static_cast<float>(y + h);
            
            
            try {
                // Extract face ROI using clamped coordinates
                cv::Rect roi_rect(x, y, w, h);
                cv::Mat faceRoi = frame(roi_rect);
                
                // Validate quality
                QualityResult quality = quality_checker_->validate(faceRoi);
                
                if (!quality.is_good_quality) {
                    result.name = "Poor Quality";
                    result.confidence = quality.quality_score * 100.0f;
                    result.status = "poor_quality";
                    stats.quality_rejections++;
                    results.push_back(result);
                    continue;
                }
                
                // Preprocess face
                std::vector<float> faceVector = detector_->forward(faceRoi);
                
                if (faceVector.size() != (160 * 160 * 3)) {
                    result.name = "Invalid";
                    result.confidence = 0.0f;
                    result.status = "poor_quality";
                    stats.quality_rejections++;
                    results.push_back(result);
                    continue;
                }
                
                // Create ONNX tensor
                Ort::Value face_tensor = Ort::Value::CreateTensor<float>(
                    memory_info_,
                    const_cast<float*>(faceVector.data()),
                    faceVector.size(),
                    input_shape_.data(),
                    input_shape_.size()
                );
                
                // Run inference
                std::vector<Ort::Value> outputs = inception_net_->forward(face_tensor);
                std::vector<int64_t> output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
                float* embedding = outputs[0].GetTensorMutableData<float>();
                
                
                stats.total_inferences_run++;
                
                // Query database for recognition
                PostgresPerson person = db_->get_recognition(
                    std::vector<float>(embedding, embedding + output_shape[1]),
                    config_.recognition_threshold
                );
                
                if (person.confidence != 0.0) {
                    result.name = person.name;
                    result.confidence = person.confidence;
                    result.status = "recognized";
                    stats.successful_recognitions++;
                } else {
                    result.name = "Unknown";
                    result.confidence = 0.0f;
                    result.status = "unknown";
                    stats.similarity_rejections++;
                }
                
                results.push_back(result);
                
            } catch (const std::exception& e) {
                std::cerr << "Face processing error: " << e.what() << std::endl;
                result.name = "Error";
                result.confidence = 0.0f;
                result.status = "error";
                results.push_back(result);
            }
        }
        
        return results;
    }
};

// WebSocket session class
class Session : public std::enable_shared_from_this<Session> {
private:
    websocket::stream<beast::tcp_stream> ws_;
    beast::flat_buffer buffer_;
    std::shared_ptr<FaceRecognizer> recognizer_;
    ServerStats& stats_;
    uint64_t session_id_;
    
public:
    Session(tcp::socket&& socket, std::shared_ptr<FaceRecognizer> recognizer, ServerStats& stats, uint64_t session_id)
        : ws_(std::move(socket)),
          recognizer_(recognizer),
          stats_(stats),
          session_id_(session_id) {
    }
    
    void run() {
        net::dispatch(
            ws_.get_executor(),
            beast::bind_front_handler(
                &Session::on_run,
                shared_from_this()
            )
        );
    }
    
    void on_run() {
        ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::server));
        ws_.set_option(websocket::stream_base::decorator(
            [](websocket::response_type& res) {
                res.set(http::field::server, "Face Recognition WebSocket Server");
            }
        ));
        
        ws_.async_accept(
            beast::bind_front_handler(
                &Session::on_accept,
                shared_from_this()
            )
        );
    }
    
    void on_accept(beast::error_code ec) {
        if (ec) {
            std::cerr << "Accept error: " << ec.message() << std::endl;
            return;
        }
        
        std::cout << "Client " << session_id_ << " connected" << std::endl;
        do_read();
    }
    
    void do_read() {
        ws_.async_read(
            buffer_,
            beast::bind_front_handler(
                &Session::on_read,
                shared_from_this()
            )
        );
    }
    
    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        
        if (ec == websocket::error::closed) {
            std::cout << "Client " << session_id_ << " disconnected" << std::endl;
            return;
        }
        
        if (ec) {
            std::cerr << "Read error: " << ec.message() << std::endl;
            return;
        }
        
        // Process the message
        std::string message = beast::buffers_to_string(buffer_.data());
        buffer_.consume(buffer_.size());
        
        std::string response = processMessage(message);
        
        if (!response.empty()) {
        
            
            // Send response
            ws_.text(true);
            ws_.async_write(
                net::buffer(response),
                beast::bind_front_handler(
                    &Session::on_write,
                    shared_from_this()
                )
            );
        } else {
            // No response to send, continue reading next message
            do_read();
        }
    }
    
    void on_write(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        
        if (ec) {
            std::cerr << "Write error: " << ec.message() << std::endl;
            return;
        }
        
        
        // Read next message
        do_read();
    }
    
    std::string processMessage(const std::string& message) {
        try {
            
            // Parse JSON
            json data = json::parse(message);
            std::string type = data.value("type", "");
            
            if (type == "frame") {
                auto start_time = std::chrono::high_resolution_clock::now();
                
                int frame_id = data.value("frame_id", 0);
                std::string frame_base64 = data.value("frame", "");
                int tracker_id = data.value("tracker_id", 0);
                
                // Debug: Print base64 length and first few characters
                
                // Extract bbox if present
                float bbox_xmin = 0.0f;
                float bbox_ymin = 0.0f;
                if (data.contains("bbox")) {
                    auto bbox = data["bbox"];
                    bbox_xmin = bbox.value("xmin", 0.0f);
                    bbox_ymin = bbox.value("ymin", 0.0f);
                }
                
                // Decode frame
                cv::Mat frame = Base64::decodeToMat(frame_base64);
                
                if (frame.empty()) {
                    return buildErrorResponse(frame_id, "Failed to decode frame");
                }
                
                // Debug: Print image shape
                
                // Process frame
                std::vector<RecognitionResult> results = recognizer_->processFrame(frame, stats_, tracker_id);
                
                // Adjust bbox coordinates if bbox was provided (like Python identify_bbox function)
                if (bbox_xmin != 0.0f || bbox_ymin != 0.0f) {
                    for (auto& result : results) {
                        // Reposition bbox coordinates
                        result.xmin += bbox_xmin;
                        result.ymin += bbox_ymin;
                        result.xmax += bbox_xmin;
                        result.ymax += bbox_ymin;
                        
                    }
                }
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                float processing_time = duration.count() / 1000.0f;
                
                return buildResultResponse(frame_id, results, processing_time, tracker_id);
                
            } else if (type == "get_stats") {
                return buildStatsResponse();
            }
            
            return buildErrorResponse(0, "Unknown message type");
            
        } catch (const std::exception& e) {
            std::cerr << "Message processing error: " << e.what() << std::endl;
            return buildErrorResponse(0, std::string("Processing error: ") + e.what());
        }
    }
    
    std::string buildResultResponse(int frame_id, const std::vector<RecognitionResult>& results, float processing_time, int tracker_id) {
        
        // Filter out "Poor Quality" results
        std::vector<RecognitionResult> filtered_results;
        for (const auto& result : results) {
            if (result.status != "poor_quality") {
                filtered_results.push_back(result);
            } else {
            }
        }
        
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "{"
            << "\"type\": \"result\","
            << "\"frame_id\": " << frame_id << ","
            << "\"recognition_results\": [";
        
        for (size_t i = 0; i < filtered_results.size(); ++i) {
            if (i > 0) oss << ",";
            oss << filtered_results[i].toJSON();
        }
        
        oss << "],"
            << "\"processing_time\": " << processing_time << ","
            << "\"tracker_id\": " << tracker_id << ","
            << stats_.toJSON()
            << "}";
        
        std::string response = oss.str();
        
        return response;
    }
    
    std::string buildErrorResponse(int frame_id, const std::string& error_message) {
        std::ostringstream oss;
        oss << "{"
            << "\"type\": \"error\","
            << "\"frame_id\": " << frame_id << ","
            << "\"error\": \"" << error_message << "\""
            << "}";
        return oss.str();
    }
    
    std::string buildStatsResponse() {
        std::ostringstream oss;
        oss << "{"
            << "\"type\": \"stats\","
            << stats_.toJSON()
            << "}";
        return oss.str();
    }
};

// Listener class
class Listener : public std::enable_shared_from_this<Listener> {
private:
    net::io_context& ioc_;
    tcp::acceptor acceptor_;
    std::shared_ptr<FaceRecognizer> recognizer_;
    ServerStats& stats_;
    std::atomic<uint64_t> session_counter_{0};
    
public:
    Listener(net::io_context& ioc, tcp::endpoint endpoint, 
             std::shared_ptr<FaceRecognizer> recognizer, ServerStats& stats)
        : ioc_(ioc),
          acceptor_(net::make_strand(ioc)),
          recognizer_(recognizer),
          stats_(stats) {
        
        beast::error_code ec;
        
        acceptor_.open(endpoint.protocol(), ec);
        if (ec) {
            std::cerr << "Open error: " << ec.message() << std::endl;
            return;
        }
        
        acceptor_.set_option(net::socket_base::reuse_address(true), ec);
        if (ec) {
            std::cerr << "Set option error: " << ec.message() << std::endl;
            return;
        }
        
        acceptor_.bind(endpoint, ec);
        if (ec) {
            std::cerr << "Bind error: " << ec.message() << std::endl;
            return;
        }
        
        acceptor_.listen(net::socket_base::max_listen_connections, ec);
        if (ec) {
            std::cerr << "Listen error: " << ec.message() << std::endl;
            return;
        }
    }
    
    void run() {
        do_accept();
    }
    
private:
    void do_accept() {
        acceptor_.async_accept(
            net::make_strand(ioc_),
            beast::bind_front_handler(
                &Listener::on_accept,
                shared_from_this()
            )
        );
    }
    
    void on_accept(beast::error_code ec, tcp::socket socket) {
        if (ec) {
            std::cerr << "Accept error: " << ec.message() << std::endl;
        } else {
            uint64_t session_id = ++session_counter_;
            std::make_shared<Session>(std::move(socket), recognizer_, stats_, session_id)->run();
        }
        
        do_accept();
    }
};

// Main function
int main(int argc, char* argv[]) {
    try {
        // Load configuration
        Config config;
        std::string config_file = "config.ini";
        if (argc > 1) {
            config_file = argv[1];
        }
        config.load(config_file);
        
        // Create face recognizer
        std::cout << "Initializing face recognizer..." << std::endl;
        auto recognizer = std::make_shared<FaceRecognizer>(config);
        
        // Create statistics
        ServerStats stats;
        
        // Create IO context
        auto const num_threads = std::max<int>(1, std::thread::hardware_concurrency());
        net::io_context ioc{num_threads};
        
        // Create and launch listener
        auto const address = net::ip::make_address(config.server_host);
        auto const port = static_cast<unsigned short>(config.server_port);
        
        std::cout << "Starting WebSocket server on " << config.server_host << ":" << config.server_port << std::endl;
        
        std::make_shared<Listener>(
            ioc,
            tcp::endpoint{address, port},
            recognizer,
            stats
        )->run();
        
        std::cout << "Server started successfully with " << num_threads << " threads" << std::endl;
        
        // Run the I/O service on multiple threads
        std::vector<std::thread> threads;
        threads.reserve(num_threads - 1);
        for (int i = 0; i < num_threads - 1; ++i) {
            threads.emplace_back([&ioc] { ioc.run(); });
        }
        ioc.run();
        
        for (auto& t : threads) {
            t.join();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

