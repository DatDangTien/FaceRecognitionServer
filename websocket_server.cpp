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

#include "src/utils/config.hpp"
#include "src/utils/base64.hpp"
#include "src/recognizer/face_recognizer.hpp"

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

// JSON parsing using nlohmann/json
using json = nlohmann::json;


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
                std::vector<RecognitionResult> results = recognizer_->processFrame(frame, tracker_id);
                stats_.total_frames_processed++;
                for (auto& result : results) {
                    if (result.status == "recognized") {
                        stats_.successful_recognitions++;
                    } else if (result.status == "poor_quality") {
                        stats_.quality_rejections++;
                    }
                    if (result.status == "unknown") {
                        stats_.similarity_rejections++;
                    }
                }
                
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

