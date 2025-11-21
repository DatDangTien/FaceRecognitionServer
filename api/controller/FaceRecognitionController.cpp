#include "FaceRecognitionController.hpp"
#include <cstdlib>
#include <stdexcept>
#include <opencv2/opencv.hpp>

FaceRecognitionController::FaceRecognitionController(const std::shared_ptr<ObjectMapper>& objectMapper)
    : oatpp::web::server::api::ApiController(objectMapper)
{
    // Initialize recognizer
    config_ = std::make_shared<Config>();
    std::string config_path = std::getenv("CONFIG_PATH") ? std::getenv("CONFIG_PATH") : "config.ini";
    if (!config_->load(config_path)) {
        throw std::runtime_error("Failed to load config file: " + config_path);
    }
    recognizer_ = std::make_shared<FaceRecognizer>(*config_);
}

std::shared_ptr<oatpp::web::protocol::http::outgoing::Response> FaceRecognitionController::handleRoot() {
    auto response = RootResponseDto::createShared();
    response->message = "Face Recognition API";
    response->version = "1.0.0";
    
    auto endpoints = Fields<String>::createShared();
    endpoints["recognize"] = String("/recognize");
    endpoints["register"] = String("/register");
    endpoints["health"] = String("/health");
    response->endpoints = endpoints;
    
    return createDtoResponse(Status::CODE_200, response);
}

std::shared_ptr<oatpp::web::protocol::http::outgoing::Response> FaceRecognitionController::handleHealth() {
    auto response = HealthResponseDto::createShared();
    response->status = "healthy";
    response->recognizer_initialized = (recognizer_ != nullptr);
    return createDtoResponse(Status::CODE_200, response);
}

cv::Mat FaceRecognitionController::extractImageFromBinary(const oatpp::String& body, bool& image_found) {
    cv::Mat img;
    image_found = false;
    
    if (!body || body->empty()) {
        return img;
    }
    
    try {
        // Convert oatpp::String to vector<uint8_t>
        const std::string& body_str = std::string(*body);
        std::vector<uint8_t> binary_data(body_str.begin(), body_str.end());
        
        // Decode binary image
        img = ImageUtils::decodeBinaryImage(binary_data);
        if (!img.empty()) {
            image_found = true;
        }
    } catch (const std::exception& e) {
        // Failed to decode binary image
        image_found = false;
    }
    
    return img;
}

cv::Mat FaceRecognitionController::extractImageFromJson(const oatpp::String& body, bool& image_found) {
    cv::Mat img;
    image_found = false;
    
    if (!body) {
        return img;
    }
    
    try {
        auto requestDto = getDefaultObjectMapper()->readFromString<oatpp::Object<RecognizeRequestDto>>(body);
        if (requestDto && requestDto->base64_image) {
            std::string base64_str = std::string(*requestDto->base64_image);
            img = ImageUtils::decodeBase64Image(base64_str);
            if (!img.empty()) {
                image_found = true;
            }
        }
    } catch (...) {
        // Try simple string parsing as fallback
        std::string body_str = std::string(*body);
        size_t pos = body_str.find("\"base64_image\"");
        if (pos != std::string::npos) {
            pos = body_str.find(":", pos);
            if (pos != std::string::npos) {
                pos = body_str.find("\"", pos) + 1;
                size_t end_pos = body_str.find("\"", pos);
                if (end_pos != std::string::npos) {
                    std::string base64_str = body_str.substr(pos, end_pos - pos);
                    img = ImageUtils::decodeBase64Image(base64_str);
                    if (!img.empty()) {
                        image_found = true;
                    }
                }
            }
        }
    }
    
    return img;
}

cv::Mat FaceRecognitionController::extractImageFromMultipart(const std::shared_ptr<oatpp::web::protocol::http::incoming::Request>& request, bool& image_found) {
    cv::Mat img;
    image_found = false;
    
    if (!request) {
        return img;
    }
    
    try {
        // Create multipart container
        auto multipart = std::make_shared<oatpp::web::mime::multipart::PartList>(request->getHeaders());
        
        // Create multipart reader
        oatpp::web::mime::multipart::Reader multipartReader(multipart.get());
        
        // Configure to read parts into memory (max 10MB)
        multipartReader.setDefaultPartReader(oatpp::web::mime::multipart::createInMemoryPartReader(10 * 1024 * 1024));
        
        // Read multipart body
        request->transferBody(&multipartReader);
        
        // Get the file part
        auto filePart = multipart->getNamedPart("file");
        if (filePart) {
            auto payload = filePart->getPayload();
            if (payload) {
                auto inMemoryData = payload->getInMemoryData();
                if (inMemoryData) {
                    // Convert to binary data
                    const std::string& data_str = std::string(*inMemoryData);
                    std::vector<uint8_t> binary_data(data_str.begin(), data_str.end());
                    
                    // Decode image
                    img = ImageUtils::decodeBinaryImage(binary_data);
                    if (!img.empty()) {
                        image_found = true;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        // Failed to parse multipart
        image_found = false;
    }
    
    return img;
}

cv::Mat FaceRecognitionController::extractImageFromMultipartRegister(const std::shared_ptr<oatpp::web::protocol::http::incoming::Request>& request, bool& image_found, std::string& name) {
    cv::Mat img;
    image_found = false;
    name = "";
    
    if (!request) {
        return img;
    }
    
    try {
        // Create multipart container
        auto multipart = std::make_shared<oatpp::web::mime::multipart::PartList>(request->getHeaders());
        
        // Create multipart reader
        oatpp::web::mime::multipart::Reader multipartReader(multipart.get());
        
        // Configure to read parts into memory (max 10MB)
        multipartReader.setDefaultPartReader(oatpp::web::mime::multipart::createInMemoryPartReader(10 * 1024 * 1024));
        
        // Read multipart body
        request->transferBody(&multipartReader);
        
        // Get the name field (if provided in form)
        auto namePart = multipart->getNamedPart("name");
        if (namePart) {
            auto payload = namePart->getPayload();
            if (payload) {
                auto inMemoryData = payload->getInMemoryData();
                if (inMemoryData) {
                    name = std::string(*inMemoryData);
                }
            }
        }
        
        // If name not in form, try query parameter
        if (name.empty()) {
            auto name_param = request->getQueryParameter("name");
            if (name_param) {
                name = std::string(*name_param);
            }
        }
        
        // Get the file part
        auto filePart = multipart->getNamedPart("file");
        if (filePart) {
            auto payload = filePart->getPayload();
            if (payload) {
                auto inMemoryData = payload->getInMemoryData();
                if (inMemoryData) {
                    // Convert to binary data
                    const std::string& data_str = std::string(*inMemoryData);
                    std::vector<uint8_t> binary_data(data_str.begin(), data_str.end());
                    
                    // Decode image
                    img = ImageUtils::decodeBinaryImage(binary_data);
                    if (!img.empty()) {
                        image_found = true;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        // Failed to parse multipart
        image_found = false;
    }
    
    return img;
}

cv::Mat FaceRecognitionController::extractImageFromRequest(const std::shared_ptr<oatpp::web::protocol::http::incoming::Request>& request, bool& image_found) {
    cv::Mat img;
    image_found = false;
    
    if (!request) {
        return img;
    }
    
    // Get Content-Type header
    auto contentType = request->getHeader("Content-Type");
    std::string content_type_str = contentType ? std::string(*contentType) : "";
    
    // Check for multipart/form-data first
    if (content_type_str.find("multipart/form-data") != std::string::npos) {
        img = extractImageFromMultipart(request, image_found);
        if (image_found) {
            return img;
        }
    }
    
    // Read body for other content types
    auto body = request->readBodyToString();
    if (!body || body->empty()) {
        return img;
    }
    
    // Try JSON parsing (for base64 images)
    if (content_type_str.find("application/json") != std::string::npos || 
        content_type_str.empty()) {
        img = extractImageFromJson(body, image_found);
        if (image_found) {
            return img;
        }
    }
    
    // If content type is unknown or empty, try JSON
    if (!image_found && content_type_str.empty()) {
        img = extractImageFromJson(body, image_found);
    }
    
    return img;
}

cv::Mat FaceRecognitionController::extractImageFromJsonRegister(const oatpp::String& body, bool& image_found, std::string& name) {
    cv::Mat img;
    image_found = false;
    name = "";
    
    if (!body) {
        return img;
    }
    
    try {
        auto requestDto = getDefaultObjectMapper()->readFromString<oatpp::Object<RegisterRequestDto>>(body);
        if (requestDto) {
            if (requestDto->name) {
                name = std::string(*requestDto->name);
            }
            if (requestDto->base64_image) {
                std::string base64_str = std::string(*requestDto->base64_image);
                img = ImageUtils::decodeBase64Image(base64_str);
                if (!img.empty()) {
                    image_found = true;
                }
            }
        }
    } catch (...) {
        // Try simple string parsing as fallback
        std::string body_str = std::string(*body);
        
        // Parse name (string)
        size_t pos = body_str.find("\"name\"");
        if (pos != std::string::npos) {
            pos = body_str.find(":", pos);
            if (pos != std::string::npos) {
                size_t end_pos = body_str.find_first_of(",}", pos);
                if (end_pos != std::string::npos) {
                    std::string name_str = body_str.substr(pos + 1, end_pos - pos - 1);
                    // Remove whitespace and quotes
                    name_str.erase(0, name_str.find_first_not_of(" \t\n\r\""));
                    name_str.erase(name_str.find_last_not_of(" \t\n\r\"") + 1);
                    name = name_str;
                }
            }
        }
        
        // Parse base64_image
        pos = body_str.find("\"base64_image\"");
        if (pos != std::string::npos) {
            pos = body_str.find(":", pos);
            if (pos != std::string::npos) {
                pos = body_str.find("\"", pos) + 1;
                size_t end_pos = body_str.find("\"", pos);
                if (end_pos != std::string::npos) {
                    std::string base64_str = body_str.substr(pos, end_pos - pos);
                    img = ImageUtils::decodeBase64Image(base64_str);
                    if (!img.empty()) {
                        image_found = true;
                    }
                }
            }
        }
    }
    
    return img;
}

cv::Mat FaceRecognitionController::extractImageFromRegisterRequest(const std::shared_ptr<oatpp::web::protocol::http::incoming::Request>& request, bool& image_found, std::string& name) {
    cv::Mat img;
    image_found = false;
    name = "";
    
    if (!request) {
        return img;
    }
    
    // Get Content-Type header
    auto contentType = request->getHeader("Content-Type");
    std::string content_type_str = contentType ? std::string(*contentType) : "";
    
    // Check for multipart/form-data first
    if (content_type_str.find("multipart/form-data") != std::string::npos) {
        img = extractImageFromMultipartRegister(request, image_found, name);
        if (image_found) {
            return img;
        }
    }
    
    // Read body for other content types
    auto body = request->readBodyToString();
    if (!body || body->empty()) {
        return img;
    }
    
    // Try JSON parsing (for base64 images with name)
    if (content_type_str.find("application/json") != std::string::npos || 
        content_type_str.empty()) {
        img = extractImageFromJsonRegister(body, image_found, name);
        if (image_found) {
            return img;
        }
    }
    
    // If content type is unknown or empty, try JSON
    if (!image_found && content_type_str.empty()) {
        img = extractImageFromJsonRegister(body, image_found, name);
    }
    
    return img;
}

oatpp::Object<RecognizeResponseDto> FaceRecognitionController::convertResultsToDto(const std::vector<RecognitionResult>& results) {
    auto response = RecognizeResponseDto::createShared();
    auto faces = List<Object<RecognitionResultDto>>::createShared();
    
    for (const auto& result : results) {
        auto faceDto = RecognitionResultDto::createShared();
        faceDto->name = result.name;
        faceDto->confidence = result.confidence;
        faceDto->status = result.status;
        faceDto->xmin = result.xmin;
        faceDto->ymin = result.ymin;
        faceDto->xmax = result.xmax;
        faceDto->ymax = result.ymax;
        faceDto->person_id = result.person_id;
        faces->push_back(faceDto);
    }
    
    response->faces = faces;
    response->total_faces = static_cast<int32_t>(results.size());
    
    return response;
}

std::shared_ptr<oatpp::web::protocol::http::outgoing::Response> FaceRecognitionController::handleRecognize(const std::shared_ptr<IncomingRequest>& request) {
    try {
        // Extract image from request (supports both binary and JSON/base64)
        bool image_found = false;
        cv::Mat img = extractImageFromRequest(request, image_found);
        
        // Validate image
        if (!image_found || img.empty()) {
            auto error = oatpp::String("Either image file or base64_image must be provided");
            return createResponse(Status::CODE_400, error);
        }
        
        // Ensure correct format
        img = ImageUtils::ensureCorrectFormat(img);
        if (img.empty()) {
            auto error = oatpp::String("Failed to decode image");
            return createResponse(Status::CODE_400, error);
        }
        
        // Process frame
        std::vector<RecognitionResult> results = recognizer_->processFrame(img, 0);
        
        // Convert to DTO
        auto response = convertResultsToDto(results);
        
        return createDtoResponse(Status::CODE_200, response);
        
    } catch (const std::exception& e) {
        auto error = oatpp::String("Error processing image: " + std::string(e.what()));
        return createResponse(Status::CODE_500, error);
    }
}

std::shared_ptr<oatpp::web::protocol::http::outgoing::Response> FaceRecognitionController::handleRegister(const std::shared_ptr<IncomingRequest>& request) {
    try {
        // Extract image and name from request (supports both binary and JSON/base64)
        bool image_found = false;
        std::string name = "";
        cv::Mat img = extractImageFromRegisterRequest(request, image_found, name);
        
        // Validate image
        if (!image_found || img.empty()) {
            auto error = oatpp::String("Either image file or base64_image must be provided");
            return createResponse(Status::CODE_400, error);
        }
        
        // Validate name (must not be empty)
        if (name.empty()) {
            auto error = oatpp::String("Name field must be provided and cannot be empty");
            return createResponse(Status::CODE_400, error);
        }
        
        // Ensure correct format
        img = ImageUtils::ensureCorrectFormat(img);
        if (img.empty()) {
            auto error = oatpp::String("Failed to decode image");
            return createResponse(Status::CODE_400, error);
        }
        
        // Register face
        auto result = recognizer_->registerFace(img, name);
        
        // Create response
        auto response = RegisterResponseDto::createShared();
        response->success = result.first;
        if (result.first) {
            response->message = "Face registered successfully for name: " + name + ": " + result.second;
        } else {
            response->message = "Failed to register face for name: " + name + ": " + result.second;
        }
        
        return createDtoResponse(Status::CODE_200, response);
        
    } catch (const std::exception& e) {
        auto error = oatpp::String("Error registering face: " + std::string(e.what()));
        return createResponse(Status::CODE_500, error);
    }
}
