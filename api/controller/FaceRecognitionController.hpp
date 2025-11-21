#ifndef FACE_RECOGNITION_CONTROLLER_HPP
#define FACE_RECOGNITION_CONTROLLER_HPP

#include "oatpp/web/server/api/ApiController.hpp"
#include "oatpp/core/macro/codegen.hpp"
#include "oatpp/core/macro/component.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"
#include "oatpp/web/mime/multipart/PartList.hpp"
#include "oatpp/web/mime/multipart/Reader.hpp"
#include "oatpp/web/mime/multipart/InMemoryDataProvider.hpp"
#include "dto/RecognitionDto.hpp"
#include "../../src/recognizer/face_recognizer.hpp"
#include "../../src/utils/config.hpp"
#include "utils/ImageUtils.hpp"
#include <memory>

#include OATPP_CODEGEN_BEGIN(ApiController)

class FaceRecognitionController : public oatpp::web::server::api::ApiController {
private:
    std::shared_ptr<FaceRecognizer> recognizer_;
    std::shared_ptr<Config> config_;
    
    // Helper methods - implementation in .cpp file
    std::shared_ptr<oatpp::web::protocol::http::outgoing::Response> handleRoot();
    std::shared_ptr<oatpp::web::protocol::http::outgoing::Response> handleHealth();
    std::shared_ptr<oatpp::web::protocol::http::outgoing::Response> handleRecognize(const std::shared_ptr<oatpp::web::protocol::http::incoming::Request>& request);
    std::shared_ptr<oatpp::web::protocol::http::outgoing::Response> handleRegister(const std::shared_ptr<oatpp::web::protocol::http::incoming::Request>& request);
    
    // Internal helper methods
    cv::Mat extractImageFromRequest(const std::shared_ptr<oatpp::web::protocol::http::incoming::Request>& request, bool& image_found);
    cv::Mat extractImageFromRegisterRequest(const std::shared_ptr<oatpp::web::protocol::http::incoming::Request>& request, bool& image_found, std::string& name);
    cv::Mat extractImageFromMultipart(const std::shared_ptr<oatpp::web::protocol::http::incoming::Request>& request, bool& image_found);
    cv::Mat extractImageFromMultipartRegister(const std::shared_ptr<oatpp::web::protocol::http::incoming::Request>& request, bool& image_found, std::string& name);
    cv::Mat extractImageFromBinary(const oatpp::String& body, bool& image_found);
    cv::Mat extractImageFromJson(const oatpp::String& body, bool& image_found);
    cv::Mat extractImageFromJsonRegister(const oatpp::String& body, bool& image_found, std::string& name);
    oatpp::Object<RecognizeResponseDto> convertResultsToDto(const std::vector<RecognitionResult>& results);
    
public:
    FaceRecognitionController(const std::shared_ptr<ObjectMapper>& objectMapper);
    
    // Endpoint declarations - minimal logic, delegate to implementation methods
    ENDPOINT("GET", "/", root) {
        return handleRoot();
    }
    
    ENDPOINT("GET", "/health", health) {
        return handleHealth();
    }
    
    ENDPOINT("POST", "/recognize", recognize,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        return handleRecognize(request);
    }
    
    ENDPOINT("POST", "/register", register_,
             REQUEST(std::shared_ptr<IncomingRequest>, request)) {
        return handleRegister(request);
    }
};

#include OATPP_CODEGEN_END(ApiController)

#endif // FACE_RECOGNITION_CONTROLLER_HPP
