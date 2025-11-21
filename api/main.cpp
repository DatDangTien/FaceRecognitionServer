#include "oatpp/network/Server.hpp"
#include "oatpp/network/tcp/server/ConnectionProvider.hpp"
#include "oatpp/network/Address.hpp"
#include "oatpp/web/server/HttpConnectionHandler.hpp"
#include "oatpp/web/server/HttpRouter.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"
#include "oatpp/core/macro/component.hpp"
#include "controller/FaceRecognitionController.hpp"
#include "src/utils/config.hpp"
#include <iostream>
#include <memory>

void run() {
    // Create ObjectMapper for serialization
    auto objectMapper = oatpp::parser::json::mapping::ObjectMapper::createShared();
    
    // Create HTTP Router
    auto router = oatpp::web::server::HttpRouter::createShared();
    
    // Create Controller
    auto controller = std::make_shared<FaceRecognitionController>(objectMapper);
    router->addController(controller);
    
    // Create HTTP ConnectionHandler
    auto connectionHandler = oatpp::web::server::HttpConnectionHandler::createShared(router);
    
    // Load configuration
    Config config;
    config.load("config.ini");
    
    std::string host = config.api_host;
    v_uint16 port = static_cast<v_uint16>(config.api_port);
    
    // Create ConnectionProvider
    auto connectionProvider = oatpp::network::tcp::server::ConnectionProvider::createShared(
        oatpp::network::Address(oatpp::String(host.c_str()), port)
    );
    
    // Create Server
    oatpp::network::Server server(
        connectionProvider,
        connectionHandler
    );
    
    std::cout << "Face Recognition API Server running on " << host << ":" << port << std::endl;
    std::cout << "Endpoints:" << std::endl;
    std::cout << "  GET  /" << std::endl;
    std::cout << "  GET  /health" << std::endl;
    std::cout << "  POST /recognize" << std::endl;
    std::cout << "  POST /register" << std::endl;
    
    // Run server
    server.run();
}

int main(int argc, char** argv) {
    oatpp::base::Environment::init();
    
    try {
        run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        oatpp::base::Environment::destroy();
        return 1;
    }
    
    oatpp::base::Environment::destroy();
    return 0;
}

