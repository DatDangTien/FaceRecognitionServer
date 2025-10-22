#include "config.hpp"
#include <algorithm>
#include <cctype>

Config::Config() {
    setDefaults();
}

void Config::setDefaults() {
    // Database defaults
    db_host = "localhost";
    db_port = 5433;
    db_name = "healthmed";
    db_user = "paperless";
    db_password = "paperless";
    
    // Recognition defaults
    recognition_threshold = 0.3f;
    detection_interval = 5;
    
    // Quality thresholds (from Python config)
    blur_threshold = 100.0f;
    min_face_size = 60.0f;
    dark_ratio_threshold = 0.4f;
    bright_ratio_threshold = 0.3f;
    pose_threshold = 1000.0f;
    quality_threshold = 0.5f;
    
    // Server defaults
    server_host = "0.0.0.0";
    server_port = 8764;
    
    // Model paths
    models_path = "./models";
    inception_model_path = "./models/inception.onnx";
}

std::string Config::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

void Config::parseFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << filename << std::endl;
        std::cerr << "Using default configuration." << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        // Parse key=value
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));
            configMap[key] = value;
        }
    }
    
    file.close();
}

std::string Config::getValue(const std::string& key, const std::string& defaultValue) {
    auto it = configMap.find(key);
    if (it != configMap.end()) {
        return it->second;
    }
    return defaultValue;
}

int Config::getValueInt(const std::string& key, int defaultValue) {
    auto it = configMap.find(key);
    if (it != configMap.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            std::cerr << "Warning: Invalid integer value for " << key << std::endl;
        }
    }
    return defaultValue;
}

float Config::getValueFloat(const std::string& key, float defaultValue) {
    auto it = configMap.find(key);
    if (it != configMap.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {
            std::cerr << "Warning: Invalid float value for " << key << std::endl;
        }
    }
    return defaultValue;
}

bool Config::load(const std::string& filename) {
    parseFile(filename);
    
    // Load database configuration
    db_host = getValue("db_host", db_host);
    db_port = getValueInt("db_port", db_port);
    db_name = getValue("db_name", db_name);
    db_user = getValue("db_user", db_user);
    db_password = getValue("db_password", db_password);
    
    // Load recognition settings
    recognition_threshold = getValueFloat("recognition_threshold", recognition_threshold);
    detection_interval = getValueInt("detection_interval", detection_interval);
    
    // Load quality thresholds
    blur_threshold = getValueFloat("blur_threshold", blur_threshold);
    min_face_size = getValueFloat("min_face_size", min_face_size);
    dark_ratio_threshold = getValueFloat("dark_ratio_threshold", dark_ratio_threshold);
    bright_ratio_threshold = getValueFloat("bright_ratio_threshold", bright_ratio_threshold);
    pose_threshold = getValueFloat("pose_threshold", pose_threshold);
    quality_threshold = getValueFloat("quality_threshold", quality_threshold);
    
    // Load server settings
    server_host = getValue("server_host", server_host);
    server_port = getValueInt("server_port", server_port);
    
    // Load model paths
    models_path = getValue("models_path", models_path);
    inception_model_path = getValue("inception_model_path", inception_model_path);
    
    std::cout << "Configuration loaded successfully" << std::endl;
    std::cout << "Database: " << db_host << ":" << db_port << "/" << db_name << std::endl;
    std::cout << "Server: " << server_host << ":" << server_port << std::endl;
    std::cout << "Recognition threshold: " << recognition_threshold << std::endl;
    
    return true;
}

