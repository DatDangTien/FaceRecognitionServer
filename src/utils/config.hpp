#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>

class Config {
public:
    // Database configuration
    std::string db_host;
    int db_port;
    std::string db_name;
    std::string db_user;
    std::string db_password;
    
    // Recognition settings
    float recognition_threshold;
    int detection_interval;
    
    // Quality thresholds
    bool quality_check;
    float blur_threshold;
    float min_face_size;
    float dark_ratio_threshold;
    float bright_ratio_threshold;
    float pose_threshold;
    float quality_threshold;
    
    // Server settings
    std::string server_host;
    int server_port;
    
    // Model paths
    std::string models_path;
    std::string inception_model_path;

    bool visualize;
    bool benchmark;
    
    Config();
    bool load(const std::string& filename);
    void setDefaults();
    
private:
    std::map<std::string, std::string> configMap;
    void parseFile(const std::string& filename);
    std::string trim(const std::string& str);
    std::string getValue(const std::string& key, const std::string& defaultValue);
    int getValueInt(const std::string& key, int defaultValue);
    float getValueFloat(const std::string& key, float defaultValue);
    bool getValueBool(const std::string& key, bool defaultValue);
};

#endif // CONFIG_HPP

