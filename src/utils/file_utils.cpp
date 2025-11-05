#include "file_utils.hpp"
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

std::vector<std::string> get_image_files(const std::string& data_dir) {
    std::vector<std::string> image_files;
    std::vector<std::string> image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
        ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF", ".WEBP"
    };
    
    if (!fs::exists(data_dir) || !fs::is_directory(data_dir)) {
        std::cerr << "Directory does not exist or is not a directory: " << data_dir << std::endl;
        return image_files;
    }
    
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            
            // Check if the extension is in our list of image extensions
            if (std::find(image_extensions.begin(), image_extensions.end(), extension) != image_extensions.end()) {
                image_files.push_back(entry.path().filename().string());
            }
        }
    }
    
    return image_files;
}

