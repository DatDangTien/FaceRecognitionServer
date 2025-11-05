#include "src/recognizer/face_recognizer.hpp"
#include "src/utils/config.hpp"
#include "src/utils/file_utils.hpp"

//// rm -rf build; mkdir build;cd build;cmake ..;make;cd ..
//// ./build/register "John Doe" ./data/got.jpg

int main(int argc, char **argv) {
    // ./build/register /datadir (contain Name.jpg) 
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <data_dir> or " << argv[0] << " <name> <image_path>" << std::endl;
        return 1;
    }
    if (argc == 2) {
        std::string data_dir = argv[1];
        std::vector<std::string> image_files = get_image_files(data_dir);
        for (const auto& image_file : image_files) {
            std::string name = image_file.substr(0, image_file.find('.'));
            std::string image_path = data_dir + "/" + image_file;
            cv::Mat img = cv::imread(image_path);
            if (img.empty()) {
                std::cerr << "Failed to load image" << image_path << std::endl;
                continue;
            }
            Config config;
            config.load("config.ini");
            FaceRecognizer recognizer(config);
            bool success = recognizer.registerFace(img, name);
            if (success) {
                std::cout << "Face registered successfully" << std::endl;
            }
            else {
                std::cerr << "Failed to register face" << std::endl;
                continue;
            }
        }
        return 0;
    }
    
    std::string name = argv[1];
    std::string image_path = argv[2];
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    Config config;
    config.load("config.ini");
    FaceRecognizer recognizer(config);
    bool success = recognizer.registerFace(img, name);
    if (success) {
        std::cout << "Face registered successfully" << std::endl;
        return 0;
    }
    else {
        std::cerr << "Failed to register face" << std::endl;
        return 1;
    }
}
