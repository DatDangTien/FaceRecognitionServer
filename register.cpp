#include "src/recognizer/face_recognizer.hpp"
#include "src/utils/config.hpp"

//// rm -rf build; mkdir build;cd build;cmake ..;make;cd ..
//// ./build/register "John Doe" ./data/got.jpg

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <name> <image_path>" << std::endl;
        return 1;
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
    }
    else {
        std::cerr << "Failed to register face" << std::endl;
        return 1;
    }
}
