#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>

#include "mtcnn/detector.h"
#include "draw.hpp"
#include "mtcnn/onnx_module.h"
#include <pqxx/pqxx>
#include "postgres.hpp"

//// rm -rf build; mkdir build;cd build;cmake ..;make;cd ..
//// ./build/infer_photo ./models ./data/got.jpg

int main(int argc, char **argv) {

  if (argc < 3) {
        std::cerr << "Usage " << ": "
            << "<app_binary> "
            << "<path_to_models_dir>"
            << "<path_to_test_image>\n";
        return 1;
  }

  std::string modelPath = argv[1];

  ProposalNetwork::Config pConfig;
  pConfig.caffeModel = modelPath + "/det1.caffemodel";
  pConfig.protoText = modelPath + "/det1.prototxt";
  pConfig.threshold = 0.6f;

  RefineNetwork::Config rConfig;
  rConfig.caffeModel = modelPath + "/det2.caffemodel";
  rConfig.protoText = modelPath + "/det2.prototxt";
  rConfig.threshold = 0.7f;

  OutputNetwork::Config oConfig;
  oConfig.caffeModel = modelPath + "/det3.caffemodel";
  oConfig.protoText = modelPath + "/det3.prototxt";
  oConfig.threshold = 0.7f;

  MTCNNDetector detector(pConfig, rConfig, oConfig);

  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "face_embedding");
  Ort::SessionOptions session_options;
  SubNet InceptionNet(env, session_options, "../../models/inception.onnx");
  std::vector<int64_t> input_shape = {1, 3, 160, 160};
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  Postgres db("localhost", 5433, "healthmed", "paperless", "paperless");
  float recognition_threshold = 0.3;

  cv::Mat img = cv::imread(argv[2]);

  std::vector<Face> faces;
  std::vector<float> conf;

  {
    faces = detector.detect(img, 20.f, 0.709f);
  }

  std::cout << "Number of faces found in the supplied image - " << faces.size()
            << std::endl;

  std::vector<rectPoints> data;
  std::vector<PostgresPerson> people;

  for (size_t i = 0; i < faces.size(); ++i) {
    conf.push_back(faces[i].score);
    cv::Mat faceRoi = img(faces[i].bbox.getRect());
    std::vector<float> faceVector = detector.forward(faceRoi);
    std::cout << "Face vector shape: " << faceVector.size() << std::endl;
    if (faceVector.size() != (160 * 160 * 3)) {
      // faces.erase(faces.begin() + i);
      std::cout << "Invalid face tensor" << std::endl;
      people.push_back(PostgresPerson(0, "Invalid", 0.0, 0.0));
      continue;
    }
    // Debug value
    for (size_t j = 0; j < 5; ++j) {
      if (j > faceVector.size()) break;
      std::cout << faceVector[j] << " ";
    }
    std::cout << std::endl;

    Ort::Value face_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(faceVector.data()),
        faceVector.size(),
        input_shape.data(),
        input_shape.size()
    );
    std::vector<Ort::Value> outputs = InceptionNet.forward(face_tensor);
    std::vector<int64_t> output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Embedding shape: ";
    for (size_t j = 0; j < output_shape.size(); ++j) {
        std::cout << output_shape[j] << " ";
    }
    std::cout << std::endl;
    float* embedding = outputs[0].GetTensorMutableData<float>();
    try {
      PostgresPerson person = db.get_recognition(std::vector<float>(embedding, embedding + output_shape[1]), recognition_threshold);
      if (person.confidence != 0.0) {
        people.push_back(person);
      } else {
        people.push_back(PostgresPerson(0, "Unknown", 0.0, 0.0));
      }
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return 1;
    }
  }

  // show the image with faces in it
  std::cout << "Num faces: " << faces.size() << std::endl;
  
  // Check if any faces were found
  if (faces.size() == 0) {
    std::cout << "⚠️  No faces detected in the image." << std::endl;
    std::cout << "Displaying original image..." << std::endl;
    cv::imshow("test-oc", img);
    cv::waitKey(0);
    return 0;
  }
  
  for (size_t i = 0; i < faces.size(); ++i) {
    std::vector<cv::Point> pts;
    for (int p = 0; p < NUM_PTS; ++p) {
      pts.push_back(
          cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
    }

    std::string label = std::to_string(faces[i].score).substr(0, 4) + "|" + people[i].name + ": " + std::to_string(people[i].confidence);
    auto rect = faces[i].bbox.getRect();
    auto d = std::make_tuple(rect, pts, "Conf: " + label);
    data.push_back(d);
  }

  float mean_conf = std::accumulate(conf.begin(), conf.end(), 0.0f) / conf.size();
  std::cout << "Mean confidence: " << mean_conf << std::endl;
  
  auto resultImg = drawRectsAndPoints(img, data);
  cv::imshow("face-recognition", resultImg);
  cv::waitKey(0);

  return 0;
}
