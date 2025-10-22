#include "detector.h"
#include "helpers.h"

// OpenCV 4.0 update
#define CV_BGR2RGB cv::COLOR_BGR2RGB
#define CV_BGRA2RGB cv::COLOR_BGR2RGB


MTCNNDetector::MTCNNDetector(const ProposalNetwork::Config &pConfig,
                             const RefineNetwork::Config &rConfig,
                             const OutputNetwork::Config &oConfig) {
  _pnet = std::unique_ptr<ProposalNetwork>(new ProposalNetwork(pConfig));
  _rnet = std::unique_ptr<RefineNetwork>(new RefineNetwork(rConfig));
  _onet = std::unique_ptr<OutputNetwork>(new OutputNetwork(oConfig));
}

std::vector<Face> MTCNNDetector::detect(const cv::Mat &img,
                                        const float minFaceSize,
                                        const float scaleFactor) {

  cv::Mat rgbImg;
  if (img.channels() == 3) {
    cv::cvtColor(img, rgbImg, CV_BGR2RGB);
  } else if (img.channels() == 4) {
    cv::cvtColor(img, rgbImg, CV_BGRA2RGB);
  }
  if (rgbImg.empty()) {
    return std::vector<Face>();
  }
  rgbImg.convertTo(rgbImg, CV_32FC3);
  rgbImg = rgbImg.t();

  // Run Proposal Network to find the initial set of faces
  std::vector<Face> faces = _pnet->run(rgbImg, minFaceSize, scaleFactor);

  // Early exit if we do not have any faces
  if (faces.empty()) {
    return faces;
  }

  // Run Refine network on the output of the Proposal network
  faces = _rnet->run(rgbImg, faces);

  // Early exit if we do not have any faces
  if (faces.empty()) {
    return faces;
  }

  // Run Output network on the output of the Refine network
  faces = _onet->run(rgbImg, faces);

  for (size_t i = 0; i < faces.size(); ++i) {
    std::swap(faces[i].bbox.x1, faces[i].bbox.y1);
    std::swap(faces[i].bbox.x2, faces[i].bbox.y2);
    for (int p = 0; p < NUM_PTS; ++p) {
      std::swap(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]);
    }
  }

  return faces;
}

std::vector<float> MTCNNDetector::forward(const cv::Mat &img, const float minFaceSize,
                                          const float scaleFactor, const int imageSize,
                                          const int margin, float *prob) {
  // Detect faces
  std::vector<Face> faces = detect(img, minFaceSize, scaleFactor);

  // Return empty vector if no face detected
  if (faces.empty()) {
    return std::vector<float>();
  }

  // Select the largest face (similar to select_largest=True in Python)
  size_t largestIdx = 0;
  float largestArea = 0;
  for (size_t i = 0; i < faces.size(); ++i) {
    float area = (faces[i].bbox.x2 - faces[i].bbox.x1) *
                 (faces[i].bbox.y2 - faces[i].bbox.y1);
    if (area > largestArea) {
      largestArea = area;
      largestIdx = i;
    }
  }

  Face &selectedFace = faces[largestIdx];
  
  // Return probability if requested
  if (prob != nullptr) {
    *prob = selectedFace.score;
  }

  // Add margin to bounding box
  float box_width = selectedFace.bbox.x2 - selectedFace.bbox.x1;
  float box_height = selectedFace.bbox.y2 - selectedFace.bbox.y1;
  
  int x1 = static_cast<int>(selectedFace.bbox.x1 - margin);
  int y1 = static_cast<int>(selectedFace.bbox.y1 - margin);
  int x2 = static_cast<int>(selectedFace.bbox.x2 + margin);
  int y2 = static_cast<int>(selectedFace.bbox.y2 + margin);

  // Clamp to image boundaries
  x1 = std::max(0, x1);
  y1 = std::max(0, y1);
  x2 = std::min(img.cols, x2);
  y2 = std::min(img.rows, y2);

  // Create rectangle for cropping
  cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
  
  // Crop face region using helper function
  cv::Mat faceCrop = cropImage(img, roi);
  
  if (faceCrop.empty()) {
    return std::vector<float>();
  }

  // Resize to target size (default 160x160)
  cv::Mat faceResized;
  cv::resize(faceCrop, faceResized, cv::Size(imageSize, imageSize),
             0, 0, cv::INTER_LINEAR);

  // Convert to float
  cv::Mat faceFloat;
  faceResized.convertTo(faceFloat, CV_32FC3);

  // Apply fixed_image_standardization: (pixel - 127.5) / 128.0
  faceFloat = (faceFloat - 127.5) / 128.0;

  // Convert from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
  // Split channels
  std::vector<cv::Mat> channels(3);
  cv::split(faceFloat, channels);

  // Create flattened tensor vector in CHW format
  // Total size: 3 * imageSize * imageSize
  std::vector<float> faceTensor;
  faceTensor.reserve(3 * imageSize * imageSize);
  
  // Flatten each channel and append to vector
  // Order: [R_channel, G_channel, B_channel]
  for (int c = 0; c < 3; ++c) {
    // Each channel is continuous, so we can directly copy its data
    if (channels[c].isContinuous()) {
      float* data = channels[c].ptr<float>(0);
      faceTensor.insert(faceTensor.end(), data, data + imageSize * imageSize);
    } else {
      // If not continuous, copy row by row
      for (int i = 0; i < imageSize; ++i) {
        float* rowData = channels[c].ptr<float>(i);
        faceTensor.insert(faceTensor.end(), rowData, rowData + imageSize);
      }
    }
  }

  return faceTensor;
}
