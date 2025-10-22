#ifndef _include_opencv_mtcnn_detector_h_
#define _include_opencv_mtcnn_detector_h_

#include "face.h"
#include "onet.h"
#include "pnet.h"
#include "rnet.h"

class MTCNNDetector {
private:
  std::unique_ptr<ProposalNetwork> _pnet;
  std::unique_ptr<RefineNetwork> _rnet;
  std::unique_ptr<OutputNetwork> _onet;

public:
  MTCNNDetector(const ProposalNetwork::Config &pConfig,
                const RefineNetwork::Config &rConfig,
                const OutputNetwork::Config &oConfig);
  std::vector<Face> detect(const cv::Mat &img, const float minFaceSize,
                           const float scaleFactor);
  
  // Extract face tensor as flattened float vector (CHW format)
  // For imageSize=160: returns vector of size 3*160*160 = 76,800 floats
  // Format: [R_channel(160x160), G_channel(160x160), B_channel(160x160)]
  // Values: normalized as (pixel - 127.5) / 128.0, range ~[-1, 1]
  // Returns empty vector if no face is detected
  std::vector<float> forward(const cv::Mat &img, const float minFaceSize = 20.0f,
                             const float scaleFactor = 0.709f, const int imageSize = 160,
                             const int margin = 0, float *prob = nullptr);
};

#endif
