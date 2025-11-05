#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tuple>


using rectPoints = std::tuple<cv::Rect, std::vector<cv::Point>, std::string>;

static cv::Mat drawRectsAndPoints(const cv::Mat &img,
                                  const std::vector<rectPoints> data,
                                  const double fps = 0.0) {
  cv::Mat outImg;
  img.convertTo(outImg, CV_8UC3);

  for (auto &d : data) {
    auto rect = std::get<0>(d);
    auto pts = std::get<1>(d);
    auto text = std::get<2>(d);
    
    cv::rectangle(outImg, rect, cv::Scalar(0, 255, 255), 2);
    cv::Point textPos = cv::Point(rect.tl().x, rect.br().y - 20); // Position text 20px from bottom edge
    cv::putText(outImg, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

    if (fps > 0.0) {
      cv::putText(outImg, "FPS: " + std::to_string(fps).substr(0, 4), cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }
    
    for (size_t i = 0; i < pts.size(); ++i) {
      cv::circle(outImg, pts[i], 3, cv::Scalar(0, 255, 255), 1);
    }
  }
  return outImg;
}
