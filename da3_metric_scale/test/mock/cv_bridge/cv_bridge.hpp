// cv_bridge stand-in for tests (mono8 / rgb8 / 32FC1 only)
#pragma once
#include <memory>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "../ros_mock.hpp"
namespace cv_bridge {
struct CvImage {cv::Mat image; using ConstPtr = std::shared_ptr<const CvImage>;};
inline std::shared_ptr<const CvImage> toCvShare(
  const std::shared_ptr<const sensor_msgs::msg::Image> & m, const std::string & enc = "")
{
  auto out = std::make_shared<CvImage>();
  int type = m->encoding == "32FC1" ? CV_32FC1 : m->encoding == "rgb8" ? CV_8UC3 : CV_8UC1;
  cv::Mat src(m->height, m->width, type, const_cast<uint8_t *>(m->data.data()), m->step);
  if (enc == "mono8" && type == CV_8UC3) {cv::cvtColor(src, out->image, cv::COLOR_RGB2GRAY);}
  else {out->image = src.clone();}
  return out;
}
}
