#ifndef DEPTH_SEGMENTATION_COMMON_H_
#define DEPTH_SEGMENTATION_COMMON_H_

#include <string>

#include <opencv2/rgbd.hpp>

namespace depth_segmentation {

const static std::string kDebugWindowName = "DebugImages";

struct SurfaceNormalParams {
  size_t window_size = 7;
  size_t method = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS;
};

}  // depth_segmentation

#endif  // DEPTH_SEGMENTATION_COMMON_H_
