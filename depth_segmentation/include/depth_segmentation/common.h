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

struct MaxDistanceMapParams {
  size_t window_size = 3;
  bool use_mask = false;
  bool use_threshold = true;
  bool exclude_nan_as_max = false;
  double noise_thresholding_factor = 50.0;
  double sensor_noise_param_1 = 0.0012;  // From Nguyen et al. (2012)
  double sensor_noise_param_2 = 0.0019;  // From Nguyen et al. (2012)
  double sensor_noise_param_3 = 0.0001;  // From Nguyen et al. (2012)
  double sensor_min_distance = 0.2;
};

struct IsNan {
  template <class T>
  bool operator()(T const& p) const {
    return std::isnan(p);
  }
};

struct IsNotNan {
  template <class T>
  bool operator()(T const& p) const {
    return !std::isnan(p);
  }
};

}  // depth_segmentation

#endif  // DEPTH_SEGMENTATION_COMMON_H_
