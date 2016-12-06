#ifndef DEPTH_SEGMENTATION_COMMON_H_
#define DEPTH_SEGMENTATION_COMMON_H_

#include <string>

#include <glog/logging.h>
#include <opencv2/rgbd.hpp>

namespace depth_segmentation {

const static std::string kDebugWindowName = "DebugImages";

struct SurfaceNormalParams {
  SurfaceNormalParams() {
    CHECK_EQ(window_size % 2, 1);
    CHECK_LT(window_size, 8);
    CHECK_GT(window_size, 0);
  }
  size_t window_size = 7;
  // size_t method = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD;
  size_t method = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS;
  // size_t method = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_SRI;
  bool display = false;
};

struct MaxDistanceMapParams {
  MaxDistanceMapParams() { CHECK_EQ(window_size % 2, 1); }
  size_t window_size = 3;
  bool display = false;
  bool exclude_nan_as_max_distance = false;
  bool ignore_nan_coordinates = false;  // TODO(ff): This probably doesn't make
                                        // a lot of sense -> consider removing
                                        // it.
  bool use_threshold = true;
  double noise_thresholding_factor = 8.0;
  double sensor_noise_param_1 = 0.0012;  // From Nguyen et al. (2012)
  double sensor_noise_param_2 = 0.0019;  // From Nguyen et al. (2012)
  double sensor_noise_param_3 = 0.0001;  // From Nguyen et al. (2012)
  double sensor_min_distance = 0.2;
};

struct MinConcavityMapParams {
  MinConcavityMapParams() { CHECK_EQ(window_size % 2, 1); }
  size_t window_size = 5;
  size_t step_size = 1;
  bool display = false;
  bool use_threshold = true;
  double threshold = 0.94;
};

struct FinalEdgeMapParams {
  bool display = false;
};

struct LabelMapParams {
  bool display = true;
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
