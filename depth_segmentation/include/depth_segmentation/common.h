#ifndef DEPTH_SEGMENTATION_COMMON_H_
#define DEPTH_SEGMENTATION_COMMON_H_

#include <string>

#include <glog/logging.h>
#include <opencv2/rgbd.hpp>
#include <opencv2/viz/vizcore.hpp>

namespace depth_segmentation {

const static std::string kDebugWindowName = "DebugImages";

enum SurfaceNormalEstimationMethod {
  kLinemod = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD,
  kFals = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS,
  kSri = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_SRI,
};

struct SurfaceNormalParams {
  SurfaceNormalParams() {
    CHECK_EQ(window_size % 2u, 1u);
    CHECK_LT(window_size, 8u);
    CHECK_GT(window_size, 0u);
  }
  size_t window_size = 7u;
  size_t method = SurfaceNormalEstimationMethod::kFals;
  bool display = false;
};

struct MaxDistanceMapParams {
  MaxDistanceMapParams() { CHECK_EQ(window_size % 2u, 1u); }
  size_t window_size = 3u;
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

struct MinConvexityMapParams {
  MinConvexityMapParams() { CHECK_EQ(window_size % 2u, 1u); }
  size_t morphological_opening_size = 1u;
  size_t window_size = 5u;
  size_t step_size = 1u;
  bool display = false;
  bool use_morphological_opening = true;
  bool use_threshold = true;
  double min_convexity_threshold = 0.94;
};

struct FinalEdgeMapParams {
  size_t morphological_opening_size = 1u;
  size_t morphological_closing_size = 1u;
  bool use_morphological_opening = true;
  bool use_morphological_closing = true;
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

void visualizeDepthMap(const cv::Mat& depth_map, cv::viz::Viz3d* viz_3d) {
  viz_3d->setBackgroundColor(cv::viz::Color::gray());
  viz_3d->showWidget("cloud",
                     cv::viz::WCloud(depth_map, cv::viz::Color::red()));
  viz_3d->showWidget("coo", cv::viz::WCoordinateSystem(1.5));
  viz_3d->spinOnce(0, true);
}

void visualizeDepthMapWithNormals(const cv::Mat& depth_map,
                                  const cv::Mat& normals,
                                  cv::viz::Viz3d* viz_3d) {
  viz_3d->setBackgroundColor(cv::viz::Color::gray());
  viz_3d->showWidget("cloud",
                     cv::viz::WCloud(depth_map, cv::viz::Color::red()));
  viz_3d->showWidget("normals",
                     cv::viz::WCloudNormals(depth_map, normals, 1, 0.001f,
                                            cv::viz::Color::green()));
  viz_3d->showWidget("coo", cv::viz::WCoordinateSystem(1.5));
  viz_3d->spinOnce(0, true);
}

}  // depth_segmentation

#endif  // DEPTH_SEGMENTATION_COMMON_H_
