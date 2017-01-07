#ifndef DEPTH_SEGMENTATION_COMMON_H_
#define DEPTH_SEGMENTATION_COMMON_H_

#include <string>

#include <glog/logging.h>
#include <opencv2/highgui.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/viz/vizcore.hpp>

namespace depth_segmentation {

const static std::string kDebugWindowName = "DebugImages";

enum SurfaceNormalEstimationMethod {
  kFals = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS,
  kLinemod = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD,
  kSri = cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_SRI,
  kOwn = 3,
};

struct SurfaceNormalParams {
  SurfaceNormalParams() {
    CHECK_EQ(window_size % 2u, 1u);
    CHECK_GT(window_size, 1u);
    if (method != SurfaceNormalEstimationMethod::kOwn) {
      CHECK_LT(window_size, 8u);
    }
  }
  size_t window_size = 11u;
  size_t method = SurfaceNormalEstimationMethod::kOwn;
  bool display = false;
  double distance_factor_threshold = 0.01;
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
  double noise_thresholding_factor = 6.0;
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
  double threshold = 0.95;
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

struct Params {
  Params()
      : final_edge(), label(), max_distance(), min_convexity(), normals() {}
  FinalEdgeMapParams final_edge;
  LabelMapParams label;
  MaxDistanceMapParams max_distance;
  MinConvexityMapParams min_convexity;
  SurfaceNormalParams normals;
};

void visualizeDepthMap(const cv::Mat& depth_map, cv::viz::Viz3d* viz_3d) {
  CHECK(!depth_map.empty());
  CHECK_EQ(depth_map.type(), CV_32FC3);
  CHECK_NOTNULL(viz_3d);
  viz_3d->setBackgroundColor(cv::viz::Color::gray());
  viz_3d->showWidget("cloud",
                     cv::viz::WCloud(depth_map, cv::viz::Color::red()));
  viz_3d->showWidget("coo", cv::viz::WCoordinateSystem(1.5));
  viz_3d->spinOnce(0, true);
}

void visualizeDepthMapWithNormals(const cv::Mat& depth_map,
                                  const cv::Mat& normals,
                                  cv::viz::Viz3d* viz_3d) {
  CHECK(!depth_map.empty());
  CHECK_EQ(depth_map.type(), CV_32FC3);
  CHECK(!normals.empty());
  CHECK_EQ(normals.type(), CV_32FC3);
  CHECK_EQ(depth_map.size(), normals.size());
  CHECK_NOTNULL(viz_3d);
  viz_3d->setBackgroundColor(cv::viz::Color::gray());
  viz_3d->showWidget("cloud",
                     cv::viz::WCloud(depth_map, cv::viz::Color::red()));
  viz_3d->showWidget("normals",
                     cv::viz::WCloudNormals(depth_map, normals, 50, 0.02f,
                                            cv::viz::Color::green()));
  viz_3d->showWidget("coo", cv::viz::WCoordinateSystem(1.5));
  viz_3d->spinOnce(0, true);
}

void computeCovariance(const cv::Mat& neighborhood, const cv::Vec3f& mean,
                       const size_t neighborhood_size, cv::Mat* covariance) {
  CHECK(!neighborhood.empty());
  CHECK_GT(neighborhood_size, 0u);
  CHECK_NOTNULL(covariance);

  *covariance = cv::Mat::zeros(3, 3, CV_32F);

  for (size_t i = 0u; i < neighborhood_size; ++i) {
    cv::Vec3f point;
    point[0] = neighborhood.at<float>(0, i) - mean[0];
    point[1] = neighborhood.at<float>(1, i) - mean[1];
    point[2] = neighborhood.at<float>(2, i) - mean[2];

    covariance->at<float>(1, 1) += point[1] * point[1];
    covariance->at<float>(1, 2) += point[1] * point[2];
    covariance->at<float>(2, 2) += point[2] * point[2];

    point *= point[0];
    covariance->at<float>(0, 0) += point[0];
    covariance->at<float>(0, 1) += point[1];
    covariance->at<float>(0, 2) += point[2];
  }
  // Assign the symmetric elements of the covariance matrix.
  covariance->at<float>(1, 0) = covariance->at<float>(0, 1);
  covariance->at<float>(2, 0) = covariance->at<float>(0, 2);
  covariance->at<float>(2, 1) = covariance->at<float>(1, 2);
}

size_t findNeighborhood(const cv::Mat& depth_map, const size_t window_size,
                        const float max_distance, const size_t x,
                        const size_t y, cv::Mat* neighborhood,
                        cv::Vec3f* mean) {
  CHECK(!depth_map.empty());
  CHECK_GT(window_size, 0u);
  CHECK_EQ(window_size % 2u, 1u);
  CHECK_GE(max_distance, 0.0f);
  CHECK_GE(x, 0u);
  CHECK_GE(y, 0u);
  CHECK_LT(x, depth_map.cols);
  CHECK_LT(y, depth_map.rows);
  CHECK_NOTNULL(neighborhood);
  CHECK_NOTNULL(mean);

  size_t neighborhood_size = 0u;
  *neighborhood = cv::Mat::zeros(3, window_size * window_size, CV_32FC1);
  cv::Vec3f mid_point = depth_map.at<cv::Vec3f>(y, x);
  for (size_t y_idx = 0u; y_idx < window_size; ++y_idx) {
    int y_filter_idx = y + y_idx - window_size / 2u;
    if (y_filter_idx < 0 || y_filter_idx >= depth_map.rows) {
      continue;
    }
    for (size_t x_idx = 0u; x_idx < window_size; ++x_idx) {
      int x_filter_idx = x + x_idx - window_size / 2u;
      if (x_filter_idx < 0 || x_filter_idx >= depth_map.cols) {
        continue;
      }
      CHECK_GE(y_filter_idx, 0u);
      CHECK_GE(x_filter_idx, 0u);
      CHECK_LT(y_filter_idx, depth_map.rows);
      CHECK_LT(x_filter_idx, depth_map.cols);

      cv::Vec3f filter_point =
          depth_map.at<cv::Vec3f>(y_filter_idx, x_filter_idx);

      // Compute Euclidean distance between filter_point and mid_point.
      cv::Vec3f difference = mid_point - filter_point;
      float euclidean_dist = cv::sqrt(difference.dot(difference));
      if (euclidean_dist < max_distance) {
        // Add the filter_point to neighborhood set.
        for (size_t coordinate = 0; coordinate < 3; ++coordinate) {
          neighborhood->at<float>(coordinate, neighborhood_size) =
              filter_point[coordinate];
        }
        ++neighborhood_size;
        *mean += filter_point;
      }
    }
  }
  *mean /= static_cast<float>(neighborhood_size);
  return neighborhood_size;
}

/*! \brief Compute point normals of a depth image.
 *
 * Compute the point normals by looking at a neighborhood around each pixel.
 * We're taking a standard squared kernel, where we discard points that are too
 * far away from the center point (by evaluating the Euclidean distance).
 */
void computeOwnNormals(const SurfaceNormalParams& params,
                       const cv::Mat& depth_map, cv::Mat* normals) {
  CHECK(!depth_map.empty());
  CHECK_EQ(depth_map.type(), CV_32FC3);
  CHECK_NOTNULL(normals);
  CHECK_EQ(depth_map.size(), normals->size());

  cv::Mat neighborhood =
      cv::Mat::zeros(3, params.window_size * params.window_size, CV_32FC1);
  cv::Mat eigenvalues;
  cv::Mat eigenvectors;
  cv::Mat covariance(3, 3, CV_32FC1);
  covariance = cv::Mat::zeros(3, 3, CV_32FC1);
  cv::Vec3f mean;
  cv::Vec3f mid_point;

  float float_nan = std::numeric_limits<float>::quiet_NaN();
#pragma omp parallel for private(neighborhood, eigenvalues, eigenvectors, \
                                 covariance, mean, mid_point)
  for (size_t y = 0u; y < depth_map.rows; ++y) {
    for (size_t x = 0u; x < depth_map.cols; ++x) {
      mid_point = depth_map.at<cv::Vec3f>(y, x);
      // Skip point if z value is nan.
      if (cvIsNaN(mid_point[2])) {
        continue;
      }
      float max_distance = params.distance_factor_threshold * mid_point[2];
      mean[0] = 0.0f;
      mean[1] = 0.0f;
      mean[2] = 0.0f;

      size_t neighborhood_size =
          findNeighborhood(depth_map, params.window_size, max_distance, x, y,
                           &neighborhood, &mean);
      if (neighborhood_size > 1u) {
        computeCovariance(neighborhood, mean, neighborhood_size, &covariance);
        // Compute Eigen vectors.
        cv::eigen(covariance, eigenvalues, eigenvectors);
        // Get the Eigenvector corresponding to the smallest Eigenvalue.
        const size_t n_th_eigenvector = 2u;
        for (size_t coordinate = 0u; coordinate < 3u; ++coordinate) {
          normals->at<cv::Vec3f>(y, x)[coordinate] =
              eigenvectors.at<float>(n_th_eigenvector, coordinate);
        }
        // Re-Orient normals to point towards camera
        if (normals->at<cv::Vec3f>(y, x)[2] > 0.0f) {
          normals->at<cv::Vec3f>(y, x) = -normals->at<cv::Vec3f>(y, x);
        }
      } else {
        for (size_t coordinate = 0u; coordinate < 3u; ++coordinate) {
          normals->at<cv::Vec3f>(y, x)[coordinate] = float_nan;
        }
      }
    }
  }
}
}  // depth_segmentation

#endif  // DEPTH_SEGMENTATION_COMMON_H_
