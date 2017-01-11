#include "depth_segmentation/depth_segmentation.h"

#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo/photo.hpp>

namespace depth_segmentation {

CameraTracker::CameraTracker(const DepthCamera& depth_camera,
                             const RgbCamera& rgb_camera)
    : depth_camera_(depth_camera),
      rgb_camera_(rgb_camera),
      world_transform_(4, 4, CV_64FC1),
      transform_(4, 4, CV_64FC1) {
  world_transform_ = cv::Mat::eye(4, 4, CV_64FC1);
  transform_ = cv::Mat::eye(4, 4, CV_64FC1);
}

void CameraTracker::initialize(const std::string odometry_type) {
  CHECK(depth_camera_.initialized());
  CHECK(rgb_camera_.initialized());
  CHECK(!depth_camera_.getCameraMatrix().empty());
  CHECK(std::find(kCameraTrackerNames.begin(), kCameraTrackerNames.end(),
                  odometry_type) != kCameraTrackerNames.end());
  odometry_ = cv::rgbd::Odometry::create(odometry_type);
  odometry_->setCameraMatrix(depth_camera_.getCameraMatrix());

  LOG(INFO) << "CameraTracker initialized";
}

bool CameraTracker::computeTransform(const cv::Mat& src_rgb_image,
                                     const cv::Mat& src_depth_image,
                                     const cv::Mat& dst_rgb_image,
                                     const cv::Mat& dst_depth_image,
                                     const cv::Mat& src_depth_mask,
                                     const cv::Mat& dst_depth_mask) {
  CHECK(!src_rgb_image.empty());
  CHECK(!src_depth_image.empty());
  CHECK(!dst_rgb_image.empty());
  CHECK(!dst_depth_image.empty());
  CHECK(!src_depth_mask.empty());
  CHECK(!dst_depth_mask.empty());
  CHECK_EQ(src_rgb_image.size(), dst_rgb_image.size());
  CHECK_EQ(src_depth_image.size(), dst_depth_image.size());
  CHECK_EQ(src_rgb_image.size(), src_depth_image.size());
  CHECK_EQ(src_depth_image.size(), src_depth_mask.size());
  CHECK_EQ(dst_depth_image.size(), dst_depth_mask.size());
  CHECK(!world_transform_.empty());

  cv::Ptr<cv::rgbd::OdometryFrame> src_frame(new cv::rgbd::OdometryFrame(
      src_rgb_image, src_depth_image, src_depth_mask));
  cv::Ptr<cv::rgbd::OdometryFrame> dst_frame(new cv::rgbd::OdometryFrame(
      dst_rgb_image, dst_depth_image, dst_depth_mask));
  bool success = odometry_->compute(src_frame, dst_frame, transform_);
  if (success) {
    world_transform_ = transform_ * world_transform_;
  }
  return success;
}
void CameraTracker::visualize(const cv::Mat old_depth_image,
                              const cv::Mat new_depth_image) const {
  CHECK(!old_depth_image.empty());
  CHECK(!new_depth_image.empty());
  CHECK_EQ(old_depth_image.size(), new_depth_image.size());
  CHECK_EQ(old_depth_image.type(), CV_32FC1);
  CHECK_EQ(new_depth_image.type(), CV_32FC1);

  // Place both depth images into one.
  cv::Size size_old_depth = old_depth_image.size();
  cv::Size size_new_depth = new_depth_image.size();
  cv::Mat combined_depth(size_old_depth.height,
                         size_old_depth.width + size_new_depth.width, CV_32FC1);
  cv::Mat left(combined_depth,
               cv::Rect(0, 0, size_old_depth.width, size_old_depth.height));
  old_depth_image.copyTo(left);
  cv::Mat right(combined_depth,
                cv::Rect(size_old_depth.width, 0, size_new_depth.width,
                         size_new_depth.height));
  new_depth_image.copyTo(right);

  // Adjust the colors, such that the depth images look nicer.
  double min;
  double max;
  cv::minMaxIdx(combined_depth, &min, &max, 0, 0,
                cv::Mat(combined_depth == combined_depth));
  combined_depth -= min;
  cv::Mat adjusted_depth;
  cv::convertScaleAbs(combined_depth, adjusted_depth,
                      static_cast<double>(kImageRange) / (max - min));

  cv::imshow(kDebugWindowName, adjusted_depth);
  cv::waitKey(1);
}

void CameraTracker::createMask(const cv::Mat& depth, cv::Mat* mask) {
  CHECK(!depth.empty());
  CHECK_EQ(depth.type(), CV_32FC1);
  CHECK_NOTNULL(mask);
  CHECK(depth.size() == mask->size());
  for (size_t y = 0u; y < depth.rows; ++y) {
    for (size_t x = 0u; x < depth.cols; ++x) {
      if (cvIsNaN(depth.at<float>(y, x)) || depth.at<float>(y, x) > kMaxDepth ||
          depth.at<float>(y, x) <= FLT_EPSILON)
        mask->at<uchar>(y, x) = 0;
    }
  }
}

void CameraTracker::dilateFrame(cv::Mat& image, cv::Mat& depth) {
  CHECK(!image.empty());
  CHECK_EQ(image.type(), CV_8UC1);
  CHECK(!depth.empty());
  CHECK_EQ(depth.type(), CV_32FC1);
  CHECK_EQ(depth.size(), image.size());

  cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(kImageRange));
  createMask(depth, &mask);

  image.setTo(kImageRange, ~mask);
  cv::Mat min_image;
  cv::erode(image, min_image, cv::Mat());

  image.setTo(0, ~mask);
  cv::Mat max_image;
  cv::dilate(image, max_image, cv::Mat());

  depth.setTo(FLT_MAX, ~mask);
  cv::Mat min_depth;
  cv::erode(depth, min_depth, cv::Mat());

  depth.setTo(0, ~mask);
  cv::Mat max_depth;
  cv::dilate(depth, max_depth, cv::Mat());

  cv::Mat dilated_mask;
  cv::dilate(mask, dilated_mask, cv::Mat(), cv::Point(-1, -1), 1);
  for (size_t y = 0u; y < depth.rows; ++y) {
    for (size_t x = 0u; x < depth.cols; ++x) {
      if (!mask.at<uchar>(y, x) && dilated_mask.at<uchar>(y, x)) {
        image.at<uchar>(y, x) = static_cast<uchar>(
            0.5f * (static_cast<float>(min_image.at<uchar>(y, x)) +
                    static_cast<float>(max_image.at<uchar>(y, x))));
        depth.at<float>(y, x) =
            0.5f * (min_depth.at<float>(y, x) + max_depth.at<float>(y, x));
      }
    }
  }
}

void DepthSegmenter::initialize() {
  CHECK(depth_camera_.initialized());
  CHECK_EQ(params_.normals.window_size % 2, 1u);
  rgbd_normals_ = cv::rgbd::RgbdNormals(
      depth_camera_.getWidth(), depth_camera_.getHeight(), CV_32F,
      depth_camera_.getCameraMatrix(), params_.normals.window_size,
      params_.normals.method);
  LOG(INFO) << "DepthSegmenter initialized";
}

void DepthSegmenter::dynamicReconfigureCallback(
    depth_segmentation::DepthSegmenterConfig& config, uint32_t level) {
  // Surface normal params.
  if (config.normals_window_size % 2u != 1u) {
    // Resetting the config value to its previous value.
    config.normals_window_size = params_.normals.window_size;
    LOG(ERROR) << "Set the normals window size to an odd number.";
    return;
  }
  if (config.normals_window_size < 1u) {
    // Resetting the config value to its previous value.
    config.normals_window_size = params_.normals.window_size;
    LOG(ERROR) << "Set the normals window size to an odd value of at least 3.";
    return;
  }
  if (config.normals_method !=
          SurfaceNormalEstimationMethod::kDepthWindowFilter &&
      config.normals_window_size >= 8u) {
    // Resetting the config value to its previous value.
    config.normals_window_size = params_.normals.window_size;
    LOG(ERROR) << "Only normal method Own supports normal window sizes larger "
                  "than 7.";
    return;
  }
  params_.normals.method = config.normals_method;
  params_.normals.distance_factor_threshold =
      config.normals_distance_factor_threshold;
  params_.normals.window_size = config.normals_window_size;
  params_.normals.display = config.normals_display;

  // Max distance map params.
  if (config.max_distance_window_size % 2u != 1u) {
    // Resetting the config value to its previous value.
    config.max_distance_window_size = params_.max_distance.window_size;
    LOG(ERROR) << "Set the max distnace window size to an odd number.";
    return;
  }
  params_.max_distance.display = config.max_distance_display;
  params_.max_distance.exclude_nan_as_max_distance =
      config.max_distance_exclude_nan_as_max_distance;
  params_.max_distance.ignore_nan_coordinates =
      config.max_distance_ignore_nan_coordinates;
  params_.max_distance.noise_thresholding_factor =
      config.max_distance_noise_thresholding_factor;
  params_.max_distance.sensor_min_distance =
      config.max_distance_sensor_min_distance;
  params_.max_distance.sensor_noise_param_1 =
      config.max_distance_sensor_noise_param_1;
  params_.max_distance.sensor_noise_param_2 =
      config.max_distance_sensor_noise_param_2;
  params_.max_distance.sensor_noise_param_3 =
      config.max_distance_sensor_noise_param_3;
  params_.max_distance.use_threshold = config.max_distance_use_threshold;
  params_.max_distance.window_size = config.max_distance_window_size;

  // Min convexity map params.
  if (config.min_convexity_window_size % 2u != 1u) {
    // Resetting the config value to its previous value.
    config.min_convexity_window_size = params_.min_convexity.window_size;
    LOG(ERROR) << "Set the min convexity window size to an odd number.";
    return;
  }
  params_.min_convexity.morphological_opening_size =
      config.min_convexity_morphological_opening_size;
  params_.min_convexity.step_size = config.min_convexity_step_size;
  params_.min_convexity.use_morphological_opening =
      config.min_convexity_use_morphological_opening;
  params_.min_convexity.use_threshold = config.min_convexity_use_threshold;
  params_.min_convexity.threshold = config.min_convexity_threshold;
  params_.min_convexity.mask_threshold = config.min_convexity_mask_threshold;

  params_.min_convexity.display = config.min_convexity_display;
  params_.min_convexity.window_size = config.min_convexity_window_size;

  // Final edge map params.
  params_.final_edge.morphological_opening_size =
      config.final_edge_morphological_opening_size;
  params_.final_edge.morphological_closing_size =
      config.final_edge_morphological_closing_size;
  params_.final_edge.use_morphological_opening =
      config.final_edge_use_morphological_opening;
  params_.final_edge.use_morphological_closing =
      config.final_edge_use_morphological_closing;

  // Label map params.
  params_.label.display = config.label_display;

  LOG(INFO) << "Dynamic Reconfigure Request.";
}

void DepthSegmenter::computeDepthMap(const cv::Mat& depth_image,
                                     cv::Mat* depth_map) {
  CHECK(!depth_image.empty());
  CHECK_EQ(depth_image.type(), CV_32FC1);
  CHECK_NOTNULL(depth_map);
  CHECK_EQ(depth_image.size(), depth_map->size());
  CHECK_EQ(depth_map->type(), CV_32FC3);
  CHECK(!depth_camera_.getCameraMatrix().empty());

  cv::rgbd::depthTo3d(depth_image, depth_camera_.getCameraMatrix(), *depth_map);
}

void DepthSegmenter::computeMaxDistanceMap(const cv::Mat& depth_map,
                                           cv::Mat* max_distance_map) {
  CHECK(!depth_map.empty());
  CHECK_EQ(depth_map.type(), CV_32FC3);
  CHECK_NOTNULL(max_distance_map);
  CHECK_EQ(max_distance_map->type(), CV_32FC1);
  // Check if window_size is odd.
  CHECK_EQ(params_.max_distance.window_size % 2, 1u);

  max_distance_map->setTo(cv::Scalar(0.0f));

  const size_t kernel_size = params_.max_distance.window_size;
  const size_t n_kernels = kernel_size * kernel_size - 1u;

  // Define the n kernels and compute the filtered images.
  for (size_t i = 0u; i < n_kernels + 1u; ++i) {
    if (i == n_kernels / 2u) {
      continue;
    }
    cv::Mat kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_32FC1);
    kernel.at<float>(i) = -1.0f;
    kernel.at<float>(n_kernels / 2u) = 1.0f;

    // Compute the filtered images.
    cv::Mat filtered_image(depth_map.size(), CV_32FC3);
    cv::filter2D(depth_map, filtered_image, CV_32FC3, kernel);

    // Calculate the norm over the three channels.
    std::vector<cv::Mat> channels(3);
    cv::split(filtered_image, channels);
    cv::Mat distance_map(depth_map.size(), CV_32FC1);
    if (params_.max_distance.ignore_nan_coordinates) {
      // Ignore nan values for the distance calculation.
      cv::Mat mask_0 = cv::Mat(channels[0] == channels[0]);
      cv::Mat mask_1 = cv::Mat(channels[1] == channels[1]);
      cv::Mat mask_2 = cv::Mat(channels[2] == channels[2]);
      mask_0.convertTo(mask_0, CV_32FC1);
      mask_1.convertTo(mask_1, CV_32FC1);
      mask_2.convertTo(mask_2, CV_32FC1);
      distance_map = mask_0.mul(channels[0].mul(channels[0])) +
                     mask_1.mul(channels[1].mul(channels[1])) +
                     mask_2.mul(channels[2].mul(channels[2]));
    } else {
      // If at least one of the coordinates is nan the distance will be nan.
      distance_map = channels[0].mul(channels[0]) +
                     channels[1].mul(channels[1]) +
                     channels[2].mul(channels[2]);
    }

    if (params_.max_distance.exclude_nan_as_max_distance) {
      cv::Mat mask = cv::Mat(distance_map == distance_map);
      mask.convertTo(mask, CV_32FC1);
      distance_map = mask.mul(distance_map);
    }
    // Individually set the maximum pixel value of the two matrices.
    cv::max(*max_distance_map, distance_map, *max_distance_map);
  }

  cv::sqrt(*max_distance_map, *max_distance_map);
  std::vector<cv::Mat> channels(3);
  cv::split(depth_map, channels);

  // Threshold the max_distance_map to get an edge map.
  if (params_.max_distance.use_threshold) {
    for (size_t i = 0u; i < depth_map.cols * depth_map.rows; ++i) {
      // Threshold the distance map based on Nguyen et al. (2012) noise model.
      // TODO(ff): Theta should be the angle between the normal and the camera
      // direction. (Here, a mean value is used, as suggested by Tateno et al.
      // (2016))
      static constexpr float theta = 30.f * CV_PI / 180.f;
      float z = (channels[2]).at<float>(i);
      float sigma_axial_noise =
          params_.max_distance.sensor_noise_param_1 +
          params_.max_distance.sensor_noise_param_2 *
              (z - params_.max_distance.sensor_min_distance) *
              (z - params_.max_distance.sensor_min_distance) +
          params_.max_distance.sensor_noise_param_3 / cv::sqrt(z) * theta *
              theta / (CV_PI / 2.0f - theta) * (CV_PI / 2.0f - theta);
      if (max_distance_map->at<float>(i) >
          sigma_axial_noise * params_.max_distance.noise_thresholding_factor) {
        max_distance_map->at<float>(i) = 1.0f;
      } else {
        max_distance_map->at<float>(i) = 0.0f;
      }
    }
  }
  if (params_.max_distance.display) {
    static const std::string kWindowName = "MaxDistanceMap";
    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(kWindowName, *max_distance_map);
    cv::waitKey(1);
  }
}

void DepthSegmenter::computeNormalMap(const cv::Mat& depth_map,
                                      cv::Mat* normal_map) {
  CHECK(!depth_map.empty());
  CHECK(depth_map.type() == CV_32FC3 &&
            (params_.normals.method == SurfaceNormalEstimationMethod::kFals ||
             params_.normals.method == SurfaceNormalEstimationMethod::kSri ||
             params_.normals.method ==
                 SurfaceNormalEstimationMethod::kDepthWindowFilter) ||
        (depth_map.type() == CV_32FC1 || depth_map.type() == CV_16UC1 ||
         depth_map.type() == CV_32FC3) &&
            params_.normals.method == SurfaceNormalEstimationMethod::kLinemod);
  CHECK_NOTNULL(normal_map);
  if (params_.normals.method !=
      SurfaceNormalEstimationMethod::kDepthWindowFilter) {
    rgbd_normals_(depth_map, *normal_map);
  } else {
    computeOwnNormals(params_.normals, depth_map, normal_map);
  }
  if (params_.normals.display) {
    static const std::string kWindowName = "NormalMap";
    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    // Taking the negative values of the normal map, as all normals point in
    // negative z-direction.
    cv::imshow(kWindowName, -*normal_map);
    cv::waitKey(1);
  }
}

void DepthSegmenter::computeMinConvexityMap(const cv::Mat& depth_map,
                                            const cv::Mat& normal_map,
                                            cv::Mat* min_convexity_map) {
  CHECK(!depth_map.empty());
  CHECK(!normal_map.empty());
  CHECK_EQ(depth_map.type(), CV_32FC3);
  CHECK_EQ(normal_map.type(), CV_32FC3);
  CHECK_EQ(depth_map.size(), normal_map.size());
  CHECK_NOTNULL(min_convexity_map);
  CHECK_EQ(min_convexity_map->type(), CV_32FC1);
  CHECK_EQ(depth_map.size(), min_convexity_map->size());
  // Check if window_size is odd.
  CHECK_EQ(params_.min_convexity.window_size % 2, 1u);
  min_convexity_map->setTo(cv::Scalar(10.0f));

  const size_t kernel_size = params_.min_convexity.window_size +
                             (params_.min_convexity.step_size - 1u) *
                                 (params_.min_convexity.window_size - 1u);
  const size_t n_kernels =
      params_.min_convexity.window_size * params_.min_convexity.window_size -
      1u;
  // Define the n point-wise distance kernels and compute the filtered images.
  // The kernels for i look as follows (e.g. window_size = 5, i = 6):
  //     0  0  0  0  0
  //     0  1  0  0  0
  //     0  0 -1  0  0
  //     0  0  0  0  0
  //     0  0  0  0  0
  for (size_t i = 0u; i < n_kernels + 1u;
       i += static_cast<size_t>(i % kernel_size == kernel_size) * kernel_size +
            params_.min_convexity.step_size) {
    if (i == n_kernels / 2u) {
      continue;
    }
    cv::Mat difference_kernel =
        cv::Mat::zeros(kernel_size, kernel_size, CV_32FC1);
    difference_kernel.at<float>(i) = 1.0f;
    difference_kernel.at<float>(n_kernels / 2u) = -1.0f;

    // Compute the filtered images.
    cv::Mat difference_map(depth_map.size(), CV_32FC3);
    cv::filter2D(depth_map, difference_map, CV_32FC3, difference_kernel);

    // Calculate the dot product over the three channels of difference_map and
    // normal_map.
    cv::Mat difference_times_normal(depth_map.size(), CV_32FC3);
    difference_times_normal = difference_map.mul(-normal_map);
    std::vector<cv::Mat> channels(3);
    cv::split(difference_times_normal, channels);
    cv::Mat vector_projection(depth_map.size(), CV_32FC1);
    vector_projection = channels[0] + channels[1] + channels[2];

    // TODO(ff): Check if params_.min_convexity.mask_threshold should be
    // mid-point distance dependent.
    // maybe do something like:
    // std::vector<cv::Mat> depth_map_channels(3);
    // cv::split(depth_map, depth_map_channels);
    // vector_projection = vector_projection.mul(depth_map_channels[2]);

    cv::Mat concavity_mask(depth_map.size(), CV_32FC1);
    cv::Mat convexity_mask(depth_map.size(), CV_32FC1);

    // Split the projected vector images into convex and concave
    // regions/masks.
    constexpr float kMaxBinaryValue = 1.0f;
    cv::threshold(vector_projection, convexity_mask,
                  params_.min_convexity.mask_threshold, kMaxBinaryValue,
                  cv::THRESH_BINARY);
    cv::threshold(vector_projection, concavity_mask,
                  params_.min_convexity.mask_threshold, kMaxBinaryValue,
                  cv::THRESH_BINARY_INV);

    cv::Mat normal_kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_32FC1);
    normal_kernel.at<float>(i) = 1.0f;

    cv::Mat filtered_normal_image = cv::Mat::zeros(normal_map.size(), CV_32FC3);
    cv::filter2D(normal_map, filtered_normal_image, CV_32FC3, normal_kernel);

    // TODO(ff): Create a function for this mulitplication and projections.
    cv::Mat normal_times_filtered_normal(depth_map.size(), CV_32FC3);
    normal_times_filtered_normal = normal_map.mul(filtered_normal_image);
    std::vector<cv::Mat> normal_channels(3);
    cv::split(normal_times_filtered_normal, normal_channels);
    cv::Mat normal_vector_projection(depth_map.size(), CV_32FC1);
    normal_vector_projection =
        normal_channels[0] + normal_channels[1] + normal_channels[2];
    normal_vector_projection = concavity_mask.mul(normal_vector_projection);

    cv::Mat convexity_map = cv::Mat::ones(depth_map.size(), CV_32FC1);
    convexity_map = convexity_mask + normal_vector_projection;

    // Individually set the minimum pixel value of the two matrices.
    cv::min(*min_convexity_map, convexity_map, *min_convexity_map);
  }

  if (params_.min_convexity.use_threshold) {
    constexpr float kMaxBinaryValue = 1.0f;
    cv::threshold(*min_convexity_map, *min_convexity_map,
                  params_.min_convexity.threshold, kMaxBinaryValue,
                  cv::THRESH_BINARY);
  }

  if (params_.min_convexity.use_morphological_opening) {
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2u * params_.min_convexity.morphological_opening_size + 1u,
                 2u * params_.min_convexity.morphological_opening_size + 1u),
        cv::Point(params_.min_convexity.morphological_opening_size,
                  params_.min_convexity.morphological_opening_size));
    cv::morphologyEx(*min_convexity_map, *min_convexity_map, cv::MORPH_OPEN,
                     element);
  }

  if (params_.min_convexity.display) {
    static const std::string kWindowName = "MinConcavityMap";
    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(kWindowName, *min_convexity_map);
    cv::waitKey(1);
  }
}

void DepthSegmenter::computeFinalEdgeMap(const cv::Mat& convexity_map,
                                         const cv::Mat& distance_map,
                                         cv::Mat* edge_map) {
  CHECK(!convexity_map.empty());
  CHECK(!distance_map.empty());
  CHECK_EQ(convexity_map.type(), CV_32FC1);
  CHECK_EQ(distance_map.type(), CV_32FC1);
  CHECK_EQ(convexity_map.size(), distance_map.size());
  CHECK_NOTNULL(edge_map);
  if (params_.final_edge.use_morphological_opening) {
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2u * params_.final_edge.morphological_opening_size + 1u,
                 2u * params_.final_edge.morphological_opening_size + 1u),
        cv::Point(params_.final_edge.morphological_opening_size,
                  params_.final_edge.morphological_opening_size));

    cv::morphologyEx(convexity_map, convexity_map, cv::MORPH_OPEN, element);
  }
  if (params_.final_edge.use_morphological_closing) {
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2u * params_.final_edge.morphological_closing_size + 1u,
                 2u * params_.final_edge.morphological_closing_size + 1u),
        cv::Point(params_.final_edge.morphological_closing_size,
                  params_.final_edge.morphological_closing_size));
    cv::morphologyEx(distance_map, distance_map, cv::MORPH_CLOSE, element);
  }

  *edge_map = convexity_map - distance_map;
  if (params_.final_edge.display) {
    static const std::string kWindowName = "FinalEdgeMap";
    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    imshow(kWindowName, *edge_map);
    cv::waitKey(1);
  }
}

void DepthSegmenter::findBlobs(const cv::Mat& binary,
                               std::vector<std::vector<cv::Point2i>>* labels) {
  CHECK(!binary.empty());
  CHECK_EQ(binary.type(), CV_32FC1);
  CHECK_NOTNULL(labels)->clear();
  cv::Mat label_image;
  binary.convertTo(label_image, CV_32SC1);

  // Labels start at 2 as we use 0 for background and 1 for unlabled.
  size_t label_count = 2u;
  for (size_t y = 0u; y < label_image.rows; ++y) {
    for (size_t x = 0u; x < label_image.cols; ++x) {
      if (label_image.at<int>(y, x) != 1) {
        continue;
      }
      cv::Rect rect;
      cv::floodFill(label_image, cv::Point(x, y), label_count, &rect, 0, 0,
                    cv::FLOODFILL_FIXED_RANGE);
      std::vector<cv::Point2i> blob;
      size_t rect_size_y = rect.y + rect.height;
      size_t rect_size_x = rect.x + rect.width;
      for (size_t i = rect.y; i < rect_size_y; ++i) {
        for (size_t j = rect.x; j < rect_size_x; ++j) {
          if (label_image.at<int>(i, j) != label_count) {
            continue;
          }
          blob.push_back(cv::Point2i(j, i));
        }
      }
      if (blob.size() > 1u) {
        labels->push_back(blob);
        ++label_count;
      }
    }
  }
}

void DepthSegmenter::inpaintImage(const cv::Mat& image, cv::Mat* inpainted) {
  CHECK(false) << "THIS IS UNTESTED AND PROBABLY SOMEWHAT WRONG.";
  CHECK(!image.empty());
  CHECK_NOTNULL(inpainted);
  cv::Mat border_image;
  cv::Mat inpainted_8bit;
  double inpaint_radius = 3.0;
  int make_border = 1;
  cv::copyMakeBorder(image, border_image, make_border, make_border, make_border,
                     make_border, cv::BORDER_REPLICATE);
  border_image.convertTo(border_image, CV_8UC3, 255.0f);
  cv::inpaint(border_image, (border_image == border_image), inpainted_8bit,
              inpaint_radius, cv::INPAINT_TELEA);
  inpainted_8bit.convertTo(inpainted_8bit, CV_32FC3, 1.0f / 255.0f);
  *inpainted = inpainted_8bit(
      cv::Rect(make_border, make_border, image.cols, image.rows));
  cv::namedWindow("inpainted", CV_WINDOW_AUTOSIZE);
  cv::imshow("inpainted", *inpainted);
  cv::waitKey(1);
}

void DepthSegmenter::labelMap(const cv::Mat& edge_map, cv::Mat* labeled_map) {
  CHECK(!edge_map.empty());
  CHECK_EQ(edge_map.type(), CV_32FC1);
  CHECK_NOTNULL(labeled_map);
  cv::RNG rng(12345);
  cv::Mat binary_edge_map;

  std::vector<std::vector<cv::Point2i>> labels;
  constexpr float kEdgeMapThresholdValue = 0.0f;
  constexpr float kMaxBinaryValue = 1.0f;
  cv::threshold(edge_map, binary_edge_map, kEdgeMapThresholdValue,
                kMaxBinaryValue, cv::THRESH_BINARY);
  findBlobs(binary_edge_map, &labels);
  cv::Mat output = cv::Mat::zeros(binary_edge_map.size(), CV_8UC3);

  // Randomly color the labels
  for (size_t i = 0u; i < labels.size(); ++i) {
    unsigned char r = 255 * (rand() / (1.0 + RAND_MAX));
    unsigned char g = 255 * (rand() / (1.0 + RAND_MAX));
    unsigned char b = 255 * (rand() / (1.0 + RAND_MAX));

    for (size_t j = 0u; j < labels[i].size(); ++j) {
      int x = labels[i][j].x;
      int y = labels[i][j].y;

      output.at<cv::Vec3b>(y, x)[0] = b;
      output.at<cv::Vec3b>(y, x)[1] = g;
      output.at<cv::Vec3b>(y, x)[2] = r;
    }
  }
  if (params_.label.display) {
    static const std::string kWindowName = "LabelMap";
    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    imshow(kWindowName, output);
    cv::waitKey(1);
  }
  *labeled_map = output;
}

}  // namespace depth_segmentation
