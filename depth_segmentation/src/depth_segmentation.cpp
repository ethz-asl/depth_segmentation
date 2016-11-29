#include "depth_segmentation/depth_segmentation.h"

#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

namespace depth_segmentation {

CameraTracker::CameraTracker(const DepthCamera& depth_camera,
                             const RgbCamera& rgb_camera)
    : depth_camera_(depth_camera),
      rgb_camera_(rgb_camera),
      world_transform_(4, 4, CV_64FC1),
      transform_(4, 4, CV_64FC1) {
  world_transform_ = cv::Mat::eye(4, 4, CV_64FC1);
  transform_ = cv::Mat::eye(4, 4, CV_64FC1);
#ifdef DISPLAY_DEPTH_IMAGES
  cv::namedWindow(kDebugWindowName, cv::WINDOW_AUTOSIZE);
#endif  // DISPLAY_DEPTH_IMAGES
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
  CHECK_EQ(surface_normal_params_.window_size % 2, 1);
  rgbd_normals_ = cv::rgbd::RgbdNormals(
      depth_camera_.getWidth(), depth_camera_.getHeight(), CV_32F,
      depth_camera_.getCameraMatrix(), surface_normal_params_.window_size,
      surface_normal_params_.method);
  LOG(INFO) << "DepthSegmenter initialized";
}

void DepthSegmenter::computeDepthMap(const cv::Mat& depth_image,
                                     cv::Mat* depth_map) {
  CHECK(!depth_image.empty());
  CHECK_EQ(depth_image.type(), CV_32FC1);
  CHECK_NOTNULL(depth_map);
  CHECK_EQ(depth_image.size(), depth_map->size());
  CHECK_EQ(depth_map->type(), CV_32FC3);

  cv::rgbd::depthTo3d(depth_image, depth_camera_.getCameraMatrix(), *depth_map);
}

void DepthSegmenter::computeMaxDistanceMap(const cv::Mat& depth_map,
                                           cv::Mat* max_distance_map) {
  CHECK(!depth_map.empty());
  CHECK_EQ(depth_map.type(), CV_32FC3);
  CHECK_NOTNULL(max_distance_map);
  CHECK_EQ(max_distance_map->type(), CV_32FC1);
  // Check if window_size is odd.
  CHECK_EQ(max_distance_map_params_.window_size % 2, 1);

  max_distance_map->setTo(cv::Scalar(0.0f));

  const size_t kernel_size = max_distance_map_params_.window_size;
  const size_t n_kernels = kernel_size * kernel_size - 1u;

  // Define the n kernels and compute the filtered images.
  for (size_t i = 0u; i < n_kernels + 1u; ++i) {
    if (i == n_kernels / 2u) {
      continue;
    }
    cv::Mat kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_32FC1);
    kernel.at<float>(i) = -1;
    kernel.at<float>(n_kernels / 2u) = 1;

    // Compute the filtered images.
    cv::Mat filtered_image(depth_map.size(), CV_32FC3);
    cv::filter2D(depth_map, filtered_image, CV_32FC3, kernel);

    // Calculate the norm over the three channels.
    std::vector<cv::Mat> channels(3);
    cv::split(filtered_image, channels);
    cv::Mat distance_map(depth_map.size(), CV_32FC1);
    if (max_distance_map_params_.ignore_nan_coordinates) {
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

    if (max_distance_map_params_.exclude_nan_as_max_distance) {
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
  if (max_distance_map_params_.use_threshold) {
    for (size_t i = 0u; i < depth_map.cols * depth_map.rows; ++i) {
      // Threshold the distance map based on Nguyen et al. (2012) noise model.
      // TODO(ff): Theta should be the angle between the normal and the camera
      // direction. (Here, a mean value is used, as suggested by Tateno et al.
      // (2016))
      static constexpr float theta = 30.f * CV_PI / 180.f;
      float z = (channels[2]).at<float>(i);
      float sigma_axial_noise =
          max_distance_map_params_.sensor_noise_param_1 +
          max_distance_map_params_.sensor_noise_param_2 *
              (z - max_distance_map_params_.sensor_min_distance) *
              (z - max_distance_map_params_.sensor_min_distance) +
          max_distance_map_params_.sensor_noise_param_3 / cv::sqrt(z) * theta *
              theta / (CV_PI / 2.0f - theta) * (CV_PI / 2.0f - theta);
      if (max_distance_map->at<float>(i) >
          sigma_axial_noise *
              max_distance_map_params_.noise_thresholding_factor) {
        max_distance_map->at<float>(i) = 1.0f;
      } else {
        max_distance_map->at<float>(i) = 0.0f;
      }
    }
  }
#ifdef DISPLAY_DISTANCE_MAP_IMAGES
  cv::imshow(kDebugWindowName, *max_distance_map);
  cv::waitKey(1);
#endif  // DISPLAY_DISTANCE_MAP_IMAGES
}

void DepthSegmenter::computeNormalMap(const cv::Mat& depth_map,
                                      cv::Mat* normal_map) {
  CHECK(!depth_map.empty());
  size_t normal_method = surface_normal_params_.method;
  CHECK(depth_map.type() == CV_32FC3 &&
            (normal_method == cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS ||
             normal_method == cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_SRI) ||
        (depth_map.type() == CV_32FC1) &&
            normal_method ==
                cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);
  CHECK_NOTNULL(normal_map);

  rgbd_normals_(depth_map, *normal_map);
#ifdef DISPLAY_NORMAL_IMAGES
  // Taking the negative values of the normal map, as all normals point in
  // negative z-direction.
  imshow(kDebugWindowName, -*normal_map);
  cv::waitKey(1);
#endif  // DISPLAY_NORMAL_IMAGES
}

void DepthSegmenter::computeMinConcavityMap(const cv::Mat& depth_map,
                                            const cv::Mat& normal_map,
                                            cv::Mat* min_concavity_map) {}

}  // namespace depth_segmentation
