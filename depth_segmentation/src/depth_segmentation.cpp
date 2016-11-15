#include "depth_segmentation/depth_segmentation.h"

namespace depth_segmentation {

CameraTracker::CameraTracker() : transform_() {}

void CameraTracker::initialize(const size_t width, const size_t height,
                               const cv::Mat& rgb_camera_matrix,
                               const cv::Mat& depth_camera_matrix,
                               const std::string odometry_type) {
  depth_camera_matrix_ = depth_camera_matrix;
  rgb_camera_matrix_ = rgb_camera_matrix;
  odometry_ = cv::rgbd::Odometry::create(odometry_type);
  odometry_->setCameraMatrix(depth_camera_matrix_);
  world_transform_ = cv::Mat::eye(4, 4, CV_64FC1);
  transform_ = cv::Mat::eye(4, 4, CV_64FC1);
  LOG(INFO) << "CameraTracker initialized";
}

bool CameraTracker::computeTransform(const cv::Mat& src_rgb_image,
                                     const cv::Mat& src_depth_image,
                                     const cv::Mat& dst_rgb_image,
                                     const cv::Mat& dst_depth_image) {
  CV_Assert(!src_rgb_image.empty());
  CV_Assert(!src_depth_image.empty());
  CV_Assert(!dst_rgb_image.empty());
  CV_Assert(!dst_depth_image.empty());

  cv::Ptr<cv::rgbd::OdometryFrame> src_frame(new cv::rgbd::OdometryFrame(
      src_rgb_image, src_depth_image,
      cv::Mat(src_rgb_image.size(), CV_8UC1, cv::Scalar(255))));
  cv::Ptr<cv::rgbd::OdometryFrame> dst_frame(new cv::rgbd::OdometryFrame(
      dst_rgb_image, dst_depth_image,
      cv::Mat(dst_rgb_image.size(), CV_8UC1, cv::Scalar(255))));
  bool success = odometry_->compute(src_frame, dst_frame, transform_);
  if (success) {
    world_transform_ *= transform_;
  }
  return success;
}
}
