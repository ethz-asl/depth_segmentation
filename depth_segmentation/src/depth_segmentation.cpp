#include "depth_segmentation/depth_segmentation.h"

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
  cv::minMaxIdx(combined_depth, &min, &max);
  combined_depth -= min;
  cv::Mat adjusted_depth;
  cv::convertScaleAbs(combined_depth, adjusted_depth,
                      (double)kImageRange / double(max - min));

  cv::imshow(kDebugWindowName, adjusted_depth);
  cv::waitKey(0);
}

void CameraTracker::createMask(const cv::Mat& depth, cv::Mat* mask) {
  CHECK(!depth.empty());
  CHECK(depth.type() == CV_32FC1);
  CHECK(depth.size() == mask->size());
  CHECK(mask);
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
  CHECK(image.type() == CV_8UC1);
  CHECK(!depth.empty());
  CHECK(depth.type() == CV_32FC1);
  CHECK(depth.size() == image.size());

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
  LOG(INFO) << depth_camera_.getCameraMatrix();
  rgbd_normals_ =
      cv::rgbd::RgbdNormals(depth_camera_.getWidth(), depth_camera_.getHeight(),
                            CV_32FC1, depth_camera_.getCameraMatrix());
}

void DepthSegmenter::computeDepthMap(const cv::Mat& depth_image,
                                     cv::Mat* depth_map) {
  CHECK(!depth_image.empty());
  CHECK(depth_image.type() == CV_32FC1);
  CHECK_NOTNULL(depth_map);
  cv::rgbd::depthTo3d(depth_image, depth_camera_.getCameraMatrix(), *depth_map);
}

void DepthSegmenter::computeNormalMap(const cv::Mat& depth_map,
                                      cv::Mat* normal_map) {
  CHECK(!depth_map.empty());
  CHECK(depth_map.type() == CV_32FC1);
  CHECK_NOTNULL(normal_map);
  rgbd_normals_(depth_map, *normal_map);
}
}
