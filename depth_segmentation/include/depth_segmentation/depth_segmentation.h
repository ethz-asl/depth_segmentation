#ifndef DEPTH_SEGMENTATION_DEPTH_SEGMENTATION_H_
#define DEPTH_SEGMENTATION_DEPTH_SEGMENTATION_H_

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>

namespace depth_segmentation {

static const std::string kDebugWindowName = "DebugImages";
class CameraTracker {
 public:
  CameraTracker();
  void initialize(const size_t width, const size_t height,
                  const cv::Mat& rgb_camera_matrix,
                  const cv::Mat& depth_camera_matrix,
                  const std::string odometry_type);

  void initialize(const size_t width, const size_t height,
                  const cv::Mat& rgb_camera_matrix,
                  const cv::Mat& depth_camera_matrix) {
    return initialize(width, height, rgb_camera_matrix, depth_camera_matrix,
                      kCameraTrackerNames[CameraTrackerType::kRgbdICPOdometry]);
  }

  bool computeTransform(const cv::Mat& src_rgb_image,
                        const cv::Mat& src_depth_image,
                        const cv::Mat& dst_rgb_image,
                        const cv::Mat& dst_depth_image,
                        const cv::Mat& src_depth_mask,
                        const cv::Mat& dst_depth_mask);
  bool computeTransform(const cv::Mat& dst_rgb_image,
                        const cv::Mat& dst_depth_image,
                        const cv::Mat& dst_depth_mask) {
    return computeTransform(rgb_image_, depth_image_, dst_rgb_image,
                            dst_depth_image, depth_mask_, dst_depth_mask);
  }
  void createMask(const cv::Mat& depth, cv::Mat* mask);
  void dilateFrame(cv::Mat& image, cv::Mat& depth);

  inline void setDepthImage(const cv::Mat& depth_image) {
    depth_image_ = depth_image;
  }
  inline void setDepthMask(const cv::Mat& depth_mask) {
    depth_mask_ = depth_mask;
  }
  inline void setRgbImage(const cv::Mat& rgb_image) { rgb_image_ = rgb_image; }
  inline void setDepthCameraMatrix(const cv::Mat& camera_matrix) {
    depth_camera_matrix_ = camera_matrix;
  }
  inline void setRgbCameraMatrix(const cv::Mat& camera_matrix) {
    rgb_camera_matrix_ = camera_matrix;
  }
  inline cv::Mat getTransform() const { return transform_; }
  inline cv::Mat getWorldTransform() const { return world_transform_; }
  inline cv::Mat getRgbImage() const { return rgb_image_; }
  inline cv::Mat getDepthImage() const { return depth_image_; }
  inline cv::Mat getDepthMask() const { return depth_mask_; }

  void visualize(const cv::Mat old_depth_image,
                 const cv::Mat new_depth_image) const;

  enum CameraTrackerType {
    kRgbdICPOdometry = 0,
    kRgbdOdometry = 1,
    kICPOdometry = 2,
  };
  static constexpr size_t kImageRange = 255;
  static constexpr double kMinDepth = 0.15;
  static constexpr double kMaxDepth = 10.0;
  const std::vector<std::string> kCameraTrackerNames = {
      "RgbdICPOdometry", "RgbdOdometry", "ICPOdometry"};

 private:
  cv::Mat depth_camera_matrix_;
  cv::Mat rgb_camera_matrix_;
  cv::Ptr<cv::rgbd::Odometry> odometry_;
  cv::Mat rgb_image_;
  cv::Mat depth_image_;
  cv::Mat depth_mask_;
  cv::Mat transform_;
  cv::Mat world_transform_;
};

void depthMap(const cv::Mat& depth_image, cv::Mat* depth_map);
void maxDistanceMap(const cv::Mat& image, cv::Mat* distance_map);
void normalMap(const cv::Mat image, cv::Mat* normal_map);
void minConcavityMap(const cv::Mat& image, cv::Mat* concavity_map);
void concavityAwareNormalMap(const cv::Mat& concavity_map,
                             const cv::Mat& normal_map, cv::Mat* combined_map);
void edgeMap(const cv::Mat& image, cv::Mat* edge_map);
void labelMap(const cv::Mat& edge_map, cv::Mat* labeled_map);
}

#endif  // DEPTH_SEGMENTATION_DEPTH_SEGMENTATION_H_
