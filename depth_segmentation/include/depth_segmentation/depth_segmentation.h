
#ifndef DEPTH_SEGMENTATION_DEPTH_SEGMENTATION_H_
#define DEPTH_SEGMENTATION_DEPTH_SEGMENTATION_H_

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>

namespace depth_segmentation {

class CameraTracker {
 public:
  CameraTracker();
  void initialize(const size_t width, const size_t height,
                  const cv::Mat& rgb_camera_matrix,
                  const cv::Mat& depth_camera_matrix,
                  const std::string odometry_type = "RgbdICPOdometry");

  bool computeTransform(const cv::Mat& src_rgb_image,
                        const cv::Mat& src_depth_image,
                        const cv::Mat& dst_rgb_image,
                        const cv::Mat& dst_depth_image);

  inline void setDepthImage(const cv::Mat& depth_image) {
    depth_image_ = depth_image;
  }
  inline void setRgbImage(const cv::Mat& rgb_image) { rgb_image_ = rgb_image; }
  inline void setDepthCameraMatrix(const cv::Mat& camera_matrix) {
    depth_camera_matrix_ = camera_matrix;
  }
  inline void setRgbCameraMatrix(const cv::Mat& camera_matrix) {
    rgb_camera_matrix_ = camera_matrix;
  }
  inline cv::Mat getTransform() { return transform_; }
  inline cv::Mat getWorldTransform() { return world_transform_; }
  inline cv::Mat getRgbImage() { return rgb_image_; }
  inline cv::Mat getDepthImage() { return depth_image_; }

  bool computeTransform(const cv::Mat& dst_rgb_image,
                        const cv::Mat& dst_depth_image) {
    return computeTransform(rgb_image_, depth_image_, dst_rgb_image,
                            dst_depth_image);
  }

 private:
  cv::Mat depth_camera_matrix_;
  cv::Mat rgb_camera_matrix_;
  cv::Ptr<cv::rgbd::Odometry> odometry_;
  cv::Mat rgb_image_;
  cv::Mat depth_image_;
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
