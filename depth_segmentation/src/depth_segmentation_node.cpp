#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "depth_segmentation/depth_segmentation.h"
#include "depth_segmentation/ros_common.h"

class DepthSegmentationNode {
 public:
  DepthSegmentationNode()
      : node_handle_("~"),
        image_transport_(node_handle_),
        camera_info_ready_(false),
        depth_image_sub_(image_transport_, depth_segmentation::kDepthImageTopic,
                         1),
        rgb_image_sub_(image_transport_, depth_segmentation::kRgbImageTopic, 1),
        depth_info_sub_(node_handle_, depth_segmentation::kDepthCameraInfoTopic,
                        1),
        rgb_info_sub_(node_handle_, depth_segmentation::kRgbCameraInfoTopic, 1),
        image_sync_policy_(ImageSyncPolicy(10), depth_image_sub_,
                           rgb_image_sub_),
        camera_info_sync_policy_(CameraInfoSyncPolicy(10), depth_info_sub_,
                                 rgb_info_sub_),
        depth_camera_(),
        rgb_camera_(),
        surface_normal_params_(),
        max_distance_map_params_(),
        min_concavity_map_params_(),
        camera_tracker_(depth_camera_, rgb_camera_),
        depth_segmenter_(depth_camera_, surface_normal_params_,
                         max_distance_map_params_, min_concavity_map_params_) {
    image_sync_policy_.registerCallback(
        boost::bind(&DepthSegmentationNode::imageCallback, this, _1, _2));
    camera_info_sync_policy_.registerCallback(
        boost::bind(&DepthSegmentationNode::cameraInfoCallback, this, _1, _2));
  }

 private:
  ros::NodeHandle node_handle_;
  image_transport::ImageTransport image_transport_;
  tf::TransformBroadcaster transform_broadcaster_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      ImageSyncPolicy;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>
      CameraInfoSyncPolicy;

  bool camera_info_ready_;
  depth_segmentation::DepthCamera depth_camera_;
  depth_segmentation::RgbCamera rgb_camera_;

  depth_segmentation::SurfaceNormalParams surface_normal_params_;
  depth_segmentation::MaxDistanceMapParams max_distance_map_params_;
  depth_segmentation::MinConcavityMapParams min_concavity_map_params_;

  depth_segmentation::CameraTracker camera_tracker_;
  depth_segmentation::DepthSegmenter depth_segmenter_;

  message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> rgb_info_sub_;

  image_transport::SubscriberFilter depth_image_sub_;
  image_transport::SubscriberFilter rgb_image_sub_;

  message_filters::Synchronizer<ImageSyncPolicy> image_sync_policy_;
  message_filters::Synchronizer<CameraInfoSyncPolicy> camera_info_sync_policy_;

  void publish_tf(const cv::Mat cv_transform, const ros::Time& timestamp) {
    // Rotate such that the world frame initially aligns with the camera_link
    // frame.
    static const cv::Mat kWorldAlign =
        (cv::Mat_<double>(4, 4) << 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat cv_transform_world_aligned = cv_transform * kWorldAlign;

    tf::Vector3 translation_tf(cv_transform_world_aligned.at<double>(0, 3),
                               cv_transform_world_aligned.at<double>(1, 3),
                               cv_transform_world_aligned.at<double>(2, 3));

    tf::Matrix3x3 rotation_tf;
    for (size_t i = 0u; i < 3u; ++i) {
      for (size_t j = 0u; j < 3u; ++j) {
        rotation_tf[j][i] = cv_transform_world_aligned.at<double>(j, i);
      }
    }
    tf::Transform transform;
    transform.setOrigin(translation_tf);
    transform.setBasis(rotation_tf);

    transform_broadcaster_.sendTransform(tf::StampedTransform(
        transform, timestamp, depth_segmentation::kTfDepthCameraFrame,
        depth_segmentation::kTfWorldFrame));
  }

  void imageCallback(const sensor_msgs::Image::ConstPtr& depth_msg,
                     const sensor_msgs::Image::ConstPtr& rgb_msg) {
    if (camera_info_ready_) {
      cv_bridge::CvImagePtr cv_depth_image;
      cv_depth_image = cv_bridge::toCvCopy(
          depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);

      cv::Mat rescaled_depth(cv_depth_image->image.size(), CV_32FC1);
      cv::rgbd::rescaleDepth(cv_depth_image->image, CV_32FC1, rescaled_depth);

      cv_bridge::CvImagePtr cv_rgb_image;
      cv_rgb_image =
          cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::RGB8);
      cv::Mat bw_image(cv_rgb_image->image.size(), CV_8UC1);

      cvtColor(cv_rgb_image->image, bw_image, cv::COLOR_RGB2GRAY);

      cv::Mat mask(bw_image.size(), CV_8UC1,
                   cv::Scalar(depth_segmentation::CameraTracker::kImageRange));
      if (!camera_tracker_.getRgbImage().empty() &&
          !camera_tracker_.getDepthImage().empty()) {
#ifdef WRITE_IMAGES
        cv::imwrite(std::to_string(cv_rgb_image->header.stamp.toSec()) +
                        "_rgb_image.png",
                    cv_rgb_image->image);
        cv::imwrite(std::to_string(cv_rgb_image->header.stamp.toSec()) +
                        "_bw_image.png",
                    bw_image);
        cv::imwrite(std::to_string(depth_msg->header.stamp.toSec()) +
                        "_depth_image.png",
                    rescaled_depth);
        cv::imwrite(
            std::to_string(depth_msg->header.stamp.toSec()) + "_depth_mask.png",
            mask);
#endif  // WRITE_IMAGES

#ifdef DISPLAY_DEPTH_IMAGES
        camera_tracker_.visualize(camera_tracker_.getDepthImage(),
                                  rescaled_depth);
#endif  // DISPLAY_DEPTH_IMAGES

        // Compute transform from tracker.
        if (camera_tracker_.computeTransform(bw_image, rescaled_depth, mask)) {
          publish_tf(camera_tracker_.getWorldTransform(),
                     depth_msg->header.stamp);

          cv::Mat depth_map(depth_camera_.getWidth(), depth_camera_.getHeight(),
                            CV_32FC3);
          depth_segmenter_.computeDepthMap(rescaled_depth, &depth_map);

          // Compute normal map.
          cv::Mat normal_map(depth_map.size(), CV_32FC3);
          if (surface_normal_params_.method ==
                  cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS ||
              surface_normal_params_.method ==
                  cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_SRI) {
            depth_segmenter_.computeNormalMap(depth_map, &normal_map);
          } else if (surface_normal_params_.method ==
                     cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD) {
            depth_segmenter_.computeNormalMap(rescaled_depth, &normal_map);
          }

          // Compute maximum distance map.
          cv::Mat distance_map(depth_camera_.getWidth(),
                               depth_camera_.getHeight(), CV_32FC1);
          depth_segmenter_.computeMaxDistanceMap(depth_map, &distance_map);

          // Update the member images to the new images.
          // TODO(ff): Consider only doing this, when we are far enough away
          // from a frame. (Which basically means we would set a keyframe.)
          depth_camera_.setImage(rescaled_depth);
          depth_camera_.setMask(mask);
          rgb_camera_.setImage(bw_image);
        } else {
          LOG(ERROR) << "Failed to compute Transform.";
        }
      } else {
        depth_camera_.setImage(rescaled_depth);
        depth_camera_.setMask(mask);
        rgb_camera_.setImage(bw_image);
      }
    }
  }

  void cameraInfoCallback(
      const sensor_msgs::CameraInfo::ConstPtr& depth_camera_info_msg,
      const sensor_msgs::CameraInfo::ConstPtr& rgb_camera_info_msg) {
    if (camera_info_ready_) {
      return;
    }

    sensor_msgs::CameraInfo depth_info;
    depth_info = *depth_camera_info_msg;
    Eigen::Vector2d depth_image_size(depth_info.width, depth_info.height);

    cv::Mat K_depth = cv::Mat::eye(3, 3, CV_32FC1);
    K_depth.at<float>(0, 0) = depth_info.K[0];
    K_depth.at<float>(0, 2) = depth_info.K[2];
    K_depth.at<float>(1, 1) = depth_info.K[4];
    K_depth.at<float>(1, 2) = depth_info.K[5];
    K_depth.at<float>(2, 2) = depth_info.K[8];

    depth_camera_.initialize(depth_image_size.x(), depth_image_size.y(),
                             CV_32FC1, K_depth);

    sensor_msgs::CameraInfo rgb_info;
    rgb_info = *rgb_camera_info_msg;
    Eigen::Vector2d rgb_image_size(rgb_info.width, rgb_info.height);

    cv::Mat K_rgb = cv::Mat::eye(3, 3, CV_32FC1);
    K_rgb.at<float>(0, 0) = rgb_info.K[0];
    K_rgb.at<float>(0, 2) = rgb_info.K[2];
    K_rgb.at<float>(1, 1) = rgb_info.K[4];
    K_rgb.at<float>(1, 2) = rgb_info.K[5];
    K_rgb.at<float>(2, 2) = rgb_info.K[8];

    rgb_camera_.initialize(rgb_image_size.x(), rgb_image_size.y(), CV_8UC1,
                           K_rgb);

    depth_segmenter_.initialize();
    camera_tracker_.initialize(
        camera_tracker_.kCameraTrackerNames
            [camera_tracker_.CameraTrackerType::kRgbdICPOdometry]);

    camera_info_ready_ = true;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ros::init(argc, argv, "depth_segmentation_node");
  DepthSegmentationNode depth_segmentation_node;

  while (ros::ok()) {
    ros::spin();
  }

  return EXIT_SUCCESS;
}
