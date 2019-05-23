#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <mask_rcnn_ros/Result.h>

#include "depth_segmentation/depth_segmentation.h"
#include "depth_segmentation/ros_common.h"

// typedef pcl::PointSurfel PointType;

struct PointSurfelLabel {
  PCL_ADD_POINT4D;
  PCL_ADD_NORMAL4D;
  PCL_ADD_RGB;
  uint8_t label;
  uint8_t instance;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointSurfelLabel,
    (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
        float, normal_y, normal_y)(float, normal_z, normal_z)(float, rgb, rgb)(
        uint8_t, label, label)(uint8_t, instance, instance))

typedef PointSurfelLabel PointType;

class DepthSegmentationNode {
 public:
  DepthSegmentationNode()
      : node_handle_("~"),
        image_transport_(node_handle_),
        camera_info_ready_(false),
        depth_image_sub_(image_transport_, depth_segmentation::kDepthImageTopic,
                         100),
        rgb_image_sub_(image_transport_, depth_segmentation::kRgbImageTopic,
                       100),
        // label_image_sub_(image_transport_,
        // depth_segmentation::kLabelImageTopic,
        //                  100),
        segmentation_result_sub_(node_handle_,
                                 depth_segmentation::kSegmentationTopic, 1),
        depth_info_sub_(node_handle_, depth_segmentation::kDepthCameraInfoTopic,
                        100),
        rgb_info_sub_(node_handle_, depth_segmentation::kRgbCameraInfoTopic,
                      100),
        image_sync_policy_(ImageSyncPolicy(30), depth_image_sub_,
                           rgb_image_sub_, segmentation_result_sub_),
        camera_info_sync_policy_(CameraInfoSyncPolicy(10), depth_info_sub_,
                                 rgb_info_sub_),
        depth_camera_(),
        rgb_camera_(),
        params_(),
        camera_tracker_(depth_camera_, rgb_camera_),
        depth_segmenter_(depth_camera_, params_) {
    image_sync_policy_.registerCallback(
        boost::bind(&DepthSegmentationNode::imageCallback, this, _1, _2, _3));
    camera_info_sync_policy_.registerCallback(
        boost::bind(&DepthSegmentationNode::cameraInfoCallback, this, _1, _2));
    point_cloud2_segment_pub_ =
        node_handle_.advertise<sensor_msgs::PointCloud2>("object_segment",
                                                         1000);
    point_cloud2_scene_pub_ =
        node_handle_.advertise<sensor_msgs::PointCloud2>("segmented_scene", 1);
  }

 private:
  ros::NodeHandle node_handle_;
  image_transport::ImageTransport image_transport_;
  tf::TransformBroadcaster transform_broadcaster_;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image, mask_rcnn_ros::Result>
      ImageSyncPolicy;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>
      CameraInfoSyncPolicy;

  bool camera_info_ready_;
  depth_segmentation::DepthCamera depth_camera_;
  depth_segmentation::RgbCamera rgb_camera_;

  depth_segmentation::Params params_;

 public:
  depth_segmentation::CameraTracker camera_tracker_;
  depth_segmentation::DepthSegmenter depth_segmenter_;

 private:
  message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> rgb_info_sub_;
  message_filters::Subscriber<mask_rcnn_ros::Result> segmentation_result_sub_;

  image_transport::SubscriberFilter depth_image_sub_;
  image_transport::SubscriberFilter rgb_image_sub_;
  // image_transport::SubscriberFilter label_image_sub_;
  // image_transport::SubscriberFilter segmentation_result_sub_;

  ros::Publisher point_cloud2_segment_pub_;
  ros::Publisher point_cloud2_scene_pub_;

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

  void publish_segments(
      const std::vector<depth_segmentation::Segment>& segments,
      const std_msgs::Header& header) {
    CHECK_GT(segments.size(), 0u);
    pcl::PointCloud<PointType>::Ptr scene_pcl(new pcl::PointCloud<PointType>);
    for (depth_segmentation::Segment segment : segments) {
      CHECK_GT(segment.points.size(), 0u);
      pcl::PointCloud<PointType>::Ptr segment_pcl(
          new pcl::PointCloud<PointType>);
      for (std::size_t i = 0; i < segment.points.size(); ++i) {
        PointType point_pcl;
        point_pcl.x = segment.points[i][0];
        point_pcl.y = segment.points[i][1];
        point_pcl.z = segment.points[i][2];
        point_pcl.normal_x = segment.normals[i][0];
        point_pcl.normal_y = segment.normals[i][1];
        point_pcl.normal_z = segment.normals[i][2];
        point_pcl.r = segment.original_colors[i][0];
        point_pcl.g = segment.original_colors[i][1];
        point_pcl.b = segment.original_colors[i][2];
        if (segment.semantic_label.size() > 0) {
          point_pcl.label = *(segment.semantic_label.begin());
        } else {
          point_pcl.label = 0u;
        }
        if (segment.instance_label.size() > 0) {
          point_pcl.instance = *(segment.instance_label.begin());
        } else {
          point_pcl.instance = 0u;
        }

        // LOG(ERROR) << "Segment instance: " << unsigned(point_pcl.instance);

        segment_pcl->push_back(point_pcl);
        scene_pcl->push_back(point_pcl);
      }
      sensor_msgs::PointCloud2 pcl2_msg;
      pcl::toROSMsg(*segment_pcl, pcl2_msg);
      pcl2_msg.header.stamp = header.stamp;
      pcl2_msg.header.frame_id = header.frame_id;
      point_cloud2_segment_pub_.publish(pcl2_msg);
    }
    // Just for rviz also publish the whole scene, as otherwise only ~10
    // segments are shown:
    // https://github.com/ros-visualization/rviz/issues/689
    sensor_msgs::PointCloud2 pcl2_msg;
    pcl::toROSMsg(*scene_pcl, pcl2_msg);
    pcl2_msg.header.stamp = header.stamp;
    pcl2_msg.header.frame_id = header.frame_id;
    point_cloud2_scene_pub_.publish(pcl2_msg);
  }

  void segmentationFromROSMsg(
      const mask_rcnn_ros::Result::ConstPtr& segmentation_msg,
      depth_segmentation::Segmentation* segmentation) {
    segmentation->masks.reserve(segmentation_msg->masks.size());
    segmentation->labels.reserve(segmentation_msg->masks.size());
    for (int i = 0; i < segmentation_msg->masks.size(); ++i) {
      cv_bridge::CvImagePtr cv_mask_image;
      cv_mask_image = cv_bridge::toCvCopy(segmentation_msg->masks[i],
                                          sensor_msgs::image_encodings::MONO8);
      segmentation->masks.push_back(cv_mask_image->image.clone());
      segmentation->labels.push_back(segmentation_msg->class_ids[i]);
    }
  }
  // void imageCallback(const sensor_msgs::Image::ConstPtr& depth_msg,
  //                    const sensor_msgs::Image::ConstPtr& rgb_msg,
  //                    const sensor_msgs::Image::ConstPtr& label_msg) {

  void imageCallback(const sensor_msgs::Image::ConstPtr& depth_msg,
                     const sensor_msgs::Image::ConstPtr& rgb_msg,
                     const mask_rcnn_ros::Result::ConstPtr& segmentation_msg) {
    if (camera_info_ready_) {
      voxblox::timing::Timer preprocessing("rgbd_preprocessing");
      cv_bridge::CvImagePtr cv_depth_image;
      cv::Mat rescaled_depth;

      if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        cv_depth_image = cv_bridge::toCvCopy(
            depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
      } else if (depth_msg->encoding ==
                 sensor_msgs::image_encodings::TYPE_32FC1) {
        cv_depth_image = cv_bridge::toCvCopy(
            depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
      }

      rescaled_depth = cv::Mat(cv_depth_image->image.size(), CV_32FC1);
      cv::rgbd::rescaleDepth(cv_depth_image->image, CV_32FC1, rescaled_depth);
      // rescaled_depth = rescaled_depth * 1.05f;

      cv_bridge::CvImagePtr cv_rgb_image;
      cv_rgb_image =
          cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat bw_image(cv_rgb_image->image.size(), CV_8UC1);

      cvtColor(cv_rgb_image->image, bw_image, cv::COLOR_RGB2GRAY);

      // cv_bridge::CvImagePtr cv_label_image;
      // cv_label_image = cv_bridge::toCvCopy(
      //     label_msg, sensor_msgs::image_encodings::TYPE_16UC1);
      depth_segmentation::Segmentation segmentation;
      segmentationFromROSMsg(segmentation_msg, &segmentation);

      cv::Mat mask(bw_image.size(), CV_8UC1,
                   cv::Scalar(depth_segmentation::CameraTracker::kImageRange));

      preprocessing.Stop();

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
        constexpr bool kUseTracker = false;
        if (kUseTracker) {
          if (camera_tracker_.computeTransform(bw_image, rescaled_depth,
                                               mask)) {
            publish_tf(camera_tracker_.getWorldTransform(),
                       depth_msg->header.stamp);
          } else {
            LOG(ERROR) << "Failed to compute Transform.";
          }
        }

        voxblox::timing::Timer compute_depth_map("compute_depth_map");
        cv::Mat depth_map(depth_camera_.getWidth(), depth_camera_.getHeight(),
                          CV_32FC3);
        depth_segmenter_.computeDepthMap(rescaled_depth, &depth_map);
        compute_depth_map.Stop();

        // Compute normal map.
        voxblox::timing::Timer compute_normal_map("compute_normal_map");
        cv::Mat normal_map(depth_map.size(), CV_32FC3, 0.0f);
        if (params_.normals.method ==
                depth_segmentation::SurfaceNormalEstimationMethod::kFals ||
            params_.normals.method ==
                depth_segmentation::SurfaceNormalEstimationMethod::kSri ||
            params_.normals.method ==
                depth_segmentation::SurfaceNormalEstimationMethod::
                    kDepthWindowFilter) {
          depth_segmenter_.computeNormalMap(depth_map, &normal_map);
        } else if (params_.normals.method ==
                   depth_segmentation::SurfaceNormalEstimationMethod::
                       kLinemod) {
          depth_segmenter_.computeNormalMap(cv_depth_image->image, &normal_map);
        }
        compute_normal_map.Stop();

        voxblox::timing::Timer compute_edge_map("compute_edge_map");

        // Compute depth discontinuity map.
        cv::Mat discontinuity_map = cv::Mat::zeros(
            depth_camera_.getWidth(), depth_camera_.getHeight(), CV_32FC1);
        if (params_.depth_discontinuity.use_discontinuity) {
          depth_segmenter_.computeDepthDiscontinuityMap(rescaled_depth,
                                                        &discontinuity_map);
        }

        // Compute maximum distance map.
        cv::Mat distance_map = cv::Mat::zeros(
            depth_camera_.getWidth(), depth_camera_.getHeight(), CV_32FC1);
        if (params_.max_distance.use_max_distance) {
          depth_segmenter_.computeMaxDistanceMap(depth_map, &distance_map);
        }

        // Compute minimum convexity map.
        cv::Mat convexity_map = cv::Mat::zeros(
            depth_camera_.getWidth(), depth_camera_.getHeight(), CV_32FC1);
        if (params_.min_convexity.use_min_convexity) {
          depth_segmenter_.computeMinConvexityMap(depth_map, normal_map,
                                                  &convexity_map);
        }

        // Compute final edge map.
        cv::Mat edge_map(depth_camera_.getWidth(), depth_camera_.getHeight(),
                         CV_32FC1);
        depth_segmenter_.computeFinalEdgeMap(convexity_map, distance_map,
                                             discontinuity_map, &edge_map);

        compute_edge_map.Stop();

        cv::Mat label_map(edge_map.size(), CV_32FC1);
        cv::Mat remove_no_values =
            cv::Mat::zeros(edge_map.size(), edge_map.type());
        edge_map.copyTo(remove_no_values, rescaled_depth == rescaled_depth);
        edge_map = remove_no_values;
        std::vector<depth_segmentation::Segment> segments;
        std::vector<cv::Mat> segment_masks;
        depth_segmenter_.labelMap(cv_rgb_image->image, rescaled_depth,
                                  cv_depth_image->image, segmentation,
                                  depth_map, edge_map, normal_map,
                                  rgb_msg->header.stamp.toSec(), &label_map,
                                  &segment_masks, &segments);

        if (segments.size() > 0) {
          publish_segments(segments, depth_msg->header);
        }

        // Update the member images to the new images.
        // TODO(ff): Consider only doing this, when we are far enough away
        // from a frame. (Which basically means we would set a keyframe.)
        depth_camera_.setImage(rescaled_depth);
        depth_camera_.setMask(mask);
        rgb_camera_.setImage(bw_image);
      } else {
        depth_camera_.setImage(rescaled_depth);
        depth_camera_.setMask(mask);
        rgb_camera_.setImage(bw_image);
      }

      ROS_INFO_STREAM("Timings: " << std::endl
                                  << voxblox::timing::Timing::Print());
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
  LOG(INFO) << "Starting depth segmentation ... ";
  ros::init(argc, argv, "depth_segmentation_node");
  DepthSegmentationNode depth_segmentation_node;

  dynamic_reconfigure::Server<depth_segmentation::DepthSegmenterConfig>
      reconfigure_server;
  dynamic_reconfigure::Server<depth_segmentation::DepthSegmenterConfig>::
      CallbackType dynamic_reconfigure_function;

  dynamic_reconfigure_function = boost::bind(
      &depth_segmentation::DepthSegmenter::dynamicReconfigureCallback,
      &depth_segmentation_node.depth_segmenter_, _1, _2);
  reconfigure_server.setCallback(dynamic_reconfigure_function);

  while (ros::ok()) {
    ros::spin();
  }

  return EXIT_SUCCESS;
}
