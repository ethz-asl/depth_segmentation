#!/usr/bin/env python
"""
Subscribe to an image topic (/image_raw) of type sensor_msgs/Image
Apply filter(s) to the resulting image, republish as /image_filtered
"""
from __future__ import print_function
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class SubThenFilter:
    def __init__(self):
        self.sub = rospy.Subscriber(
            "/image_raw", Image, self.image_callback, queue_size=1)
        self.pub = rospy.Publisher("/image_filtered", Image, queue_size=1)
        self.bridge = CvBridge()
        self.median_blur_size = 3
        self.use_median_blur = False

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)

        cv_image = np.nan_to_num(cv_image)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilate_iterations = 1
        cv_image = cv2.dilate(cv_image, kernel, dilate_iterations)

        if self.use_median_blur:
            cv_image = cv2.medianBlur(cv_image, self.median_blur_size)

        try:
            msg = self.bridge.cv2_to_imgmsg(cv_image, "passthrough")
            data.data = msg.data
            self.pub.publish(data)
        except CvBridgeError as e:
            print(e)


if __name__ == "__main__":
    rospy.init_node("image_filter", anonymous=True)
    sf = SubThenFilter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")
cv2.destroyAllWindows()
