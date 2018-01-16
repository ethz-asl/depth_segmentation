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
        # cv2.normalize(cv_image, cv_image, 16000, 0, cv2.NORM_MINMAX)
        # img8 = (cv_image / 256).astype('uint8')
        # ret, mask = cv2.threshold(img8, 1, 255, cv2.THRESH_BINARY_INV)

        # dst = cv2.inpaint(img8, mask, 3, cv2.INPAINT_NS)
        # dst = cv2.inpaint(img8, mask_inv, 3, cv2.INPAINT_TELEA)
        # cv_image = (dst * 256).astype('uint16')

        kernel = np.ones((3, 3), np.uint8)
        cv_image = cv2.dilate(cv_image, kernel, 3)

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
