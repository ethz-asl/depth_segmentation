#!/usr/bin/env python3

import sys
sys.path.insert(
    1,
    '/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/lib-dynload/'
)
sys.path.insert(
    1,
    '/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/'
)
sys.path.insert(1, '/opt/ros/lunar/lib/python3.6/site-packages/')
sys.path.insert(
    1, '/usr/local/Cellar/opencv/3.3.0_2/lib/python3.6/site-packages/')

sys.path.insert(1, '/usr/local/lib/python3.6/site-packages')

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import argparse
import os
import rospy

from sensor_msgs.msg import Image

try:
    import cv2 as cv
except ImportError:
    raise ImportError(
        'Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
        'configure environemnt variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)'
    )

inWidth = 1280  # 300
inHeight = 720  # 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

classNames = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
              'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor')
frame = None

bridge = CvBridge()


def rgb_image_callback(msg):
    """ RGB image callback."""
    global frame
    # print("got image.")
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    print(type(msg.data))
    # mat = np.asarray(msg.data, dtype=np.uint8)
    # frame = msg.data


if __name__ == "__main__":
    rospy.init_node('ssd_dector_ros', disable_signals=True)
    rate = rospy.Rate(30)

    full_path_to_file = os.path.realpath(__file__)
    path_list = full_path_to_file.split('/')
    ssd_folder = "/".join(path_list[0:-2])
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rgb_image_topic",
        default="/camera/rgb/image_raw",
        help=
        "ROS RGB camera image topic. If unspecified, will use the video flag.")
    parser.add_argument(
        "--video",
        help="path to video file. If empty, camera's stream will be used")
    parser.add_argument(
        "--prototxt",
        default=ssd_folder + "/nets/MobileNetSSD_deploy.prototxt",
        # default=ssd_folder + "/nets/MobileNet_deploy.prototxt",
        help="path to caffe prototxt")
    parser.add_argument(
        "-c",
        "--caffemodel",
        default=ssd_folder + "/models/MobileNetSSD_deploy.caffemodel",
        # default=ssd_folder + "/models/MobileNet_deploy.caffemodel",
        help="path to caffemodel file, download it here: "
        "https://github.com/chuanqi305/MobileNet-SSD/")
    parser.add_argument(
        "--thr",
        default=0.2,
        help="confidence threshold to filter out weak detections")
    args = parser.parse_args()

    net = cv.dnn.readNetFromCaffe(args.prototxt, args.caffemodel)
    print(args.rgb_image_topic)
    if args.rgb_image_topic:
        rgb_sub = rospy.Subscriber(
            args.rgb_image_topic, Image, rgb_image_callback, queue_size=1)
    elif args.video:
        cap = cv.VideoCapture(args.video)
    else:
        cap = cv.VideoCapture(0)

    if args.rgb_image_topic:
        while frame is None:
            print("waiting for image to arrive.")
            rate.sleep()
        print("got an image.")
    else:
        ret, frame = cap.read()
    (inHeight, inWidth, inChannels) = frame.shape
    inScaleFactor = 0.007843
    scale = 0.5
    inWidth = int(scale * inWidth)
    inHeight = int(scale * inHeight)
    WHRatio = inWidth / float(inHeight)
    frame_tmp = None
    while not rospy.is_shutdown():
        # Capture frame-by-frame
        if args.rgb_image_topic:
            pass
        else:
            ret, frame = cap.read()
        if frame_tmp == frame:
            continue
        frame_tmp = frame
        blob = cv.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight),
                                    meanVal)
        net.setInput(blob)
        detections = net.forward()

        cols = frame.shape[1]
        rows = frame.shape[0]

        if cols / float(rows) > WHRatio:
            cropSize = (int(rows * WHRatio), rows)
        else:
            cropSize = (cols, int(cols / WHRatio))

        y1 = int((rows - cropSize[1]) / 2)
        y2 = int(y1 + cropSize[1])
        x1 = int((cols - cropSize[0]) / 2)
        x2 = int(x1 + cropSize[0])
        frame = frame[y1:y2, x1:x2]

        cols = frame.shape[1]
        rows = frame.shape[0]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args.thr:
                class_id = int(detections[0, 0, i, 1])

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom),
                             (xRightTop, yRightTop), (0, 255, 0))
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv.getTextSize(
                    label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                             (xLeftBottom + labelSize[0],
                              yLeftBottom + baseLine), (255, 255,
                                                        255), cv.FILLED)
                cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("detections", frame)
        if cv.waitKey(1) >= 0:
            break
