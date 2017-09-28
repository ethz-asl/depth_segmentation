#!/usr/bin/env python3
import numpy as np
import argparse
import os
import sys

sys.path.insert(
    1,
    '/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/lib-dynload/'
)
sys.path.insert(
    1,
    '/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/'
)
sys.path.insert(1, '/usr/local/lib/python3.6/site-packages')

try:
    import cv2 as cv
except ImportError:
    raise ImportError(
        'Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
        'configure environemnt variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)'
    )

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

classNames = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
              'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor')

if __name__ == "__main__":
    full_path_to_file = os.path.realpath(__file__)
    path_list = full_path_to_file.split('/')
    ssd_folder = "/".join(path_list[0:-2])
    parser = argparse.ArgumentParser()
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
    if args.video:
        cap = cv.VideoCapture(args.video)
    else:
        cap = cv.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
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
