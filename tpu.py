import argparse
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from threading import Thread
from PIL import Image
from DetectionTF import Detection
import utils
from shapely.geometry import Point, Polygon

import cv2
import numpy as np

from centroidtracker import DirectionCentroidTracker, CentroidTracker

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--modeldir", help="Folder the .tflite file is located in", required=True
)
parser.add_argument(
    "--src",
    help="Source for opencv VideoCapture",
    default="rtsp://192.168.1.90/stream/profile1=r",
)
parser.add_argument(
    "--threshold",
    help="Minimum confidence threshold for displaying detected objects",
    default=0.5,
)
parser.add_argument("--output", help="Write video to file", default="")

args = parser.parse_args()

MODEL_NAME = args.modeldir
SRC = args.src
THRESHOLD = float(args.threshold)
OUTPUT = args.output

# Get path to current working directory
CWD_PATH = os.getcwd()

# Initialize video stream
stream = cv2.VideoCapture(SRC)

time.sleep(1)

SRC_WIDTH = int(stream.get(3))
SRC_HEIGHT = int(stream.get(4))

frame_num = 0

p1 = (0, 0)
p2 = (1000, 0)
p3 = (1000, 1080)
p4 = (0, 1080)

coords = [p1, p2, p3, p4]

# Create a Polygon
poly = Polygon(coords)
inPolygon = defaultdict(lambda: False)
intruders = []

# cenTracker = DirectionCentroidTracker(p1, p2, maxDisappeared=20)
cenTracker = CentroidTracker(maxDisappeared=40)

detector = Detection(MODEL_NAME)

if OUTPUT:
    out = cv2.VideoWriter(
        OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), 30, (SRC_WIDTH, SRC_HEIGHT)
    )

while True:
    __, frame1 = stream.read()
    if frame1 is None and OUTPUT:
        out.release()
        break
    frame = frame1.copy()

    im = Image.fromarray(frame)

    detection_res = detector.detect_objects(im, THRESHOLD)
    rects = list(map(lambda x: tuple(x.bounding_box.flatten()), detection_res))

    for detected_obj in detection_res:
        if detected_obj.label_id == 0:
            utils.draw_label(
                frame,
                tuple(detected_obj.bounding_box.flatten()),
                f"{detected_obj.score}",
            )
            utils.draw_detection_box(frame, tuple(detected_obj.bounding_box.flatten()))

    objects = cenTracker.update(frame, rects)

    for (objectID, centroid) in objects.items():

        # isThatSide = cenTracker.isBottomOrRightSide(centroid[0], centroid[1])
        # if isThatSide and objectID in cenTracker.topLeftSide:
        #     cenTracker.topLeftSide.remove(objectID)
        #     cenTracker.bottomRightSide.append(objectID)
        #     # moved from top left to btm right, call API!
        #     bottomRightCount += 1
        # elif not isThatSide and objectID in cenTracker.bottomRightSide:
        #     cenTracker.bottomRightSide.remove(objectID)
        #     cenTracker.topLeftSide.append(objectID)
        #     # moved top btm right to top left, call API!!
        #     topLeftCount += 1
        # totalCount = bottomRightCount - topLeftCount

        # utils.draw_centroid(frame, centroid, objectID)

        utils.draw_centroid(frame, centroid, objectID)
        if Point(centroid).within(poly):
            if not inPolygon[objectID]:
                inPolygon[objectID] = True
                # print(f'{objectID} has intruded into the polygon')
                intruders.append(objectID)
            # inPolygon[objectID] += 1
            # print(inPolygon[objectID])
            # if inPolygon[objectID] >= 50:
            #     print(f'{objectID} in polygon for {inPolygon[objectID]} frames')
        pass

    # Draw Line / Polygon
    # cv2.line(frame, p1, p2, (0, 0, 255), 2)
    for i in range(len(coords)):
        c1 = coords[i - 1]
        c2 = coords[i]
        cv2.line(frame, c1, c2, (0, 0, 255), 2)

    info = [("Intruders", intruders)]
    if type(cenTracker) == DirectionCentroidTracker:
        info = [
            ("Total", cenTracker.totalCount),
            ("Entered", cenTracker.bottomRightCount),
            ("Exited", cenTracker.topLeftCount),
        ]

    for (i, (k, v)) in enumerate(info):
        text = f"{k}, {v}"
        cv2.putText(
            frame,
            text,
            (30, 200 - ((i * 30) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    if OUTPUT:
        out.write(frame)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.namedWindow("Window")  # Create a named window
    cv2.moveWindow("Window", 40, 30)  # Move it to (40,30)
    cv2.imshow("Window", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
