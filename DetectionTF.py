# import the necessary packages
import os
from pathlib import Path

import cv2
import numpy as np
from edgetpu.detection.engine import DetectionCandidate, DetectionEngine
from PIL import Image, JpegImagePlugin

# from tflite_runtime.interpreter import Interpreter, load_delegate


class Detection:
    def __init__(self, model_dir: str):
        PATH_TO_MODEL = (
            Path(Path.cwd() / model_dir, "detect_edgetpu.tflite").absolute().as_posix()
        )
        # PATH_TO_MODEL = Path(Path.cwd() / 'models' / model_dir, 'detect_edgetpu.tflite').absolute().as_posix()
        self.engine = DetectionEngine(PATH_TO_MODEL)

    def detect_objects(self, image, threshold):
        """Detect objects in image using EdgeTPU Detection Engine

        For reference: https://coral.ai/docs/reference/edgetpu.detection.engine/.
        
        Args:
            image (obj): A numpy.ndarray or PIL.Image representation of the
                image for object detection.
        
        Returns:
            list: List of edgetpu.detection.engine.DetectionCanditate objects
                representing objects detected in given image.
        """

        # if type(image) == np.ndarray:
        #     img = Image.fromarray(image)
        # elif type(image) == JpegImagePlugin.JpegImageFile:
        #     img = image
        detection_res = self.engine.detect_with_image(
            image,
            relative_coord=False,
            threshold=threshold,
            keep_aspect_ratio=False,
            # resample=Image.BICUBIC,
            top_k=3,
        )

        return detection_res


def check_interest_area(rect, interest_area):
    """Check if rectangle is within interest area.
    
    Args:
        rect (tuple): A tuple representing the rectangle bounding box
            (x1, y1, x2, y2).
        interest_area (dict): A dictionary containing values about interest area.
    
    Returns:
        boolean: True if rectangle is within interest_area. False otherwise.
    """
    (x1, y1, x2, y2) = rect
    if interest_area["isActive"]:
        minx = interest_area["minx"]
        miny = interest_area["miny"]
        maxx = interest_area["maxx"]
        maxy = interest_area["maxy"]

        w = x2 - x1
        h = y2 - y1
        center_x = x1 + w / 2
        center_y = y1 + h / 2
        if center_x < minx or center_x > maxx or center_y < miny or center_y > maxy:
            return False
    return True


def filter_detection_objects(obj_list, interest_area):
    """Filter through a list of rectangles and return rectangles that are
    within interest_area.
    
    Args:
        obj_list (list): A list of edgetpu.detection.engine.DetectionCandidate objects.
        interest_area (dict): A dictionary containing values about interest area.

    Returns:
        list: A list of edgetpu.detection.engine.DetectionCandate objects with
            bounding boxes within the interest area.
    """
    ret_list = []
    for obj in obj_list:
        box = tuple(obj.bounding_box.flatten())
        if check_interest_area(box, interest_area):
            ret_list.append(obj)

    return ret_list


def draw_detection_box(image, rect):
    (x1, y1, x2, y2) = rect
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def draw_detection_boxes(image, rects):
    for rect in rects:
        image = draw_detection_box(image, rect)
    return image
