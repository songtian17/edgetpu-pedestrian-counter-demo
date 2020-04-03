# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

import base64
import threading
from collections import OrderedDict

import cv2
import numpy as np
from scipy.spatial import distance as dist
from shapely.geometry import Point, Polygon
from collections import defaultdict


class CentroidTracker:
    def __init__(self, maxDisappeared=50, notify=False):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.target_rect = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, frame, centroid, rect):
        """Register new centroid by inserting the ObjectID and centroid as a
        key value pair in the objects dictionary of the class.
        
        Args:
            frame (numpy.ndarray): A numpy.ndarray representation of the image
                containing the centroid.
            centroid (numpy.ndarray): A numpy.ndarray object representing a centroid.
            rect (tuple): A tuple representing the bounding box of an object
                (x1, y2, x2, y2).
        """

        # when registering an object we use the next available object
        # ID to store the centroid

        objectId = self.nextObjectID

        # ---------------- Method overwritten in child ----------------
        self.on_register(frame, rect, objectId)
        # ---------------- Method overwritten in child ----------------

        self.objects[objectId] = centroid

        self.disappeared[objectId] = 0
        try:
            self.target_rect[objectId] = rect
        except IndexError:
            self.target_rect[objectId] = rect
            pass

        # uncomment next line to auto increment stored
        self.nextObjectID += 1

    def on_register(self, frame, rect, objectId):
        # does things on register (to be overwritten in child classes)
        pass

    def deregister(self, objectID):
        """Deregister a centroid by deleting its ObjectID from the objects dictionary.
        
        Args:
            objectID (int): Object ID of the centroid in objects dictionary.
        """
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries

        # ---------------- Method overwritten in child ----------------
        self.on_deregister(objectID)
        # ---------------- Method overwritten in child ----------------

        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.target_rect[objectID]
        print("deregister {}".format(objectID))

    def on_deregister(self, objectID):
        pass

    def update(self, frame, rects):
        """Method called every step to check and register and deregister centroids.
        
        Args:
            frame (numpy.ndarray): A numpy.ndarray representation of a video frame.
            rects (list): A list of tuples representing bounding boxes of
                objects detected in the frame.
        """

        # ---------------- Method overwritten in child ----------------
        self.on_update()
        # ---------------- Method overwritten in child ----------------

        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:

            # loop over any existing tracked objects and mark them
            # as disappeared
            disappeared_copy = self.disappeared.copy()
            for objectID in disappeared_copy.keys():
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                disappear_duration = self.disappeared[objectID]
                if disappear_duration > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        boundingBoxes = OrderedDict()
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            boundingBoxes[i] = (startX, startY, endX, endY)
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        # print(len(self.objects))
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(frame, inputCentroids[i], boundingBoxes[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())

            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # print("d {0} ".format(np.array(objectCentroids)))
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # print("{0} and {1}".format(rows,cols))
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # if distance (D[row][col]) > 50:
                if D[row][col] > 200:
                    usedRows.add(row)
                    usedCols.add(col)
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]

                self.objects[objectID] = inputCentroids[col]
                try:
                    self.target_rect[objectID] = boundingBoxes[col]
                # because of threading we have to ignore indexerror
                except IndexError:
                    # self.target_rect[objectID] = ['UNKNOWN',boundingBoxes[col]]
                    pass

                # self.objects[objectID] = [[inputCentroids[col]],self.lpr_ocr.plate]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        # threading.Thread(target=self.deregister(objectID)).start()
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(frame, inputCentroids[col], boundingBoxes[col])

        # return the set of trackable objects
        return self.objects

    def on_update(self):
        pass


class DirectionCentroidTracker(CentroidTracker):
    def __init__(self, point1, point2, maxDisappeared=30):
        super().__init__(maxDisappeared=maxDisappeared)
        self.x1, self.y1 = point1
        self.x2, self.y2 = point2
        self.topLeftSide = []
        self.bottomRightSide = []
        self.topLeftCount = 0
        self.bottomRightCount = 0
        self.totalCount = 0

    def isBottomOrRightSide(self, centroidX, centroidY):
        return (
            (self.x2 - self.x1) * (centroidY - self.y1)
            - (centroidX - self.x1) * (self.y2 - self.y1)
        ) > 0

    def on_register(self, frame, rect, objectID):
        x1, y1, x2, y2 = rect
        centroidX, centroidY = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if self.isBottomOrRightSide(centroidX, centroidY):
            self.bottomRightSide.append(objectID)
        else:
            self.topLeftSide.append(objectID)

    def on_deregister(self, objectID):
        centroidX, centroidY = self.objects[objectID]

        isThatSide = self.isBottomOrRightSide(centroidX, centroidY)

        if isThatSide and objectID in self.topLeftSide:
            self.topLeftSide.remove(objectID)
            self.bottomRightCount += 1
        elif not isThatSide and objectID in self.bottomRightSide:
            self.bottomRightSide.remove(objectID)
            self.topLeftCount += 1
        else:
            # centroid stays on same side
            pass
        self.totalCount = self.bottomRightCount - self.topLeftCount
        pass
