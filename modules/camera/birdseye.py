import json
import yaml
import cv2 as cv
import numpy as np
from .helpers import abspath, draw_grid, parse_coordinates



class BirdseyeView():
    """
    A helper class for converting images and image coordinates
    to global cordinates
    """

    def __init__(self, camera_name):
        """Load the configuration file"""
        filepath = "locations/{0}/config.yaml".format(camera_name)
        filepath = abspath(filepath)
        with open(filepath, "r") as stream:
            self.config = yaml.load(stream)


    def get_original_image(self):
        """Return the original image"""
        return cv.imread(abspath(self.config["local"]["image"]))


    def get_transformed_image(self):
        """
        Return the birdseye image as a numpy array
        """
        scale_x = self.config["local"]["scale_x"]
        scale_y = self.config["local"]["scale_y"]
        local_image = cv.imread(abspath(self.config["local"]["image"]))
        rect = parse_coordinates(self.config["local"]["points"]).astype(np.float32)

        # Extract the four image corners
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        # compute the perspective transform matrix and then apply it
        M = cv.getPerspectiveTransform(rect, dst)

        # Write the transformation matrix to a JSON file
        data = {"transformation_matrix": M.tolist()}
        filepath = abspath(self.config["transformation"]["file"])
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        global_size = (maxWidth+600, maxHeight+100)
        global_image = cv.warpPerspective(local_image, M, global_size)
        global_image = cv.resize(global_image, (0,0), fx=scale_x, fy=scale_y)
        return global_image


    def get_transformation_matrix(self):
        """Return the transformation matrix"""
        filename = abspath(self.config["transformation"]["file"])
        with open(filename, "r") as f:
            data = json.load(f)
        return np.array(data["transformation_matrix"])


    def transform_to_global(self,pts):
        """
        Map a local point to global coordinates so it can be plotted on the birdseye image
        The top left corner is considered the origin
        @pts. A nx2 matrix, where each row is a point
        """
        scale_x = self.config["local"]["scale_x"]
        scale_y = self.config["local"]["scale_y"]

        if pts.shape[1]!=2:
            raise ValueError("pts must be a matrix of 2 dimensional points")

        # Load the transformation matrix from JSON
        M = self.get_transformation_matrix()

        # The perspective transformation matrix expects the z-dimension to be 1
        pts = np.c_[pts, np.ones(pts.shape[0])]

        transformed = np.matmul(M, pts.T)
        global_pts = transformed[0:2]/transformed[2]
        scaled_pts = np.array([[scale_x], [scale_y]])*global_pts

        return scaled_pts.T

