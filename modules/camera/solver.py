"""
Load a image of the road from Auburn
Calculate the mapping from the image to global coordinates
Plot a grid on the original camera image
"""

import os
import math
import cv2 as cv
import numpy as np
from cv2 import LINE_AA

ALPHA = 0.3
BLUE = (255,0,0)
RED = (0,0,255)
GREEN = (0,255,0)
CYAN = (204,176,74)
RADIUS = 5


def abspath(relpath, ensure_exists=False):
    """Return the absolute path relative to this file"""
    root = os.path.dirname(__file__)
    abspath = os.path.join(root,relpath)
    if ensure_exists and not os.path.exists(abspath):
        raise IOError("File does not exist: "+abspath)
    return abspath


images = []
inputfile = abspath("locations/auburn/images/intersection.png",True)
outfile = abspath("locations/auburn/images/intersection-lanes.png")
undistfile = abspath("locations/auburn/images/intersection-undist.png")
birdseye_file = abspath("locations/auburn/images/intersection-birdseye.png")

output = cv.imread(inputfile)
overlay = output.copy()
size = output.shape
total = 0


lane1a = np.array([
    (183,359),
    (0,  152),
    (58, 144),
    (298,338),
])

lane1b = np.array([
    (190,368),
    (307,348),
    (700,692),
    (700,700),
    (470,700)
])

lane2 = np.array([
    (164,364),
    (57, 383),
    (0,  269),
    (0,  161)
])

lane3 = np.array([
    (50,384),
    (0, 396),
    (0, 282)
])

local_points = np.array([
    (183, 359), # 0: Origin
    (0,   152), # 1: Distane lane corner
    (57,  384), # 2: X axis point
    (238, 428), # 3: y axis point
    (558, 359), # 4: y axis point
    #(176, 125), # 5: Driveway
    (95,  458)  # 6: Left side of crossing
])


local_points = local_points

global_points = np.array([
    (0,           0,    0), # 0: Origin
    (0,      -1500,     0), # 1:
    ( 300,       0,     0), # 2: X axis point
    (0,        200,     0), # 3: Y axis point
    (-700,    200,      0), # 4: Y axis point
    #(-7,       -15,    0), # 5: Driveway
    (300,     200,     0)  # 6:
])

local_points = local_points.astype(np.float32)
global_points = global_points.astype(np.float32)



def run():
    cv.fillConvexPoly(output, lane1a, BLUE)
    cv.fillConvexPoly(output, lane1b, BLUE)
    cv.fillConvexPoly(output, lane2, RED)
    cv.fillConvexPoly(output, lane3, GREEN)
    cv.imwrite(outfile,output)

    # make everything transparent
    cv.addWeighted(overlay, 1-ALPHA, output, ALPHA, 0, output)


    # Draw all the reference points on the image
    for row in range(len(local_points)):
        center = tuple(local_points[row,:].astype(int))
        textpos = (center[0]+5,center[1])
        radius = 3
        cv.circle(output, center, radius, RED, lineType=LINE_AA)
        cv.putText(output,str(row), textpos, cv.FONT_HERSHEY_SIMPLEX, 0.4, RED, lineType=LINE_AA)

    # Draw axis labels
    cv.putText(output, "X", (20,396), cv.FONT_HERSHEY_SIMPLEX, 0.4, RED, lineType=LINE_AA)
    cv.putText(output, "Y", (258,428), cv.FONT_HERSHEY_SIMPLEX, 0.4, RED, lineType=LINE_AA)



    # Camera internals
    focal_length = size[1]
    center = (size[1]/4, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                            )

    print("Camera Matrix :\n {0}".format(camera_matrix))
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv.solvePnP(global_points, local_points, camera_matrix, dist_coeffs)

    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    (origin, jacobian) = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    (x_axis, jacobian) = cv.projectPoints(np.array([(1.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    (y_axis, jacobian) = cv.projectPoints(np.array([(0.0, 1.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    (z_axis, jacobian) = cv.projectPoints(np.array([(0.0, 0.0, 1.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    origin = tuple(origin.astype(np.int).tolist()[0][0])
    x_axis = tuple(x_axis.astype(np.int).tolist()[0][0])
    y_axis = tuple(y_axis.astype(np.int).tolist()[0][0])
    z_axis = tuple(z_axis.astype(np.int).tolist()[0][0])

    cv.line(output, origin, x_axis, GREEN, 2, lineType=LINE_AA)
    cv.line(output, origin, y_axis, GREEN, 2, lineType=LINE_AA)
    #cv.line(output, origin, z_axis, GREEN, 2, lineType=LINE_AA)


    points = []
    xmin = -15
    xmax = 15
    ymin = -15
    ymax = 15

    # Create grid lines
    for x in range(xmin,xmax):
        points.append((x,ymin,0)) # right
        points.append((x,ymax,0)) # left

    for y in range(xmin,xmax):
        points.append((xmin,y,0)) # right
        points.append((xmax,y,0)) # left

    # Move grid lines into local coords
    points = np.array(points).astype(np.float32)
    points, jacobian = cv.projectPoints(points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # Draw gridlines
    for i in range(0, points.shape[0]-1, 2):
        p1 = tuple(points[i,:,:][0])
        p2 = tuple(points[i+1,:,:][0])
        cv.line(output, p1, p2, CYAN, 1, lineType=LINE_AA)

    # Map the global points back to local
    gpoints, jacobian = cv.projectPoints(global_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # Draw global points
    for i in range(0, len(gpoints)):
        radius = 2
        center = tuple(gpoints[i,:,:][0])
        cv.circle(output, center, radius, BLUE, lineType=LINE_AA)


    # Create y grid lines
    cv.imwrite(outfile,output)

    # Undistort the original image
    # This doesn't actually do anything because we set distortion to zero
    undistorted = cv.undistort(output, camera_matrix, dist_coeffs)

    # Get the camera matrix
    # Requires that the points are in the same plane
    #matrix, mask = cv.findHomography(local_points,global_points, method=cv.RANSAC)
    #matrix = matrix/100
    #print(rotation_vector.T)
    #matrix,jac = cv.Rodrigues(rotation_vector.T)
    #print("Transformation Matrix\n",matrix)
    #birdseye = cv.warpPerspective(output, matrix, (5000,8000))

    padded = cv.copyMakeBorder(output, 100, 0, 500, 500, cv.BORDER_CONSTANT)

    local_points = 

    cv.imwrite(birdseye_file, padded)




