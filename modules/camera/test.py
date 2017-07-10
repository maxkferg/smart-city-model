import os
import cv2 as cv
import numpy as np
from .birdseye import BirdseyeView
from .helpers import abspath, draw_grid, parse_coordinates

BLUE = (150,50,50)
GREEN = (0,200,0)
CAMERA = "auburn"

def run():
    view = BirdseyeView(CAMERA)
    local_image = view.get_original_image()
    local_points = parse_coordinates(view.config["local"]["points"])
    local_markers = parse_coordinates(view.config["local"]["markers"])

    # Output image files
    debug_output = abspath(view.config["local"]["debug"])
    birdseye_output = abspath(view.config["global"]["image"])

    # Draw a polygon grid for debugging
    points = local_points.reshape((-1,1,2))
    debug_image = np.copy(local_image)
    debug_image = cv.polylines(debug_image, points, True, GREEN, thickness=10, lineType=cv.LINE_AA)
    draw_grid(debug_image, local_points, 20)

    # Draw the markers on the camera image
    for marker in view.config["local"]["markers"]:
        position = (marker["x"],marker["y"])
        cv.circle(debug_image, position, 3, BLUE, thickness=10, lineType=cv.LINE_AA)
    print("Saving annotated frame to: ", os.path.relpath(debug_output))
    cv.imwrite(debug_output, debug_image)

    # Get the transformed birdseye image
    global_image = view.get_transformed_image()

    # Get the transformed markers
    global_markers = view.transform_to_global(local_markers)

    # Plot the markers in the global image
    for row in range(global_markers.shape[0]):
        position = (int(global_markers[row,0]), int(global_markers[row,1]))
        cv.circle(global_image, position, 3, BLUE, thickness=10, lineType=cv.LINE_AA)

    print("Saving birdseye frame to: ", os.path.relpath(birdseye_output))
    cv.imwrite(birdseye_output, global_image)


