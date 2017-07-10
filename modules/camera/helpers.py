import os
import itertools
import cv2 as cv
import numpy as np
from numpy import linspace

CYAN = (204,176,74)

def abspath(relpath, ensure_exists=False):
    """Return the absolute path relative to this file"""
    root = os.path.dirname(__file__)
    abspath = os.path.join(root,relpath)
    if ensure_exists and not os.path.exists(abspath):
        raise IOError("File does not exist: " + abspath)
    return abspath



def parse_coordinates(pts, dimensions=2):
    """
    Return a coordinates array from configuration settings
    @pts. An array of dict containing x,y,z keys
    """
    nrows = len(pts)
    coordinates = np.zeros((nrows,dimensions), np.int32)
    for i,row in enumerate(pts):
        if dimensions==2:
            coordinates[i,:] = (row["x"],row["y"])
        elif dimensions==3:
            coordinates[i,:] = (row["x"],row["y"],row["z"])
        else:
            raise ValueError("Number of dimensions must be 2 or 3")
    return coordinates



def draw_grid(image, corners, n=2, color=CYAN):
    """
    Draw an n x n grid inside the area bounded by corners
    """
    # Calculate points along edges
    (tl, tr, br, bl) = corners
    top = zip(linspace(tl[0],tr[0],n), linspace(tl[1],tr[1],n))
    bottom = zip(linspace(bl[0],br[0],n), linspace(bl[1],br[1],n))
    left = zip(linspace(tl[0],bl[0],n), linspace(tl[1],bl[1],n))
    right = zip(linspace(tr[0],br[0],n), linspace(tr[1],br[1],n))

    # Calculate grid lines
    horz_lines = zip(top,bottom)
    vert_lines = zip(left,right)

    # Draw grid lines
    for line in itertools.chain(horz_lines,vert_lines):
        p1 = (int(line[0][0]), int(line[0][1]))
        p2 = (int(line[1][0]), int(line[1][1]))
        cv.line(image, p1, p2, color, 1, lineType=cv.LINE_AA)









