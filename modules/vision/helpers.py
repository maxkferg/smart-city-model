import os
import math
import matplotlib
import cv2 as cv
import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip
from collections import namedtuple

from .nets import ssd_vgg_300
from .nets import ssd_common
from .preprocessing import ssd_vgg_preprocessing


colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


def abspath(relpath, ensure_exists=False):
    """Return the absolute path relative to this file"""
    root = os.path.dirname(__file__)
    abspath = os.path.join(root,relpath)
    if ensure_exists and not os.path.exists(abspath):
        raise IOError("File does not exist: " + abspath)
    return abspath


def bboxes_as_pixels(img, bboxes):
    """
    Return a list of bounding boxes, in pixels
    Returns boxes in form {x,y,width,height}
    """
    centers = []
    w = img.shape[1] # Width is number of cols
    h = img.shape[0] # Height is number of rows
    for i in range(bboxes.shape[0]):
        box = bboxes[i]
        centers.append({
            'x': w*(box[1]+box[3])/2,
            'y': h*(box[0]+box[2])/2,
            'width': w*(box[3]-box[1]),
            'height': h*(box[2]-box[0])
        })
    return centers


def bboxes_draw_on_img(img, scores, bboxes, colors, thickness=2, show_text=True):
    """
    Drawing bounding boxes on an image, with additional text if wanted...
    """
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[i % len(colors)]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        if show_text:
            s = '%s: %s' % ('Car', scores[i])
            p1 = (p1[0]-5, p1[1])
            cv.putText(img, s, p1[::-1], cv.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
    return img



def bboxes_overlap(bbox, bboxes):
    """Computing overlap score between bboxes1 and bboxes2.
    Note: bboxes1 can be multi-dimensional.
    """
    if bboxes.ndim == 1:
        bboxes = np.expand_dims(bboxes, 0)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes[:, 0], bbox[0])
    int_xmin = np.maximum(bboxes[:, 1], bbox[1])
    int_ymax = np.minimum(bboxes[:, 2], bbox[2])
    int_xmax = np.minimum(bboxes[:, 3], bbox[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    vol2 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    score1 = int_vol / vol1
    score2 = int_vol / vol2
    return np.maximum(score1, score2)



def bboxes_nms_intersection_avg(classes, scores, bboxes, threshold=0.5):
    """Apply non-maximum selection to bounding boxes with score averaging.
    The NMS algorithm works as follows: go over the list of boxes, and for each, see if
    boxes with lower score overlap. If yes, averaging their scores and coordinates, and
    consider it as a valid detection.

    Arguments:
      classes, scores, bboxes: SSD network output.
      threshold: Overlapping threshold between two boxes.
    Return:
      classes, scores, bboxes: Classes, scores and bboxes of objects detected after applying NMS.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    new_bboxes = np.copy(bboxes)
    new_scores = np.copy(scores)
    new_elements = np.ones_like(scores)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            sub_bboxes = bboxes[(i+1):]
            sub_scores = scores[(i+1):]
            overlap = bboxes_overlap(new_bboxes[i], sub_bboxes)
            mask = np.logical_and(overlap > threshold, keep_bboxes[(i+1):])
            while np.sum(mask):
                keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], ~mask)
                # Update boxes...
                tmp_scores = np.reshape(sub_scores[mask], (sub_scores[mask].size, 1))
                new_bboxes[i] = new_bboxes[i] * new_scores[i] + np.sum(sub_bboxes[mask] * tmp_scores, axis=0)
                new_scores[i] += np.sum(sub_scores[mask])
                new_bboxes[i] = new_bboxes[i] / new_scores[i]
                new_elements[i] += np.sum(mask)

                # New overlap with the remaining?
                overlap = bboxes_overlap(new_bboxes[i], sub_bboxes)
                mask = np.logical_and(overlap > threshold, keep_bboxes[(i+1):])

    new_scores = new_scores / new_elements
    idxes = np.where(keep_bboxes)
    return classes[idxes], new_scores[idxes], new_bboxes[idxes]



car = namedtuple('car', ['n_frames', 'bbox', 'speed', 'score', 'idx'])

def update_car_collection(cars, scores, bboxes, overlap_threshold=0.5, smoothing=0.3, n_frames=15):
    """Update a car collection.
    The algorithm works as follows: it first tries to match cars from the collection with
    new bounding boxes. For every match, the car coordinates and speed are updated accordingly.
    If there are remaining boxes at the end of the matching process, every one of them is considered
    as a new car.
    Finally, the algorithm also checks in how many of the last N frames the object has been detected.
    This allows to remove false positives, which usually only appear on a very few frames.

    Arguments:
      cars: List of car objects.
      scores, bboxes: Scores and boxes of newly detected objects.

    Return:
      cars collection updated.
    """
    detected_bboxes = np.zeros(scores.shape, dtype=np.bool)
    new_cars = []
    for i, c in enumerate(cars):
        # Car bbox prediction, using speed.
        cbbox = c.bbox + np.concatenate([c.speed] * 2)
        # Overlap with detected bboxes.
        overlap = bboxes_overlap(cbbox, bboxes)
        mask = np.logical_and(overlap > overlap_threshold, ~detected_bboxes)
        # Some detection overlap with prior.
        if np.sum(mask) > 0:
            detected_bboxes[mask] = True
            sub_scores = np.reshape(scores[mask], (scores[mask].size, 1))
            nbbox = np.sum(bboxes[mask] * sub_scores, axis=0) / np.sum(sub_scores)

            # Update car parameters.
            new_cbbox = smoothing * nbbox + (1 - smoothing) * cbbox
            nspeed = np.sum(np.reshape(new_cbbox - cbbox, (2, 2)), axis=0)
            new_speed = nspeed * smoothing + (1 - smoothing) * c.speed
            new_score = np.sum(sub_scores) / np.sum(mask)
            new_score = smoothing * new_score + (1 - smoothing) * c.score
            new_cars.append(car(n_frames=np.minimum(c.n_frames + 1, n_frames),
                                bbox=new_cbbox,
                                speed=new_speed,
                                score=new_score,
                                idx=c.idx))
        else:
            # Keep the same one, with just a position update.
            if c.n_frames > 1:
                new_cars.append(car(n_frames=c.n_frames-1,
                                    bbox=cbbox,
                                    speed=c.speed,
                                    score=c.score,
                                    idx=c.idx))
    max_idx = max([0] + [c.idx for c in new_cars]) + 1
    # Add remaining boxes.
    for i in range(scores.size):
        if not detected_bboxes[i]:
            new_cars.append(car(n_frames=1,
                                bbox=bboxes[i],
                                speed=np.zeros((2, ), dtype=bboxes.dtype),
                                score=scores[i],
                                idx=max_idx))
            max_idx += 1

    # Sort list of car by size.
    sorted(new_cars, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    return new_cars


def plot_image(img, title='', figsize=(24, 9)):
    return
    f, axes = plt.subplots(1, 1, figsize=figsize)
    f.tight_layout()
    axes.imshow(img)
    axes.set_title(title, fontsize=20)
