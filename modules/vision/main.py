import os
import math
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm
from collections import namedtuple

from .helpers import bboxes_draw_on_img, bboxes_nms_intersection_avg, update_car_collection, abspath, colors_tableau
from .nets import ssd_vgg_300
from .nets import ssd_common
from .preprocessing import ssd_vgg_preprocessing

CHECKPOINT = abspath('checkpoints/ssd_model.ckpt')
TEST_INPUT = abspath('test/input')
TEST_OUTPUT = abspath('test/output')

slim = tf.contrib.slim

# Build up the convolutional network and load the checkpoint.
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, (300, 300), resize=ssd_vgg_preprocessing.Resize.NONE)
image_4d = tf.expand_dims(image_pre, 0)

# Network parameters.
params = ssd_vgg_300.SSDNet.default_params
params = params._replace(num_classes=8)

# SSD network construction.
reuse = True if 'ssd' in locals() else None
ssd = ssd_vgg_300.SSDNet(params)
with slim.arg_scope(ssd.arg_scope(weight_decay=0.0005)):
    predictions, localisations, logits, end_points = ssd.net(image_4d, is_training=False, reuse=reuse)



class ImageNetwork():
    """
    Convenient wrapper around the SSD neural network
    """

    def __init__(self, select_threshold=0.5, nms_threshold=0.5):
        """Set the network parameters"""
        self.cars = []
        self.nms_threshold = nms_threshold
        self.select_threshold = select_threshold


    def __enter__(self):
        """
        Create the session
        Load the Neural Network
        Clean up everything on __exit__
        """
        self.sess = tf.Session()

        # Initialize variables.
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # Restore SSD model.
        saver = tf.train.Saver()
        saver.restore(self.sess, CHECKPOINT)

        # Save back model to clean the checkpoint?
        save_clean = False
        if save_clean:
            ckpt_filename_save = './checkpoints/ssd_model.ckpt'
            saver.save(self.sess, ckpt_filename_save, write_meta_graph=True, write_state=True)
        return self


    def __exit__(self, *args):
        """
        Clean up the session
        """
        self.sess.close()


    def process_image(self, img):
        """
        Process an image through SSD network.

        Arguments:
          sess: A tensorflow session with loaded weights
          img: Numpy array containing an image.
          select_threshold: Classification threshold (i.e. probability threshold for car detection).
        Return:
          rclasses, rscores, rbboxes: Classes, scores and bboxes of objects detected.
        """
        classes, scores, bboxes = self._process_image(img)
        classes_nms, scores_nms, bboxes_nms = bboxes_nms_intersection_avg(classes, scores, bboxes, self.nms_threshold)
        return classes_nms, scores_nms, bboxes_nms


    def _process_image(self, img):
        """
        Process an image through SSD network.
        Does not apply the NMS algorithms to remove conflicting bounding boxes

        Arguments:
          sess: A tensorflow session with loaded weights
          img: Numpy array containing an image.
          select_threshold: Classification threshold (i.e. probability threshold for car detection).
        Return:
          rclasses, rscores, rbboxes: Classes, scores and bboxes of objects detected.
        """
        # Resize image to height 300.
        factor = 300. / float(img.shape[0])
        img = cv.resize(img, (0,0), fx=factor, fy=factor)
        # Run SSD network and get class prediction and localization.
        rpredictions, rlocalisations = self.sess.run([predictions, localisations], feed_dict={img_input: img})

        # Get anchor boxes for this image shape.
        ssd.update_feature_shapes(rpredictions)
        anchors = ssd.anchors(img.shape, dtype=np.float32)

        # Compute classes and bboxes from the net outputs: decode SSD output.
        rclasses, rscores, rbboxes, rlayers, ridxes = ssd_common.ssd_bboxes_select(
                rpredictions, rlocalisations, anchors,
                threshold=self.select_threshold, img_shape=img.shape, num_classes=ssd.params.num_classes, decode=True)

        # Remove other classes than cars.
        idxes = (rclasses == 1)
        rclasses = rclasses[idxes]
        rscores = rscores[idxes]
        rbboxes = rbboxes[idxes]
        # Sort boxes by score.
        rclasses, rscores, rbboxes = ssd_common.bboxes_sort(rclasses, rscores, rbboxes, top_k=400, priority_inside=True, margin=0.0)
        return rclasses, rscores, rbboxes


    def ssd_process_frame(self, img, select_threshold=0.5, nms_threshold=0.2, overlap_threshold=0.4, smoothing=0.25):
        """Process a video frame through SSD network, apply NMS algorithm and draw bounding boxes.

        Arguments:
          img: Numpy array containing an image.
          select_threshold: Classification threshold (i.e. probability threshold for car detection).
          nms_threshold: NMS threshold.
          overlap_threshold: Overlap threshold used for updating cars collection.
          smoothing: Smoothing factor over frames.
        Return:
          image with bounding boxes.
        """
        # Resize image to height 300.
        factor = 300. / float(img.shape[0])
        img = cv.resize(img, (0,0), fx=factor, fy=factor)
        # Run SSD network and get class prediction and localization.
        rpredictions, rlocalisations = self.sess.run([predictions, localisations], feed_dict={img_input: img})

        # Get anchor boxes for this image shape.
        ssd.update_feature_shapes(rpredictions)
        anchors = ssd.anchors(img.shape, dtype=np.float32)

        # Compute classes and bboxes from the net outputs: decode SSD output.
        rclasses, rscores, rbboxes, rlayers, ridxes = ssd_common.ssd_bboxes_select(
                rpredictions, rlocalisations, anchors,
                threshold=select_threshold, img_shape=img.shape, num_classes=ssd.params.num_classes, decode=True)

        # Remove other classes than cars.
        idxes = (rclasses == 1)
        rclasses = rclasses[idxes]
        rscores = rscores[idxes]
        rbboxes = rbboxes[idxes]
        # Sort boxes by score.
        rclasses, rscores, rbboxes = ssd_common.bboxes_sort(rclasses, rscores, rbboxes,
                                                            top_k=400, priority_inside=True, margin=0.0)
        # Apply NMS.
        rclasses, rscores, rbboxes = bboxes_nms_intersection_avg(rclasses, rscores, rbboxes, threshold=nms_threshold)
        # Update cars collection.
        n_frames=15
        self.cars = update_car_collection(self.cars, rscores, rbboxes,
                                                       overlap_threshold, smoothing, n_frames=n_frames)

        # Draw bboxes
        cbboxes = [c.bbox for c in self.cars if c.n_frames > n_frames - 5]
        cindexes = [c.idx for c in self.cars if c.n_frames > n_frames - 5]
        if len(cbboxes):
            cbboxes = np.stack(cbboxes)
            cindexes = np.stack(cindexes)
            bboxes_draw_on_img(img, cindexes, cbboxes, colors_tableau, thickness=2)
        return img







