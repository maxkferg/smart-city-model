# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts KITTI data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'image_2'. Similarly, bounding box annotations are supposed to be
stored in the 'label_2'

This TensorFlow script converts the training and validation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing PNG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'PNG'

    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import os.path
import sys
import random
import numpy as np
import tensorflow as tf

from modules.storage.models import VideoModel, FrameModel
from moviepy.editor import VideoFileClip
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature


def convert_to_example(image_data, shape, labels, labels_text, truncated, occluded,
                        alpha, bboxes, dimensions, locations, rotation_y):
    """Build an Example proto for an image example.

    Args:
      image_data: string, PNG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    # Transpose bboxes, dimensions and locations.
    bboxes = list(map(list, zip(*bboxes)))
    dimensions = list(map(list, zip(*dimensions)))
    locations = list(map(list, zip(*locations)))
    # Iterators.
    it_bboxes = iter(bboxes)
    it_dims = iter(dimensions)
    its_locs = iter(locations)

    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),    # TODO: FIX THIS
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data),
            'object/label': int64_feature(labels),
            'object/label_text': bytes_feature(labels_text),
            'object/truncated': float_feature(truncated),
            'object/occluded': int64_feature(occluded),
            'object/alpha': float_feature(alpha),
            'object/bbox/xmin': float_feature(next(it_bboxes, [])),
            'object/bbox/ymin': float_feature(next(it_bboxes, [])),
            'object/bbox/xmax': float_feature(next(it_bboxes, [])),
            'object/bbox/ymax': float_feature(next(it_bboxes, [])),
            'object/dimensions/height': float_feature(next(it_dims, [])),
            'object/dimensions/width': float_feature(next(it_dims, [])),
            'object/dimensions/length': float_feature(next(it_dims, [])),
            'object/location/x': float_feature(next(its_locs, [])),
            'object/location/y': float_feature(next(its_locs, [])),
            'object/location/z': float_feature(next(its_locs, [])),
            'object/rotation_y': float_feature(rotation_y),
            }))
    return example



def _get_output_filename(output_dir, name):
    return '%s/%s.tfrecord' % (output_dir, name)



def image_to_bytes(image_data, sess):
    """Return the image as png encoded bytes"""
    inputs = tf.placeholder(dtype=tf.uint8, shape=image_data.shape)
    encoded_png = tf.image.encode_png(inputs)
    return sess.run(encoded_png, feed_dict={inputs: image_data})



def normalize_bounding_boxes(bboxes, image_shape):
    """Shift the bboxes into the [0,1.0] range"""
    h = image_shape[0]
    w = image_shape[1]
    output = []
    for bbox in bboxes:
        output.append([
            (bbox['x']) / w,
            (bbox['y']) / h,
            (bbox['x']+bbox['width']) / w,
            (bbox['y']+bbox['height']) / h
        ])
    return output



def run(dataset_dir, output_dir, name='kitti_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    tf_filename = _get_output_filename(output_dir, name)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')

    # quick hack
    video_id = dataset_dir

    with tf.Session() as sess:
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            metadata = VideoModel.get(video_id)
            framedata = list(FrameModel.query(hash_key=video_id))
            # Download the file
            localfile = metadata.download()
            clip = VideoFileClip(localfile)
            # Iterate over ever frame and add it to the buffer
            for i,image_data in enumerate(clip.iter_frames()):
                objects = framedata[i].objects_train
                bboxes = normalize_bounding_boxes(objects, image_data.shape)

                # Convert the record into a standard format
                example = convert_to_example(
                    image_data=image_to_bytes(image_data, sess),
                    shape=list(image_data.shape),
                    labels=[1 for _ in objects],
                    labels_text=[],#[o['label'] for o in objects],
                    truncated=[],
                    occluded=[],
                    alpha=[],
                    bboxes=bboxes,
                    dimensions=[],
                    locations=[],
                    rotation_y=[]
                )
                # Write the record to file
                tfrecord_writer.write(example.SerializeToString())
                print('.', end='', flush=True)

        print('\nFinished converting the Database dataset!')
