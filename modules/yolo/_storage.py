#!/usr/bin/env python

# Copyright 2016 Google Inc.
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

"""Demonstrates how to connect to Cloud Bigtable and run some basic operations.

Prerequisites:

- Create a Cloud Bigtable cluster.
  https://cloud.google.com/bigtable/docs/creating-cluster
- Set your Google Application Default Credentials.
  https://developers.google.com/identity/protocols/application-default-credentials
"""
import struct
from google.cloud import bigtable


def get_int(row, column_family_id, column_id):
    """Return a float from a cell"""
    column_id = column_id.encode("utf-8")
    bytecode = row.cells[column_family_id][column_id][0].value
    return struct.unpack('>i', bytecode)[0]

def store_int(row, column_family_id, column_id, number):
    """Store a float in a cell"""
    column_id = column_id.encode("utf-8")
    return row.set_cell(column_family_id, column_id, struct.pack('>i', number))


def get_float(row, column_family_id, column_id):
    """Return a float from a cell"""
    column_id = column_id.encode("utf-8")
    bytecode = row.cells[column_family_id][column_id][0].value
    return struct.unpack('>d', bytecode)[0]

def store_float(row, column_family_id, column_id, number):
    """Store a float in a cell"""
    column_id = column_id.encode("utf-8")
    return row.set_cell(column_family_id, column_id, struct.pack('>d', number))



class VideoFrame():
    """
    A frame of data
    """
    def __init__(self,video_num,frame_num,x,y,width,height):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.video_num = video_num # Video number
        self.frame_num = frame_num # Frame number

    def __str__(self):
        return "<VideoFrame v:{0} f:{1}>".format(self.video_num, self.frame_num)



class CloudStorage():
    """
    Store objects in Google BigTable
    """
    column_family_id = "cf1"

    def __init__(self, project_id, instance_id, table_id):
        """
        Create the table for frame metadata
        """
        self.instance_id = instance_id
        self.project_id = project_id
        self.table_id = table_id
        self.client = bigtable.Client(project=project_id,admin=True)
        self.instance = self.client.instance(instance_id)
        self.table = self.instance.table(table_id)
        # Check for table existance
        # self.initialize()


    def to_list(self):
        """
        Serialize as a list
        """
        return [
            self.x,
            self.y,
            self.width,
            self.height
        ]


    def initialize(self):
        """
        Create the table for frame metadata
        """
        # We need an admin client to create a table
        client = bigtable.Client(project=self.project_id, admin=True)
        instance = client.instance(self.instance_id)

        # Create table
        print('Creating the {} table.'.format(table_id))
        table = instance.table(self.table_id)
        table.create()

        # Create column family
        self.column_family_id = 'cf1'
        cf1 = table.column_family(self.column_family_id)
        cf1.create()


    def store_frame(self, frame):
        """
        Store a frame object in the database
        """
        row_key = 'frame-{0}-{1}'.format(frame.video_num, frame.frame_num)
        row = self.table.row(row_key)
        store_float(row, self.column_family_id, "x", frame.x)
        store_float(row, self.column_family_id, "y", frame.y)
        store_float(row, self.column_family_id, "width", frame.width)
        store_float(row, self.column_family_id, "height", frame.height)
        store_int(row, self.column_family_id, "frame-num", frame.frame_num)
        store_int(row, self.column_family_id, "video-num", frame.video_num)
        row.commit()
        print('Saved: ',row_key)


    def get_frame(self, video_num, frame_num):
        """
        Get a frame from the database
        """
        row_key = 'frame-{0}-{1}'.format(video_num, frame_num).encode("utf-8")
        row = self.table.read_row(row_key)
        return self._make_videoframe(row)


    def iter_frames(self):
        """
        Return an iterator over frames
        """
        partial_rows = self.table.read_rows()
        def iterator():
            while True:
                partial_rows.consume_next()
                if not len(partial_rows._rows):
                    return
                for key,row in partial_rows._rows.items():
                    frame = self._make_videoframe(row)
                    print(frame)
                    yield frame
                partial_rows._rows = {}
        return iterator()


    def _make_videoframe(self,row):
        """Make a VideoFrame from a BigTable Row"""
        data = {}
        cells = row.cells[self.column_family_id]
        data["x"] = get_float(row, self.column_family_id, "x")
        data["y"] = get_float(row, self.column_family_id, "y")
        data["width"] = get_float(row, self.column_family_id, "width")
        data["height"] = get_float(row, self.column_family_id, "height")
        data["frame_num"] = get_int(row, self.column_family_id, "frame-num")
        data["video_num"] = get_int(row, self.column_family_id, "video-num")
        return VideoFrame(**data)




table_id = "video-frames"
instance_id = "intersection-data"
project_id = "api-project-829690318459"
db = CloudStorage(project_id, instance_id, table_id)

