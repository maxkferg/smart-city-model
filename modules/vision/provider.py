import numpy as np
import tensorflow as tf
from modules.storage.models import VideoModel, FrameModel
from moviepy.editor import VideoFileClip



class DatabaseProvider():

    def __init__(self, nbuffer=1000):
        self.buffer = []
        self.nbuffer = nbuffer
        self.fillBuffer('1500327953')

    def get(self,keys):
        if len(self.buffer):
            item = self.buffer.pop()
            return [item[key] for key in keys]
        return None

    def fillBuffer(self,video_id):
        """Download a video and fill all the frames in the buffer"""
        metadata = VideoModel.get(video_id)
        framedata = list(FrameModel.query(hash_key=video_id))
        # Download the file
        localfile = metadata.download()
        clip = VideoFileClip(localfile)
        # Iterate over ever frame and add it to the buffer
        for i,frame in enumerate(clip.iter_frames()):
            objects = framedata[i].objects_train
            bboxes = [(i['x'],i['y'],i['width'],i['height']) for i in objects]
            self.buffer.append({
                'image': tf.constant(frame),
                'shape': frame.shape,
                'object/label': [i['label'] for i in objects],
                'object/bbox': np.array(bboxes,dtype=np.float32)
            })

