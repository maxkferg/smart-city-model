import os
import time
import boto3
import requests
from pynamodb.models import Model
from pynamodb.attributes import *

temp = "test/temp"
resource = boto3.resource('s3')
bucket = resource.Bucket('smart-traffic-monitoring')



class VideoModel(Model):
    """
    A Video that is stored in S3
    """
    class Meta:
        table_name = "intersection-videos"

    id = UnicodeAttribute(hash_key=True)
    camera = UnicodeAttribute()
    timestamp = UTCDateTimeAttribute()
    filename = UnicodeAttribute()

    @classmethod
    def create(cls,camera,created,videopath):
        """Create a new object using a local filepath"""
        timestamp = str(int(created.timestamp()))
        filename = "{0}-{1}.mpeg".format(camera,timestamp)
        bucket.upload_file(videopath, Key=filename)
        return cls(id=timestamp,camera=camera,timestamp=created,filename=filename)

    def download(self):
        """Download the video file"""
        path = os.path.join(temp,self.filename)
        bucket.download_file(self.filename, path)
        print("Downloaded",self.filename,"to",path)
        return path



class FrameModel(Model):
    """
    A single frame of a video with automatically labelled cars
    """
    class Meta:
        table_name = "intersection-frames"

    id = UnicodeAttribute(hash_key=True)
    camera = UnicodeAttribute()
    video_id = UnicodeAttribute()
    frame_number = NumberAttribute()
    objects_train = ListAttribute()
    objects_test = ListAttribute()
    n_objects_train = NumberAttribute()
    n_objects_test = NumberAttribute()

    def save(self,*args,**kwargs):
        """Update the metadata on save"""
        self.n_objects_train = len(self.objects_train)
        self.n_objects_test = len(self.objects_test)
        super().save(*args,**kwargs)





