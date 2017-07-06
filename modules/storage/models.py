import boto3
from pynamodb.models import Model
from pynamodb.attributes import *


resource = boto3.resource('s3')
bucket = resource.Bucket('smart-traffic-monitoring')


class VideoModel(Model):
    """
    A Video that is stored in S3
    """
    class Meta:
        table_name = "intersection-videos"

    id = NumberAttribute(hash_key=True)
    camera = UnicodeAttribute()
    timestamp = UTCDateTimeAttribute()
    filename = UnicodeAttribute()

    @classmethod
    def create(cls,camera,datetime,filepath):
        """Create a new object using a local filepath"""
        uid = int(time.mktime(datetime))
        filename = "{0}-{1}".format(camera,timestamp)
        bucket.upload_file(filename, Key=filepath)
        return cls(uid,camera,timestamp,filename)



class FrameModel(Model):
    """
    A single frame of a video with automatically labelled cars
    """
    class Meta:
        table_name = "intersection-frames"

    id = NumberAttribute(hash_key=True)
    video_id = UnicodeAttribute()
    frame_number = UnicodeAttribute()
    objects_train = ListAttribute()
    objects_test = ListAttribute()





