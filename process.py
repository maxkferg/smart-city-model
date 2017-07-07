import pafy
import requests
import numpy
import cv2 as cv
from moviepy.editor import VideoFileClip
from modules.storage.models import VideoModel,FrameModel
from modules.yolo.yolo_pipeline import *



class ImageProcessor:
    """Process a single video frame"""

    def __init__(self,video):
        self.video = video
        self.frame_num = 0

    def process(self,image):
        try:
            self.save_annotated_image(image)
            network, frame = vehicle_only_yolo(image)
            results = network.result_list
        except Exception as e:
            print("Frame processing failed:",e)
            frame = image
            results = []
        self.save_result_to_db(results)
        self.frame_num += 1
        return frame

    def save_result_to_db(self,results):
        """Save the frame result to the database"""
        for result in results:
            for i,item in enumerate(result):
                if isinstance(item, (np.float32,np.float64)):
                    result[i] = float(item)
        obj = FrameModel(
            id="{0}-{1}".format(self.video.id, self.frame_num),
            camera=self.video.camera,
            video_id=self.video.id,
            frame_number=self.frame_num,
            objects_train=results,
            objects_test=[]
        )
        obj.save()
        return obj

    def save_annotated_image(self,image):
        """Save the image of the road"""
        image = image[:,:,::-1]
        filepath = "test/data/road-{0}-{1}.png".format(self.video.timestamp, self.frame_num)
        cv.imwrite(filepath,image)



class VideoProcessor:
    """Process videos"""
    output = "test/data/{0}"

    def process_video(self,video):
        agent = ImageProcessor(video)
        localfile = video.download()
        clip = VideoFileClip(localfile)
        clip = clip.fl_image(agent.process)
        # Save the video for debugging
        outfile = video.filename.replace("mpeg","mp4")
        output = self.output.format(outfile)
        clip.write_videofile(output, audio=False)
        print("Saved video to: ",output)



if __name__ == "__main__":
    processor = VideoProcessor()
    for item in VideoModel.scan(camera__eq='auburn'):
        print("Processing",item)
        processor.process_video(item)
