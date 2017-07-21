import pafy
import requests
import cv2 as cv
import numpy as np
from moviepy.editor import VideoFileClip
from modules.storage.models import VideoModel,FrameModel
from modules.ssd.main import ImageNetwork as SSD
from modules.ssd.helpers import bboxes_as_pixels, bboxes_draw_on_img, colors_tableau
from modules.camera.birdseye import BirdseyeView


BLUE = (255,0,0)
birdseye = BirdseyeView("auburn")
birdseye_image = birdseye.get_transformed_image()



class ImageProcessor:
    """Process a single video frame"""

    def __init__(self,network,metadata):
        self.metadata = metadata
        self.frame_num = 0
        self.network = network
        self.birdseye = birdseye


    def process(self,image):
        return self.process_perspective(image)


    def process_birdseye(self,image):
        """Process an image in birdseye"""
        try:
            classes, scores, bboxes = self.network.process_image(image)
            pts = bboxes_as_pixels(image, bboxes)[:,:2]
            global_pts = birdseye.transform_to_global(pts)
            for row in range(global_pts.shape[0]):
                x,y = global_pts[row,:]
                print(x,y)
                frame = cv.circle(birdseye_image, (int(x), int(y)), 10, BLUE, -1)
        except Exception as e:
            print("Frame processing failed:",e)
            frame = image
            results = []
        self.frame_num += 1
        return frame


    def process_perspective(self,image):
        """Process an image in perspective"""
        try:
            classes, scores, bboxes = self.network.process_image(image)
            results = bboxes_as_pixels(image, bboxes)
            frame = bboxes_draw_on_img(image, scores, bboxes, colors_tableau, thickness=2, show_text=True)
            self.save_result_to_db(classes, scores, results)
        except Exception as e:
            print("Frame processing failed:",e)
            frame = image
            results = []
        self.frame_num += 1
        return frame


    def save_result_to_db(self, classes, scores, bboxes):
        """Save the frame result to the database"""
        box_objects = []
        for clss,score,box in zip(classes,scores,bboxes):
            box_objects.append({
                'label': 'car',
                'x': float(box['x']),
                'y': float(box['y']),
                'width': float(box['width']),
                'height': float(box['height']),
                'confidence': float(score)
            })
        obj = FrameModel(
            video_id=self.metadata.id,
            frame_number=self.frame_num,
            camera=self.metadata.camera,
            objects_train=[],
            objects_test=box_objects
        )
        obj.save()
        return obj


    def save_annotated_image(self,image):
        """Save the image of the road"""
        image = image[:,:,::-1]
        filepath = "test/data/road-{0}-{1}.png".format(self.metadata.timestamp, self.frame_num)
        cv.imwrite(filepath,image)



class VideoProcessor:
    """
    Process videos

    Currently downloads a video and labels it
    The labelled video is saved to a local file
    """
    output = "test/data/{0}"

    def __init__(self,network):
        self.network = network

    def process_video(self,metadata):
        """
        Process a video file
        @metadata. A Dynamodb video object
        """
        agent = ImageProcessor(self.network, metadata)
        localfile = metadata.download()
        print(localfile)
        clip = VideoFileClip(localfile)
        clip = clip.fl_image(agent.process)
        # Save the video for debugging
        output = self.output.format(metadata.filename)
        clip.write_videofile(output, audio=False)
        print("Saved video to: ",output)



if __name__ == "__main__":
    with SSD(select_threshold=0.65, nms_threshold=0.6) as ssd:
        processor = VideoProcessor(ssd)
        for item in VideoModel.scan(camera__eq='alabama'):
            print("Processing",item)
            processor.process_video(item)




