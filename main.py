import cv2
from moviepy.editor import VideoFileClip
from modules.ssd.main import ImageNetwork

"""
def pipeline_yolo(img):
    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_yolo(img, img_lane_augmented, lane_info)
    return output

def process_image(img):
    output = vehicle_only_yolo(img)
    return output


def process_video():
    video_output = 'examples/project_YOLO.mp4'
    clip1 = VideoFileClip("examples/project_video.mp4").subclip(30,32)
    clip = clip1.fl_image(pipeline_yolo)
    clip.write_videofile(video_output, audio=False)
"""


def process_video(video_input, video_output):
    """Process a video using the SSD network"""
    with ImageNetwork() as ssd: 
        clip = VideoFileClip(video_input).subclip(26,31)
        clip = clip.fl_image(ssd.ssd_process_frame)
        clip.write_videofile(video_output, audio=False)


if __name__ == "__main__":
    # SSD Pipeline
    video_input = 'test/project_video.mp4'
    video_output = 'test/labelled_ssd.mp4'
    process_video(video_input, video_output)






