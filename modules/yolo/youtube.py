import cv2
import pafy
import requests
from moviepy.editor import VideoFileClip
from yolo_pipeline import *
from lane import *


def pull_videos():
    """
    Return a list of video urls from the server
    """
    url = "https://www.youtube.com/watch?v=cjuskMMYlLA"
    video = pafy.new(url)
    metadata_url = None
    for s in video.streams:
        if s.resolution == "1280x720":
            metadata_url = s.url

    r = requests.get(metadata_url)
    videos = r.text.split()
    videos = [v for v in videos if not v.startswith('#')]
    return videos


def download_video(url):
    """Download video and return filename"""
    path = "videos/temp.mpeg"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in r:
                f.write(chunk)
    return path



class ImageProcessor:

    def __init__(self,video_num):
        self.frame_num = 0
        self.video_num = video_num

    def process(self,image):
        output = vehicle_only_yolo(image, self.video_num, self.frame_num)
        self.frame_num += 1
        return output

    def save(self,image):
        """Save the image of the road"""
        image = image[:,:,::-1]
        filepath = "data/frames/road-{0}-{1}.png".format(self.video_num, self.frame_num)
        cv2.imwrite(filepath,image)



def process_video(url,output,i):
    agent = ImageProcessor(i)
    clip = VideoFileClip(url)
    clip = clip.crop(x_center=600, y_center=360, width=700, height=700)
    clip = clip.fl_image(agent.process)
    clip.write_videofile(output, audio=False)


if __name__ == "__main__":
    urls = pull_videos()
    for i,url in enumerate(urls):
        try:
            print("Processing ",url)
            filename = download_video(url)
            output = 'videos/alabama-{0}.mp4'.format(i)
            process_video(filename,output,i)
        except Exception as e:
            print(e)