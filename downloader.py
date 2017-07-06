import cv2
import pafy
import requests
from storage.models import VideoModel

def pull_videos(url):
    """
    Return a list of video urls from the server
    """
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


def downloader(camera,url):
    """Continuously download videos from a source"""
    urls = pull_videos(url)
    for i,url in enumerate(urls):
        try:
            print("Downloading ",url)
            filename = download_video(url)
            VideoModel.create(filename)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    camera = "auburn"
    link = "https://www.youtube.com/watch?v=cjuskMMYlLA"
    downloader(camera,link)