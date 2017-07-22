import os
import matplotlib
import numpy as np
from moviepy.editor import VideoFileClip
from .main import ImageNetwork
from .provider import DatabaseProvider
from .helpers import bboxes_draw_on_img, bboxes_nms_intersection_avg, update_car_collection, plot_image, colors_tableau


ROOT = os.path.dirname(os.path.realpath(__file__))
CHECKPOINT = os.path.join(ROOT,'checkpoints','ssd_model.ckpt')
TEST_INPUT = os.path.join(ROOT,'test','input')
TEST_OUTPUT = os.path.join(ROOT,'test','output')


# Test the provider
provider = DatabaseProvider()
item = True
while item:
    item = provider.get(['image', 'shape', 'object/label','object/bbox'])
    print(item)


# Image Pipeline
def run():
    with ImageNetwork(select_threshold=0.8, nms_threshold=0.5) as network:
        # Load a sample image.
        image_names = sorted(os.listdir(TEST_INPUT))
        input_path = os.path.join(TEST_INPUT,image_names[5])
        img = matplotlib.image.imread(input_path)

        print('Input image shape:', img.shape)
        plot_image(img, 'Original frame', (10, 10))

        # SSD network on image.
        rclasses, rscores, rbboxes = network._process_image(img)

        # Draw bboxes of detected objects.
        img_bboxes = np.copy(img)
        bboxes_draw_on_img(img_bboxes, rscores, rbboxes, colors_tableau, thickness=2, show_text=False)

        plot_image(img_bboxes, 'Raw SSD network output: multiple detections.', (10, 10))

        # Apply Non-Maximum-Selection
        rclasses_nms, rscores_nms, rbboxes_nms = network.process_image(img)

        # Draw bboxes
        img_bboxes = np.copy(img)
        bboxes_draw_on_img(img_bboxes, rscores_nms, rbboxes_nms, colors_tableau, thickness=2)

        plot_image(img_bboxes, 'SSD network output + Non Maximum Suppression.', (10, 10))


        # Vehicle detection: images
        def process_image(img, select_threshold=0.8, nms_threshold=0.5):
            # SSD network + NMS on image.
            rclasses, rscores, rbboxes = network.process_image(img, select_threshold)
            rclasses, rscores, rbboxes = bboxes_nms_intersection_avg(rclasses, rscores, rbboxes, threshold=nms_threshold)
            # Draw bboxes of detected objects.
            bboxes_draw_on_img(img, rscores, rbboxes, colors_tableau, thickness=2, show_text=True)
            return img

        # Load a sample image.
        image_names = sorted(os.listdir(TEST_INPUT))
        for name in image_names:
            if name.endswith("jpg"):
                print(name)
                input_path = os.path.join(TEST_INPUT, name)
                output_path = os.path.join(TEST_OUTPUT, name)
                img = matplotlib.image.imread(input_path)
                img = process_image(img, select_threshold=0.6, nms_threshold=0.5)
                matplotlib.image.imsave(output_path, img, format='jpg')


        # Vehicle detection: videos
        network.cars = []

        # Selection parameters.
        select_threshold=0.7
        nms_threshold=0.5

        video_input = os.path.join(TEST_INPUT,'test_video.mp4')
        video_output = os.path.join(TEST_OUTPUT,'test_video_cars.mp4')
        clip1 = VideoFileClip(video_input)
        white_clip = clip1.fl_image(lambda x: network.ssd_process_frame(x, select_threshold, nms_threshold))
        white_clip.write_videofile(video_output, audio=False)
        network.cars = []

        # Selection parameters.
        select_threshold=0.7
        nms_threshold=0.5

        video_input = os.path.join(TEST_INPUT,'project_video.mp4')
        video_output = os.path.join(TEST_OUTPUT,'project_video_cars.mp4')
        clip1 = VideoFileClip(video_input)
        white_clip = clip1.fl_image(lambda x: network.ssd_process_frame(x, select_threshold, nms_threshold))
        white_clip.write_videofile(video_output, audio=False)


        network.cars = []

        # Selection parameters.
        select_threshold=0.6
        nms_threshold=0.3

        video_input = os.path.join(TEST_INPUT,'challenge_video.mp4')
        video_output = os.path.join(TEST_OUTPUT,'challenge_video_cars.mp4')
        clip1 = VideoFileClip(video_input)
        white_clip = clip1.fl_image(lambda x:network.ssd_process_frame(x, select_threshold, nms_threshold))
        white_clip.write_videofile(video_output, audio=False)
