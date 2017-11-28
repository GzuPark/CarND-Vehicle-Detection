from yolo_small import *
from moviepy.editor import VideoFileClip

yolo = yolo_tf()

def process(img, detect=yolo):
    detect_from_file(detect, img)
    result = show_results(img, detect)
    return result

yolo_result = "yolo_project_video_result.mp4"
yolo_clip = VideoFileClip("project_video.mp4")
result_clip = yolo_clip.fl_image(process)
result_clip.write_videofile(yolo_result, audio=False)
