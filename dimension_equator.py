import cv2
import os
import argparse
from PIL import Image

def resize_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.resize(target_size, Image.LANCZOS)
    image.save(image_path)
    print(f"Resized image saved at {image_path}")

def resize_video(video_path, target_size):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = video_path.replace(".mp4", "_resized.mp4")
    out = cv2.VideoWriter(temp_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), target_size)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        out.write(frame)
    
    cap.release()
    out.release()
    
    os.replace(temp_video_path, video_path)
    print(f"Resized video saved at {video_path}")

def main(image_path, video_path):
    image = Image.open(image_path)
    cap = cv2.VideoCapture(video_path)
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_width, image_height = image.size
    
    target_size = (max(image_width, video_width), max(image_height, video_height))
    
    resize_image(image_path, target_size)
    resize_video(video_path, target_size)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("video", help="Path to the video file")
    args = parser.parse_args()
    
    main(args.image, args.video)
