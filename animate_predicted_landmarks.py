import cv2
import numpy as np
import os
import re
from glob import glob

def natural_sort_key(s):
    """
    Sort strings with embedded numbers naturally.
    E.g., ["1.png", "2.png", "10.png"] instead of ["1.png", "10.png", "2.png"]
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', os.path.basename(s))]

def create_animation(image_folder, output_file="animation.mp4", fps=24):
    """
    Create an animation from a series of PNG images and save as MP4.
    Images are used as-is without any cropping.
    
    Parameters:
    - image_folder: Folder containing the PNG images
    - output_file: Name of the output video file
    - fps: Frames per second for the output video
    """
    # Get all PNG files
    image_files = glob(os.path.join(image_folder, "*.png"))
    
    # Sort them naturally (numerically)
    image_files = sorted(image_files, key=natural_sort_key)
    
    if not image_files:
        print(f"No PNG files found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} images to animate")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Error: Could not read image {image_files[0]}")
        return
        
    height, width, layers = first_image.shape
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Add each image to the video
    for i, image_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        # Read the image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Warning: Could not read image {image_file}, skipping")
            continue
            
        # Add to video as-is without cropping
        video.write(img)
    
    # Release the video writer
    video.release()
    print(f"Animation saved as {output_file}")

if __name__ == "__main__":
    # You can change these parameters as needed
    image_folder = "predicted_landmarks"  # Folder containing your PNG files
    output_file = "animation.mp4"
    fps = 9  # Frames per second
    
    create_animation(image_folder, output_file, fps)