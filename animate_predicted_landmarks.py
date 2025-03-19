# import cv2
# import numpy as np
# import os
# import re
# from glob import glob

# def natural_sort_key(s):
#     """
#     Sort strings with embedded numbers naturally.
#     E.g., ["1.png", "2.png", "10.png"] instead of ["1.png", "10.png", "2.png"]
#     """
#     return [int(text) if text.isdigit() else text.lower()
#             for text in re.split(r'(\d+)', os.path.basename(s))]

# def create_animation(image_folder, output_file="animation.mp4", fps=24):
#     """
#     Create an animation from a series of PNG images and save as MP4.
    
#     Parameters:
#     - image_folder: Folder containing the PNG images
#     - output_file: Name of the output video file
#     - fps: Frames per second for the output video
#     """
#     # Get all PNG files
#     image_files = glob(os.path.join(image_folder, "*.png"))
    
#     # Sort them naturally (numerically)
#     image_files = sorted(image_files, key=natural_sort_key)
    
#     if not image_files:
#         print(f"No PNG files found in {image_folder}")
#         return
    
#     print(f"Found {len(image_files)} images to animate")
    
#     # Read the first image to get dimensions
#     first_image = cv2.imread(image_files[0])
#     height, width, layers = first_image.shape
    
#     # Initialize the video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
#     video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
#     # Add each image to the video
#     for i, image_file in enumerate(image_files):
#         print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
#         img = cv2.imread(image_file)
#         video.write(img)
    
#     # Release the video writer
#     video.release()
#     print(f"Animation saved as {output_file}")

# if __name__ == "__main__":
#     # You can change these parameters as needed
#     image_folder = "predicted_landmarks"  # Folder containing your PNG files
#     output_file = "animation.mp4"
#     fps = 8  # Frames per second
    
#     create_animation(image_folder, output_file, fps)

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

def crop_landmark_image(image):
    """
    Crop the image to focus on just the facial landmarks.
    This function removes unnecessary borders and title text.
    
    Parameters:
    - image: Input image containing facial landmarks
    
    Returns:
    - Cropped image with just the facial landmarks
    """
    # Find all non-black pixels (landmarks)
    green_pixels = np.where(image[:, :, 1] > 100)  # Green channel threshold
    
    if len(green_pixels[0]) == 0:  # No landmarks found
        return image
    
    # Get bounding box of landmarks
    min_y, max_y = np.min(green_pixels[0]), np.max(green_pixels[0])
    min_x, max_x = np.min(green_pixels[1]), np.max(green_pixels[1])
    
    # Add padding around landmarks
    padding = 50
    min_y = max(0, min_y - padding)
    min_x = max(0, min_x - padding)
    max_y = min(image.shape[0], max_y + padding)
    max_x = min(image.shape[1], max_x + padding)
    
    # Crop the image
    cropped_image = image[min_y:max_y, min_x:max_x]
    
    return cropped_image

def create_animation(image_folder, output_file="animation.mp4", fps=24, temp_folder="temp_cropped"):
    """
    Create an animation from a series of PNG images and save as MP4.
    Images are first cropped to focus on the facial landmarks.
    
    Parameters:
    - image_folder: Folder containing the PNG images
    - output_file: Name of the output video file
    - fps: Frames per second for the output video
    - temp_folder: Temporary folder to store cropped images
    """
    # Get all PNG files
    image_files = glob(os.path.join(image_folder, "*.png"))
    
    # Sort them naturally (numerically)
    image_files = sorted(image_files, key=natural_sort_key)
    
    if not image_files:
        print(f"No PNG files found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} images to animate")
    
    # Create temporary folder for cropped images if it doesn't exist
    os.makedirs(temp_folder, exist_ok=True)
    
    cropped_image_files = []
    
    # Process each image
    for i, image_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        # Read the image
        img = cv2.imread(image_file)
        
        # Crop the image
        cropped_img = crop_landmark_image(img)
        
        # Save the cropped image
        cropped_file = os.path.join(temp_folder, os.path.basename(image_file))
        cv2.imwrite(cropped_file, cropped_img)
        cropped_image_files.append(cropped_file)
    
    if not cropped_image_files:
        print("No images were successfully cropped")
        return
        
    # Read the first cropped image to get dimensions
    first_image = cv2.imread(cropped_image_files[0])
    height, width, layers = first_image.shape
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Add each cropped image to the video
    for cropped_file in cropped_image_files:
        img = cv2.imread(cropped_file)
        video.write(img)
    
    # Release the video writer
    video.release()
    print(f"Animation saved as {output_file}")
    
    # Clean up temporary files if needed
    # Uncomment the following lines to automatically delete temporary files
    for cropped_file in cropped_image_files:
        os.remove(cropped_file)
    os.rmdir(temp_folder)

if __name__ == "__main__":
    # You can change these parameters as needed
    image_folder = "predicted_landmarks"  # Folder containing your PNG files
    output_file = "animation.mp4"
    fps = 9  # Frames per second
    temp_folder = "temp_cropped"  # Temporary folder for cropped images
    
    create_animation(image_folder, output_file, fps, temp_folder)