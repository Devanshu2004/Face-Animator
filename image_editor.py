# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm

# def crop_image(input_path, output_path, x_min=200, x_max=600, y_min=75, y_max=500):
#     """
#     Crop an image to focus on the facial landmarks for phoneme visualization.
    
#     Parameters:
#     -----------
#     input_path : str
#         Path to the input image
#     output_path : str
#         Path where the cropped image will be saved
#     x_min, x_max, y_min, y_max : int
#         Cropping coordinates
#     """
#     # Read the image
#     img = cv2.imread(input_path)
    
#     if img is None:
#         print(f"Error: Could not read image from {input_path}")
#         return False
    
#     # Get image dimensions
#     height, width = img.shape[:2]
    
#     # Ensure crop coordinates are within image bounds
#     x_min = max(0, x_min)
#     y_min = max(0, y_min)
#     x_max = min(width, x_max)
#     y_max = min(height, y_max)
    
#     # Crop the image
#     cropped_img = img[y_min:y_max, x_min:x_max]
    
#     # Save the cropped image
#     cv2.imwrite(output_path, cropped_img)
#     return True

# def batch_crop_images(input_folder, output_folder, x_min=200, x_max=600, y_min=75, y_max=500):
#     """
#     Apply the same cropping to all numbered images in a folder.
    
#     Parameters:
#     -----------
#     input_folder : str
#         Path to the folder containing input images
#     output_folder : str
#         Path to the folder where cropped images will be saved
#     x_min, x_max, y_min, y_max : int
#         Cropping coordinates
#     """
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         print(f"Created output directory: {output_folder}")
    
#     # Get all numbered PNG files in the input folder
#     image_files = [f for f in os.listdir(input_folder) if f.split('.')[0].isdigit() and f.endswith('.png')]
#     image_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort numerically
    
#     print(f"Found {len(image_files)} image files to process.")
    
#     # Process each image
#     successful = 0
#     failed = 0
    
#     for filename in tqdm(image_files, desc="Cropping images"):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)
        
#         if crop_image(input_path, output_path, x_min, x_max, y_min, y_max):
#             successful += 1
#         else:
#             failed += 1
    
#     print(f"Processing complete: {successful} images successfully cropped, {failed} failed.")

# # Execute the function
# if __name__ == "__main__":
#     input_folder = "predicted_landmarks"
#     output_folder = "cropped_landmarks"
    
#     batch_crop_images(
#         input_folder,
#         output_folder,
#         x_min=200,  # Left boundary
#         x_max=600,  # Right boundary
#         y_min=75,   # Top boundary
#         y_max=500   # Bottom boundary
#     )
    
    
#     # Remove all files and subdirectories in the directory
#     for file in os.listdir("predicted_landmarks"):
#         file_path = os.path.join("predicted_landmarks", file)
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)  # Remove file or symlink
#         elif os.path.isdir(file_path):
#             for subfile in os.listdir(file_path):  
#                 os.unlink(os.path.join(file_path, subfile))  # Remove files in subdirectory
#             os.rmdir(file_path)  # Remove the empty subdirectory

#     # Now remove the empty directory
#     os.rmdir("predicted_landmarks")

import cv2
import numpy as np
import os

def convert_white_to_black(input_path, output_path):
    """
    Reads an image, converts all white pixels to black, and saves the result.
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the output image
    """
    # Read the image
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return False
    
    # Define white (or near white) as pixels with all values above 240
    white_mask = np.all(img >= 240, axis=2)
    
    # Set all white pixels to black [0,0,0]
    img[white_mask] = [0, 0, 0]
    
    # Remove text by detecting non-black, non-green pixels
    # For text like axis labels, numbers, etc.
    # Define green with some tolerance
    lower_green = np.array([0, 150, 0])
    upper_green = np.array([100, 255, 100])
    green_mask = cv2.inRange(img, lower_green, upper_green)
    
    # Create a mask that keeps only green elements
    # Everything that's not green becomes black
    not_green_mask = (green_mask == 0)
    
    # Apply the mask to the image
    for c in range(3):
        img[:,:,c][not_green_mask] = 0
    
    # Save the result
    cv2.imwrite(output_path, img)
    print(f"Converted image saved to {output_path}")
    return True

def process_all_images(input_folder, output_folder):
    """
    Process all PNG images in the input folder and save results to output folder.
    
    Args:
        input_folder (str): Folder containing input images
        output_folder (str): Folder to save processed images
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Get all PNG files in the input folder
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    if not input_files:
        print(f"No PNG files found in {input_folder}")
        return
    
    print(f"Found {len(input_files)} PNG files to process.")
    
    # Process each file
    success_count = 0
    for filename in input_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        if convert_white_to_black(input_path, output_path):
            success_count += 1
    
    print(f"Processing complete: {success_count} of {len(input_files)} images successfully processed.")

# Usage
if __name__ == "__main__":
    input_folder = "predicted_landmarks"
    output_folder = "cropped_landmarks"
    process_all_images(input_folder, output_folder)