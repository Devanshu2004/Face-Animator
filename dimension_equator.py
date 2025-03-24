import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import glob

def detect_face_landmarks(image_path_or_array):
    """
    Detect face landmarks in an image and calculate distances from face boundaries to image boundaries.
    
    Args:
        image_path_or_array: Path to the input image or numpy array
        
    Returns:
        dict: Distances from face boundaries to image boundaries and image dimensions
    """
    # Initialize MediaPipe Face Detection and Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    # Read the image
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        if image is None:
            raise ValueError(f"Could not read image from {image_path_or_array}")
    else:
        image = image_path_or_array
    
    # Convert to RGB (MediaPipe requires RGB input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    # Get all green points (landmarks) from the image - specific for the provided image
    # This is an alternative method since the image already contains landmarks
    landmarks_px = []
    green_mask = cv2.inRange(image, (0, 200, 0), (10, 255, 10))
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            landmarks_px.append((cx, cy))
    
    # If no green points found, try MediaPipe
    if not landmarks_px:
        # Process the image with MediaPipe Face Mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(image_rgb)
        
        # Check if face is detected
        if not results.multi_face_landmarks:
            return {"error": "No face detected in the image"}
        
        # Extract face landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        landmarks_px = [(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks]
    
    # Initialize values for finding extremes
    top_y = float('inf')
    bottom_y = -float('inf')
    left_x = float('inf')
    right_x = -float('inf')
    
    # Find the extreme landmarks
    for x, y in landmarks_px:
        if y < top_y:
            top_y = y
        if y > bottom_y:
            bottom_y = y
        if x < left_x:
            left_x = x
        if x > right_x:
            right_x = x
    
    # Calculate distances
    distance_to_top = top_y
    distance_to_bottom = h - bottom_y
    distance_to_left = left_x
    distance_to_right = w - right_x
    
    return {
        "top": {
            "distance": distance_to_top,
            "position": top_y
        },
        "bottom": {
            "distance": distance_to_bottom,
            "position": bottom_y
        },
        "left": {
            "distance": distance_to_left,
            "position": left_x
        },
        "right": {
            "distance": distance_to_right,
            "position": right_x
        },
        "image_dimensions": {
            "width": w,
            "height": h
        }
    }

def crop_image_using_reference_boundaries(reference_boundaries, target_image_path, output_path):
    """
    Crop a target image to match face boundary distances from a reference image.
    
    Args:
        reference_boundaries (dict): Boundary information from the reference image
        target_image_path (str): Path to the image that needs to be cropped
        output_path (str): Path to save the cropped image
        
    Returns:
        dict: Information about the crop operation
    """
    # Get face boundaries for the target image
    target_boundaries = detect_face_landmarks(target_image_path)
    if "error" in target_boundaries:
        return {"error": target_boundaries["error"], "file": target_image_path}
    
    # Open the target image with PIL for cropping
    target_img = Image.open(target_image_path)
    target_width, target_height = target_img.size
    
    # Get the face boundary positions from the target image
    target_top = target_boundaries["top"]["position"]
    target_bottom = target_boundaries["bottom"]["position"]
    target_left = target_boundaries["left"]["position"]
    target_right = target_boundaries["right"]["position"]
    
    # Get the desired distances from the reference image
    ref_top_distance = reference_boundaries["top"]["distance"]
    ref_bottom_distance = reference_boundaries["bottom"]["distance"]
    ref_left_distance = reference_boundaries["left"]["distance"]
    ref_right_distance = reference_boundaries["right"]["distance"]
    
    # Calculate new dimensions to maintain the same distances
    face_height = target_bottom - target_top
    face_width = target_right - target_left
    
    new_height = face_height + ref_top_distance + ref_bottom_distance
    new_width = face_width + ref_left_distance + ref_right_distance
    
    # Calculate crop box coordinates
    # We need to adjust the current image to get the desired distances
    crop_top = max(0, target_top - ref_top_distance)
    crop_bottom = min(target_height, target_bottom + ref_bottom_distance)
    crop_left = max(0, target_left - ref_left_distance)
    crop_right = min(target_width, target_right + ref_right_distance)
    
    # Check if the image needs padding instead of cropping
    need_padding = (
        crop_top < 0 or 
        crop_bottom > target_height or 
        crop_left < 0 or 
        crop_right > target_width
    )
    
    result = {
        "file": target_image_path,
        "new_dimensions": (0, 0),
        "crop_or_pad_info": "",
        "method": ""
    }
    
    if need_padding:
        # Create a new blank image with the desired dimensions
        padded_img = Image.new('RGB', (new_width, new_height), (0, 0, 0))  # Black background
        
        # Calculate paste position
        paste_x = max(0, ref_left_distance - target_left)
        paste_y = max(0, ref_top_distance - target_top)
        
        # Paste the original image onto the padded image
        padded_img.paste(target_img, (paste_x, paste_y))
        padded_img.save(output_path)
        
        result["new_dimensions"] = (new_width, new_height)
        result["crop_or_pad_info"] = (paste_x, paste_y, target_width, target_height)
        result["method"] = "padded"
    else:
        # Crop the image
        crop_box = (crop_left, crop_top, crop_right, crop_bottom)
        cropped_img = target_img.crop(crop_box)
        cropped_img.save(output_path)
        
        result["new_dimensions"] = (crop_right - crop_left, crop_bottom - crop_top)
        result["crop_or_pad_info"] = crop_box
        result["method"] = "cropped"
    
    return result

def apply_same_crop_to_image(crop_info, target_image_path, output_path):
    """
    Apply the same crop or padding to a target image based on previously determined crop info.
    
    Args:
        crop_info (dict): Crop information from a previous operation
        target_image_path (str): Path to the image that needs to be cropped
        output_path (str): Path to save the cropped image
    
    Returns:
        dict: Information about the crop operation
    """
    target_img = Image.open(target_image_path)
    
    result = {
        "file": target_image_path,
        "method": crop_info["method"]
    }
    
    if crop_info["method"] == "padded":
        # Extract padding information
        paste_x, paste_y, _, _ = crop_info["crop_or_pad_info"]
        new_width, new_height = crop_info["new_dimensions"]
        
        # Create a new blank image with the same dimensions
        padded_img = Image.new('RGB', (new_width, new_height), (0, 0, 0))  # Black background
        
        # Paste the original image onto the padded image
        padded_img.paste(target_img, (paste_x, paste_y))
        padded_img.save(output_path)
        
        result["new_dimensions"] = (new_width, new_height)
    else:
        # Extract crop box
        crop_box = crop_info["crop_or_pad_info"]
        
        # Crop the image
        cropped_img = target_img.crop(crop_box)
        cropped_img.save(output_path)
        
        result["new_dimensions"] = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
    
    return result

def process_all_images(reference_image_path, target_directory, output_directory, use_first_image_as_reference=True):
    """
    Process all images in a directory, using either a reference image or the first image in the directory
    to determine crop boundaries.
    
    Args:
        reference_image_path (str): Path to the reference image (used if use_first_image_as_reference is False)
        target_directory (str): Directory containing the images to be processed
        output_directory (str): Directory to save the processed images
        use_first_image_as_reference (bool): Whether to use the first image in the directory as reference
    
    Returns:
        list: Results for each processed image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get list of image files
    image_files = sorted(glob.glob(os.path.join(target_directory, "*.png")))
    if not image_files:
        return {"error": f"No PNG images found in {target_directory}"}
    
    results = []
    reference_boundaries = None
    first_image_crop_info = None
    
    # Get reference boundaries
    if use_first_image_as_reference:
        first_image = image_files[0]
        reference_boundaries = detect_face_landmarks(first_image)
        
        if "error" in reference_boundaries:
            return {"error": f"Error processing reference image: {reference_boundaries['error']}"}
        
        print(f"Using first image {os.path.basename(first_image)} as reference")
    else:
        try:
            reference_boundaries = detect_face_landmarks(reference_image_path)
            if "error" in reference_boundaries:
                return {"error": f"Error processing reference image: {reference_boundaries['error']}"}
            
            print(f"Using {os.path.basename(reference_image_path)} as reference")
        except Exception as e:
            return {"error": f"Could not process reference image: {e}"}
    
    # Process first image
    first_output_path = os.path.join(output_directory, os.path.basename(image_files[0]))
    first_result = crop_image_using_reference_boundaries(reference_boundaries, image_files[0], first_output_path)
    results.append(first_result)
    
    if "error" in first_result:
        return {"error": f"Error processing first image: {first_result['error']}"}
    
    # Store crop info from first image
    first_image_crop_info = {
        "new_dimensions": first_result["new_dimensions"],
        "crop_or_pad_info": first_result["crop_or_pad_info"],
        "method": first_result["method"]
    }
    
    # Process all other images using the same crop info
    for image_file in image_files[1:]:
        output_path = os.path.join(output_directory, os.path.basename(image_file))
        
        # Apply the same crop to this image
        result = apply_same_crop_to_image(first_image_crop_info, image_file, output_path)
        results.append(result)
        
        print(f"Processed {os.path.basename(image_file)} → {os.path.basename(output_path)}")
    
    return results

# Main execution
if __name__ == "__main__":
    # Parameters
    reference_image_path = "face_texture.png"  # Optional external reference image
    target_directory = "cropped_landmarks"  # Directory with images to process
    output_directory = "predicted_landmarks"  # Directory to save processed images
    use_first_image_as_reference = True  # Use the first image (0.png) as reference
    
    try:
        print(f"Starting batch processing of images in {target_directory}")
        results = process_all_images(
            reference_image_path, 
            target_directory, 
            output_directory,
            use_first_image_as_reference
        )
        
        if isinstance(results, dict) and "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"\nProcessing complete. Summary:")
            print(f"- Total images processed: {len(results)}")
            print(f"- Method used: {results[0]['method']}")
            print(f"- Output dimensions: {results[0]['new_dimensions']}")
            print(f"- Processed images saved to: {output_directory}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    # Remove all files and subdirectories in the directory
    for file in os.listdir("cropped_landmarks"):
        file_path = os.path.join("cropped_landmarks", file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # Remove file or symlink
        elif os.path.isdir(file_path):
            for subfile in os.listdir(file_path):  
                os.unlink(os.path.join(file_path, subfile))  # Remove files in subdirectory
            os.rmdir(file_path)  # Remove the empty subdirectory

    # Now remove the empty directory
    os.rmdir("cropped_landmarks")