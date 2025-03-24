import os
import give_timestamps_to_video
import generate_model
import predict_facial_landmarks
import image_editor
import dimension_equator
import animate_predicted_landmarks

input_image = "face_texture.png"
input_video = "input_video.mp4"

def main():
    folder_name = "temporary_usage_files"

    # Check if folder exists
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        # Delete all files inside the folder
        for file in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Removes empty directories only

        # Remove the folder after clearing contents
        os.rmdir(folder_name)
        print(f"Deleted existing '{folder_name}' folder.")

    # Create a new folder
    os.mkdir(folder_name)
    
    # Generate timestamps of the facial landmarks from the input training video
    video_path = "input_video.mp4"  # Replace with your video file path
    output_csv_path = "temporary_usage_files/aligned_output.csv" # Output CSV file path
    give_timestamps_to_video.main(input_video, output_csv_path)
    
    # Train the model
    generate_model.main(output_csv_path, input_video)
    
    # Now, for the new given text predict landmarks
    predict_facial_landmarks.predict_landmarks_from_text_input()
    
    # Remove the extra space from the predicted landmarks
    image_editor.process_all_images("predicted_landmarks", "cropped_landmarks")
    
    
    # Remove all files and subdirectories in the directory
    for file in os.listdir("predicted_landmarks"):
        file_path = os.path.join("predicted_landmarks", file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # Remove file or symlink
        elif os.path.isdir(file_path):
            for subfile in os.listdir(file_path):  
                os.unlink(os.path.join(file_path, subfile))  # Remove files in subdirectory
            os.rmdir(file_path)  # Remove the empty subdirectory

    # Now remove the empty directory
    os.rmdir("predicted_landmarks")
    
    target_directory = "cropped_landmarks"  # Directory with images to process
    output_directory = "predicted_landmarks"  # Directory to save processed images
    use_first_image_as_reference = True  # Use the first image (0.png) as reference
    
    try:
        print(f"Starting batch processing of images in {target_directory}")
        results = dimension_equator.process_all_images(
            input_image, 
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
    
    image_folder = "predicted_landmarks"  # Folder containing your PNG files
    output_file = "animation.mp4"
    fps = 9  # Frames per second
    
    animate_predicted_landmarks.create_animation(image_folder, output_file, fps)

if __name__ == "__main__":
    main()
