import os
import give_timestamps_to_video
import generate_model
import predict_facial_landmarks
import image_editor
import dimension_equator
import animate_predicted_landmarks

input_image = "face_texture.png"
input_video = "input_video.mp4"

def clear_directory(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    for subfile in os.listdir(file_path):
                        os.unlink(os.path.join(file_path, subfile))
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        try:
            os.rmdir(dir_path)
            print(f"Deleted existing '{dir_path}' folder.")
        except OSError:
            print(f"Failed to delete {dir_path}")

def main():
    temp_folders = [
        "temporary_usage_files",
        "predicted_landmarks",
        "cropped_landmarks"
    ]
    
    for folder in temp_folders:
        clear_directory(folder)
    
    os.mkdir("temporary_usage_files")
    
    # Generate timestamps
    output_csv_path = "temporary_usage_files/aligned_output.csv"
    give_timestamps_to_video.main(input_video, output_csv_path)
    
    # Train the model
    generate_model.main(output_csv_path, input_video)
    
    # Predict landmarks
    predict_facial_landmarks.predict_landmarks_from_text_input()
    
    # Process images
    image_editor.process_all_images("predicted_landmarks", "cropped_landmarks")
    clear_directory("predicted_landmarks")
    
    # Dimension processing
    try:
        print("Starting batch processing of images in cropped_landmarks")
        results = dimension_equator.process_all_images(
            input_image, 
            "cropped_landmarks", 
            "predicted_landmarks",
            True
        )
        
        if isinstance(results, dict) and "error" in results:
            print(f"Error: {results['error']}")
        else:
            print("\nProcessing complete. Summary:")
            print(f"- Total images processed: {len(results)}")
            print(f"- Method used: {results[0]['method']}")
            print(f"- Output dimensions: {results[0]['new_dimensions']}")
            print(f"- Processed images saved to: predicted_landmarks")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    clear_directory("cropped_landmarks")
    
    # Create animation
    animate_predicted_landmarks.create_animation("predicted_landmarks", "animation.mp4", 9)

if __name__ == "__main__":
    main()