import os
import give_timestamps_to_video
import generate_model
import predict_facial_landmarks
import image_editor
import dimension_equator
import animate_predicted_landmarks
import animate_speech

input_image_static = "face_texture.png"
input_video_static = "input_video.mp4"
model_name_static = "devanshu_v2"
text_to_generate_static = "Hi! I am your personal assistant Jarvis (female one). I will be assisting you throughout your entire working process. Let me know if you need my help."
output_location_static = "outputs/intro.mp4"

def clear_directory(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            print(file)
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

def generate_video(model_name, text_to_generate, input_image, output_location, input_video=None):
    temp_folders = [
        "temporary_usage_files",
        "predicted_landmarks",
        "cropped_landmarks"
    ]
    
    # Train the model if data doesn't exist
    if not os.path.exists(f"models/{model_name}/{model_name}.h5"):
        for folder in temp_folders:
            clear_directory(folder)
        
        os.mkdir("temporary_usage_files")
        
        # Generate timestamps
        output_csv_path = "temporary_usage_files/aligned_output.csv"
        give_timestamps_to_video.main(input_video, output_csv_path)
        
        print("#######################")
        print(os.path.exists(f"models/{model_name}/{model_name}.h5"))
        print("The model doesn't exist!\nMaking a new one!")
        print("#######################")
        generate_model.main(output_csv_path, input_video, model_name=model_name)
    
    # Predict landmarks
    predict_facial_landmarks.predict_landmarks_from_text_input(model_name, text_to_generate)
    
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
    
    output_dir = output_location.rsplit("/", 1)
    if not os.path.exists(output_dir[0]):
        os.mkdir(output_dir[0])
    # animate_speech.main(text_to_generate)
    animate_speech.main(text_to_generate, output_location)
    
    clear_directory("predicted_landmarks")
    print("Successfully deleted 'predicted_landmarks'")

if __name__ == "__main__":
    generate_video(
        model_name=model_name_static,
        text_to_generate=text_to_generate_static,
        input_image=input_image_static,
        output_location=output_location_static,
        input_video=input_video_static
        )
