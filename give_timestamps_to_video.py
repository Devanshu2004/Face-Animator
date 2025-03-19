import os
import pandas as pd
import ffmpeg
import whisper
import re
import pronouncing

# Step 1: Extract audio from the video using ffmpeg
def extract_audio_from_video(video_path, audio_output_path):
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_output_path, acodec='pcm_s16le', ar='16000')  # WAV format, 16kHz sample rate
            .run(overwrite_output=True)
        )
        print(f"Audio extracted and saved to {audio_output_path}")
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode('utf-8')}")
        raise

# Step 2: Transcribe audio and get word-level timestamps using Whisper
def transcribe_audio_with_whisper(audio_path):
    # Load the Whisper model
    model = whisper.load_model("base")  # You can use "small", "medium", or "large" for better accuracy

    # Transcribe the audio with word-level timestamps
    result = model.transcribe(audio_path, word_timestamps=True)

    # Extract word-level timestamps
    aligned_data = []
    for segment in result["segments"]:
        for word in segment["words"]:
            aligned_data.append({
                "word": word["word"].strip(),
                "start_time": word["start"],
                "end_time": word["end"]
            })

    return aligned_data

# New function: Convert word to phonetic representation
def get_phonetic_representation(word):
    """
    Convert a word to its phonetic representation using the CMU Pronouncing Dictionary.
    Returns the phonetic representation or an indication if not found.
    """
    # Clean the word (remove punctuation and convert to lowercase)
    clean_word = re.sub(r'[^\w\s]', '', word.lower())
    
    if not clean_word:  # If the word is just punctuation
        return "[no text]"
        
    # Get phonetic pronunciation from pronouncing dictionary
    pronunciations = pronouncing.phones_for_word(clean_word)
    if pronunciations:
        # Use the first pronunciation (most common)
        return pronunciations[0]
    else:
        # If word not in dictionary
        return "[not found]"

# Step 3: Add phonetic representations and save to CSV
def save_to_csv_with_phonetics(aligned_data, output_csv_path):
    # Convert to DataFrame
    df = pd.DataFrame(aligned_data)
    
    # Add phonetic representation column
    df['phonetic'] = df['word'].apply(get_phonetic_representation)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Aligned data with phonetics saved to {output_csv_path}")

# Main function
def main(video_path, output_csv_path):
    # Step 1: Extract audio from the video
    audio_output_path = "temp_audio.wav"
    extract_audio_from_video(video_path, audio_output_path)

    # Step 2: Transcribe audio and get word-level timestamps
    aligned_data = transcribe_audio_with_whisper(audio_output_path)

    # Step 3: Add phonetic representations and save to CSV
    save_to_csv_with_phonetics(aligned_data, output_csv_path)

    # Clean up temporary audio file
    os.remove(audio_output_path)
    print("Temporary audio file removed.")

# Run the script
if __name__ == "__main__":
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
    
    video_path = "input_video.mp4"  # Replace with your video file path
    output_csv_path = "temporary_usage_files/aligned_output.csv" # Output CSV file path

    main(video_path, output_csv_path)