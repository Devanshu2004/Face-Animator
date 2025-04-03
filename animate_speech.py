import os
import subprocess
from gtts import gTTS
import sys

print("Python Version:", sys.version)
print("Python Executable:", sys.executable)
print("Sys Path:", sys.path)

def generate_audio_from_text(text, output_audio_file="speech.mp3"):
    """
    Generate audio from text using Google Text-to-Speech
    
    Parameters:
    - text: Text to convert to speech
    - output_audio_file: File path to save the generated audio
    
    Returns:
    - Path to the generated audio file
    """
    print(f"Generating audio for text: {text}")
    try:
        # Create a gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save the audio file
        tts.save(output_audio_file)
        print(f"Audio saved to {output_audio_file}")
        return output_audio_file
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def get_video_duration(video_file):
    """
    Get the duration of a video file using FFmpeg
    
    Parameters:
    - video_file: Path to the video file
    
    Returns:
    - Duration in seconds as a float
    """
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            video_file
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout.strip())
        print(f"Video duration: {duration:.2f}s")
        return duration
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0

def get_audio_duration(audio_file):
    """
    Get the duration of an audio file using FFmpeg
    
    Parameters:
    - audio_file: Path to the audio file
    
    Returns:
    - Duration in seconds as a float
    """
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            audio_file
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout.strip())
        print(f"Audio duration: {duration:.2f}s")
        return duration
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0

def merge_audio_with_video(video_file, audio_file, output_file="final_animation.mp4"):
    """
    Merge audio with video using FFmpeg
    
    Parameters:
    - video_file: Path to the input video file
    - audio_file: Path to the audio file to merge
    - output_file: Path to save the merged video
    """
    try:
        video_duration = get_video_duration(video_file)
        audio_duration = get_audio_duration(audio_file)
        
        # Check if audio is longer than video
        if audio_duration > video_duration:
            print(f"Warning: Audio ({audio_duration:.2f}s) is longer than video ({video_duration:.2f}s)")
            print("Creating a looped version of the video to match audio duration")
            
            # Create a temporary file for the looped video
            temp_looped_video = "temp_looped_video.mp4"
            
            # Calculate how many times to loop the video
            num_loops = int(audio_duration / video_duration) + 1
            print(f"Looping video {num_loops} times")
            
            # Create a file with a list of the input video repeated
            list_file = "input_list.txt"
            with open(list_file, "w") as f:
                for _ in range(num_loops):
                    f.write(f"file '{video_file}'\n")
            
            # Use FFmpeg to concatenate the videos
            concat_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c', 'copy',
                temp_looped_video
            ]
            subprocess.run(concat_cmd, check=True)
            
            # Now trim the looped video to match audio duration exactly
            trim_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', temp_looped_video,
                '-ss', '0',
                '-to', str(audio_duration),
                '-c', 'copy',
                'temp_trimmed_video.mp4'
            ]
            subprocess.run(trim_cmd, check=True)
            
            # Use the trimmed video for the final merge
            video_to_use = 'temp_trimmed_video.mp4'
            
        elif video_duration > audio_duration:
            print(f"Warning: Video ({video_duration:.2f}s) is longer than audio ({audio_duration:.2f}s)")
            print("Trimming video to match audio duration")
            
            # Trim the video to match audio duration
            trim_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', video_file,
                '-ss', '0',
                '-to', str(audio_duration),
                '-c', 'copy',
                'temp_trimmed_video.mp4'
            ]
            subprocess.run(trim_cmd, check=True)
            
            # Use the trimmed video for the final merge
            video_to_use = 'temp_trimmed_video.mp4'
        else:
            # Durations match, use original video
            video_to_use = video_file
        
        # Merge the audio with the video
        merge_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', video_to_use,
            '-i', audio_file,
            '-map', '0:v',  # Take video from first input
            '-map', '1:a',  # Take audio from second input
            '-c:v', 'copy',  # Copy video codec
            '-shortest',  # Finish encoding when the shortest input stream ends
            output_file
        ]
        
        print(f"Merging audio and video using command: {' '.join(merge_cmd)}")
        subprocess.run(merge_cmd, check=True)
        
        # Clean up temporary files
        for temp_file in ['temp_looped_video.mp4', 'temp_trimmed_video.mp4', 'input_list.txt']:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"Successfully merged audio and video to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error merging audio with video: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(text_to_generate, output_location):
    # The text from your animation
    text = text_to_generate
    
    # Generate audio from the text
    audio_file = generate_audio_from_text(text)
    
    if audio_file and os.path.exists(audio_file):
        # Merge the audio with the animation video
        video_file = "animation.mp4"
        output_file = output_location
        
        if os.path.exists(video_file):
            print(output_file)
            merged_file = merge_audio_with_video(video_file, audio_file, output_file)
            
            if merged_file and os.path.exists(merged_file):
                print(f"Process completed successfully. Final video saved as {merged_file}")
            else:
                print("Failed to merge audio and video.")
        else:
            print(f"Error: Video file '{video_file}' not found.")
    else:
        print("Failed to generate audio file.")
    
    os.remove("speech.mp3")
    os.remove("animation.mp4")
    print("Removed 'speech.mp3' and 'animation.mp4' successfully")

if __name__ == "__main__":
    main()