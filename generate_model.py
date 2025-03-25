import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Load and preprocess phonetic data
def load_and_preprocess_phonetics(csv_path):
    """Load phonetic data from CSV and preprocess it."""
    df = pd.read_csv(csv_path)
    
    # Create timestamps for each phoneme (we'll need these to align with video frames)
    phonemes_with_times = []
    
    for _, row in df.iterrows():
        word = row['word']
        start_time = row['start_time']
        end_time = row['end_time']
        phonetic = row['phonetic'].split()
        
        # Calculate estimated duration for each phoneme in the word
        if len(phonetic) > 0:
            phoneme_duration = (end_time - start_time) / len(phonetic)
            
            for i, phoneme in enumerate(phonetic):
                phoneme_start = start_time + i * phoneme_duration
                phoneme_end = phoneme_start + phoneme_duration
                phonemes_with_times.append({
                    'word': word,
                    'phoneme': phoneme,
                    'start_time': phoneme_start,
                    'end_time': phoneme_end
                })
    
    return pd.DataFrame(phonemes_with_times)

# Step 2: Extract facial landmarks from video
def extract_facial_landmarks(video_path, landmark_indices=None, fps=30):
    """
    Extract facial landmarks from video using MediaPipe.
    
    Parameters:
        video_path (str): Path to the video file
        landmark_indices (list or None): List of specific landmark indices to extract.
        If None, all 468 landmarks will be extracted.
        fps (int): Desired frames per second for processing
    
    Returns:
        tuple: (landmarks_array, timestamps)
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    # Initialize MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps
        
        print(f"Video duration: {duration:.2f} seconds, FPS: {video_fps}, Total frames: {total_frames}")
        
        # Container for landmarks
        all_landmarks = []
        timestamps = []
        
        # Determine how many coordinates we need to store
        total_landmarks = 468  # MediaPipe Face Mesh has 468 landmarks
        
        # If specific landmarks are requested, prepare to extract just those
        if landmark_indices is not None:
            # Ensure landmark_indices is a list
            if not isinstance(landmark_indices, list):
                landmark_indices = [landmark_indices]
            # Validate indices are in range
            landmark_indices = [i for i in landmark_indices if 0 <= i < total_landmarks]
            if not landmark_indices:
                print("No valid landmark indices provided. Using all landmarks.")
                landmark_indices = None
        
        frame_count = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Calculate timestamp for this frame
            timestamp = frame_count / video_fps
            timestamps.append(timestamp)
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image with MediaPipe
            results = face_mesh.process(image_rgb)
            
            # If face landmarks are detected, save them
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                if landmark_indices is None:
                    # Extract all landmark coordinates (x, y, z)
                    frame_landmarks = []
                    for landmark in face_landmarks.landmark:
                        frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                else:
                    # Extract only the specified landmark coordinates
                    frame_landmarks = []
                    for idx in landmark_indices:
                        landmark = face_landmarks.landmark[idx]
                        frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                all_landmarks.append(frame_landmarks)
            else:
                # If no face detected, add zeros
                if landmark_indices is None:
                    all_landmarks.append([0] * (total_landmarks * 3))  # 468 landmarks with x, y, z
                else:
                    all_landmarks.append([0] * (len(landmark_indices) * 3))  # Specified landmarks with x, y, z
            
            frame_count += 1
        
        cap.release()
        
        return np.array(all_landmarks), timestamps

# Step 3: Align phonemes with facial landmarks
def align_phonemes_with_landmarks(phonemes_df, landmarks, timestamps):
    """Align phonemes with corresponding facial landmarks based on timestamps."""
    aligned_data = []
    
    for _, phoneme_row in phonemes_df.iterrows():
        phoneme_start = phoneme_row['start_time']
        phoneme_end = phoneme_row['end_time']
        phoneme = phoneme_row['phoneme']
        
        # Find frames within the phoneme's time range
        frame_indices = [i for i, ts in enumerate(timestamps) 
                            if phoneme_start <= ts < phoneme_end]
        
        # Skip phonemes with no corresponding frames
        if not frame_indices:
            continue
        
        # Use the landmarks from the middle of the phoneme duration
        mid_frame_idx = frame_indices[len(frame_indices) // 2]
        aligned_data.append({
            'phoneme': phoneme,
            'landmarks': landmarks[mid_frame_idx]
        })
    
    return aligned_data

# Step 4: Prepare data for training
def prepare_training_data(aligned_data):
    """Prepare phoneme and landmark data for neural network training."""
    # Extract phonemes and create a vocabulary
    all_phonemes = [item['phoneme'] for item in aligned_data]
    unique_phonemes = sorted(set(all_phonemes))
    phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(unique_phonemes)}
    
    # Create one-hot encoded vectors for phonemes
    X = np.zeros((len(aligned_data), len(unique_phonemes)))
    for i, item in enumerate(aligned_data):
        X[i, phoneme_to_idx[item['phoneme']]] = 1
    
    # Extract landmarks
    y = np.array([item['landmarks'] for item in aligned_data])
    
    return X, y, phoneme_to_idx

# Step 5: Create a more advanced neural network model
def create_model(input_shape, output_shape):
    """Create a neural network model for phoneme-to-landmark prediction."""
    model = Sequential([
        Dense(512, activation='tanh', input_shape=(input_shape,)),
        Dropout(0.25),
        Dense(1024, activation='tanh'),
        Dropout(0.25),
        Dense(1024, activation='tanh'),
        Dropout(0.25),
        Dense(output_shape, activation='linear')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    
    return model

# Step 6: Create a sequence-based model for context-aware predictions
def create_sequence_model(num_phonemes, output_shape, seq_length=5):
    """Create a sequence-based model that considers phoneme context."""
    inputs = Input(shape=(seq_length, num_phonemes))
    
    x = Bidirectional(LSTM(256, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256))(x)
    x = Dropout(0.2)(x)
    
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(output_shape, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    
    return model

# Step 7: Prepare sequence data
def prepare_sequence_data(X, y, phoneme_to_idx, seq_length=5):
    """Prepare sequence data for training the LSTM model."""
    num_samples = len(X)
    num_phonemes = len(phoneme_to_idx)
    
    # Create empty arrays for sequences
    X_seq = np.zeros((num_samples - seq_length + 1, seq_length, num_phonemes))
    y_seq = np.zeros((num_samples - seq_length + 1, y.shape[1]))
    
    # Fill the arrays
    for i in range(num_samples - seq_length + 1):
        X_seq[i] = X[i:i+seq_length]
        y_seq[i] = y[i+seq_length-1]  # Predict landmarks for the last phoneme in sequence
    
    return X_seq, y_seq

# Step 8: Visualization functions
def visualize_landmarks(landmarks, image_shape=(427, 640)):
    """Visualize facial landmarks on a blank image with consistent scaling and centering."""
    # Create a blank image
    image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    
    # Reshape landmarks to (num_landmarks, 3)
    landmarks_reshaped = landmarks.reshape(-1, 3)
    
    # Skip if no valid landmarks
    if np.all(landmarks_reshaped == 0):
        return image
    
    # Filter out zero coordinates (which might indicate missing landmarks)
    valid_points = landmarks_reshaped[~np.all(landmarks_reshaped == 0, axis=1)]
    
    if len(valid_points) == 0:
        return image
        
    # Calculate the bounding box of valid landmarks
    min_x, min_y = np.min(valid_points[:, :2], axis=0)
    max_x, max_y = np.max(valid_points[:, :2], axis=0)
    
    # Calculate current width and height of the landmarks
    width = max_x - min_x
    height = max_y - min_y
    
    # Calculate aspect ratio of the face
    aspect_ratio = width / height if height > 0 else 1
    
    # Set target size to occupy 70% of the image but maintain aspect ratio
    target_width = 0.7 * image_shape[1]
    target_height = target_width / aspect_ratio  # Adjust height based on width to maintain aspect ratio
    
    # If the target height is too large, adjust both dimensions
    if target_height > 0.7 * image_shape[0]:
        target_height = 0.7 * image_shape[0]
        target_width = target_height * aspect_ratio
    
    # Determine the scaling factor
    scale_x = target_width / width if width > 0 else 1
    scale_y = target_height / height if height > 0 else 1
    
    # Use the same scale for both dimensions to maintain aspect ratio
    scale = min(scale_x, scale_y)
    
    # Calculate the center of the landmarks
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Calculate the target center (center of image)
    target_center_x = image_shape[1] / 2
    target_center_y = image_shape[0] / 2
    
    # Plot each landmark
    for i, (x, y, z) in enumerate(landmarks_reshaped):
        if x == 0 and y == 0 and z == 0:
            continue  # Skip zero landmarks
            
        # Scale the point relative to the landmark center
        scaled_x = ((x - center_x) * scale) + target_center_x
        scaled_y = ((y - center_y) * scale) + target_center_y
        
        # Convert to integer coordinates for drawing
        image_x = int(scaled_x)
        image_y = int(scaled_y)
        
        # Make sure the point is within image boundaries
        if 0 <= image_x < image_shape[1] and 0 <= image_y < image_shape[0]:
            # Draw the landmark point
            cv2.circle(image, (image_x, image_y), 1, (0, 255, 0), -1)
    
    return image

# def visualize_landmarks(landmarks, image_shape=(480, 640)):
#     """
#     Visualize facial landmarks with connected polygons and gradients.
    
#     Parameters:
#         landmarks (np.array): Array of facial landmarks
#         image_shape (tuple): Dimensions of the output image
    
#     Returns:
#         np.array: Visualized image with facial landmarks
#     """
#     # Create a blank image with white background
#     image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
#     image.fill(255)  # White background
    
#     # Reshape landmarks to (num_landmarks, 3)
#     landmarks_reshaped = landmarks.reshape(-1, 3)
    
#     # Skip if no valid landmarks
#     if np.all(landmarks_reshaped == 0):
#         return image
    
#     # Filter out zero coordinates (which might indicate missing landmarks)
#     valid_points = landmarks_reshaped[~np.all(landmarks_reshaped == 0, axis=1)]
    
#     if len(valid_points) == 0:
#         return image
        
#     # Calculate the bounding box of valid landmarks
#     min_x, min_y = np.min(valid_points[:, :2], axis=0)
#     max_x, max_y = np.max(valid_points[:, :2], axis=0)
    
#     # Calculate current width and height of the landmarks
#     width = max_x - min_x
#     height = max_y - min_y
    
#     # Calculate aspect ratio of the face
#     aspect_ratio = width / height if height > 0 else 1
    
#     # Set target size to occupy 70% of the image but maintain aspect ratio
#     target_width = 0.7 * image_shape[1]
#     target_height = target_width / aspect_ratio
    
#     # If the target height is too large, adjust both dimensions
#     if target_height > 0.7 * image_shape[0]:
#         target_height = 0.7 * image_shape[0]
#         target_width = target_height * aspect_ratio
    
#     # Determine the scaling factor
#     scale_x = target_width / width if width > 0 else 1
#     scale_y = target_height / height if height > 0 else 1
    
#     # Use the same scale for both dimensions to maintain aspect ratio
#     scale = min(scale_x, scale_y)
    
#     # Calculate the center of the landmarks
#     center_x = (min_x + max_x) / 2
#     center_y = (min_y + max_y) / 2
    
#     # Calculate the target center (center of image)
#     target_center_x = image_shape[1] / 2
#     target_center_y = image_shape[0] / 2
    
#     # Prepare lists to store scaled points
#     scaled_points = []
    
#     # Scale and convert landmarks to image coordinates
#     for i, (x, y, z) in enumerate(landmarks_reshaped):
#         if x == 0 and y == 0 and z == 0:
#             continue  # Skip zero landmarks
            
#         # Scale the point relative to the landmark center
#         scaled_x = ((x - center_x) * scale) + target_center_x
#         scaled_y = ((y - center_y) * scale) + target_center_y
        
#         # Convert to integer coordinates
#         image_x = int(scaled_x)
#         image_y = int(scaled_y)
        
#         # Make sure the point is within image boundaries
#         if 0 <= image_x < image_shape[1] and 0 <= image_y < image_shape[0]:
#             scaled_points.append((image_x, image_y))
    
#     # Define key face regions with landmark indices
#     face_regions = [
#         # Contour of the face
#         list(range(0, 17)),  # Jaw line
#         list(range(17, 22)),  # Left eyebrow
#         list(range(22, 27)),  # Right eyebrow
#         list(range(36, 42)),  # Left eye
#         list(range(42, 48)),  # Right eye
#         list(range(48, 60)),  # Outer lips
#         list(range(60, 68))   # Inner lips
#     ]
    
#     # Create gradient colors for different regions
#     region_colors = [
#         (173, 216, 230),  # Light blue for face contour
#         (255, 192, 203),  # Pink for eyebrows
#         (173, 255, 47),   # Green-yellow for eyes
#         (255, 160, 122)   # Light salmon for lips
#     ]
    
#     # Draw polygons for each region
#     for i, region in enumerate(face_regions):
#         # Ensure we have enough points in the region
#         region_points = [scaled_points[idx] for idx in region if idx < len(scaled_points)]
        
#         if len(region_points) > 2:
#             # Create a gradient overlay
#             color = region_colors[i % len(region_colors)]
            
#             # Create a temporary image for this region
#             region_mask = np.zeros_like(image)
            
#             # Draw filled polygon
#             cv2.fillPoly(region_mask, [np.array(region_points)], color)
            
#             # Apply soft alpha blending
#             alpha = 0.3  # Transparency factor
#             cv2.addWeighted(region_mask, alpha, image, 1 - alpha, 0, image)
    
#     return image

def compare_landmarks(true_landmarks, pred_landmarks, image_shape=(480, 640)):
    """Compare true and predicted landmarks on an image."""
    # Create a blank image
    image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    
    # Reshape landmarks
    true_reshaped = true_landmarks.reshape(-1, 3)
    pred_reshaped = pred_landmarks.reshape(-1, 3)
    
    # Get all valid points
    all_points = np.vstack([true_reshaped, pred_reshaped])
    valid_points = all_points[~np.all(all_points == 0, axis=1)]
    
    if len(valid_points) == 0:
        return image
        
    # Calculate the bounding box of all landmarks
    min_x, min_y = np.min(valid_points[:, :2], axis=0)
    max_x, max_y = np.max(valid_points[:, :2], axis=0)
    
    # Calculate current width and height
    width = max_x - min_x
    height = max_y - min_y
    
    # Calculate aspect ratio
    aspect_ratio = width / height if height > 0 else 1
    
    # Set target size to occupy 70% of the image but maintain aspect ratio
    target_width = 0.7 * image_shape[1]
    target_height = target_width / aspect_ratio
    
    # If the target height is too large, adjust both dimensions
    if target_height > 0.7 * image_shape[0]:
        target_height = 0.7 * image_shape[0]
        target_width = target_height * aspect_ratio
    
    # Determine the scaling factor
    scale_x = target_width / width if width > 0 else 1
    scale_y = target_height / height if height > 0 else 1
    scale = min(scale_x, scale_y)
    
    # Calculate the center of all landmarks
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Calculate the target center (center of image)
    target_center_x = image_shape[1] / 2
    target_center_y = image_shape[0] / 2
    
    # Plot each landmark pair
    for i in range(len(true_reshaped)):
        # True landmarks in green
        if not np.all(true_reshaped[i] == 0):
            true_x = ((true_reshaped[i, 0] - center_x) * scale) + target_center_x
            true_y = ((true_reshaped[i, 1] - center_y) * scale) + target_center_y
            
            image_x = int(true_x)
            image_y = int(true_y)
            
            if 0 <= image_x < image_shape[1] and 0 <= image_y < image_shape[0]:
                cv2.circle(image, (image_x, image_y), 1, (0, 255, 0), -1)
        
        # Predicted landmarks in blue
        if not np.all(pred_reshaped[i] == 0):
            pred_x = ((pred_reshaped[i, 0] - center_x) * scale) + target_center_x
            pred_y = ((pred_reshaped[i, 1] - center_y) * scale) + target_center_y
            
            image_x = int(pred_x)
            image_y = int(true_y)
            
            if 0 <= image_x < image_shape[1] and 0 <= image_y < image_shape[0]:
                cv2.circle(image, (image_x, image_y), 1, (255, 0, 0), -1)
    
    return image

# def compare_landmarks(true_landmarks, pred_landmarks, image_shape=(480, 640)):
#     """Compare true and predicted landmarks on an image."""
#     # Create a blank image
#     image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    
#     # Reshape landmarks
#     true_reshaped = true_landmarks.reshape(-1, 3)
#     pred_reshaped = pred_landmarks.reshape(-1, 3)
    
#     # Plot each landmark pair
#     for i in range(len(true_reshaped)):
#         # True landmarks in green
#         true_x = int(true_reshaped[i, 0] * image_shape[1])
#         true_y = int(true_reshaped[i, 1] * image_shape[0])
#         cv2.circle(image, (true_x, true_y), 1, (0, 255, 0), -1)
        
#         # Predicted landmarks in blue
#         pred_x = int(pred_reshaped[i, 0] * image_shape[1])
#         pred_y = int(pred_reshaped[i, 1] * image_shape[0])
#         cv2.circle(image, (pred_x, pred_y), 1, (255, 0, 0), -1)
    
#     return image

# def combine_landmarks(original_landmarks, predicted_landmarks, landmark_points, landmark_points_to_measure):
#     """
#     Combine predicted landmarks and original landmarks into a complete 468-point face mesh.
    
#     Parameters:
#         original_landmarks (np.array): Original landmarks array (from extract_facial_landmarks)
#         predicted_landmarks (np.array): Predicted landmarks array (from model prediction)
#         landmark_points (list): List of indices for points in original_landmarks
#         landmark_points_to_measure (list): List of indices for points in predicted_landmarks
    
#     Returns:
#         np.array: Combined landmarks array with all 468 points in proper order
#     """
#     # Get the number of samples
#     num_samples = predicted_landmarks.shape[0]
    
#     # Create empty arrays for the combined landmarks
#     # Each sample will have 468 landmarks with x, y, z coordinates
#     combined_landmarks = np.zeros((num_samples, 468 * 3))
    
#     # For each sample in the prediction
#     for sample_idx in range(num_samples):
#         # Get the current sample's predicted landmarks
#         current_pred = predicted_landmarks[sample_idx]
        
#         # Get corresponding sample from original landmarks
#         # This assumes alignment between predicted and original samples
#         current_orig = original_landmarks[sample_idx] if sample_idx < len(original_landmarks) else np.zeros_like(original_landmarks[0])
        
#         # Create a temporary array to hold all landmarks for this sample
#         full_landmarks = np.zeros(468 * 3)
        
#         # First, fill in the predicted landmarks
#         for i, point_idx in enumerate(landmark_points_to_measure):
#             # Each landmark has x, y, z coordinates
#             full_idx = point_idx * 3
#             pred_idx = i * 3
            
#             # Copy the x, y, z values
#             full_landmarks[full_idx:full_idx+3] = current_pred[pred_idx:pred_idx+3]
        
#         # Then, fill in the original landmarks
#         for i, point_idx in enumerate(landmark_points):
#             # Each landmark has x, y, z coordinates
#             full_idx = point_idx * 3
#             orig_idx = i * 3
            
#             # Copy the x, y, z values
#             full_landmarks[full_idx:full_idx+3] = current_orig[orig_idx:orig_idx+3]
        
#         # Store the full landmarks for this sample
#         combined_landmarks[sample_idx] = full_landmarks
    
#     return combined_landmarks

def combine_landmarks(original_landmarks, predicted_landmarks, landmark_points, landmark_points_to_measure):
    """
    Combine predicted landmarks and original landmarks into a complete 468-point face mesh.
    
    Parameters:
        original_landmarks (np.array): Original landmarks array (from extract_facial_landmarks)
        predicted_landmarks (np.array): Predicted landmarks array (from model prediction)
        landmark_points (list): List of indices for points in original_landmarks
        landmark_points_to_measure (list): List of indices for points in predicted_landmarks
    
    Returns:
        np.array: Combined landmarks array with all 468 points in proper order
    """
    # Print shapes and sizes for debugging
    print(f"Original landmarks shape: {original_landmarks.shape}")
    print(f"Predicted landmarks shape: {predicted_landmarks.shape}")
    print(f"Number of landmark_points: {len(landmark_points)}")
    print(f"Number of landmark_points_to_measure: {len(landmark_points_to_measure)}")
    
    # Get the number of samples
    num_samples = predicted_landmarks.shape[0]
    
    # Create empty arrays for the combined landmarks
    # Each sample will have 468 landmarks with x, y, z coordinates
    combined_landmarks = np.zeros((num_samples, 468 * 3))
    
    # Reshape predicted landmarks if needed
    # If the predicted landmarks are flattened (one dimension per sample)
    if len(predicted_landmarks.shape) == 2 and predicted_landmarks.shape[1] % 3 == 0:
        # Calculate how many landmarks are in the predictions
        num_pred_landmarks = predicted_landmarks.shape[1] // 3
        
        # Check if this matches our expected number
        if num_pred_landmarks != len(landmark_points_to_measure):
            print(f"Warning: Predicted landmarks ({num_pred_landmarks}) doesn't match landmark_points_to_measure ({len(landmark_points_to_measure)})")
    
    # Ensure original_landmarks is properly shaped
    if isinstance(original_landmarks, np.ndarray) and len(original_landmarks.shape) == 1:
        # If it's a flat array, reshape it to have 3 coordinates per landmark
        original_landmarks = original_landmarks.reshape(-1, 3)
    
    # Ensure we have enough landmarks in original_landmarks
    if len(original_landmarks) < len(landmark_points):
        print(f"Warning: Not enough landmarks in original_landmarks ({len(original_landmarks)}) compared to landmark_points ({len(landmark_points)})")
        # Pad with zeros if needed
        padding = np.zeros((len(landmark_points) - len(original_landmarks), 3))
        original_landmarks = np.vstack([original_landmarks, padding])
    
    # For each sample in the prediction
    for sample_idx in range(num_samples):
        # Get the current sample's predicted landmarks
        current_pred = predicted_landmarks[sample_idx]
        
        # Create a temporary array to hold all landmarks for this sample
        full_landmarks = np.zeros(468 * 3)
        
        # First, fill in the predicted landmarks
        for i, point_idx in enumerate(landmark_points_to_measure):
            if i * 3 + 2 < len(current_pred):  # Make sure we have enough data
                # Each landmark has x, y, z coordinates
                full_idx = point_idx * 3
                pred_idx = i * 3
                
                # Copy the x, y, z values
                full_landmarks[full_idx:full_idx+3] = current_pred[pred_idx:pred_idx+3]
        
        # Then, fill in the original landmarks
        for i, point_idx in enumerate(landmark_points):
            if i < len(original_landmarks):  # Make sure we have enough data
                # Each landmark has x, y, z coordinates
                full_idx = point_idx * 3
                
                # Copy the x, y, z values
                full_landmarks[full_idx:full_idx+3] = original_landmarks[i]
        
        # Store the full landmarks for this sample
        combined_landmarks[sample_idx] = full_landmarks
    
    return combined_landmarks

# Step 9: Main function to orchestrate the process
def main(csv_path, video_path):
    """Main function to train and evaluate the model."""
    # Load and preprocess phonetic data
    print("Loading phonetic data...")
    phonemes_df = load_and_preprocess_phonetics(csv_path)
    
    # landmark_points_to_measure = [13, 14, 17, 46, 52, 53, 55, 58, 61, 63, 65, 66, 70, 78, 80, 81, 82, 84, 
    #                                 87, 88, 91, 95, 105, 107, 135, 136, 138, 140, 146, 148, 149, 150, 152,
    #                                 169, 170, 171, 172, 175, 176, 178, 181, 191, 192, 214, 215, 276, 282,
    #                                 283, 285, 288, 291, 293, 295, 296, 300, 308, 310, 311, 312, 314, 317,
    #                                 318, 321, 324, 334, 336, 364, 365, 367, 369, 375, 377, 378, 379, 394,
    #                                 395, 396, 397, 400, 402, 405, 415, 416, 434, 435]
    
    # landmark_points = []
    # for point in range(468):
    #     if point not in landmark_points_to_measure:
    #         landmark_points.append(point)
    
    # Extract facial landmarks from video
    print("Extracting facial landmarks from video...")
    landmarks, timestamps = extract_facial_landmarks(video_path)
    # landmarks_filtered, timestamps = extract_facial_landmarks(video_path, landmark_indices=landmark_points_to_measure)
    
    # Align phonemes with landmarks
    print("Aligning phonemes with landmarks...")
    aligned_data = align_phonemes_with_landmarks(phonemes_df, landmarks, timestamps)
    
    # Prepare data for training
    print("Preparing training data...")
    X, y, phoneme_to_idx = prepare_training_data(aligned_data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the simple model
    print("Training the simple model...")
    simple_model = create_model(X.shape[1], y.shape[1])
    simple_model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Prepare sequence data
    print("Preparing sequence data...")
    X_seq, y_seq = prepare_sequence_data(X, y, phoneme_to_idx)
    
    # Split sequence data
    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    
    # Create and train the sequence model
    print("Training the sequence model...")
    seq_model = create_sequence_model(X.shape[1], y.shape[1])
    seq_model.fit(
        X_seq_train, y_seq_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_seq_test, y_seq_test),
        verbose=1
    )
    
    # Evaluate models
    print("Evaluating models...")
    simple_loss = simple_model.evaluate(X_test, y_test)
    
    # # Use the first seq_length samples from test set
    seq_test_samples = min(len(X_seq_test), len(X_test) - 4)
    seq_loss = seq_model.evaluate(X_seq_test[:seq_test_samples], y_seq_test[:seq_test_samples])
    
    print(f"Simple model loss: {simple_loss}")
    print(f"Sequence model loss: {seq_loss}")
    
    # Make predictions
    simple_preds = simple_model.predict(X_test)
    # simple_preds = seq_model.predict(X_seq_test)
    
    # # Get the number of samples to process
    # num_samples = min(len(simple_preds), len(landmarks))
    
    # # Extract landmarks for the same samples
    # landmarks_for_samples = landmarks[:num_samples]
    
    # # Combine the landmarks
    # print("Combining landmarks...")
    # combined_landmarks = combine_landmarks(
    #     landmarks_for_samples,
    #     simple_preds[:num_samples],
    #     landmark_points,
    #     landmark_points_to_measure
    # )
    
    # print(f"Shape of combined landmarks: {combined_landmarks.shape}")
    # print(f"Number of landmark points: {combined_landmarks.shape[1]/3}")
    
    # # Save the combined landmarks
    # np.save("combined_landmarks.npy", combined_landmarks)
    # print(combined_landmarks)
    
    # Visualize some results
    print("Visualizing results...")
    for i in range(min(5, len(X_test))):
        true_image = visualize_landmarks(y_test[i])
        pred_image = visualize_landmarks(simple_preds[i])
        compare_image = compare_landmarks(y_test[i], simple_preds[i])
        
        # Display the images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(true_image)
        axes[0].set_title("True Landmarks")
        axes[1].imshow(pred_image)
        axes[1].set_title("Predicted Landmarks")
        axes[2].imshow(compare_image)
        axes[2].set_title("Comparison (Green=True, Blue=Pred)")
        
        plt.tight_layout()
        # plt.savefig(f"comparison_{i}.png")
        plt.close()
    
    # Save models
    simple_model.save("temporary_usage_files/phoneme_to_landmark_simple.h5")
    # seq_model.save("temporary_usage_files/phoneme_to_landmark_sequence.h5")
    
    # Save the phoneme mapping for inference
    with open("temporary_usage_files/phoneme_mapping.txt", "w") as f:
        for phoneme, idx in phoneme_to_idx.items():
            f.write(f"{phoneme},{idx}\n")
    
    print("Training and evaluation complete!")

# Step 10: Function for inference on new text
def predict_landmarks_for_text(text, phoneme_to_idx, model, phoneme_dict):
    """Predict facial landmarks for new text."""
    # This function would use a pronunciation dictionary to convert text to phonemes
    # Then use the trained model to predict landmarks
    # For simplicity, we'll assume we have a function to convert text to phonemes
    
    # This is a placeholder - you would need a proper text-to-phoneme converter
    phonemes = text_to_phonemes(text, phoneme_dict)
    
    # Convert phonemes to one-hot encoded vectors
    X_new = np.zeros((len(phonemes), len(phoneme_to_idx)))
    for i, phoneme in enumerate(phonemes):
        if phoneme in phoneme_to_idx:
            X_new[i, phoneme_to_idx[phoneme]] = 1
    
    # Make predictions
    predictions = model.predict(X_new)
    
    return predictions

# Helper function for text to phoneme conversion
def text_to_phonemes(text, phoneme_dict):
    """Convert text to phonemes using a pronunciation dictionary."""
    # This is a simplified version - you would need a proper dictionary
    words = text.lower().split()
    phonemes = []
    
    for word in words:
        if word in phoneme_dict:
            phonemes.extend(phoneme_dict[word].split())
        else:
            # Fallback for unknown words
            phonemes.extend(['UNK'])
    
    return phonemes

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    csv_path = "temporary_usage_files/aligned_output.csv"
    video_path = "input_video.mp4"
    
    main(csv_path, video_path)
