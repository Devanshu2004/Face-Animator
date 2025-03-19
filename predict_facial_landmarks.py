# Import necessary libraries from both files
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from g2p import make_g2p

# Import functions from temp.py
from generate_model import visualize_landmarks, text_to_phonemes, combine_landmarks


# # Function to extract facial landmarks from the input image for reference
# def extract_facial_landmarks(image_path, landmark_indices=None):
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

#     image = cv2.imread(image_path)

#     # Convert the image to RGB (MediaPipe requires RGB images)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Process the image to detect facial landmarks
#     results = face_mesh.process(rgb_image)

#     all_landmarks = []
#     total_landmarks = 468

#     # If specific landmarks are requested, prepare to extract just those
#     if landmark_indices is not None:
#         # Ensure landmark_indices is a list
#         if not isinstance(landmark_indices, list):
#             landmark_indices = [landmark_indices]
#         # Validate indices are in range
#         landmark_indices = [i for i in landmark_indices if 0 <= i < total_landmarks]
#         if not landmark_indices:
#             print("No valid landmark indices provided. Using all landmarks.")
#             landmark_indices = None
    
#     # Check if landmarks are detected
#     if results.multi_face_landmarks:
#         face_landmarks = results.multi_face_landmarks[0]
        
#         # Extract only the specified landmark coordinates
#         frame_landmarks = []
#         for idx in landmark_indices:
#             landmark = face_landmarks.landmark[idx]
#             frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
#         all_landmarks.extend(frame_landmarks)
    
#     return np.array(all_landmarks)

# Function from trial.py
def text_to_phonetic(text):
    # Create a grapheme-to-phoneme converter for English to ARPABET
    transducer = make_g2p('eng', 'eng-arpabet')
    
    # Convert the text to its phonetic equivalent
    phonetic_text = transducer(text).output_string
    
    return phonetic_text

# Modified text_to_phonemes function that uses the g2p conversion
def text_to_phonemes_g2p(text):
    # Get ARPABET phonemes using the g2p converter
    phonetic_text = text_to_phonetic(text)
    
    # Split the phonetic text into individual phonemes
    # Note: ARPABET phonemes might need formatting to match your model's expected format
    phonemes = phonetic_text.split()
    
    return phonemes

# Function to prepare phoneme input for the model
def prepare_phoneme_input(phonemes, phoneme_to_idx):
    # Convert phonemes to one-hot encoded vectors
    X_new = np.zeros((len(phonemes), len(phoneme_to_idx)))
    for i, phoneme in enumerate(phonemes):
        # Strip any numbers (stress markers) from phonemes if present
        clean_phoneme = ''.join([c for c in phoneme if not c.isdigit()])
        
        if clean_phoneme in phoneme_to_idx:
            X_new[i, phoneme_to_idx[clean_phoneme]] = 1
        else:
            # Handle unknown phonemes
            print(f"Warning: Phoneme '{clean_phoneme}' not in training set")
    
    return X_new

# Main function to integrate everything
def predict_landmarks_from_text_input():
    
    # Load the saved model
    model_path = "temporary_usage_files/phoneme_to_landmark_simple.h5"  # or sequence model
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    model = load_model(model_path)
    print("Model loaded successfully.")
    
    # Load the phoneme mapping
    phoneme_mapping_path = "temporary_usage_files/phoneme_mapping.txt"
    if not os.path.exists(phoneme_mapping_path):
        print(f"Error: Phoneme mapping file '{phoneme_mapping_path}' not found.")
        return
    
    phoneme_to_idx = {}
    with open(phoneme_mapping_path, "r") as f:
        for line in f:
            if ',' in line:
                phoneme, idx = line.strip().split(',')
                phoneme_to_idx[phoneme] = int(idx)
    
    print(f"Loaded {len(phoneme_to_idx)} phonemes from mapping file.")
    
    # Get user input
    text = "Hi, I am your current AI assistant, and I am willing to help you with your extremely auspicious project. Let me know, if you need my help."
    
    # Convert to phonemes using g2p
    phonemes = text_to_phonemes_g2p(text)
    print("Converted to phonemes:", phonemes)
    
    # Prepare input for model
    X_new = prepare_phoneme_input(phonemes, phoneme_to_idx)
    
    # Check if we have any valid phonemes to predict
    if np.sum(X_new) == 0:
        print("Error: None of the phonemes in the input text match the training data.")
        return
    
    # Make predictions
    print("Predicting facial landmarks...")
    predictions = model.predict(X_new)
    
    # Visualize the predicted landmarks
    print("Generating visualizations...")
    output_dir = "predicted_landmarks"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, landmark_set in enumerate(predictions):
        image = visualize_landmarks(landmark_set)
        
        # Save the visualization
        output_path = os.path.join(output_dir, f"{i}.png")
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"Predicted landmarks for phoneme: {phonemes[i]}")
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved visualization for phoneme {phonemes[i]} to {output_path}")
    
    print(f"Generated {len(predictions)} landmark visualizations in {output_dir}/")


if __name__ == "__main__":
    predict_landmarks_from_text_input()
