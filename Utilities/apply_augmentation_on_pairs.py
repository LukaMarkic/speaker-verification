import os
import numpy as np
import pandas as pd
import random
from skimage.transform import resize

def apply_random_gaussian_noise(spectrogram, mean=0.0, std=1.5):
    noise = np.random.normal(mean, std, spectrogram.shape)
    return spectrogram + noise

def apply_amplitude_scaling(spectrogram, scale_factor=1.35):
    return spectrogram * scale_factor

def apply_random_amplitude_scaling(spectrogram, min_scale=0.9, max_scale=1.3):
    scale_factor = np.random.uniform(min_scale, max_scale)
    return spectrogram * scale_factor

def augment_and_replace(train_files, augmentation_type, augmentation_percentage):
    # Calculate the number of files to augment
    num_files_to_augment = int(len(train_files) * augmentation_percentage)
    
    # Randomly select files to augment
    files_to_augment = train_files if augmentation_percentage == 1 else random.sample(train_files, num_files_to_augment)

    total_files = len(files_to_augment)
    completed_files = 0

    for pair_file in files_to_augment:
        pair_file_path = os.path.join(image_dir, pair_file)
        # Check if file exists and is not empty
        if not os.path.exists(pair_file_path) or os.path.getsize(pair_file_path) == 0:
            print(f"Skipping {pair_file} as it does not exist or is empty.")
            continue
        
        try:
            # Load the combined spectrogram pair
            combined_spectrograms = np.load(pair_file_path)
        except Exception as e:
            print(f"Error loading {pair_file}: {e}")
            continue
        
        if combined_spectrograms.shape[0] != 2:
            print(f"Skipping {pair_file} as it does not contain two spectrograms.")
            continue
        
        spectrogram1 = combined_spectrograms[0]
        spectrogram2 = combined_spectrograms[1]
        
        # Ensure spectrograms are resized to (100, 50) to match the dimensions used in generate spectrograms
        spectrogram1 = resize(spectrogram1, (100, 50))
        spectrogram2 = resize(spectrogram2, (100, 50))
        
        # Apply the chosen augmentation
        if augmentation_type == 'gaussian_noise':
            spectrogram1 = apply_random_gaussian_noise(spectrogram1)
            spectrogram2 = apply_random_gaussian_noise(spectrogram2)
        elif augmentation_type == 'amplitude_scaling':
            spectrogram1 = apply_amplitude_scaling(spectrogram1)
            spectrogram2 = apply_amplitude_scaling(spectrogram2)
        elif augmentation_type == 'random_amplitude_scaling':
            spectrogram1 = apply_random_amplitude_scaling(spectrogram1)
            spectrogram2 = apply_random_amplitude_scaling(spectrogram2)
        
        # Combine the augmented pair back into a single array
        augmented_combined_spectrograms = np.array([spectrogram1, spectrogram2])
        
        # Save the augmented pair (replace the original pair)
        np.save(pair_file_path, augmented_combined_spectrograms)
        
        # Update progress
        completed_files += 1
        if completed_files % (total_files // 10) == 0:
            print(f"Progress: {100 * completed_files // total_files}% completed")

# Directory containing numpy image files
image_dir = "Spectrogram Pairs TRAIN"

# Load the split distribution from the CSV file
df = pd.read_csv('split_distribution_basic.csv')
train_files = df['train_files'].dropna().tolist()


# Choose augmentation type: 'gaussian_noise', 'amplitude_scaling', or 'random_amplitude_scaling'
augmentation_type = 'random_amplitude_scaling'  # Change this as desired

# Set the percentage of training files to augment (e.g., 0.5 for 50%)
augmentation_percentage = 1  # Change this as desired

# Apply augmentation and replace training pairs
augment_and_replace(train_files, augmentation_type, augmentation_percentage)
print("Augmentation has been applied")
