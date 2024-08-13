import os
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to split the dataset into training, validation, and testing sets
def get_split(image_files):
    positive_pairs = []
    negative_pairs = []

    # Iterate through all combinations of file pairs
    for pair in image_files:
        pair = pair.split('/')[-1]
        file1, file2 = pair.split('-')
        if file1[:8] == file2[:8]:
            positive_pairs.append(pair)
        else:
            negative_pairs.append(pair)

    # Split positive and negative pairs into training, validation, and testing sets
    train_positive, remaining_positive = train_test_split(positive_pairs, test_size=0.3, random_state=42)
    val_positive, test_positive = train_test_split(remaining_positive, test_size=2 / 3, random_state=42)

    train_negative, remaining_negative = train_test_split(negative_pairs, test_size=0.3, random_state=42)
    val_negative, test_negative = train_test_split(remaining_negative, test_size=2 / 3, random_state=42)

    # Combine positive and negative pairs for training, validation, and testing
    train_files = train_positive + train_negative
    val_files = val_positive + val_negative
    test_files = test_positive + test_negative

    # Shuffle the datasets
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    return train_files, val_files, test_files

# Directory containing numpy image files
image_dir = "Spectrogram Pairs"

# List all image files
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.npy')]
# Split dataset into training, validation, and testing sets
train_files, val_files, test_files = get_split(image_files)

# Save the split distribution to a CSV file
split_distribution = {
    'train_files': train_files,
    'val_files': val_files,
    'test_files': test_files
}
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in split_distribution.items()]))
df.to_csv('split_distribution_basic.csv', index=False)
