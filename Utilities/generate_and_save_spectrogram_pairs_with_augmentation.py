import os
import numpy as np
import librosa
import random
import pandas as pd
from skimage.transform import resize


def get_random_elements(array_elements, number_of_elements):
    random_sample = []
    if array_elements:
        sample_size = min(len(array_elements), number_of_elements)
        random_sample = random.sample(array_elements, sample_size)
    return random_sample

def apply_random_gaussian_noise_to_audio(audio, mean=0.0, min_std=0.0003, max_std=0.002):
    std = np.random.uniform(min_std, max_std)
    noise = np.random.normal(mean, std, audio.shape).astype(np.float32)  # Podesavanje s tipa podataka flaot64 na float32 kako bi se uskladile velicine
    noisy_audio = audio + noise
    return noisy_audio, std

def apply_random_amplitude_scaling_to_spectrogram(spectrogram, min_scale=0.9, max_scale=1.2):
    scale_factor = np.random.uniform(min_scale, max_scale)
    scaled_spectrogram = spectrogram / scale_factor
    scaled_spectrogram = np.clip(scaled_spectrogram, -80, 0)
    return scaled_spectrogram


def preprocess_audio(audio_file, sr=16000, target_duration=0.82, augmentation=None):
    y, sr = librosa.load(audio_file, sr=sr)

    # Use voice activity detection (VAD) to find speech segments
    vad_segments = librosa.effects.split(y, top_db=29)

    # Find the boundaries of the first three words
    word_boundaries = []
    word_count = 0
    for segment in vad_segments:
        if word_count >= 3:
            break
        if segment[1] - segment[0] > 0.16 * sr:  # Ensure segment duration is reasonable
            word_boundaries.append(segment)
            word_count += 1

    # Combine the segments corresponding to the first three words
    combined_segment = np.concatenate([y[boundary[0]:boundary[1]] for boundary in word_boundaries], axis=0)

    # Pad or trim combined segment to target duration
    target_samples = int(target_duration * sr)
    combined_segment = combined_segment[:target_samples]

    avg_std = 0

    # Apply augmentations based on parameters
    if augmentation == "gaussian":
        combined_segment, std = apply_random_gaussian_noise_to_audio(combined_segment)
        avg_std = std
    
    # Compute spectrogram for the augmented audio
    spectrogram = librosa.feature.melspectrogram(
        y=combined_segment, sr=sr, n_fft=512, hop_length=256, n_mels=128
    )
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Apply amplitude scaling to the spectrogram
    if augmentation == "amplitude":
        log_spectrogram = apply_random_amplitude_scaling_to_spectrogram(log_spectrogram)

    resized_spectrogram = resize(log_spectrogram, (100, 50))
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=-1)

    return resized_spectrogram, avg_std


def delete_files_in_directory(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Get a list of all files in the directory
    files = os.listdir(directory)

    for file in files:
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"Skipped non-file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def generate_data_for_saving(pairs, spectrogram_dict, output_directory='./Spectrogram Pairs'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for pair in pairs:
        #pair = pair.split('.npy')[0]
        #file1, file2 = pair.split('-')
        cleaned_pair = pair.strip("()").replace("'", "")
        # Split the string by the comma
        file1, file2 = cleaned_pair.split(", ")
        spectrogram_1 = spectrogram_dict[file1]
        spectrogram_2 = spectrogram_dict[file2]
        np.save(os.path.join(output_directory, f"{file1}-{file2}.npy"), [spectrogram_1[:, :, 0], spectrogram_2[:, :, 0]])

    print("All pairs augmented.")


audio_dir = "./GRID/TRAIN"
audio_files = [os.path.join("./GRID/TRAIN", f) for f in os.listdir("./GRID/TRAIN") if f.endswith('.wav')]

spectrogram_dict = {}
std_values = []
for audio_file in audio_files:
    # Example usage with Gaussian noise augmentation
    spectrogram, std = preprocess_audio(audio_file, augmentation="amplitude")
    if std != 0:
        std_values.append(std)
    name = os.path.basename(audio_file)
    name = name.split(".wav")[0]
    spectrogram_dict[name] = spectrogram

average_std = np.mean(std_values)
print(f"Average standard deviation of Gaussian noise applied: {average_std}")

df = pd.read_csv("pairs_with_labels_TRAIN.csv")
all_pairs =  df['Pairs'].dropna().tolist()

#df = pd.read_csv('split_distribution_basic.csv')
#all_pairs = df['train_files'].dropna().tolist()
#all_pairs = [os.path.join("Spectrogram Pairs TRAIN", f) for f in os.listdir("Spectrogram Pairs TRAIN") if f.endswith('.npy')]
print(len(all_pairs))
#delete_files_in_directory("Spectrogram Pairs TRAIN")

# Save spectrogram pairs
generate_data_for_saving(all_pairs, spectrogram_dict, output_directory="Spectrogram Pairs TRAIN")
