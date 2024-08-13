import os
import numpy as np
import librosa
import random
import pandas as pd
from skimage.transform import resize

def create_pairs_with_labels(audio_files):
    # Step 1: Create unique positive pairs for every audio class
    positive_pairs = []
    class_positive_counts = {}

    # Generate positive pairs for each class
    for file in audio_files:
        class_name = file[:8]
        if class_name not in class_positive_counts:
            class_positive_counts[class_name] = 0

        for other_file in audio_files:
            if file != other_file and other_file[:8] == class_name:
                # Ensure only one pair is added for each unique pair of files
                if (file, other_file) not in positive_pairs and (other_file, file) not in positive_pairs:
                    positive_pairs.append((file, other_file))
                    class_positive_counts[class_name] += 1

    # Step 2: Create negative pairs for each audio class
    negative_pairs = []
    used_count = {}  # Track the number of negative pairs added for each class

    for class_name in class_positive_counts.keys():
        positive_count = class_positive_counts[class_name]
        used_speaker = []
        # Select 20% of other songs of the same speaker
        same_class_speaker_files = [file for file in audio_files if file[:8] == class_name]
        other_speaker_songs = [file for file in audio_files if file[:4] == class_name[:4] and file not in same_class_speaker_files]
        other_songs_count = max(1, int(positive_count * 0.2))  # Ensure at least 1 song is selected
        first_other_speaker_songs = random.sample(other_speaker_songs, min(other_songs_count, len(other_speaker_songs)))
        second_other_songs_count = [file for file in other_speaker_songs if file[:7] == class_name[:7]and file not in same_class_speaker_files and file not in first_other_speaker_songs]
        second_other_songs_count = get_random_elements(second_other_songs_count, 3)
                                  
        for other_file in first_other_speaker_songs:
            if class_name not in used_count:
                used_count[class_name] = 0
            if used_count[class_name] < positive_count:
                negative_pairs.append((same_class_speaker_files[0], other_file))
                used_count[class_name] += 1

                # Pair with other speakers of the same audio file class
        for other_file in second_other_songs_count:
            if class_name not in used_count:
                used_count[class_name] = 0
            if used_count[class_name] < positive_count:
                negative_pairs.append((same_class_speaker_files[0], other_file))
                used_count[class_name] += 1

        for other_speaker_file in audio_files:
            if other_speaker_file[:4] != class_name[:4] and other_speaker_file[4:8] == class_name[4:8]:
                if class_name not in used_count:
                    used_count[class_name] = 0
                if used_count[class_name] < positive_count and other_speaker_file[:4] not in used_speaker:
                    used_speaker.append(other_speaker_file[:4])
                    negative_pairs.append((same_class_speaker_files[0], other_speaker_file))
                    used_count[class_name] += 1
        
        # Pair with remaining songs randomly
        random_songs = [file for file in audio_files if file not in same_class_speaker_files]
        random.shuffle(random_songs)
        for random_song in random_songs[:positive_count]:
            if class_name not in used_count:
                used_count[class_name] = 0
            if used_count[class_name] < positive_count:
                negative_pairs.append((same_class_speaker_files[0], random_song))
                used_count[class_name] += 1
        

    # Assign labels
    positive_labels = [1] * len(positive_pairs)
    negative_labels = [0] * len(negative_pairs)

    all_pairs = positive_pairs + negative_pairs
    all_labels = positive_labels + negative_labels

    return all_pairs, all_labels

def get_random_elements(array_elements, number_of_elements):
    random_sample = []
    if array_elements:
        sample_size = min(len(array_elements), number_of_elements)
        random_sample = random.sample(array_elements, sample_size)
    return random_sample

def preprocess_audio(audio_file, sr=16000, target_duration=0.82):
    y, sr = librosa.load(audio_file, sr=sr)

    # Use voice activity detection (VAD) to find speech segments
    vad_segments = librosa.effects.split(y, top_db=29)
    # Find the boundaries of the first three words
    if len(vad_segments) == 0:
        return
    word_boundaries = []
    word_count = 0
    for segment in vad_segments:
        if word_count >= 3:
            break
        if segment[1] - segment[0] > 0.16 * sr:  # Ensure segment duration is reasonable
            word_boundaries.append(segment)
            word_count += 1

    # Extract the segments corresponding to the first three words
    speech_segments_aligned = []
    for boundary in word_boundaries:
        start_sample = boundary[0]
        end_sample = boundary[1]
        speech_segment = y[start_sample:end_sample]

        # Pad or trim segment to target duration
        target_samples = int(target_duration * sr)
        speech_segments_aligned.append(speech_segment[:target_samples])

    if not speech_segments_aligned:
        raise ValueError("No speech segments found in the audio file.")

    # Compute spectrogram for the first word segment
    spectrogram = librosa.feature.melspectrogram(
        y=speech_segments_aligned[0], sr=sr, n_fft=512, hop_length=256, n_mels=128
    )
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    resized_spectrogram = resize(log_spectrogram, (100, 50))
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=-1)

    return resized_spectrogram

def preprocess_audio(audio_file, sr=16000, target_duration=0.82):
    y, sr = librosa.load(audio_file, sr=sr)
    vad_segments = librosa.effects.split(y, top_db=29)
    
    word_boundaries = []
    word_count = 0
    for segment in vad_segments:
        if word_count >= 3:
            break
        if segment[1] - segment[0] > 0.16 * sr:
            word_boundaries.append(segment)
            word_count += 1

    combined_segment = np.concatenate([y[boundary[0]:boundary[1]] for boundary in word_boundaries], axis=0)
    target_samples = int(target_duration * sr)
    combined_segment = combined_segment[:target_samples]

    spectrogram = librosa.feature.melspectrogram(
        y=combined_segment, sr=sr, n_fft=512, hop_length=256, n_mels=128
    )
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    resized_spectrogram = resize(log_spectrogram, (100, 50))
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=-1)

    return resized_spectrogram

def generate_data_for_saving(pairs, spectrogram_dict, output_directory='./Spectrogram Pairs'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for pair in pairs:
        file1, file2 = pair
        #pair = pair.split('.npy')[0]
        #file1, file2 = pair.split('-')
        #cleaned_pair = pair.strip("()").replace("'", "")
        # Split the string by the comma
        #file1, file2 = cleaned_pair.split(", ")
        spectrogram_1 = spectrogram_dict[file1]
        spectrogram_2 = spectrogram_dict[file2]
        np.save(os.path.join(output_directory, f"{file1}-{file2}.npy"), [spectrogram_1[:, :, 0], spectrogram_2[:, :, 0]])
    
    print("All pairs generated.")

#data_type = "TEST"
#audio_dir = f"./GRID/{data_type}"
audio_files = [os.path.join("./GRID/Additional Speakers COMINED", f) for f in os.listdir("./GRID/Additional Speakers COMINED") if f.endswith('.wav')]

spectrogram_dict = {}
for audio_file in audio_files:
    spectrogram = preprocess_audio(audio_file)
    name = os.path.basename(audio_file)
    name = name.split(".wav")[0]
    spectrogram_dict[name] = spectrogram

#df = pd.read_csv("pairs_with_labels_TEST.csv")
#all_pairs =  df['Pairs'].dropna().tolist()
#df = pd.read_csv("split_distribution_basic.csv")
#all_pairs =  df['val_files'].dropna().tolist() + df['test_files'].dropna().tolist()
all_pairs, all_labels = create_pairs_with_labels(spectrogram_dict.keys())
print(len(all_pairs))
# Create a DataFrame
df = pd.DataFrame({'Pairs': all_pairs, 'Label': all_labels})

# Save the DataFrame to a CSV file
#df.to_csv(f'pairs_with_labels_{data_type}.csv', index=False)
df.to_csv(f'pairs_with_labels_Additional_Spkeakers_COMBINED.csv', index=False)

# Save spectrogram pairs
generate_data_for_saving(all_pairs, spectrogram_dict, output_directory=f"Spectrogram Pairs Additional")
