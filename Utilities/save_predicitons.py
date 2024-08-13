import os
import numpy as np
import random
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def load_batch(remaining_files, batch_size, files_directory=""):
    batch_files = []
    batch_images = []
    while len(batch_images) < batch_size and remaining_files:
        file_name = random.choice(remaining_files)
        try:
            spectrogram = np.load(os.path.join(files_directory, file_name))
            batch_images.append(spectrogram)
            batch_files.append(file_name)
            remaining_files.remove(file_name)
        except (EOFError, ValueError) as e:
            print(f"Skipping file {file_name} due to loading error: {e}")
    labels = get_pair_labels(batch_files)
    return np.array(batch_images), np.array(labels), remaining_files

def get_pair_labels(file_names):
    labels = []
    for pair in file_names:
        file1, file2 = pair.split('-')
        file1 = file1.split('/')[-1]
        label = 1 if file1[:8] == file2[:8] else 0
        labels.append(label)
    return labels

def predict_labels(siamese_model, batch_images):
    batch_images_1 = [pair[0] for pair in batch_images]
    batch_images_2 = [pair[1] for pair in batch_images]
    predictions = siamese_model.predict([np.array(batch_images_1), np.array(batch_images_2)], verbose=0)
    return predictions[:, 0]


def save_predictions_and_labels(true_labels, predictions, dataset_type, formatted_time):
    df = pd.DataFrame({'True_Labels': true_labels, 'Predictions': predictions})
    df.to_csv(f"Predictions_{dataset_type}_{formatted_time}.csv", index=False)

def evaluate_model(siamese_model, dataset_files, dataset_type, batch_size, formatted_time):
    steps = (len(dataset_files) // batch_size) + (len(dataset_files) % batch_size != 0)
    remaining_files = dataset_files[:]
    all_true_labels = []
    all_predictions = []

    for step in range(steps):
        if step == steps - 1:
            items_per_step = len(remaining_files)
        else:
            items_per_step = batch_size

        while True:
            batch_images, labels, remaining_files = load_batch(remaining_files, items_per_step)
            if len(batch_images) == items_per_step:
                break

        raw_predictions = predict_labels(siamese_model, batch_images)
        all_true_labels.extend(labels)
        all_predictions.extend(raw_predictions)

        print(f"Step {step + 1}/{steps} - Batch size: {len(batch_images)}, Labels: {len(labels)}, Predictions: {len(raw_predictions)}")

    save_predictions_and_labels(all_true_labels, all_predictions, dataset_type, formatted_time)


def main():
    model_path = "./../latest_siamese_model_e-34_f.h5"
    #pairs_file = 'split_distribution_basic.csv'

    #df = pd.read_csv(pairs_file)
    #train_files = df['train_files'].dropna().tolist() 
    #val_files = df['val_files'].dropna().tolist()
    #test_files = df['test_files'].dropna().tolist()

    #train_files = [os.path.join("Spectrogram Pairs TRAIN", f) for f in os.listdir("Spectrogram Pairs TRAIN") if f.endswith('.npy')]
    #val_files = [os.path.join("Spectrogram Pairs VALIDATION", f) for f in os.listdir("Spectrogram Pairs VALIDATION") if f.endswith('.npy')]
    #test_files = [os.path.join("Spectrogram Pairs TEST", f) for f in os.listdir("Spectrogram Pairs TEST") if f.endswith('.npy')]

    additionals_files = [os.path.join("Spectrogram Pairs Additional", f) for f in os.listdir("Spectrogram Pairs Additional") if f.endswith('.npy')]

    batch_size = 512
    current_time = datetime.now()
    formatted_time = current_time.strftime("%d-%m-%Y")
    siamese_model = load_model(model_path)

    #evaluate_model(siamese_model, train_files, "train", batch_size, formatted_time)
    #evaluate_model(siamese_model, val_files, "val", batch_size, formatted_time)
    #evaluate_model(siamese_model, test_files, "test", batch_size, formatted_time)
    evaluate_model(siamese_model, additionals_files, "additional", batch_size, formatted_time)
if __name__ == "__main__":
    main()

