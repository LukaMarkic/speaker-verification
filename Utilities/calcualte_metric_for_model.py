import os
import numpy as np
import random
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def load_batch(remaining_files, batch_size, files_directory="Spectrogram Pairs Combined"):
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

def create_confusion_matrix(true_labels, predictions, dataset_type, formatted_time, threshold=0.5):
    predictions = np.array(predictions)
    binary_predictions = (predictions >= threshold).astype(int)
    conf_matrix = confusion_matrix(true_labels, binary_predictions)
    print(f"Confusion Matrix ({dataset_type}):")
    print(conf_matrix)
    if not os.path.exists(f"Confusion_Matrices_{formatted_time}"):
        os.makedirs(f"Confusion_Matrices_{formatted_time}")
    np.savetxt(f"Confusion_Matrices_{formatted_time}/confusion_matrix_{dataset_type}_{formatted_time}.csv", conf_matrix, delimiter=",")

def calculate_metrics(true_labels, predictions, threshold=0.5):
    predictions = np.array(predictions)
    binary_predictions = (predictions >= threshold).astype(int)
    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions)
    recall = recall_score(true_labels, binary_predictions)
    specificity = recall_score(true_labels, binary_predictions, pos_label=0)
    return accuracy, precision, recall, specificity

def save_metrics(metrics, dataset_type, formatted_time):
    df = pd.DataFrame([metrics], columns=['Accuracy', 'Precision', 'Recall', 'Specificity'])
    df.to_csv(f"Metrics_{dataset_type}_{formatted_time}.csv", index=False)

def compute_eer(fpr, tpr):
    fnr = 1 - tpr
    eer_threshold = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold] + fnr[eer_threshold]) / 2
    return eer, eer_threshold

def getTitle(dataset_type):
    if dataset_type == "train":
        return 'FAR i FRR krivulje skupa za učenje'
    elif dataset_type == "val":
        return 'FAR i FRR krivulje validacijskog skupa'
    elif dataset_type == "test":
        return 'FAR i FRR krivulje testnog skupa'
    else:
        return 'FAR i FRR krivulje'

def plot_far_frr_graph(true_labels, predictions, dataset_type, formatted_time):
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    fnr = 1 - tpr
    eer, eer_threshold = compute_eer(fpr, tpr)

    plt.figure(figsize=(18, 10))
    plt.grid()
    plt.plot(thresholds, fpr, label='FAR')
    plt.plot(thresholds, fnr, label='FRR')
    plt.scatter(thresholds[eer_threshold], fpr[eer_threshold], c='red', label=f'EER = {eer:.2f}')
    plt.xlabel('Prag')
    plt.ylabel('Stupanj pogreške')
    plt.title(getTitle(dataset_type=dataset_type))
    plt.legend(loc='best')
    if not os.path.exists(f"FAR_FRR_Graphs_{formatted_time}"):
        os.makedirs(f"FAR_FRR_Graphs_{formatted_time}")
    plt.savefig(f"FAR_FRR_Graphs_{formatted_time}/far_frr_graph_{dataset_type}_{formatted_time}.png")
    plt.close()
    
    eer_df = pd.DataFrame({'EER': [eer], 'Threshold': [thresholds[eer_threshold]]})
    eer_df.to_csv(f"FAR_FRR_Graphs_{formatted_time}/eer_threshold_{dataset_type}_{formatted_time}.csv", index=False)

def evaluate_model(siamese_model, dataset_files, dataset_type, batch_size, formatted_time, spectrogram_directory):
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
            batch_images, labels, remaining_files = load_batch(remaining_files, items_per_step, files_directory=spectrogram_directory)
            if len(batch_images) == items_per_step:
                break

        raw_predictions = predict_labels(siamese_model, batch_images)
        all_true_labels.extend(labels)
        all_predictions.extend(raw_predictions)

        print(f"Step {step + 1}/{steps} - Batch size: {len(batch_images)}, Labels: {len(labels)}, Predictions: {len(raw_predictions)}")

    create_confusion_matrix(all_true_labels, all_predictions, dataset_type, formatted_time)
    metrics = calculate_metrics(all_true_labels, all_predictions)
    save_metrics(metrics, dataset_type, formatted_time)
    plot_far_frr_graph(all_true_labels, all_predictions, dataset_type, formatted_time)

def main():
    model_path = "./../latest_siamese_model_e-34_f.h5"

    #pairs_file = 'split_distribution_basic.csv'
    #df = pd.read_csv(pairs_file)
    #train_files = df['train_files'].dropna().tolist() 
    #val_files = df['val_files'].dropna().tolist()
    #test_files = df['test_files'].dropna().tolist()

    train_files = [os.path.join("Spectrogram Pairs TRAIN", f) for f in os.listdir("Spectrogram Pairs TRAIN") if f.endswith('.npy')]
    val_files = [os.path.join("Spectrogram Pairs VALIDATION", f) for f in os.listdir("Spectrogram Pairs VALIDATION") if f.endswith('.npy')]
    test_files = [os.path.join("Spectrogram Pairs TEST", f) for f in os.listdir("Spectrogram Pairs TEST") if f.endswith('.npy')]

    batch_size = 512
    current_time = datetime.now()
    formatted_time = current_time.strftime("%d-%m-%Y")
    siamese_model = load_model(model_path)

    evaluate_model(siamese_model, train_files, "train", batch_size, formatted_time, "")
    evaluate_model(siamese_model, val_files, "val", batch_size, formatted_time, "")
    evaluate_model(siamese_model, test_files, "test", batch_size, formatted_time, "")

if __name__ == "__main__":
    main()
