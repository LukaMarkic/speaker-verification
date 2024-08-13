import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, Model, Sequential
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D, Activation, Dropout
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
from keras.callbacks import Callback
import keras.backend as K
import pandas as pd

class DynamicLearningRateScheduler(Callback):
    def __init__(self, factor=0.65, patience=6, threshold=0.1, min_lr=1e-5, verbose=1):
        super(DynamicLearningRateScheduler, self).__init__()
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        self.wait = 0
        self.best_divergence = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        if train_loss is not None and val_loss is not None:
            divergence = abs(train_loss - val_loss)
            
            if divergence < self.best_divergence - self.threshold:
                self.best_divergence = divergence
                self.wait = 0
            else:
                self.wait += 1
                
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor * train_loss/val_loss
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print(f"\nEpoch {epoch + 1}: reducing learning rate to {new_lr}.")
                            
                        self.wait = 0

    def on_train_begin(self, logs=None):
        self.best_divergence = float('inf')
        self.wait = 0
# Define a Lambda layer to compute Euclidean distance
def euclidean_distance(tensors):
    x, y = tensors
    sum_squared = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))
    
def create_siamese_network(input_shape):
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)

    convnet = Sequential([
        Conv2D(12, 3, input_shape=input_shape),
        Activation('relu'),
        Conv2D(24, 3, padding="same"),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(24, 3),
        Activation('relu'),
        Conv2D(36, 3, padding="same"),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(64, 2),
        Activation('relu'),
        Conv2D(128, 2),
        Activation('relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(32),
        Activation('sigmoid')
    ])

    encoded_l = convnet(input_1)
    encoded_r = convnet(input_2)
    
    L1_layer = Lambda(lambda tensor: tf.abs(tensor[0] - tensor[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid')(L1_distance)
    '''
    L2_layer = Lambda(euclidean_distance)
    L2_distance = L2_layer([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid')(L2_distance)
    '''
    siamese_model = Model(inputs=[input_1, input_2], outputs=prediction)

    return siamese_model

# Function to load a batch of images
def load_batch(image_dir, remaining_files, batch_size):
    # Select batch_size number of files randomly
    batch_files = random.choices(remaining_files, k=batch_size)
    # Create a new list without the selected files
    remaining_files = [file_name for file_name in remaining_files if file_name not in batch_files]
    batch_images = [np.load(os.path.join(image_dir, file_name)) for file_name in batch_files]  # Extracting spectrograms
    labels = get_pair_labels(batch_files)  # Extracting labels
    return np.array(batch_images), np.array(labels), remaining_files

# Function to create pairs and labels for training
def get_pair_labels(file_names):
    labels = []
    for pair in file_names:
        file1, file2 = pair.split('-')
        file1 = file1.split('/')[-1]
        label = 1 if file1[:8] == file2[:8] else 0
        labels.append(label)
    return labels

def get_split(direcotry = 'split_distribution_basic.csv'):
    # Load the split distribution from the CSV file
    df = pd.read_csv(direcotry)
    train_files = df['train_files'].dropna().tolist()
    val_files = df['val_files'].dropna().tolist()
    test_files = df['test_files'].dropna().tolist()

    return train_files, val_files, test_files


# Function to predict labels for a batch of images
def predict_labels(siamese_model, batch_images):
    batch_images_1 = [pair[0] for pair in batch_images]
    batch_images_2 = [pair[1] for pair in batch_images]
    predictions = siamese_model.predict([np.array(batch_images_1), np.array(batch_images_2)], verbose=0)
    rounded_predictions = np.round(predictions).astype(int)
    return rounded_predictions, predictions[:, 0]


# Directory containing numpy image files
image_dir = "Spectrogram Pairs Combined"

# Split dataset into training, validation, and testing sets
train_files, val_files, _ = get_split()

# Training loop
batch_size = 512
epochs = 120
current_time = datetime.now()
# Format the current date and time as a string
formatted_time = current_time.strftime("%d-%m-%Y")
start_epoch = 0
dataset_type = "Gaussian_Combined"
output_model_directory = f"{dataset_type}_{formatted_time}"
date = "01-07-2024"  #30-05-2024, 31-05-2024
prev_train_loss = 1
prev_val_loss = 1
print(output_model_directory)

input_shape = (100, 50, 1)
siamese_model = create_siamese_network(input_shape)
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
if start_epoch > 0:
    siamese_model = load_model(f"{dataset_type}_{date}/latest_siamese_model_e-{start_epoch}_f.h5")

steps_per_epoch = (len(train_files) // batch_size) if len(train_files) % batch_size == 0 else (
            len(train_files) // batch_size) + 1
validation_steps = (len(val_files) // batch_size) if len(val_files) % batch_size == 0 else (
            len(val_files) // batch_size) + 1

print(len(train_files) % batch_size)
train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []
test_loss_history = []
test_accuracy_history = []
lr_change_counter = 0

dynamic_lr_scheduler = DynamicLearningRateScheduler(factor=0.91, patience=3, threshold=0.1, min_lr=1e-6, verbose=1)
dynamic_lr_scheduler.set_model(siamese_model)

for epoch in range(epochs - start_epoch):
    print(f"Epoch {start_epoch + epoch + 1}/{epochs}:")

    if start_epoch + epoch == 0:
        siamese_model.optimizer.lr = 0.00074

    total_train_loss = 0
    total_train_accuracy = 0
    train_steps = 0
    remainig_training_files = train_files[:]
    remainig_val_files = val_files[:]
    
    for step in range(steps_per_epoch):
        # Load a batch of images
        items_per_step = batch_size if step != (steps_per_epoch - 1) else len(train_files) % batch_size
        while True:
            batch_images, labels, remainig_training_files = load_batch(image_dir, remainig_training_files, items_per_step)
            if step != (steps_per_epoch - 1) and len(batch_images) == batch_size:  # Check if all images are loaded
                break
            elif step == (steps_per_epoch - 1) and len(batch_images) == len(train_files) % batch_size:
                break

        batch_images_1 = [pair[0] for pair in batch_images]
        batch_images_2 = [pair[1] for pair in batch_images]

        # Train the model on the batch
        loss, accuracy = siamese_model.train_on_batch([np.array(batch_images_1), np.array(batch_images_2)], labels)

        # Print progress bar
        progress = (step + 1) / steps_per_epoch * 100
        print(f"\r{(step + 1)}/{steps_per_epoch}: [{'=' * int(progress / 5):20}]", end="", flush=True)

        total_train_loss += loss
        total_train_accuracy += accuracy
        train_steps += 1

    avg_train_loss = total_train_loss / train_steps
    avg_train_accuracy = total_train_accuracy / train_steps
    print(f"Average Training Loss: {avg_train_loss}, Average Training Accuracy: {avg_train_accuracy}")
    prev_train_loss = avg_train_loss

    # Validation
    total_val_loss = 0
    total_val_accuracy = 0
    val_steps = 0

    # Evaluate the model on the validation set
    for step in range(validation_steps):
        items_per_step = batch_size if step != (validation_steps - 1) else len(val_files) % batch_size
        while True:
            val_batch_images, val_labels, remainig_val_files = load_batch(image_dir, remainig_val_files, items_per_step)
            if step != (validation_steps - 1) and len(
                    val_batch_images) == batch_size:  # Check if all images are loaded
                break
            elif step == (validation_steps - 1) and len(
                    val_batch_images) == len(val_files) % batch_size:
                break

        val_batch_images_1 = [pair[0] for pair in val_batch_images]
        val_batch_images_2 = [pair[1] for pair in val_batch_images]
        # Evaluate the model on the validation batch
        val_loss, val_accuracy = siamese_model.evaluate([np.array(val_batch_images_1), np.array(val_batch_images_2)],
                                                        val_labels, verbose=0)
        # Print progress bar
        progress = (step + 1) / validation_steps * 100
        print(f"\r{(step + 1)}/{validation_steps}: [{'=' * int(progress / 5):20}]", end="", flush=True)

        total_val_loss += val_loss
        total_val_accuracy += val_accuracy
        val_steps += 1

    avg_val_loss = total_val_loss / val_steps
    avg_val_accuracy = total_val_accuracy / val_steps
    print(f"Average Validation Loss: {avg_val_loss}, Average Validation Accuracy: {avg_val_accuracy}")
    prev_val_loss = avg_val_loss

   
    if not os.path.exists(output_model_directory):
        os.makedirs(output_model_directory)
    siamese_model.save(f"{output_model_directory}/latest_siamese_model_e-{start_epoch + epoch + 1}_f.h5")

    # Save loss and accuracy history
    train_loss_history.append(avg_train_loss)
    train_accuracy_history.append(avg_train_accuracy)
    val_loss_history.append(avg_val_loss)
    val_accuracy_history.append(avg_val_accuracy)

    history_data = pd.DataFrame({
    'train_loss': train_loss_history,
    'train_accuracy': train_accuracy_history,
    'val_loss': val_loss_history,
    'val_accuracy': val_accuracy_history,
    })

    history_data.to_csv(f"{dataset_type}_results_per_epoch_{start_epoch}_{formatted_time}.csv", index=False)
    
    # Update the learning rate dynamically
    logs = {'loss': avg_train_loss, 'val_loss': avg_val_loss}
    dynamic_lr_scheduler.on_epoch_end(epoch, logs)

# Save loss and accuracy history to file
history_data = pd.DataFrame({
    'train_loss': train_loss_history,
    'train_accuracy': train_accuracy_history,
    'val_loss': val_loss_history,
    'val_accuracy': val_accuracy_history,
    })

history_data.to_csv(f"results_per_epoch_{start_epoch}_{formatted_time}.csv", index=False)

