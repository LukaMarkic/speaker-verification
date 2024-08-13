import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve
import os

def calculate_metrics_from_file(csv_file, threshold=0.5, output_dir='./Metrics', dataset_type=""):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract true labels and predictions, and convert predictions to floats
    true_labels = df['True_Labels'].values
    predictions = df['Predictions'].astype(float).values

    # Convert predictions to binary based on threshold
    binary_predictions = (predictions >= threshold).astype(int)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, binary_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Save confusion matrix to file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(os.path.join(output_dir, f'confusion_matrix_t_{threshold}_{dataset_type}.csv'), conf_matrix, delimiter=",")

    # Calculate metrics
    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions)
    recall = recall_score(true_labels, binary_predictions)
    specificity = recall_score(true_labels, binary_predictions, pos_label=0)

    # Save metrics to file
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, f'metrics_t_{threshold}_{dataset_type}.csv'), index=False)

    # Print metrics
    print("Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    csv_file = './Predictions_additional_03-07-2024.csv'  # Path to your CSV file
    calculate_metrics_from_file(csv_file=csv_file, threshold=0.5, dataset_type = "additional")
