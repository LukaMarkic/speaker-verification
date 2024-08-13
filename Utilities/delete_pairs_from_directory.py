import os
import pandas as pd

def remove_files(directory, files_to_remove):
    for file_name in files_to_remove:
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    print("All pairs have been removed.")

if __name__ == "__main__":
    directory = 'Spectrogram Pairs Combined'
    pairs_name_file = 'split_distribution_basic.csv'
    df = pd.read_csv(pairs_name_file)
    train_files = df['train_files'].dropna().tolist()
    
    remove_files(directory, train_files)
