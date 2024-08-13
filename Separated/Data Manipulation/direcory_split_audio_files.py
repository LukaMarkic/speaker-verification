import os
import shutil

def move_files(src_directory, train_directory, val_directory, test_directory):
    train_indices = {"1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",
                     "1011", "1012", "1013", "1014", "1015", "1016", "1017", "1018", "1020", "1021",
                     "1022", "1023", "1024", "1025"}
    val_indices = {"1019", "1026", "1029"}
    test_indices = {"1027", "1028", "1030", "1032", "1031", "1033", "1034"}

    # Ensure the target directories exist
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(val_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    # Iterate over all files in the source directory
    for file_name in os.listdir(src_directory):
        file_path = os.path.join(src_directory, file_name)
        if os.path.isfile(file_path):
            prefix = file_name[:4]
            if prefix in train_indices:
                target_directory = train_directory
            elif prefix in val_indices:
                target_directory = val_directory
            elif prefix in test_indices:
                target_directory = test_directory
            else:
                print(f"Skipping file with unrecognized prefix: {file_name}")
                continue

            # Move the file to the target directory
            shutil.move(file_path, os.path.join(target_directory, file_name))
            print(f"Moved {file_name} to {target_directory}")

if __name__ == "__main__":
    src_directory = "GRID/COMBINED"
    train_directory = "GRID/TRAIN"
    val_directory = "GRID/VALIDATION"
    test_directory = "GRID/TEST"

    move_files(src_directory, train_directory, val_directory, test_directory)

