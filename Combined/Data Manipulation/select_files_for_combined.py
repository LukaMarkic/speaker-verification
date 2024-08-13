import os
import shutil

def copy_files(source_base_directory, target_directory):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Define the exception prefixes
    exception_prefixes = ['bga', 'bgb', 'bgi', 'gbw', 'lra', 'lrb', 'lri', 'lrw', 
                          'pwa', 'pwb', 'pwi', 'pww', 'sba', 'sbb', 'sbi', 'sbw']
    
    # Iterate over directories s1 to s34
    for i in range(1, 35):
        directory = os.path.join(source_base_directory, f"s{i}")
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Skipping.")
            continue
        
        # Create a dictionary to store files based on their first eight characters
        files_dict = {}

        # Iterate over all files in the directory
        for filename in os.listdir(directory):
            # Extract the first eight characters of the filename
            prefix = filename[:8]
            
            # Check if the prefix is in the exceptions list
            if prefix[5:] in exception_prefixes:
                # Add the file to the dictionary to potentially copy later
                if prefix not in files_dict:
                    files_dict[prefix] = []
                files_dict[prefix].append(filename)
            else:
                # Add the file to the dictionary to potentially copy later
                if prefix not in files_dict:
                    files_dict[prefix] = []
                files_dict[prefix].append(filename)

        # Iterate over the dictionary and copy files according to the logic
        for prefix, files in files_dict.items():
            if prefix[5:] in exception_prefixes:
                # Select 14 files or less
                selected_files = files[:14]
            else:
                # Select only 8 files
                selected_files = files[:8]
            
            # Copy the selected files to the target directory
            for file_to_copy in selected_files:
                source_path = os.path.join(directory, file_to_copy)
                destination_path = os.path.join(target_directory, file_to_copy)
                shutil.copy(source_path, destination_path)
                print(f"Copied {file_to_copy} to {target_directory}")

# Example usage:
source_base_directory = "./GRID"
target_directory = "./GRID/COMBINED"
copy_files(source_base_directory, target_directory)
