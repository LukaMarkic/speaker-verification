import os

def delete_files(directory):
    # Create a dictionary to store files based on their first eight characters
    files_dict = {}
    number_of_deleted_files = 14

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Extract the first eight characters of the filename
        prefix = filename[:8]
        
        # Check if the prefix is in the exceptions list
        if prefix[5:] in ['bga', 'bgb', 'bgi', 'gbw', 'lra', 'lrb', 'lri', 'lrw', 
                          'pwa', 'pwb', 'pwi', 'pww', 'sba', 'sbb', 'sbi', 'sbw']:
            # Add the file to the dictionary to potentially delete later
            if prefix not in files_dict:
                files_dict[prefix] = []
            files_dict[prefix].append(filename)

    # Iterate over the dictionary and delete all but the first 8 files for each prefix
    for prefix, files in files_dict.items():
        if len(files) > number_of_deleted_files:
            # Sort the files by modification time (oldest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
            # Delete all but the first 8 files
            for file_to_delete in files[number_of_deleted_files:]:
                os.remove(os.path.join(directory, file_to_delete))

# Example usage:
directory = "./GRID/COMBINED"
delete_files(directory)
