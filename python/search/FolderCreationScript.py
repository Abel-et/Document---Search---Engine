import os

# Define the package path (adjust according to your structure)
package_path = "path_to/ C:\Users\hp\Searches\python\search\"

  # replace 'path_to' with the actual path to your package

# Folder names to create
folders_to_create = ['templates', 'preloaded_files']

# Create folders
for folder in folders_to_create:
    folder_path = os.path.join(package_path, folder)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")