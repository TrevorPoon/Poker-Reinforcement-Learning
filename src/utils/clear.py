import os
import shutil

# Define directories to clear
directories_to_clear = [
    "result/log",
    "result/runs",
    "result/slurm",
    "models/"
]

# Clear each directory
for directory in directories_to_clear:
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Remove the directory and all its contents
        os.makedirs(directory)  # Recreate the empty directory
