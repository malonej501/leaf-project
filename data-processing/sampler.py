import os
import random
import shutil

# Set the paths to the input directory and the output directory
input_dir = "/home/m/malone/leaf_storage/random_walks/leaves_full_1-5-23"
output_dir = "/home/m/malone/leaf_storage/accuracy_test_500"

# Get a list of all files in the input directory
files = os.listdir(input_dir)

n = 10

for i in range(0, n):
    # Loop through each file in the input directory
    for file in files:
        # Create the full path to the file
        file_path = os.path.join(input_dir, file)

        # Check if the file is a directory
        if os.path.isdir(file_path):
            # Get a list of all files in the subdirectory
            subfiles = os.listdir(file_path)

            # Choose a random subfile from the subdirectory
            subfile = random.choice(subfiles)

            # Create the full path to the subfile
            subfile_path = os.path.join(file_path, subfile)

            # Check if the subfile is a directory
            if os.path.isdir(subfile_path):
                # Get a list of all .png files in the subdirectory
                png_files = [f for f in os.listdir(subfile_path) if f.endswith(".png")]

                # Choose a random .png file from the subdirectory
                png_file = random.choice(png_files)

                # Create the full path to the .png file
                png_file_path = os.path.join(subfile_path, png_file)

                # Get the new filename
                new_filename = f"{file}_{i+1}.png"

                # Create the full path to the new file
                new_file_path = os.path.join(output_dir, new_filename)

                # Copy the .png file to the output directory and rename it
                shutil.copy(png_file_path, new_file_path)
