import os
import imageio
import numpy as np
import math
from PIL import Image, ImageSequence
from skimage.transform import resize

wd = "/home/m/malone/leaf_storage/random_walks/leaves_full_1-5-23"
wd1 = "/home/m/malone/leaf_storage/random_walks"

def convert_png_to_gif(folder_path, gif_path):
    images = []
    
    # Iterate over all files in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path)
            
            # Convert RGBA images to RGB
            if img.mode == "RGBA":
                img = img.convert("RGB")
            
            images.append(img)
    
    # Save the images as a GIF
    images[0].save(gif_path, save_all=True, append_images=images[1:], loop=0, duration=200)

def convert_png_to_gif_batch(wd):

    for leafdirectory in os.listdir(wd):
        print(f"Current = {leafdirectory}")
        leafdirectory_path = os.path.join(wd, leafdirectory)
        for walkdirectory in os.listdir(leafdirectory_path):
            walkdirectory_path = os.path.join(leafdirectory_path, walkdirectory)

            folder_path = walkdirectory_path
            gif_path = walkdirectory_path + "/animation.gif"
                    
            images = []
            
            # Iterate over all files in the folder
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith(".png"):
                    file_path = os.path.join(folder_path, filename)
                    img = Image.open(file_path)
                    
                    # Convert RGBA images to RGB
                    if img.mode == "RGBA":
                        img = img.convert("RGB")
                    
                    images.append(img)
            
            # Save the images as a GIF
            images[0].save(gif_path, save_all=True, append_images=images[1:], loop=0)


def combine_gifs_in_folders(panel_width=None):
    gif_paths = []
    
    # Iterate through subfolders
    for leafdirectory in os.listdir(wd): 
        print(f"Current = {leafdirectory}")
        leafdirectory_path = os.path.join(wd, leafdirectory)
        walkdirectory_path = leafdirectory_path + "/walk0"
        for dirpath, dirnames, filenames in os.walk(walkdirectory_path):
            for filename in filenames:
                if filename == "animation.gif":
                    gif_path = os.path.join(dirpath, filename)
                    gif_paths.append(gif_path)

    # Calculate the number of columns and rows in the panel
    num_gifs = len(gif_paths)
    panel_size = math.ceil(math.sqrt(num_gifs))
    panel_width = panel_size * 256
    panel_height = panel_size * 256

    panel = imageio.imread(gif_paths[0])
    panel = resize(panel, (panel_height, panel_width))
    durations = []

    for gif_path in gif_paths:
        gif_frames = imageio.mimread(gif_path)
        gif_duration = gif_frames[0].duration
        durations.append(gif_duration)
        gif_resized = resize(gif_frames[0], (256, 256))
        panel = imageio.hstack([panel, gif_resized])

    imageio.mimsave(os.path.join(wd1, "aggregated.gif"), panel, duration=durations)
    
#combine_gifs_in_folders()

# Example usage
#folder_path = "/home/jamesmalone/Desktop/p1_82_3_giftest"
#gif_path = "/home/jamesmalone/Desktop/p1_82_3_giftest/giftest.gif"
#convert_png_to_gif(folder_path, gif_path)

convert_png_to_gif_batch(wd)