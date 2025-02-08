import os
import shutil
import re

sourcedir = "/home/m/malone/leaf_storage/random_walks/leaves_full_11-10-23_MUT3"
endir = "/home/m/malone/leaf_storage/random_walks/firstandlast/firstandlast_11-10-23_MUT3"

os.makedirs(endir, exist_ok=True)
os.makedirs(endir + "/first", exist_ok=True)
os.makedirs(endir + "/last", exist_ok=True)

def copy_first_png(walkdir_path, png_list, endir):
    if png_list:
        firstpng_path = os.path.join(walkdir_path, png_list[0])
        shutil.copy(firstpng_path, endir + "/first")

def copy_last_png(walkdir_path, png_list, endir):
    if png_list:
        lastpng_path = os.path.join(walkdir_path, png_list[-1])
        shutil.copy(lastpng_path, endir + "/last")


def main(sourcedir, endir):
    for leafdir in os.listdir(sourcedir):
        print(f"Current = {leafdir}")
        leafdir_path = os.path.join(sourcedir, leafdir)
        for walkdir in os.listdir(leafdir_path):
            walkdir_path = os.path.join(leafdir_path, walkdir)
            png_list = [file for file in os.listdir(walkdir_path) if file.endswith(".png")]
            #sort by last continuous stretch of numbers in the filename
            png_list_sorted = sorted(png_list, key=lambda x: int(re.search(r'\d+(?=\D*$)', x).group(0)) if re.search(r'\d+(?=\D*$)', x) else 0)
            # for i in png_list:
            #     print(int(re.search(r'\d+(?=\D*$)', i).group(0)) if re.search(r'\d+(?=\D*$)', i) else 0)
            copy_first_png(walkdir_path, png_list_sorted, endir)
            copy_last_png(walkdir_path, png_list_sorted, endir)

main(sourcedir, endir)
