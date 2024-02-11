import pandas as pd
import os
import urllib.request
from PIL import Image


def download():
    if os.listdir("download_imgs"):
        print("download_imgs is not empty! Terminating.")

    else:
        intersect = pd.read_csv(
            "sample_eud_21-1-24/Naturalis_eud_sample_Janssens_intersect_21-01-24.csv"
        )

        for index, row in intersect.iterrows():
            try:
                species = row["species"]
                print(index, species)
                url = row["accessURI"]
                urllib.request.urlretrieve(url, f"temp.png")
                img = Image.open(r"temp.png")
                img.save(f"download_imgs/{species}{index}.png")
            except:
                None


download()
