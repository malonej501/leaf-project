import pandas as pd
from datetime import datetime
import os
import urllib.request
from PIL import Image


species_per_family = 200  # specify no. species to sample per angiosperm family - if there aren't enough species in the database, it will take the maximum number.

angio_fams = pd.read_csv("../APG_IV/APG_IV_ang_fams.csv")
eud_fams = pd.read_csv("../APG_IV/APG_IV_eud_fams.csv")

current_date = datetime.now().strftime("%d-%m-%y")


def filter_to_angio_or_eud():

    chunk_size = 100000  # Adjust the chunk size as needed
    # Initialize an empty list to store the intersected dataframes
    intersect_dfs = []

    # Read and process the data in chunks
    for i, chunk_occurrence in enumerate(
        pd.read_csv(
            "botany-20240108.dwca/Occurrence.txt",
            chunksize=chunk_size,
            low_memory=False,
        )
    ):
        print(f"Row number: {i * chunk_size}")
        # Perform the intersection with angio_fams for the current chunk
        intersect_chunk = pd.merge(
            chunk_occurrence, angio_fams, on="family", how="inner"
        )

        # Subset chunk to rows representing a species
        intersect_chunk = intersect_chunk[intersect_chunk["taxonRank"] == "species"]

        # Append the intersected chunk to the list
        intersect_dfs.append(intersect_chunk)

    # Concatenate the list of intersected dataframes into a single dataframe
    intersect = pd.concat(intersect_dfs, ignore_index=True)

    intersect.to_csv(
        "~/Documents/Leaf Project/Naturalis/Naturalis_ang_species_occurrence.csv",
        index=False,
    )


def sample_families(sample_fams):
    print("Reading data...")
    ang_sp_full = pd.read_csv(
        "Naturalis_ang_species_occurrence.csv",
        low_memory=False,
    )
    print("Done!")

    print("Removing duplicate species...")
    sp_list = ang_sp_full["genus"].str.cat(ang_sp_full["specificEpithet"], sep="_")
    ang_sp_full.insert(0, "species", sp_list)
    ang_sp_full_clean = ang_sp_full.drop_duplicates(
        subset="species", keep="first"
    ).reset_index(drop=True)
    print("Done!")

    sample_dfs = []

    for i, family in enumerate(sample_fams["family"]):
        print(i, family)
        fam = ang_sp_full_clean[ang_sp_full_clean["family"] == family]
        if len(fam) >= species_per_family:
            fam_samp = fam.sample(n=species_per_family)
        else:
            fam_samp = fam.sample(n=len(fam))
        sample_dfs.append(fam_samp)

    sample = pd.concat(sample_dfs, ignore_index=True)

    sample = sample.rename(
        columns={"id": "CoreId"}
    )  # do this because the variable is called CoreId in Multimedia.txt

    sample.to_csv(
        f"./Naturalis_occurrence_eud_sample_{current_date}.csv",
        index=False,
    )


def img_from_sample():
    print("Reading data...")
    sample = pd.read_csv(
        "sample_eud_13-1-24/Naturalis_occurrence_eud_sample_13-01-24.csv"
    )
    print("Done!")

    multimedia = pd.read_csv("botany-20240108.dwca/Multimedia.txt")

    # Shouldn't be any duplicate species as merging on CoreId, not species
    intersect = pd.merge(multimedia, sample, on="CoreId", how="inner")

    intersect.to_csv(
        f"./Naturalis_multimedia_eud_sample_{current_date}.csv",
        index=False,
    )


def download_imgs():
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


if __name__ == "__main__":
    # sample_families(eud_fams)
    img_from_sample()
