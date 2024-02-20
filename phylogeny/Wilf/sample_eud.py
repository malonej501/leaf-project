import pandas as pd
from datetime import datetime
import os
import urllib.request
from PIL import Image


species_per_family = 100  # specify no. species to sample per angiosperm family - if there aren't enough species in the database, it will take the maximum number.

angio_fams = pd.read_csv("APG_IV_ang_fams.csv")
eud_fams = pd.read_csv("APG_IV_eud_fams.csv")

current_date = datetime.now().strftime("%d-%m-%y")


def filter_to_angio_or_eud(fams):

    chunk_size = 100000  # Adjust the chunk size as needed
    # Initialize an empty list to store the intersected dataframes
    intersect_dfs = []

    # Read and process the data in chunks
    for i, chunk_occurrence in enumerate(
        pd.read_csv(
            "supplemental_data_v1.0/Master_inventory_leavesdb_v1.0.csv",
            chunksize=chunk_size,
            low_memory=False,
        )
    ):
        print(f"Row number: {i * chunk_size}")
        # Perform the intersection with angio_fams for the current chunk
        intersect_chunk = pd.merge(chunk_occurrence, fams, on="Family", how="inner")

        # Subset chunk to rows representing a species
        # intersect_chunk = intersect_chunk[intersect_chunk["taxonRank"] == "species"]

        # Append the intersected chunk to the list
        intersect_dfs.append(intersect_chunk)

    # Concatenate the list of intersected dataframes into a single dataframe
    intersect = pd.concat(intersect_dfs, ignore_index=True)

    # Remove fossil specimens
    intersect_no_fossil = intersect[intersect["Collection_type"] != "fossil"]

    intersect_no_fossil.to_csv(
        "wilf_eud_species.csv",
        index=False,
    )


def sample_families(sample_fams):
    print("Reading data...")
    sp_full = pd.read_csv(
        "wilf_eud_species.csv",
        low_memory=False,
    )
    print("Done!")

    sp_list = sp_full["Genus"].str.cat(sp_full["species"], sep="_")
    sp_full.insert(0, "genus_species", sp_list)
    # print("Removing duplicate species...")
    # ang_sp_full_clean = sp_full.drop_duplicates(
    #     subset="species", keep="first"
    # ).reset_index(drop=True)
    # print("Done!")

    sample_dfs = []

    for i, family in enumerate(sample_fams["Family"]):
        print(i, family)
        fam = sp_full[sp_full["Family"] == family]
        if len(fam) >= species_per_family:
            fam_samp = fam.sample(n=species_per_family)
        else:
            fam_samp = fam.sample(n=len(fam))
        sample_dfs.append(fam_samp)

    sample = pd.concat(sample_dfs, ignore_index=True)

    # sample = sample.rename(
    #     columns={"id": "CoreId"}
    # )  # do this because the variable is called CoreId in Multimedia.txt

    print(sample)
    print(len(set(sample["Family"])))
    print(len(set(sample["genus_species"])))

    sample.to_csv(
        f"./Wilf_eud_sample_{current_date}.csv",
        index=False,
    )


if __name__ == "__main__":
    # sample_families(eud_fams)
    # img_from_sample()
    # filter_to_angio_or_eud(eud_fams)
    sample_families(eud_fams)
