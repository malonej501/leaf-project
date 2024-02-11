import pandas as pd
from datetime import datetime

species_per_family = 200  # specify no. species to sample per angiosperm family - if there aren't enough species in the database, it will take the maximum number.

angio_fams = pd.read_csv("../APG_IV/APG_IV_ang_fams.csv")
eud_fams = pd.read_csv("../APG_IV/APG_IV_eud_fams.csv")

current_date = datetime.now().strftime("%d-%m-%y")


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


if __name__ == "__main__":
    # sample_families(eud_fams)
    img_from_sample()
