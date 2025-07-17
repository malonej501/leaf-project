import os
from datetime import datetime
import urllib.request
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


# specify no. species to sample per angiosperm family - if there aren't
# enough species in the database, it will take the maximum number.
SP_PER_FAM = -1  # set to -1 to disable subsampling

FILTER = "ang"  # "ang" for angiosperms, "eud" for eudicots
ANGIO_FAMS = pd.read_csv("../APG_IV/APG_IV_ang_fams.csv")
EUD_FAMS = pd.read_csv("../APG_IV/APG_IV_eud_fams.csv")
CHUNK_SIZE = 100000  # For reading large .csv files


CURR_DATE = datetime.now().strftime("%d-%m-%y")

START_FROM = 0


def filter_to_angio_or_eud():
    """Filter the occurrence data to only include angiosperm or eudicot
    lineages and save to .csv."""

    # Initialize an empty list to store the intersected dataframes
    intersect_dfs = []

    # Read and process the data in chunks
    for i, chunk_occurrence in enumerate(
        pd.read_csv(
            "botany-20240108.dwca/Occurrence.txt",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
    ):
        print(f"Row number: {i * CHUNK_SIZE}")
        filt = ANGIO_FAMS if FILTER == "ang" else EUD_FAMS
        # intersect with angio or eud fams for the current chunk
        intersect_chunk = pd.merge(
            chunk_occurrence, filt, on="family", how="inner"
        )

        # Subset chunk to rows representing a species
        intersect_chunk = intersect_chunk[
            intersect_chunk["taxonRank"] == "species"]

        # Append the intersected chunk to the list
        intersect_dfs.append(intersect_chunk)

    # Concatenate the list of intersected dataframes into a single dataframe
    intersect = pd.concat(intersect_dfs, ignore_index=True)

    intersect.to_csv(
        "~/Documents/Leaf Project/Naturalis/" +
        "Naturalis_ang_species_occurrence.csv",
        index=False,
    )


def sample_families():
    """Sample a specified number of species from each family in the
    sample_fams dataframe and save to a .csv file."""

    sample_fams = ANGIO_FAMS if FILTER == "ang" else EUD_FAMS

    print("Reading data...")
    ang_sp_full = pd.read_csv(
        "botany-20240108.dwca/Naturalis_ang_species_occurrence.csv",
        low_memory=False,
    )
    print("Done!")

    print("Removing duplicate species...")
    sp_list = ang_sp_full["genus"].str.cat(
        ang_sp_full["specificEpithet"], sep="_")
    ang_sp_full.insert(0, "species", sp_list)
    ang_sp_full_clean = ang_sp_full.drop_duplicates(
        subset="species", keep="first"
    ).reset_index(
        drop=True
    )  # remove duplicate species
    ang_sp_full_clean = ang_sp_full_clean.drop_duplicates(
        subset="id", keep="first"
    ).reset_index(
        drop=True
    )  # remove any completely identical rows
    print("Done!")

    sample_dfs = []

    if SP_PER_FAM != -1:
        for i, family in enumerate(sample_fams["family"]):
            print(i, family)
            fam = ang_sp_full_clean[ang_sp_full_clean["family"] == family]
            if len(fam) >= SP_PER_FAM:
                fam_samp = fam.sample(n=SP_PER_FAM)
            else:
                fam_samp = fam.sample(n=len(fam))
            sample_dfs.append(fam_samp)

        sample = pd.concat(sample_dfs, ignore_index=True)
    else:
        sample = ang_sp_full_clean  # no subsampling

    sample = sample.rename(
        columns={"id": "CoreId"}
    )  # do this because the variable is called CoreId in Multimedia.txt

    sample.to_csv(
        f"./Naturalis_occurrence_{FILTER}_sppfam{SP_PER_FAM}_{CURR_DATE}.csv",
        index=False,
    )


def img_from_sample():
    """Get the intersection of occurrence data and multimedia data
    based on CoreId and save to a .csv file."""

    print("Reading data...")
    # sample = pd.read_csv(
    #     "jan_zun_nat_ang_09-10-24/Naturalis_occurrence
    # _ang_sppfam-1_11-07-25.csv"
    # )
    sample_fname = "Naturalis_occurrence_ang_sppfam-1_11-07-25"
    sample = pd.read_csv(sample_fname + ".csv")
    print("Done!")

    multimedia = pd.read_csv("botany-20240108.dwca/Multimedia.txt")

    # Shouldn't be any duplicate species as merging on CoreId, not species
    intersect = pd.merge(multimedia, sample, on="CoreId", how="inner")

    intersect.to_csv(
        f"./{sample_fname}_multimedia.csv",
        index=False,
    )


def download_imgs():
    """Download multimedia sample from Naturalis database to download_imgs/"""
    if os.listdir("download_imgs"):
        print("download_imgs is not empty! Terminating.")

    else:
        # intersect = pd.read_csv(
        #     "sample_eud_21-1-24/Naturalis_eud_sample_Janssens_
        # intersect_21-01-24.csv"
        # )
        intersect = pd.read_csv(
            "jan_zun_nat_ang_11-07-25/jan_zun_union_nat_species_11-07-25.csv")
        print(f"Downloading {len(intersect)} images...")

        failed_indicies = []
        for index, row in intersect.iloc[START_FROM:].iterrows():
            try:
                species = row["species"]
                print(index, species)
                url = row["accessURI"]
                urllib.request.urlretrieve(url, "temp.png")
                img = Image.open(r"temp.png")
                img.save(f"download_imgs/{species}{index}.png")
            except (urllib.error.URLError, OSError) as e:
                failed_indicies.append(index)
                print(f"Error: Download failed - {e}")

        failed_to_download = intersect.iloc[failed_indicies]
        if not failed_to_download.empty:
            failed_to_download.to_csv("download_failed.csv", index=False)


def get_all_genera():
    """Return the taxon ranks of all genera in the Naturalis database"""

    # Initialize an empty list to store the intersected dataframes
    unique_genera_dfs = []

    # Read and process the data in chunks
    for i, chunk_occurrence in enumerate(
        pd.read_csv(
            "botany-20240108.dwca/Occurrence.txt",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
    ):
        print(f"Row number: {i * CHUNK_SIZE}")
        unique_genera_chunk = chunk_occurrence.drop_duplicates(
            subset=["class", "order", "family", "genus"], keep="first")
        print(f"Dropped {len(chunk_occurrence) - len(unique_genera_chunk)} " +
              "duplicates")

        unique_genera_dfs.append(unique_genera_chunk)

    # Concatenate the list of intersected dataframes into a single dataframe
    concat = pd.concat(unique_genera_dfs, ignore_index=True)
    print(f"Total number of genera: {len(concat)}")

    # Drop any remaining duplicated genera
    unique_genera = concat.drop_duplicates(
        subset=["class", "order", "family", "genus"], keep="first")
    print(f"Dropped further {len(concat) - len(unique_genera)} duplicates")

    # Subset to just taxon rank columns
    unique_genera = unique_genera[["class", "order", "family", "genus"]]

    # Remove rows containing missing values
    unique_genera_clean = unique_genera.dropna()
    print(f"Dropped {len(unique_genera) - len(unique_genera_clean)} " +
          "rows with missing values")
    print(f"Final number of genera: {len(unique_genera_clean)}")

    unique_genera_clean.to_csv(
        f"Naturalis_unique_genera_{CURR_DATE}.csv",
        index=False,
    )


def plot_taxon_distribution():
    """Plot the no. taxa per family in the specified samples of from
    Naturalis."""
    # datapath1 = "sample_eud_zuntini_10-09-24/Naturalis_multimedia_eud_sample
    # _10-09-24_zuntini_intercept_genera_labelled.csv"
    # datapath2 = "sample_eud_21-1-24/Naturalis_eud_sample_Janssens_intersect
    # _labelled_21-01-24.csv"
    # datapath1 = "sample_eud_zun_equal_fam_16-09-24/Naturalis_multimedia_eud
    # _sample_10-09-24_zuntini_intercept_genera_labelled_equal_fam.csv"
    # datapath2 = "sample_eud_jan_equal_fam_16-09-24/Naturalis_eud_sample
    # _Janssens_intersect_labelled_21-01-24_equal_fam.csv"
    # datapath1 = "jan_zun_nat_ang_26-09-24/jan_zun_union_nat_genus
    # _labelled.csv"
    # datapath2 = "jan_zun_nat_ang_26-09-24/jan_nat_genus.csv"
    # datapath3 = "jan_zun_nat_ang_26-09-24/zun_nat_genus.csv"
    datapath1 = "jan_zun_nat_ang_26-09-24/jan_nat_genus_labelled.csv"
    datapath2 = "jan_zun_nat_ang_26-09-24/zun_nat_genus_labelled.csv"

    datapaths = [datapath1, datapath2]  # , datapath3]
    samples = [pd.read_csv(datapath) for datapath in datapaths]
    # u = samples[0]
    # js = samples[1]
    # zs = samples[2]
    # j = u[u["CoreId"].isin(js["CoreId"])]
    # j.to_csv("jan_zun_nat_ang_26-09-24/jan_nat_genus_labelled.csv",
    # index=False)
    # print(len(j))
    # z = u[u["CoreId"].isin(zs["CoreId"])]
    # z.to_csv("jan_zun_nat_ang_26-09-24/zun_nat_genus_labelled.csv",
    # index=False)
    # print(len(z))
    # exit()

    datanames = []
    for datapath in datapaths:
        datanames.append(datapath.split("/")[1].split(".")[0])
    # data_full = pd.concat(samples)
    # print(data_full)
    freqs = [pd.DataFrame(sample["family"].value_counts())
             for sample in samples]
    for i, freq in enumerate(freqs):
        freq.rename(columns={"count": f"count_{datanames[i]}"}, inplace=True)
    freqs_merged = pd.merge(*freqs, on="family", how="outer").reset_index()
    freqs_merged.sort_values(
        by=freqs_merged.columns[1], ascending=False, inplace=True)
    print(datanames)
    print(freqs)
    print(freqs_merged)
    # plt.figure(figsize=(18, 6))
    _, ax = plt.subplots(figsize=(18, 6))
    index = np.arange(len(freqs_merged["family"]))
    bar_width = 0.35
    ax.bar(
        index - bar_width / 2,
        freqs_merged[f"count_{datanames[0]}"],
        bar_width,
        label=f"count_{datanames[0]}",
        color=sns.color_palette("colorblind")[0],
    )
    ax.bar(
        index + bar_width / 2,
        freqs_merged[f"count_{datanames[1]}"],
        bar_width,
        label=f"count_{datanames[1]}",
        color=sns.color_palette("colorblind")[1],
    )
    ax.set_xlabel(f"Family N={len(freqs_merged)}")
    ax.set_ylabel("Count")
    ax.set_xticks(index)
    ax.set_xticklabels(freqs_merged["family"])
    ax.legend(loc="upper right")
    plt.xticks(rotation=90, fontsize=6)
    plt.subplots_adjust(bottom=0.205, left=0.035, right=0.985)
    plt.show()
    # freq.plot(
    #     kind="bar", xlabel=f"Family N={len(freq)}", ylabel="Count",
    # title=dataname
    # )
    # plt.xticks(fontsize=6)
    # plt.tight_layout
    # plt.subplots_adjust(bottom=0.205, left=0.035, right=0.985)
    # plt.show()


def taxon_difference(level):
    """Find the intersection of the specified taxon level between two
    samples of Naturalis data and print the results."""

    datapath1 = ("sample_eud_zuntini_10-09-24/Naturalis_multimedia_eud_sample"
                 "_10-09-24_zuntini_intercept_genera_labelled.csv")
    datapath2 = ("sample_eud_21-1-24/Naturalis_eud_sample_Janssens_intersect"
                 "_labelled_21-01-24.csv")
    zun_nat = pd.read_csv(datapath1)
    jan_nat = pd.read_csv(datapath2)

    zun = set(zun_nat[level])
    jan = set(jan_nat[level])

    intersect = zun & jan

    print(intersect)
    print(len(intersect))
    print(zun_nat)

    return intersect


def equalise_taxon_sample(level, export):
    """Equalise the no. samples of the specified taxon level
    between two samples of Naturalis data and save the results to .csv 
    files."""

    datapath1 = ("sample_eud_zuntini_10-09-24/Naturalis_multimedia_eud_sample"
                 "_10-09-24_zuntini_intercept_genera_labelled.csv")
    datapath2 = ("sample_eud_21-1-24/Naturalis_eud_sample_Janssens_intersect_"
                 "labelled_21-01-24_2416.csv")
    zun_nat = pd.read_csv(datapath1)
    jan_nat = pd.read_csv(datapath2)
    print(len(set(jan_nat["species"])))

    zun_nat_counts = pd.DataFrame(zun_nat[level].value_counts()).reset_index()
    jan_nat_counts = pd.DataFrame(jan_nat[level].value_counts()).reset_index()
    data_all = pd.merge(zun_nat_counts, jan_nat_counts, on=level, how="outer")
    data_all["min"] = data_all[["count_x", "count_y"]].min(axis=1)
    data_all.dropna(inplace=True)
    data_all.reset_index(drop=True, inplace=True)
    # pd.set_option("display.max_rows", None)  # Show all rows
    print(data_all)
    print(sum(data_all["min"]))

    zun_nat_reduced = pd.DataFrame()
    jan_nat_reduced = pd.DataFrame()
    for _, row in data_all.iterrows():
        tax_group = row[level]
        count = row["min"]

        zun_nat_filt = zun_nat[zun_nat[level] == tax_group]
        zun_nat_filt_samp = zun_nat_filt.sample(n=int(count), replace=False)
        zun_nat_reduced = pd.concat([zun_nat_reduced, zun_nat_filt_samp])

        jan_nat_filt = jan_nat[jan_nat[level] == tax_group]
        jan_nat_filt_samp = jan_nat_filt.sample(n=int(count), replace=False)
        jan_nat_reduced = pd.concat([jan_nat_reduced, jan_nat_filt_samp])

    zun_nat_reduced.reset_index(drop=True, inplace=True)
    jan_nat_reduced.reset_index(drop=True, inplace=True)
    if export:
        zun_nat_reduced.to_csv(
            "Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept_" +
            f"genera_labelled_equal_{level}.csv",
            index=False,
        )
        jan_nat_reduced.to_csv(
            "Naturalis_eud_sample_Janssens_intersect_labelled_21-01-24_" +
            f"equal_{level}.csv",
            index=False,
        )

    print(zun_nat_reduced)
    print(jan_nat_reduced)

    z_labels = zun_nat_reduced[["genus", "shape"]]
    print(z_labels)
    if export:
        z_labels.to_csv(
            f"zuntini_genera_equal_{level}_phylo_nat_class.txt",
            sep="\t",
            index=False,
            header=False,
        )
    j_labels = jan_nat_reduced[["species", "shape"]]
    print(j_labels)
    if export:
        j_labels.to_csv(
            f"jan_equal_{level}_phylo_nat_class.txt",
            sep="\t",
            index=False,
            header=False,
        )


if __name__ == "__main__":
    # sample_families()
    img_from_sample()
    # download_imgs()
    # plot_taxon_distribution()
    # taxon_difference(level="family")
    # equalise_taxon_sample(level="genus", export=False)
    # get_all_genera()
