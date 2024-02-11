import pandas as pd

angio_fams = pd.read_csv("~/Documents/Leaf Project/APG_IV/APG_IV_families_clean.csv")

chunk_size = 100000  # Adjust the chunk size as needed


def generate_intersect():
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
