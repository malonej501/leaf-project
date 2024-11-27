import pandas as pd
import os
import re
from datetime import datetime

wd = "jan_zun_nat_ang_26-09-24"
file_name = "jan_zun_union_nat_genus.csv"
data_name = file_name.split(".")[0]


shapes = pd.DataFrame()

filelist = []
for file in os.listdir(wd):
    if "img_labels" in file and file.endswith(".csv") and "full" not in file:
        filelist.append(file)

sorted_file_list = sorted(
    filelist,
    key=lambda x: int(re.search(r"\d+", x).group()),
)

for file in sorted_file_list:
    if "img_labels" in file and file.endswith(".csv") and "full" not in file:
        print(file)
        df = pd.read_csv(f"./{wd}/{file}")
        shapes = pd.concat([shapes, df], ignore_index=True)

print(set(shapes["shape"]))

# shapes.replace("y", "a", inplace=True)
# shapes.replace("f", "a", inplace=True)
# shapes.replace("s", "a", inplace=True)
# shapes.replace("v", "a", inplace=True)

print(len(shapes))
print(shapes["shape"].value_counts())

# remove numerical name component
shapes["species"] = shapes["species"].replace("\d+", "", regex=True)
# drop any remaining duplicates
shapes.drop_duplicates(subset="species", keep="first", inplace=True)
shapes.to_csv(f"./{data_name}_labels_full.csv", index=False)

# drop ambiguous species
shapes_unambig = shapes[shapes["shape"] != "a"].reset_index(drop=True)
print(f"Unambiguous species count: {len(shapes_unambig)}")

shapes_unambig["shape"].replace({"u": 0, "l": 1, "d": 2, "c": 3}, inplace=True)
# print(shapes_unambig)

shapes_unambig.to_csv(
    f"./{data_name}_labels_unambig_full.csv", sep="\t", index=False, header=False
)

species_full = pd.read_csv(os.path.join(wd, file_name))
# drop any duplicatesikk
species_full_clean = species_full.drop_duplicates(subset="species", keep="first")


species_full_labelled = pd.merge(
    species_full_clean, shapes_unambig, on="species", how="inner"
)  # .reset_index(drop=True)
print(species_full_labelled)
# print(species_full_clean["species"].value_counts())


species_full_labelled.to_csv(
    f"./{data_name}_labelled.csv",
    index=False,
)
