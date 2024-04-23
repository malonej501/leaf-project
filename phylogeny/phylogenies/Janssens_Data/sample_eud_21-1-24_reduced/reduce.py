import pandas as pd

data = pd.read_csv("Naturalis_eud_sample_Janssens_intersect_labelled_21-01-24.csv")

print(data)
print(len(set(data["family"])))

species_per_family = 3

sample_dfs = []
for family in set(data["family"]):
    all_fam_rows = data[data["family"] == family]
    if len(all_fam_rows) > species_per_family:
        fam_samp = all_fam_rows.sample(n=species_per_family)
    else:
        fam_samp = all_fam_rows
    sample_dfs.append(fam_samp)

sample = pd.concat(sample_dfs, ignore_index=True)
print(sample)
sample.to_csv(
    "Naturalis_eud_sample_Janssens_intersect_labelled_21-01-24_reduced.csv", index=False
)
