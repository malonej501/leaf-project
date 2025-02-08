import pandas as pd
import numpy as np

files = [
    "sample_eud_13-1-24/img_labels_full_2024-01-14 15:55:27.817077.csv",
    "sample_eud_21-1-24/img_labels_full_2024-02-03 20:17:59.633309.csv",
]

labels = []

for file in files:
    label = pd.read_csv(file)
    labels.append(label)
full = labels[0].merge(labels[1], on="species")

n_conflicts = sum(full["shape_x"] != full["shape_y"])
prop_conflicts = n_conflicts / len(full)


conflicts_by_shape = []
n_shapes = []
for shape in ["u", "l", "d", "c"]:

    full_filt = full[full["shape_x"] == shape]
    shape_conflicts = sum((full_filt["shape_x"] != full_filt["shape_y"]))
    print(shape_conflicts)
    if shape_conflicts:
        prop_conflicts_by_shape = shape_conflicts / len(full_filt)
        conflicts_by_shape.append(prop_conflicts_by_shape)
    else:
        conflicts_by_shape.append(np.NaN)

print(f"Total conflicts = {n_conflicts}")
print(f"Proportion conflicts = {prop_conflicts}")
print(f"proportion conflicts by shape (u,l,d,c):\n\t{conflicts_by_shape}")

print(full["shape_x"].value_counts())
print(full["shape_y"].value_counts())
