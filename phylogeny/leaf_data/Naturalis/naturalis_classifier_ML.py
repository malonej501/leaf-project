import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from fastai.vision.all import *
import os

img_wd = "sample_eud_21-1-24/Naturalis_eud_sample_Janssens_intersect_21-01-24/"
img_labels = pd.read_csv(
    "sample_eud_21-1-24/img_labels_unambig_full.csv",
    sep="\t",
    names=["species", "shape"],
)
print(img_labels)
train_labels = img_labels.sample(n=len(img_labels) // 4)
print(train_labels)

# return the full file names of of the training set
train_imgs = []
for partial_str in train_labels["species"]:
    train_imgs.extend(
        [
            full_str
            for i, full_str in enumerate(os.listdir(img_wd))
            if partial_str in full_str
        ]
    )

print(train_imgs[0:10])
print(len(train_imgs))
print(len(set(train_imgs)))


def get_label(filename):
    label = ""
    for i, partial_str in enumerate(img_labels["species"]):
        if partial_str in filename:
            label = str(img_labels["shape"][i])
            break
    return label


dls = ImageDataLoaders.from_name_func(
    img_wd,
    get_image_files(img_wd),
    valid_pct=0.4,  # 40% for testing
    seed=1,
    label_func=get_label,
    item_tfms=Resize(224),
)

# dls.valid.show_batch()

learn = cnn_learner(dls, resnet34, metrics=error_rate, pretrained=True)
learn.fine_tune(epochs=10)
learn.export()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(6)
