from pdict import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil

wd = "/home/m/malone/GitHub/leaf-project/vlab-5.0-3609-ubuntu-20_04/oofs/ext/NPHLeafModels_1.01"
if os.path.exists(wd + "/LeafGenerator/pplots"):
    shutil.rmtree(wd + "/LeafGenerator/pplots")
os.makedirs(wd + "/LeafGenerator/pplots")

pdict_sub = {}

for key, value in pdict.items():
    if "true" not in value and "false" not in value and "M_PI" not in value:
        pdict_sub[key] = value

print(pdict_sub)
print(pdict_sub.values())

# pdict_subf = pd.DataFrame(pdict_sub)

for key, value in pdict_sub.items():
    x_values = range(len(value))
    plt.bar(x_values, value)
    plt.title(key)
    plt.savefig(wd + f"/LeafGenerator/pplots/{key}.png")
    plt.clf()

exit()
print(pdict_sub)
exit()


# pval = list(pdict.values())
p_val = []
p_keys = []
for i, value in enumerate(pdict.values()):
    print(value)
    if "true" not in value or "false" not in value or "M_PI" not in value:
        p_val.append(value)
        p_keys.append(pdict.keys()[i])

print(p_val)
