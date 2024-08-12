import pandas as pd
import seaborn as sns

data = pd.read_csv("Janssens_Data/sample_eud_21-1-24/img_labels_unambig_full_21-1-24.txt", sep="\t", header=None, names=["species","shape"])

nat_data = pd.read_csv("Janssens_Data/sample_eud_21-1-24/Naturalis_eud_sample_Janssens_intersect_labelled_21-01-24.csv")

order_labels = pd.read_csv("Janssens_Data/sample_eud_21-1-24/order_labels.csv")
print(nat_data)
print(nat_data["genus"])
print(nat_data["specificEpithet"])
nat_data["species"] = nat_data["genus"] + "_" + nat_data["specificEpithet"]
nat_sub = nat_data.loc[:, ["species","order","shape"]]

# data.reset_index(names="node", inplace=True)
print(data)
print(nat_sub)



data.insert(1, "command", "range")
nat_sub.insert(1, "command", "range")
nat_sub.insert(1, "end", nat_sub["species"])
#order_labels.insert(2, "command", "range")
order_labels.insert(2, "line_width", "1")
order_labels.insert(2, "line_style", "solid")
order_labels.insert(2, "line_colour", "#000000")
order_labels.insert(2, "gradient_colour", "#000000")
order_labels.insert(2, "fill_colour", "#000000")
order_labels["label_colour"] = "#000000"
order_labels["label_size"] = "40"



palette = sns.color_palette("colorblind", as_cmap=True)

cmap = {0:palette[0],1:palette[1],2:palette[2],3:palette[3]}
shapemap = {0:"Unlobed",1:"Lobed",2:"Dissected",3:"Compound"}

data["colour"] = data["shape"].map(cmap)
data["label"] = data["shape"].map(shapemap)


#nat_sub["pos"] = -1
#nat_sub["colour"] = "#000000"
#nat_sub["style"] = "normal"
#nat_sub["size"] = 1
#nat_sub["rotation"] = 0


data.drop(columns = ["shape"], inplace=True)
nat_sub.drop(columns = ["shape"], inplace=True)
print(data)
print(nat_sub)
print(order_labels)
print(len(set(nat_sub["order"])))
#nat_sub = nat_sub.sort_values(by="order")
#print(nat_sub)


# data.to_csv("jan_phylo_nat_class_shapeannotations.txt",sep="\t",header=False,index=False)
# nat_sub.to_csv("jan_phylo_nat_class_dataset_text.txt",sep="\t",header=False,index=False)
order_labels.to_csv("jan_phylo_nat_class_order_annotations.txt",sep="\t",header=False,index=False)
