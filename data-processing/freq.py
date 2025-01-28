from matplotlib import pyplot as plt
import pandas as pd
import numpy as np# values from stochastic_character_map in an_rec.R - averages from 100 simulations

# Create the DataFrame with the specified index
phylo_transitions = pd.DataFrame({
    "transition": ["uu", "ul", "ud", "uc", "lu", "ll", "ld", "lc", "du", "dl", "dd", "dc", "cu", "cl", "cd", "cc"],
    "count": [None, 43.42, 25.74, 90.68, 4.92, None, 6.57, 3.54, 5.46, 7.95, None, 4.56, 7.67, 3.41, 2.17, None]
})

def get_walks():
    """Return the details of random walks along with transition type at each step"""
    walks = pd.read_csv("MUT2.2.csv") # Not 100% sure if this is the correct file - perhaps some data is missing?
    walks["nextshape"] = walks["shape"].shift(+1) # get transitions by shifting shape columns down by one and combining
    walks["transition"] = walks["shape"] + walks["nextshape"]
    walks.loc[walks["step"] == 0, "transition"] = walks["first_cat"] + walks["shape"] # replace 0th step with first_cat + shape

    return walks

def get_phylo():
    """Return the shape data for phylogenetic trees"""
    phylo = pd.read_csv("../phylogeny/shape_data/labels_final/zun_genus_phylo_nat_class_26-09-24.txt", sep="\t", header=None, names=["genus","shape"])
    phylo["shape"] = phylo["shape"].replace({0:"u",1:"l",2:"d",3:"c"}) # replace numerical values with letters
    return phylo

def plot_trans_freq(log_scale=False, rm_self=False):
    """Plot the frequency of each transition type"""
    walks = get_walks()
    freq = walks["transition"].value_counts() # count transitions
   
    if rm_self: # remove self transitions
        freq = freq.drop(["uu","ll","dd","cc"])
        print(freq)
    if log_scale:
        freq = np.log(freq)
        print(freq)
    
    freq.plot(kind="bar")
    plt.xlabel("Transition")
    plt.ylabel("Proportion")
    plt.show()

def plot_trans_prop(rm_self=False):
    """Plot the proportion of each transition type"""
    walks = get_walks()
    freq = walks["transition"].value_counts() # count transitions

    if rm_self: # remove self transitions before calculating proportion
        freq = freq.drop(["uu","ll","dd","cc"])

    prop = freq / freq.sum() # calculate proportion    
    prop.plot(kind="bar")
    plt.xlabel("Transition")
    plt.ylabel("Proportion")
    plt.show()

def plot_phylo_sim_shape_freq(drop=0, e=False):
    """Plot the frequency/proportion of each shape in the phylogenies and simulation"""
    walks = get_walks()
    if drop > 0:
        walks = walks[walks["step"] >= drop].reset_index() # drop first n steps
    phylo = get_phylo()
    wfreq = walks["shape"].value_counts() # count shapes
    wprop = (wfreq / wfreq.sum()).rename("Simulation") # calculate proportion
    pfreq = phylo["shape"].value_counts()
    pprop = (pfreq / pfreq.sum()).rename("Phylogeny")
    freq = pd.concat([wfreq,pfreq], axis=1)
    prop = pd.concat([wprop,pprop], axis=1)

    

    prop.plot(kind="bar")
    plt.xlabel("Shape")
    plt.ylabel("Proportion")
    plt.savefig(f"zun_MUT2.2-drop{drop}_shape_prop.pdf") if e else None
    plt.show()

def plot_phylo_sim_trans_prop(drop=0, rm_self=False, e=False):
    walks = get_walks()
    if drop > 0:
        walks = walks[walks["step"] >= drop].reset_index()
    wfreq = walks["transition"].value_counts() # count transitions
    if rm_self:
        wfreq = wfreq.drop(["uu","ll","dd","cc"])
    wprop = wfreq / wfreq.sum() # calculate proportion of transitions in simulation
    wprop = wprop.reset_index()
    wprop.columns = ["transition","Simulation"]

    phylo_transitions["prop"] = phylo_transitions["count"] / phylo_transitions["count"].sum() # calculate proportion of transitions in phylogeny
    pprop = phylo_transitions.drop("count", axis=1)
    pprop.columns = ["transition","Phylogeny"]

    print(wprop)
    print(pprop)

    prop = pd.merge(wprop, pprop, on="transition", how="outer") # merge sim and phylo data
    prop = prop.set_index("transition")
    if rm_self:
        prop = prop.drop(["uu","ll","dd","cc"])
    
    print(prop)
    

    prop.plot(kind="bar")
    plt.xlabel("Transition")
    plt.ylabel("Proportion")
    plt.savefig(f"zun_MUT2.2-drop{drop}_trans_prop.pdf") if e else None
    plt.show()

if __name__ == "__main__":
    # plot_phylo_sim_trans_prop(rm_self=True, drop=0)
    plot_phylo_sim_shape_freq(drop=0)