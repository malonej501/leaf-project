import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import itertools


DROP = 0  # drop first n steps

# values from stochastic_character_map in an_rec.R - averages from 100
# simulations
# Create the DataFrame with the specified index
phylo_transitions = pd.DataFrame({
    "transition": ["uu", "ul", "ud", "uc",
                   "lu", "ll", "ld", "lc",
                   "du", "dl", "dd", "dc",
                   "cu", "cl", "cd", "cc"],
    "count": [None, 43.42, 25.74, 90.68,
              4.92, None, 6.57, 3.54,
              5.46, 7.95, None, 4.56,
              7.67, 3.41, 2.17, None]
})


def get_walks():
    """Return the details of random walks along with transition type at each 
    step"""
    # walks = pd.read_csv("MUT2_320_mle_23-04-25.csv")
    walks = pd.read_csv("MUT5_320_mcmc_23-07-25_1.csv")
    # get transitions by shifting shape columns down by one and combining
    walks["prevshape"] = walks["shape"].shift(+1)
    walks["transition"] = walks["shape"] + walks["prevshape"]
    walks.loc[walks["step"] == 0, "transition"] = walks["first_cat"] + \
        walks["shape"]  # replace 0th step with first_cat + shape
    # Get all unique leafids and walkids (or use your reference list)
    all_leafids = walks['leafid'].unique()
    all_walkids = walks['walkid'].unique()
    # Generate all possible combinations
    all_combos = pd.DataFrame(list(itertools.product(all_leafids, all_walkids)),
                              columns=['leafid', 'walkid'])
    # Merge with your actual data to find missing combinations
    merged = all_combos.merge(walks[['leafid', 'walkid']], on=[
                              'leafid', 'walkid'], how='left', indicator=True)
    # Rows with _merge == 'left_only' are missing in your data
    missing = merged[merged['_merge'] == 'left_only']
    print(f"MISSING: {missing}")
    return walks


def get_phylo():
    """Return the shape data for phylogenetic trees"""
    phylo = pd.read_csv("../phylogeny/shape_data/labels_final/"
                        "zun_genus_phylo_nat_class_26-09-24.txt",
                        sep="\t", header=None, names=["genus", "shape"])
    # replace numerical values with letters
    phylo["shape"] = phylo["shape"].replace({0: "u", 1: "l", 2: "d", 3: "c"})
    return phylo


def plot_trans_freq_sim(log_scale=False, rm_self=False):
    """Plot the frequency of each transition type"""
    walks = get_walks()
    if DROP > 0:
        # drop first n steps
        walks = walks[walks["step"] >= DROP].reset_index()
    freq = walks["transition"].value_counts()  # count transitions
    print(freq)
    print(freq.sum())

    if rm_self:  # remove self transitions
        freq = freq.drop(["uu", "ll", "dd", "cc"])
        print(freq)
    if log_scale:
        freq = np.log(freq)
        print(freq)

    freq.plot(kind="bar")
    plt.xlabel("Transition")
    plt.ylabel("Proportion")
    plt.title("Simulation")
    plt.show()


def plot_trans_prop_sim(rm_self=False):
    """Plot the proportion of each transition type"""
    walks = get_walks()
    if DROP > 0:
        # drop first n steps
        walks = walks[walks["step"] >= DROP].reset_index()
    freq = walks["transition"].value_counts()  # count transitions

    if rm_self:  # remove self transitions before calculating proportion
        freq = freq.drop(["uu", "ll", "dd", "cc"])

    prop = freq / freq.sum()  # calculate proportion
    prop.plot(kind="bar")
    plt.xlabel("Transition")
    plt.ylabel("Proportion")
    plt.title("Simulation")
    plt.show()


def plot_shape_freq_phylo_sim(e=False):
    """Plot the frequency/proportion of each shape in the phylogenies and 
    simulation"""
    walks = get_walks()
    if DROP > 0:
        # drop first n steps
        walks = walks[walks["step"] >= DROP].reset_index()
    phylo = get_phylo()
    wfreq = walks["shape"].value_counts()  # count shapes
    wprop = (wfreq / wfreq.sum()).rename("Simulation")  # calculate proportion
    pfreq = phylo["shape"].value_counts()
    pprop = (pfreq / pfreq.sum()).rename("Phylogeny")
    freq = pd.concat([wfreq, pfreq], axis=1)
    prop = pd.concat([wprop, pprop], axis=1)

    prop.plot(kind="bar")
    plt.xlabel("Shape")
    plt.ylabel("Proportion")
    if e:
        plt.savefig(f"zun_MUT2.2-drop{DROP}_shape_prop.pdf")
    plt.show()


def plot_trans_prop_phylo_sim(rm_self=False, e=False):
    """Plot the proportion of each transition type in the phylogenies and
    simulation, either including diagonal transitions or not"""
    walks = get_walks()
    if DROP > 0:
        walks = walks[walks["step"] >= DROP].reset_index()
    wfreq = walks["transition"].value_counts()  # count transitions
    if rm_self:
        wfreq = wfreq.drop(["uu", "ll", "dd", "cc"])
    wprop = wfreq / wfreq.sum()  # calculate prop of transitions in simulation
    wprop = wprop.reset_index()
    wprop.columns = ["transition", "Simulation"]

    # calculate proportion of transitions in phylogeny
    phylo_transitions["prop"] = phylo_transitions["count"] / \
        phylo_transitions["count"].sum()
    pprop = phylo_transitions.drop("count", axis=1)
    pprop.columns = ["transition", "Phylogeny"]

    print(wprop)
    print(pprop)

    prop = pd.merge(wprop, pprop, on="transition",
                    how="outer")  # merge sim and phylo data
    prop = prop.sort_values("Simulation", ascending=False)
    prop = prop.set_index("transition")
    if rm_self:
        prop = prop.drop(["uu", "ll", "dd", "cc"])

    print(prop)

    prop.plot(kind="bar")
    plt.xlabel("Transition")
    plt.ylabel("Proportion")
    if e:
        plt.savefig(f"zun_MUT2.2-drop{DROP}_trans_prop.pdf")
    plt.show()


if __name__ == "__main__":
    plot_trans_prop_phylo_sim(rm_self=True)
    plot_shape_freq_phylo_sim()
    # plot_trans_freq_sim()
