import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.stats import kruskal
from scipy import linalg
import copy
import sympy as sp
from sympy import *

wd = "mcmc/unif_wilf_20-02-24/"
filename = "mcmc_unif_20-02-24_wilf.csv"

##### Getting Probs #####


T = 0.1  # set value of T to enumerate probabilities
N_trees = 1  # declare the number of trees used to estimate the parameters

rates_full = pd.read_csv(wd + filename)
# print(rates_full)
prob_tab = pd.DataFrame(
    {
        "p00": [],
        "p01": [],
        "p02": [],
        "p03": [],
        "p10": [],
        "p11": [],
        "p12": [],
        "p13": [],
        "p20": [],
        "p21": [],
        "p22": [],
        "p23": [],
        "p30": [],
        "p31": [],
        "p32": [],
        "p33": [],
    }
)

#### Insert new columns for stationary rates

rates_full_wstat = copy.deepcopy(rates_full)
rates_full_wstat.insert(0, "q00", 0 - rates_full.iloc[:, 0:3].sum(axis=1))
rates_full_wstat.insert(5, "q11", 0 - rates_full.iloc[:, 3:6].sum(axis=1))
rates_full_wstat.insert(10, "q22", 0 - rates_full.iloc[:, 6:9].sum(axis=1))
rates_full_wstat.insert(15, "q33", 0 - rates_full.iloc[:, 9:12].sum(axis=1))
# print(rates_full_wstat)

#### Calculate Probabilities give t=0.1


def matrixfromrow(dataframe, i):
    row = dataframe.iloc[i].values
    rate_matrix = row.reshape(4, 4)
    return rate_matrix


def rowfrommatrix(matrix, prob_tab):
    row_arr = matrix.reshape(1, 16)
    row_df = pd.DataFrame(row_arr, columns=list(prob_tab.columns))
    return row_df


def getprobs(Q):
    ## Manual Way
    # t = sp.symbols("t")
    # # D - eigenvalues, C - eigenvectors
    # D, C = np.linalg.eig(Q)
    # Cinv = np.linalg.inv(C)
    # Ddiag = np.diagflat(np.array([sp.exp(val * t) for val in D]))
    # P = np.matmul(np.matmul(C, Ddiag), Cinv)
    # evaluate = np.vectorize(lambda expr: expr.subs(t, T))
    # Peval = evaluate(P)

    ## Quick Way
    Peval = scipy.linalg.expm(Q * T)
    # exit()
    return Peval


def rates_probs_mean(prob_tab):

    for i in range(0, len(rates_full_wstat)):
        rates = matrixfromrow(rates_full_wstat, i)
        probs = getprobs(rates)
        row = rowfrommatrix(probs, prob_tab)
        prob_tab = pd.concat([prob_tab, row], ignore_index=True)

    print(prob_tab)
    prob_tab.to_csv(wd + f"probs_t{T}_{filename}", index=false)

    return prob_tab


def rates_mean_probs(prob_tab):
    print(rates_full_wstat)
    print(rates_full_wstat.mean())
    means = rates_full_wstat.mean()
    means_reshape = np.reshape(means, (4, 4))
    probs = getprobs(means_reshape)
    print(probs)
    prob_tab = rowfrommatrix(probs, prob_tab)
    prob_tab.to_csv(wd + f"probs_t{T}_{filename}.csv", index=false)


##### Plotting #####


sns.set_palette("colorblind")
order = ["u", "l", "d", "c"]


#### Data processing


def translong(data, dtype):
    full_trans = data.T.reset_index(names="transition")
    full_trans["first_cat"] = [str[1] for str in full_trans["transition"]]
    full_trans["first_cat"] = full_trans["first_cat"].replace(
        {"0": "u", "1": "l", "2": "d", "3": "c"}
    )
    full_trans["last_cat"] = [str[2] for str in full_trans["transition"]]
    full_trans["last_cat"] = full_trans["last_cat"].replace(
        {"0": "u", "1": "l", "2": "d", "3": "c"}
    )

    if dtype == "rate":
        full_long = pd.melt(
            full_trans,
            id_vars=["transition", "first_cat", "last_cat"],
            var_name="tree",
            value_name="rate",
        )
    elif dtype == "prob":
        full_long = pd.melt(
            full_trans,
            id_vars=["transition", "first_cat", "last_cat"],
            var_name="tree",
            value_name="rate",
        )
    else:
        return ValueError("incorrect dtype")
    return full_trans, full_long


probs_full = rates_probs_mean(prob_tab)
rates_full_trans, rates_full_long = translong(rates_full, "rate")
probs_full_trans, probs_full_long = translong(probs_full, "prob")
print(rates_full_long)

#### Plot Uncertainty of Probabilities

colours = sns.color_palette("colorblind")


def box1():
    sns.boxplot(data=probs_full, orient="v", palette=colours)
    plt.xlabel("Transition type")
    plt.ylabel(f"Probability (t={T})")
    plt.title(f"Uncertainty of pobabilities\nN={N_trees}, t={t}, {filename}")
    # sns.violinplot(data=rates_full, inner="quart")
    plt.savefig(wd + f"prob_uncert_t{T}_{filename}.png")
    plt.show()
    plt.clf()


def catplot1():
    order = ["u", "l", "d", "c"]
    labels = ["unlobed(u)", "lobed(l)", "dissected(d)", "compound(c)"]

    probs_full_long_sorted = probs_full_long.sort_values(
        by=["last_cat"], key=lambda x: x.map({v: i for i, v in enumerate(order)})
    )

    probs_full_long_sorted_uniq = probs_full_long_sorted[
        probs_full_long_sorted["first_cat"] != probs_full_long_sorted["last_cat"]
    ]

    g = sns.catplot(
        x="first_cat",
        y="rate",
        hue="last_cat",
        data=probs_full_long_sorted,
        kind="bar",
        # kind="violin",
        # density_norm="width",
        order=order,
        palette=colours,
        height=5,
        aspect=1,
    )
    g.set_xticklabels(labels=labels)
    g.set_axis_labels("Initial Shape", "Mean Probability")
    g._legend.set_title("Final Shape")
    g.fig.suptitle(f"Geeta et al. (2012)\nN={N_trees}, t={T}, {filename}")
    plt.xticks(fontsize=9)
    plt.ylim(0, 1)
    plt.subplots_adjust(right=0.8, top=0.89, bottom=0.15)
    plt.savefig(wd + f"probs_uncert_t{T}_{filename}.png", dpi=100)
    plt.clf()


#### Plot Uncertainty of Rates


def catplot2():
    order = ["u", "l", "d", "c"]
    labels = ["unlobed(u)", "lobed(l)", "dissected(d)", "compound(c)"]

    rates_full_long_sorted = rates_full_long.sort_values(
        by=["last_cat"], key=lambda x: x.map({v: i for i, v in enumerate(order)})
    )

    rates_full_long_sorted_uniq = rates_full_long_sorted[
        rates_full_long_sorted["first_cat"] != rates_full_long_sorted["last_cat"]
    ]

    g = sns.catplot(
        x="first_cat",
        y="rate",
        hue="last_cat",
        data=rates_full_long_sorted,
        kind="bar",
        order=order,
        palette=colours,
        height=5,
        aspect=1,
    )
    g.set_xticklabels(labels=labels)
    g.set_axis_labels("Initial Shape", "Evolutionary rate")
    g._legend.set_title("Final Shape")
    g.fig.suptitle(f"Uncertainty of rates\nN={N_trees}, {filename}")
    plt.xticks(fontsize=9)
    plt.subplots_adjust(right=0.8, top=0.89, bottom=0.15)
    plt.savefig(wd + f"rates_uncert_{filename}.png", dpi=100)
    plt.clf()


def box2():
    sns.boxplot(data=rates_full, orient="v", palette=colours)
    plt.xlabel("Rate parameter")
    plt.ylabel("Evolutionary rate")
    plt.title(f"Uncertainty of ML evolutionary rates\nN={N_trees} {filename}")
    # sns.violinplot(data=rates_full, inner="quart")
    plt.savefig(wd + f"rates_uncert_{filename}.png")
    plt.show()
    plt.clf()


def curves_phylogeny():

    tee = sp.symbols("t")

    t_vals = np.linspace(0, 0.2, 500)

    QMCMC_df = rates_full
    QMCMC_mean = pd.DataFrame(rates_full_wstat.mean()).transpose()
    print(QMCMC_mean)
    QMCMC = matrixfromrow(QMCMC_mean, 0)
    print(QMCMC)

    results = []

    for t_val in t_vals:
        result = np.array([])
        QMCMC_t = QMCMC * t_val
        result = linalg.expm(QMCMC_t)
        results.append(result)

    plot_data = {"t": [], "first_cat": [], "shape": [], "P": []}

    for i, matrix in enumerate(results):
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                plot_data["t"].append(t_vals[i])
                plot_data["first_cat"].append(row)
                plot_data["shape"].append(column)
                plot_data["P"].append(matrix[row, column])

    plot_data = pd.DataFrame(plot_data)
    # mapping = {"u": 0, "l": 1, "d": 2, "c": 3}
    mapping = {0: "u", 1: "l", 2: "d", 3: "c"}
    plot_data["first_cat"].replace(mapping, inplace=True)
    plot_data["shape"].replace(mapping, inplace=True)
    print(plot_data)

    g = sns.relplot(
        data=plot_data,
        x="t",
        y="P",
        col="first_cat",
        hue="shape",
        kind="line",
        col_wrap=2,
        col_order=order,
        hue_order=order,
        facet_kws={"sharey": False},
    )  # .fig.suptitle(filename)
    plt.show()


def kruskal_test(rates_full):
    kruskal_result = kruskal(
        rates_full["q01"],
        rates_full["q02"],
        rates_full["q03"],
        rates_full["q10"],
        rates_full["q12"],
        rates_full["q13"],
        rates_full["q20"],
        rates_full["q21"],
        rates_full["q23"],
        rates_full["q30"],
        rates_full["q31"],
        rates_full["q32"],
    )

    print(kruskal_result)


# rates_probs_mean(prob_tab)
# catplot1()
# catplot2()

curves_phylogeny()
catplot1()
catplot2()
