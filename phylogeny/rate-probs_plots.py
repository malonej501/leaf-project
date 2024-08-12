import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.stats import kruskal
from scipy import linalg
import copy
import os
import sympy as sp
from sympy import *

##### Getting Rates #####

rates_map = {
    "q01": ("u", "l"),
    "q02": ("u", "d"),
    "q03": ("u", "c"),
    "q10": ("l", "u"),
    "q12": ("l", "d"),
    "q13": ("l", "c"),
    "q20": ("d", "u"),
    "q21": ("d", "l"),
    "q23": ("d", "c"),
    "q30": ("c", "u"),
    "q31": ("c", "l"),
    "q32": ("c", "d"),
}

rates_map1 = {
    "0": ("u", "l"),
    "1": ("u", "d"),
    "2": ("u", "c"),
    "3": ("l", "u"),
    "4": ("l", "d"),
    "5": ("l", "c"),
    "6": ("d", "u"),
    "7": ("d", "l"),
    "8": ("d", "c"),
    "9": ("c", "u"),
    "10": ("c", "l"),
    "11": ("c", "d"),
}


def get_rates_single():
    wd = "Geeta/mcmc/Geeta_23-04-24/"
    filename = "561AngLf09_D.csv"
    rates_full = pd.read_csv(wd + filename)

    #### Insert new columns for stationary rates

    rates_full_wstat = copy.deepcopy(rates_full)
    rates_full_wstat.insert(0, "q00", 0 - rates_full.iloc[:, 0:3].sum(axis=1))
    rates_full_wstat.insert(5, "q11", 0 - rates_full.iloc[:, 3:6].sum(axis=1))
    rates_full_wstat.insert(10, "q22", 0 - rates_full.iloc[:, 6:9].sum(axis=1))
    rates_full_wstat.insert(15, "q33", 0 - rates_full.iloc[:, 9:12].sum(axis=1))
    # print(rates_full_wstat)

    return rates_full, rates_full_wstat


def get_rates_batch(directory):
    data = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            path = os.path.join(directory, filename)
            df = pd.read_csv(path)
            df["phylo-class"] = filename[:-4]
            data.append(df)

    data_concat = pd.concat(data, ignore_index=True)

    return data_concat


def normalise_rates(all_rates):
    all_rates["group_max"] = (
        all_rates.groupby("phylo-class").transform("max").max(axis=1)
    )
    print(all_rates)
    print(set(all_rates["group_max"]))

    norm = all_rates.iloc[:, 0:12].div(all_rates["group_max"], axis=0)
    norm["phylo-class"] = all_rates["phylo-class"]
    print(norm)
    return norm


##### Getting Probs #####


T = 0.1  # set value of T to enumerate probabilities
N_trees = 1  # declare the number of trees used to estimate the parameters
order = ["u", "l", "d", "c"]
labels = ["unlobed(u)", "lobed(l)", "dissected(d)", "compound(c)"]


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


def rates_probs_mean(prob_tab, rates_full_wstat, wd, filename):

    for i in range(0, len(rates_full_wstat)):
        rates = matrixfromrow(rates_full_wstat, i)
        probs = getprobs(rates)
        row = rowfrommatrix(probs, prob_tab)
        prob_tab = pd.concat([prob_tab, row], ignore_index=True)

    print(prob_tab)
    prob_tab.to_csv(wd + f"probs_t{T}_{filename}.csv", index=false)

    return prob_tab


def rates_mean_probs(prob_tab, rates_full_wstat, wd, filename):
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


#### Plot Uncertainty of Probabilities

colours = sns.color_palette("colorblind")


def box1(probs_full, wd, filename):
    sns.boxplot(data=probs_full, orient="v", palette=colours)
    plt.xlabel("Transition type")
    plt.ylabel(f"Probability (t={T})")
    plt.title(f"Uncertainty of pobabilities\nN={N_trees}, t={t}, {filename}")
    # sns.violinplot(data=rates_full, inner="quart")
    plt.savefig(wd + f"prob_uncert_t{T}_{filename}.png")
    plt.show()
    plt.clf()


def catplot1(probs_full_long, wd, filename):
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


def catplot2(rates_full_long, wd, filename):

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


def box2(rates_full, wd, filename):
    sns.boxplot(data=rates_full, orient="v", palette=colours)
    plt.xlabel("Rate parameter")
    plt.ylabel("Evolutionary rate")
    plt.title(f"Uncertainty of ML evolutionary rates\nN={N_trees} {filename}")
    # sns.violinplot(data=rates_full, inner="quart")
    plt.savefig(wd + f"rates_uncert_{filename}.png")
    plt.show()
    plt.clf()


def curves_phylogeny(rates_full, rates_full_wstat, wd, filename):

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


def plot_rates_batch(rates):
    mapping = {"0": "u", "1": "l", "2": "d", "3": "c"}
    col_order = [
        "geeta_phylo_geeta_class",
        "solt_phylo_geeta_class",
        "jan_phylo_geeta_class",
        "geeta_phylo_nat_class",
        "solt_phylo_nat_class",
        "jan_phylo_nat_class",
    ]

    rates_long = pd.melt(
        rates, id_vars=["phylo-class"], var_name="transition", value_name="rate"
    )
    rates_long["first_cat"] = [str[1] for str in rates_long["transition"]]
    rates_long["first_cat"] = rates_long["first_cat"].replace(mapping)
    rates_long["last_cat"] = [str[2] for str in rates_long["transition"]]
    rates_long["last_cat"] = rates_long["last_cat"].replace(mapping)

    print(rates_long)
    g = sns.catplot(
        data=rates_long,
        y="rate",
        x="first_cat",
        hue="last_cat",
        col="phylo-class",
        col_wrap=3,
        order=order,
        hue_order=order,
        col_order=col_order,
        palette="colorblind",
        kind="bar",
        # height=3,
        # aspect=1.4,
    )
    g.set_xticklabels(labels=labels, fontsize=12)
    g.set_axis_labels("Initial Shape", "Normalised Evolutionary Rate", fontsize=16)
    g._legend.set_title("Final Shape", prop={"size": 16})
    g.set_titles(size=12)
    for text in g._legend.get_texts():
        text.set_fontsize(12)
    for ax in g.axes.flat:
        ax.tick_params(axis="y", labelsize=12)
    plt.subplots_adjust(right=0.92)
    # plt.tight_layout()
    plt.show()


def rates_batch_stats(rates):
    print(rates)
    stats = rates.groupby("phylo-class").agg(["mean", "median", "sem"])
    statsT = stats.T
    # statsT.to_csv("phylogenetic_rates_norm_stats.csv")
    means = rates.groupby("phylo-class").agg(["mean"])
    meansdiff = pd.DataFrame()
    meansdiff["q01"] = means["q01"] - means["q10"]
    meansdiff["q02"] = means["q02"] - means["q20"]
    meansdiff["q03"] = means["q03"] - means["q30"]
    meansdiff["q10"] = means["q10"] - means["q01"]
    meansdiff["q12"] = means["q12"] - means["q21"]
    meansdiff["q13"] = means["q13"] - means["q31"]
    meansdiff["q20"] = means["q20"] - means["q02"]
    meansdiff["q21"] = means["q21"] - means["q12"]
    meansdiff["q23"] = means["q23"] - means["q32"]
    meansdiff["q30"] = means["q30"] - means["q03"]
    meansdiff["q31"] = means["q31"] - means["q13"]
    meansdiff["q32"] = means["q32"] - means["q23"]
    print(meansdiff)
    meansdiff.T.to_csv("phylogenetic_meanrates_diff_norm.csv")


def plot_rates_trace_hist(rates):
    # rates_sub = rates[rates["phylo-class"] == "jan_phylo_nat_class"].reset_index(
    #     drop=True
    # )
    rates_sub = rates[
        rates["phylo-class"] == "solt_phylo_geeta_class_norm_prior"
    ].reset_index(drop=True)
    rates_sub.reset_index(inplace=True)
    rates_sub.drop(columns=["phylo-class"], inplace=True)
    rates_sub_long = pd.melt(
        rates_sub, id_vars=["index"], var_name="transition", value_name="rate"
    )
    print(rates_sub_long)

    sns.displot(
        data=rates_sub_long, x="rate", col="transition", col_wrap=4, kind="hist"
    )
    plt.show()
    sns.relplot(
        data=rates_sub_long,
        y="rate",
        x="index",
        col="transition",
        col_wrap=4,
        kind="line",
    )
    plt.show()


def concat_posteriors():
    wd = "../data-processing/markov_fitter_reports/emcee/err_MUT2_2"
    files = []
    for file in os.listdir(wd):
        if file.endswith(".csv"):
            posterior = pd.read_csv(os.path.join(wd, file))
            files.append(posterior)
    posterior_concat = pd.concat(files)
    posterior_concat.to_csv("posterior_concat.csv", index=False)


def med_diff(summary):
    dfs = []
    for dataset in set(summary["Dataset_"]):
        dataset_sub = summary[summary["Dataset_"] == dataset].reset_index(drop=True)
        # create column with reverse transition types for each row
        dataset_sub["transition_rev"] = (
            dataset_sub["transition_"].str[2] + "→" + dataset_sub["transition_"].str[0]
        )
        # order alphabetically by the reverse transition types and find the difference
        dataset_sub_rev = dataset_sub.sort_values(by="transition_rev").reset_index(
            drop=True
        )
        dataset_sub["rate_norm_median_diff"] = (
            dataset_sub["rate_norm_median"] - dataset_sub_rev["rate_norm_median"]
        )
        dfs.append(dataset_sub)
    summary_new = pd.concat(dfs).reset_index(drop=True)
    return summary_new


def plot_phylo_and_sim_rates():
    phylo_rates = get_rates_batch(directory="all_rates/uniform_1010000steps")
    # s1_lb = pd.read_csv(
    #     "../data-processing/markov_fitter_reports/emcee/avg/MUT1/emcee_run_log_lb.csv"
    # )
    # s1_lb["phylo-class"] = "MUT1_simulation_lb"
    # s1_mean = pd.read_csv(
    #     "../data-processing/markov_fitter_reports/emcee/avg/MUT1/emcee_run_log_mean.csv"
    # )
    # s1_mean["phylo-class"] = "MUT1_simulation"
    # s1_ub = pd.read_csv(
    #     "../data-processing/markov_fitter_reports/emcee/avg/MUT1/emcee_run_log_ub.csv"
    # )
    # s1_ub["phylo-class"] = "MUT1_simulation_ub"

    # s2_lb = pd.read_csv(
    #     "../data-processing/markov_fitter_reports/emcee/avg/MUT2.2/emcee_run_log_lb.csv"
    # )
    # s2_lb["phylo-class"] = "MUT2_simulation_lb"
    # s2_mean = pd.read_csv(
    #     "../data-processing/markov_fitter_reports/emcee/avg/MUT2.2/emcee_run_log_mean.csv"
    # )
    # s2_mean["phylo-class"] = "MUT2_simulation"
    # s2_ub = pd.read_csv(
    #     "../data-processing/markov_fitter_reports/emcee/avg/MUT2.2/emcee_run_log_ub.csv"
    # )
    # s2_ub["phylo-class"] = "MUT2_simulation_ub"

    s1 = pd.read_csv(
        "../data-processing/markov_fitter_reports/emcee/leaf_uncert_posteriors_MUT1.csv"
    )
    s2 = pd.read_csv(
        "../data-processing/markov_fitter_reports/emcee/leaf_uncert_posteriors_MUT2.csv"
    )
    s1["phylo-class"] = "MUT1_simulation"
    s2["phylo-class"] = "MUT2_simulation"
    sim_rates = pd.concat([s1, s2]).reset_index(drop=True)

    # sim_rates = pd.concat([sim_rates1, sim_rates2]).reset_index(drop=True)
    # sim_rates = pd.read_csv("../data-processing/emcee_run_log.csv")
    # sim_rates_norm = sim_rates.div(sim_rates.max(axis=None))
    # sim_rates_norm["phylo-class"] = "MUT2.2_simulation"
    name_map = {
        "0": "q01",
        "1": "q02",
        "2": "q03",
        "3": "q10",
        "4": "q12",
        "5": "q13",
        "6": "q20",
        "7": "q21",
        "8": "q23",
        "9": "q30",
        "10": "q31",
        "11": "q32",
    }
    sim_rates = sim_rates.rename(columns=name_map)

    phylo_sim = pd.concat([sim_rates, phylo_rates]).reset_index(drop=True)
    phylo_sim = phylo_sim.rename(columns={"phylo-class": "Dataset"})
    phylo_sim_long = pd.melt(
        phylo_sim, id_vars=["Dataset"], var_name="transition", value_name="rate"
    )
    # Normalise by dividing by the mean mean transition rate for each dataset
    phylo_sim_long["mean_rate"] = phylo_sim_long.groupby(["Dataset", "transition"])[
        "rate"
    ].transform("mean")

    phylo_sim_long["mean_mean"] = phylo_sim_long.groupby(["Dataset"])[
        "mean_rate"
    ].transform("mean")
    phylo_sim_long["rate_norm"] = phylo_sim_long["rate"] / phylo_sim_long["mean_mean"]
    phylo_sim_long["initial_shape"], phylo_sim_long["final_shape"] = zip(
        *phylo_sim_long["transition"].map(rates_map)
    )
    print(phylo_sim_long)

    # Select the datasets to plot e.g. MUT2.2 simulation and 2 phylogenies
    # phylo_sim_sub = phylo_sim_long[
    #     phylo_sim_long["Dataset"].isin(
    #         [
    #             "MUT2.2_simulation",
    #             "MUT1_simulation",
    #             "geeta_phylo_geeta_class",
    #             "jan_phylo_nat_class",
    #         ]
    #     )
    # ].reset_index(drop=True)
    phylo_sim_sub = phylo_sim_long

    # phylo_sim_sub["Dataset"] = phylo_sim_sub["Dataset"].replace(
    #     {
    #         "MUT1_simulation": "Simulation 1",
    #         "MUT2.2_simulation": "Simulation 2",
    #         "jan_phylo_nat_class": "Phylogeny 1",
    #         "geeta_phylo_geeta_class": "Phylogeny 2",
    #     }
    # )
    rate_map = {
        "q01": "u→l",
        "q02": "u→d",
        "q03": "u→c",
        "q10": "l→u",
        "q12": "l→d",
        "q13": "l→c",
        "q20": "d→u",
        "q21": "d→l",
        "q23": "d→c",
        "q30": "c→u",
        "q31": "c→l",
        "q32": "c→d",
    }
    phylo_sim_sub["transition"] = phylo_sim_sub["transition"].replace(rate_map)
    print(phylo_sim_sub)
    # exit()

    summary = (
        phylo_sim_sub.groupby(["Dataset", "transition"])[["rate_norm", "rate"]]
        .agg(["mean", "median", "count", "std", "sem"])
        .reset_index()
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]

    # get differences between back and forth median rates for arrow plots
    summary = med_diff(summary)
    print(summary)
    # summary.to_csv("sim_phylo_rates_stats_12-08-24.csv", index=False)
    # exit()

    # # summary["mcmc_std_frac"] = summary["std"] / summary["mean"]
    # print(summary)

    # load fractional errors from transition counts
    # MUT1_std_frac = pd.read_csv(
    #     "../data-processing/markov_fitter_reports/emcee/avg/MUT1/MUT1_counts.csv"
    # )
    # MUT1_std_frac["Dataset"] = "MUT1_simulation"
    # MUT2_std_frac = pd.read_csv(
    #     "../data-processing/markov_fitter_reports/emcee/avg/MUT2.2/MUT2.2_counts.csv"
    # )
    # MUT2_std_frac["Dataset"] = "MUT2_simulation"
    # leaf_std_frac = pd.concat([MUT1_std_frac, MUT2_std_frac])
    # leaf_std_frac.rename(columns={"std_frac": "leaf_std_frac"}, inplace=True)
    # # drop stasis
    # leaf_std_frac.drop(
    #     leaf_std_frac[leaf_std_frac["transition"].isin(["uu", "ll", "dd", "cc"])].index,
    #     inplace=True,
    # )
    # # merge leaf error with mcmc_error - only introduces data for simulation rows
    # summary = pd.merge(
    #     summary,
    #     leaf_std_frac[["Dataset", "transition", "leaf_std_frac"]],
    #     on=["Dataset", "transition"],
    #     how="outer",
    # )
    # summary["std_frac_quad"] = np.sqrt(
    #     (summary["mcmc_std_frac"] ** 2) + (summary["leaf_std_frac"] ** 2)
    # )
    # # Make an error column which contains mcmc std for phylogeny datasets and the leaf count std and mcmc std added in quadrature for simulation datasets
    # summary["err"] = summary["std_frac_quad"] * summary["mean"]
    # summary["err"].fillna(summary["std"], inplace=True)
    # print(summary)

    # summary["err"] = 1.96 * summary["sem"]
    # print(summary)
    # exit()

    # summary["ub"] = summary["mean"] + (1.96 * summary["sem"])
    # summary["lb"] = summary["mean"] - (1.96 * summary["sem"])
    # summary["ub"] = summary["mean"] + summary["std"]
    # summary["lb"] = summary["mean"] - summary["std"]

    # print(summary)
    # MUT1 = summary[summary["Dataset"].isin(["MUT1_simulation_mean"])][
    #     ["Dataset", "transition", "mean"]
    # ].reset_index(drop=True)
    # MUT1["lb"] = summary[summary["Dataset"] == "MUT1_simulation_lb"]["lb"]
    # MUT1["ub"] = summary[summary["Dataset"] == "MUT1_simulation_ub"]["ub"].reset_index(
    #     drop=True
    # )
    # MUT1["Dataset"] = "MUT1_simulation"
    # print(MUT1)

    # MUT2 = summary[summary["Dataset"].isin(["MUT2_simulation_mean"])][
    #     ["Dataset", "transition", "mean"]
    # ].reset_index(drop=True)
    # MUT2["lb"] = summary[summary["Dataset"] == "MUT2_simulation_lb"]["lb"].reset_index(
    #     drop=True
    # )
    # MUT2["ub"] = summary[summary["Dataset"] == "MUT2_simulation_ub"]["ub"].reset_index(
    #     drop=True
    # )
    # MUT2["Dataset"] = "MUT2_simulation"
    # print(MUT2)

    # summary = pd.concat([MUT1, MUT2, summary])
    # summary["mean-lb"] = abs(summary["mean"] - summary["lb"])
    # summary["ub-mean"] = abs(summary["ub"] - summary["mean"])
    # print(summary)

    # plot_order = ["Simulation 1", "Simulation 2", "Phylogeny 1", "Phylogeny 2"]
    plot_order = [
        "MUT1_simulation",
        "MUT2_simulation",
        "jan_phylo_nat_class",
        "jan_phylo_geeta_class",
        "zuntini_phylo_nat_class",
        "zuntini_phylo_geeta_class",
        "geeta_phylo_nat_class",
        "geeta_phylo_geeta_class",
        # "geeta_phylo_geeta_class_23-04-24_17_each",
        # "geeta_phylo_geeta_class_23-04-24_shuff",
        # "geeta_phylo_geeta_class_23-04-24_mle",
        # "geeta_phylo_geeta_class_23-04-24_17_each_mle",
        # "geeta_phylo_geeta_class_23-04-24_shuff_mle",
    ]
    fig, axes = plt.subplots(
        nrows=4, ncols=4, figsize=(10, 8)
    )  # , layout="constrained")

    counter = -1
    legend_labels = []
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i == j:
                axes[i, j].axis("off")
            if i != j:
                counter += 1
                transition = list(rate_map.values())[counter]
                plot_data = phylo_sim_sub[phylo_sim_sub["transition"] == transition]
                print(plot_data)
                rates = []
                for k, dataset in enumerate(plot_order):
                    # dset.append(dataset)
                    rates.append(
                        plot_data["rate_norm"][
                            plot_data["Dataset"] == dataset
                        ].squeeze()
                    )

                    # err = plot_data["sem"][
                    #     plot_data["Dataset"] == dataset
                    # ].squeeze()  # rate_lb = plot_data["mean-lb"][
                    #     plot_data["Dataset"] == dataset
                    # ].squeeze()

                    # rate_ub = plot_data["ub-mean"][
                    #     plot_data["Dataset"] == dataset
                    # ].squeeze()
                    # ax.axvspan(
                    #     # "MUT1_simulation",
                    #     # "MUT2_simulation",
                    #     -0.5,
                    #     1.5,
                    #     color="grey",
                    #     alpha=0.04,
                    #     linewidth=0,
                    # )
                    # ax.errorbar(
                    #     x=dataset,
                    #     y=rate,
                    #     yerr=err,
                    #     fmt="o",
                    #     ms=4,
                    #     color=sns.color_palette("colorblind")[k],
                    #     capsize=2.5,
                    #     capthick=1.5,
                    #     elinewidth=1.5,
                    # )

                    if dataset not in legend_labels:
                        legend_labels.append(dataset)

                # ax.axvline(3.5, linestyle="--", color="grey", alpha=0.5)
                # ax.text(1, 5, "MCMC", color="grey")
                # ax.text(4.5, 5, "MLE", color="grey")
                ax.axvline(2.5, linestyle="--", color="grey", alpha=0.5)
                # bp = ax.boxplot(
                #     rates,
                #     patch_artist=True,
                #     # showmeans=True,
                #     # meanline=True,
                #     showfliers=False,
                # )
                bp = ax.violinplot(
                    rates,
                    showmedians=True,
                    showextrema=False,
                )

                # for median in bp["medians"]:
                #     # median.set_visible(False)
                #     median.set(color="black")
                # for k, box in enumerate(bp["boxes"]):
                #     box.set_facecolor(sns.color_palette("colorblind")[k])
                for k, pc in enumerate(bp["bodies"]):
                    # pc.set_facecolor(sns.color_palette("colorblind")[k % 3])
                    # pc.set_facecolor(sns.color_palette("colorblind")[-((k - 1) // -2)])
                    pc.set_facecolor(sns.color_palette("colorblind")[k])
                    # pc.set_edgecolor("black")
                    pc.set_alpha(1)
                bp["cmedians"].set_colors("black")
                ax.set_title(transition)
                ax.set_ylim(0, 6)
            if j == 0:
                ax.set_ylabel("Rate")
            if i == 3:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    # plot_order,
                    ["S1", "S2", "P1", "P2", "P3", "P4", "P5", "P6"],
                    fontsize=9,
                )
                ax.set_xlabel("Dataset")
            if j == 3 and i == 2:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    # plot_order,
                    ["S1", "S2", "P1", "P2", "P3", "P4", "P5", "P6"],
                    fontsize=9,
                )
            if (i, j) == (0, 1):
                ax.set_ylabel("Rate")
            if j != 0 and (i, j) != (0, 1):
                ax.set_yticklabels([])
            if i != 3 and (i, j) != (2, 3):
                ax.set_xticklabels([])
            if (i, j) == (2, 3):
                ax.set_xlabel("Dataset")
    labels_alt = [
        "S1: MUT1 Simulation",
        "S2: MUT2 Simulation",
        "P1: Janssens et al. (2020)\nphylogeny, Naturalis\nclassification",
        "P2: Janssens et al. (2020)\nphylogeny, Geeta et al.\n(2012) classification",
        # "P3: Soltis et al. (2011)\nphylogeny, Naturalis\nclassification",
        # "P4: Soltis et al. (2011)\nphylogeny, Geeta et al.\n(2012) classification",
        "P3: Zuntini et al. (2024)\nphylogeny, Naturalis\nclassification",
        "P4: Zuntini et al. (2024)\nphylogeny, Geeta et al.\n(2012) classification",
        "P5: Geeta et al. (2012)\nphylogeny, Naturalis\nclassification",
        "P6: Geeta et al. (2012)\nphylogeny, Geeta et al.\n(2012) classification",
        # "P7: jan_phylo_nat_class\n21-01-24_95_each",
        # "P8: jan_phylo_nat_class\n_21-01-24_shuff",
    ]
    # labels_alt = [
    #     "P1: Geeta et al. (2012)\nphylogeny, Geeta et al.\n(2012) classification",
    #     "P2: geeta_phylo_geeta_\nclass_23-04-24_17_each\n_mcmc",
    #     "P3: geeta_phylo_geeta_\nclass_23-04-24_17_each\n_mle",
    #     "P4: geeta_phylo_geeta_\nclass_23-04-24_shuff\n_mcmc",
    #     "P5: geeta_phylo_geeta_\nclass_23-04-24_shuff\n_mle",
    # ]
    # labels_alt = [
    #     "P1: original mcmc",
    #     "P2: 17 each, mcmc",
    #     "P3: shuffled, mcmc",
    #     "P4: original mle",
    #     "P5: 17 each, mle",
    #     "P6: shuffled, mle",
    # ]
    # labels_alt = [
    #     "P1: jan_phylo_nat_class",
    #     "P2: jan_phylo_geeta_class",
    #     "P3: solt_phylo_nat_class",
    #     "P4: solt_phylo_geeta_class",
    #     "P5: geeta_phylo_nat_class",
    #     "P6: geeta_phylo_geeta_class",
    # ]

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            # color=sns.color_palette("colorblind")[i % 3],
            # color=sns.color_palette("colorblind")[-((i - 1) // -2)],
            color=sns.color_palette("colorblind")[i],
        )
        for i, label in enumerate(
            labels_alt
        )  # change back to legend_labels if you want the default
    ]
    fig.legend(
        legend_handles,
        labels_alt,
        loc="right",
        title="Dataset",
        # loc="outside center right",
        # bbox_to_anchor=(1.2, 0.5),
        # fontsize=11,
        ncol=1,
    )
    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.25, wspace=0.2, bottom=0.18)
    plt.subplots_adjust(hspace=0.2, wspace=0.2, right=0.745)
    plt.show()

    #### For the Shuffle test

    plot_order = [
        # "MUT1_simulation",
        # "MUT2_simulation",
        # "jan_phylo_nat_class",
        # "jan_phylo_geeta_class",
        # "zuntini_phylo_nat_class",
        # "zuntini_phylo_geeta_class",
        # "geeta_phylo_nat_class",
        "geeta_phylo_geeta_class",
        "geeta_phylo_geeta_class_23-04-24_17_each",
        "geeta_phylo_geeta_class_23-04-24_shuff",
        "geeta_phylo_geeta_class_23-04-24_mle",
        "geeta_phylo_geeta_class_23-04-24_17_each_mle",
        "geeta_phylo_geeta_class_23-04-24_shuff_mle",
    ]
    fig, axes = plt.subplots(
        nrows=4, ncols=4, figsize=(10, 8)
    )  # , layout="constrained")

    counter = -1
    legend_labels = []
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i == j:
                axes[i, j].axis("off")
            if i != j:
                counter += 1
                transition = list(rate_map.values())[counter]
                plot_data = phylo_sim_sub[phylo_sim_sub["transition"] == transition]
                print(plot_data)
                rates = []
                for k, dataset in enumerate(plot_order):
                    # dset.append(dataset)
                    rates.append(
                        plot_data["rate_norm"][
                            plot_data["Dataset"] == dataset
                        ].squeeze()
                    )

                    # err = plot_data["sem"][
                    #     plot_data["Dataset"] == dataset
                    # ].squeeze()  # rate_lb = plot_data["mean-lb"][
                    #     plot_data["Dataset"] == dataset
                    # ].squeeze()

                    # rate_ub = plot_data["ub-mean"][
                    #     plot_data["Dataset"] == dataset
                    # ].squeeze()
                    # ax.axvspan(
                    #     # "MUT1_simulation",
                    #     # "MUT2_simulation",
                    #     -0.5,
                    #     1.5,
                    #     color="grey",
                    #     alpha=0.04,
                    #     linewidth=0,
                    # )
                    # ax.errorbar(
                    #     x=dataset,
                    #     y=rate,
                    #     yerr=err,
                    #     fmt="o",
                    #     ms=4,
                    #     color=sns.color_palette("colorblind")[k],
                    #     capsize=2.5,
                    #     capthick=1.5,
                    #     elinewidth=1.5,
                    # )

                    if dataset not in legend_labels:
                        legend_labels.append(dataset)

                ax.axvline(3.5, linestyle="--", color="grey", alpha=0.5)
                ax.text(1, 5, "MCMC", color="grey")
                ax.text(4.5, 5, "MLE", color="grey")
                # ax.axvline(2.5, linestyle="--", color="grey", alpha=0.5)
                # bp = ax.boxplot(
                #     rates,
                #     patch_artist=True,
                #     # showmeans=True,
                #     # meanline=True,
                #     showfliers=False,
                # )
                bp = ax.violinplot(
                    rates,
                    showmedians=True,
                    showextrema=False,
                )

                # for median in bp["medians"]:
                #     # median.set_visible(False)
                #     median.set(color="black")
                # for k, box in enumerate(bp["boxes"]):
                #     box.set_facecolor(sns.color_palette("colorblind")[k])
                for k, pc in enumerate(bp["bodies"]):
                    pc.set_facecolor(sns.color_palette("colorblind")[k % 3])
                    # pc.set_facecolor(sns.color_palette("colorblind")[-((k - 1) // -2)])
                    # pc.set_facecolor(sns.color_palette("colorblind")[k])
                    # pc.set_edgecolor("black")
                    pc.set_alpha(1)
                bp["cmedians"].set_colors("black")
                ax.set_title(transition)
                ax.set_ylim(0, 6)
            if j == 0:
                ax.set_ylabel("Rate")
            if i == 3:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    # plot_order,
                    ["P1", "P2", "P3", "P4", "P5", "P6"],
                    fontsize=9,
                )
                ax.set_xlabel("Dataset")
            if j == 3 and i == 2:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    # plot_order,
                    ["P1", "P2", "P3", "P4", "P5", "P6"],
                    fontsize=9,
                )
            if (i, j) == (0, 1):
                ax.set_ylabel("Rate")
            if j != 0 and (i, j) != (0, 1):
                ax.set_yticklabels([])
            if i != 3 and (i, j) != (2, 3):
                ax.set_xticklabels([])
            if (i, j) == (2, 3):
                ax.set_xlabel("Dataset")

    labels_alt = [
        "P1: original mcmc",
        "P2: 17 each, mcmc",
        "P3: shuffled, mcmc",
        "P4: original mle",
        "P5: 17 each, mle",
        "P6: shuffled, mle",
    ]

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=sns.color_palette("colorblind")[i % 3],
        )
        for i, label in enumerate(
            labels_alt
        )  # change back to legend_labels if you want the default
    ]
    fig.legend(
        legend_handles,
        labels_alt,
        loc="right",
        title="Dataset",
        # loc="outside center right",
        # bbox_to_anchor=(1.2, 0.5),
        # fontsize=11,
        ncol=1,
    )
    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.25, wspace=0.2, bottom=0.18)
    plt.subplots_adjust(hspace=0.2, wspace=0.2, right=0.745)
    plt.show()


if __name__ == "__main__":

    # plot_rates_trace_hist(rates)
    # phylo_rates_norm = normalise_rates(phylo_rates)
    plot_phylo_and_sim_rates()
    # concat_posteriors()
    # rates_batch_stats(rates_norm)
    # plot_rates_batch(rates_norm)
    # concat_posteriors()
