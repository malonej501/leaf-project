import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal
from scipy import stats
import copy
import os
from PIL import Image
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from sympy import *
import schemdraw
from schemdraw import flow
from schemdraw import elements as elm
from datetime import date

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

rates_map2 = {
    "0": "u→l",
    "1": "u→d",
    "2": "u→c",
    "3": "l→u",
    "4": "l→d",
    "5": "l→c",
    "6": "d→u",
    "7": "d→l",
    "8": "d→c",
    "9": "c→u",
    "10": "c→l",
    "11": "c→d",
}

rates_map3 = {
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

rates_map4 = {
    "0": "u→l",
    "1": "u→c",
    "2": "l→u",
    "3": "l→c",
    "4": "c→u",
    "5": "c→l",
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


def concat_posteriors():
    wd = "../data-processing/markov_fitter_reports/emcee/err_MUT2_2"
    files = []
    for file in os.listdir(wd):
        if file.endswith(".csv"):
            posterior = pd.read_csv(os.path.join(wd, file))
            files.append(posterior)
    posterior_concat = pd.concat(files)
    posterior_concat.to_csv("posterior_concat.csv", index=False)


def rate_diff(summary):
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
        # dataset_sub["rate_norm_median_diff"] = (
        #     dataset_sub["rate_norm_median"] - dataset_sub_rev["rate_norm_median"]
        # )
        dataset_sub["rate_norm_mean_diff"] = (
            dataset_sub["rate_norm_mean"] - dataset_sub_rev["rate_norm_mean"]
        )
        dfs.append(dataset_sub)
    summary_new = pd.concat(dfs).reset_index(drop=True)
    return summary_new


def import_phylo_and_sim_rates(plot_order, calc_diff):
    # to return differences between rates rather than raw rates, set calc_diff to true
    #### import data ####
    phylo_rates = get_rates_batch(directory="rates/uniform_1010000steps")
    # s1 = pd.read_csv(
    #     "../data-processing/markov_fitter_reports/emcee/leaf_uncert_posteriors_MUT1.csv"
    # )
    s1 = pd.read_csv(
        "../data-processing/markov_fitter_reports/emcee/posteriors_MUT1_mcmc_04-12-24.csv"
    )
    # s2 = pd.read_csv(
    #     "../data-processing/markov_fitter_reports/emcee/leaf_uncert_posteriors_MUT2.csv"
    # )
    s2 = pd.read_csv(
        "../data-processing/markov_fitter_reports/emcee/posteriors_MUT2_mcmc_04-12-24.csv"
    ) 
    s1["phylo-class"] = "MUT1_simulation"
    s2["phylo-class"] = "MUT2_simulation"
    sim_rates = pd.concat([s1, s2]).reset_index(drop=True)

    sim_rates = sim_rates.rename(columns=rates_map2)
    phylo_rates = phylo_rates.rename(columns=rates_map3)

    phylo_sim = pd.concat([sim_rates, phylo_rates]).reset_index(drop=True)
    phylo_sim = phylo_sim.rename(columns={"phylo-class": "Dataset"})

    # Calculate differences
    if calc_diff:
        transitions = rates_map3.values()
        for fwd in transitions:
            bwd = fwd[2] + "→" + fwd[0]
            if fwd != bwd:
                col_name = f"{fwd}-{bwd}"
                phylo_sim[col_name] = phylo_sim[fwd] - phylo_sim[bwd]
    # phylo_sim["l→u-u→l"] = phylo_sim["l→u"] - phylo_sim["u→l"]
    # phylo_sim["d→u-u→d"] = phylo_sim["d→u"] - phylo_sim["u→d"]
    # phylo_sim["c→u-u→c"] = phylo_sim["c→u"] - phylo_sim["u→c"]
    # phylo_sim["l→d-d→l"] = phylo_sim["l→d"] - phylo_sim["d→l"]
    # phylo_sim["l→c-c→l"] = phylo_sim["l→c"] - phylo_sim["c→l"]
    # phylo_sim["d→c-c→d"] = phylo_sim["d→c"] - phylo_sim["c→d"]
    # # phylo_sim = phylo_sim.drop(columns=rates_map3.values())
    phylo_sim_long = pd.melt(
        phylo_sim,
        id_vars=["Dataset"],
        var_name="transition",
        value_name="rate",
    )
    phylo_sim_long["dataname"] = phylo_sim_long["Dataset"].apply(
        lambda x: x.split("_class", 1)[0] + "_class"
    )
    # filter to only rows with dataset in the plot_order list
    phylo_sim_long = phylo_sim_long[
        phylo_sim_long["Dataset"].isin(plot_order)
    ].reset_index(drop=True)

    return phylo_sim_long


def import_phylo_ML_rates(ML_data, plot_order, calc_diff):
    # ML_rates = pd.read_csv("rates/ML3_mean_rates_all.csv")
    ML_rates = pd.read_csv(f"rates/ML/{ML_data}.csv")
    ML_rates.drop(
        columns=[
            "Lh",
            "Root P(0)",
            "Root P(1)",
            "Root P(2)",
            "Root P(3)",
            "Unnamed: 11",
        ],
        inplace=True,
        errors="ignore",
    )
    ML_rates = ML_rates.rename(columns=rates_map3)
    ML_rates["dataname"] = ML_rates["dataset"].apply(
        lambda x: x.split("_class", 1)[0] + "_class"
    )
    if calc_diff:
        # Calculate differences
        transitions = rates_map3.values()
        for fwd in transitions:
            bwd = fwd[2] + "→" + fwd[0]
            if fwd != bwd:
                col_name = f"{fwd}-{bwd}"
                ML_rates[col_name] = ML_rates[fwd] - ML_rates[bwd]

        # ML_rates["l→u-u→l"] = ML_rates["l→u"] - ML_rates["u→l"]
        # ML_rates["d→u-u→d"] = ML_rates["d→u"] - ML_rates["u→d"]
        # ML_rates["c→u-u→c"] = ML_rates["c→u"] - ML_rates["u→c"]
        # ML_rates["l→d-d→l"] = ML_rates["l→d"] - ML_rates["d→l"]
        # ML_rates["l→c-c→l"] = ML_rates["l→c"] - ML_rates["c→l"]
        # ML_rates["d→c-c→d"] = ML_rates["d→c"] - ML_rates["c→d"]
        # ML_rates = ML_rates.drop(columns=rates_map3.values())

    ML_rates_long = pd.melt(
        ML_rates,
        id_vars=["dataname", "dataset"],
        var_name="transition",
        value_name="rate",
    )
    # filter to only rows with dataset in the plot_order list
    ML_plot_order = [x.split("_class", 1)[0] + "_class" for x in plot_order]
    ML_rates_long = ML_rates_long[
        ML_rates_long["dataname"].isin(ML_plot_order)
    ].reset_index(drop=True)
    # ML_rates_long.rename(
    #     columns={"dataset": "Dataset"}
    # )  # rename to match column in mcmc rates
    return ML_rates_long

def import_sim_ML_rates(calc_diff):
    mut1_ml = pd.read_csv(f"../data-processing/markov_fitter_reports/emcee/ML_MUT1_mcmc_04-12-24.csv")
    mut1_ml["Dataset"] = "MUT1_simulation"
    mut2_ml = pd.read_csv(f"../data-processing/markov_fitter_reports/emcee/ML_MUT2_mcmc_04-12-24.csv")
    mut2_ml["Dataset"] = "MUT2_simulation"
    sim_ml = pd.concat([mut1_ml, mut2_ml], ignore_index=True)
    sim_ml = sim_ml.rename(columns=rates_map2)
    
    if calc_diff:
        transitions = rates_map3.values()
        for fwd in transitions:
            bwd = fwd[2] + "→" + fwd[0]
            if fwd != bwd:
                col_name = f"{fwd}-{bwd}"
                sim_ml[col_name] = sim_ml[fwd] - sim_ml[bwd]

    sim_ml_long = pd.melt(sim_ml, id_vars="Dataset",var_name="transition",value_name="rate")
    sim_ml_long["dataname"] = sim_ml_long["Dataset"] + "_class"
    return sim_ml_long

def normalise_rates(phylo_sim_long, ML_phylo_rates_long, ML_sim_rates_long, norm_method):
    #### Normalise the rates across datasets ####
    phylo_sim_long_filt = phylo_sim_long[  # only calculate the mean rate for transitions, not transition differences
        phylo_sim_long["transition"].isin(rates_map3.values())
    ]
    phylo_sim_long["mean_rate"] = phylo_sim_long_filt.groupby(
        ["Dataset", "transition"]
    )["rate"].transform(
        "mean"
    )  # get mean rate for each transition per dataset

    if norm_method == "meanmean":
        # get the mean mean transition rate per dataset (i.e. the centre of the rates for that dataset)
        phylo_sim_long["mean_mean"] = phylo_sim_long.groupby(["Dataset"])[
            "mean_rate"
        ].transform("mean")
        # normalise by dividing by the mean mean transition rate for each dataset
        phylo_sim_long["rate_norm"] = (
            phylo_sim_long["rate"] / phylo_sim_long["mean_mean"]
        )

        # merge mcmc mean-means with ML-rates
        ML_phylo_rates_long = pd.merge(
            ML_phylo_rates_long,
            phylo_sim_long[["dataname", "mean_mean"]].drop_duplicates(
                subset=["dataname"]
            ),
            on="dataname",
        )
        ML_sim_rates_long = pd.merge(
            ML_sim_rates_long,
            phylo_sim_long[["dataname", "mean_mean"]].drop_duplicates(
                subset=["dataname"]
            ),
            on="dataname",
        )

        # normalise ML rates
        ML_phylo_rates_long["rate_norm"] = (
            ML_phylo_rates_long["rate"] / ML_phylo_rates_long["mean_mean"]
        )
        ML_sim_rates_long["rate_norm"] = (
            ML_sim_rates_long["rate"] / ML_sim_rates_long["mean_mean"]
        )
        # phylo_sim_long["initial_shape"], phylo_sim_long["final_shape"] = zip(
        #     *phylo_sim_long["transition"].map(rates_map)
        # )

    elif norm_method == "zscore":
        # get the mean mean transition rate per dataset (i.e. the centre of the rates for that dataset)
        phylo_sim_long["mean_mean"] = phylo_sim_long.groupby(["Dataset"])[
            "mean_rate"
        ].transform("mean")
        # get the stdev of the mean transition rate per dataset
        phylo_sim_long["std_mean"] = phylo_sim_long.groupby("Dataset")[
            "mean_rate"
        ].transform("std")
        # z-score normalisation
        phylo_sim_long["rate_norm"] = (
            phylo_sim_long["rate"] - phylo_sim_long["mean_mean"]
        ) / phylo_sim_long["std_mean"]
        # normalise ML rates
        ML_phylo_rates_long["rate_norm"] = (
            ML_phylo_rates_long["rate"] - phylo_sim_long["mean_mean"]
        ) / phylo_sim_long["std_mean"]

    elif norm_method == "zscore+2.7":
        # get the mean mean transition rate per dataset (i.e. the centre of the rates for that dataset)
        phylo_sim_long["mean_mean"] = phylo_sim_long.groupby(["Dataset"])[
            "mean_rate"
        ].transform("mean")
        # get the stdev of the mean transition rate per dataset
        phylo_sim_long["std_mean"] = phylo_sim_long.groupby("Dataset")[
            "mean_rate"
        ].transform("std")
        # z-score normalisation
        phylo_sim_long["rate_norm"] = (
            (phylo_sim_long["rate"] - phylo_sim_long["mean_mean"])
            / phylo_sim_long["std_mean"]
        ) + 2.7  # move data up by 2.7 to get rid of negatives
        # normalise ML rates
        ML_phylo_rates_long["rate_norm"] = (
            (ML_phylo_rates_long["rate"] - phylo_sim_long["mean_mean"])
            / phylo_sim_long["std_mean"]
        ) + 2.7

    elif norm_method == "zscore_global":
        phylo_sim_long["dataset_mean"] = phylo_sim_long.groupby(["Dataset"])[
            "rate"
        ].transform(
            "mean"
        )  # get mean overall rate for each dataset
        # get the stdev of rates per dataset
        phylo_sim_long["dataset_std"] = phylo_sim_long.groupby("Dataset")[
            "rate"
        ].transform("std")
        # zscore normalisation
        phylo_sim_long["rate_norm"] = (
            phylo_sim_long["rate"] - phylo_sim_long["dataset_mean"]
        ) / phylo_sim_long["dataset_std"]
        # normalise ML rates
        ML_phylo_rates_long["rate_norm"] = (
            ML_phylo_rates_long["rate"] - phylo_sim_long["dataset_mean"]
        ) / phylo_sim_long["dataset_std"]

    elif norm_method == "minmax":
        # min max normalisation
        phylo_sim_long["min_mean"] = phylo_sim_long.groupby("Dataset")[
            "mean_rate"
        ].transform("min")
        phylo_sim_long["max_mean"] = phylo_sim_long.groupby("Dataset")[
            "mean_rate"
        ].transform("max")
        phylo_sim_long["rate_norm"] = (
            phylo_sim_long["rate"] - phylo_sim_long["min_mean"]
        ) / (phylo_sim_long["max_mean"] - phylo_sim_long["min_mean"])
        # normalise ML rates
        ML_phylo_rates_long["rate_norm"] = (
            ML_phylo_rates_long["rate"] - phylo_sim_long["min_mean"]
        ) / (phylo_sim_long["max_mean"] - phylo_sim_long["min_mean"])

    else:
        raise RuntimeError(
            "Invalid normalisation method. Ensure norm_method argument is correct."
        )

    #### generate summary statistics for all normalised rate estimates ####

    summary = (
        phylo_sim_long.groupby(["Dataset", "transition"])[["rate_norm", "rate"]]
        .agg(["mean", "median", "count", "std", "sem"])
        .reset_index()
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]

    # get differences between back and forth median rates for arrow plots
    summary = rate_diff(summary)
    print(summary)
    summary.to_csv(
        f"sim_phylo_rates_stats_{norm_method}_mixedpriors_{str(date.today())}.csv", index=False
    )
    print(ML_phylo_rates_long)
    print(
        ML_phylo_rates_long[ML_phylo_rates_long["dataname"] == "jan_phylo_geeta_class"]
    )

    return phylo_sim_long, ML_phylo_rates_long, ML_sim_rates_long, summary


def plot_phylo_and_sim_rates(norm_method, ML_data, legend):
    plot_order = [
        "MUT1_simulation",
        "MUT2_simulation",
        # "jan_phylo_nat_class_uniform0-100_rj_scaletrees0.001_3",
        # "jan_phylo_geeta_class_uniform0-100_rj_scaletrees0.001_3",
        # "solt_phylo_nat_class_uniform0-100_rj_4",
        # "solt_phylo_geeta_class_uniform0-100_rj_4",
        # "zuntini_phylo_nat_class_uniform0-100_rj_scaletrees0.001_3",
        # "zuntini_phylo_geeta_class_uniform0-100_rj_scaletrees0.001_3",
        # "geeta_phylo_nat_class_uniform0-100_rj_4",
        # "geeta_phylo_geeta_class_uniform0-100_rj_4",
        # "jan_phylo_nat_class_uniform0-0.1_res_2",
        "jan_phylo_nat_class_uniform0-0.1_1",
        # "jan_equal_fam_phylo_nat_class_uniform0-0.1_3",
        # "jan_equal_genus_phylo_nat_class_uniform0-0.1_4",
        # "jan_phylo_geeta_class_uniform0-100_2",
        # "solt_phylo_nat_class_uniform0-100_2",
        # "solt_phylo_geeta_class_uniform0-100_2",
        # "zuntini_phylo_nat_class_uniform0-0.1_1",
        # "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_res_2",
        "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_2",
        # "zuntini_genera_equal_fam_phylo_nat_class_uniform0-0.1_3",
        # "zuntini_genera_equal_genus_phylo_nat_class_uniform0-0.1_4",
        # "zuntini_phylo_geeta_class_uniform0-0.1_1",
        # "geeta_phylo_nat_class_uniform0-100_2",
        # "geeta_phylo_geeta_class_uniform0-100_2",
        "geeta_phylo_geeta_class_uniform0-100_4",
        # "jan_phylo_nat_class_uniform0-0.1_rj_1",
        # "jan_phylo_geeta_class_uniform0-0.1_rj_1",
        # "solt_phylo_nat_class_uniform0-100_rj_4",
        # "solt_phylo_geeta_class_uniform0-100_rj_4",
        # "zuntini_phylo_nat_class_uniform0-0.1_rj_1",
        # "zuntini_phylo_geeta_class_uniform0-0.1_rj_1",
        # "geeta_phylo_nat_class_uniform0-100_rj_4",
        # "geeta_phylo_geeta_class_uniform0-100_rj_4",
        # "geeta_phylo_nat_class_uniform0-100_2",
        # "geeta_phylo_geeta_class_uniform0-100_2",
        # "geeta_phylo_geeta_class_23-04-24_17_each",
        # "geeta_phylo_geeta_class_23-04-24_shuff",
        # "geeta_phylo_geeta_class_23-04-24_mle",
        # "geeta_phylo_geeta_class_23-04-24_17_each_mle",
        # "geeta_phylo_geeta_class_23-04-24_shuff_mle",
    ]

    ML_phylo_rates_long = import_phylo_ML_rates(ML_data)
    phylo_sim_long = import_phylo_and_sim_rates(plot_order)
    phylo_sim_long, ML_phylo_rates_long, summary = normalise_rates(
        phylo_sim_long, ML_phylo_rates_long, norm_method
    )

    #### plotting ####

    plt.rcParams["font.family"] = "CMU Serif"
    if legend:
        fig, axes = plt.subplots(
            nrows=4,
            ncols=4,
            figsize=(10, 8),  # sharey=True
        )  # , layout="constrained")
    else:
        fig, axes = plt.subplots(
            nrows=4,
            ncols=4,
            figsize=(8, 8),  # sharey=True
        )  # , layout="constrained")
    counter = -1
    legend_labels = []
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i == j:
                axes[i, j].axis("off")
            if i != j:
                counter += 1
                transition = list(rates_map2.values())[counter]
                plot_data = phylo_sim_long[phylo_sim_long["transition"] == transition]
                ML_plot_data = ML_phylo_rates_long[
                    ML_phylo_rates_long["transition"] == transition
                ]
                rates = []
                ML_rates = []
                for k, dataset in enumerate(plot_order):
                    # dset.append(dataset)
                    rates.append(
                        plot_data["rate_norm"][
                            plot_data["Dataset"] == dataset
                        ].squeeze()
                    )

                    # get ML_rates in correct order
                    x = ML_plot_data[
                        ML_plot_data["dataname"].apply(lambda x: x in dataset)
                    ].reset_index(drop=True)
                    if not x.empty:
                        ML_rates.append(x.loc[0, "rate_norm"])
                    elif x.empty:
                        ML_rates.append(np.NaN)

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
                bp = ax.boxplot(
                    rates,
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    showfliers=False,
                    meanprops=dict(color="black", linestyle="-"),
                )
                # bp = ax.violinplot(
                #     rates,
                #     showmedians=True,
                #     showextrema=False,
                # )

                # for i, pos in enumerate(
                #     range(1, len(plot_order) + 1)
                # ):  # Boxplot x-positions are 1-based index
                #     print(i, pos)
                #     print(ML_rates[i])
                #     ax.plot(
                #         pos, ML_rates[i], "ro", markersize=10
                #     )  # 'ro' specifies red color and circle marker

                for median in bp["medians"]:
                    median.set_visible(False)
                    # median.set(color="black")
                for k, box in enumerate(bp["boxes"]):
                    box.set_facecolor(sns.color_palette("colorblind")[k])
                # for k, pc in enumerate(bp["bodies"]):
                #     # pc.set_facecolor(sns.color_palette("colorblind")[k % 3])
                #     # pc.set_facecolor(sns.color_palette("colorblind")[-((k - 1) // -2)])
                #     pc.set_facecolor(sns.color_palette("colorblind")[k])
                #     # pc.set_edgecolor("black")
                #     pc.set_alpha(1)
                # bp["cmedians"].set_colors("black")
                # ax.set_title(transition)
                if norm_method == "zscore":
                    ax.set_ylim(-2.7, 8)  # for z-score normalisation
                elif norm_method == "zscore+2.7":
                    ax.set_ylim(0, 10.7)  # for z-score norm + 2.7
                elif norm_method == "zscore_global":
                    ax.set_ylim(-2, 5)
                elif norm_method == "meanmean":
                    ax.set_ylim(0, 8)  # for mean-mean normalisation
                elif norm_method == "minmax":
                    ax.set_ylim(-0.5, 3)  # for min-max normalisation

                # plot ML values
                if ML_data:
                    pos = list(range(1, len(plot_order) + 1))
                    ax.scatter(
                        pos, ML_rates, color="black", zorder=5, s=8, facecolors="white"
                    )  # , marker="D")

            # if j == 0:
            #     ax.set_ylabel("Normalised rate")
            xticklabs = ["S1", "S2"]
            xticklabs.extend([f"P{i}" for i in range(1, len(plot_order) - 1)])
            if i == 3:
                # ax.set_xticks(
                #     list(range(1, len(plot_order) + 1)),
                #     # plot_order,
                #     # ["S1", "S2", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
                #     xticklabs,
                #     fontsize=9,
                # )
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    ["MUT1", "MUT2", "Janssens", "Zuntini", "Geeta"],
                    fontsize=9,
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                # ax.set_xlabel("Dataset")
            if j == 3 and i == 2:
                # ax.set_xticks(
                #     list(range(1, len(plot_order) + 1)),
                #     # plot_order,
                #     # ["S1", "S2", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
                #     xticklabs,
                #     fontsize=9,
                # )
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    ["MUT1", "MUT2", "Janssens", "Zuntini", "Geeta"],
                    fontsize=9,
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                # ax.set_xlabel("Dataset")

            # if (i, j) == (0, 1):
            #     ax.set_ylabel("Unlobed")
            if j != 0 and (i, j) != (0, 1):
                ax.set_yticklabels([])
            if i != 3 and (i, j) != (2, 3):
                ax.set_xticklabels([])

    fig.supxlabel("Dataset")
    fig.supylabel("Normalised rate")

    labels_alt = plot_order
    # labels_alt = [
    #     "S1: MUT1 Simulation",
    #     "S2: MUT2 Simulation",
    #     "P1: Janssens et al. (2020)\nphylogeny, Naturalis\nclassification",
    #     # "P2: Janssens et al. (2020)\nphylogeny, Geeta et al.\n(2012) classification",
    #     # "P3: Soltis et al. (2011)\nphylogeny, Naturalis\nclassification",
    #     # "P4: Soltis et al. (2011)\nphylogeny, Geeta et al.\n(2012) classification",
    #     "P5: Zuntini et al. (2024)\nphylogeny, Naturalis\nclassification",
    #     # "P6: Zuntini et al. (2024)\nphylogeny, Geeta et al.\n(2012) classification",
    #     # "P7: Geeta et al. (2012)\nphylogeny, Naturalis\nclassification",
    #     "P8: Geeta et al. (2012)\nphylogeny, Geeta et al.\n(2012) classification",
    #     # "P7: jan_phylo_nat_class\n21-01-24_95_each",
    #     # "P8: jan_phylo_nat_class\n_21-01-24_shuff",
    # ]
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

    if legend:
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
    if legend:
        if (
            norm_method == "zscore"
            or norm_method == "zscore+2.7"
            or norm_method == "zscore_global"
        ):
            plt.subplots_adjust(
                hspace=0.2, wspace=0.2, right=0.745, left=0.064
            )  # for z-score-norm
        else:
            plt.subplots_adjust(
                hspace=0.2, wspace=0.2, right=0.745, left=0.044
            )  # for min-max-norm or mean-mean-norm
    else:
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()


def plot_phylo_and_sim_rates_restricted(norm_method, ML_data):
    plot_order = [
        "MUT1_simulation",
        "MUT2_simulation",
        "jan_phylo_nat_class_uniform0-0.1_red_1",
        "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_red_1",
        "geeta_phylo_geeta_class_uniform0-100_red_1",
    ]

    ML_phylo_rates_long = import_phylo_ML_rates(ML_data)
    phylo_sim_long = import_phylo_and_sim_rates(plot_order)
    phylo_sim_long, ML_phylo_rates_long, summary = normalise_rates(
        phylo_sim_long, ML_phylo_rates_long, norm_method
    )

    #### plotting ####

    # # remove dissected as all
    # phylo_sim_long = phylo_sim_long[phylo_sim_long["transition"] != "d→u"]
    # phylo_sim_long = phylo_sim_long[phylo_sim_long["transition"] != "d→l"]
    # phylo_sim_long = phylo_sim_long[phylo_sim_long["transition"] != "d→c"]
    # phylo_sim_long = phylo_sim_long[phylo_sim_long["transition"] != "u→d"]
    # phylo_sim_long = phylo_sim_long[phylo_sim_long["transition"] != "l→d"]
    # phylo_sim_long = phylo_sim_long[phylo_sim_long["transition"] != "c→d"]

    plt.rcParams["font.family"] = "CMU Serif"
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(10, 8),  # sharey=True
    )  # , layout="constrained")

    counter = -1
    legend_labels = []
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i == j:
                axes[i, j].axis("off")
            if i != j:
                counter += 1
                transition = list(rates_map4.values())[counter]
                plot_data = phylo_sim_long[phylo_sim_long["transition"] == transition]
                ML_plot_data = ML_phylo_rates_long[
                    ML_phylo_rates_long["transition"] == transition
                ]
                rates = []
                ML_rates = []
                for k, dataset in enumerate(plot_order):
                    # dset.append(dataset)
                    rates.append(
                        plot_data["rate_norm"][
                            plot_data["Dataset"] == dataset
                        ].squeeze()
                    )

                    # get ML_rates in correct order
                    x = ML_plot_data[
                        ML_plot_data["dataname"].apply(lambda x: x in dataset)
                    ].reset_index(drop=True)
                    if not x.empty:
                        ML_rates.append(x.loc[0, "rate_norm"])
                    elif x.empty:
                        ML_rates.append(np.NaN)

                    if dataset not in legend_labels:
                        legend_labels.append(dataset)

                ax.axvline(2.5, linestyle="--", color="grey", alpha=0.5)
                bp = ax.boxplot(
                    rates,
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    showfliers=False,
                    meanprops=dict(color="black", linestyle="-"),
                )

                for median in bp["medians"]:
                    median.set_visible(False)
                    # median.set(color="black")
                for k, box in enumerate(bp["boxes"]):
                    box.set_facecolor(sns.color_palette("colorblind")[k])
                ax.set_title(transition)
                if norm_method == "zscore":
                    ax.set_ylim(-2.7, 8)  # for z-score normalisation
                elif norm_method == "zscore+2.7":
                    ax.set_ylim(0, 10.7)  # for z-score norm + 2.7
                elif norm_method == "zscore_global":
                    ax.set_ylim(-2, 5)
                elif norm_method == "meanmean":
                    ax.set_ylim(0, 8)  # for mean-mean normalisation
                elif norm_method == "minmax":
                    ax.set_ylim(-0.5, 3)  # for min-max normalisation

                # plot ML values
                if ML_data:
                    pos = list(range(1, len(plot_order) + 1))
                    ax.scatter(
                        pos, ML_rates, color="black", zorder=5, s=7
                    )  # , marker="D")

            if j == 0:
                ax.set_ylabel("Normalised rate")
            xticklabs = ["S1", "S2"]
            xticklabs.extend([f"P{i}" for i in range(1, len(plot_order) - 1)])
            if i == 2:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    xticklabs,
                    fontsize=9,
                )
                ax.set_xlabel("Dataset")
            if j == 2 and i == 1:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    xticklabs,
                    fontsize=9,
                )
            if (i, j) == (0, 1):
                ax.set_ylabel("Normalised rate")
            if j != 0 and (i, j) != (0, 1):
                ax.set_yticklabels([])
            if i != 2 and (i, j) != (1, 2):
                ax.set_xticklabels([])
            if (i, j) == (1, 2):
                ax.set_xlabel("Dataset")
    # labels_alt = plot_order
    labels_alt = [
        "S1: MUT1 Simulation",
        "S2: MUT2 Simulation",
        "P1: Janssens et al. (2020)\nphylogeny, Naturalis\nclassification",
        #     "P2: Janssens et al. (2020)\nphylogeny, Geeta et al.\n(2012) classification",
        #     "P3: Soltis et al. (2011)\nphylogeny, Naturalis\nclassification",
        #     "P4: Soltis et al. (2011)\nphylogeny, Geeta et al.\n(2012) classification",
        "P2: Zuntini et al. (2024)\nphylogeny, Naturalis\nclassification",
        #     "P6: Zuntini et al. (2024)\nphylogeny, Geeta et al.\n(2012) classification",
        "P3: Geeta et al. (2012)\nphylogeny, Naturalis\nclassification",
        #     "P8: Geeta et al. (2012)\nphylogeny, Geeta et al.\n(2012) classification",
        #     # "P7: jan_phylo_nat_class\n21-01-24_95_each",
        #     # "P8: jan_phylo_nat_class\n_21-01-24_shuff",
    ]

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
    if (
        norm_method == "zscore"
        or norm_method == "zscore+2.7"
        or norm_method == "zscore_global"
    ):
        plt.subplots_adjust(
            hspace=0.2, wspace=0.2, right=0.745, left=0.064
        )  # for z-score-norm
    else:
        plt.subplots_adjust(
            hspace=0.2,
            wspace=0.2,
            right=0.745,
        )  # for min-max-norm or mean-mean-norm
    plt.show()


def plot_phylo_and_sim_rates_with_leaf_icons(norm_method, ML_data, legend):
    plot_order = [
        "MUT1_simulation",
        "MUT2_simulation",
        # "jan_phylo_nat_class_uniform0-0.1_1",
        # "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_2",
        # "geeta_phylo_geeta_class_uniform0-100_4",
        "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "geeta_phylo_geeta_class_uniform0-100_genus_1",
    ]

    ML_phylo_rates_long = import_phylo_ML_rates(ML_data, plot_order, calc_diff=False)
    phylo_sim_long = import_phylo_and_sim_rates(plot_order, calc_diff=False)
    phylo_sim_long, ML_phylo_rates_long, summary = normalise_rates(
        phylo_sim_long, ML_phylo_rates_long, norm_method
    )

    #### plotting ####

    plt.rcParams["font.family"] = "CMU Serif"
    if legend:
        fig, axes = plt.subplots(
            nrows=4,
            ncols=4,
            figsize=(10, 8),  # sharey=True
        )  # , layout="constrained")
    else:
        fig, axes = plt.subplots(
            nrows=5,
            ncols=5,
            figsize=(9, 9),  # sharey=True
        )  # , layout="constrained")
    counter = -1
    legend_labels = []
    print(axes)
    for i in range(1, 5):
        for j in range(0, 4):
            ax = axes[i, j]
            if i - 1 == j:
                ax.axis("off")
            if i - 1 != j:
                counter += 1
                transition = list(rates_map2.values())[counter]
                plot_data = phylo_sim_long[phylo_sim_long["transition"] == transition]
                ML_plot_data = ML_phylo_rates_long[
                    ML_phylo_rates_long["transition"] == transition
                ]
                rates = []
                ML_rates = []
                for k, dataset in enumerate(plot_order):
                    # dset.append(dataset)
                    rates.append(
                        plot_data["rate_norm"][
                            plot_data["Dataset"] == dataset
                        ].squeeze()
                    )

                    # get ML_rates in correct order
                    x = ML_plot_data[
                        ML_plot_data["dataname"].apply(lambda x: x in dataset)
                    ].reset_index(drop=True)
                    if not x.empty:
                        ML_rates.append(x.loc[0, "rate_norm"])
                    elif x.empty:
                        ML_rates.append(np.nan)
                    if dataset not in legend_labels:
                        legend_labels.append(dataset)
                ax.axvline(2.5, linestyle="--", color="grey", alpha=0.5)
                # bp = ax.boxplot(
                #     rates,
                #     patch_artist=True,
                #     showmeans=True,
                #     meanline=True,
                #     showfliers=False,
                #     meanprops=dict(color="black", linestyle="-"),
                # )
                bp = ax.violinplot(rates, showextrema=False, showmeans=True)

                # for median in bp["medians"]:
                #     median.set_visible(False)
                # for k, box in enumerate(bp["boxes"]):
                #     box.set_facecolor(sns.color_palette("colorblind")[k])
                # ax.set_title(transition)
                if norm_method == "zscore":
                    ax.set_ylim(-2.7, 8)  # for z-score normalisation
                elif norm_method == "zscore+2.7":
                    ax.set_ylim(0, 10.7)  # for z-score norm + 2.7
                elif norm_method == "zscore_global":
                    ax.set_ylim(-2, 5)
                elif norm_method == "meanmean":
                    ax.set_ylim(0, 8)  # for mean-mean normalisation
                elif norm_method == "minmax":
                    ax.set_ylim(-0.5, 3)  # for min-max normalisation

                # plot ML values
                if ML_data:
                    pos = list(range(1, len(plot_order) + 1))
                    ax.scatter(
                        pos, ML_rates, color="black", zorder=5, s=8, facecolors="white"
                    )  # , marker="D")

            xticklabs = ["S1", "S2"]
            xticklabs.extend([f"P{i}" for i in range(1, len(plot_order) - 1)])
            if i == 4:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    ["MUT1", "MUT2", "Janssens", "Zuntini", "Geeta"],
                    fontsize=9,
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            if (i, j) == (3, 3):
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    ["MUT1", "MUT2", "Janssens", "Zuntini", "Geeta"],
                    fontsize=9,
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            if j != 0 and (i, j) != (1, 1):
                ax.set_yticklabels([])
            if i != 4 and (i, j) != (3, 3):
                ax.set_xticklabels([])

    ### plot leaf images ####
    icon_filenames = [
        "leaf_p7a_0_0.png",
        "leaf_p8ae_0_0.png",
        "leaf_pd1_0_0.png",
        "leaf_pc1_alt_0_0.png",
    ]
    icons = [os.path.join("uldc_model_icons", path) for path in icon_filenames]
    icon_imgs = [Image.open(path) for path in icons]
    img_width, img_height = icon_imgs[1].size
    scale_factor = 0.5
    shape_cats = ["Unlobed", "Lobed", "Dissected", "Compound"]

    for j in range(0, 4):
        ax = axes[0, j]
        ax.axis("off")
        ax.imshow(icon_imgs[j])
        ax.text(img_width / 2, img_height, shape_cats[j], ha="center", va="top")
        ax.set_xlim(img_width / scale_factor, (-img_width / 2) / scale_factor)
        ax.set_ylim(img_height, -(img_height / scale_factor))
    for i in range(1, 5):
        ax = axes[i, 4]
        ax.axis("off")
        ax.imshow(icon_imgs[i - 1])
        ax.text(img_width / 2, img_height, shape_cats[i - 1], ha="center", va="top")
        ax.set_xlim(0, (img_width / scale_factor) + ((img_width / 2) / scale_factor))
        ax.set_ylim(img_height / scale_factor, (-img_height / 2) / scale_factor)
    axes[0, 4].axis("off")

    xlab_pos = 0.43
    ylab_pos = 0.45
    fig.supxlabel("Dataset", x=xlab_pos, ha="center")
    fig.supylabel("Normalised rate", y=ylab_pos, ha="center", va="center")
    fig.text(xlab_pos, 0.9, "Final shape", ha="center", va="center", fontsize=12)
    fig.text(
        0.9,
        ylab_pos,
        "Initial shape",
        ha="center",
        va="center",
        rotation=270,
        fontsize=12,
    )
    plt.tight_layout()
    if legend:
        if (
            norm_method == "zscore"
            or norm_method == "zscore+2.7"
            or norm_method == "zscore_global"
        ):
            plt.subplots_adjust(
                hspace=0.2, wspace=0.2, right=0.745, left=0.064
            )  # for z-score-norm
        else:
            plt.subplots_adjust(
                hspace=0.2, wspace=0.2, right=0.745, left=0.044
            )  # for min-max-norm or mean-mean-norm
    else:
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()


def plot_rates_diff(norm_method, ML_data):
    plot_order = [
        "MUT1_simulation",
        "MUT2_simulation",
        # "jan_phylo_nat_class_uniform0-0.1_5",
        # "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_5",
        # "geeta_phylo_geeta_class_uniform0-100_6",
        "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "geeta_phylo_geeta_class_uniform0-100_genus_1",
        # "jan_equal_genus_phylo_nat_class_uniform0-0.1_4",
        # "zuntini_genera_equal_genus_phylo_nat_class_uniform0-0.1_4",
        # "geeta_phylo_geeta_class_uniform0-100_4",
    ]

    # plot_titles = ["u-l", "u-d", "u-c", "l-d", "l-c", "d-c"]
    plot_titles = [
        "l→u-u→l",
        "d→u-u→d",
        "c→u-u→c",
        "l→d-d→l",
        "l→c-c→l",
        "d→c-c→d",
    ]
    # plot_transitions = ["u→l", "u→d", "u→c", "l→d", "l→c", "d→c"]
    plot_transitions = ["l→u", "d→u", "c→u", "l→d", "l→c", "d→c"]

    ML_phylo_rates_long = import_phylo_ML_rates(ML_data, plot_order, calc_diff=True)
    phylo_sim_long = import_phylo_and_sim_rates(plot_order, calc_diff=True)
    phylo_sim_long, ML_phylo_rates_long, summary = normalise_rates(
        phylo_sim_long, ML_phylo_rates_long, norm_method
    )

    # mcmc_diff, ml_diff = get_raw_rates_diff(phylo_sim_long, ML_phylo_rates_long)

    #### plotting ####

    plt.rcParams["font.family"] = "CMU Serif"

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(12, 9),  # sharey=True
    )  # , layout="constrained")
    counter = -1
    legend_labels = []
    print(axes)
    for i, transition in enumerate(plot_titles):
        ax = axes.flat[i]
        # mcmc_plot_data = mcmc_diff[mcmc_diff["transition"] == transition]
        # ml_plot_data = ml_diff[ml_diff["transition"] == transition]
        mcmc_plot_data = phylo_sim_long[phylo_sim_long["transition"] == transition]
        ml_plot_data = ML_phylo_rates_long[
            ML_phylo_rates_long["transition"] == transition
        ]

        rates = []
        ml_rates = []
        for k, dataset in enumerate(plot_order):
            # rates.append(
            #     mcmc_plot_data["rate_norm_diff"][
            #         mcmc_plot_data["Dataset"] == dataset
            #     ].squeeze()
            # )
            rates.append(
                mcmc_plot_data["rate_norm"][
                    mcmc_plot_data["Dataset"] == dataset
                ].squeeze()
            )

            # get ML_rates in correct order
            x = ml_plot_data[
                ml_plot_data["dataname"].apply(lambda x: x in dataset)
            ].reset_index(drop=True)
            if not x.empty:
                # ml_rates.append(x.loc[0, "rate_norm_diff"])
                ml_rates.append(x.loc[0, "rate_norm"])
            elif x.empty:
                ml_rates.append(np.nan)
        ax.axvline(2.5, linestyle="--", color="grey", alpha=0.5)
        ax.axhline(0, linestyle="--", color="C1", alpha=0.5)
        bp = ax.violinplot(
            rates,
            # patch_artist=True,
            # showmeans=True,
            # meanline=True,
            # showfliers=False,
            # meanprops=dict(color="black", linestyle="-"),
            showextrema=False,
            showmeans=True,
        )

        # for median in bp["medians"]:
        #     median.set_visible(False)
        # for k, box in enumerate(bp["boxes"]):
        #     box.set_facecolor(sns.color_palette("colorblind")[k])
        ax.set_title(plot_titles[i])
        ax.set_ylim(-8, 8)
        ax.set_xticks(
            list(range(1, len(plot_order) + 1)),
            ["MUT1", "MUT2", "Janssens", "Zuntini", "Geeta"],
            fontsize=9,
        )
        ax.set_xticklabels(ax.get_xticklabels())  # , rotation=45)
        # if norm_method == "zscore":
        #     ax.set_ylim(-2.7, 8)  # for z-score normalisation
        # elif norm_method == "zscore+2.7":
        #     ax.set_ylim(0, 10.7)  # for z-score norm + 2.7
        # elif norm_method == "zscore_global":
        #     ax.set_ylim(-2, 5)
        # elif norm_method == "meanmean":
        #     ax.set_ylim(0, 8)  # for mean-mean normalisation
        # elif norm_method == "minmax":
        #     ax.set_ylim(-0.5, 3)  # for min-max normalisation

        # plot ML values
        if ML_data:
            pos = list(range(1, len(plot_order) + 1))
            ax.scatter(
                pos, ml_rates, color="black", zorder=5, s=8, facecolors="white"
            )  # , marker="D")
    fig.supxlabel("Dataset")
    fig.supylabel("Difference between back and forwards normalised rate")
    plt.show()


def plot_phylo_and_sim_rates_shuffled():

    phylo_sim_long = import_phylo_and_sim_rates()

    #### For the Shuffle test

    plot_order = [
        # "MUT1_simulation",
        # "MUT2_simulation",
        # "jan_phylo_nat_class",
        # "jan_phylo_geeta_class",
        # "zuntini_phylo_nat_class",
        # "zuntini_phylo_geeta_class",
        # "geeta_phylo_nat_class",
        # "geeta_phylo_geeta_class",
        # "geeta_phylo_geeta_class_23-04-24_17_each",
        # "geeta_phylo_geeta_class_23-04-24_shuff",
        # "geeta_phylo_geeta_class_23-04-24_mle",
        # "geeta_phylo_geeta_class_23-04-24_17_each_mle",
        # "geeta_phylo_geeta_class_23-04-24_shuff_mle",
        "jan_phylo_nat_class_prior_0-1_burnin100000_run4",
        # "jan_phylo_nat_class_21-01-24_95_each_prior_0-1_burnin100000_run2",
        "jan_phylo_nat_class_21-01-24_shuff_prior_0-1_burnin100000_run1",
        # "jan_phylo_nat_class_21-01-24_cenrich_sub_prior_0-1_burnin100000_run1",
        # "jan_phylo_nat_class_mle",
        "jan_phylo_nat_class_21-01-24_95_each_mle",
        # "jan_phylo_nat_class_21-01-24_shuff_mle",
        "jan_phylo_nat_class_21-01-24_cenrich_sub_mle",
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
                transition = list(rates_map2.values())[counter]
                plot_data = phylo_sim_long[phylo_sim_long["transition"] == transition]
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

                ax.axvline(2.5, linestyle="--", color="grey", alpha=0.5)
                # ax.axvline(4.5, linestyle="--", color="grey", alpha=0.5)
                ax.text(1, 5, "MCMC", color="grey")
                ax.text(3, 5, "MLE", color="grey")
                # ax.text(1.5, 5, "MCMC", color="grey")
                # ax.text(6, 5, "MLE", color="grey")
                # ax.axvline(2.5, linestyle="--", color="grey", alpha=0.5)
                bp = ax.boxplot(
                    rates,
                    patch_artist=True,
                    # showmeans=True,
                    # meanline=True,
                    showfliers=False,
                )
                # bp = ax.violinplot(
                #     rates,
                #     showmedians=True,
                #     showextrema=False,
                # )

                # for median in bp["medians"]:
                #     # median.set_visible(False)
                #     median.set(color="black")
                # for k, box in enumerate(bp["boxes"]):
                #     box.set_facecolor(sns.color_palette("colorblind")[k])
                for k, pc in enumerate(bp["bodies"]):
                    pc.set_facecolor(
                        sns.color_palette("colorblind")[k % int(0.5 * len(plot_order))]
                    )
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
                    [f"P{i}" for i in range(1, len(plot_order) + 1)],
                    fontsize=9,
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.set_xlabel("Dataset")
            if j == 3 and i == 2:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    # plot_order,
                    [f"P{i}" for i in range(1, len(plot_order) + 1)],
                    fontsize=9,
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.set_xlabel("Dataset")

            if (i, j) == (0, 1):
                ax.set_ylabel("Noramlised rate")
            if j != 0 and (i, j) != (0, 1):
                ax.set_yticklabels([])
            if i != 3 and (i, j) != (2, 3):
                ax.set_xticklabels([])

    labels_alt = [
        # "P1: original mcmc",
        # "P2: 17 each, mcmc",
        # "P3: shuffled, mcmc",
        # "P4: original mle",
        # "P5: 17 each, mle",
        # "P6: shuffled, mle",
        "P1: original, prior0-1 mcmc",
        # "P2: 95 each, prior0-1 mcmc",
        "P2: shuffled, prior0-1 mcmc",
        # "P4: c-enriched, prior0-1 mcmc",
        # "P5: original mle",
        "P3: 95 each, mle",
        # "P7: shuffled, mle",
        "P4: c-enriched, mle",
    ]

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=sns.color_palette("colorblind")[i % int(0.5 * len(plot_order))],
        )
        for i, label in enumerate(
            labels_alt
        )  # change back to legend_labels if you want the default
    ]
    fig.legend(
        legend_handles,
        labels_alt,
        loc="bottom",
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


def arrow_plot_schemedraw():
    rate_data = pd.read_csv("sim_phylo_rates_stats_meanmean_mixedpriors_18-09-24.csv")
    cmap = plt.get_cmap("viridis")
    rate_data["std_c"] = rate_data["rate_norm_std"].apply(lambda x: cmap(x))
    print(rate_data["rate_norm_mean_diff"])
    print(set(rate_data["Dataset_"]))
    datasets = set(rate_data["Dataset_"])
    node_dist = 8
    for dataset in datasets:
        plot_data = rate_data[rate_data["Dataset_"] == dataset]
        with schemdraw.Drawing() as d:
            u = flow.Circle().at((0, 0)).label("U")
            l = flow.Circle().at((node_dist, 0)).label("L")
            d = flow.Circle().at((node_dist, -node_dist)).label("D")
            c = flow.Circle().at((0, -node_dist)).label("C")

            for i, row in plot_data.iterrows():
                at = row["transition_"][0]
                to = row["transition_"][2]
                rate = row["rate_norm_mean_diff"] * 10
                rate_c = row["std_c"]

                if rate > 0:
                    if at == "l" and to == "u":
                        elm.Arc2(k=-0.3, arrow="->", lw=rate, color=rate_c).at(l.NW).to(
                            u.NE
                        )
                        # elm.Arrowhead(headwith=rate * 10, headlength=rate * 10)
                    if at == "u" and to == "l":
                        elm.Arc2(k=-0.3, arrow="->", lw=rate, color=rate_c).at(u.SE).to(
                            l.SW
                        )
                    if at == "d" and to == "l":
                        elm.Arc2(k=-0.3, arrow="->", lw=rate, color=rate_c).at(d.NE).to(
                            l.SE
                        )
                    if at == "l" and to == "d":
                        elm.Arc2(k=-0.3, arrow="->", lw=rate, color=rate_c).at(l.SW).to(
                            d.NW
                        )
                    if at == "c" and to == "d":
                        elm.Arc2(k=-0.3, arrow="->", lw=rate, color=rate_c).at(c.SE).to(
                            d.SW
                        )
                    if at == "d" and to == "c":
                        elm.Arc2(k=-0.3, arrow="->", lw=rate, color=rate_c).at(d.NW).to(
                            c.NE
                        )
                    if at == "c" and to == "u":
                        elm.Arc2(k=-0.3, arrow="->", lw=rate, color=rate_c).at(c.NE).to(
                            u.SE
                        )
                    if at == "u" and to == "c":
                        elm.Arc2(k=-0.3, arrow="->", lw=rate, color=rate_c).at(u.SW).to(
                            c.NW
                        )
                    #### diagonals
                    if at == "c" and to == "l":
                        elm.Arc2(k=-0.3, arrow="->", lw=rate, color=rate_c).at(c.E).to(
                            l.S
                        )
                    if at == "l" and to == "c":
                        elm.Arc2(k=-0.3, arrow="->", lw=rate, color=rate_c).at(l.W).to(
                            c.N
                        )
                    if at == "d" and to == "u":
                        elm.Arc2(k=0.3, arrow="->", lw=rate, color=rate_c).at(d.W).to(
                            u.S
                        )
                    if at == "u" and to == "d":
                        elm.Arc2(k=0.3, arrow="->", lw=rate, color=rate_c).at(u.E).to(
                            d.N
                        )


def blank_diagram(
    fig_width=9,
    fig_height=8,
    bg_color="white",
):
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.set_facecolor(bg_color)

    # ax.tick_params(bottom=False, top=False, left=False, right=False)
    # ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    return fig, ax


def arc(ax, startpoint, endpoint, curvature, rate, rate_c, rate_std, significant, c_val):
    style = ArrowStyle(
        "Simple",
        head_length=1,
        head_width=1.5,
        tail_width=1,  # length_includes_head=True
    )
    # increase c_val to increase the curviness of the arrows

    if not rate_std:

        if significant:
            arrow = FancyArrowPatch(
                (startpoint[0], startpoint[1]),  # Start point
                (endpoint[0], endpoint[1]),  # End point
                connectionstyle=f"arc3,rad={curvature}.{c_val}",  # Curvature
                mutation_scale=10,  # Arrow head size
                arrowstyle="-|>",  # Arrow style
                # arrowstyle=style,
                color=rate_c,  # Arrow color
                lw=rate,  # Line width
            )
        else:
            arrow = FancyArrowPatch(
                (startpoint[0], startpoint[1]),  # Start point
                (endpoint[0], endpoint[1]),  # End point
                connectionstyle=f"arc3,rad={curvature}.{c_val}",  # Curvature
                mutation_scale=10,  # Arrow head size
                arrowstyle="-|>",  # Arrow style
                # linestyle="-",
                # arrowstyle=style,
                color=rate_c,  # Arrow color
                lw=rate,  # Line width
                zorder=0,
            )
        arrow.set_joinstyle("miter")
        # arrow.set_
        arrow.set_capstyle("butt")
        ax.add_patch(arrow)

    if rate_std:
        arrow_ub = FancyArrowPatch(
            (startpoint[0], startpoint[1]),  # Start point
            (endpoint[0], endpoint[1]),  # End point
            connectionstyle=f"arc3,rad={curvature}.{c_val}",  # Curvature
            mutation_scale=10,  # Arrow head size
            arrowstyle="-|>",  # Arrow style
            # arrowstyle=style,
            color="black",  # Arrow color
            fill=False,
            # alpha=0.2,
            # linestyle="--",
            lw=rate + rate_std + 10,  # Line width
        )
        arrow_ub.set_joinstyle("miter")
        arrow = FancyArrowPatch(
            (startpoint[0], startpoint[1]),  # Start point
            (endpoint[0], endpoint[1]),  # End point
            connectionstyle=f"arc3,rad={curvature}.{c_val}",  # Curvature
            mutation_scale=10,  # Arrow head size
            arrowstyle="-|>",  # Arrow style
            # arrowstyle=style,
            alpha=0.2,
            color=rate_c,  # Arrow color
            lw=rate,  # Line width
        )
        arrow.set_joinstyle("miter")
        arrow_lb = FancyArrowPatch(
            (startpoint[0], startpoint[1]),  # Start point
            (endpoint[0], endpoint[1]),  # End point
            connectionstyle=f"arc3,rad={curvature}.{c_val}",  # Curvature
            mutation_scale=10,  # Arrow head size
            arrowstyle="-|>",  # Arrow style
            # arrowstyle=style,
            # color="white",  # Arrow color
            fill=False,
            color="white",
            # linestyle="--",
            lw=rate - rate_std,  # Line width
        )
        arrow_lb.set_joinstyle("miter")

        # ax.add_patch(arrow)
        ax.add_patch(arrow_ub)
        ax.add_patch(arrow_lb)
        # arrow_ub = mpl.lines.Line2D(
        #     (startpoint[0], startpoint[1]),
        #     (endpoint[0], endpoint[1]),
        #     linestyle="--",
        #     lw=rate + rate_std,
        # )


def nodes(ax, c_proper, rad, texts, icon_imgs):
    # textplacement goes from top right clockwise
    # text_placement = ((-0.7, 0.7), (0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)) # original
    text_placement = ((-0.1, 1), (0.1, 1), (0.1, -1), (-0.1, -1))
    for i, center in enumerate(c_proper.values()):
        x, y = center
        theta = np.linspace(0, 2 * np.pi, 100)
        # ax.plot(
        #     x + rad * np.cos(theta),
        #     y + rad * np.sin(theta),
        #     color="black",
        # )
        # resize_img = icon_imgs[i].resize((1024, 984), Image.LANCZOS)

        # img_box = mpl.offsetbox.OffsetImage(resize_img, zoom=0.04)
        # ab = mpl.offsetbox.AnnotationBbox(
        #     img_box, (x, y), frameon=False, box_alignment=(0.5, 0.5)
        # )
        # ax.add_artist(ab)
        w, h = icon_imgs[i].size
        sf = 0.0021

        ax.imshow(
            icon_imgs[i],
            extent=[
                x - (w * sf / 2),
                x + (w * sf / 2),
                y - (h * sf / 2),
                y + (h * sf / 2),
            ],
            rasterized=True,
            # interpolation="none",
        )

        ax.text(
            x + text_placement[i][0],
            y + text_placement[i][1],
            texts[i],
            horizontalalignment="left" if i == 1 or i == 2 else "right",
            verticalalignment="center",
            color="black",
        )


def arrow_plot(norm_method, ML_data, colourise):
    plt.rcParams["font.family"] = "CMU Serif"
    plot_order = [
        "MUT1_simulation",
        "MUT2_simulation",
        # "jan_phylo_nat_class_uniform0-0.1_5",
        # "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_5",
        # "geeta_phylo_geeta_class_uniform0-100_6",
        "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "geeta_phylo_geeta_class_uniform0-100_genus_1",
        # "jan_equal_genus_phylo_nat_class_uniform0-0.1_4",
        # "zuntini_genera_equal_genus_phylo_nat_class_uniform0-0.1_4",
    ]
    icon_filenames = [
        "leaf_p7a_0_0.png",
        "leaf_p8ae_0_0.png",
        "leaf_pd1_0_0.png",
        "leaf_pc1_alt_0_0.png",
    ]
    icons = [os.path.join("uldc_model_icons", path) for path in icon_filenames]
    icon_imgs = [Image.open(path) for path in icons]
    img_width, img_height = icon_imgs[1].size
    scale_factor = 0.5
    shape_cats = ["Unlobed", "Lobed", "Dissected", "Compound"]

    ML_phylo_rates_long = import_phylo_ML_rates(ML_data, plot_order, calc_diff=True)
    phylo_sim_long = import_phylo_and_sim_rates(plot_order, calc_diff=True)
    phylo_sim_long, ML_phylo_rates_long, summary = normalise_rates(
        phylo_sim_long, ML_phylo_rates_long, norm_method
    )
    rate_data = test_rates_diff_from_zero(phylo_sim_long)
    # rate_data = pd.read_csv("sim_phylo_rates_stats_meanmean_mixedpriors_18-09-24.csv")
    # rate_data.rename(columns={"Dataset_": "dataset"}, inplace=True)
    # rate_data["transition"] = (
    #     rate_data["transition_"] + "-" + rate_data["transition_rev"]
    # )
    # prop_over_zero = pd.read_csv("rates_diff_std_prop_over_zero.csv")
    # prop_over_zero_filt = prop_over_zero[["dataset", "transition", "prop_over_zero"]]
    # print(rate_data)
    # print(prop_over_zero)
    # print(prop_over_zero_filt)
    rate_data["fwd"] = rate_data["transition"].str[:3]
    rate_data["bwd"] = rate_data["transition"].str[-3:]
    # print(rate_data)
    rate_data.to_csv("rate_data_bw_fw.csv", index=False)

    cmap = plt.get_cmap("viridis")
    rate_data["std_c"] = rate_data["std"].apply(lambda x: cmap(x))
    print(rate_data)
    # print(rate_data["rate_norm_mean_diff"])
    # print(set(rate_data["dataset"]))
    datasets = set(rate_data["dataset"])
    node_dist = 8

    # centers = [(2, 6), (6, 6), (6, 2), (2, 2)]
    c_proper = {"u": (2, 6), "l": (6, 6), "d": (6, 2), "c": (2, 2)}

    rad = 0.5
    texts = [
        "Unlobed",
        "Lobed",
        "Dissected",
        "Compound",
    ]
    c = {
        "uN": (2, 6 + rad),
        "uE": (2 + rad, 6),
        "uS": (2, 6 - rad),
        "uW": (2 - rad, 6),
        "lN": (6, 6 + rad),
        "lE": (6 + rad, 6),
        "lS": (6, 6 - rad),
        "lW": (6 - rad, 6),
        "dN": (6, 2 + rad),
        "dE": (6 + rad, 2),
        "dS": (6, 2 - rad),
        "dW": (6 - rad, 2),
        "cN": (2, 2 + rad),
        "cE": (2 + rad, 2),
        "cS": (2, 2 - rad),
        "cW": (2 - rad, 2),
    }

    # Draw circles with text in the center
    if len(plot_order) > 1:
        fig, axs = plt.subplots(2, 3, figsize=(14, 9))
        for ax in axs.flat:
            ax.axis("off")
    else:
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        axs.axis("off")

    for d, dataset in enumerate(plot_order):
        ax = axs.flat[d] if len(plot_order) > 1 else axs
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)

        plot_data = rate_data[rate_data["dataset"] == dataset]
        # fig, ax = blank_diagram()
        nodes(ax, c_proper, rad, texts, icon_imgs)
        for i, row in plot_data.iterrows():
            # print(row)
            at = row["fwd"][0]
            to = row["fwd"][2]
            r = row["mean_rate_norm"] * 10
            # rc = row["std_c"] if colourise else "black"
            rc = "lightgrey" if row["prop_over_zero"] < 0.90 else "black"
            sig = False if row["prop_over_zero"] < 0.90 else True
            # rs = row["rate_norm_std"]
            rs = False  # set to false to disable multi-arrows
            c_val = 2 # increase to increase the curviness of the arrows

            if r > 0:
                if at + to == "ul":
                    arc(ax, c["uS"], c["lS"], "+", r, rc, rs, sig, c_val)
                if at + to == "lu":
                    arc(ax, c["lN"], c["uN"], "+", r, rc, rs, sig, c_val)
                if at + to == "ld":
                    arc(ax, c["lE"], c["dE"], "-", r, rc, rs, sig, c_val)
                if at + to == "dl":
                    arc(ax, c["dE"], c["lE"], "+", r, rc, rs, sig, c_val)
                if at + to == "dc":
                    arc(ax, c["dS"], c["cS"], "-", r, rc, rs, sig, c_val)
                if at + to == "cd":
                    arc(ax, c["cS"], c["dS"], "+", r, rc, rs, sig, c_val)
                if at + to == "cu":
                    arc(ax, c["cW"], c["uW"], "-", r, rc, rs, sig, c_val)
                if at + to == "uc":
                    arc(ax, c["uE"], c["cE"], "-", r, rc, rs, sig, c_val)

                    #### diagonals ####
                if at == "u" and to == "d":
                    arc(ax, c["uS"], c["dW"], "+", r, rc, rs, sig, 3)
                if at == "d" and to == "u":
                    arc(ax, c["dW"], c["uS"], "-", r, rc, rs, sig, 3)
                if at == "c" and to == "l":
                    arc(ax, c["cE"], c["lS"], "+", r, rc, rs, sig, 3)
                if at == "l" and to == "c":
                    arc(ax, c["lS"], c["cE"], "-", r, rc, rs, sig, 3)
        ax.set_title(dataset, fontsize=8)
        if colourise:
            norm = mpl.colors.Normalize(
                vmin=min(rate_data["rate_norm_std"]),
                vmax=max(rate_data["rate_norm_std"]),
            )
            cbar_ax = fig.add_axes(
                [0.85, 0.25, 0.03, 0.5]
            )  # Position (left, bottom, width, height)
            fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                shrink=0.2,
                label="Normalised stdev",
                cax=cbar_ax,
            )
    plt.tight_layout()
    # plt.savefig("arrow_plot.pdf", format="pdf", dpi=1200)
    # plt.show()
    return axs, phylo_sim_long, ML_phylo_rates_long, plot_order

    # with schemdraw.Drawing() as d:
    #         u = flow.Circle().at((0, 0)).label("U")
    #         l = flow.Circle().at((4, 0)).label("L")
    #         d = flow.Circle().at((4, -4)).label("D")
    #         c = flow.Circle().at((0, -4)).label("C")
    #         elm.Arc2(k=-0.3, arrow="->").at(l.NW).to(u.NE)
    #         elm.Arc2(k=-0.3, arrow="->").at(u.SE).to(l.SW)
    #         elm.Arc2(k=-0.3, arrow="->").at(d.NE).to(l.SE)
    #         elm.Arc2(k=-0.3, arrow="->").at(l.SW).to(d.NW)
    #         elm.Arc2(k=-0.3, arrow="->").at(c.SE).to(d.SW)
    #         elm.Arc2(k=-0.3, arrow="->").at(d.NW).to(c.NE)
    #         elm.Arc2(k=-0.3, arrow="->").at(c.NE).to(u.SE)
    #         elm.Arc2(k=-0.3, arrow="->").at(u.SW).to(c.NW)
    #         #### diagonals
    #         elm.Arc2(k=-0.3, arrow="->").at(c.E).to(l.S)
    #         elm.Arc2(k=-0.3, arrow="->").at(l.W).to(c.N)
    #         elm.Arc2(k=0.3, arrow="->").at(d.W).to(u.S)
    #         elm.Arc2(k=0.3, arrow="->").at(u.E).to(d.N)


def get_phylo_stats():
    labels = []
    wd = "phylogenies/final_data/labels"
    for file in os.listdir(wd):
        df = pd.read_csv(
            os.path.join(wd, file), sep="\t", header=None, names=["taxa", "shape"]
        )
        dataset = file[:-4]
        df.insert(0, "dataset", dataset)
        labels.append(df)
    labels_df = pd.concat(labels)
    print(labels_df)
    counts = labels_df.groupby("dataset")["shape"].value_counts().unstack(fill_value=0)
    print(counts)
    counts.to_csv("phylo_stats.csv")


def test_rates_diff_from_zero_old(norm_method, ML_data):
    plot_order = [
        "MUT1_simulation",
        "MUT2_simulation",
        # "jan_phylo_nat_class_uniform0-0.1_1",
        # "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_2",
        # "geeta_phylo_geeta_class_uniform0-100_4",
        "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "geeta_phylo_geeta_class_uniform0-100_genus_1",
    ]
    transitions = [
        "l→u-u→l",
        "d→u-u→d",
        "c→u-u→c",
        "l→d-d→l",
        "l→c-c→l",
        "d→c-c→d",
    ]
    ML_phylo_rates_long = import_phylo_ML_rates(ML_data, plot_order, calc_diff=True)
    phylo_sim_long = import_phylo_and_sim_rates(plot_order, calc_diff=True)
    phylo_sim_long, ML_phylo_rates_long, summary = normalise_rates(
        phylo_sim_long, ML_phylo_rates_long, norm_method
    )

    # for name, group in phylo_sim_long
    results = []
    for i, transition in enumerate(transitions):
        mcmc_plot_data = phylo_sim_long[phylo_sim_long["transition"] == transition]
        for name, group in mcmc_plot_data.groupby("Dataset"):
            n = len(group)
            ng0 = len(group[group["rate_norm"] > 0])

            mean = group["rate_norm"].mean()
            # t_stat, p_val = stats.ttest_1samp(group["rate_norm"], 0)
            std = group["rate_norm"].std()
            results.append(
                {
                    "dataset": name,
                    "transition": transition,
                    "mean_rate_norm": mean,
                    # "t_stat": t_stat,
                    # "p_val": p_val,
                    "std": std,
                    "lb": mean - std,
                    "ub": mean + std,
                    "n": n,
                    "prop_over_zero": ng0 / n,
                }
            )
    r = pd.DataFrame(results)
    print(r)
    # r.to_csv("rates_diff_t_test.csv", index=False)

    # error = [r["mean_rate_norm"] - r["lb"], r["mean_rate_norm"] + r["ub"]]
    # fig, axes = plt.subplots(
    #     nrows=2,
    #     ncols=3,
    #     figsize=(12, 9),  # sharey=True
    # )  # , layout="constrained")

    # for ax, transition in zip(axes.flat, r["transition"].unique()):
    #     r_filt = r[r["transition"] == transition]
    #     ax.errorbar(
    #         r_filt["dataset"],
    #         r_filt["mean_rate_norm"],
    #         yerr=r_filt["std"] * 2,
    #         fmt="o",
    #         capsize=5,
    #         elinewidth=1,
    #         markersize=8,
    #         label="Data points with error",
    #     )
    #     ax.axhline(0, linestyle="--", color="C1", alpha=0.5)
    #     ax.set_title(transition)
    #     ax.set_xticks(
    #         list(range(0, len(plot_order))),
    #         ["MUT1", "MUT2", "Janssens", "Zuntini", "Geeta"],
    #         fontsize=9,
    #     )
    # plt.tight_layout()
    # plt.show()
    # print(r)


def arrow_w_viol(norm_method, ML_data, layout):
    plt.rcParams["font.family"] = "CMU Serif"
    plot_order = [
        "MUT1_simulation",
        "MUT2_simulation",
        # "jan_phylo_nat_class_uniform0-0.1_5",
        # "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_5",
        # "geeta_phylo_geeta_class_uniform0-100_6",
        "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "geeta_phylo_geeta_class_uniform0-100_genus_1",
        # "jan_equal_genus_phylo_nat_class_uniform0-0.1_4",
        # "zuntini_genera_equal_genus_phylo_nat_class_uniform0-0.1_4",
    ]
    icon_filenames = [
        "leaf_p7a_0_0.png",
        "leaf_p8ae_0_0.png",
        "leaf_pd1_0_0.png",
        "leaf_pc1_alt_0_0.png",
    ]
    transition_filenames = [
        "ulvd.png",
        "udvd.png",
        "ucvd.png",
        "ldvd.png",
        "lcvd.png",
        "dcvd.png",
    ]
    icons = [os.path.join("uldc_model_icons", path) for path in icon_filenames]
    transition_icons = [
        os.path.join("uldc_model_icons", path) for path in transition_filenames
    ]
    icon_imgs = [Image.open(path) for path in icons]
    trans_imgs = [Image.open(path) for path in transition_icons]
    trans_img_w, trans_img_h = trans_imgs[1].size
    img_width, img_height = icon_imgs[1].size
    scale_factor = 0.5
    shape_cats = ["Unlobed", "Lobed", "Dissected", "Compound"]

    ML_phylo_rates_long = import_phylo_ML_rates(ML_data, plot_order, calc_diff=True)
    ML_sim_rates_long = import_sim_ML_rates(calc_diff=True)
    phylo_sim_long = import_phylo_and_sim_rates(plot_order, calc_diff=True)
    phylo_sim_long, ML_phylo_rates_long, ML_sim_rates_long, summary = normalise_rates(
        phylo_sim_long, ML_phylo_rates_long, ML_sim_rates_long, norm_method
    )

    rate_data = test_rates_diff_from_zero(phylo_sim_long)
    # rate_data = pd.read_csv("sim_phylo_rates_stats_meanmean_mixedpriors_18-09-24.csv")
    # rate_data.rename(columns={"Dataset_": "dataset"}, inplace=True)
    # rate_data["transition"] = (
    #     rate_data["transition_"] + "-" + rate_data["transition_rev"]
    # )
    # prop_over_zero = pd.read_csv("rates_diff_std_prop_over_zero.csv")
    # prop_over_zero_filt = prop_over_zero[["dataset", "transition", "prop_over_zero"]]
    # print(rate_data)
    # print(prop_over_zero)
    # print(prop_over_zero_filt)
    rate_data["fwd"] = rate_data["transition"].str[:3]
    rate_data["bwd"] = rate_data["transition"].str[-3:]
    # print(rate_data)
    rate_data.to_csv(f"rate_data_bw_fw_{str(date.today())}.csv", index=False)

    cmap = plt.get_cmap("viridis")
    rate_data["std_c"] = rate_data["std"].apply(lambda x: cmap(x))
    print(rate_data)
    # print(rate_data["rate_norm_mean_diff"])
    # print(set(rate_data["dataset"]))
    datasets = set(rate_data["dataset"])
    node_dist = 8

    # centers = [(2, 6), (6, 6), (6, 2), (2, 2)]
    c_proper = {"u": (2, 6), "l": (6, 6), "d": (6, 2), "c": (2, 2)}

    rad = 0.5
    texts = ["U","L","D","C"] if layout == "v" else ["","","",""]
    # texts = ["","","",""] # labels for graph nodes

    c = {
        "uN": (2, 6 + rad),
        "uE": (2 + rad, 6),
        "uS": (2, 6 - rad),
        "uW": (2 - rad, 6),
        "lN": (6, 6 + rad),
        "lE": (6 + rad, 6),
        "lS": (6, 6 - rad),
        "lW": (6 - rad, 6),
        "dN": (6, 2 + rad),
        "dE": (6 + rad, 2),
        "dS": (6, 2 - rad),
        "dW": (6 - rad, 2),
        "cN": (2, 2 + rad),
        "cE": (2 + rad, 2),
        "cS": (2, 2 - rad),
        "cW": (2 - rad, 2),
    }
    transitions = [
        "l→u-u→l",
        "d→u-u→d",
        "c→u-u→c",
        "l→d-d→l",
        "l→c-c→l",
        "d→c-c→d",
    ]
    plot_titles = [
        # "Simulation\nMUT1",
        # "Simulation\nMUT2",
        # "Phylogeny\nJanssens et al., 2020",
        # "Phylogeny\nZuntini et al., 2024",
        # "Phylogeny\nGeeta et al., 2012",
        "MUT1",
        "MUT2",
        "Janssens et al. (2020)",
        "Zuntini et al. (2024)",
        "Geeta et al. (2012)",
    ]

    # Draw circles with text in the center

    if layout == "h":
        fig_h = 6
        fig_w = 14
        fig, axs = plt.subplots(
            2,
            len(plot_order),
            figsize=(fig_w, fig_h),
            # layout="constrained",  # , gridspec_kw={"height_ratios": [3, 1]}
        )
    elif layout == "v":
        fig_h = 4 * len(plot_order)
        fig_w = 8
        fig, axs = plt.subplots(
            len(plot_order) * 2,
            2,
            figsize=(fig_h, fig_w),
            gridspec_kw={
                "height_ratios": [
                    3 if i % 2 == 0 else 1 for i in range(len(plot_order) * 2)
                ],
                "width_ratios": [2, 1],
            },
        )
    # elif layout == "c":  # only works with len(plot_order) == 1
    #     fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    else:
        raise (RuntimeError("incorrect layout argument"))

    if len(plot_order) == 1:
        axs = [axs]
    for i, row_ in enumerate(axs):
        print(row_)
        for j in range(0, len(row_)):
            ax = row_[j]
            if layout == "v" and i % 2 == 1:
                ax.axis("off")
                continue
            dataset = plot_order[j] if layout == "h" else plot_order[i // 2]
            if (layout == "h" and i == 0) or (layout == "v" and j == 0):
                ax.axis("off")
                if layout == "v":
                    ax = plt.subplot2grid(shape=(2, 2), loc=(i // 2, 0))
                    ax.axis("off")
                ax.set_xlim(0, 8)
                ax.set_ylim(0, 8)
                # ax.axis("off")

                plot_data = rate_data[rate_data["dataset"] == dataset]
                # print(plot_data)
                # fig, ax = blank_diagram()
                nodes(ax, c_proper, rad, texts, icon_imgs)
                for k, row in plot_data.iterrows():
                    # print(row)
                    at = row["fwd"][0]
                    to = row["fwd"][2]
                    r = row["mean_rate_norm"] * 4 # increase the multiplier to scale the width of the arrows
                    # rc = row["std_c"] if colourise else "black"
                    rc = "lightgrey" if row["prop_over_zero"] < 0.90 else "black"
                    sig = False if row["prop_over_zero"] < 0.90 else True
                    # rs = row["rate_norm_std"]
                    rs = False  # set to false to disable multi-arrows
                    c_val = 2 # increase to increase the curviness of the arrows
                    dc_val = 5 # increas to increase diagonal arrow curviness


                    if r > 0:
                        if at + to == "ul":
                            arc(ax, c["uS"], c["lS"], "+", r, rc, rs, sig, c_val)
                        if at + to == "lu":
                            arc(ax, c["lN"], c["uN"], "+", r, rc, rs, sig, c_val)
                        if at + to == "ld":
                            arc(ax, c["lE"], c["dE"], "-", r, rc, rs, sig, c_val)
                        if at + to == "dl":
                            arc(ax, c["dE"], c["lE"], "+", r, rc, rs, sig, c_val)
                        if at + to == "dc":
                            arc(ax, c["dS"], c["cS"], "-", r, rc, rs, sig, c_val)
                        if at + to == "cd":
                            arc(ax, c["cS"], c["dS"], "+", r, rc, rs, sig, c_val)
                        if at + to == "cu":
                            arc(ax, c["cW"], c["uW"], "-", r, rc, rs, sig, c_val)
                        if at + to == "uc":
                            arc(ax, c["uE"], c["cE"], "-", r, rc, rs, sig, c_val)

                            #### diagonals ####
                        if at == "u" and to == "d":
                            arc(ax, c["uS"], c["dW"], "+", r, rc, rs, sig, dc_val)
                        if at == "d" and to == "u":
                            arc(ax, c["dW"], c["uS"], "-", r, rc, rs, sig, dc_val)
                        if at == "c" and to == "l":
                            arc(ax, c["cE"], c["lS"], "+", r, rc, rs, sig, dc_val)
                        if at == "l" and to == "c":
                            arc(ax, c["lS"], c["cE"], "-", r, rc, rs, sig, dc_val)
                # title = plot_order[j] if layout == "h" else plot_order[i // 2]
                title = plot_titles[j] if layout == "h" else plot_titles[i // 2]
                ax.set_title(title, fontsize=9)
            if (layout == "h" and i == 1) or (layout == "v" and j == 1):

                mcmc_plot_data = phylo_sim_long[phylo_sim_long["Dataset"] == dataset]
                ml_phylo_plot_data = ML_phylo_rates_long[
                    ML_phylo_rates_long["dataname"].apply(lambda x: x in dataset)
                ]
                ml_sim_plot_data = ML_sim_rates_long[ML_sim_rates_long["Dataset"].apply(lambda x: x in dataset)]
                print(ml_sim_plot_data)
                rates = []
                ml_rates = []
                for transition in transitions:
                    rates.append(
                        mcmc_plot_data["rate_norm"][
                            mcmc_plot_data["transition"] == transition
                        ].squeeze()
                    )
                    if not ml_phylo_plot_data.empty:
                        x = ml_phylo_plot_data[
                            ml_phylo_plot_data["transition"] == transition
                        ].reset_index(drop=True)

                        if not x.empty:
                            # ml_rates.append(x.loc[0, "rate_norm_diff"])
                            ml_rates.append(x.loc[0, "rate_norm"])
                    if not ml_sim_plot_data.empty:
                        x = ml_sim_plot_data[
                            ml_sim_plot_data["transition"] == transition
                        ].reset_index(drop=True)
                        if not x.empty:
                            ml_rates.append(x.loc[0, "rate_norm"])
                    if ml_phylo_plot_data.empty and ml_sim_plot_data.empty:
                        ml_rates.append(np.nan)
                print(ml_rates)

                p = ax.violinplot(
                    rates,
                    # patch_artist=True,
                    # showmeans=True,
                    # meanline=True,
                    # showfliers=False,
                    # meanprops=dict(color="black", linestyle="-"),
                    showextrema=False,
                    showmeans=True,
                )

                if ML_data:
                    pos = list(range(1, len(transitions) + 1))
                    ax.scatter(
                        pos, ml_rates, color="black", zorder=5, s=8, facecolors="white"
                    )  # , marker="D")
                ax.axhline(0, linestyle="--", color="C1", alpha=0.5)
                ax.set_ylim(-8, 8)
                ax.set_xticks(
                    list(range(1, len(transitions) + 1)),
                    transitions,
                    fontsize=9,
                )
                # print(ax.get_position())
                xl, yl, xh, yh = np.array(ax.get_position()).ravel()
                # print(xl, yl, xh, yh)
                w = xh - xl
                h = yh - yl
                # size = 0.05
                size = (
                    0.1 if layout == "h" else 0.08
                )  # worked well for horizontal and multiple vertical
                # A = np.random.random(size=(5, 5))
                # ax.matshow(A)
                for x, xtick_pos in enumerate(list(range(1, len(transitions) + 1))):
                    xp = (
                        xl
                        + (w / (len(transitions)) * xtick_pos)
                        - (0.5 * (w / (len(transitions))))
                    )
                    # xp = xtick_pos
                    ax1 = fig.add_axes(
                        [xp - (size * 0.5), yl - size - (0.1 * size), size, size]
                    )
                    ax1.axison = False
                    ax1.imshow(trans_imgs[x])  # , transform=ax.transAxes)
                    # ax.imshow(
                    #     trans_imgs[x],
                    #     aspect="auto",
                    #     extent=(xtick_pos - size, xtick_pos + size, -0.1, 0.1 + size),
                    # )
                # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                ax.set_xticklabels([])
                ax.set_ylabel("Net normalised rate")
                if layout == "h" and j > 0:
                    ax.set_ylabel("")
                    # ax.set_yticklabels([])

    # Create legend
    labels = ["Compound","Dissected","Lobed","Unlobed"]
    axlgnd = fig.add_axes([0.88, 0.4 ,0.1,0.2]) # left_pos, bottom_pos, width, height
    icon_w, icon_h = icon_imgs[0].size
    axlgnd.set_ylim(0, len(icon_imgs) * icon_h)
    for i, img in enumerate(reversed(icon_imgs)):
        axlgnd.imshow(img, extent=(0, icon_w, i*icon_h, (i*icon_h) + icon_h))
        axlgnd.text(icon_w, (i*icon_h) + (icon_h * 0.5), labels[i], ha="left",va="center",fontsize=9)
        axlgnd.axis("off")

    # plt.tight_layout()
    # plt.savefig("arrow_violin_plot.pdf", format="pdf", dpi=1200)
    plt.savefig("arrow_violin_plot.svg", format="svg", dpi=1200)
    plt.show()


def test_rates_diff_from_zero(phylo_sim_long):

    phylo_sim_long_filt = phylo_sim_long[
        phylo_sim_long["transition"].str.contains(
            r"^[a-zA-Z]→[a-zA-Z]-[a-zA-Z]→[a-zA-Z]$"
        )
    ]
    transitions = phylo_sim_long_filt["transition"].unique()
    results = []
    for i, transition in enumerate(transitions):
        mcmc_plot_data = phylo_sim_long[phylo_sim_long["transition"] == transition]
        for name, group in mcmc_plot_data.groupby("Dataset"):
            n = len(group)
            ng0 = len(group[group["rate_norm"] > 0])

            mean = group["rate_norm"].mean()
            # t_stat, p_val = stats.ttest_1samp(group["rate_norm"], 0)
            std = group["rate_norm"].std()
            results.append(
                {
                    "dataset": name,
                    "transition": transition,
                    "mean_rate_norm": mean,
                    # "t_stat": t_stat,
                    # "p_val": p_val,
                    "std": std,
                    "lb": mean - std,
                    "ub": mean + std,
                    "n": n,
                    "prop_over_zero": ng0 / n,
                }
            )
    r = pd.DataFrame(results)
    return r


if __name__ == "__main__":

    # Normalisation method options: meanmean, zscore, zscore+2.7, zscore_global, minmax
    # Set ML_data to False to hide ML points

    # plot_rates_trace_hist(rates)
    # phylo_rates_norm = normalise_rates(phylo_rates)
    # plot_phylo_and_sim_rates(
    #     norm_method="meanmean", ML_data="ML4_equal_fam_mean_rates_all", legend=False
    # )
    # plot_phylo_and_sim_rates_with_leaf_icons(
    #     norm_method="meanmean", ML_data="ML3_mean_rates_all", legend=False
    # )
    # plot_phylo_and_sim_rates_with_leaf_icons(
    #     norm_method="minmax", ML_data="ML6_genus_mean_rates_all", legend=False
    # )
    # test_rates_diff_from_zero(
    #     norm_method="meanmean", ML_data="ML6_genus_mean_rates_all"
    # )
    # arrow_plot(norm_method="meanmean", ML_data="ML4_mean_rates_all", colourise=False)
    arrow_w_viol(norm_method="meanmean", ML_data="ML6_genus_mean_rates_all", layout="h")
    # plot_rates_diff(norm_method="meanmean", ML_data="ML4_mean_rates_all")
    # plot_rates_diff(norm_method="meanmean", ML_data="ML6_genus_mean_rates_all")
# plot_phylo_and_sim_rates_restricted(
#     norm_method="minmax", ML_data="ML_red_1_mean_rates_all"
# )
# get_phylo_stats()
# concat_posteriors()
# rates_batch_stats(rates_norm)
# plot_rates_batch(rates_norm)
# concat_posteriors()
