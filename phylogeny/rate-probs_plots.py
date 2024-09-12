import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal
import copy
import os
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


def import_phylo_and_sim_rates(plot_order):
    #### import data ####
    phylo_rates = get_rates_batch(directory="all_rates/uniform_1010000steps")
    s1 = pd.read_csv(
        "../data-processing/markov_fitter_reports/emcee/leaf_uncert_posteriors_MUT1.csv"
    )
    s2 = pd.read_csv(
        "../data-processing/markov_fitter_reports/emcee/leaf_uncert_posteriors_MUT2.csv"
    )
    s1["phylo-class"] = "MUT1_simulation"
    s2["phylo-class"] = "MUT2_simulation"
    sim_rates = pd.concat([s1, s2]).reset_index(drop=True)

    sim_rates = sim_rates.rename(columns=rates_map2)
    phylo_rates = phylo_rates.rename(columns=rates_map3)

    phylo_sim = pd.concat([sim_rates, phylo_rates]).reset_index(drop=True)
    phylo_sim = phylo_sim.rename(columns={"phylo-class": "Dataset"})
    phylo_sim_long = pd.melt(
        phylo_sim, id_vars=["Dataset"], var_name="transition", value_name="rate"
    )
    phylo_sim_long["dataname"] = phylo_sim_long["Dataset"].apply(
        lambda x: x.split("_class", 1)[0] + "_class"
    )
    # filter to only rows with dataset in the plot_order list
    phylo_sim_long = phylo_sim_long[
        phylo_sim_long["Dataset"].isin(plot_order)
    ].reset_index(drop=True)

    return phylo_sim_long


def import_phylo_ML_rates():
    ML_rates = pd.read_csv("all_rates/ML3_mean_rates_all.csv")
    ML_rates.drop(
        columns=["Lh", "Root P(0)", "Root P(1)", "Root P(2)", "Root P(3)"], inplace=True
    )
    ML_rates = ML_rates.rename(columns=rates_map3)
    ML_rates = ML_rates.rename(columns={"dataset": "dataname"})
    ML_rates_long = pd.melt(
        ML_rates, id_vars=["dataname"], var_name="transition", value_name="rate"
    )

    return ML_rates_long


def normalise_rates(phylo_sim_long, ML_phylo_rates_long, norm_method):
    #### Normalise the rates across datasets ####
    phylo_sim_long["mean_rate"] = phylo_sim_long.groupby(["Dataset", "transition"])[
        "rate"
    ].transform(
        "mean"
    )  # get mean rate for each transition per dataset
    print(phylo_sim_long)

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
        print(ML_phylo_rates_long)
        # normalise ML rates
        ML_phylo_rates_long["rate_norm"] = (
            ML_phylo_rates_long["rate"] / ML_phylo_rates_long["mean_mean"]
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
        f"sim_phylo_rates_stats_{norm_method}_mixedpriors_03-09-24.csv", index=False
    )
    print(ML_phylo_rates_long)
    print(
        ML_phylo_rates_long[ML_phylo_rates_long["dataname"] == "jan_phylo_geeta_class"]
    )

    return phylo_sim_long, ML_phylo_rates_long, summary


def plot_phylo_and_sim_rates(norm_method, show_ML: bool):
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
        "jan_phylo_nat_class_uniform0-0.1_1",
        "jan_phylo_geeta_class_uniform0-100_2",
        "solt_phylo_nat_class_uniform0-100_2",
        "solt_phylo_geeta_class_uniform0-100_2",
        # "zuntini_phylo_nat_class_uniform0-0.1_1",
        "zuntini_phylo_nat_class_10-09-24_class_uniform0-0.1_2",
        "zuntini_phylo_geeta_class_uniform0-0.1_1",
        "geeta_phylo_nat_class_uniform0-100_2",
        "geeta_phylo_geeta_class_uniform0-100_2",
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

    ML_phylo_rates_long = import_phylo_ML_rates()
    phylo_sim_long = import_phylo_and_sim_rates(plot_order)
    phylo_sim_long, ML_phylo_rates_long, summary = normalise_rates(
        phylo_sim_long, ML_phylo_rates_long, norm_method
    )

    #### plotting ####

    plt.rcParams["font.family"] = "CMU Serif"
    fig, axes = plt.subplots(
        nrows=4,
        ncols=4,
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
                if show_ML:
                    pos = list(range(1, len(plot_order) + 1))
                    ax.scatter(pos, ML_rates, color="red", zorder=5, s=7, marker="D")

            if j == 0:
                ax.set_ylabel("Normalised rate")
            if i == 3:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    # plot_order,
                    ["S1", "S2", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
                    fontsize=9,
                )
                ax.set_xlabel("Dataset")
            if j == 3 and i == 2:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    # plot_order,
                    ["S1", "S2", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
                    fontsize=9,
                )
            if (i, j) == (0, 1):
                ax.set_ylabel("Normalised rate")
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
        "P3: Soltis et al. (2011)\nphylogeny, Naturalis\nclassification",
        "P4: Soltis et al. (2011)\nphylogeny, Geeta et al.\n(2012) classification",
        "P5: Zuntini et al. (2024)\nphylogeny, Naturalis\nclassification",
        "P6: Zuntini et al. (2024)\nphylogeny, Geeta et al.\n(2012) classification",
        "P7: Geeta et al. (2012)\nphylogeny, Naturalis\nclassification",
        "P8: Geeta et al. (2012)\nphylogeny, Geeta et al.\n(2012) classification",
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
                ax.set_xlabel("Dataset")
            if j == 3 and i == 2:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    # plot_order,
                    [f"P{i}" for i in range(1, len(plot_order) + 1)],
                    fontsize=9,
                )
            if (i, j) == (0, 1):
                ax.set_ylabel("Noramlised rate")
            if j != 0 and (i, j) != (0, 1):
                ax.set_yticklabels([])
            if i != 3 and (i, j) != (2, 3):
                ax.set_xticklabels([])
            if (i, j) == (2, 3):
                ax.set_xlabel("Dataset")

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


if __name__ == "__main__":

    # Normalisation method options: meanmean, zscore, zscore+2.7, zscore_global, minmax

    # plot_rates_trace_hist(rates)
    # phylo_rates_norm = normalise_rates(phylo_rates)
    plot_phylo_and_sim_rates(norm_method="meanmean", show_ML=True)
    # get_phylo_stats()
    # concat_posteriors()
    # rates_batch_stats(rates_norm)
    # plot_rates_batch(rates_norm)
    # concat_posteriors()
