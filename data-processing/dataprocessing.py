import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats, linalg, spatial
import sys
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.gofplots import qqplot
import matplotlib.gridspec as gridspec
import seaborn.objects as so
from itertools import product
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sympy as sp
from scipy.integrate import odeint
from PIL import Image

wd = "leaves_full_21-9-23_MUT2.2_CLEAN"
# wd = "leaves_full_15-9-23_MUT1_CLEAN"
# resd = "/home/m/malone/leaf_storage/random_walks"
wd1 = "../vlab-5.0-3609-ubuntu-20_04/oofs/ext/NPHLeafModels_1.01/LeafGenerator"

sys.path.insert(1, wd1)

sns.set_palette("colorblind")
sns_palette = sns.color_palette("colorblind")
order = ["u", "l", "d", "c"]

from pdict import *

first_cats = pd.DataFrame(
    {
        "leafid": [
            "pc1_alt",
            "pd1",
            "pl1",
            "pu1",
            "pl2",
            "pu2",
            "pc3_alt",
            "pl3",
            "pu3",
            "pc4",
            "pl4",
            "pc5_alt",
            "p6af",
            "p6i",
            "p7a",
            "p7g",
            "p8ae",
            "p8i",
            "p9b",
            "p12b",
            "p12de",
            "p12f",
            "p10c7",
            "p0_121",
            "p12c7",
            "p1_35",
            "p1_82",
            "p2_78_alt",
            "p3_60",
            "p6_81",
            "p7_43",
            "p7_92",
            "p1_122_alt",
            "p1_414",
            "p2_149_alt",
            "p2_195",
            "p2_346_alt",
            "p4_121",
            "p4_510",
            "p5_122",
            "p5_249",
            "p5_909",
            "p6_163_alt",
            "p7_277",
            "p7_437",
            "p9_129",
            "p6_1155",
            "p8_1235",
        ],
        "first_cat": [
            "c",
            "d",
            "c",
            "l",
            "d",
            "u",
            "c",
            "l",
            "u",
            "c",
            "l",
            "c",
            "u",
            "u",
            "u",
            "u",
            "l",
            "l",
            "l",
            "d",
            "c",
            "c",
            "d",
            "u",
            "c",
            "d",
            "u",
            "u",
            "u",
            "d",
            "c",
            "u",
            "l",
            "d",
            "d",
            "u",
            "c",
            "u",
            "d",
            "u",
            "u",
            "d",
            "l",
            "c",
            "c",
            "u",
            "c",
            "d",
        ],
    }
)


def concatenator():
    dfs = []
    print(f"\n\nCurrent directory: {wd}\n\n")

    for leafdirectory in os.listdir(wd):
        print(f"Current = {leafdirectory}")
        leafdirectory_path = os.path.join(wd, leafdirectory)
        for walkdirectory in os.listdir(leafdirectory_path):
            walkdirectory_path = os.path.join(leafdirectory_path, walkdirectory)
            for file in os.listdir(walkdirectory_path):
                if (
                    file.endswith(".csv")
                    and "shape_report" in file
                    and "shape_report1" not in file
                ):
                    df = pd.read_csv(os.path.join(walkdirectory_path, file))
                    df.insert(0, "leafid", leafdirectory)
                    df.insert(1, "walkid", int(re.findall(r"\d+", walkdirectory)[0]))
                    dfs.append(df)

    # result = pd.concat(dfs, ignore_index=True)
    # result.to_csv("/home/m/malone/leaf_storage/random_walks/result.csv", index=False)

    return dfs


def param_concatenator():
    dfs = []

    for leafdirectory in os.listdir(wd):
        print(f"Current = {leafdirectory}")
        leafdirectory_path = os.path.join(wd, leafdirectory)
        for walkdirectory in os.listdir(leafdirectory_path):
            walkdirectory_path = os.path.join(leafdirectory_path, walkdirectory)
            for file in os.listdir(walkdirectory_path):
                if file.endswith(".csv") and "report" in file and "shape" not in file:
                    df = pd.read_csv(
                        os.path.join(walkdirectory_path, file), header=None
                    )
                    df.insert(0, "leafid", leafdirectory)
                    dfs.append(df)

    # result = pd.concat(dfs, ignore_index=True)
    # result.to_csv("/home/m/malone/leaf_storage/random_walks/result.csv", index=False)

    return dfs


def firstandlast_alt():
    dfs_by_leaf = concatenator()

    firstandlast = pd.DataFrame(columns=["leafid", "walkid", "first_cat", "last_cat"])

    for df in dfs_by_leaf:
        if not df.empty:
            print("######")
            print(df.iloc[0]["leafid"])
            first_cat = first_cats[first_cats["leafid"] == df.iloc[0]["leafid"]][
                "first_cat"
            ].values[
                0
            ]  # don't understand this code so commented out as of 28-8-23
            last_cat = df.iloc[-1]["shape"]
            row_dict = {
                "leafid": df.iloc[0]["leafid"],
                "walkid": df.iloc[0]["walkid"],
                "first_cat": first_cat,
                "last_cat": last_cat,
            }
            row_df = pd.DataFrame.from_dict(row_dict, orient="index").T
            firstandlast = pd.concat([firstandlast, row_df], ignore_index=True)

        else:
            row_dict = {
                "leafid": np.nan,
                "walkid": np.nan,
                "first_cat": np.nan,
                "last_cat": np.nan,
            }
            row_df = pd.DataFrame.from_dict(row_dict, orient="index").T
            firstandlast = pd.concat([firstandlast, row_df], ignore_index=True)
            print("#### No leaves found")

    print(firstandlast)
    firstandlast.to_csv(
        "/home/m/malone/leaf_storage/random_walks/firstandlast1.csv", index=False
    )
    return firstandlast


def frequency_alt():
    firstandlastdf = firstandlast_alt()
    leafgrouped = (
        firstandlastdf.groupby(["leafid", "first_cat", "last_cat"])
        .size()
        .reset_index(name="ntransitions")
    )
    print(leafgrouped)
    # add in blank rows
    transitions = (
        leafgrouped.groupby(["leafid", "last_cat", "first_cat"])["ntransitions"]
        .sum()
        .unstack("last_cat", fill_value=0)
        .stack()
        .reset_index(name="ntransitions")
    )
    transitions["transition"] = (
        transitions["first_cat"] + "->" + transitions["last_cat"]
    )
    transitions["nwalks"] = transitions.groupby(["leafid", "first_cat"])[
        "ntransitions"
    ].transform("sum")
    transitions["proportion"] = transitions["ntransitions"] / transitions["nwalks"]
    # transitions.to_csv("/home/m/malone/leaf_storage/random_walks/transitions.csv", index=False)
    # print(transitions)

    for category in transitions["transition"].unique():
        plt.clf()
        subset = transitions[transitions["transition"] == category]["proportion"]
        print(list(subset))
        # qqplot(subset, line="s")
        # plt.show()
        stat, p = stats.shapiro(subset)
        print(f"{category} stat = {stat} p = {p}")

    averagetransitions = (
        transitions.groupby(["first_cat", "last_cat"])["proportion"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "average_prop", "std": "std", "count": "nleaves"})
    )
    averagetransitions["sterr"] = averagetransitions["std"] / np.sqrt(
        averagetransitions["nleaves"]
    )
    averagetransitions["lb"] = averagetransitions["average_prop"] - (
        1.96 * averagetransitions["sterr"]
    )
    averagetransitions["ub"] = averagetransitions["average_prop"] + (
        1.96 * averagetransitions["sterr"]
    )
    averagetransitions["lb"] = averagetransitions["lb"].clip(lower=0)
    averagetransitions["ub"] = averagetransitions["ub"].clip(upper=1)
    averagetransitions["last_cat"] = averagetransitions["last_cat"].replace(
        "unclassified", "X"
    )
    averagetransitions["transition"] = (
        averagetransitions["first_cat"]
        + r"$\rightarrow$"
        + averagetransitions["last_cat"]
    )
    averagetransitions.to_csv(
        "/home/m/malone/leaf_storage/random_walks/average_transitions.csv", index=False
    )
    print(averagetransitions)

    colours = ["#41342C", "#158471", "#97AF25", "#CAC535"]

    sns.barplot(
        x="transition",
        y="average_prop",
        data=averagetransitions,
        yerr=[
            averagetransitions["average_prop"] - averagetransitions["lb"],
            averagetransitions["ub"] - averagetransitions["average_prop"],
        ],
        palette=colours,
    )
    plt.xlabel("Transition")
    plt.ylabel("Mean Probability")
    labels = [
        r"$\rightarrow$u",
        r"$\rightarrow$l",
        r"$\rightarrow$d",
        r"$\rightarrow$c",
    ]
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colours[::-1]]
    plt.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0, 1)
    )  # , title="Last Shape")
    plt.xticks(rotation=45)
    plt.show()

    # test for equal transition proportions for all transition types and leafids
    # DOESNT WORK ON TRANSITION DATA WITH ALL THE ZERO PROPORTIONS INCLUDED, AS PROPORTIONS NO LONGER NORAMLLY DISTRIBUTED
    # leafgrouped["transition"] = leafgrouped["first_cat"] + "->" + leafgrouped["last_cat"]
    # leafgrouped["proportion"] = leafgrouped["ntransitions"] / 10
    # print(leafgrouped)
    # model = ols("proportion ~ transition + leafid", data=transitions).fit()
    # print(sm.stats.anova_lm(model, typ=2))
    # print(model.rsquared)
    # stat, p = stats.shapiro(transitions["proportion"])
    # print(f"Shapiro-Wilk Test\nstat: {stat}, p-value: {p}")
    # qqplot(transitions["proportion"], line='s')
    # plt.show()
    # plt.clf()
    # sns.histplot(averagetransitions["average_prop"])
    # plt.show()

    # Kruksall wallis test - nonparameteric ANOVA
    grouped_input = [
        transition["proportion"] for _, transition in transitions.groupby("transition")
    ]
    result = stats.kruskal(*grouped_input)

    # Print the results
    print(result)

    # Null hypothesis rates away from unlobed are the same as rates towards lobed
    propsaway = transitions.loc[
        transitions["transition"].isin(["u->l", "u->d", "u->c"]), "proportion"
    ].tolist()
    propstowards = transitions.loc[
        transitions["transition"].isin(["l->u", "d->u", "c->u"]), "proportion"
    ].tolist()
    toandfro = {
        "Probability": propsaway + propstowards,
        "Transition": [r"u$\rightarrow$" for _ in propsaway]
        + [r"$\rightarrow$u" for _ in propstowards],
    }
    toandfrodf = pd.DataFrame(toandfro)

    # Test for normal distribution
    stat, p = stats.shapiro(propsaway)
    print(f"SHAPIRO_WILK propsaway stat = {stat} p = {p}")
    stat, p = stats.shapiro(propstowards)
    print(f"SHAPIRO_WILK propstowards stat = {stat} p = {p}")

    # One tailed - test if propstowards distribution is greater than propsaway
    result = stats.mannwhitneyu(propstowards, propsaway, alternative="greater")
    print(f"MANN_WHITNEY {result}")

    plt.clf()
    sns.violinplot(
        x="Transition",
        y="Probability",
        data=toandfrodf,
        cut=0,
        scale="count",
        palette=colours[1:3],
    )
    plt.show()

    # tukey = pairwise_tukeyhsd(transitions["proportion"], transitions["transition"])
    # print(tukey.summary())

    # groupbyleafid = firstandlastdf.groupby(["leafid"]).size().reset_index(name="ntransitions")
    # print(groupbyleafid)


def heatmap():
    mydata = np.array(
        [
            [0.753, 0.159, 0.008, 0.000],
            [0.800, 0.144, 0.056, 0.000],
            [0.800, 0.082, 0.116, 0.000],
            [0.554, 0.046, 0.208, 0.192],
        ]
    )
    geetarerun = np.array(
        [
            [0.824, 0.035, 0.051, 0.090],
            [0.778, 0.035, 0.088, 0.100],
            [0.286, 0.016, 0.666, 0.032],
            [0.706, 0.033, 0.151, 0.109],
        ]
    )

    myflat = mydata.flatten()
    Gflat = geetarerun.flatten()

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.hist(myflat)
    # ax2.hist(Gflat)
    # plt.show()

    result = stats.mannwhitneyu(myflat, Gflat)

    # Perform Kolmogorov-Smirnov test
    statistic, p_value = stats.kstest(myflat, Gflat)

    # Print the results
    print("Kolmogorov-Smirnov test statistic:", statistic)
    print("p-value:", p_value)

    exit()

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # Plot heatmaps
    ax1 = plt.subplot(gs[0])
    sns.heatmap(mydata, cmap="Blues", annot=False, ax=ax1, cbar=False)
    ax1.set_xlabel("End shape")
    ax1.set_ylabel("Start shape")
    ax1.set_xticklabels(["u", "l", "d", "c"])
    ax1.set_yticklabels(["u", "l", "d", "c"], rotation=0)

    ax2 = plt.subplot(gs[1])
    sns.heatmap(
        geetarerun,
        cmap="Blues",
        annot=False,
        ax=ax2,
        cbar=True,
        cbar_ax=fig.add_axes([0.9, 0.125, 0.025, 0.75]),
        cbar_kws={"label": "Probability"},
    )
    ax2.set_xlabel("End shape")
    ax2.set_ylabel("Start shape")
    ax2.set_xticklabels(["u", "l", "d", "c"])
    ax2.set_yticklabels(["u", "l", "d", "c"], rotation=0)

    # Add titles and adjust layout
    ax1.set_title("a. My Data")
    ax2.set_title("b. Geeta et al. (2012)")
    plt.subplots_adjust(right=0.85)
    plt.show()


# heatmap()


def firstswitch():
    order = ["u", "l", "d", "c"]
    dfs = concatenator()

    firstswitchdf = pd.DataFrame(
        columns=[
            "leafid",
            "walkid",
            "first_cat",
            "first_switch_cat",
            "nsteps_to_switch",
        ]
    )

    for df in dfs:
        if not df.empty:
            first_cat = df.iloc[0]["shape"]
            leafid = df.iloc[0]["leafid"]
            walkid = df.iloc[0]["walkid"]
            for i, value in enumerate(df["shape"]):
                if value != first_cat:
                    # row_dict = {
                    #     "leafid": df.iloc[0]["leafid"],
                    #     "walkid": df.iloc[0]["walkid"],
                    #     "first_cat": first_cat,
                    #     "first_switch_cat": value,
                    #     "nsteps_to_switch": i,
                    # }
                    row = [leafid, walkid, first_cat, value, i]
                    # print(row)
                    break
                else:
                    row = [leafid, walkid, first_cat, value, np.NAN]

            firstswitchdf.loc[len(firstswitchdf)] = row
        else:
            print("#### Encountered empty shape report")

    firstswitchdf.to_csv(
        "/home/m/malone/leaf_storage/random_walks/firstswitchdf.csv", index=False
    )

    t_sorted = firstswitchdf.sort_values(
        by=["first_switch_cat"],
        key=lambda x: x.map({v: i for i, v in enumerate(order)}),
    )
    print(t_sorted)

    sns.catplot(
        x="first_cat",
        y="nsteps_to_switch",
        hue="first_switch_cat",
        kind="bar",
        data=t_sorted.dropna(),
        order=order,
        palette="colorblind",
    )
    plt.show()


def paramspace():
    order = ["u", "l", "d", "c"]
    # paramshapess = param_concatenator()
    # paramshapes = pd.concat(paramshapess, ignore_index=True)
    paramshapes = pd.read_csv("MUT2.2_trajectories_param.csv")

    starting_leaves = pd.DataFrame(pdict.values()).transpose()

    only_valid_leaves = paramshapes[
        paramshapes.iloc[:, 3].str.contains("passed")
    ].reset_index(drop=True)
    # remove the metadata columns at the start and the leaf shape metrics columns from the classifier at the end
    only_valid_leaves = only_valid_leaves.iloc[:, 5:-6]
    only_valid_leaves.columns = range(only_valid_leaves.shape[1])
    # paramshapes_sub = only_valid_leaves.iloc[:, [0, 1, 2, 3, 36, 67]]
    # # print(paramshapes_sub.iloc[:, 3].str.contains("passed"))

    # shapess = concatenator()
    # shapes = pd.concat(shapess, ignore_index=True)
    shapes = pd.read_csv("MUT2.2_trajectories_shape.csv")

    # paramshapes_sub["shape"] = shapes["shape"]
    # paramshapes_sub.columns = [
    #     "leafid",
    #     "walkid",
    #     "step",
    #     "pass",
    #     "p1",
    #     "p2",
    #     "shape",
    # ]

    # combine starting leaves and random walk leaves into one dataframe to do pca with, then separate later
    data = pd.concat([starting_leaves, only_valid_leaves], ignore_index=True)
    data = data.replace(
        {" 'true'": 1, "true": 1, " 'false'": 0, "false": 0, " nan": np.nan}
    )
    # data = data.replace({"M_PI": np.pi}, regex=True)
    # data = data.applymap(
    #     lambda x: x.strip(" '") if isinstance(x, str) else x
    # )  # remove space and quotations from the angle parameters
    # data = data.applymap(
    #     lambda x: eval(x.replace("M_PI", str(np.pi))) if isinstance(x, str) else x
    # )
    data = data.loc[:, ~data.apply(lambda col: any("M_PI" in str(x) for x in col))]
    data = data.loc[:, ~data.apply(lambda col: any("#define" in str(x) for x in col))]
    data = data.dropna(axis=1, how="any")
    print(f"No. starting leaves {len(starting_leaves)}")
    # shapes_all = list(first_cats["first_cat"].values) + list(shapes["shape"])
    # data["shape"] = shapes_all

    # for shape in order:
    #     data_sub = data[data["shape"] == shape]
    #     data_sub = data_sub.iloc[:, :-1]
    #     hull = spatial.ConvexHull(data_sub, qhull_options="QJ")
    #     print(hull.volume)

    # data.to_csv("/home/m/malone/leaf_storage/random_walks/result.csv", index=False)
    scaled_data = StandardScaler().fit_transform(data)

    pca_params = PCA(n_components=2)
    princip_params = pca_params.fit_transform(scaled_data)
    explained_variance_ratio = pca_params.explained_variance_ratio_
    princip_df = pd.DataFrame(data=princip_params, columns=["pc1", "pc2"])
    princip_df_starting_leaves = princip_df.iloc[: len(starting_leaves)]
    princip_df_starting_leaves["shape"] = list(first_cats["first_cat"].values)
    princip_df_result = princip_df.iloc[len(starting_leaves) :].reset_index(drop=True)
    princip_df_result["shape"] = shapes["shape"]
    print(len(shapes), len(princip_df_result))
    # Generate convex hulls
    hulls = []
    for shape in order:
        PCA_sub = princip_df_result[princip_df_result["shape"] == shape]
        PCA_sub = PCA_sub[["pc1", "pc2"]]
        hull = spatial.ConvexHull(PCA_sub)
        hulls.append(hull)

    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]
    plt.rcParams["font.family"] = "CMU Serif"
    fig, axs = plt.subplots(
        2, 2, figsize=(7, 8), sharex="all", sharey="all", layout="constrained"
    )
    counter = -1
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            counter += 1
            shape = order[counter]
            plot_data = princip_df_result[
                princip_df_result["shape"] == shape
            ].reset_index(drop=True)
            unlobed_data = princip_df_result[
                princip_df_result["shape"] == "u"
            ].reset_index(drop=True)
            # plot walk_data
            ax.scatter(
                plot_data["pc1"],
                plot_data["pc2"],
                s=10,
                # color="black",
                color=sns_palette[counter],
                alpha=0.1,
            )
            # plot initial points
            ax.scatter(
                x=princip_df_starting_leaves["pc1"],
                y=princip_df_starting_leaves["pc2"],
                c=princip_df_starting_leaves["shape"].map(
                    {
                        "u": sns_palette[0],
                        "l": sns_palette[1],
                        "d": sns_palette[2],
                        "c": sns_palette[3],
                    }
                ),
                edgecolor="white",
                linewidth=0.8,
            )
            # plot convex hulls for walk data
            hull = hulls[counter]
            hull_unlobed = hulls[0]
            for simplex in hull_unlobed.simplices:
                ax.plot(
                    unlobed_data["pc1"][simplex],
                    unlobed_data["pc2"][simplex],
                    color="grey",
                )
            for simplex in hull.simplices:
                # print(simplex)
                # print(princip_df_result["pc1"][simplex])
                ax.plot(
                    plot_data["pc1"][simplex],
                    plot_data["pc2"][simplex],
                    color="red",
                )
            ax.set_title(
                f"{order_full[counter]} h-vol:{round(hull.volume, 2)}", fontsize=14
            )
            # if j > 0:
            # ax.set_yticklabels([])
            if j == 0:
                ax.set_ylabel(
                    f"PC2 ({(explained_variance_ratio[1] * 100):.2f}%)", fontsize=14
                )
            # if i == 0:
            # ax.set_xticklabels([])
            if i == 1:
                ax.set_xlabel(
                    f"PC1 ({(explained_variance_ratio[0] * 100):.2f}%)", fontsize=14
                )
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color, markersize=10
        )
        for color in sns_palette
    ]
    legend = fig.legend(
        legend_handles,
        ["unlobed", "lobed", "dissected", "compound"],
        loc="outside lower center",
        title="Shape",
        fontsize=13,
        ncol=4,
    )
    title = legend.get_title()
    title.set_fontsize(14)

    plt.show()


def overallfreq():
    dfs = concatenator()

    shapecounts = pd.DataFrame(columns=["leafid", "walkid", "shape", "count"])

    dfscombinedwalkids = []
    for df in dfs:
        for wid in df["walkids"]:
            combinedwalkids = df.groupby("walkid")["wid"].sum()

    for df in dfs:
        counts = df.groupby(["shape"]).size().reset_index(name="count")
        for i, row in counts.iterrows():
            row_dict = {
                "leafid": df.iloc[0]["leafid"],
                "walkid": df.iloc[0]["walkid"],
                "shape": row["shape"],
                "count": row["count"],
            }

            shapecounts = shapecounts.append(row_dict, ignore_index=True)

    print(shapecounts)
    shapecounts.to_csv(
        "/home/m/malone/leaf_storage/random_walks/shapecounts.csv", index=False
    )


def stack_plot():
    dfs = concatenator()

    for walk in dfs:
        walk["step"] = walk.index.values

    concat = pd.concat(dfs, ignore_index=True)
    concat = pd.merge(concat, first_cats[["leafid", "first_cat"]], on="leafid")
    mapping = {"u": 0, "l": 1, "d": 2, "c": 3}
    concat["dummy"] = concat["shape"].map(mapping)

    grouped = (
        concat.groupby(["first_cat", "step", "shape"]).size().reset_index(name="count")
    )

    lfirst = grouped[grouped["first_cat"] == "l"]
    print(lfirst)

    sns.set_theme()
    # plots = []
    fig, axs = plt.subplots(2, 2)
    # fig.text(0.5, 0.01, 'Step', ha='center')
    # fig.text(0.01, 0.5, "Count", va='center', rotation='vertical')
    for i, cat in enumerate(["c", "d", "l", "u"]):
        lfirst = grouped[grouped["first_cat"] == cat]

        reference = pd.DataFrame(
            list(product(grouped["step"].unique(), ["u", "l", "d", "c"])),
            columns=["step", "shape"],
        )

        lfirst_filled = reference.merge(lfirst, how="left").fillna({"count": 0})

        u = lfirst_filled.loc[lfirst_filled["shape"] == "u", "count"]
        l = lfirst_filled.loc[lfirst_filled["shape"] == "l", "count"]
        d = lfirst_filled.loc[lfirst_filled["shape"] == "d", "count"]
        c = lfirst_filled.loc[lfirst_filled["shape"] == "c", "count"]
        print(u)
        print(l)
        print(d)
        print(c)

        row = i // 2
        col = i % 2
        ax = axs[row, col]

        colours = ["#41342C", "#158471", "#97AF25", "#CAC535"]
        labels = ["c", "d", "l", "u"]
        ax.stackplot(
            range(0, 120),
            c.values,
            d.values,
            l.values,
            u.values,
            colors=colours,
            labels=labels,
            alpha=0.7,
        )
        # plt.xlabel("Step")
        # plt.ylabel("Count")
        if cat == "u":
            title = ax.set_title("d. Starting Shape: Unlobed")
        if cat == "l":
            title = ax.set_title("c. Starting Shape: Lobed")
        if cat == "d":
            title = ax.set_title("b. Starting Shape: Dissected")
        if cat == "c":
            title = ax.set_title("a. Starting Shape: Compound")
        title.set_fontsize(15)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # plt.subplots_adjust(right=0.7)
        # plots.append(fig)
        ax.set_xlabel("No. parameter changes")
        ax.set_ylabel("No. leaves per category")

    plt.tight_layout()
    plt.show()


def prop_curves():
    dfs = concatenator()

    for walk in dfs:
        walk["step"] = walk.index.values

    concat = pd.concat(dfs, ignore_index=True)
    concat = pd.merge(concat, first_cats[["leafid", "first_cat"]], on="leafid")
    mapping = {"u": 0, "l": 1, "d": 2, "c": 3}
    concat["dummy"] = concat["shape"].map(mapping)
    print(concat)

    # no. each shape type for each step of each first_cat
    grouped_by_first_cat = (
        concat.groupby(["first_cat", "step", "shape"])
        .size()
        .reset_index(name="total_shape_firstcat")
    )
    # total no. leaves per step for every first_cat
    grouped_by_first_cat_total = (
        grouped_by_first_cat.groupby(["first_cat", "step"])
        .agg(total_firstcat=("total_shape_firstcat", "sum"))
        .reset_index()
    )

    grouped_by_first_cat = grouped_by_first_cat.merge(
        grouped_by_first_cat_total, on=["first_cat", "step"]
    )
    grouped_by_first_cat["proportion"] = (
        grouped_by_first_cat["total_shape_firstcat"]
        / grouped_by_first_cat["total_firstcat"]
    )
    print(grouped_by_first_cat)

    # get total no. leaves per shape, per step for each leaf
    grouped_by_leaf = (
        concat.groupby(["leafid", "first_cat", "step", "shape"])
        .size()
        .reset_index(name="total_shape_leafid")
    )

    # get total no. leaves per step per leaf
    grouped_by_leaf_total = (
        grouped_by_leaf.groupby(["leafid", "step"])
        .agg(total_leafid=("total_shape_leafid", "sum"))
        .reset_index()
    )

    grouped_by_leaf = grouped_by_leaf.merge(
        grouped_by_leaf_total, on=["leafid", "step"]
    )

    grouped_by_leaf = grouped_by_leaf.merge(
        grouped_by_first_cat_total, on=["first_cat", "step"]
    )

    # get proportion of each shape per step per leaf
    grouped_by_leaf["proportion"] = (
        grouped_by_leaf["total_shape_leafid"] / grouped_by_leaf["total_leafid"]
    )

    # grouped_by_leaf["first_cat"] = grouped_by_leaf["first_cat"].map(
    #     {"u": "Unlobed", "l": "Lobed", "d": "Dissected", "c": "Compound"}
    # )
    # grouped_by_leaf["shape"] = grouped_by_leaf["shape"].map(
    #     {"u": "Unlobed", "l": "Lobed", "d": "Dissected", "c": "Compound"}
    # )

    print(grouped_by_leaf)

    # plt.rcParams["font.family"] = "CMU Serif"
    g = (
        sns.relplot(
            data=grouped_by_leaf,
            x="step",
            y="proportion",
            col="first_cat",
            hue="shape",
            kind="line",
            col_wrap=2,
            hue_order=order,
            col_order=order,
            errorbar="ci",  # 95% confidence interval calculated with bootstrapping see https://seaborn.pydata.org/tutorial/error_bars.html
        )
        .set_axis_labels("Step", "Mean Prop.")
        .set(xlim=(0, 60))
    )

    custom_titles = [
        "Initial shape: Unlobed",
        "Initial shape: Lobed",
        "Initial shape: Dissected",
        "Initial shape: Compound",
    ]  # Add your titles here
    for ax, title in zip(g.axes.flat, custom_titles):
        ax.set_title(title)

    plt.show()


def curves_phylogeny():

    tee = sp.symbols("t")

    t_vals = np.linspace(0, 1, 500)
    # QMCMC are average rates from BayesTraitsV4_ 4 - 12 - 23 _mcmc
    QMCMC = np.array(
        [
            [-7.554229919, 4.90788243, 0.648523446, 1.997824043],
            [77.126084851, -166.160628819, 30.208611133, 58.825932835],
            [45.248729997, 38.483670211, -119.743221002, 36.010820794],
            [16.253673911, 16.234659459, 11.907356073, -44.395689443],
        ]
    )
    # QML are average rates from /multistatererun/multistate_rerun _d . csv
    QML = np.array(
        [
            [-7.33971033333333, 5.66013596969697, 0.265320606060606, 1.41425375757576],
            [99.9958611414141, -162.866899474747, 0.259923232323232, 62.6111151010101],
            [
                3.19065408080808,
                0.991705666666667,
                -4.24180003030303,
                0.0594402828282828,
            ],
            [19.7934216262626, 11.5333476161616, 5.61182216161616, -36.9385914040404],
        ]
    )

    results = []

    for t_val in t_vals:
        result = np.array([])
        QML_t = QML * t_val
        result = linalg.expm(QML_t)
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

    sns.relplot(
        data=plot_data,
        x="t",
        y="P",
        col="first_cat",
        hue="shape",
        kind="line",
        col_wrap=2,
        col_order=order,
        hue_order=order,
        facet_kws={
            "sharey": False
        },  # Uncomment this if you want each row to have its own y-axis scale
    )
    plt.show()


def curves_CTMC_MLEsimfit():

    # Get timeseries data
    dfs = concatenator()
    for walk in dfs:
        walk["step"] = walk.index.values
    concat = pd.concat(dfs, ignore_index=True)
    concat = pd.merge(concat, first_cats[["leafid", "first_cat"]], on="leafid")
    timeseries = (
        concat.groupby(["first_cat", "step", "shape"])
        .size()
        .reset_index(name="total_shape_firstcat")
    )
    timeseries_total = (
        timeseries.groupby(["first_cat", "step"])
        .agg(total_firstcat=("total_shape_firstcat", "sum"))
        .reset_index()
    )
    timeseries = timeseries.merge(timeseries_total, on=["first_cat", "step"])
    timeseries["proportion"] = (
        timeseries["total_shape_firstcat"] / timeseries["total_firstcat"]
    )
    # add initial state to the timeseries
    timeseries["step"] = timeseries["step"] + 1
    for i in order:
        timeseries.loc[-1] = {
            "first_cat": i,
            "step": 0,
            "shape": i,
            "total_shape_firstcat": np.nan,
            "total_firstcat": np.nan,
            "proportion": 1,
        }
        timeseries.index = timeseries.index + 1
        timeseries = timeseries.sort_index()
    print(timeseries)

    # produce curves from MLE inferred rates
    mle = pd.read_csv("MUT2.2_MLE_rates.csv")
    t_vals = np.linspace(0, 120, 120)
    curves = []
    Q = np.array(mle["rate"].values).reshape(4, 4)
    QL = np.array(mle["LB"].values).reshape(4, 4)
    QU = np.array(mle["UB"].values).reshape(4, 4)
    for t in t_vals:
        Pt = linalg.expm(Q * t)
        PLt = linalg.expm(QL * t)
        PUt = linalg.expm(QU * t)
        curves.append([Pt, PLt, PUt])

    plot_data = {"t": [], "first_cat": [], "shape": [], "P": [], "lb": [], "ub": []}

    for i, list in enumerate(curves):
        for j, matrix in enumerate(list):
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    if j == 0:
                        plot_data["t"].append(t_vals[i])
                        plot_data["first_cat"].append(row)
                        plot_data["shape"].append(column)
                        plot_data["P"].append(matrix[row, column])
                    elif j == 1:
                        plot_data["lb"].append(matrix[row, column])
                    elif j == 2:
                        plot_data["ub"].append(matrix[row, column])

    plot_data = pd.DataFrame(plot_data)
    mapping = {0: "u", 1: "l", 2: "d", 3: "c"}
    plot_data["first_cat"].replace(mapping, inplace=True)
    plot_data["shape"].replace(mapping, inplace=True)
    # replace any upper bound value greater than 1 with 1 (because they are probabilities)
    plot_data["ub"] = plot_data["ub"].clip(upper=1)
    print(plot_data)
    plot_data_long = pd.melt(
        plot_data,
        id_vars=["t", "first_cat", "shape"],
        value_vars=["P", "lb", "ub"],
        var_name="variable",
        value_name="value",
    )
    print(plot_data_long)

    cmap = matplotlib.colors.ListedColormap(sns.color_palette("colorblind"))

    # Create subplots
    fig, axs = plt.subplots(
        nrows=len(order) // 2, ncols=2, figsize=(12, 8), layout="constrained"
    )

    # Flatten axs for easy iteration
    axs = axs.flatten()

    # Plot for each category in first_cat
    lines = []
    for i, cat in enumerate(order):
        ax = axs[i]
        cat_data = plot_data[plot_data["first_cat"] == cat]
        timeseries_cat_data = timeseries[timeseries["first_cat"] == cat]
        for j, shape in enumerate(order):
            shape_data = cat_data[cat_data["shape"] == shape]
            timeseries_shape_data = timeseries_cat_data[
                timeseries_cat_data["shape"] == shape
            ]
            (line1,) = ax.plot(
                timeseries_shape_data["step"],
                timeseries_shape_data["proportion"],
                label=shape,
                c=sns_palette[j],
                linestyle="--",
            )
            (line,) = ax.plot(
                shape_data["t"],
                shape_data["P"],
                label=shape,
                c=sns_palette[j],
                linestyle="-",  # cmap=cmap
            )
            ax.fill_between(
                shape_data["t"],
                shape_data["lb"],
                shape_data["ub"],
                alpha=0.3,
                # cmap=cmap,
            )
            lines.append(line)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_title(f"Initial shape: {cat}")
        ax.set_xlabel("t")
        ax.set_ylabel("P")

    fig.legend(lines, order, loc="outside right", title="shape")
    plt.show()


def curves_CTMC_mcmcsimfit():
    # Get mcmc data
    mcmc_data = pd.read_csv(
        "markov_fitter_reports/emcee/24chains_25000steps_15000burnin/MUT2.2_emcee_run_log_24-04-24.csv"
    )
    # mcmc_data = pd.read_csv("emcee_run_log.csv")
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
    mcmc_data = mcmc_data.rename(columns=name_map)
    # infer stasis rates
    mcmc_data.insert(0, "q00", -mcmc_data["q01"] - mcmc_data["q02"] - mcmc_data["q03"])
    mcmc_data.insert(5, "q11", -mcmc_data["q10"] - mcmc_data["q12"] - mcmc_data["q13"])
    mcmc_data.insert(10, "q22", -mcmc_data["q20"] - mcmc_data["q21"] - mcmc_data["q23"])
    mcmc_data.insert(15, "q33", -mcmc_data["q30"] - mcmc_data["q31"] - mcmc_data["q32"])
    print(mcmc_data)

    # Calculate means
    mcmc_summary = mcmc_data.mean().reset_index()
    mcmc_summary.columns = ["transition", "mean_rate"]
    # Calculate confidence intervals
    confidence_intervals = {}
    for col in mcmc_data.columns:
        data = mcmc_data[col].dropna()
        confidence_intervals[col] = (
            np.mean(data) - (1.96 * stats.sem(data)),
            np.mean(data) + (1.96 * stats.sem(data)),
        )

        # stats.norm.interval(
        #    0.95, loc=np.mean(data), scale=np.std(data) / np.sqrt(len(data))
        # )
    mcmc_summary["lb"] = [i[0] for i in confidence_intervals.values()]
    mcmc_summary["ub"] = [i[1] for i in confidence_intervals.values()]
    print(mcmc_summary)

    # Get timeseries data
    dfs = concatenator()
    for walk in dfs:
        walk["step"] = walk.index.values
    concat = pd.concat(dfs, ignore_index=True)
    concat = pd.merge(concat, first_cats[["leafid", "first_cat"]], on="leafid")
    timeseries = (
        concat.groupby(["first_cat", "step", "shape"])
        .size()
        .reset_index(name="total_shape_firstcat")
    )
    timeseries_total = (
        timeseries.groupby(["first_cat", "step"])
        .agg(total_firstcat=("total_shape_firstcat", "sum"))
        .reset_index()
    )
    timeseries = timeseries.merge(timeseries_total, on=["first_cat", "step"])
    timeseries["proportion"] = (
        timeseries["total_shape_firstcat"] / timeseries["total_firstcat"]
    )
    # add initial state to the timeseries
    timeseries["step"] = timeseries["step"] + 1
    for i in order:
        timeseries.loc[-1] = {
            "first_cat": i,
            "step": 0,
            "shape": i,
            "total_shape_firstcat": np.nan,
            "total_firstcat": np.nan,
            "proportion": 1,
        }
        timeseries.index = timeseries.index + 1
        timeseries = timeseries.sort_index()
    print(timeseries)

    # produce curves from mcmc inferred rates
    t_vals = np.linspace(0, 120, 120)
    curves = []
    Q = np.array(mcmc_summary["mean_rate"].values).reshape(4, 4)
    QL = np.array(mcmc_summary["lb"].values).reshape(4, 4)
    QU = np.array(mcmc_summary["ub"].values).reshape(4, 4)
    for t in t_vals:
        Pt = linalg.expm(Q * t)
        PLt = linalg.expm(QL * t)
        PUt = linalg.expm(QU * t)
        curves.append([Pt, PLt, PUt])

    plot_data = {"t": [], "first_cat": [], "shape": [], "P": [], "lb": [], "ub": []}

    for i, list in enumerate(curves):
        for j, matrix in enumerate(list):
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    if j == 0:
                        plot_data["t"].append(t_vals[i])
                        plot_data["first_cat"].append(row)
                        plot_data["shape"].append(column)
                        plot_data["P"].append(matrix[row, column])
                    elif j == 1:
                        plot_data["lb"].append(matrix[row, column])
                    elif j == 2:
                        plot_data["ub"].append(matrix[row, column])

    plot_data = pd.DataFrame(plot_data)
    mapping = {0: "u", 1: "l", 2: "d", 3: "c"}
    plot_data["first_cat"].replace(mapping, inplace=True)
    plot_data["shape"].replace(mapping, inplace=True)
    # replace any upper bound value greater than 1 with 1 (because they are probabilities)
    plot_data["ub"] = plot_data["ub"].clip(upper=1)
    print(plot_data)
    plot_data_long = pd.melt(
        plot_data,
        id_vars=["t", "first_cat", "shape"],
        value_vars=["P", "lb", "ub"],
        var_name="variable",
        value_name="value",
    )
    print(plot_data_long)

    cmap = matplotlib.colors.ListedColormap(sns.color_palette("colorblind"))

    # Create subplots
    fig, axs = plt.subplots(
        nrows=len(order) // 2, ncols=2, figsize=(12, 8), layout="constrained"
    )

    # Flatten axs for easy iteration
    axs = axs.flatten()

    # Plot for each category in first_cat
    lines = []
    for i, cat in enumerate(order):
        ax = axs[i]
        cat_data = plot_data[plot_data["first_cat"] == cat]
        timeseries_cat_data = timeseries[timeseries["first_cat"] == cat]
        for j, shape in enumerate(order):
            shape_data = cat_data[cat_data["shape"] == shape]
            timeseries_shape_data = timeseries_cat_data[
                timeseries_cat_data["shape"] == shape
            ]
            (line1,) = ax.plot(
                timeseries_shape_data["step"],
                timeseries_shape_data["proportion"],
                label=shape,
                c=sns_palette[j],
                linestyle="--",
            )
            (line,) = ax.plot(
                shape_data["t"],
                shape_data["P"],
                label=shape,
                c=sns_palette[j],
                linestyle="-",  # cmap=cmap
            )
            ax.fill_between(
                shape_data["t"],
                shape_data["lb"],
                shape_data["ub"],
                alpha=0.3,
                # cmap=cmap,
            )
            lines.append(line)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_title(f"Initial shape: {cat}")
        ax.set_xlabel("t")
        ax.set_ylabel("P")

    fig.legend(lines, order, loc="outside right", title="shape")
    plt.show()


def plot_data_from_probcurves(curves, t_vals):
    plot_data = {"t": [], "first_cat": [], "shape": [], "P": [], "lb": [], "ub": []}

    for i, list in enumerate(curves):
        for j, matrix in enumerate(list):
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    if j == 0:
                        plot_data["t"].append(t_vals[i])
                        plot_data["first_cat"].append(row)
                        plot_data["shape"].append(column)
                        plot_data["P"].append(matrix[row, column])
                    elif j == 1:
                        plot_data["lb"].append(matrix[row, column])
                    elif j == 2:
                        plot_data["ub"].append(matrix[row, column])

    plot_data = pd.DataFrame(plot_data)
    mapping = {0: "u", 1: "l", 2: "d", 3: "c"}
    plot_data["first_cat"].replace(mapping, inplace=True)
    plot_data["shape"].replace(mapping, inplace=True)
    # replace any upper bound value greater than 1 with 1 (because they are probabilities)
    plot_data["ub"] = plot_data["ub"].clip(upper=1)
    return plot_data


def plot_sim_and_phylogeny_curves():

    lb = 5
    ub = 95

    #### Get phylo-rates ####
    phylo_dir = "../phylogeny/rates/uniform_1010000steps"
    # phylo = "jan_phylo_nat_class"  # the phylo-class to use for the curves
    # phylo = "jan_phylo_nat_class_21-01-24_95_each"
    # phylo = "geeta_phylo_geeta_class_23-04-24_shuff"
    # phylo = "zuntini_phylo_nat_class"
    phylo = "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1"
    # phylo = "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1"
    phylo_xlim = 200  # 0.1
    phylo_rates_list = []

    for filename in os.listdir(phylo_dir):
        if filename.endswith(".csv"):
            path = os.path.join(phylo_dir, filename)
            df = pd.read_csv(path)
            df["phylo-class"] = filename[:-4]
            phylo_rates_list.append(df)

    phylo_rates = pd.concat(phylo_rates_list, ignore_index=True)
    # Choose phylogeny for curves
    phylo_rates = phylo_rates[phylo_rates["phylo-class"] == phylo]
    phylo_rates.drop(columns="phylo-class", inplace=True)
    phylo_rates.reset_index(drop=True, inplace=True)
    # Insert stasis rates
    phylo_rates.insert(
        0,
        "q00",
        -phylo_rates["q01"] - phylo_rates["q02"] - phylo_rates["q03"],
    )
    phylo_rates.insert(
        5,
        "q11",
        -phylo_rates["q10"] - phylo_rates["q12"] - phylo_rates["q13"],
    )
    phylo_rates.insert(
        10,
        "q22",
        -phylo_rates["q20"] - phylo_rates["q21"] - phylo_rates["q23"],
    )
    phylo_rates.insert(
        15,
        "q33",
        -phylo_rates["q30"] - phylo_rates["q31"] - phylo_rates["q32"],
    )

    # Calculate means
    phylo_summary = phylo_rates.mean().reset_index()
    phylo_summary.columns = ["transition", "mean_rate"]
    # Calculate confidence intervals
    confidence_intervals = {}
    for col in phylo_rates.columns:
        data = phylo_rates[col].dropna()
        confidence_intervals[col] = (
            # np.mean(data) - (1.96 * stats.sem(data)), # sterr for mean
            # np.mean(data) + (1.96 * stats.sem(data)),
            # np.mean(data) - np.std(data),
            # np.mean(data) + np.std(data),
            np.percentile(data, lb),  # calculate credible interval from posterior
            np.percentile(data, ub),
        )

    phylo_summary["lb"] = [i[0] for i in confidence_intervals.values()]
    phylo_summary["ub"] = [i[1] for i in confidence_intervals.values()]

    #### Get sim-rates ####

    sim_rates = (
        pd.read_csv("markov_fitter_reports/emcee/leaf_uncert_posteriors_MUT2.csv") * 0.1
    )
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

    # for sim_rates in avg_list:
    sim_rates = sim_rates.rename(columns=name_map)
    sim_rates.insert(0, "q00", -sim_rates["q01"] - sim_rates["q02"] - sim_rates["q03"])
    sim_rates.insert(5, "q11", -sim_rates["q10"] - sim_rates["q12"] - sim_rates["q13"])
    sim_rates.insert(
        10,
        "q22",
        -sim_rates["q20"] - sim_rates["q21"] - sim_rates["q23"],
    )
    sim_rates.insert(
        15,
        "q33",
        -sim_rates["q30"] - sim_rates["q31"] - sim_rates["q32"],
    )

    # Calculate means
    sim_summary = sim_rates.mean().reset_index()
    sim_summary.columns = ["transition", "mean_rate"]
    # Calculate confidence intervals
    confidence_intervals = {}
    for col in sim_rates.columns:
        data = sim_rates[col].dropna()
        confidence_intervals[col] = (
            # np.mean(data) - (1.96 * stats.sem(data)),
            # np.mean(data) + (1.96 * stats.sem(data)),
            # np.mean(data) - np.std(data),
            # np.mean(data) + np.std(data),
            # np.mean(data) - (200 * np.var(data, ddof=1)),
            # np.mean(data) + (200 * np.var(data, ddof=1)),
            np.percentile(data, lb),
            np.percentile(data, ub),
        )
    sim_summary["lb"] = [i[0] for i in confidence_intervals.values()]
    sim_summary["ub"] = [i[1] for i in confidence_intervals.values()]
    # sim_summaries.append(sim_summary)
    print(sim_summary)

    #### Get sim timeseries data ####
    concat = pd.read_csv("MUT2.2_trajectories_shape.csv")
    print(concat)

    # total of each shape per step for each leafid
    timeseries = (
        concat.groupby(["leafid", "first_cat", "step", "shape"])
        .size()
        .reset_index(name="total_shape_firstcat")
    )
    print(timeseries)

    # no. active walks per step for each leafid
    timeseries_total = (
        timeseries.groupby(["leafid", "first_cat", "step"])
        .agg(total_firstcat=("total_shape_firstcat", "sum"))
        .reset_index()
    )
    print(timeseries_total)

    # proportion of active walks in each shape category for each leafid
    timeseries = timeseries.merge(timeseries_total, on=["leafid", "first_cat", "step"])
    timeseries["proportion"] = (
        timeseries["total_shape_firstcat"] / timeseries["total_firstcat"]
    )
    print(timeseries)

    # mean proportion of active walks in each shape category for all leaves in each first_cat
    timeseries = (
        timeseries.groupby(["first_cat", "step", "shape"])
        .agg(mean_prop=("proportion", "mean"), sterr=("proportion", "sem"))
        .reset_index()
    )
    timeseries["lb"] = timeseries["mean_prop"] - 1.96 * timeseries["sterr"]
    timeseries["ub"] = timeseries["mean_prop"] + 1.96 * timeseries["sterr"]

    # add initial state to the timeseries
    timeseries["step"] = timeseries["step"] + 1
    for i in order:
        for j in order:
            if i == j:
                timeseries.loc[-1] = {
                    "first_cat": i,
                    "step": 0,
                    "shape": i,
                    # "total_shape_firstcat": np.nan,
                    # "total_firstcat": np.nan,
                    # "proportion": 1,
                    "mean_prop": 1,
                    "sterr": 0,
                    "lb": 1,
                    "ub": 1,
                }
            else:
                timeseries.loc[-1] = {
                    "first_cat": i,
                    "step": 0,
                    "shape": j,
                    "mean_prop": 0,
                    "sterr": 0,
                    "lb": 0,
                    "ub": 0,
                }
            timeseries.index = timeseries.index + 1
            timeseries = timeseries.sort_index()
    print(timeseries)

    # print(phylo_rates)
    # print(phylo_summary)
    # print(sim_rates)
    # print(sim_summary)
    # print(timeseries)

    # produce phylo-curves
    t_vals = np.linspace(0, phylo_xlim, 120)

    phylo_curves = []
    Q = np.array(phylo_summary["mean_rate"].values).reshape(4, 4)
    QL = np.array(phylo_summary["lb"].values).reshape(4, 4)
    QU = np.array(phylo_summary["ub"].values).reshape(4, 4)
    for t in t_vals:
        Pt = linalg.expm(Q * t)
        PLt = linalg.expm(QL * t)
        PUt = linalg.expm(QU * t)
        phylo_curves.append([Pt, PLt, PUt])

    phylo_plot = pd.DataFrame(plot_data_from_probcurves(phylo_curves, t_vals))

    # produce sim-curves

    t_vals = np.linspace(0, 120, 120)
    sim_curves = []
    Q = np.array(sim_summary["mean_rate"].values).reshape(4, 4)
    QL = np.array(sim_summary["lb"].values).reshape(4, 4)
    QU = np.array(sim_summary["ub"].values).reshape(4, 4)
    for t in t_vals:
        Pt = linalg.expm(Q * t)
        PLt = linalg.expm(QL * t)
        PUt = linalg.expm(QU * t)
        sim_curves.append([Pt, PLt, PUt])

    sim_plot = pd.DataFrame(plot_data_from_probcurves(sim_curves, t_vals))

    print(phylo_plot)
    print(sim_plot)

    # Create subplots
    plt.rcParams["font.family"] = "CMU Serif"
    fig, axs = plt.subplots(
        nrows=5,
        ncols=5,
        figsize=(9, 9),
        # sharey=True,
        gridspec_kw={"height_ratios": [3, 3, 3, 1, 3]},
    )

    lines = []
    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]

    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            idx = j - 1

            if i < 1 or j < 1:
                ax.axis("off")
                continue
            else:
                cat = order[idx]
                ax.set_ylim(0, 1)
                if idx != 0:
                    ax.set_yticklabels([])
                if i == 1:  # timeseries data on the left
                    cat_data = timeseries[timeseries["first_cat"] == cat]
                    cat_data = cat_data.rename(columns={"mean_prop": "P", "step": "t"})
                    ax.set_title(order_full[idx])
                    ax.set_xlim(0, 60)
                    ax.set_xticks(np.arange(0, 61, 20))
                    ax.set_xlabel("Step")
                    if idx == 0:
                        # ax.set_title("MUT2 Data")
                        ax.set_ylabel("Mean Prop.")
                        # ax.annotate(  # This adds the row labels
                        #     f"Initial shape\n{order_full[j]}",
                        #     xy=(0.5, 1),
                        #     xytext=(-ax.yaxis.labelpad - 5, 0),
                        #     xycoords=ax.yaxis.label,
                        #     textcoords="offset points",
                        #     size="large",
                        #     ha="right",
                        #     va="center",
                        # )

                if i == 2:  # simulation ctmc in centre column
                    cat_data = sim_plot[sim_plot["first_cat"] == cat]
                    # ax.set_title(order_full[j])
                    ax.set_xlabel("Step")
                    ax.set_xlim(0, 60)
                    ax.set_xticks(np.arange(0, 61, 20))
                    if idx == 0:
                        ax.set_ylabel("P")

                if i == 3:
                    ax.axis("off")

                if i == 4:  # phylogeny data on the right
                    cat_data = phylo_plot[phylo_plot["first_cat"] == cat]
                    # ax.set_title(order_full[i])
                    ax.set_xlim(0, phylo_xlim)
                    ax.set_xlabel("Branch length (Myr)")
                    if idx == 0:
                        ax.set_ylabel("P")
                    #     ax.set_title(
                    #         "Zuntini et al. (2024)\nphylogeny, Naturalis\nclassification CTMC",
                    #         # "jan_phylo\n_nat_class_21\n-01-24_95_each"
                    #         # phylo,
                    #     )

                if i != 3:
                    for s, shape in enumerate(order):
                        shape_data = cat_data[cat_data["shape"] == shape]
                        (line,) = ax.plot(
                            shape_data["t"],
                            shape_data["P"],
                            label=shape,
                            c=sns_palette[s],
                            linestyle="-",  # cmap=cmap
                        )
                        ax.fill_between(
                            shape_data["t"],
                            shape_data["lb"],
                            shape_data["ub"],
                            alpha=0.2,
                            # cmap=cmap,
                        )
                        lines.append(line)
                    # if i < 3:
                    #     ax.set_xticklabels([])
                    # if i == 3:
                    #     if j == 0 or j == 1:
                    #         ax.set_xlabel("Step")
                    #     else:
                    #         ax.set_xlabel("Myr")

    icon_filenames = [
        "u.png",
        "l.png",
        "d.png",
        "c.png",
    ]
    icons = [os.path.join("uldc_model_icons", path) for path in icon_filenames]
    icon_imgs = [Image.open(path) for path in icons]
    img_width, img_height = icon_imgs[1].size
    sf = 1.1
    shape_cats = ["Unlobed", "Lobed", "Dissected", "Compound"]

    for j in range(0, 4):
        ax = axs[0, j + 1]
        # ax.axis("off")
        ax.imshow(icon_imgs[j])
        # ax.text(img_width / 2, img_height, shape_cats[j], ha="center", va="top")
        # ax.set_xlim(img_width / sf, (-img_width / 2) / sf)
        ax.set_xlim(0 + (img_width * (sf - 1)), img_width - (img_width * (sf - 1)))
        ax.set_ylim(img_height, -(img_height / sf))
    for idx, i in enumerate([1, 2, 4]):
        ax = axs[i, 0]

        labs = [
            "Simulation Data\nMUT2",
            "Simulation CTMC\nMUT2",
            "Phylogeny CTMC\nZuntini et al. (2024)",
            # "Zuntini et al. (2024)\nphylogeny, Naturalis\nclassification CTMC",
        ]

        ax.text(
            0.2,
            0.5,
            labs[idx],
            ha="center",
            va="center",
        )
    #     ax.axis("off")
    #     ax.imshow(icon_imgs[i - 1])
    #     ax.text(img_width / 2, img_height, shape_cats[i - 1], ha="center", va="top")
    #     ax.set_xlim(0, (img_width / scale_factor) + ((img_width / 2) / scale_factor))
    #     ax.set_ylim(img_height / scale_factor, (-img_height / 2) / scale_factor)
    # axes[0, 4].axis("off")

    legend = fig.legend(
        lines,
        order_full,
        loc="outside right",
        title="Final shape",
        # fontsize=11,
        ncol=1,
    )
    title = legend.get_title()
    title.set_fontsize(11)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.17, hspace=0.17, right=0.84)
    plt.savefig("curves.pdf", format="pdf", dpi=1200)
    plt.show()


def plot_sim_and_phylogeny_curves_vert():

    lb = 30
    ub = 70

    #### Get phylo-rates ####
    phylo_dir = "../phylogeny/rates/uniform_1010000steps"
    # phylo = "jan_phylo_nat_class"  # the phylo-class to use for the curves
    # phylo = "jan_phylo_nat_class_21-01-24_95_each"
    # phylo = "geeta_phylo_geeta_class_23-04-24_shuff"
    # phylo = "zuntini_phylo_nat_class"
    phylo = "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1"
    # phylo = "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1"
    phylo_xlim = 200  # 0.1
    phylo_rates_list = []

    for filename in os.listdir(phylo_dir):
        if filename.endswith(".csv"):
            path = os.path.join(phylo_dir, filename)
            df = pd.read_csv(path)
            df["phylo-class"] = filename[:-4]
            phylo_rates_list.append(df)

    phylo_rates = pd.concat(phylo_rates_list, ignore_index=True)
    # Choose phylogeny for curves
    phylo_rates = phylo_rates[phylo_rates["phylo-class"] == phylo]
    phylo_rates.drop(columns="phylo-class", inplace=True)
    phylo_rates.reset_index(drop=True, inplace=True)
    # Insert stasis rates
    phylo_rates.insert(
        0,
        "q00",
        -phylo_rates["q01"] - phylo_rates["q02"] - phylo_rates["q03"],
    )
    phylo_rates.insert(
        5,
        "q11",
        -phylo_rates["q10"] - phylo_rates["q12"] - phylo_rates["q13"],
    )
    phylo_rates.insert(
        10,
        "q22",
        -phylo_rates["q20"] - phylo_rates["q21"] - phylo_rates["q23"],
    )
    phylo_rates.insert(
        15,
        "q33",
        -phylo_rates["q30"] - phylo_rates["q31"] - phylo_rates["q32"],
    )

    # Calculate means
    phylo_summary = phylo_rates.mean().reset_index()
    phylo_summary.columns = ["transition", "mean_rate"]
    # Calculate confidence intervals
    confidence_intervals = {}
    for col in phylo_rates.columns:
        data = phylo_rates[col].dropna()
        confidence_intervals[col] = (
            # np.mean(data) - (1.96 * stats.sem(data)),
            # np.mean(data) + (1.96 * stats.sem(data)),
            # np.mean(data) - np.std(data),
            # np.mean(data) + np.std(data),
            np.percentile(data, lb),  # calculate credible interval from posterior
            np.percentile(data, ub),
        )

    phylo_summary["lb"] = [i[0] for i in confidence_intervals.values()]
    phylo_summary["ub"] = [i[1] for i in confidence_intervals.values()]

    #### Get sim-rates ####
    # sim_rates = pd.read_csv(
    #     "../data-processing/markov_fitter_reports/emcee/24chains_25000steps_15000burnin/emcee_run_log_24-04-24.csv"
    # )
    # sim_rates = pd.read_csv(
    #     "markov_fitter_reports/emcee/24chains_25000steps_15000burnin/MUT2.2_emcee_run_log_24-04-24.csv"
    # )
    # mean = pd.read_csv("markov_fitter_reports/emcee/avg/MUT2.2/emcee_run_log_mean.csv")
    # ub = pd.read_csv("markov_fitter_reports/emcee/avg/MUT2.2/emcee_run_log_ub.csv")
    # lb = pd.read_csv("markov_fitter_reports/emcee/avg/MUT2.2/emcee_run_log_lb.csv")
    sim_rates = (
        pd.read_csv("markov_fitter_reports/emcee/leaf_uncert_posteriors_MUT2.csv") * 0.1
    )

    # sim_rates = pd.concat([mean, ub, lb])
    # sim_rates = ub
    # avg_list = [lb, mean, ub]

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

    sim_summaries = []

    # for sim_rates in avg_list:
    sim_rates = sim_rates.rename(columns=name_map)
    sim_rates.insert(0, "q00", -sim_rates["q01"] - sim_rates["q02"] - sim_rates["q03"])
    sim_rates.insert(5, "q11", -sim_rates["q10"] - sim_rates["q12"] - sim_rates["q13"])
    sim_rates.insert(
        10,
        "q22",
        -sim_rates["q20"] - sim_rates["q21"] - sim_rates["q23"],
    )
    sim_rates.insert(
        15,
        "q33",
        -sim_rates["q30"] - sim_rates["q31"] - sim_rates["q32"],
    )

    # Calculate means
    sim_summary = sim_rates.mean().reset_index()
    sim_summary.columns = ["transition", "mean_rate"]
    # Calculate confidence intervals
    confidence_intervals = {}
    for col in sim_rates.columns:
        data = sim_rates[col].dropna()
        confidence_intervals[col] = (
            # np.mean(data) - (1.96 * stats.sem(data)),
            # np.mean(data) + (1.96 * stats.sem(data)),
            # np.mean(data) - np.std(data),
            # np.mean(data) + np.std(data),
            # np.mean(data) - (200 * np.var(data, ddof=1)),
            # np.mean(data) + (200 * np.var(data, ddof=1)),
            np.percentile(data, lb),
            np.percentile(data, ub),
        )
    sim_summary["lb"] = [i[0] for i in confidence_intervals.values()]
    sim_summary["ub"] = [i[1] for i in confidence_intervals.values()]
    # sim_summaries.append(sim_summary)
    print(sim_summary)

    # Use the mean of the emcee mean, ub of emcee ub, lb of emcee lb
    # sim_summary = pd.DataFrame(
    #     {
    #         "transition": sim_summaries[1]["transition"],
    #         "mean_rate": sim_summaries[1]["mean_rate"],
    #         "lb": sim_summaries[0]["lb"],
    #         "ub": sim_summaries[2]["ub"],
    #     }
    # )
    # print(sim_summary)

    #### Get sim timeseries data ####
    # dfs = concatenator()
    # for walk in dfs:
    #     walk["step"] = walk.index.values
    # concat = pd.concat(dfs, ignore_index=True)
    # concat = pd.merge(concat, first_cats[["leafid", "first_cat"]], on="leafid")
    concat = pd.read_csv("MUT2.2_trajectories_shape.csv")
    print(concat)

    # total of each shape per step for each leafid
    timeseries = (
        concat.groupby(["leafid", "first_cat", "step", "shape"])
        .size()
        .reset_index(name="total_shape_firstcat")
    )
    print(timeseries)

    # no. active walks per step for each leafid
    timeseries_total = (
        timeseries.groupby(["leafid", "first_cat", "step"])
        .agg(total_firstcat=("total_shape_firstcat", "sum"))
        .reset_index()
    )
    print(timeseries_total)

    # proportion of active walks in each shape category for each leafid
    timeseries = timeseries.merge(timeseries_total, on=["leafid", "first_cat", "step"])
    timeseries["proportion"] = (
        timeseries["total_shape_firstcat"] / timeseries["total_firstcat"]
    )
    print(timeseries)

    # mean proportion of active walks in each shape category for all leaves in each first_cat
    timeseries = (
        timeseries.groupby(["first_cat", "step", "shape"])
        .agg(mean_prop=("proportion", "mean"), sterr=("proportion", "sem"))
        .reset_index()
    )
    timeseries["lb"] = timeseries["mean_prop"] - 1.96 * timeseries["sterr"]
    timeseries["ub"] = timeseries["mean_prop"] + 1.96 * timeseries["sterr"]

    # add initial state to the timeseries
    timeseries["step"] = timeseries["step"] + 1
    for i in order:
        for j in order:
            if i == j:
                timeseries.loc[-1] = {
                    "first_cat": i,
                    "step": 0,
                    "shape": i,
                    # "total_shape_firstcat": np.nan,
                    # "total_firstcat": np.nan,
                    # "proportion": 1,
                    "mean_prop": 1,
                    "sterr": 0,
                    "lb": 1,
                    "ub": 1,
                }
            else:
                timeseries.loc[-1] = {
                    "first_cat": i,
                    "step": 0,
                    "shape": j,
                    "mean_prop": 0,
                    "sterr": 0,
                    "lb": 0,
                    "ub": 0,
                }
            timeseries.index = timeseries.index + 1
            timeseries = timeseries.sort_index()
    print(timeseries)

    # print(phylo_rates)
    # print(phylo_summary)
    # print(sim_rates)
    # print(sim_summary)
    # print(timeseries)

    # produce phylo-curves
    t_vals = np.linspace(0, phylo_xlim, 120)

    phylo_curves = []
    Q = np.array(phylo_summary["mean_rate"].values).reshape(4, 4)
    QL = np.array(phylo_summary["lb"].values).reshape(4, 4)
    QU = np.array(phylo_summary["ub"].values).reshape(4, 4)
    for t in t_vals:
        Pt = linalg.expm(Q * t)
        PLt = linalg.expm(QL * t)
        PUt = linalg.expm(QU * t)
        phylo_curves.append([Pt, PLt, PUt])

    phylo_plot = pd.DataFrame(plot_data_from_probcurves(phylo_curves, t_vals))

    # produce sim-curves

    t_vals = np.linspace(0, 120, 120)
    sim_curves = []
    Q = np.array(sim_summary["mean_rate"].values).reshape(4, 4)
    QL = np.array(sim_summary["lb"].values).reshape(4, 4)
    QU = np.array(sim_summary["ub"].values).reshape(4, 4)
    for t in t_vals:
        Pt = linalg.expm(Q * t)
        PLt = linalg.expm(QL * t)
        PUt = linalg.expm(QU * t)
        sim_curves.append([Pt, PLt, PUt])

    sim_plot = pd.DataFrame(plot_data_from_probcurves(sim_curves, t_vals))

    print(phylo_plot)
    print(sim_plot)

    # Create subplots
    plt.rcParams["font.family"] = "CMU Serif"
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(9, 9), sharey=True)

    lines = []
    order_full = ["unlobed", "lobed", "dissected", "compound"]

    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            cat = order[i]
            ax.set_ylim(0, 1)
            if j == 0:  # timeseries data on the left
                cat_data = timeseries[timeseries["first_cat"] == cat]
                cat_data = cat_data.rename(columns={"mean_prop": "P", "step": "t"})
                # ax.set_title(order_full[i])
                if i == 0:
                    ax.set_title("MUT2 Data")
                ax.set_ylabel("Mean Prop.")
                ax.annotate(  # This adds the row labels
                    f"Initial shape\n{order_full[i]}",
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    size="large",
                    ha="right",
                    va="center",
                )
                ax.set_xlim(0, 60)
            if j == 1:  # simulation ctmc in centre column
                cat_data = sim_plot[sim_plot["first_cat"] == cat]
                # ax.set_title(order_full[i])
                if i == 0:
                    ax.set_title("MUT2 CTMC")
                ax.set_ylabel("P")
                ax.set_xlim(0, 60)
            if j == 2:  # phylogeny data on the right
                cat_data = phylo_plot[phylo_plot["first_cat"] == cat]
                ax.set_ylabel("P")
                # ax.set_title(order_full[i])
                if i == 0:
                    ax.set_title(
                        "Zuntini et al. (2024)\nphylogeny, Naturalis\nclassification CTMC",
                        # "jan_phylo\n_nat_class_21\n-01-24_95_each"
                        # phylo,
                    )
                ax.set_xlim(0, phylo_xlim)

            for s, shape in enumerate(order):
                shape_data = cat_data[cat_data["shape"] == shape]
                (line,) = ax.plot(
                    shape_data["t"],
                    shape_data["P"],
                    label=shape,
                    c=sns_palette[s],
                    linestyle="-",  # cmap=cmap
                )
                ax.fill_between(
                    shape_data["t"],
                    shape_data["lb"],
                    shape_data["ub"],
                    alpha=0.2,
                    # cmap=cmap,
                )
                lines.append(line)
            if i < 3:
                ax.set_xticklabels([])
            if i == 3:
                if j == 0 or j == 1:
                    ax.set_xlabel("Step")
                else:
                    ax.set_xlabel("Myr")
    legend = fig.legend(
        lines,
        order_full,
        loc="outside right",
        title="Final shape",
        # fontsize=11,
        ncol=1,
    )
    title = legend.get_title()
    title.set_fontsize(11)
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    plt.show()


def randomwalk_rates_firstswitch():
    dfs = concatenator()

    for walk in dfs:
        walk["step"] = walk.index.values

    # final_rows = [df.iloc[-1] for df in dfs if not df.empty] # to get the state at the end of each walk
    # Get the step at the first shape switch or the last row if no switch
    firstswitch_dfs = []
    for df in dfs:
        if not df.empty:
            first_cat = first_cats[first_cats["leafid"] == df.iloc[0]["leafid"]][
                "first_cat"
            ].values[0]
            walk_length = len(df)
            for i, value in enumerate(df["shape"]):
                if (
                    i + 1 < walk_length
                ):  # if the shape never changes, append the final step anyway
                    if value != first_cat:
                        firstswitch_dfs.append(df.iloc[i])
                        break
                else:
                    firstswitch_dfs.append(df.iloc[i])

    concat = pd.concat(firstswitch_dfs, axis=1).T.reset_index(drop=True)
    concat = pd.merge(concat, first_cats[["leafid", "first_cat"]], on="leafid")
    mapping = {"u": 0, "l": 1, "d": 2, "c": 3}
    concat["shape_id"] = concat["shape"].map(mapping)
    print(concat)

    # frequency of shapes at firstswitch step grouped by leafid
    grouped_by_leaf = (
        concat.groupby(["leafid", "first_cat", "shape"])
        .agg(mean_step=("step", "mean"))
        .reset_index()
    )
    print(grouped_by_leaf)
    # make a rate metric
    grouped_by_leaf["mean_step"] = grouped_by_leaf["mean_step"].replace(0, np.nan)
    grouped_by_leaf["100-mean_step"] = 100 - grouped_by_leaf["mean_step"]

    sns.catplot(
        data=grouped_by_leaf,
        y="100-mean_step",
        x="first_cat",
        hue="shape",
        kind="bar",
        order=order,
        hue_order=order,
    )
    plt.show()


def randomwalk_rates_step60():
    dfs = concatenator()

    for walk in dfs:
        walk["step"] = walk.index.values

    concat = pd.concat(dfs, ignore_index=True)
    concat = pd.merge(concat, first_cats[["leafid", "first_cat"]], on="leafid")
    mapping = {"u": 0, "l": 1, "d": 2, "c": 3}
    concat["shape_id"] = concat["shape"].map(mapping)
    print(concat)

    step60 = concat[concat["step"] == 60].reset_index(drop=True)
    print(step60)

    grouped_by_leaf = (
        step60.groupby(["leafid", "first_cat", "shape"])
        .size()
        .reset_index(name="total_shape_leafid")
    )
    print(grouped_by_leaf)

    sns.catplot(
        data=grouped_by_leaf,
        y="total_shape_leafid",
        x="first_cat",
        hue="shape",
        kind="bar",
        order=order,
        hue_order=order,
    )
    plt.show()


def randomwalk_rates_allswitch():
    dfs = concatenator()

    trans_time = {
        "uu": [],
        "ul": [],
        "ud": [],
        "uc": [],
        "lu": [],
        "ll": [],
        "ld": [],
        "lc": [],
        "du": [],
        "dl": [],
        "dd": [],
        "dc": [],
        "cu": [],
        "cl": [],
        "cd": [],
        "cc": [],
    }

    for walk in dfs:
        walk["step"] = walk.index.values
        walk = walk[walk["step"] <= 60].reset_index(drop=True)
        if not walk.empty:
            first_cat = first_cats[first_cats["leafid"] == walk.iloc[0]["leafid"]][
                "first_cat"
            ].values[0]
            current_shape = first_cat
            holding_nsteps = 0
            for i, row in walk.iterrows():
                if row["shape"] == current_shape:
                    holding_nsteps += 1
                else:
                    if i == 0:
                        end_shape = first_cat
                    else:
                        end_shape = walk["shape"][i - 1]
                    next_shape = row["shape"]
                    transition = end_shape + next_shape
                    trans_time[transition].append(holding_nsteps)
                    current_shape = row["shape"]
                    holding_nsteps = 0

    flat = [(key, value) for key, values in trans_time.items() for value in values]
    trans_time_df = pd.DataFrame(flat, columns=["transition", "holding_nsteps"])
    trans_time_df["transition_count"] = trans_time_df["transition"].map(
        trans_time_df["transition"].value_counts()
    )
    trans_time_df["count/hold"] = (
        trans_time_df["transition_count"] / trans_time_df["holding_nsteps"]
    )
    print(trans_time_df)

    sns.catplot(
        data=trans_time_df,
        y="count/hold",
        x="transition",
        kind="box",
    )
    plt.show()
    # concat = pd.concat(dfs, ignore_index=True)
    # concat = pd.merge(concat, first_cats[["leafid", "first_cat"]], on="leafid")
    # mapping = {"u": 0, "l": 1, "d": 2, "c": 3}
    # concat["shape_id"] = concat["shape"].map(mapping)

    # step60 = concat[concat["step"] <= 60].reset_index(drop=True)
    # print(step60)

    # leafwalk = step60.groupby(["leafid", "walkid"])


def MLE_rates_barplot():
    data = pd.read_csv("MUT2.2_MLE_rates.csv")
    replacement = {
        "State 1": "u",
        "State 2": "l",
        "State 3": "d",
        "State 4": "c",
        "State.1": "u",
        "State.2": "l",
        "State.3": "d",
        "State.4": "c",
    }
    data["initial_shape"] = data["initial_shape"].replace(replacement)
    data["final_shape"] = data["final_shape"].replace(replacement)
    data = data[data["initial_shape"] != data["final_shape"]]
    data.rename(
        columns={
            "rate": "MLE",
            "initial_shape": "Initial Shape",
            "final_shape": "Final Shape",
        },
        inplace=True,
    )
    print(data)
    # yerr = (data["SE"] * 1.96).to_list()
    # print(yerr)
    # plt.figure(figsize=(10, 6))
    # plt.bar(data["transition"], data["rate"], yerr=yerr)
    # plt.show()
    melt = pd.melt(
        data,
        id_vars=["Initial Shape", "Final Shape", "SE", "transition"],
        value_vars=["MLE", "LB", "UB"],
        value_name="rate",
    )
    print(melt)
    order_full = ["unlobed", "lobed", "dissected", "compound"]
    labels = ["unlobed(u)", "lobed(l)", "dissected(d)", "compound(c)"]
    g = sns.catplot(
        data=melt,
        y="rate",
        x="Initial Shape",
        hue="Final Shape",
        kind="bar",
        errorbar=lambda x: (x.min(), x.max()),
        order=order,
        hue_order=order,
    )
    g.set_xticklabels(labels=labels)
    plt.ylabel("Evolutionary Rate")
    plt.show()


if __name__ == "__main__":
    # curves_CTMC_MLEsimfit()
    # curves_CTMC_mcmcsimfit()
    # MLE_rates_barplot()
    # randomwalk_rates_firstswitch()
    plot_sim_and_phylogeny_curves()
    # plot_sim_and_phylogeny_curves_vert()

    # stack_plot()
    # paramspace()

    prop_curves()

    # curves_phylogeny()

    # randomwalk_rates_allswitch()
