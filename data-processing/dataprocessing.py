import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
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

wd = "leaves_full_21-9-23_MUT2.2_CLEAN"
# resd = "/home/m/malone/leaf_storage/random_walks"
wd1 = "../vlab-5.0-3609-ubuntu-20_04/oofs/ext/NPHLeafModels_1.01/LeafGenerator"

sys.path.insert(1, wd1)

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
    paramshapess = param_concatenator()
    paramshapes = pd.concat(paramshapess, ignore_index=True)

    starting_leaves = pd.DataFrame(pdict.values()).transpose()

    only_valid_leaves = paramshapes[
        paramshapes.iloc[:, 3].str.contains("passed")
    ].reset_index(drop=True)
    # remove the metadata columns at the start and the leaf shape metrics columns from the classifier at the end
    only_valid_leaves = only_valid_leaves.iloc[:, 5:-6]
    only_valid_leaves.columns = range(only_valid_leaves.shape[1])
    # paramshapes_sub = only_valid_leaves.iloc[:, [0, 1, 2, 3, 36, 67]]
    # # print(paramshapes_sub.iloc[:, 3].str.contains("passed"))

    shapess = concatenator()
    shapes = pd.concat(shapess, ignore_index=True)

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

    data.to_csv("/home/m/malone/leaf_storage/random_walks/result.csv", index=False)
    scaled_data = StandardScaler().fit_transform(data)

    pca_params = PCA(n_components=2)
    princip_params = pca_params.fit_transform(scaled_data)
    princip_df = pd.DataFrame(data=princip_params, columns=["pc1", "pc2"])
    princip_df_starting_leaves = princip_df.iloc[: len(starting_leaves)]
    princip_df_starting_leaves["shape"] = list(first_cats["first_cat"].values)
    princip_df_result = princip_df.iloc[len(starting_leaves) :]
    princip_df_result["shape"] = shapes["shape"]
    print(len(shapes), len(princip_df_result))
    # sns.displot( # for heatplot
    g = sns.displot(
        x="pc1",
        y="pc2",
        hue="shape",
        col="shape",
        col_wrap=2,
        col_order=order,
        hue_order=order,
        kind="hist",
        data=princip_df_result,
        palette="colorblind",
        bins=100,
        # alpha=0.5,
    )
    # sns.relplot(
    #     x="pc1",
    #     y="pc2",
    #     hue="shape",
    #     col="shape",
    #     col_wrap=2,
    #     col_order=order,
    #     hue_order=order,
    #     kind="scatter",
    #     data=princip_df_starting_leaves,
    #     palette="colorblind",
    #     ax=g.axes,
    # )

    # overlay starting leaves on the heatmap
    for ax in g.axes.flat:
        sns.scatterplot(
            x="pc1",
            y="pc2",
            hue="shape",
            hue_order=order,
            data=princip_df_starting_leaves,
            palette="colorblind",
            legend=False,
            linewidth=0,
            # edgecolor="black",
            ax=ax,
        )
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
    print(dfs)

    # mapping = {"u":0, "l":1, "c":2, "d":3}
    # concat = pd.concat(dfs, ignore_index=True)
    # concat = pd.merge(concat, first_cats[['leafid', 'first_cat']], on='leafid')
    # concat["dummy"] = concat["shape"].map(mapping)
    # concat["step"] = concat.index.values

    # sns.lineplot(data=concat, x="step", y="dummy", hue="first_cat")
    # plt.xlabel("Step")
    # plt.ylabel("Shape")
    # plt.show()

    # for i, walk in enumerate(dfs):
    #     mapping = {"u":0, "l":1, "c":2, "d":3}
    #     color_map = {"u":"blue", "l":"red", "c":"green", "d":"yellow"}

    #     walk["first_cat"] = first_cats[first_cats["leafid"] == walk.iloc[0]["leafid"]]["first_cat"].values[0]
    #     #walk["first_cat"] = walk["first_cat"].map(color_map)
    #     #walk["first_cat"] = walk["first_cat"].map(mapping)
    #     walk["dummy"] = walk["shape"].map(mapping)
    #     walk["step"] = walk.index.values
    #     #print(walk)

    #     if walk["first_cat"][0] == "u":
    #         sns.lineplot(data=walk, x="step", y="dummy", color="blue", alpha=0.1)
    #     if walk["first_cat"][0]  == "l":
    #         sns.lineplot(data=walk, x="step", y="dummy", color="red", alpha=0.1)
    #     if walk["first_cat"][0]  == "d":
    #         sns.lineplot(data=walk, x="step", y="dummy", color="green", alpha=0.1)
    #     if walk["first_cat"][0]  == "c":
    #         sns.lineplot(data=walk, x="step", y="dummy", color="yellow", alpha=0.1)

    #     #plt.plot(walk.index.values, walk["dummy"].values, c=walk["first_cat"], alpha=0.1)
    #     #sns.lineplot(x=walk.index.values, y=walk["dummy"].values, hue=walk["first_cat"].to_numpy(), alpha=0.1)
    #     #sns.lineplot(data=walk, x="step", y="dummy", hue="first_cat", alpha=0.1)

    # plt.legend()
    # plt.xlabel("Step")
    # plt.ylabel("Shape")
    # plt.show()

    # Normalise by the length of each walk
    # fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # for i, cat in enumerate(["u", "l", "d", "c"]):
    #     ax = axs[i//2, i%2]
    #     for walk in dfs:
    #         walk = pd.merge(walk, first_cats[['leafid', 'first_cat']], on='leafid')
    #         if walk["first_cat"][0] == cat:
    #             mapping = {"u":0, "l":1, "d":2, "c":3}
    #             walk["dummy"] = walk["shape"].map(mapping)
    #             walk["frac_through_walk"] = (walk.reset_index().index)/len(walk)
    #             walk["step"] = walk.index.values
    #             ax.plot(walk["frac_through_walk"], walk["dummy"], color="black", alpha=0.1)
    #     ax.set_xlabel("Frac. through walk")
    #     ax.set_ylabel("Shape")
    #     ax.set_title(f"First Category: {cat}")
    #     ax.set_yticks(range(4))

    # plt.tight_layout()
    # plt.show()

    for walk in dfs:
        walk["step"] = walk.index.values

    concat = pd.concat(dfs, ignore_index=True)
    concat = pd.merge(concat, first_cats[["leafid", "first_cat"]], on="leafid")
    mapping = {"u": 0, "l": 1, "d": 2, "c": 3}
    concat["dummy"] = concat["shape"].map(mapping)

    grouped = (
        concat.groupby(["first_cat", "step", "shape"]).size().reset_index(name="count")
    )
    print(grouped)

    # https://seaborn.pydata.org/generated/seaborn.objects.Area.html
    # colours = ["#41342C","#158471","#97AF25","#CAC535"]
    # (
    #     so.Plot(grouped, "step", "count").facet("first_cat", wrap=2)
    #     .add(so.Area(alpha=.7), so.Stack(), legend=True, color="shape")
    #     .layout(engine = None)
    #     .show()
    # )

    lfirst = grouped[grouped["first_cat"] == "l"]
    print(lfirst)

    # (
    #     so.Plot(lfirst, "step", "count")
    #     .add(so.Area(alpha=.7), so.Stack(), legend=False, color="shape")
    #     .show()
    # )

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
        .reset_index(name="total_first_cat_step_shape")
    )
    # total no. leaves per step for every first_cat
    grouped_by_first_cat_total = (
        grouped_by_first_cat.groupby(["first_cat", "step"])
        .agg(total_first_cat_step=("total_first_cat_step_shape", "sum"))
        .reset_index()
    )

    grouped_by_first_cat = grouped_by_first_cat.merge(
        grouped_by_first_cat_total, on=["first_cat", "step"]
    )
    grouped_by_first_cat["proportion"] = (
        grouped_by_first_cat["total_first_cat_step_shape"]
        / grouped_by_first_cat["total_first_cat_step"]
    )
    print(grouped_by_first_cat)

    # get total no. leaves per shape, per step for each leaf
    grouped_by_leaf = (
        concat.groupby(["leafid", "first_cat", "step", "shape"])
        .size()
        .reset_index(name="count")
    )
    # get total no. leaves per step per leaf
    grouped_by_leaf_total = (
        grouped_by_leaf.groupby(["leafid", "first_cat", "step"])
        .agg(total_leafid_first_cat_step=("count", "sum"))
        .reset_index()
    )

    grouped_by_leaf = grouped_by_leaf.merge(
        grouped_by_leaf_total, on=["leafid", "first_cat", "step"]
    )
    # get proportion of each shape per step per leaf
    grouped_by_leaf["proportion"] = (
        grouped_by_leaf["count"] / grouped_by_leaf["total_leafid_first_cat_step"]
    )

    # grouped_by_leaf_avg_by_first_cat = (
    #     grouped_by_leaf.groupby(["first_cat", "step", "shape"])
    #     .agg(mean_prop=("proportion", "mean"))
    #     .reset_index()
    # )

    # grouped_by_leaf_std_by_first_cat = (
    #     grouped_by_leaf.groupby(["first_cat", "step"])
    #     .agg(std=("proportion", "std"))
    #     .reset_index()
    # )
    # # get std for the means of the proportions for each step for each first_cat
    # grouped_by_leaf_avg_by_first_cat = grouped_by_leaf_avg_by_first_cat.merge(
    #     grouped_by_leaf_std_by_first_cat, on=["first_cat", "step"]
    # )

    # print(grouped_by_leaf_avg_by_first_cat)
    # print(grouped_by_leaf_avg_by_first_cat["std"].describe())

    concat_grouped_by_first_cat = concat.merge(
        grouped_by_first_cat, on=["first_cat", "step", "shape"]
    )

    # captures within first_cat uncertainty
    concat_grouped_by_leaf = concat.merge(
        grouped_by_leaf, on=["leafid", "first_cat", "step", "shape"]
    )

    grouped_by_leaf_total = grouped_by_leaf_total.merge(
        grouped_by_first_cat, on=["first_cat", "step"]
    )

    grouped_by_leaf_total["proportion_leafid"] = (
        grouped_by_leaf_total["total_first_cat_step_shape"]
        / grouped_by_leaf_total["total_leafid_first_cat_step"]
    )

    grouped_by_leaf_leafid_total = (
        grouped_by_leaf.groupby(["leafid", "first_cat", "step", "shape"])
        .agg(total_leafid_shape_step=("count", "sum"))
        .reset_index()
    )

    grouped_by_leaf_total = grouped_by_leaf_total.merge(
        grouped_by_leaf_leafid_total, on=["leafid", "first_cat", "step", "shape"]
    )

    print(grouped_by_leaf_total)

    # sns.relplot(
    #     data=grouped_by_leaf_total,
    #     x="step",
    #     y="proportion_leafid",
    #     col="first_cat",
    #     hue="shape",
    #     kind="line",
    #     col_wrap=2,
    #     # errorbar="sd",
    # ).set_axis_labels("step", "average proportion")

    # sns.relplot(
    #     data=grouped_by_leaf_total,
    #     x="step",
    #     y="proportion_first_cat/leafid",
    #     col="first_cat",
    #     hue="shape",
    #     kind="line",
    #     col_wrap=2,
    #     # errorbar="sd",
    # ).set_axis_labels("step", "average proportion")

    sns.relplot(
        data=grouped_by_first_cat,
        x="step",
        y="proportion",
        col="first_cat",
        hue="shape",
        kind="line",
        col_wrap=2,
        # errorbar="sd",
    )

    plt.show()


# stack_plot()
# paramspace()

prop_curves()
