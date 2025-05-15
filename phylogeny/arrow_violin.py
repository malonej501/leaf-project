import os
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from PIL import Image


LAYOUT = "h"
SF = 4  # 6 good for v with 2 plots # scale factor for the thickness of the arrows
C_VAL = 2  # increase to increase the curviness of the arrows
DC_VAL = 5  # increas to increase diagonal arrow curviness
# the credible interval for shading the arrows grey or black
# (>CI must be above or below zero to be black)
CI = 0.90

NORM_MTHD = "meanmean"
ML_DATA = "ML6_genus_mean_rates_all"
# sim1 = "MUT1_mcmc_11-12-24"
# sim2 = "MUT2_mcmc_11-12-24"
SIM1 = "MUT1_06-02-25"
SIM2 = "MUT2_mcmc_05-02-25"

PLOT_ORDER = [
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


def get_rates_batch(directory):
    """Retreive all phylo rates and concatenate into a single dataframe"""
    data = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            path = os.path.join(directory, filename)
            df = pd.read_csv(path)
            df["phylo-class"] = filename[:-4]
            data.append(df)

    data_concat = pd.concat(data, ignore_index=True)

    return data_concat


def calc_net_rate(rates):
    """Calculate the net rate for each transition by subtracting the reverse 
    transition rate from the forward transition rate"""
    transitions = rates_map3.values()
    for fwd in transitions:
        bwd = fwd[2] + "→" + fwd[0]
        if fwd != bwd:
            col_name = f"{fwd}-{bwd}"
            rates[col_name] = rates[fwd] - rates[bwd]
    return rates


def import_phylo_and_sim_rates(calc_diff):
    """Import Q-matrix posteriors for phylo and sim and calculate net rates"""
    # to return diff between rates rather than raw rates, set calc_diff true
    #### import data ####
    phylo_rates = get_rates_batch(directory="rates/uniform_1010000steps")
    s1 = pd.read_csv(f"../data-processing/markov_fitter_reports/emcee/{SIM1}/"
                     f"posteriors_{SIM1}.csv")
    s2 = pd.read_csv(f"../data-processing/markov_fitter_reports/emcee/{SIM2}/"
                     f"posteriors_{SIM2}.csv")
    s1["phylo-class"] = "MUT1_simulation"
    s2["phylo-class"] = "MUT2_simulation"

    sim_rates = pd.concat([s1, s2]).reset_index(drop=True)
    sim_rates = sim_rates.rename(columns=rates_map2)
    phylo_rates = phylo_rates.rename(columns=rates_map3)
    phylo_sim = pd.concat([sim_rates, phylo_rates]).reset_index(drop=True)
    phylo_sim = phylo_sim.rename(columns={"phylo-class": "Dataset"})

    phylo_sim = calc_net_rate(phylo_sim) if calc_diff else phylo_sim

    phylo_sim_long = pd.melt(
        phylo_sim,
        id_vars=["Dataset"],
        var_name="transition",
        value_name="rate",
    )
    phylo_sim_long["dataname"] = phylo_sim_long["Dataset"].apply(
        lambda x: x.split("_class", 1)[0] + "_class"
    )
    # filter to only rows with dataset in the PLOT_ORDER list
    phylo_sim_long = phylo_sim_long[
        phylo_sim_long["Dataset"].isin(PLOT_ORDER)
    ].reset_index(drop=True)

    return phylo_sim_long


def import_phylo_ml_rates(calc_diff):
    """Import Q-matrix ML estimates for phylo"""
    qml = pd.read_csv(f"rates/ML/{ML_DATA}.csv")
    qml.drop(
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
    qml = qml.rename(columns=rates_map3)
    qml["dataname"] = qml["dataset"].apply(
        lambda x: x.split("_class", 1)[0] + "_class"
    )

    qml = calc_net_rate(qml) if calc_diff else qml

    qml_long = pd.melt(
        qml,
        id_vars=["dataname", "dataset"],
        var_name="transition",
        value_name="rate",
    )
    # filter to only rows with dataset in the PLOT_ORDER list
    ml_plot_order = [x.split("_class", 1)[0] + "_class" for x in PLOT_ORDER]
    qml_long = qml_long[
        qml_long["dataname"].isin(ml_plot_order)
    ].reset_index(drop=True)
    return qml_long


def test_rates_diff_from_zero(phylo_sim_long):
    """Return fraction of normalised rate posteriors that are > 0"""
    phylo_sim_long_filt = phylo_sim_long[
        phylo_sim_long["transition"].str.contains(
            r"^[a-zA-Z]→[a-zA-Z]-[a-zA-Z]→[a-zA-Z]$"
        )
    ]
    transitions = phylo_sim_long_filt["transition"].unique()
    results = []
    for trans in transitions:
        mcmc_plot_data = phylo_sim_long[phylo_sim_long["transition"] == trans]
        for name, group in mcmc_plot_data.groupby("Dataset"):
            n = len(group)
            ng0 = len(group[group["rate_norm"] > 0])  # number of rates > 0

            mean = group["rate_norm"].mean()
            # t_stat, p_val = stats.ttest_1samp(group["rate_norm"], 0)
            std = group["rate_norm"].std()
            results.append(
                {
                    "dataset": name,
                    "transition": trans,
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


def arc(ax, startpoint, endpoint, curvature, rate, rate_c, rate_std,
        significant, cur):
    """Draw an arc between two points with a given curvature and thickness
     proportional to rate"""
    # increase cur to increase the curviness of the arrows
    if not rate_std:
        if significant:
            arrow = FancyArrowPatch(
                (startpoint[0], startpoint[1]),  # Start point
                (endpoint[0], endpoint[1]),  # End point
                connectionstyle=f"arc3,rad={curvature}.{cur}",  # Curvature
                mutation_scale=10,  # Arrow head size
                arrowstyle="-|>",  # Arrow style
                color=rate_c,  # Arrow color
                lw=rate,  # Line width
            )
        else:
            arrow = FancyArrowPatch(
                (startpoint[0], startpoint[1]),  # Start point
                (endpoint[0], endpoint[1]),  # End point
                connectionstyle=f"arc3,rad={curvature}.{cur}",  # Curvature
                mutation_scale=10,  # Arrow head size
                arrowstyle="-|>",  # Arrow style
                color=rate_c,  # Arrow color
                lw=rate,  # Line width
                zorder=0,
            )
        arrow.set_joinstyle("miter")
        arrow.set_capstyle("butt")
        ax.add_patch(arrow)

    if rate_std:
        arrow_ub = FancyArrowPatch(
            (startpoint[0], startpoint[1]),  # Start point
            (endpoint[0], endpoint[1]),  # End point
            connectionstyle=f"arc3,rad={curvature}.{cur}",  # Curvature
            mutation_scale=10,  # Arrow head size
            arrowstyle="-|>",  # Arrow style
            color="black",  # Arrow color
            fill=False,
            lw=rate + rate_std + 10,  # Line width
        )
        arrow_ub.set_joinstyle("miter")
        arrow = FancyArrowPatch(
            (startpoint[0], startpoint[1]),  # Start point
            (endpoint[0], endpoint[1]),  # End point
            connectionstyle=f"arc3,rad={curvature}.{cur}",  # Curvature
            mutation_scale=10,  # Arrow head size
            arrowstyle="-|>",  # Arrow style
            alpha=0.2,
            color=rate_c,  # Arrow color
            lw=rate,  # Line width
        )
        arrow.set_joinstyle("miter")
        arrow_lb = FancyArrowPatch(
            (startpoint[0], startpoint[1]),  # Start point
            (endpoint[0], endpoint[1]),  # End point
            connectionstyle=f"arc3,rad={curvature}.{cur}",  # Curvature
            mutation_scale=10,  # Arrow head size
            arrowstyle="-|>",  # Arrow style
            fill=False,
            color="white",
            lw=rate - rate_std,  # Line width
        )
        arrow_lb.set_joinstyle("miter")

        # ax.add_patch(arrow)
        ax.add_patch(arrow_ub)
        ax.add_patch(arrow_lb)


def nodes(ax, c_proper, rad, texts, icon_imgs):
    """Draw nodes with leaf icons and text"""
    # textplacement goes from top right clockwise
    # text_placement = ((-0.7, 0.7), (0.7, 0.7), (0.7, -0.7), (-0.7, -0.7))
    text_placement = ((-0.1, 1), (0.1, 1), (0.1, -1), (-0.1, -1))
    for i, center in enumerate(c_proper.values()):
        x, y = center
        theta = np.linspace(0, 2 * np.pi, 100)
        w, h = icon_imgs[i].size
        scf = 0.0021

        ax.imshow(
            icon_imgs[i],
            extent=[
                x - (w * scf / 2),
                x + (w * scf / 2),
                y - (h * scf / 2),
                y + (h * scf / 2),
            ],
            rasterized=True,
        )

        ax.text(
            x + text_placement[i][0],
            y + text_placement[i][1],
            texts[i],
            horizontalalignment="left" if i in [1, 2] else "right",
            verticalalignment="center",
            color="black",
        )


def import_sim_ml_rates(calc_diff):
    """Import Q-matrix ML estimates for sim"""
    mut1_ml = pd.read_csv(
        f"../data-processing/markov_fitter_reports/emcee/{SIM1}/ML_{SIM1}.csv")
    mut1_ml["Dataset"] = "MUT1_simulation"
    mut2_ml = pd.read_csv(
        f"../data-processing/markov_fitter_reports/emcee/{SIM2}/ML_{SIM2}.csv")
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

    sim_ml_long = pd.melt(sim_ml, id_vars="Dataset",
                          var_name="transition", value_name="rate")
    sim_ml_long["dataname"] = sim_ml_long["Dataset"] + "_class"
    return sim_ml_long


def normalise_rates(phylo_sim_long, ml_phylo_rates_long, ml_sim_rates_long):
    """
    Normalise the rates across datasets with the specified method
    NORM_MTHD = "meanmean", "zscore", "zscore+2.7", "zscore_global", "minmax"
    """
    # only calculate the mean rate for transitions, not transition differences
    phylo_sim_long_filt = phylo_sim_long[
        phylo_sim_long["transition"].isin(rates_map3.values())
    ]
    phylo_sim_long["mean_rate"] = phylo_sim_long_filt.groupby(
        ["Dataset", "transition"]
    )["rate"].transform(
        "mean"
    )  # get mean rate for each transition per dataset

    if NORM_MTHD == "meanmean":
        # get the mean mean transition rate per dataset (i.e. the centre of the
        # rates for that dataset)
        phylo_sim_long["mean_mean"] = phylo_sim_long.groupby(["Dataset"])[
            "mean_rate"
        ].transform("mean")
        # normalise by dividing by the mean mean transition rate for each
        # dataset
        phylo_sim_long["rate_norm"] = (
            phylo_sim_long["rate"] / phylo_sim_long["mean_mean"]
        )

        # merge mcmc mean-means with ML-rates
        ml_phylo_rates_long = pd.merge(
            ml_phylo_rates_long,
            phylo_sim_long[["dataname", "mean_mean"]].drop_duplicates(
                subset=["dataname"]
            ),
            on="dataname",
        )
        ml_sim_rates_long = pd.merge(
            ml_sim_rates_long,
            phylo_sim_long[["dataname", "mean_mean"]].drop_duplicates(
                subset=["dataname"]
            ),
            on="dataname",
        )
        # normalise ML rates
        ml_phylo_rates_long["rate_norm"] = (
            ml_phylo_rates_long["rate"] / ml_phylo_rates_long["mean_mean"]
        )
        ml_sim_rates_long["rate_norm"] = (
            ml_sim_rates_long["rate"] / ml_sim_rates_long["mean_mean"]
        )
        # phylo_sim_long["initial_shape"], phylo_sim_long["final_shape"] = zip(
        #     *phylo_sim_long["transition"].map(rates_map)
        # )

    elif NORM_MTHD == "zscore":
        # get the mean mean transition rate per dataset (i.e. the centre of the
        # rates for that dataset)
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
        ml_phylo_rates_long["rate_norm"] = (
            ml_phylo_rates_long["rate"] - phylo_sim_long["mean_mean"]
        ) / phylo_sim_long["std_mean"]

    elif NORM_MTHD == "zscore+2.7":
        # get the mean mean transition rate per dataset (i.e. the centre of the
        # rates for that dataset)
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
        ml_phylo_rates_long["rate_norm"] = (
            (ml_phylo_rates_long["rate"] - phylo_sim_long["mean_mean"])
            / phylo_sim_long["std_mean"]
        ) + 2.7

    elif NORM_MTHD == "zscore_global":
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
        ml_phylo_rates_long["rate_norm"] = (
            ml_phylo_rates_long["rate"] - phylo_sim_long["dataset_mean"]
        ) / phylo_sim_long["dataset_std"]

    elif NORM_MTHD == "minmax":
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
        ml_phylo_rates_long["rate_norm"] = (
            ml_phylo_rates_long["rate"] - phylo_sim_long["min_mean"]
        ) / (phylo_sim_long["max_mean"] - phylo_sim_long["min_mean"])

    else:
        raise RuntimeError("Invalid NORM_MTHD argument.")

    return phylo_sim_long, ml_phylo_rates_long, ml_sim_rates_long


def draw_arc(r, at, to, ax, c, rc, rs, sig, c_val, dc_val):
    """Draw the appropriate arcs between nodes depending on the node
    identities."""
    if r > 0:
        if at + to == "ul":
            arc(ax, c["uS"], c["lS"], "+",
                r, rc, rs, sig, c_val)
        if at + to == "lu":
            arc(ax, c["lN"], c["uN"], "+",
                r, rc, rs, sig, c_val)
        if at + to == "ld":
            arc(ax, c["lE"], c["dE"], "-",
                r, rc, rs, sig, c_val)
        if at + to == "dl":
            arc(ax, c["dE"], c["lE"], "+",
                r, rc, rs, sig, c_val)
        if at + to == "dc":
            arc(ax, c["dS"], c["cS"], "-",
                r, rc, rs, sig, c_val)
        if at + to == "cd":
            arc(ax, c["cS"], c["dS"], "+",
                r, rc, rs, sig, c_val)
        if at + to == "cu":
            arc(ax, c["cW"], c["uW"], "-",
                r, rc, rs, sig, c_val)
        if at + to == "uc":
            arc(ax, c["uE"], c["cE"], "-",
                r, rc, rs, sig, c_val)

        #### diagonals ####
        if at == "u" and to == "d":
            arc(ax, c["uS"], c["dW"], "+",
                r, rc, rs, sig, dc_val)
        if at == "d" and to == "u":
            arc(ax, c["dW"], c["uS"], "-",
                r, rc, rs, sig, dc_val)
        if at == "c" and to == "l":
            arc(ax, c["cE"], c["lS"], "+",
                r, rc, rs, sig, dc_val)
        if at == "l" and to == "c":
            arc(ax, c["lS"], c["cE"], "-",
                r, rc, rs, sig, dc_val)


def arrow_w_viol():
    """Plot arrows and violin plots for the phylo and sim rates"""
    # plt.rcParams["font.family"] = "CMU Serif"

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

    ml_phylo_rates_long = import_phylo_ml_rates(calc_diff=True)
    ml_sim_rates_long = import_sim_ml_rates(calc_diff=True)
    phylo_sim_long = import_phylo_and_sim_rates(calc_diff=True)
    phylo_sim_long, ml_phylo_rates_long, ml_sim_rates_long = normalise_rates(
        phylo_sim_long, ml_phylo_rates_long, ml_sim_rates_long
    )

    rate_data = test_rates_diff_from_zero(phylo_sim_long)
    rate_data["fwd"] = rate_data["transition"].str[:3]
    rate_data["bwd"] = rate_data["transition"].str[-3:]
    rate_data.to_csv(f"rate_data_bw_fw_{str(date.today())}.csv", index=False)

    cmap = plt.get_cmap("viridis")
    rate_data["std_c"] = rate_data["std"].map(cmap)
    rate_data.to_csv(
        f"rate_data_{NORM_MTHD}_{str(date.today())}.csv", index=False)

    # centers = [(2, 6), (6, 6), (6, 2), (2, 2)]
    c_proper = {"u": (2, 6), "l": (6, 6), "d": (6, 2), "c": (2, 2)}
    rad = 0.5  # padding between leaf icon and circle where arrows join
    texts = ["Unlobed", "Lobed", "Dissected",
             "Compound"] if LAYOUT == "v" else ["", "", "", ""]
    c = {
        "uN": (2, 6 + rad),  # define arrow attachment points for nodes
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
        "MUT1",
        "MUT2",
        "Janssens et al. (2020)",
        "Zuntini et al. (2024)",
        "Geeta et al. (2012)",
    ]

    # Draw circles with text in the center
    if LAYOUT == "h":
        fig_h = 6
        fig_w = 14
        fig, axs = plt.subplots(
            2,
            len(PLOT_ORDER),
            figsize=(fig_w, fig_h),
        )
    elif LAYOUT == "v":
        fig_h = 4 * len(PLOT_ORDER)
        fig_w = 8
        fig, axs = plt.subplots(
            len(PLOT_ORDER) * 2,
            2,
            figsize=(fig_h, fig_w),
            gridspec_kw={
                "height_ratios": [
                    3 if i % 2 == 0 else 1 for i in range(len(PLOT_ORDER) * 2)
                ],
                "width_ratios": [2, 1],
            },
        )
    else:
        raise RuntimeError("incorrect LAYOUT argument")
    if len(PLOT_ORDER) == 1:
        axs = [axs]
    dataset = None
    for i, row_ in enumerate(axs):
        for j, ax in enumerate(row_):
            if LAYOUT == "v" and i % 2 == 1:
                ax.axis("off")
                continue
            dataset = PLOT_ORDER[j] if LAYOUT == "h" else PLOT_ORDER[i // 2]
            if (LAYOUT == "h" and i == 0) or (LAYOUT == "v" and j == 0):
                ax.axis("off")
                if LAYOUT == "v":
                    ax = plt.subplot2grid(shape=(2, 2), loc=(i // 2, 0))
                    ax.axis("off")
                ax.set_xlim(0, 8)
                ax.set_ylim(0, 8)

                plot_data = rate_data[rate_data["dataset"] == dataset]
                nodes(ax, c_proper, rad, texts, icon_imgs)
                for _, row in plot_data.iterrows():
                    at = row["fwd"][0]
                    to = row["fwd"][2]
                    # increase the multiplier to scale the width of the arrows
                    r = row["mean_rate_norm"] * SF  # adjusted rate
                    rc = "lightgrey" if row["prop_over_zero"] < CI else "black"
                    sig = False if row["prop_over_zero"] < CI else True
                    rs = False  # set to false to disable multi-arrows

                    draw_arc(r, at, to, ax, c, rc, rs, sig,
                             C_VAL, DC_VAL)  # draw arrow

                title = plot_titles[j] if LAYOUT == "h" \
                    else plot_titles[i // 2]
                ax.set_title(title, fontsize=9)
            if (LAYOUT == "h" and i == 1) or (LAYOUT == "v" and j == 1):

                mcmc_plot_data = phylo_sim_long[
                    phylo_sim_long["Dataset"] == dataset]
                ml_phylo_plot_data = ml_phylo_rates_long[
                    ml_phylo_rates_long["dataname"].apply(
                        lambda x: x in dataset)
                ]
                ml_sim_plot_data = ml_sim_rates_long[
                    ml_sim_rates_long["Dataset"].apply(lambda x: x in dataset)]
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
                            ml_rates.append(x.loc[0, "rate_norm"])
                    if not ml_sim_plot_data.empty:
                        x = ml_sim_plot_data[
                            ml_sim_plot_data["transition"] == transition
                        ].reset_index(drop=True)
                        if not x.empty:
                            ml_rates.append(x.loc[0, "rate_norm"])
                    if ml_phylo_plot_data.empty and ml_sim_plot_data.empty:
                        ml_rates.append(np.nan)

                ax.violinplot(rates, showextrema=False, showmeans=True)

                if ML_DATA:
                    pos = list(range(1, len(transitions) + 1))
                    ax.scatter(pos, ml_rates, color="black", zorder=5, s=8,
                               facecolors="white")  # , marker="D")
                ax.axhline(0, linestyle="--", color="C1", alpha=0.5)
                ax.set_ylim(-8, 8)
                ax.set_xticks(
                    list(range(1, len(transitions) + 1)),
                    transitions,
                    fontsize=9,
                )
                xl, yl, xh, yh = np.array(ax.get_position()).ravel()
                w = xh - xl
                h = yh - yl
                size = (
                    0.1 if LAYOUT == "h" else 0.08
                )  # worked well for horizontal and multiple vertical
                for x, xtick_pos in enumerate(range(1, len(transitions) + 1)):
                    xp = (
                        xl
                        + (w / (len(transitions)) * xtick_pos)
                        - (0.5 * (w / (len(transitions))))
                    )
                    ax1 = fig.add_axes(
                        [xp - (size * 0.5), yl - size -
                         (0.1 * size), size, size]
                    )
                    ax1.axison = False
                    ax1.imshow(trans_imgs[x])
                ax.set_xticklabels([])
                ax.set_ylabel("Net normalised rate")
                if LAYOUT == "h" and j > 0:
                    ax.set_ylabel("")

    # Create legend
    labels = ["Compound", "Dissected", "Lobed", "Unlobed"]
    # left_pos, bottom_pos, width, height
    axlgnd = fig.add_axes([0.88, 0.4, 0.1, 0.2])
    icon_w, icon_h = icon_imgs[0].size
    axlgnd.set_ylim(0, len(icon_imgs) * icon_h)
    for i, img in enumerate(reversed(icon_imgs)):
        axlgnd.imshow(img, extent=(0, icon_w, i*icon_h, (i*icon_h) + icon_h))
        axlgnd.text(icon_w, (i*icon_h) + (icon_h * 0.5),
                    labels[i], ha="left", va="center", fontsize=9)
        axlgnd.axis("off")

    # plt.tight_LAYOUT()
    plt.savefig("arrow_violin_plot.svg", format="svg", dpi=1200)
    plt.show()


if __name__ == "__main__":
    arrow_w_viol()
