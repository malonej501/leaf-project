import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, ArrowStyle
import os
from datetime import date
from PIL import Image


layout="h"
sf = 4 #6 good for v with 2 plots # scale factor for the thickness of the arrows
c_val = 2 # increase to increase the curviness of the arrows
dc_val = 5 # increas to increase diagonal arrow curviness
ci = 0.90 # the credible interval for shading the arrows grey or black (>ci must be above or below zero to be black)

norm_method="meanmean"
ML_data="ML6_genus_mean_rates_all"
sim1 = "MUT1_mcmc_11-12-24"
sim2 = "MUT2_mcmc_11-12-24"

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
    transitions = rates_map3.values()
    for fwd in transitions:
        bwd = fwd[2] + "→" + fwd[0]
        if fwd != bwd:
            col_name = f"{fwd}-{bwd}"
            rates[col_name] = rates[fwd] - rates[bwd]
    return rates

def import_phylo_and_sim_rates(plot_order, calc_diff):
    # to return differences between rates rather than raw rates, set calc_diff to true
    #### import data ####
    phylo_rates = get_rates_batch(directory="rates/uniform_1010000steps")
    s1 = pd.read_csv(f"../data-processing/markov_fitter_reports/emcee/{sim1}/posteriors_{sim1}.csv")
    s2 = pd.read_csv(f"../data-processing/markov_fitter_reports/emcee/{sim2}/posteriors_{sim2}.csv") 
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
    # filter to only rows with dataset in the plot_order list
    phylo_sim_long = phylo_sim_long[
        phylo_sim_long["Dataset"].isin(plot_order)
    ].reset_index(drop=True)

    return phylo_sim_long

def import_phylo_ML_rates(plot_order, calc_diff):
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
    
    ML_rates = calc_net_rate(ML_rates) if calc_diff else ML_rates

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
    return ML_rates_long

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


def arc(ax, startpoint, endpoint, curvature, rate, rate_c, rate_std, significant, cur):
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
    # textplacement goes from top right clockwise
    # text_placement = ((-0.7, 0.7), (0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)) # original
    text_placement = ((-0.1, 1), (0.1, 1), (0.1, -1), (-0.1, -1))
    for i, center in enumerate(c_proper.values()):
        x, y = center
        theta = np.linspace(0, 2 * np.pi, 100)
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
        )

        ax.text(
            x + text_placement[i][0],
            y + text_placement[i][1],
            texts[i],
            horizontalalignment="left" if i == 1 or i == 2 else "right",
            verticalalignment="center",
            color="black",
        )

def import_sim_ML_rates(calc_diff):
    mut1_ml = pd.read_csv(f"../data-processing/markov_fitter_reports/emcee/{sim1}/ML_{sim1}.csv")
    mut1_ml["Dataset"] = "MUT1_simulation"
    mut2_ml = pd.read_csv(f"../data-processing/markov_fitter_reports/emcee/{sim2}/ML_{sim2}.csv")
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

def normalise_rates(phylo_sim_long, ML_phylo_rates_long, ML_sim_rates_long):
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

    return phylo_sim_long, ML_phylo_rates_long, ML_sim_rates_long



def arrow_w_viol():
    plt.rcParams["font.family"] = "CMU Serif"
    
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

    ML_phylo_rates_long = import_phylo_ML_rates(plot_order, calc_diff=True)
    ML_sim_rates_long = import_sim_ML_rates(calc_diff=True)
    phylo_sim_long = import_phylo_and_sim_rates(plot_order, calc_diff=True)
    phylo_sim_long, ML_phylo_rates_long, ML_sim_rates_long = normalise_rates(
        phylo_sim_long, ML_phylo_rates_long, ML_sim_rates_long
    )

    rate_data = test_rates_diff_from_zero(phylo_sim_long)
    rate_data["fwd"] = rate_data["transition"].str[:3]
    rate_data["bwd"] = rate_data["transition"].str[-3:]
    rate_data.to_csv(f"rate_data_bw_fw_{str(date.today())}.csv", index=False)

    cmap = plt.get_cmap("viridis")
    rate_data["std_c"] = rate_data["std"].apply(lambda x: cmap(x))
    rate_data.to_csv(f"rate_data_{norm_method}_{str(date.today())}.csv", index=False)

    # centers = [(2, 6), (6, 6), (6, 2), (2, 2)]
    c_proper = {"u": (2, 6), "l": (6, 6), "d": (6, 2), "c": (2, 2)}

    rad = 0.5
    texts = ["Unlobed","Lobed","Dissected","Compound"] if layout == "v" else ["","","",""]

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
    else:
        raise (RuntimeError("incorrect layout argument"))
    if len(plot_order) == 1:
        axs = [axs]
    for i, row_ in enumerate(axs):
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

                plot_data = rate_data[rate_data["dataset"] == dataset]
                nodes(ax, c_proper, rad, texts, icon_imgs)
                for k, row in plot_data.iterrows():
                    at = row["fwd"][0]
                    to = row["fwd"][2]
                    r = row["mean_rate_norm"] * sf # increase the multiplier to scale the width of the arrows
                    rc = "lightgrey" if row["prop_over_zero"] < ci else "black"
                    sig = False if row["prop_over_zero"] < ci else True
                    rs = False  # set to false to disable multi-arrows


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
                title = plot_titles[j] if layout == "h" else plot_titles[i // 2]
                ax.set_title(title, fontsize=9)
            if (layout == "h" and i == 1) or (layout == "v" and j == 1):

                mcmc_plot_data = phylo_sim_long[phylo_sim_long["Dataset"] == dataset]
                ml_phylo_plot_data = ML_phylo_rates_long[
                    ML_phylo_rates_long["dataname"].apply(lambda x: x in dataset)
                ]
                ml_sim_plot_data = ML_sim_rates_long[ML_sim_rates_long["Dataset"].apply(lambda x: x in dataset)]
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

                p = ax.violinplot(
                    rates,
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
                xl, yl, xh, yh = np.array(ax.get_position()).ravel()
                w = xh - xl
                h = yh - yl
                size = (
                    0.1 if layout == "h" else 0.08
                )  # worked well for horizontal and multiple vertical
                for x, xtick_pos in enumerate(list(range(1, len(transitions) + 1))):
                    xp = (
                        xl
                        + (w / (len(transitions)) * xtick_pos)
                        - (0.5 * (w / (len(transitions))))
                    )
                    ax1 = fig.add_axes(
                        [xp - (size * 0.5), yl - size - (0.1 * size), size, size]
                    )
                    ax1.axison = False
                    ax1.imshow(trans_imgs[x])
                ax.set_xticklabels([])
                ax.set_ylabel("Net normalised rate")
                if layout == "h" and j > 0:
                    ax.set_ylabel("")

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
    plt.savefig("arrow_violin_plot.svg", format="svg", dpi=1200)
    plt.show()

if __name__ == "__main__":
    arrow_w_viol()
