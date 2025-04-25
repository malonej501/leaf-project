import os
import pandas as pd
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from scipy import linalg
from scipy.stats import chisquare, chi2
from PIL import Image
import seaborn as sns

PLOT = 2  # type of plot to produce
# 0-three rows with error bars,
# 1-two rows with mean model,
# 2-proportion of shapes over simulation time against model predictions
# 3-chi-squared goodness of fit for CTMC proportions to sim proportions
PHYLORATES = "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1"
PHYLORATES2 = "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1"
# SIMRATES = "MUT2_mcmc_05-02-25"  # best fit so far
# SIMRATES = "test"
# SIMRATES = "MUT2_320_mle_20-03-25" # best fit so far 24-04-25
SIMRATES = "MUT2_320_mcmc_2_24-04-25"
# simdata = "MUT2.2_trajectories_shape.csv"
# simdata = "MUT5_mcmc_10-02-25.csv"
SIMDATA = "MUT2_320_mle_23-04-25.csv"
# SIMDATA = "pwalks_10_160_leaves_full_13-03-25_MUT2_CLEAN.csv"
# SIMDATA = "MUT2_mle_11-02-25.csv"
MC_ERR_SAMP = 500  # no. monte carlo samples for mcmc error propagation
T_STAT = 0  # 0=mean prop, 1=prop - stat used for the timeseries plot
SIMRATES_METHOD = 0  # 0=mcmc, 1=mle
EQ_INIT = False  # plot timeseries from equal numbers of each initial shape
LB = 2.5  # 5 #5 # credible interval for phylo and sim mcmc rates
UB = 97.5  # 95 #95
PHYLO_XLIM = 160
SIM_XLIM = 320  # 60
SIM_XON = 0  # begin x axis at this value
RESET_FIRST_CAT = False  # redefine first_cat to shape at step SIM_XON
YSCALE = "linear"  # log or linear
# show phylogeny CTMC fit ontop as well as sim data and CTMC fit in plot 2
SHOW_PHYLO = True
V = True  # verbose for debugging

# sns.set_palette("colorblind")
ORDER = ["u", "l", "d", "c"]
VAR = "mean_prop" if T_STAT == 0 else "prop"

# exclude these in concatenator to infer from equal numbers of each initshape
EXCL = ["p0_121", "p1_82", "p2_195", "p9_129", "p5_249", "pu3", "p2_78_alt",
        "p3_60",  # unlobed
        "p8_1235", "p1_35", "p12b",  # dissected
        "pc4", "p12de", "p7_437", "p2_346_alt", "p6_1155"]  # compound


def plot_data_from_probcurves(curves, t_plot, t_calc):
    """Takes a list of probability matrices and returns a dataframe with the
    transition probabilities in a format that can be plotted"""

    # attach plot time to data, along with time used for calculation
    plot_data = {"t": [], "t_calc": [], "first_cat": [],
                 "shape": [], "P": [], "lb": [], "ub": []}

    for i, li in enumerate(curves):
        for j, matrix in enumerate(li):
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    if j == 0:
                        plot_data["t_calc"].append(t_calc[i])
                        plot_data["t"].append(t_plot[i])
                        plot_data["first_cat"].append(row)
                        plot_data["shape"].append(column)
                        plot_data["P"].append(matrix[row, column])
                    elif j == 1:
                        plot_data["lb"].append(matrix[row, column])
                    elif j == 2:
                        plot_data["ub"].append(matrix[row, column])

    plot_data = pd.DataFrame(plot_data)
    mapping = {0: "u", 1: "l", 2: "d", 3: "c"}
    plot_data["first_cat"] = plot_data["first_cat"].replace(mapping)
    plot_data["shape"] = plot_data["shape"].replace(mapping)
    # replace any upper bound value greater than 1 with 1 (because prob)
    plot_data["ub"] = plot_data["ub"].clip(upper=1)
    return plot_data


def get_phylo_rates(phylo=PHYLORATES):
    """Get the mean and confidence intervals for the posterior distributions
    of all transition rates from phylogenetic mcmc inference specified by
    phylo"""

    phylo_dir = "../phylogeny/rates/uniform_1010000steps"
    phylo_rates_list = []

    for filename in os.listdir(phylo_dir):
        if filename.endswith(".csv"):
            path = os.path.join(phylo_dir, filename)
            df = pd.read_csv(path)
            df["phylo-class"] = filename[:-4]
            phylo_rates_list.append(df)

    p_rate = pd.concat(phylo_rates_list, ignore_index=True)
    # Choose phylogeny for curves
    p_rate = p_rate[p_rate["phylo-class"] == phylo]
    p_rate.drop(columns="phylo-class", inplace=True)
    p_rate.reset_index(drop=True, inplace=True)
    # Insert stasis rates
    p_rate.insert(0, "q00", -p_rate["q01"] - p_rate["q02"] - p_rate["q03"],)
    p_rate.insert(5, "q11", -p_rate["q10"] - p_rate["q12"] - p_rate["q13"],)
    p_rate.insert(10, "q22", -p_rate["q20"] - p_rate["q21"] - p_rate["q23"],)
    p_rate.insert(15, "q33", -p_rate["q30"] - p_rate["q31"] - p_rate["q32"],)

    # Calculate means
    phylo_summary = p_rate.mean().reset_index()
    phylo_summary.columns = ["transition", "mean_rate"]
    # Calculate confidence intervals
    confidence_intervals = {}
    for col in p_rate.columns:
        data = p_rate[col].dropna()
        confidence_intervals[col] = (
            np.percentile(data, LB),  # credible interval
            np.percentile(data, UB),
        )

    phylo_summary["lb"] = [i[0] for i in confidence_intervals.values()]
    phylo_summary["ub"] = [i[1] for i in confidence_intervals.values()]

    if V:
        print(f"Phylogeny rates\n{phylo_summary}")
    return phylo_summary, p_rate


def get_sim_rates():
    """Get the mean and confidence intervals for the posterior distributions
    of all transition rates from simulation mcmc inference specified by
    SIMRATES"""

    sim_rates = (
        pd.read_csv(f"markov_fitter_reports/emcee/{SIMRATES}/posteriors_"
                    f"{SIMRATES}.csv" if SIMRATES_METHOD == 0 else
                    f"markov_fitter_reports/emcee/{SIMRATES}/ML_{SIMRATES}"
                    ".csv")
    )
    name_map = {"0": "q01", "1": "q02", "2": "q03", "3": "q10", "4": "q12",
                "5": "q13", "6": "q20", "7": "q21", "8": "q23", "9": "q30",
                "10": "q31", "11": "q32"}

    # for sim_rates in avg_list:
    sim_rates = sim_rates.rename(columns=name_map)
    sim_rates.insert(
        0, "q00", -sim_rates["q01"] - sim_rates["q02"] - sim_rates["q03"])
    sim_rates.insert(
        5, "q11", -sim_rates["q10"] - sim_rates["q12"] - sim_rates["q13"])
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

    # Tweaks
    # sim_summary.loc[1, "mean_rate"] -= 0.003  # decrease q01
    # sim_summary.loc[0, "mean_rate"] += 0.003  # increase q00
    # sim_summary.loc[3, "mean_rate"] -= 0.002  # decrease q01
    # sim_summary.loc[0, "mean_rate"] += 0.002  # increase q00
    # sim_summary.loc[12, "mean_rate"] -= 0.015  # decrease q30
    # sim_summary.loc[15, "mean_rate"] += 0.015  # increase q33``

    # Calculate confidence intervals
    confidence_intervals = {}
    for col in sim_rates.columns:
        data = sim_rates[col].dropna()
        confidence_intervals[col] = (
            np.percentile(data, LB),
            np.percentile(data, UB),
        )
    sim_summary["lb"] = [i[0] for i in confidence_intervals.values()]
    sim_summary["ub"] = [i[1] for i in confidence_intervals.values()]
    if V:
        print(f"Simulation rates\n{sim_summary}")
    return sim_summary, sim_rates


def get_timeseries():
    """
    Get the mean proportion of each shape category at each step of the walk
    from the simulation data.
    T_STAT      statistic that is used to summaries the timeseries
                0   ...mean of the proportion across leafids within first_cats
                1   ...proportion across all leafids
    """
    concat = pd.read_csv(SIMDATA)

    if RESET_FIRST_CAT:
        # reset first_cat to the shape at step SIM_XON
        concat["first_cat"] = concat.groupby(["leafid", "walkid"])[
            "shape"].transform(lambda x: x.iloc[SIM_XON])
        # create new unique leafid by appending first_cat - ensure that
        # there are no duplicate leafid, step, shape rows
        concat["leafid"] = concat["leafid"] + "_" + concat["first_cat"]
        step_sim_xon = concat[concat["step"] == SIM_XON]
        mis = step_sim_xon[step_sim_xon["first_cat"] != step_sim_xon["shape"]]
        assert len(mis) == 0, f"Mismatch in first_cat and shape at {SIM_XON}"

    if EQ_INIT:
        concat = concat[~concat["leafid"].isin(EXCL)].reset_index(drop=True)
        if V:
            print(concat.drop_duplicates(
                subset=["leafid"]).sort_values(by="first_cat"))

    # get the number of leaves in each initial group
    shape_counts = concat.groupby(["leafid", "first_cat"]).size().reset_index()
    init_shape_counts = shape_counts["first_cat"].value_counts().reset_index()
    init_shape_counts = init_shape_counts.set_index("first_cat")[
        "count"].to_dict()

    if T_STAT == 0:
        # total of each shape per step for each leafid
        tseries = (
            concat.groupby(["leafid", "first_cat", "step", "shape"])
            .agg(shape_total=("shape", "size"))
            .reset_index()
        )
        # check for leafids with duplicate steps
        dup = tseries[tseries.duplicated(
            subset=["leafid", "step", "shape"])]
        assert len(dup) == 0, f"Duplicate leafid, step, shape rows: {dup}"

        # fill in transitions that didn't occur
        all_transitions = pd.MultiIndex.from_product(
            [tseries["leafid"].unique(), tseries["step"].unique(),
             {"u", "l", "d", "c"}], names=["leafid", "step", "shape"]
        )  # all possible combinations of leafid, step and shape, then reindex
        tseries = tseries.set_index(["leafid", "step", "shape"])
        tseries = tseries.reindex(
            all_transitions, fill_value=None).reset_index()

        # fill empty first-cat values with the first non nan first_cat value
        tseries["first_cat"] = tseries.groupby(
            ["leafid"])["first_cat"].transform("first")
        tseries = tseries.fillna(0)

        # no. active walks per step for each leafid
        tseries_total = (
            tseries.groupby(["leafid", "first_cat", "step"])
            .agg(no_active_walks=("shape_total", "sum"))
            .reset_index()
        )  # if using pseudo walks, will always be 1

        # proportion of active walks in each shape category for each leafid
        tseries = tseries.merge(
            tseries_total, on=["leafid", "first_cat", "step"])
        tseries["proportion"] = (
            tseries["shape_total"] / tseries["no_active_walks"]
        )  # for pseudo walks, will always be either 1 or 0

        # mean prop active walks per shape for all leaves in each first_cat
        tseries = (  # don't group by first cat if PLOT == 2
            tseries.groupby(["first_cat", "step", "shape"] if PLOT != 2 else
                            ["step", "shape"])
            .agg(mean_prop=("proportion", "mean"),
                 sterr=("proportion", "sem"),
                 n=("proportion", "size"),
                 total=("proportion", "sum"))
            .reset_index()
        )
        if PLOT == 2:  # PLOT 2 means calculated across all first_cats
            assert tseries[tseries["n"] !=
                           48].empty, "n is not 48 for all steps."

    elif T_STAT == 1:
        tseries = concat.groupby(
            ["first_cat", "step", "shape"]
        ).size().reset_index(name="shape_total")
        # no. active walks per step
        tseries_total = tseries.groupby(["first_cat", "step"]).agg(
            no_active_walks=("shape_total", "sum")).reset_index()
        # proportion of active walks in each shape category
        tseries = tseries.merge(
            tseries_total, on=["first_cat", "step"])
        tseries["prop"] = tseries["shape_total"] / \
            tseries["no_active_walks"]
        tseries["sterr"] = 0

    tseries["lb"] = tseries[VAR] - 1.96 * tseries["sterr"]
    tseries["ub"] = tseries[VAR] + 1.96 * tseries["sterr"]

    if PLOT != 2:
        # add initial state to the tseries
        tseries["step"] = tseries["step"] + 1
        for i in ORDER:
            for j in ORDER:
                if i == j:
                    tseries.loc[-1] = {"first_cat": i, "step": 0, "shape": i,
                                       VAR: 1, "sterr": 0, "lb": 1, "ub": 1}
                else:
                    tseries.loc[-1] = {"first_cat": i, "step": 0, "shape": j,
                                       VAR: 0, "sterr": 0, "lb": 0, "ub": 0}
                tseries.index = tseries.index + 1
                tseries = tseries.sort_index()
    if V:
        print(f"Simulation data\n{tseries}")
    tseries.to_csv("tseries.csv")
    return tseries


def plot_sim_and_phylogeny_curves():
    """Plot data and phylogeny and sim predictions each in separate rows"""

    #### Get phylo-rates ####
    phylo_summary, _ = get_phylo_rates()
    #### Get sim-rates ####
    sim_summary, _ = get_sim_rates()
    #### Get sim timeseries data ####
    tseries = get_timeseries()

    # produce phylo-curves
    t_vals = np.linspace(0, PHYLO_XLIM, 100)

    phylo_curves = []
    q = np.array(phylo_summary["mean_rate"].values).reshape(4, 4)
    ql = np.array(phylo_summary["lb"].values).reshape(4, 4)
    qu = np.array(phylo_summary["ub"].values).reshape(4, 4)
    for t in t_vals:
        pt = linalg.expm(q * t)
        pllt = linalg.expm(ql * t)
        puut = linalg.expm(qu * t)
        phylo_curves.append([pt, pllt, puut])

    phylo_plot = pd.DataFrame(
        plot_data_from_probcurves(phylo_curves, t_vals, t_vals))

    # produce sim-curves
    t_vals = np.linspace(SIM_XON, SIM_XLIM, 100)
    sim_curves = []
    q = np.array(sim_summary["mean_rate"].values).reshape(4, 4)
    ql = np.array(sim_summary["lb"].values).reshape(4, 4)
    qu = np.array(sim_summary["ub"].values).reshape(4, 4)
    for t in t_vals:
        pt = linalg.expm(q * t)
        pllt = linalg.expm(ql * t)
        puut = linalg.expm(qu * t)
        sim_curves.append([pt, pllt, puut])

    sim_plot = pd.DataFrame(
        plot_data_from_probcurves(sim_curves, t_vals, t_vals))

    # Create subplots
    plt.rcParams["font.family"] = "CMU Serif"
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(9, 9),
                            gridspec_kw={"height_ratios": [3, 3, 3, 1, 3]})
    lines = []
    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]

    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            idx = j - 1

            if i < 1 or j < 1:
                ax.axis("off")
                continue
            cat = ORDER[idx]
            ax.set_ylim(0, 1)
            if idx != 0:
                ax.set_yticklabels([])
            if i == 1:  # timeseries data on the left
                cat_data = tseries[tseries["first_cat"] == cat]
                cat_data = cat_data.rename(columns={VAR: "P", "step": "t"})
                ax.set_title(order_full[idx])
                ax.set_xlim(SIM_XON, SIM_XLIM+1)
                ax.set_xticks(
                    np.arange(SIM_XON, SIM_XLIM, (SIM_XLIM-SIM_XON)/4))
                ax.set_xlabel("Step")
                if idx == 0:
                    ax.set_ylabel("Mean Prop.")

            if i == 2:  # simulation ctmc in centre column
                cat_data = sim_plot[sim_plot["first_cat"] == cat]
                ax.set_xlabel("Step")
                ax.set_xlim(SIM_XON, SIM_XLIM+1)
                ax.set_xticks(
                    np.arange(SIM_XON, SIM_XLIM, (SIM_XLIM-SIM_XON)/4))
                if idx == 0:
                    ax.set_ylabel("P")

            if i == 3:
                ax.axis("off")

            if i == 4:  # phylogeny data on the right
                cat_data = phylo_plot[phylo_plot["first_cat"] == cat]
                ax.set_xlim(0, PHYLO_XLIM+1)
                ax.set_xticks(np.arange(0, PHYLO_XLIM+1, PHYLO_XLIM/4))
                ax.set_xlabel("Branch length (Myr)")
                if idx == 0:
                    ax.set_ylabel("P")
            if i != 3:
                for s, shape in enumerate(ORDER):
                    sd = cat_data[cat_data["shape"] == shape]  # shape data
                    (line,) = ax.plot(sd["t"], sd["P"], label=shape,
                                      c=sns.color_palette("colorblind")[s],
                                      linestyle="-")
                    ax.fill_between(["t"], sd["lb"], sd["ub"], alpha=0.2)
                    lines.append(line)

    icon_filenames = ["u.png", "l.png", "d.png", "c.png"]
    icons = [os.path.join("uldc_model_icons", path) for path in icon_filenames]
    icon_imgs = [Image.open(path) for path in icons]
    img_width, img_height = icon_imgs[1].size
    sf = 1.1

    for j in range(0, 4):
        ax = axs[0, j + 1]
        ax.imshow(icon_imgs[j])
        ax.set_xlim(0 + (img_width * (sf - 1)),
                    img_width - (img_width * (sf - 1)))
        ax.set_ylim(img_height, -(img_height / sf))
    for idx, i in enumerate([1, 2, 4]):
        ax = axs[i, 0]
        labs = ["Simulation Data\nMUT2",
                "Simulation CTMC\nMUT2",
                "Phylogeny CTMC\nZuntini et al. (2024)"]
        ax.text(0.2, 0.5, labs[idx], ha="center", va="center")

    legend = fig.legend(
        lines,
        order_full,
        loc="outside right",
        title="Final shape",
        ncol=1,
    )
    title = legend.get_title()
    title.set_fontsize(11)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.17, hspace=0.17, right=0.84)
    plt.savefig("curves.pdf", format="pdf", dpi=1200)
    plt.show()


def plot_sim_and_phylogeny_curves_nouncert():
    """Plot the MEAN proportion of each shape category at each step of the walk
    from the simulation data against the predicted proportions from the
    simulation and phylogeny CTMCs. The scale for the phylogeny is different
    to the simulation scale."""

    #### Get phylo-rates ####
    phylo_summary, _ = get_phylo_rates()
    #### Get sim-rates ####
    sim_summary, _ = get_sim_rates()
    #### Get sim timeseries data ####
    tseries = get_timeseries()

    # produce phylo-curves
    t_calc = np.linspace(0, PHYLO_XLIM, 100)
    t_plot = t_calc
    phylo_curves = []
    q = np.array(phylo_summary["mean_rate"].values).reshape(4, 4)
    ql = np.array(phylo_summary["lb"].values).reshape(4, 4)
    qu = np.array(phylo_summary["ub"].values).reshape(4, 4)
    for t in t_calc:
        pt = linalg.expm(q * t)
        pllt = linalg.expm(ql * t)
        puut = linalg.expm(qu * t)
        phylo_curves.append([pt, pllt, puut])
    phylo_plot = pd.DataFrame(
        plot_data_from_probcurves(phylo_curves, t_plot, t_calc))

    # produce sim-curves
    t_calc = np.linspace(SIM_XON, SIM_XLIM, 100)
    t_plot = t_calc
    if RESET_FIRST_CAT:
        t_calc = np.linspace(0, SIM_XLIM-SIM_XON, 100)
        t_plot = np.linspace(SIM_XON, SIM_XLIM, 100)
    sim_curves = []
    q = np.array(sim_summary["mean_rate"].values).reshape(4, 4)
    ql = np.array(sim_summary["lb"].values).reshape(4, 4)
    qu = np.array(sim_summary["ub"].values).reshape(4, 4)
    for t in t_calc:
        pt = linalg.expm(q * t)
        pllt = linalg.expm(ql * t)
        puut = linalg.expm(qu * t)
        sim_curves.append([pt, pllt, puut])
    sim_plot = pd.DataFrame(
        plot_data_from_probcurves(sim_curves, t_plot, t_calc))

    # Create subplots
    # plt.rcParams["font.family"] = "CMU Serif"
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(11, 7),
                            gridspec_kw={"height_ratios": [3, 3, 1, 3]})
    lines = []
    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]

    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            idx = j - 1
            sim_cat_data = pd.DataFrame()
            if i < 1 or j < 1:
                ax.axis("off")
                continue
            cat = ORDER[idx]
            if idx != 0:
                ax.set_yticklabels([])
            if i == 1:  # timeseries data on the left
                cat_data = tseries[tseries["first_cat"] == cat]
                cat_data = cat_data.rename(columns={VAR: "P", "step": "t"})
                ax.set_title(order_full[idx])
                ax.set_xlim(SIM_XON, SIM_XLIM)
                ax.set_xticks(
                    np.arange(SIM_XON, SIM_XLIM + 1, (SIM_XLIM-SIM_XON)/4))
                ax.set_xlabel("Step")
                if idx == 0:
                    ax.set_ylabel("P")
                sim_cat_data = sim_plot[sim_plot["first_cat"] == cat]
                sim_cat_data = sim_cat_data.rename(
                    columns={VAR: "P", "step": "t"})
            if i == 2:
                ax.axis("off")
            if i == 3:  # phylogeny data on the right
                cat_data = phylo_plot[phylo_plot["first_cat"] == cat]
                ax.set_xlim(0, PHYLO_XLIM)
                ax.set_xlabel("Branch length (Myr)")
                ax.set_xticks(np.arange(0, PHYLO_XLIM+1, PHYLO_XLIM/4))
                if idx == 0:
                    ax.set_ylabel("P")
            if i != 2:
                for s, shape in enumerate(ORDER):
                    sd = cat_data[cat_data["shape"] == shape]  # shape data
                    (line,) = ax.plot(sd["t"], sd["P"], label=shape,
                                      c=sns.color_palette("colorblind")[s],
                                      linestyle="--" if i == 1 else "-")
                    ax.fill_between(sd["t"], sd["lb"], sd["ub"], alpha=0.2)
                    if not sim_cat_data.empty:
                        sd = sim_cat_data[sim_cat_data["shape"] == shape]
                        (line,) = ax.plot(sd["t"], sd["P"], label=shape,
                                          c=sns.color_palette("colorblind")[s],
                                          linestyle="-")
                        lines.append(line)
            ax.set_ylim(0, 1)

            if YSCALE == "log":
                ax.set_ylim(1e-2, 1)
                ax.set_yscale(YSCALE)
                if j > 1:  # y labs off for all but leftmost panels
                    ax.set_yticklabels([])

    icon_filenames = ["u.png", "l.png", "d.png", "c.png"]
    icons = [os.path.join("uldc_model_icons", path) for path in icon_filenames]
    icon_imgs = [Image.open(path) for path in icons]
    img_width, img_height = icon_imgs[1].size
    sf = 1.1

    for j in range(0, 4):
        ax = axs[0, j + 1]
        ax.imshow(icon_imgs[j])
        ax.set_xlim(0 + (img_width * (sf - 1)),
                    img_width - (img_width * (sf - 1)))
        ax.set_ylim(img_height, -(img_height / sf))
    for idx, i in enumerate([1, 3]):
        ax = axs[i, 0]
        labs = ["Simulation Data\nand CTMC (MUT2)",
                "Phylogeny CTMC\nZuntini et al. (2024)"]
        ax.text(0.2, 0.5, labs[idx], ha="center", va="center")
    legend = fig.legend(
        lines,
        order_full,
        loc="outside right",
        title="Final shape",
        ncol=1,
    )
    title = legend.get_title()
    title.set_fontsize(11)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.17, hspace=0.17, right=0.84)
    plt.savefig("curves.pdf", format="pdf", dpi=1200)
    plt.show()


def get_timeseries_alt():
    """Return the proportion of each shape category at each step of the walk
    from the simulation data"""

    concat = pd.read_csv(SIMDATA)

    if EQ_INIT:
        concat = concat[~concat["leafid"].isin(EXCL)].reset_index(drop=True)
        if V:
            print(concat.drop_duplicates(subset=["leafid"]).sort_values(
                by="first_cat"))

    tseries = concat.groupby(["step", "shape"]).size().reset_index(
        name="shape_total")  # no. each shape at each step
    tseries["total"] = tseries.groupby("step")["shape_total"].transform(
        "sum")  # total no. shapes at each step
    tseries["prop"] = tseries["shape_total"] / tseries["total"]
    # proportion of each shape at each step

    return tseries


def get_plot_vals(rates, init_ratio, dtype):
    """Get the predicted proportion of each shape at each step of the walk
    from the simulation or phylogeny rates"""

    # no. time points must match no. steps in sim data, phylo time scale is
    # different
    t_vals = None
    if dtype == "phylo":
        t_vals = np.linspace(0, PHYLO_XLIM, SIM_XLIM)
    elif dtype == "sim":
        t_vals = np.linspace(0, SIM_XLIM, SIM_XLIM+1)

    q = np.array(rates["mean_rate"].values).reshape(4, 4)
    ql = np.array(rates["lb"].values).reshape(4, 4)
    qu = np.array(rates["ub"].values).reshape(4, 4)
    # multiply predicted transition probability by initial proportion of shapes
    # to get the predicted proportion of each shape at each step
    curve_dat = []  # store p for each t in t_vals
    for t in t_vals:
        pt = linalg.expm(q * t) * init_ratio[:, None]
        pllt = linalg.expm(ql * t) * init_ratio[:, None]
        puut = linalg.expm(qu * t) * init_ratio[:, None]
        curve_dat.append([pt, pllt, puut])
    # N.B. the phylo time scale is different to the sim time scale, but we
    # replace the phylo time scale with the sim time scale when plotting.
    # make pseudo time scale for phylo, so it can be plotted with sim data
    plot_dat = None
    if dtype == "phylo":
        plot_dat = pd.DataFrame(plot_data_from_probcurves(
            curve_dat, np.arange(SIM_XLIM), t_vals))
    elif dtype == "sim":  # sim time scale the same for plot and calc
        plot_dat = pd.DataFrame(plot_data_from_probcurves(
            curve_dat, t_vals, t_vals))

    # sum predicted proportions from different initial shapes
    plot_dat = plot_dat.groupby(["t", "shape"]).agg(
        P=("P", "sum"), lb=("lb", "sum"), ub=("ub", "sum")).reset_index()
    if dtype == "phylo":
        # add actual phylo time scale to phylo plot data for reference
        plot_dat["t_actual"] = [i for i in t_vals for _ in range(4)]

    return plot_dat


def get_plot_vals_alt(rates, init_ratio, dtype):
    """Get the predicted proportion of each shape at each step of the walk
    from the simulation or phylogeny rates"""

    # no. time points must match no. steps in sim data, phylo time scale is
    # different
    t_vals = None
    if dtype == "phylo":
        t_vals = np.linspace(0, PHYLO_XLIM, SIM_XLIM)
    elif dtype == "sim":
        t_vals = np.linspace(0, SIM_XLIM, SIM_XLIM+1)

    print(rates)
    # multiply predicted transition probability by initial proportion of shapes
    # to get the predicted proportion of each shape at each step
    curves = []  # store p for each t in t_vals
    for i in range(MC_ERR_SAMP):
        print(f"monte carlo {i}/{MC_ERR_SAMP}", end="\r")
        samp = rates.apply(  # choose random rate from each column
            lambda col: col.sample(1).values[0], axis=0)
        # recalculate diagonals
        samp["q00"] = -samp["q01"] - samp["q02"] - samp["q03"]
        samp["q11"] = -samp["q10"] - samp["q12"] - samp["q13"]
        samp["q22"] = -samp["q20"] - samp["q21"] - samp["q23"]
        samp["q33"] = -samp["q30"] - samp["q31"] - samp["q32"]
        q = np.array(samp.values).reshape(4, 4)
        curve_dat = []
        for t in t_vals:
            pt = linalg.expm(q * t) * init_ratio[:, None]
            pllt = np.zeros((4, 4))
            puut = np.zeros((4, 4))
            curve_dat.append([pt, pllt, puut])
        if dtype == "phylo":  # pseudo time scale for phylo
            plot_dat = pd.DataFrame(plot_data_from_probcurves(
                curve_dat, np.arange(SIM_XLIM), t_vals))
        elif dtype == "sim":  # sim time scale the same for plot and calc
            plot_dat = pd.DataFrame(
                plot_data_from_probcurves(curve_dat, t_vals, t_vals))
        # sum predicted proportions from different initial shapes
        plot_dat = plot_dat.groupby(["t", "shape"]).agg(
            P=("P", "sum")).reset_index()
        if dtype == "phylo":
            # add actual phylo time scale to phylo plot data for reference
            plot_dat["t_actual"] = [i for i in t_vals for _ in range(4)]
        plot_dat.insert(0, "mc_err_samp", i)
        curves.append(plot_dat)
    curves = pd.concat(curves, ignore_index=True)  # combine mc samples
    curves_sum = curves.groupby(["t", "shape"]).agg(  # median and CI
        P=("P", "median"), lb=("P", lambda x: np.percentile(x, LB)),
        ub=("P", lambda x: np.percentile(x, UB))).reset_index()
    print(curves_sum)
    return curves, curves_sum


def plot_sim_and_phylogeny_curves_new():
    """Plot the proportion of each shape category at each step of the
    walk from the simulation data against the predicted proportions from the
    simulation and phylogeny CTMCs. The scale for the phylogeny is different
    to the simulation scale."""

    al = 0.8  # alpha for lines
    af = 0.25  # alpha for fill

    #### Get phylo-rates ####
    phylo_summary_zun, raw_zun = get_phylo_rates(phylo=PHYLORATES)
    phylo_summary_jan, raw_jan = get_phylo_rates(phylo=PHYLORATES2)
    #### Get sim-rates ####
    sim_summary, raw_sim = get_sim_rates()
    #### Get sim timeseries data ####
    # tseries = get_timeseries_alt()
    tseries = get_timeseries()  # for error bar on timeseries
    tseries = tseries.rename(columns={"mean_prop": "prop"})
    init_ratio = np.array([  # get proportion of each shape at step=0 in data
        tseries[tseries["shape"] == "u"]["prop"].values[0],
        tseries[tseries["shape"] == "l"]["prop"].values[0],
        tseries[tseries["shape"] == "d"]["prop"].values[0],
        tseries[tseries["shape"] == "c"]["prop"].values[0]
    ])

    # get predicted proportions from phylo and sim rate matrices
    # sim_plot = get_plot_vals(sim_summary, init_ratio, "sim")
    sim_plot, sim_sum = get_plot_vals_alt(raw_sim, init_ratio, "sim")
    if SHOW_PHYLO:
        phylo_plot_z, ppz_sum = get_plot_vals_alt(raw_zun, init_ratio, "phylo")
        phylo_plot_j, ppj_sum = get_plot_vals_alt(raw_jan, init_ratio, "phylo")

    # Create subplots
    # plt.rcParams["font.family"] = "CMU Serif"
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(
        9, 6), sharex=not SHOW_PHYLO, sharey=False)

    def forward(x):  # map from simulation timescale to phylogeny timescale
        # x is the value on sim timescale, so multiply by ratio of phylo to
        # sim timescale
        return x * (PHYLO_XLIM / SIM_XLIM)

    def inverse(x):  # map from phylogeny time scale to simulation time scale
        return x * (SIM_XLIM / PHYLO_XLIM)

    for i, ax in enumerate(axs.flat):
        tseries_sub = tseries[tseries["shape"] == ORDER[i]]
        l_data, = ax.plot(tseries_sub["step"],  # plot timeseries
                          tseries_sub["prop"], c="C0", zorder=0)
        if {"lb", "ub"}.issubset(tseries_sub.columns):  # show error
            ax.fill_between(tseries_sub["step"], tseries_sub["lb"],
                            tseries_sub["ub"], alpha=af, color="C0", ec=None)
        sim_sub = sim_sum[sim_sum["shape"] == ORDER[i]]
        l_fit, = ax.plot(sim_sub["t"], sim_sub["P"],
                         c="C1", ls="--", alpha=0.5)
        ax.fill_between(sim_sub["t"], sim_sub["lb"], sim_sub["ub"],
                        alpha=af, color="C1", ec=None)
        if SHOW_PHYLO:
            secax = ax.secondary_xaxis('top', functions=(forward, inverse))
            ppz_sub = ppz_sum[ppz_sum["shape"] == ORDER[i]]
            l_phy_z, = ax.plot(
                ppz_sub["t"], ppz_sub["P"], c="C2", ls="--", alpha=al)
            ax.fill_between(ppz_sub["t"], ppz_sub["lb"], ppz_sub["ub"],
                            alpha=af, color="C2", ec=None)
            ppj_sub = ppj_sum[ppj_sum["shape"] == ORDER[i]]
            l_phy_j, = ax.plot(
                ppj_sub["t"], ppj_sub["P"], c="C3", ls="--", alpha=al)
            ax.fill_between(ppj_sub["t"], ppj_sub["lb"], ppj_sub["ub"],
                            alpha=af, color="C3", ec=None)
            secax.set_xlabel("Branch length (Myr)")
            ax.set_xlabel("Simulation step")
        ax.set_title(["Unlobed", "Lobed", "Dissected", "Compound"][i])
        ax.set_xlim(0, SIM_XLIM)
        ax.grid(alpha=0.3)
        # ax.set_ylim(0, 1)
        if YSCALE == "log":
            ax.set_yscale("log")
    if not SHOW_PHYLO:
        fig.supxlabel("Simulation step")
    fig.supylabel("Proportion")
    if SHOW_PHYLO:
        leg = fig.legend([l_data, l_fit, l_phy_z, l_phy_j],
                         ["Simulation Data", "Simulation CTMC",
                          "Phylogeny CTMC\nZuntini et al. (2024)",
                          "Phylogeny CTMC\nJanssens et al. (2021)"],
                         # ["Simulation Data", "Simulation CTMC", "Phylogeny CTMC"],
                         loc="center left", bbox_to_anchor=(0.75, 0.5))
    else:
        leg = fig.legend([l_data, l_fit],
                         ["Simulation Data", "Simulation CTMC"],
                         loc="center left", bbox_to_anchor=(0.75, 0.5))
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    plt.tight_layout()
    fig.subplots_adjust(right=0.75)  # make room for legend
    plt.savefig("sim_ctmc_fit.pdf", format="pdf", dpi=1200)
    plt.show()


def goodness_of_fit_prev():
    """Calculate goodness of fit of expected proportions from CTMC to observed
    proportions in random walk data."""

    #### Get sim timeseries data ####
    tseries = get_timeseries_alt()
    init_ratio = np.array([  # get proportion of each shape at step=0 in data
        tseries[tseries["shape"] == "u"]["prop"].values[0],
        tseries[tseries["shape"] == "l"]["prop"].values[0],
        tseries[tseries["shape"] == "d"]["prop"].values[0],
        tseries[tseries["shape"] == "c"]["prop"].values[0]
    ])
    print(tseries)

    # produce CTMC expected probabilities
    sim_summary, _ = get_sim_rates()
    st_vals = np.linspace(0, SIM_XLIM-1, SIM_XLIM)
    sim_curves = []
    q = np.array(sim_summary["mean_rate"].values).reshape(4, 4)
    ql = np.array(sim_summary["lb"].values).reshape(4, 4)
    qu = np.array(sim_summary["ub"].values).reshape(4, 4)
    # multiply predicted transition probability by initial proportion of shapes
    # to get the predicted proportion of each shape at each step
    for t in st_vals:
        pt = linalg.expm(q * t) * init_ratio[:, None]
        pllt = linalg.expm(ql * t) * init_ratio[:, None]
        puut = linalg.expm(qu * t) * init_ratio[:, None]
        sim_curves.append([pt, pllt, puut])
    sim_plot = pd.DataFrame(
        plot_data_from_probcurves(sim_curves, st_vals, st_vals))
    # sum predicted proportions from different initial shapes
    sim_plot = sim_plot.groupby(["t", "shape"]).agg(
        P=("P", "sum")).reset_index()

    # merge observed and expected data
    obs_exp = pd.merge(sim_plot, tseries, left_on=["t", "shape"],
                       right_on=["step", "shape"], how="left")
    obs_exp = obs_exp.rename(columns={"P": "p_exp", "prop": "p_obs"})
    obs_exp = obs_exp.drop(columns=["step", "shape_total", "total"])
    obs_exp["(o-e)^2/e"] = (obs_exp["p_obs"] -
                            obs_exp["p_exp"])**2 / obs_exp["p_exp"]
    gdness_fit = obs_exp["(o-e)^2/e"].sum()
    print(obs_exp)
    print(f"Goodness of fit: {gdness_fit}")
    # if p > 0.05, can't reject null hypothesis that obs and exp are from the
    # same distribution, thus CTMC is consistent with data
    xi = chisquare(obs_exp["p_obs"], obs_exp["p_exp"])
    print(xi)

    df = len(obs_exp) - 12  # no. independent cells in table - no. params
    # df = 12

    # Generate x values for the chi-squared distribution
    x = np.linspace(0, chi2.ppf(0.999, df), 1000)  # 99.9% of the distribution
    y = chi2.pdf(x, df)  # Calculate the chi-squared PDF
    crit = chi2.ppf(0.95, df)  # Critical value for alpha=0.05

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"Chi-squared Distribution (df={df})")
    # Highlight the critical region where H0 is rejected
    plt.fill_between(x, 0, y, where=(x >= crit), color="C1", alpha=0.5,
                     label=f"Critical Region (alpha=0.05, chi^2 > {crit:.2f})")
    # Add a vertical line for the observed chi-squared statistic
    plt.axvline(xi.statistic, color="C2", linestyle="--", lw=2,
                label=f"Observed chi^2 = {xi.statistic:.2f}"
                f", p-value = {xi.pvalue:.2f}")
    plt.xlabel("Chi-squared Statistic")
    plt.ylabel("Probability Density")
    plt.title("Chi-squared Distribution with Observed Statistic")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def goodness_of_fit_trans():
    """Test goodness of CTMC expected transitions frequencies to observed
    frequencies of transitions in different time intervals."""

    walks = pd.read_csv(SIMDATA)
    walks["prevshape"] = walks["shape"].shift(+1)
    walks.loc[walks["step"] == 0, "prevshape"] = walks["first_cat"]
    walks["transition"] = walks["prevshape"] + walks["shape"]

    n_wind = 4  # no. time windows for counting/estimating transtions
    wind_size = SIM_XLIM / n_wind
    walks["t_window"] = (walks["step"] // wind_size).astype(int)
    walks["w_start"] = walks["t_window"] * wind_size
    walks["w_end"] = (walks["t_window"] + 1) * wind_size
    # walks["is_last"] = (walks[["leafid", "walkid"]] != walks[[
    #     "leafid", "walkid"]].shift(-1)).any(axis=1)
    counts = walks.groupby(
        ["t_window", "w_start", "w_end", "transition"]
    ).size().unstack(fill_value=0).stack().reset_index(name="count_obs")
    counts["from"] = counts["transition"].str[0]
    counts["to"] = counts["transition"].str[1]
    # no. transitions from each state per window
    counts["n_from_w"] = counts.groupby(
        ["t_window", "from"])["count_obs"].transform("sum")

    print(counts.to_string()[0:10000])
    # get time per state to calculate expected no. transitions
    # steps_valid = walks[~walks["is_last"]].copy()
    # t_ht = steps_valid.groupby("t_window")["shape"].value_counts()
    # wind_states = t_ht.index.tolist()
    sim_summary = get_sim_rates()
    print(sim_summary)
    # print(t_ht)
    # st_vals = np.linspace(0, SIM_XLIM-1, SIM_XLIM)
    # sim_curves = []
    # print(sim_summary)
    q = np.array(sim_summary["mean_rate"].values).reshape(4, 4)
    print(q)
    pt = linalg.expm(q * wind_size)
    print(pt)
    pt = pd.DataFrame(pt, index=ORDER, columns=ORDER).stack().reset_index()
    pt.columns = ["from", "to", "p_w"]
    print(pt)

    obs_exp = pd.merge(counts, pt, on=["from", "to"], how="left")
    # see Kalbfleisch and Lawless (1985) page 868
    obs_exp["count_exp"] = obs_exp["p_w"] * obs_exp["n_from_w"]

    obs_exp["(o-e)^2/e"] = (obs_exp["count_obs"] -
                            obs_exp["count_exp"])**2 / obs_exp["count_exp"]
    print(obs_exp.to_string()[0:10000])

    # for i in ORDER:
    #     for j in ORDER:
    #         # get expected trans prob for window duration
    #         counts["p_w"] = counts[counts["from"] == i]
    tee = 1.90-1.10
    pt = linalg.expm(np.array([
        [-0.136, 0.136, 0],
        [0, -2.28, 2.28],
        [0, 0.47, -0.47],
    ]) * tee)
    print(pt)
    # q = pd.DataFrame(np.array(sim_summary["mean_rate"].values).reshape(4, 4),
    #                  index=ORDER, columns=ORDER)
    xi = chisquare(obs_exp["count_obs"], obs_exp["count_exp"])
    print(xi)

    # exp_counts = pd.DataFrame([
    #     {
    #         "t_window": w,
    #         "from": i,
    #         "to": j,
    #         "exp_trans": q.loc[i, j] * t_ht[w][i]
    #     }
    #     for w in t_ht.index.get_level_values(
    #         "t_window") for i in ORDER for j in ORDER if i != j
    # ])
    # print(exp_counts)

    # merge observed and expected data
    # obs_exp = pd.merge(sim_plot, tseries, left_on=["t", "shape"],
    #                    right_on=["step", "shape"], how="left")
    # print(obs_exp)


if __name__ == "__main__":
    print("Parameters")
    print(f"Phylogeny rates: {PHYLORATES}")
    print(f"Simulation rates: {SIMRATES}")
    print(f"Simulation data: {SIMDATA}")
    print(f"Monte Carlo error sample size: {MC_ERR_SAMP}")
    print(f"T_STAT: {T_STAT}")
    print(f"Simulation rates method: {SIMRATES_METHOD}")
    print(f"Equal initial shapes: {EQ_INIT}")
    print(f"Lower bound: {LB}")
    print(f"Upper bound: {UB}")
    print(f"Phylogeny xlim: {PHYLO_XLIM}")
    print(f"Simulation xlim: {SIM_XLIM}")
    print(f"Simulation xon: {SIM_XON}")
    print(f"reset first cat: {RESET_FIRST_CAT}")
    print(f"Y scale: {YSCALE}")
    print(f"Show phylogeny: {SHOW_PHYLO}")
    print(f"Order: {ORDER}")
    print(f"Variable: {VAR}")
    print(f"Excluded leafids: {EXCL if EQ_INIT else None}")
    print(f"Verbose: {V}")
    print(f"Plot type: {PLOT}\n")
    if PLOT == 0:
        plot_sim_and_phylogeny_curves()
    if PLOT == 1:
        plot_sim_and_phylogeny_curves_nouncert()
    if PLOT == 2:
        plot_sim_and_phylogeny_curves_new()
    if PLOT == 3:
        goodness_of_fit_trans()
        # goodness_of_fit_prev()
