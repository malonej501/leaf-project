from matplotlib import pyplot as plt
from scipy import linalg
from PIL import Image
import seaborn as sns
import os
import pandas as pd
import numpy as np

plot = 1 # type of plot to produce, 0-3rows with error bars, 1-two rows with mean model
phylorates = "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1"
simrates = "MUT2_mcmc_11-12-24"
simdata = "MUT2.2_trajectories_shape.csv"
mode= 0 # the statistic used for plotting the timeseries data 0=mean proportion, 1=proportion
lb = 0#5 #5
ub = 0#95 #95
phylo_xlim = 200 
sim_xlim = 80#60

sns.set_palette("colorblind")
order = ["u", "l", "d", "c"]
var = "mean_prop" if mode == 0 else "prop"

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

def get_phylo_rates():
    
    phylo_dir = "../phylogeny/rates/uniform_1010000steps"
    phylo_rates_list = []

    for filename in os.listdir(phylo_dir):
        if filename.endswith(".csv"):
            path = os.path.join(phylo_dir, filename)
            df = pd.read_csv(path)
            df["phylo-class"] = filename[:-4]
            phylo_rates_list.append(df)

    phylo_rates = pd.concat(phylo_rates_list, ignore_index=True)
    # Choose phylogeny for curves
    phylo_rates = phylo_rates[phylo_rates["phylo-class"] == phylorates]
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

    return phylo_summary

def get_sim_rates():
    sim_rates = (
        pd.read_csv(f"markov_fitter_reports/emcee/{simrates}/posteriors_{simrates}.csv")
        # pd.read_csv(f"markov_fitter_reports/emcee//leaf_uncert_posteriors_MUT2.csv")
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
    return sim_summary

def get_timeseries():
    """
    Get time series data from raw shape trajectories. Mode controls the statistic that is used to summaries the timeseries.
    mode        0 = mean across leafids of the proportion within leafids
                1 = proportion across all leafids 
    """
    concat = pd.read_csv(simdata)
    print(concat)

    # get the number of leaves in each initial group
    shape_counts = concat.groupby(["leafid","first_cat"]).size().reset_index()
    init_shape_counts = shape_counts["first_cat"].value_counts().reset_index()
    init_shape_counts = init_shape_counts.set_index("first_cat")["count"].to_dict()
 
    if mode == 0:
        # total of each shape per step for each leafid
        timeseries = (
            concat.groupby(["leafid", "first_cat", "step", "shape"])
            .size()
            .reset_index(name="shape_total")
        )
        print(timeseries)

        # fill in transitions that didn't occur
        all_transitions = pd.MultiIndex.from_product([timeseries["leafid"].unique(),timeseries["step"].unique(), {"u","l","d","c"}], names=["leafid","step","shape"])
        timeseries = timeseries.set_index(["leafid","step","shape"]).reindex(all_transitions, fill_value=None).reset_index()
        timeseries["first_cat"] = timeseries.groupby(["leafid"])["first_cat"].transform("first") # fill empty first-cat values with the first non nan first_cat value for each leafid
        timeseries = timeseries.fillna(0)
        print(timeseries)

        # no. active walks per step for each leafid
        timeseries_total = (
            timeseries.groupby(["leafid", "first_cat", "step"])
            .agg(no_active_walks=("shape_total", "sum"))
            .reset_index()
        )
        print(timeseries_total)

        # proportion of active walks in each shape category for each leafid
        timeseries = timeseries.merge(timeseries_total, on=["leafid", "first_cat", "step"])
        print(timeseries)
        timeseries["proportion"] = (
            timeseries["shape_total"] / timeseries["no_active_walks"]
        )
        print(timeseries)

        # mean proportion of active walks in each shape category for all leaves in each first_cat
        timeseries = (
            timeseries.groupby(["first_cat", "step", "shape"])
            .agg(mean_prop=("proportion", "mean"), 
                sterr=("proportion", "sem"),
                n=("proportion", "size"),
                total=("proportion", "sum"))
            .reset_index()
        )

    elif mode == 1:
        timeseries = concat.groupby(["first_cat","step","shape"]).size().reset_index(name="shape_total")
        print(timeseries)
        # no. active walks per step
        timeseries_total = timeseries.groupby(["first_cat","step"]).agg(no_active_walks=("shape_total","sum")).reset_index()
        print(timeseries_total)
        # proportion of active walks in each shape category
        timeseries = timeseries.merge(timeseries_total, on=["first_cat","step"])
        timeseries["prop"] = timeseries["shape_total"] / timeseries["no_active_walks"]
        timeseries["sterr"] = 0


    timeseries["lb"] = timeseries[var] - 1.96 * timeseries["sterr"]
    timeseries["ub"] = timeseries[var] + 1.96 * timeseries["sterr"]

    # add initial state to the timeseries
    timeseries["step"] = timeseries["step"] + 1
    for i in order:
        for j in order:
            if i == j:
                timeseries.loc[-1] = {
                    "first_cat": i,
                    "step": 0,
                    "shape": i,
                    var: 1,
                    "sterr": 0,
                    "lb": 1,
                    "ub": 1,
                }
            else:
                timeseries.loc[-1] = {
                    "first_cat": i,
                    "step": 0,
                    "shape": j,
                    var: 0,
                    "sterr": 0,
                    "lb": 0,
                    "ub": 0,
                }
            timeseries.index = timeseries.index + 1
            timeseries = timeseries.sort_index()
    print(timeseries)
    timeseries.to_csv("timeseries.csv")
    return timeseries

def plot_sim_and_phylogeny_curves():  

    #### Get phylo-rates ####
    phylo_summary = get_phylo_rates()
    #### Get sim-rates ####
    sim_summary = get_sim_rates()
    #### Get sim timeseries data ####
    timeseries = get_timeseries()
    
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
                    cat_data = cat_data.rename(columns={var: "P", "step": "t"})
                    ax.set_title(order_full[idx])
                    ax.set_xlim(0, 60)
                    ax.set_xticks(np.arange(0, 61, 20))
                    ax.set_xlabel("Step")
                    if idx == 0:
                        ax.set_ylabel("Mean Prop.")
                  
                if i == 2:  # simulation ctmc in centre column
                    cat_data = sim_plot[sim_plot["first_cat"] == cat]
                    ax.set_xlabel("Step")
                    ax.set_xlim(0, 60)
                    ax.set_xticks(np.arange(0, 61, 20))
                    if idx == 0:
                        ax.set_ylabel("P")

                if i == 3:
                    ax.axis("off")

                if i == 4:  # phylogeny data on the right
                    cat_data = phylo_plot[phylo_plot["first_cat"] == cat]
                    ax.set_xlim(0, phylo_xlim)
                    ax.set_xlabel("Branch length (Myr)")
                    if idx == 0:
                        ax.set_ylabel("P")
                if i != 3:
                    for s, shape in enumerate(order):
                        shape_data = cat_data[cat_data["shape"] == shape]
                        (line,) = ax.plot(
                            shape_data["t"],
                            shape_data["P"],
                            label=shape,
                            c=sns.color_palette("colorblind")[s],
                            linestyle="-",
                        )
                        ax.fill_between(
                            shape_data["t"],
                            shape_data["lb"],
                            shape_data["ub"],
                            alpha=0.2,
                        )
                        lines.append(line)

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
        ax.imshow(icon_imgs[j])
        ax.set_xlim(0 + (img_width * (sf - 1)), img_width - (img_width * (sf - 1)))
        ax.set_ylim(img_height, -(img_height / sf))
    for idx, i in enumerate([1, 2, 4]):
        ax = axs[i, 0]

        labs = [
            "Simulation Data\nMUT2",
            "Simulation CTMC\nMUT2",
            "Phylogeny CTMC\nZuntini et al. (2024)",
        ]

        ax.text(
            0.2,
            0.5,
            labs[idx],
            ha="center",
            va="center",
        )

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

def plot_sim_and_phylogeny_curves_nouncert():

    # lb = 0
    # ub = 0

    #### Get phylo-rates ####
    phylo_summary = get_phylo_rates()
    #### Get sim-rates ####
    sim_summary = get_sim_rates()
    #### Get sim timeseries data ####
    timeseries = get_timeseries()

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

    # Create subplots
    plt.rcParams["font.family"] = "CMU Serif"
    fig, axs = plt.subplots(
        nrows=4,
        ncols=5,
        figsize=(9, 6),
        # sharey=True,
        gridspec_kw={"height_ratios": [3, 3, 1, 3]},
    )

    lines = []
    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]

    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            idx = j - 1
            sim_cat_data = pd.DataFrame()
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
                    cat_data = cat_data.rename(columns={var: "P", "step": "t"})
                    ax.set_title(order_full[idx])
                    ax.set_xlim(0, sim_xlim)
                    ax.set_xticks(np.arange(0, sim_xlim + 1, 20))
                    ax.set_xlabel("Step")
                    if idx == 0:
                        ax.set_ylabel("P")
                    sim_cat_data = sim_plot[sim_plot["first_cat"] == cat]
                    sim_cat_data = sim_cat_data.rename(columns={var: "P", "step": "t"})
                if i == 2:
                    ax.axis("off")
                if i == 3:  # phylogeny data on the right
                    cat_data = phylo_plot[phylo_plot["first_cat"] == cat]
                    ax.set_xlim(0, phylo_xlim)
                    ax.set_xlabel("Branch length (Myr)")
                    if idx == 0:
                        ax.set_ylabel("P")
                if i != 2:
                    for s, shape in enumerate(order):
                        shape_data = cat_data[cat_data["shape"] == shape]
                        (line,) = ax.plot(
                            shape_data["t"],
                            shape_data["P"],
                            label=shape,
                            c=sns.color_palette("colorblind")[s],
                            linestyle="--" if i == 1 else "-", 
                        )
                        ax.fill_between(
                            shape_data["t"],
                            shape_data["lb"],
                            shape_data["ub"],
                            alpha=0.2,
                        )
                        if not sim_cat_data.empty:
                            shape_data = sim_cat_data[sim_cat_data["shape"] == shape]
                            (line,) = ax.plot(
                                shape_data["t"],
                                shape_data["P"],
                                label=shape,
                                c=sns.color_palette("colorblind")[s],
                                linestyle="-",
                            )
                            lines.append(line)
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
        ax.imshow(icon_imgs[j])
        ax.set_xlim(0 + (img_width * (sf - 1)), img_width - (img_width * (sf - 1)))
        ax.set_ylim(img_height, -(img_height / sf))
    for idx, i in enumerate([1, 3]):
        ax = axs[i, 0]

        labs = [
            "Simulation Data\nand CTMC (MUT2)",
            "Phylogeny CTMC\nZuntini et al. (2024)",
        ]

        ax.text(
            0.2,
            0.5,
            labs[idx],
            ha="center",
            va="center",
        )
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


if __name__ == "__main__":
    if plot == 0:
        plot_sim_and_phylogeny_curves()
    if plot == 1:
        plot_sim_and_phylogeny_curves_nouncert()