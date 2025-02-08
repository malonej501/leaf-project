import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
import numpy as np
from io import StringIO
import multiprocessing
import subprocess
import os
import shutil
import sys

# set default parameter values
ML_data = "ML_6_genus"
burnin = 100000
datasets = [
    "geeta_phylo_geeta_class",
    "jan_genus_phylo_nat_26-09-24_class",
    "zun_genus_phylo_nat_26-09-24_class"
    ]
log_exclude = [ # columns to exclude from the log file
    # "Iteration",
    "Tree No",
    "Model string",
    "Unnamed: 19",
    "Unnamed: 22",
]

rate_map = {
    "q01" : "ul",
    "q02" : "ud",
    "q03" : "uc",
    "q10" : "lu",
    "q12" : "ld",
    "q13" : "lc",
    "q20" : "du",
    "q21" : "dl",
    "q23" : "dc",
    "q30" : "cu",
    "q31" : "cl",
    "q32" : "cd",
}
# plt.rcParams["font.family"] = "CMU Serif"
# plt.rcParams["font.size"] = 12

# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24/mcmc/prior_0-1/img_labels_unambig_full_21-1-24.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24/mcmc/prior_0-1_run2/img_labels_unambig_full_21-1-24.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24/mcmc/prior_0-1_run4_full/img_labels_unambig_full_21-1-24.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24_95_each/mcmc/original/jan_phylo_nat_class_21-01-24_95_each.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24_cenrich/mcmc/prior_0-1_run2/jan_phylo_nat_class_21-01-24_cenrich_sub.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24_shuff/mcmc/prior_0-1_run2/jan_phylo_nat_class_21-01-24.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24_95_each/mcmc/prior_exp1_run1/jan_phylo_nat_class_21-01-24_95_each.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24_95_each/mcmc/prior_0-0.05_run1/jan_phylo_nat_class_21-01-24_95_each.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24_95_each/mcmc/prior_0-0.05_resallq01_run1/jan_phylo_nat_class_21-01-24_95_each.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24_95_each/mcmc/prior_exp1_run1/jan_phylo_nat_class_21-01-24_95_each.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24_95_each/mcmc/prior_0-0.02_run1/jan_phylo_nat_class_21-01-24_95_each.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24_cenrich/mcmc/prior_0-0.02_run1/jan_phylo_nat_class_21-01-24_cenrich_sub.txt.Log.txt"
# file = "BayesTraitsV4.0.0-Linux/Janssens/sample_eud_21-1-24_shuff/mcmc/original/jan_phylo_nat_class_21-01-24_shuff.txt.Log.txt"

# file = "BayesTraitsV4.1.2-Linux/Geeta/geeta_23-04-24/mcmc/prior_0-1_run1/561AngLf09_D.txt.Log.txt"
# file = "BayesTraitsV4.1.2-Linux/Geeta/naturalis_23-04-24/mcmc/original/Naturalis_img_labels_unambig_full_21-1-24_Geeta_sub.txt.Log.txt"
# file = "BayesTraitsV4.1.2-Linux/Janssens/geeta_sub_21-02-24/mcmc/prior_exp0.1_run1/Geeta_sub_species.txt.Log.txt"
# file = "BayesTraitsV4.1.2-Linux/Janssens/sample_eud_21-1-24/mcmc/prior_exp10_rj_run1/img_labels_unambig_full_21-1-24.txt.Log.txt"
# file = "BayesTraitsV4.1.2-Linux/Soltis_2011/Geeta_04-03-24/mcmc/prior_0-1_run1/561AngLf09_D_soltis2011_T31199_sub.txt.Log.txt"
# file = "BayesTraitsV4.1.2-Linux/Soltis_2011/Naturalis_23-04-24/mcmc/prior_exp10_rj_run1/Naturalis_img_labels_unambig_full_21-1-24_soltis2011_T31199_sub.txt.Log.txt"
# file = "BayesTraitsV4.1.2-Linux/Zuntini_2024/geeta_30-06-24/mcmc/prior_0-1_run1/561AngLf09_D_zuntini2024_sub.txt.Log.txt"
# file = "BayesTraitsV4.1.2-Linux/Zuntini_2024/naturalis_30-06-24/mcmc/original/Naturalis_img_labels_unambig_full_21-1-24_zuntini2024_sub.txt.Log.txt"

# files = ["Geeta/geeta_23-04-24/mcmc/prior_0-1_run1/561AngLf09_D.txt.Log.txt",
#          "Geeta/naturalis_23-04-24/mcmc/original/Naturalis_img_labels_unambig_full_21-1-24_Geeta_sub.txt.Log.txt",
#          "Janssens/geeta_sub_21-02-24/mcmc/prior_exp0.1_run1/Geeta_sub_species.txt.Log.txt",
#          "Janssens/sample_eud_21-1-24/mcmc/prior_exp10_rj_run1/img_labels_unambig_full_21-1-24.txt.Log.txt",
#          "Soltis_2011/Geeta_04-03-24/mcmc/prior_0-1_run1/561AngLf09_D_soltis2011_T31199_sub.txt.Log.txt",
#          "Soltis_2011/Naturalis_23-04-24/mcmc/prior_exp10_rj_run1/Naturalis_img_labels_unambig_full_21-1-24_soltis2011_T31199_sub.txt.Log.txt",
#          "Zuntini_2024/geeta_30-06-24/mcmc/prior_0-1_run1/561AngLf09_D_zuntini2024_sub.txt.Log.txt",
#          "Zuntini_2024/naturalis_30-06-24/mcmc/original/Naturalis_img_labels_unambig_full_21-1-24_zuntini2024_sub.txt.Log.txt"]


# data = (
#     ("Geeta/geeta_23-04-24/561Data08.tre", "Geeta/geeta_23-04-24/561AngLf09_D.txt"),
#     (
#         "Geeta/naturalis_23-04-24/Geeta_naturalis_sub.tre",
#         "Geeta/naturalis_23-04-24/Naturalis_img_labels_unambig_full_21-1-24_Geeta_sub.txt",
#     ),
#     (
#         "Zuntini_2024/geeta_30-06-24/zuntini_geetasub.tre",
#         "Zuntini_2024/geeta_30-06-24/561AngLf09_D_zuntini2024_sub.txt",
#     ),
#     (
#         "Zuntini_2024/naturalis_30-06-24/zuntini_naturalis_sub.tre",
#         "Zuntini_2024/naturalis_30-06-24/Naturalis_img_labels_unambig_full_21-1-24_zuntini2024_sub.txt",
#     ),
#     (
#         "Soltis_2011/Geeta_04-03-24/Soltis_T31199_geetasub.tre",
#         "Soltis_2011/Geeta_04-03-24/561AngLf09_D_soltis2011_T31199_sub.txt",
#     ),
#     (
#         "Soltis_2011/Naturalis_23-04-24/Soltis_T31199_naturalis_sub.tre",
#         "Soltis_2011/Naturalis_23-04-24/Naturalis_img_labels_unambig_full_21-1-24_soltis2011_T31199_sub.txt",
#     ),
#     (
#         "Janssens/sample_eud_21-1-24/Naturalis_sample_Janssens_intersect_labelled_21-01-24.tre",
#         "Janssens/sample_eud_21-1-24/img_labels_unambig_full_21-1-24.txt",
#     ),
#     (
#         "Janssens/geeta_sub_21-02-24/Geeta_sub_Janssens.tre",
#         "Janssens/geeta_sub_21-02-24/Geeta_sub_species.txt",
#     ),
# )


def import_data():
    trees = []
    for file in os.listdir("data"):
        if file.endswith(("class.tre", "class.txt")):
            file_path = os.path.join("data", file)
            base = file_path.split(".")[0]
            if base not in trees:
                trees.append(base)

    files = sorted(tuple((f"{tree}.tre", f"{tree}.txt") for tree in trees))

    return files


def get_log(file):
    log = None
    with open(file, "r") as fh:
        lines = fh.readlines()

        # find the start of the log table
        start_index = None
        for i, line in enumerate(lines):
            if "Tree No" in line:
                start_index = i
                break
        fh.seek(0)
        log = pd.read_csv(fh, sep="\t", skiprows=start_index)

    return log


def get_ML_rates(directory):
    logs = []
    for file in os.listdir(directory):
        if file.endswith(".Log.txt"):
            data_name = file.rsplit(".")[0]
            filepath = os.path.join(directory, file)
            log = get_log(filepath)
            log_mean = log.mean()
            log_mean_df = pd.DataFrame([log_mean], columns=log.columns)
            log_mean_df.insert(0, "dataset", data_name)
            # log_mean_df.to_csv(directory + f"/{data_name}_rates_ml.csv", index=False)
            logs.append(log_mean_df)

    ML_rates = pd.concat(logs, axis=0, ignore_index=True)
    ML_rates.sort_values(by="dataset", inplace=True)
    if "Tree No" in ML_rates.columns:
        ML_rates.drop("Tree No", axis=1, inplace=True)
    if "Unnamed: 18" in ML_rates.columns:
        ML_rates.drop("Unnamed: 18", axis=1, inplace=True)
    ML_rates.to_csv(directory + "/mean_rates_all.csv", index=False)

    return ML_rates


def get_marginal_likelihood(run_name):

    marglhs = []
    for file in os.listdir("data"):
        if file.endswith(".Stones.txt"):
            data_name = file.rsplit(".")[0]
            filepath = os.path.join("data", file)
            with open(filepath, "r") as fh:
                lines = fh.readlines()
                # find marginal likelihood value
                for line in lines:
                    if "Log marginal likelihood" in line:
                        marglh_val = float(line.rsplit("\t")[1])
                        marglh = pd.DataFrame(
                            {
                                "dataset": [data_name],
                                "log_marginal_likelihood": [marglh_val],
                            }
                        )
                        marglhs.append(marglh)
    marglhs_df = pd.concat(marglhs, axis=0, ignore_index=True)
    marglhs_df.sort_values(by="dataset", inplace=True)
    marglhs_df.reset_index(inplace=True, drop=True)

    marglhs_df.to_csv(f"data/{run_name}_log_marginal_likelihoods_all.csv", index=False)
    return marglhs_df


def plot_trace_full(file, run_name, ML_data, export):
    save_fig_path = file.rsplit("/", 1)[0]
    log_file_name = file.rsplit("/", 1)[1]
    data_name = log_file_name.rsplit(".")[0]

    # Get ML data for comparison
    ML_data = pd.read_csv(f"data/{ML_data}/mean_rates_all.csv")
    # ML_data = pd.read_csv("data/ML_scaletrees0.001_1/mean_rates_all.csv")
    ML = ML_data[ML_data["dataset"] == data_name].reset_index()

    log = get_log(file)
    # summary statistics
    print(log.describe())

    # export just the cleaned posteriors from the log file
    log_no_burnin = log.loc[log["Iteration"] >= burnin]
    # rates = log_no_burnin[
    #     [
    #         "q01",
    #         "q02",
    #         "q03",
    #         "q10",
    #         "q12",
    #         "q13",
    #         "q20",
    #         "q21",
    #         "q23",
    #         "q30",
    #         "q31",
    #         "q32",
    #     ]
    # ]
    rates = log_no_burnin.filter(like="q", axis=1)
    rates.to_csv(save_fig_path + f"/{data_name}_{run_name}.csv", index=False)

    # plt.plot(log["Iteration"], log["Lh"])
    # plt.show()
    # plt.clear()
    print(log)
    num_vars = len(
        [
            var
            for var in log.columns
            if var
            not in log_exclude
        ]
    )

    fig, axes = plt.subplots(
        nrows=4,
        ncols=5,
        figsize=(15, 12),
        sharex=True,
    )

    # If only one subplot, `axes` is not a list, so we handle that case
    if num_vars == 1:
        axes = [axes]
    axes = axes.flatten()

    counter = 0
    for var in log.columns:
        if var not in log_exclude:
            axes[counter].plot(log["Iteration"], log[var])
            axes[counter].set_ylabel(var)
            axes[counter].grid(True)
            # Add ML values
            if var in ML.columns:
                axes[counter].axhline(
                    y=ML.loc[0, var], linestyle="--", color="C1", label="ML"
                )
            counter += 1

    fig.text(0.5, 0.01, "Iteration", ha="center")
    fig.suptitle(data_name)
    fig.tight_layout()
    # Adding a legend

    if export:
        plt.savefig(save_fig_path + f"/{run_name}_{data_name}_trace_single.pdf", format="pdf")

    # plt.show()
    return fig


def plot_trace(file, run_name, ML_data, export):
    save_fig_path = file.rsplit("/", 1)[0]
    log_file_name = file.rsplit("/", 1)[1]
    data_name = log_file_name.rsplit(".")[0]

    # Get ML data for comparison
    ML_data = pd.read_csv(f"data/{ML_data}/mean_rates_all.csv")
    # ML_data = pd.read_csv("data/ML_scaletrees0.001_1/mean_rates_all.csv")
    ML = ML_data[ML_data["dataset"] == data_name].reset_index()

    log = get_log(file)
    log_exclude.extend(["Lh", "Root P(0)", "Root P(1)", "Root P(2)", "Root P(3)"]) # additional log columns to exclude
    log_filt = log.drop(columns=log_exclude, errors="ignore") # ignore if not all columns in log_exclude are present
    fig, axes = plt.subplots(
        nrows=4,
        ncols=4,
        figsize=(10, 7),
        sharex=True,
        sharey=True
    )

    idx = 1 # start from 1 to skip the iteration column
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i == j:
                ax.axis("off")
                continue
            ax.plot(log_filt["Iteration"], log_filt.iloc[:, idx], c="C0")
            ax.set_ylabel(rate_map.get(log_filt.columns[idx]))
            ax.tick_params(axis="y", which="both", labelleft=True)
            ax.axhline(ML.loc[0, log_filt.columns[idx]], linestyle="--", color="C1", label="ML")   
            # ax.set_ylim(0, 1)
            idx += 1

    fig.text(0.5, 0.01, "Iteration", ha="center")
    fig.tight_layout()
    # Adding a legend

    if export:
        plt.savefig(save_fig_path + f"/{run_name}_{data_name}_trace_single.pdf", format="pdf")

    return fig

def plot_marglhs(run_name):
    marglhs = get_marginal_likelihood(run_name)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        marglhs["dataset"], marglhs["log_marginal_likelihood"], color="skyblue"
    )

    # Adding labels on top of the bars
    for bar, label in zip(bars, marglhs["log_marginal_likelihood"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,  # Adjusted for better visibility
            round(label, 2),
            ha="center",
            va="bottom",
        )
    ax.set_xticklabels(marglhs["dataset"], rotation=45)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Log Marginal Likelihood")
    plt.tight_layout()

    return fig


def run_BayesTraits(tree, labels):
    print(tree, labels)
    result = subprocess.run(
        f"./start.sh {tree} {labels}", shell=True, capture_output=True, text=True
    )

    return result.stdout


def run_select_trees(data, run_name: str, method: str, ML_data: str):
    run_dir = os.path.join("data", run_name)
    os.makedirs(run_dir, exist_ok=True)

    processes = []
    for tree, labels in data:
        process = multiprocessing.Process(target=run_BayesTraits, args=(tree, labels))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    if method == "MCMC":
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"data/{run_name}_trace.pdf")
        for _, label in data:
            logfilepath = label + ".Log.txt"
            fig = plot_trace_full(logfilepath, run_name, ML_data, export=False)
            pdf.savefig(fig)

        if os.path.exists(datasets[0] + ".txt.Stones.txt"):
            fig = plot_marglhs(run_name)
            pdf.savefig(fig)
        pdf.close()

    for file in os.listdir("data"):
        if not file.endswith(("class.tre", "class.txt")):
            source_file = os.path.join("data", file)
            destination_file = os.path.join(run_dir, file)
            if os.path.isfile(source_file):
                shutil.move(source_file, destination_file)
                print(f"Moved: {source_file} to {destination_file}")


# files = import_data()
# run_all_trees(files, "uniform1-100")
# run_select_trees(
#     [
#         "jan_phylo_nat_class",
#         "jan_phylo_geeta_class",
#         "zuntini_phylo_nat_class",
#         "zuntini_phylo_geeta_class",
#     ],
#     "ML_scaletrees0.001_1",
#     "ML"
# )
# run_select_trees(
#     datasets=["ALL"],
#     run_name="ML_red_1",
#     method="ML",
#     ML_data="None",
# )
# run_select_trees(
#     datasets=["ALL"],
#     run_name="ML_jan_zun_equal_genus_5",
#     method="ML",
#     ML_data="None",
# )

# run_select_trees(
#     datasets=["ALL"],
#     run_name="ML_7_species_genus",
#     method="ML",
#     ML_data="None",
# )
# run_select_trees(["ALL"], "uniform0-0.1_species_genus", "MCMC", "ML_7_species_genus")
# run_select_trees(["ALL"], "uniform0-100_res_1", "MCMC", "ML_res_1")
# run_select_trees(["ALL"], "exp1_1", "MCMC", "ML_3")


# run_select_trees(
#     [
#         "zuntini_phylo_nat_class_10-09-24_class",
#         "zuntini_phylo_nat_class_10-09-24_genera_class",
#     ],
#     "uniform0-1_rj_2",
#     "MCMC",
#     "ML_3",
# )
# get_ML_rates("data/ML_7_species_genus")
# get_marginal_likelihood("data/uniform0-0.1_unres_1")

def print_help():
    help_message = """
    Usage: python3 inference.py [options]

    Options:
        -h              Show this help messahe and exit.
        -id [run id]    The name given to the mcmc/mle run.
        -d  [datasets]  Pass datasets you want to run inference on, separated by ",":
                        e.g. "ALL" or "jan_phylo_nat_class-geeta_phylo_geeta_class"
        -ml [data]      Pass the ML dataset you want to compare the mcmc posteriors to.
                        e.g. "ML_7_species_genus"
        -f  [function]  Pass function you want to perform:
                        0   ...run mcmc inference
                        1   ...run mle inference
                        2   ...plot individual mcmc trace from file
    """
    print(help_message)

if __name__ == "__main__":
    args = sys.argv[1:]
    if "-h" in args:
        print_help()
    else:
        if "-id" in args:
            run_id = str(args[args.index("-id") + 1])
        if "-d" in args:
            datasets = args[args.index("-d") + 1].split(",")
            if datasets == ["ALL"]:
                data = import_data()
            else:
                data = sorted(
                    tuple(
                        (f"data/{dataset}.tre", f"data/{dataset}.txt") for dataset in datasets
                    )
                )
        if "-ml" in args:
            ML_data = str(args[args.index("-ml") + 1])
        if "-f" in args:
            func = int(args[args.index("-f") + 1])
            if func  == 0:
                method = "MCMC"
                run_select_trees(data, run_id, method, ML_data)
            if func == 1:
                method = "MLE"
                run_select_trees(data, run_id, method, ML_data)
            if func == 2:
                for _, label in data:
                    # This assumes the log file is now in the run_id directory, rather than /data
                    logfilename = label.replace("data/", "") + ".Log.txt"
                    logfilepath = f"data/{run_id}/{logfilename}"
                    plot_trace(logfilepath, run_id, ML_data, export=True)
                # plot_marglhs(run_id)