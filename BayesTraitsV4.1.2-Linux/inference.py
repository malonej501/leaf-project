import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import StringIO
import multiprocessing
import subprocess

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


data = (
    ("Geeta/geeta_23-04-24/561Data08.tre", "Geeta/geeta_23-04-24/561AngLf09_D.txt"),
    (
        "Geeta/naturalis_23-04-24/Geeta_naturalis_sub.tre",
        "Geeta/naturalis_23-04-24/Naturalis_img_labels_unambig_full_21-1-24_Geeta_sub.txt",
    ),
    (
        "Zuntini_2024/geeta_30-06-24/zuntini_geetasub.tre",
        "Zuntini_2024/geeta_30-06-24/561AngLf09_D_zuntini2024_sub.txt",
    ),
    (
        "Zuntini_2024/naturalis_30-06-24/zuntini_naturalis_sub.tre",
        "Zuntini_2024/naturalis_30-06-24/Naturalis_img_labels_unambig_full_21-1-24_zuntini2024_sub.txt",
    ),
    (
        "Soltis_2011/Geeta_04-03-24/Soltis_T31199_geetasub.tre",
        "Soltis_2011/Geeta_04-03-24/561AngLf09_D_soltis2011_T31199_sub.txt",
    ),
    (
        "Soltis_2011/Naturalis_23-04-24/Soltis_T31199_naturalis_sub.tre",
        "Soltis_2011/Naturalis_23-04-24/Naturalis_img_labels_unambig_full_21-1-24_soltis2011_T31199_sub.txt",
    ),
    (
        "Janssens/sample_eud_21-1-24/Naturalis_sample_Janssens_intersect_labelled_21-01-24.tre",
        "Janssens/sample_eud_21-1-24/img_labels_unambig_full_21-1-24.txt",
    ),
    (
        "Janssens/geeta_sub_21-02-24/Geeta_sub_Janssens.tre",
        "Janssens/geeta_sub_21-02-24/Geeta_sub_species.txt",
    ),
)


def plot_trace(file):
    save_fig_path = file.rsplit("/", 1)[0]

    # log = pd.read_csv(file, sep="\t", skiprows=57)
    # # log = pd.read_csv(file, sep="\t", skiprows=46)

    log = None
    with open(file, "r") as fh:
        lines = fh.readlines()

        # find the start of the log table
        start_index = None
        for i, line in enumerate(lines):
            print(i)
            print(line)
            if "Tree No" in line:
                start_index = i
                break
        print(start_index)
        fh.seek(0)
        log = pd.read_csv(fh, sep="\t", skiprows=start_index)

    # summary statistics
    print(log.describe())

    # export just the cleaned posteriors from the log file
    log_no_burnin = log.loc[log["Iteration"] >= 100000]
    rates = log_no_burnin[
        [
            "q01",
            "q02",
            "q03",
            "q10",
            "q12",
            "q13",
            "q20",
            "q21",
            "q23",
            "q30",
            "q31",
            "q32",
        ]
    ]
    rates.to_csv(save_fig_path + "/rates.csv")

    # plt.plot(log["Iteration"], log["Lh"])
    # plt.show()
    # plt.clear()
    print(log)
    num_vars = len(
        [
            var
            for var in log.columns
            if var
            not in [
                "Iteration",
                "Tree No",
                "Model string",
                "Unnamed: 19",
                "Unnamed: 22",
            ]
        ]
    )

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 12), sharex=True)

    # If only one subplot, `axes` is not a list, so we handle that case
    if num_vars == 1:
        axes = [axes]
    axes = axes.flatten()

    counter = 0
    for var in log.columns:
        if var not in [
            "Iteration",
            "Tree No",
            "Model string",
            "Unnamed: 19",
            "Unnamed: 22",
        ]:
            axes[counter].plot(log["Iteration"], log[var])
            axes[counter].set_ylabel(var)
            axes[counter].grid(True)
            counter += 1

    plt.xlabel("Iteration")
    plt.tight_layout()
    # Adding a legend

    plt.savefig(save_fig_path + "/trace.pdf", format="pdf")

    # plt.show()


def run_BayesTraits(tree, labels):
    print(tree, labels)
    result = subprocess.run(
        f"./start.sh {tree} {labels}", shell=True, capture_output=True, text=True
    )

    return result.stdout


def run_all_trees():

    print(f"Inference on {len(data)} trees")
    processes = []
    for tree, labels in data:
        process = multiprocessing.Process(target=run_BayesTraits, args=(tree, labels))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    for _, label in data:
        logfilepath = label + ".Log.txt"
        plot_trace(logfilepath)


run_all_trees()
