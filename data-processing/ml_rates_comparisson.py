import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ML_FITS = [
    "MUT2_320_mle_20-03-25",
    "q_p-ht_02-04-25",
    "MUT2_160-320_reinit_03-04-25",
    "MUT2_pwalks_160_10_07-04-25",
    "MUT2_mle_from0_4_11-04-25",
]
EXPORT = True  # export figure to pdf


def get_ml_rates():
    """Get the mean and confidence intervals for the posterior distributions
    of all transition rates from simulation mcmc inference specified by
    SIMRATES"""

    fits = []
    for fit in ML_FITS:
        fit_df = pd.read_csv(f"markov_fitter_reports/emcee/{fit}/ML_{fit}.csv")
        fit_df.insert(0, "fit", fit)
        fits.append(fit_df)

    fits = pd.concat(fits, ignore_index=True)
    name_map = {"0": "ul", "1": "ud", "2": "uc",
                "3": "lu", "4": "ld", "5": "lc",
                "6": "du", "7": "dl", "8": "dc",
                "9": "cu", "10": "cl", "11": "cd"}
    fits = fits.rename(columns=name_map)

    fits.insert(1, "uu", -fits["ul"] - fits["ud"] - fits["uc"])
    fits.insert(6, "ll", -fits["lu"] - fits["ld"] - fits["lc"])
    fits.insert(11, "dd", -fits["du"] - fits["dl"] - fits["dc"])
    fits.insert(16, "cc", -fits["cu"] - fits["cl"] - fits["cd"])

    return fits


def plot_ml_rates():
    """Visualise rate value from different ML fits"""
    w = 0.8  # width of the bars
    fits = get_ml_rates()
    fits = fits.melt(id_vars=["fit"], var_name="transition",
                     value_name="rate")
    n_fits = len(fits["fit"].unique())

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(8, 6), sharey=True,
                            sharex=True, layout="constrained")
    br_h = None  # bar handles
    br_l = None  # bar labels
    br_l_prefix = [
        "Original fit",
        "Prob/holding time fit",
        "First step = 160 fit",
        "Pseudo-walks fit",
        "New likelihood fit"
    ]
    for i, ax in enumerate(axs.flatten()):
        if i in [0, 5, 10, 15]:
            # ax.axis("off")  # skip diagonals
            continue
        t = list(fits["transition"].unique())[i]
        data = fits[fits["transition"] == t]
        br_h = ax.bar(
            [i - w / 2 for i in range(n_fits)],
            data["rate"],
            width=w,
            label=data["fit"],
            color=[plt.cm.tab10(i) for i in range(n_fits)],
        )
        br_l = data["fit"].to_list()
        ax.set_xticks([i - w / 2 for i in range(n_fits)])
        ax.set_xticklabels([])
        ax.set_title(t)
        ax.grid(alpha=0.3)

    fig.suptitle("ML rates from different fits")
    fig.supxlabel("Fit")
    fig.supylabel("ML rate")
    br_l = [f"{a}\n({b})" for a, b in zip(br_l_prefix, br_l)]
    fig.legend(br_h, br_l, loc="outside right center")
    if EXPORT:
        plt.savefig("ml_rates_comparisson.pdf")
    plt.show()


if __name__ == "__main__":
    plot_ml_rates()
