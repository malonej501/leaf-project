import shutil
import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# This is an alternative way to calculate the Q matrix from random walk data.
# Instead of using MLE, we use transition probabilities and average holding
# times inferred directly from the data.

# WALK_DATA = "MUT2.2.csv"  # Path to the CSV file containing walk data
# MLE_Q_DATA = "MUT2.2_MLE_rates.csv"  # CSV containing MLE rates
# Path to the CSV file containing walk data
WALK_DATA = "MUT2_320_mle_23-04-25.csv"
# WALK_DATA = "pwalks_10_160_leaves_full_13-03-25_MUT2_CLEAN.csv"
MLE_Q_DATA = "markov_fitter_reports/emcee/MUT2_320_mcmc_2_24-04-25/" \
    "ML_MUT2_320_mcmc_2_24-04-25.csv"
# MLE_Q_DATA = "markov_fitter_reports/emcee/MUT2_pwalks_160_10_07-04-25/" \
#     "ML_MUT2_pwalks_160_10_07-04-25.csv"
EXPORT_Q = False   # export the Q matrix to file
# Q_ID = "q_p-ht_02-04-25"  # ID for exporting the p/ht Q matrix
# Q_ID = "q_p-ht_contig_filt_u_03-04-25"
Q_ID = "test"
# keep only walks with first_cat in contig_filt for holding time calculation
CONTIG_FILT = ["u", "l", "d", "c"]
RM_NO_TRANS = False  # remove walks that never transition
PLOT = 3  # 0 - None, 1 - trans probs and hold times, 2 - q matrix comparison,
# 3 - hold time distribution, 4 - hold time distribution by first_cat,
# 5 - alternative hold time distribution, 99 - plot all
INCL_DIAG = False  # include diagonal transitions e.g. uu in prop calculation
V = False  # verbose


def get_walks():
    """Return the details of random walks along with transition type at each
    step"""
    walks = pd.read_csv(WALK_DATA)
    assert len(walks) == 320 * 5 * 48, (
        "The number of rows in the walk data does not match the expected " +
        "number of steps (320 leaves * 5 steps * 48 walks per leaf)."
    )
    # get transitions by shifting shape columns down by one and combining
    walks["prevshape"] = walks["shape"].shift(+1)
    # replace 0th step with first_cat
    walks.loc[walks["step"] == 0, "prevshape"] = walks["first_cat"]
    walks["transition"] = walks["shape"] + walks["prevshape"]
    if RM_NO_TRANS:
        # remove groups that never transition
        walks = walks[walks.groupby(["leafid", "walkid"])[
            "transition"].transform("nunique") > 1].reset_index(drop=True)

    return walks


def contig_state_counts(c_filt=None):
    """Calculate the number of contiguous rows for each state in each unique
    walk"""
    if c_filt is None:
        c_filt = CONTIG_FILT
    walks = get_walks()
    # optionally filter data by first_cat before calculating holding time
    walks = walks[walks["first_cat"].isin(c_filt)]

    walks["uniq_wid"] = walks["leafid"] + "_" + walks["walkid"].astype(str)

    # Group by unique walks (assuming 'walk_id' identifies unique walks)
    if "uniq_wid" not in walks.columns:
        raise ValueError("The 'uniq_wid' column is missing in the dataset.")

    def count_contig_states(group):
        # Asign unique id to each contig block of states
        group["contig_block"] = (~group["transition"].isin([
            "uu", "ll", "dd", "cc"])).cumsum()
        if V:
            print(group.to_string())
            print(group.groupby(["uniq_wid", "contig_block", "shape"]
                                ).size().reset_index(name="count").to_string())

        # Count the number of rows in each contig block
        return group.groupby(["uniq_wid",
                              "contig_block",
                              "shape",
                              "first_cat"]).size().reset_index(name="count")

    # Apply the function to each walk
    c_count = walks.groupby("uniq_wid")[["uniq_wid", "shape",
                                         "step", "transition",
                                         "first_cat"]].apply(
        count_contig_states).reset_index(drop=True)
    assert c_count["count"].sum() == 320 * 5 * 48, (
        "The total no. steps in holding times does not match the expected number "
        "of steps (320 leaves * 5 steps * 48 walks per leaf)."
    )
    h_time_avg = c_count.groupby("shape")["count"].agg(
        ht_avg="mean", ht_std="std", n_contigs="count").reset_index()
    h_time_avg["ht_se"] = h_time_avg["ht_std"] / h_time_avg["n_contigs"]**0.5

    return c_count, h_time_avg


def trans_prob():
    """Get each transition type as a proportion of total transitions from the
    same initial state. Can include diagonal transitionsin the proportion
    calculation or not"""
    walks = get_walks()
    # count transitions
    freq = walks["transition"].value_counts().reset_index()
    freq.columns = ["trans", "count"]  # Rename columns
    freq["init"] = freq["trans"].str[0]  # get initial state in transition

    # calculate proportion
    if INCL_DIAG:
        prop = freq
        prop["prop"] = freq["count"] / \
            freq.groupby("init")["count"].transform("sum")
    else:
        # remove self transitions before calculating proportion
        freq_no_diag = freq[~freq["trans"].isin(
            ["uu", "ll", "dd", "cc"])].copy()
        prop = freq_no_diag
        prop["prop"] = freq_no_diag["count"] / \
            freq_no_diag.groupby("init")["count"].transform("sum")

    return prop


def calc_q():
    """Calculate the q matrix from transition probabilities and average
    holding timees"""
    _, ht = contig_state_counts()
    tp = trans_prob()
    res = tp.merge(ht, left_on="init", right_on="shape", how="outer")
    res["q_p/ht"] = res["prop"] / res["ht_avg"]

    mle_q = pd.read_csv(MLE_Q_DATA)  # load MLE Q data from csv
    if MLE_Q_DATA != "MUT2.2_MLE_rates.csv":
        # Convert MLE Q data to long format
        mle_q = mle_q.T
        mle_q.columns = ["rate"]
        mle_q["transition"] = ["ul", "ud", "uc", "lu", "ld", "lc",
                               "du", "dl", "dc", "cu", "cl", "cd"]

    mle_q = mle_q[["transition", "rate"]]
    mle_q = mle_q.rename(columns={"transition": "trans", "rate": "q_mle"})
    res_comp = res.merge(mle_q, on="trans", how="outer")
    # reverse the order of the rows
    # res_comp = res_comp.iloc[::-1].reset_index(drop=True)

    return res_comp


def plot_q(q_data):
    """Plot the q matrix for comparisson"""
    w = 0.4  # width of the bars

    _, ax = plt.subplots(figsize=(6, 5))

    ax.bar([i - w/2 for i in range(len(q_data))], q_data["q_p/ht"],
           width=w, label="q_p/ht", color="C0")
    ax.bar([i + w/2 for i in range(len(q_data))], q_data["q_mle"],
           width=w, label="q_mle", color="C1")
    ax.set_xlabel("Transition")
    ax.set_ylabel("Q matrix value")
    ax.set_xticks(range(len(q_data)))
    ax.set_xticklabels(q_data["trans"])
    ax.legend(loc="upper right")
    ax.set_title(
        f"Q matrix comparison\n{WALK_DATA} -\n{MLE_Q_DATA.rsplit('/', 1)[-1]}")

    plt.tight_layout()
    plt.show()


def plot_prop_ht(q_data):
    """Visualise proportions and holding times"""
    fig, axs = plt.subplots(ncols=2, figsize=(8, 5))
    print(q_data)
    axs[0].bar(q_data["trans"], q_data["prop"])
    axs[0].set_xlabel("Transition")
    if INCL_DIAG:
        axs[0].set_ylabel(
            "Transition proportions\nincluding diagonal transitions")
    else:
        axs[0].set_ylabel(
            "Transition proportions\nexcluding diagonal transitions")

    axs[1].bar(q_data["init"], q_data["ht_avg"])
    axs[1].set_xlabel("State")
    axs[1].set_ylabel("Average holding time")
    fig.suptitle(
        f"Transition proportions and average holding times\n{WALK_DATA}")
    plt.tight_layout()
    plt.show()


def plot_contig_distr():
    """Visualise the distribution of holding times. Can balance the no.
    contigs or just use the maximum for each shape."""
    c_count, _ = contig_state_counts()
    min_count = c_count["shape"].value_counts().min()  # shape with min count
    c_count_balanced = c_count.groupby("shape", group_keys=False).apply(
        lambda x: x.sample(n=min_count))  # balance the counts
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7), sharex=True,
                            sharey=True)
    fig.suptitle(
        f"Holding time distribution by state\n{WALK_DATA}, "
        f"contig_filt={CONTIG_FILT}")
    fig.supxlabel("Holding time (no. steps)")
    fig.supylabel("Frequency")
    for i, shape in enumerate(["u", "l", "d", "c"]):
        ax = axs[i // 2, i % 2]
        c_count_shape = c_count_balanced[c_count_balanced["shape"] == shape]
        ax.hist(c_count_shape["count"], bins=50, range=(0, 320))
        ax.grid(alpha=0.3)
        ax.set_title(["Unlobed", "Lobed", "Dissected", "Compound"][i] +
                     fr", $N={len(c_count_shape)}$")
    plt.tight_layout()
    plt.show()


def plot_contig_distr_alt():
    """Visualise holding time distribution with violin plots."""
    c_count, _ = contig_state_counts()
    min_count = c_count["shape"].value_counts().min()  # shape with min count
    c_count_balanced = c_count.groupby("shape", group_keys=False).apply(
        lambda x: x.sample(n=min_count))  # balance the counts

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, shape in enumerate(["u", "l", "d", "c"]):
        c_count_shape = c_count_balanced[c_count_balanced["shape"] == shape]
        q1, med, q3 = np.percentile(c_count_shape["count"], [25, 50, 75])
        min, max = c_count_shape["count"].min(), c_count_shape["count"].max()
        # ax.hist(c_count_shape["count"],
        #         bins=50, alpha=0.3, label=f"State: {shape}",
        #         color=f"C{i}", range=(0, 320))
        v = ax.violinplot(c_count_shape["count"],
                          positions=[i], showmeans=False, showmedians=False,
                          widths=0.5, bw_method=0.2, showextrema=False)
        v["bodies"][0].set_facecolor("C0")
        v["bodies"][0].set_edgecolor("C0")
        # ax.vlines([i], q1, q3, color="black", lw=3)
        # ax.vlines([i], min, max, color="black", lw=1)
        # ax.scatter([i], med, color="white", marker="o", s=20, zorder=3,
        #            edgecolor="black", label=f"State: {shape}")
        ax.boxplot(c_count_shape["count"], positions=[i], widths=0.2,
                   showfliers=True, showmeans=False,
                   medianprops=dict(color="black"))
    # ax.set_xlabel("Contiguous state length (holding time)")
    # ax.set_ylabel("Frequency")
        print(len(c_count_shape))
    ax.set_xticks(range(4), ["Unlobed", "Lobed", "Dissected", "Compound"])
    ax.set_ylabel("Holding time (no. steps)")
    ax.set_title(f"Holding time distribution by state\n{WALK_DATA}, "
                 f"contig_filt={CONTIG_FILT}")
    # ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_contig_distr_by_first_cat():
    """Visualise the distribution of holding times separately for chains
    starting in each first_cat"""
    u_cc, _ = contig_state_counts(c_filt=["u"])
    l_cc, _ = contig_state_counts(c_filt=["l"])
    d_cc, _ = contig_state_counts(c_filt=["d"])
    c_cc, _ = contig_state_counts(c_filt=["c"])
    cc_fcat = pd.concat([u_cc, l_cc, d_cc, c_cc], ignore_index=True)
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 9), sharex=True)
    fig.suptitle(
        f"Holding time distribution by state and initial state\n{WALK_DATA}")
    fig.supxlabel("Contiguous state length (holding time)")
    fig.supylabel("Frequency")
    for i, first_cat in enumerate(["u", "l", "d", "c"]):
        for j, shape in enumerate(["u", "l", "d", "c"]):
            ax = axs[i, j]
            ax.hist(cc_fcat[(cc_fcat["shape"] == shape) &
                            (cc_fcat["first_cat"] == first_cat)]["count"],
                    bins=30)
            ax.grid(alpha=0.3)
            ax.set_title(f"Initial state: {first_cat}, state: {shape}")
    plt.tight_layout()
    plt.show()


def export_q(res):
    """Export the Q matrix to file in correct format for curves.py"""
    order = ["ul", "ud", "uc", "lu", "ld", "lc",
             "du", "dl", "dc", "cu", "cl", "cd"]
    res["trans"] = pd.Categorical(res["trans"], categories=order, ordered=True)
    res = res.sort_values("trans").reset_index(drop=True)
    q_curves = res["q_p/ht"].to_frame().T  # reshape dataframe to for curves.py
    q_dir = f"markov_fitter_reports/emcee/{Q_ID}"
    if os.path.exists(q_dir):
        user_input = input(
            f"The directory '{q_dir}' already exists. "
            "Overwrite? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Cancelled.")
            return
        shutil.rmtree(q_dir)  # remove existing directory
    os.mkdir(q_dir)
    q_curves.to_csv(f"{q_dir}/ML_{Q_ID}.csv", index=False, header=True)
    print(f"Q matrix exported to {q_dir}/ML_{Q_ID}.csv")

### TESTS ###


def test_holding_time():
    """Test the holding time function"""
    walks = get_walks()  # arrange
    cc, _ = contig_state_counts()

    # check for duplicate uniq_wid contig block rows
    duplicates = cc[cc.duplicated(
        subset=["uniq_wid", "contig_block"], keep=False)]  # act
    assert duplicates.empty, (
        "Duplicate contig block ids found in same unique walks")  # assert

    # check that sum of counts for each shape is equal to the total number of
    # rows in the walks
    total_contig_len = sum(cc["count"])  # act
    total_walk_rows = len(walks[walks["first_cat"].isin(CONTIG_FILT)])
    assert total_contig_len == total_walk_rows, (
        "Total contigs length does not equal total walk rows"
    )  # assert

    # check that all CONTIG_FILT rows are removed from contig counts
    assert all(cc["first_cat"].isin(CONTIG_FILT)
               ), "Contig counts contain rows CONTIG_FILT values in first_cat"

    print("Holding time tests passed")


def test_trans_prob():
    """Test the transition probability function"""
    prop = trans_prob()  # arrange
    # check the sum of rows in P matrix =1 1 e.g. Pul+Pud+Puc = 1 if no diag
    prop_sum = prop.groupby("init")["prop"].sum(
    ).reset_index()  # act - sum rows of P matrix
    assert np.all(np.isclose(prop_sum["prop"], 1.0)), (
        "Total transition proportions do not equal 1 for each initial state"
    )

    print("Transition probability tests passed")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "-e" in args:
        EXPORT_Q = True

    test_holding_time()
    test_trans_prob()
    q = calc_q()
    print(q)
    if EXPORT_Q:
        export_q(q)
    if PLOT == 1:
        plot_prop_ht(q)
    elif PLOT == 2:
        plot_q(q)
    elif PLOT == 3:
        plot_contig_distr()
    elif PLOT == 4:
        plot_contig_distr_by_first_cat()
    elif PLOT == 5:
        plot_contig_distr_alt()
    elif PLOT == 99:
        plot_prop_ht(q)
        plot_q(q)
        plot_contig_distr()
        plot_contig_distr_by_first_cat()
