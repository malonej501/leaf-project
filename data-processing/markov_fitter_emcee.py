from dataprocessing import concatenator, first_cats
import numpy as np
import pandas as pd
import scipy
import emcee
from matplotlib import pyplot as plt
import seaborn as sns
import multiprocessing
import os

lb, ub = 0, 1
ndim = 12
nwalkers = 24

rates_map = {
    0: ("u", "l"),
    1: ("u", "d"),
    2: ("u", "c"),
    3: ("l", "u"),
    4: ("l", "d"),
    5: ("l", "c"),
    6: ("d", "u"),
    7: ("d", "l"),
    8: ("d", "c"),
    9: ("c", "u"),
    10: ("c", "l"),
    11: ("c", "d"),
}

transition_map_rates = {
    "uu": (0, 0),
    "ul": (0, 1),
    "ud": (0, 2),
    "uc": (0, 3),
    "lu": (1, 0),
    "ll": (1, 1),
    "ld": (1, 2),
    "lc": (1, 3),
    "du": (2, 0),
    "dl": (2, 1),
    "dd": (2, 2),
    "dc": (2, 3),
    "cu": (3, 0),
    "cl": (3, 1),
    "cd": (3, 2),
    "cc": (3, 3),
}


def get_data():
    dfs = concatenator()

    dfs_new = []
    for walk in dfs:
        walk["step"] = walk.index.values
        walk = pd.merge(walk, first_cats[["leafid", "first_cat"]], on="leafid")
        walk.drop(
            columns=walk.columns.difference(
                ["leafid", "walkid", "first_cat", "shape", "step"]
            ),
            inplace=True,
        )
        dfs_new.append(walk)
    # concat = pd.concat(dfs, ignore_index=True)
    # concat = pd.merge(concat, first_cats[["leafid", "first_cat"]], on="leafid")
    # mapping = {"u": 0, "l": 1, "d": 2, "c": 3}
    # concat["shape_id"] = concat["shape"].map(mapping)

    return dfs_new


def get_transition_count(dfs):
    alltransitions = []
    for walk in dfs:
        if not walk.empty:
            initial_state = walk["first_cat"][0]
            steps = walk["shape"].tolist()
            for i, curr in enumerate(steps):
                if i == 0:
                    prev = initial_state
                else:
                    prev = steps[i - 1]
                    transition = prev + curr
                    alltransitions.append(transition)
    count_df = ((pd.Series(alltransitions)).value_counts()).to_frame().reset_index()
    count_df.columns = ["transition", "count"]
    return count_df


def get_transition_count_avg(dfs):
    count_dfs = []
    for walk in dfs:
        walk_transitions = []
        if not walk.empty:
            walkid = walk["walkid"][0]
            leafid = walk["leafid"][0]
            initial_state = walk["first_cat"][0]
            steps = walk["shape"].tolist()
            for i, curr in enumerate(steps):
                if i == 0:
                    prev = initial_state
                # if i > 20:
                #     break
                else:
                    prev = steps[i - 1]
                    transition = prev + curr
                    walk_transitions.append(transition)
            count_df = (
                ((pd.Series(walk_transitions)).value_counts())
                .to_frame()
                .reset_index(names="transition")
            )
            # count_df.insert(loc=0, column="walk_length", value=len(walk_transitions))
            count_df.insert(loc=0, column="walkid", value=walkid)
            count_df.insert(loc=0, column="leafid", value=leafid)
            count_df.insert(loc=0, column="first_cat", value=initial_state)
            count_dfs.append(count_df)

    # total of each transition per walkid per leafid
    counts = pd.concat(count_dfs).reset_index(drop=True)
    print(counts)
    exit()
    # counts["count_norm"] = counts["count"] / counts["walk_length"]

    # total no. each transition per leafid
    leaf_sum = (
        counts.groupby(["first_cat", "leafid", "transition"])["count"]
        .agg(["sum"])
        .reset_index()
    )
    print(leaf_sum)
    # average no. each transition across leafids
    leaf_avg = (
        leaf_sum.groupby(["transition"])["sum"]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    print(leaf_avg)
    # avg_counts["sem"] = avg_counts["sem"].fillna(
    #     0
    # )  # Beware this will give spuriously tight confidence interval - technically the interval is infinite
    # leaf_avg["ub"] = leaf_avg["mean"] + 1.96 * leaf_avg["sem"]
    # leaf_avg["lb"] = leaf_avg["mean"] - 1.96 * leaf_avg["sem"]
    # leaf_avg["ub"] = leaf_avg["mean"] + leaf_avg["std"]
    # leaf_avg["lb"] = leaf_avg["mean"] - leaf_avg["std"]
    leaf_avg["std_frac"] = leaf_avg["std"] / leaf_avg["mean"]
    transition_map = {
        "ul": "u→l",
        "ud": "u→d",
        "uc": "u→c",
        "lu": "l→u",
        "ld": "l→d",
        "lc": "l→c",
        "du": "d→u",
        "dl": "d→l",
        "dc": "d→c",
        "cu": "c→u",
        "cl": "c→l",
        "cd": "c→d",
    }
    leaf_avg["transition"] = leaf_avg["transition"].replace(transition_map)
    print(leaf_avg)
    exit()
    leaf_avg.to_csv("MUT2_counts.csv", index=False)
    print(leaf_avg)
    exit()

    mean = leaf_avg[["transition", "mean"]].rename(columns={"mean": "count"})
    ub = leaf_avg[["transition", "ub"]].rename(columns={"ub": "count"})
    lb = leaf_avg[["transition", "lb"]].rename(columns={"lb": "count"})
    sem = leaf_avg[["transition", "sem"]].rename(columns={"sem": "count"})
    std = leaf_avg[["transition", "std"]].rename(columns={"std": "count"})

    return mean, ub, lb, sem, std


def get_leaf_transitions(dfs):
    count_template = pd.DataFrame(
        {
            "transition": [
                "uu",
                "ul",
                "ud",
                "uc",
                "lu",
                "ll",
                "ld",
                "lc",
                "du",
                "dl",
                "dd",
                "dc",
                "cu",
                "cl",
                "cd",
                "cc",
            ],
            "count": [0] * 16,
        }
    )
    count_dfs = []
    for walk in dfs:
        walk_transitions = []
        if not walk.empty:
            walkid = walk["walkid"][0]
            leafid = walk["leafid"][0]
            initial_state = walk["first_cat"][0]
            steps = walk["shape"].tolist()
            for i, curr in enumerate(steps):
                if i == 0:
                    prev = initial_state
                # if i > 20:
                #     break
                else:
                    prev = steps[i - 1]
                    transition = prev + curr
                    walk_transitions.append(transition)
            walk_counts = (
                ((pd.Series(walk_transitions)).value_counts())
                .to_frame()
                .reset_index(names="transition")
            )
            count_df = pd.merge(
                count_template, walk_counts, on=["transition", "count"], how="outer"
            )
            # count_df.insert(loc=0, column="walk_length", value=len(walk_transitions))
            count_df.insert(loc=0, column="walkid", value=walkid)
            count_df.insert(loc=0, column="leafid", value=leafid)
            count_df.insert(loc=0, column="first_cat", value=initial_state)
            count_dfs.append(count_df)
    counts = pd.concat(count_dfs).reset_index(drop=True)
    leaf_sum = (
        counts.groupby(["first_cat", "leafid", "transition"])["count"]
        .agg(["sum"])
        .reset_index()
    )

    return leaf_sum


def log_prob(params):
    Q = np.array(
        [
            [-(params[0] + params[1] + params[2]), params[0], params[1], params[2]],
            [params[3], -(params[3] + params[4] + params[5]), params[4], params[5]],
            [params[6], params[7], -(params[6] + params[7] + params[8]), params[8]],
            [params[9], params[10], params[11], -(params[9] + params[10] + params[11])],
        ]
    )
    log_prob = 0
    Pt = scipy.linalg.expm(Q * 0.1)  # t=1 for every transition
    for i, transition in enumerate(transitions["transition"]):
        log_prob += transitions["count"][i] * np.log(
            Pt[transition_map_rates[transition]]
        )
    if np.isnan(log_prob):
        log_prob = -np.inf
    return log_prob


def run_mcmc():
    dfs = get_data()
    global transitions
    # transitions_total = get_transition_count(dfs)
    leaf_sum = get_leaf_transitions(dfs)
    # mean, ub, lb, sem, std = get_transition_count_avg(dfs)
    # transitions = std
    # transitions = get_transition_count(dfs)

    print(transitions)
    init_params = np.random.rand(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    state = sampler.run_mcmc(
        init_params, 25000, skip_initial_state_check=True, progress=True
    )

    samples = sampler.get_chain(flat=True, discard=15000)
    samples = pd.DataFrame(samples)
    samples.to_csv("emcee_run_log.csv", index=False)

    return samples, sampler


def run_mcmc_leaf_uncert(pid):
    n_shuffle = 10
    dfs = get_data()
    global transitions
    leaf_sum = get_leaf_transitions(dfs)
    print(leaf_sum)
    leaf_grouped = leaf_sum.groupby(["first_cat", "leafid"])
    # first_cat = np.random.choice(["u", "l", "d", "c"])
    # infer rates from the mean transition counts of 4 random leaves, 1 from each category
    for i in range(0, n_shuffle):
        np.random.seed()
        sample = []
        # pick 2 random leaves from each first_cat - they are sampled without replacement
        for shape in ["u", "l", "d", "c"]:
            leaf = np.random.choice(
                [key[1] for key in list(leaf_grouped.groups.keys()) if key[0] == shape],
                size=2,
                replace=False,
            )
            sample.extend(leaf)
        # retrieve the counts associated with the sample leaves
        leaf_sum_sub = leaf_sum[leaf_sum["leafid"].isin(sample)][
            ["transition", "sum"]
        ].rename(columns={"sum": "count"})
        # calculate the mean count for each transition type across the sample
        transitions = (
            leaf_sum_sub.groupby("transition")["count"].agg("mean").reset_index()
        )
        print(transitions)
        # infer rates from this particular sample
        init_params = np.random.rand(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        state = sampler.run_mcmc(
            init_params, 25000, skip_initial_state_check=True, progress=True
        )
        # reduce the size of the saved chain by discarding the burnin and rounding each step and recording only every thin step
        samples = np.round(
            sampler.get_chain(flat=True, discard=15000, thin=100), decimals=6
        )
        samples = pd.DataFrame(samples)
        samples.to_csv(f"emcee_run_log_{sample}_{pid}_{i}.csv", index=False)


def run_leaf_uncert_parallel():
    n_processes = 10
    processes = []
    for i in range(n_processes):
        process = multiprocessing.Process(target=run_mcmc_leaf_uncert, args=(i,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def plot_posterior(samples, sampler):

    samples_long = pd.melt(samples, var_name="parameter", value_name="rate")
    samples_long["initial_shape"], samples_long["final_shape"] = zip(
        *samples_long["parameter"].map(rates_map)
    )
    samples_long["transition"] = (
        samples_long["initial_shape"] + samples_long["final_shape"]
    )
    summary = samples_long.groupby("transition").agg({"rate": ["mean", "sem"]})
    summary.to_csv("MUT2.2_emcee_rates.csv")

    sns.displot(
        data=samples_long,
        x="rate",
        col="transition",
        col_wrap=4,
        kind="hist",
    )
    plt.show()
    plt.clf()

    order = ["u", "l", "d", "c"]
    labels = ["unlobed(u)", "lobed(l)", "dissected(d)", "compound(c)"]
    g = sns.catplot(
        data=samples_long,
        y="rate",
        x="initial_shape",
        hue="final_shape",
        kind="bar",
        palette="colorblind",
        order=order,
        hue_order=order,
    )
    g.set_xticklabels(labels=labels)
    g._legend.set_title("Final Shape")
    plt.ylabel("Evolutionary Rate")
    plt.xlabel("Initial Shape")

    plt.show()

    chain = sampler.get_chain()[
        :, 1, :
    ]  # from the left to right the indicies represent: step, chain, parameter
    # here we take all steps for all parameters from one chain
    chain = pd.DataFrame(chain)
    chain["step"] = chain.index
    chain_long = pd.melt(
        chain, id_vars=["step"], var_name="parameter", value_name="rate"
    )
    sns.relplot(
        data=chain_long, x="step", y="rate", col="parameter", col_wrap=4, kind="line"
    )
    plt.show()
    plt.clf()
    chain = []


def plot_posterior_fromfile(file):
    samples = pd.read_csv(file)
    samples_norm = samples.div(samples.max(axis=None))
    norm_long = pd.melt(samples_norm, var_name="parameter", value_name="rate")
    norm_long["parameter"] = norm_long["parameter"].astype(int)
    norm_long["initial_shape"], norm_long["final_shape"] = zip(
        *norm_long["parameter"].map(rates_map)
    )
    print(samples)
    print(norm_long)
    order = ["u", "l", "d", "c"]
    labels = ["unlobed(u)", "lobed(l)", "dissected(d)", "compound(c)"]
    g = sns.catplot(
        data=norm_long,
        y="rate",
        x="initial_shape",
        hue="final_shape",
        kind="bar",
        palette="colorblind",
        order=order,
        hue_order=order,
    )
    g.set(ylim=(0, 1))
    g.set_xticklabels(labels=labels)
    g._legend.set_title("Final Shape")
    plt.ylabel("Normalised Evolutionary Rate")
    plt.xlabel("Initial Shape")

    plt.show()


def combine_posteriors_from_file():

    directory = "markov_fitter_reports/emcee/err_MUT2"
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            # df = df.iloc[::2000, :].reset_index(drop=True)  # remove every nth row
            dfs.append(df)
            del df
    samples = pd.concat(dfs)
    # samples.to_csv("samples.csv", index=False)

    # samples = pd.read_csv("samples.csv")

    print(samples)
    print(samples.describe())

    samples_long = pd.melt(samples, var_name="parameter", value_name="rate")
    print(samples_long)

    samples_long["initial_shape"], samples_long["final_shape"] = zip(
        *samples_long["parameter"].astype(float).map(rates_map)
    )
    samples_long["transition"] = (
        samples_long["initial_shape"] + samples_long["final_shape"]
    )

    sns.displot(
        data=samples_long,
        x="rate",
        col="transition",
        col_wrap=4,
        kind="hist",
        bins=500,
    )
    plt.xlim(-0.5, 3)
    plt.show()

    sns.boxplot(data=samples_long, x="transition", y="rate", showfliers=False)
    plt.show()


if __name__ == "__main__":
    # dfs = get_data()
    # get_transition_count_avg(dfs)
    #run_leaf_uncert_parallel()
    combine_posteriors_from_file()
    # samples, sampler = run_mcmc()
    # plot_posterior(samples, sampler)
    # plot_posterior_fromfile(
    #     "markov_fitter_reports/emcee/24chains_25000steps_15000burnin/emcee_run_log_24-04-24.csv"
    # )
    # plot_posterior_fromfile("emcee_run_log.csv")
    # print(get_transition_count(get_data()))
