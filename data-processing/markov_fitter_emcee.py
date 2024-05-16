from markov_fitter import get_data, transition_map_rates
import numpy as np
import pandas as pd
import scipy
import emcee
from matplotlib import pyplot as plt
import seaborn as sns

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
            leafid = walk["leafid"][0]
            initial_state = walk["first_cat"][0]
            steps = walk["shape"].tolist()
            for i, curr in enumerate(steps):
                if i == 0:
                    prev = initial_state
                else:
                    prev = steps[i - 1]
                    transition = prev + curr
                    walk_transitions.append(transition)
            count_df = (
                ((pd.Series(walk_transitions)).value_counts())
                .to_frame()
                .reset_index(names="transition")
            )
            count_df.insert(loc=0, column="leafid", value=leafid)
            count_df.insert(loc=0, column="first_cat", value=initial_state)
            count_dfs.append(count_df)

    # total of each transition per walkid per leafid
    counts = pd.concat(count_dfs).reset_index(drop=True)
    # total of each transition per leafid
    counts = (
        counts.groupby(["leafid", "first_cat", "transition"])["count"]
        .sum()
        .reset_index(name="count")
    )
    print(counts)
    # average no. counts for each transition per first_cat
    avg_counts = (
        counts.groupby(["transition"])["count"].agg(["mean", "sem"]).reset_index()
    )
    avg_counts["ub"] = avg_counts["mean"] + 1.96 * avg_counts["sem"]
    avg_counts["lb"] = avg_counts["mean"] - 1.96 * avg_counts["sem"]
    print(avg_counts)

    mean = avg_counts[["transition", "mean"]].rename(columns={"mean": "count"})
    ub = avg_counts[["transition", "ub"]].rename(columns={"ub": "count"})
    lb = avg_counts[["transition", "lb"]].rename(columns={"lb": "count"})

    return mean, ub, lb


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
    Pt = scipy.linalg.expm(Q)  # t=1 for every transition
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
    transitions_total = get_transition_count(dfs)
    # mean, ub, lb = get_transition_count_avg(dfs)
    transitions = transitions_total

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


if __name__ == "__main__":
    # dfs = get_data()
    # get_transition_count_avg(dfs)
    samples, sampler = run_mcmc()
    plot_posterior(samples, sampler)
    # plot_posterior_fromfile(
    #     "markov_fitter_reports/emcee/24chains_25000steps_15000burnin/emcee_run_log_24-04-24.csv"
    # )
    # plot_posterior_fromfile("emcee_run_log.csv")
    # print(get_transition_count(get_data()))

    # Hi Berta, I got your feedback for the report! Thanks for your kind words, I'm glad you liked it!
