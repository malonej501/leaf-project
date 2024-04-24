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


def log_prob_old(params):
    Q = np.array(
        [
            [-(params[0] + params[1] + params[2]), params[0], params[1], params[2]],
            [params[3], -(params[3] + params[4] + params[5]), params[4], params[5]],
            [params[6], params[7], -(params[6] + params[7] + params[8]), params[8]],
            [params[9], params[10], params[11], -(params[9] + params[10] + params[11])],
        ]
    )
    log_prob = 0
    t = 1  # between each state
    Pt = scipy.linalg.expm(Q * t)
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
                    log_prob += np.log(Pt[transition_map_rates[transition]])
    if np.isnan(log_prob):
        log_prob = -np.inf
    return log_prob


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


dfs = get_data()
transitions = get_transition_count(dfs)
print(transitions)
init_params = np.random.rand(nwalkers, ndim)


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
state = sampler.run_mcmc(
    init_params, 25000, skip_initial_state_check=True, progress=True
)

# plot posterior distributions
samples = sampler.get_chain(flat=True, discard=15000)
samples = pd.DataFrame(samples)
samples.to_csv("emcee_run_log.csv")
exit()
samples_long = pd.melt(samples, var_name="parameter", value_name="rate")
samples_long["initial_shape"], samples_long["final_shape"] = zip(
    *samples_long["parameter"].map(rates_map)
)
samples_long["transition"] = samples_long["initial_shape"] + samples_long["final_shape"]
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
chain_long = pd.melt(chain, id_vars=["step"], var_name="parameter", value_name="rate")
sns.relplot(
    data=chain_long, x="step", y="rate", col="parameter", col_wrap=4, kind="line"
)
plt.show()
plt.clf()
chain = []
