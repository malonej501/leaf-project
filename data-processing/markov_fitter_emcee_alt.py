from dataprocessing import first_cats
import numpy as np
import pandas as pd
import scipy
from scipy import optimize
import emcee
from matplotlib import pyplot as plt
import multiprocessing
import os
import corner
import re

n_processes = 10 # for parallelisation
init_lb, init_ub = 0, 0.1 # lb and ub of uniform distribution for initial values
ndim = 12 # no. rate parameters
nwalkers = 24 # no. markov chains run in parallel
nsteps = 5000 #25000 # no. steps for each markov chain
nshuffle = 1# 25 # no. times the leaf dataset is shuffled
burnin = 2000 #15000 # these first iterations are discarded from the chain
thin = 10 #100 # only every thin iteration is kept
t = 1 # the value of time for each timstep in P=exp[Q*t]
wd = "leaves_full_21-9-23_MUT2.2_CLEAN" # the simulation run to fit to
# wd = "leaves_full_15-9-23_MUT1_CLEAN"
# run_id = "MUT2_mcmc_03-12-24"
run_id = "MUT2_mcmc_04-12-24"
# run_id = "MUT1_mcmc_04-12-24"

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

labels = ["ul","ud","uc","lu","ld","lc","du","dl","dc","cu","cl","cd"]

def concatenator():
    dfs = []
    print(f"\n\nCurrent directory: {wd}\n\n")
    for leafdirectory in os.listdir(wd):
        print(f"Current = {leafdirectory}")
        leafdirectory_path = os.path.join(wd, leafdirectory)
        for walkdirectory in os.listdir(leafdirectory_path):
            walkdirectory_path = os.path.join(leafdirectory_path, walkdirectory)
            for file in os.listdir(walkdirectory_path):
                if (
                    file.endswith(".csv")
                    and "shape_report" in file
                    and "shape_report1" not in file
                ):
                    df = pd.read_csv(os.path.join(walkdirectory_path, file))
                    df.insert(0, "leafid", leafdirectory)
                    df.insert(1, "walkid", int(re.findall(r"\d+", walkdirectory)[0]))
                    dfs.append(df)
    return dfs

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
    return dfs_new

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
                if i > 60: # only fit to data from the first 60 steps of each walk
                    break
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

    # total no. each transition per leafid
    leaf_sum = (
        counts.groupby(["first_cat", "leafid", "transition"])["count"]
        .agg(["sum"])
        .reset_index()
    )
    # average no. each transition across leafids
    leaf_avg = (
        leaf_sum.groupby(["transition"])["sum"]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    # print(leaf_avg)
    # avg_counts["sem"] = avg_counts["sem"].fillna(
    #     0
    # )  # Beware this will give spuriously tight confidence interval - technically the interval is infinite
    leaf_avg["ub"] = leaf_avg["mean"] + 1.96 * leaf_avg["sem"]
    leaf_avg["lb"] = leaf_avg["mean"] - 1.96 * leaf_avg["sem"]
    # leaf_avg["ub"] = leaf_avg["mean"] + leaf_avg["std"]
    # leaf_avg["lb"] = leaf_avg["mean"] - leaf_avg["std"]
    # leaf_avg["std_frac"] = leaf_avg["std"] / leaf_avg["mean"]
    # transition_map = {
    #     "ul": "u→l",
    #     "ud": "u→d",
    #     "uc": "u→c",
    #     "lu": "l→u",
    #     "ld": "l→d",
    #     "lc": "l→c",
    #     "du": "d→u",
    #     "dl": "d→l",
    #     "dc": "d→c",
    #     "cu": "c→u",
    #     "cl": "c→l",
    #     "cd": "c→d",
    # }
    # leaf_avg["transition"] = leaf_avg["transition"].replace(transition_map)
    # leaf_avg.to_csv("MUT2_counts.csv", index=False)
    mean = leaf_avg[["transition", "mean"]].rename(columns={"mean": "count"})
    ub = leaf_avg[["transition", "ub"]].rename(columns={"ub": "count"})
    lb = leaf_avg[["transition", "lb"]].rename(columns={"lb": "count"})
    sem = leaf_avg[["transition", "sem"]].rename(columns={"sem": "count"})
    std = leaf_avg[["transition", "std"]].rename(columns={"std": "count"})

    return mean, std, sem, lb, ub


def log_prior(params):  # define a uniform prior from 0 to 0.1 for every transition rate
    if all(0 <= q <= 0.1 for q in params): # all q parameters must be within the prior range to return 0
        return 0
    return -np.inf


def log_likelihood(params, t_mean, t_err):
    Q = np.array(
        [
            [-(params[0] + params[1] + params[2]), params[0], params[1], params[2]],
            [params[3], -(params[3] + params[4] + params[5]), params[4], params[5]],
            [params[6], params[7], -(params[6] + params[7] + params[8]), params[8]],
            [params[9], params[10], params[11], -(params[9] + params[10] + params[11])],
        ]
    )
    log_likelihood = 0
    Pt = scipy.linalg.expm(Q * t)  # * 0.1)  # t=1 for every transition
    for i, transition in enumerate(t_mean["transition"]):
        log_likelihood +=  t_mean["count"][i] * np.log( # see Kalbfleisch 1985 eq 3.2
            Pt[transition_map_rates[transition]]
        )
        if np.isnan(log_likelihood):
            log_likelihood = -np.inf
            break
    return log_likelihood


def log_probability(params, t_mean, t_err):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, t_mean, t_err)

def get_maximum_likelihood():
    dfs = get_data()
    t_mean, t_std, t_sem, t_lb, t_ub = get_transition_count_avg(dfs)
    print(t_mean)
    counts = get_leaf_transitions(dfs)
    counts = counts.groupby(["transition"])["sum"].agg(["sum"]).rename(columns={"sum":"count"}).reset_index()
    print(counts)
    # exit()

    nll = lambda Q: -log_likelihood(Q, counts, None)
    np.random.seed()
    init = np.random.uniform(init_lb, init_ub, ndim) # initialise 12 random numbers for Q matrix
    soln = optimize.minimize(nll, init)
    Q_ml = soln.x
    Q_ml_df = pd.DataFrame(Q_ml).T
    Q_ml_df.to_csv(f"{run_id}/ML_{run_id}.csv", index=False)
    print(f"Maximum likelihood rates: {Q_ml_df}")
    return Q_ml_df


def run_leaf_uncert_parallel_pool():
    os.mkdir(run_id)
    output_file = f'{run_id}/h_params_{run_id}.txt'

    with open(output_file, 'w') as file:
        file.write(f"Run {run_id} MCMC Hyper Parameters:\n")
        file.write(f"n_processes = {n_processes}\n")
        file.write(f"init_lb = {init_lb}, init_ub = {init_ub}\n")
        file.write(f"ndim = {ndim}\n")
        file.write(f"nwalkers = {nwalkers}\n")
        file.write(f"nsteps = {nsteps}\n")
        file.write(f"nshuffle = {nshuffle}\n")
        file.write(f"burnin = {burnin}\n")
        file.write(f"thin = {thin}\n")
        file.write(f"t = {t}\n")
        file.write(f"wd = {wd}\n")

    print("MCMC Hyper Parameters:")
    print(f"n_processes = {n_processes}")
    print(f"init_lb = {init_lb}, init_ub = {init_ub}")
    print(f"ndim = {ndim}")
    print(f"nwalkers = {nwalkers}")
    print(f"nsteps = {nsteps}")
    print(f"nshuffle = {nshuffle}")
    print(f"burnin = {burnin}")
    print(f"thin = {thin}")
    print(f"t = {t}")
    print(f"wd = {wd}")

    print(f"Parameters have been saved to {output_file}")

    dfs = get_data()
    global transitions
    # t_mean, t_std, t_sem, t_lb, t_ub = get_transition_count_avg(dfs)
    counts = get_leaf_transitions(dfs)
    counts = counts.groupby(["transition"])["sum"].agg(["sum"]).rename(columns={"sum":"count"}).reset_index()
    t_mean = counts
    t_sem = None
    print(t_mean)
    print(t_sem)

        
    np.random.seed()

    init_params = np.random.uniform(init_lb, init_ub, (nwalkers, ndim)) # generate initial values to fill Q matrix for each walker
    # set up file to save the run
    filename = f"{run_id}/{run_id}.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    with multiprocessing.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(t_mean, t_sem),pool=pool, backend=backend)
        state = sampler.run_mcmc(
            init_params, nsteps, skip_initial_state_check=True, progress=True
        )
    # reduce the size of the saved chain by discarding the burnin and rounding each step and recording only every thin step
    # samples = np.round(
    #     sampler.get_chain(flat=True, discard=burnin, thin=thin), decimals=6
    # )
    Q_ml = get_maximum_likelihood()
    plot_trace(sampler, Q_ml)
    tau = sampler.get_autocorr_time()
    print(f"No. steps until autocorrelation: {tau}")
    # samples = pd.DataFrame(samples)
    # samples.to_csv(f"{run_id}/emcee_run_log_{sample_str}_{i}.csv", index=False)
    # here we take all steps for all parameters from one chain
    # chain = pd.DataFrame(chain)
    # chain["step"] = chain.index
    # chain_long = pd.melt(
    #     chain, id_vars=["step"], var_name="parameter", value_name="rate"
    # )
    # chain_long["shuffle_id"] = i
    # chain_samples.append(chain_long)
    # chain_samples = pd.concat(chain_samples)
    # chain_samples.to_csv(f"{run_id}/emcee_run_chain1_.csv", index=False)
    
def corner_plot(sampler):
    flat_samples = sampler.get_chain(flat=True)
    corner_fig = corner.corner(flat_samples, labels=labels)
    plt.tight_layout()
    plt.show()

def plot_trace(sampler, ml_rates):
    fig, axes = plt.subplots(4, 4, figsize=(10,7), sharex=True)
    samples = sampler.get_chain()
    idx = 0
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i == j:
                ax.axis("off")
                continue
            ax.plot(samples[:,:,idx], c="C0", alpha=0.3) # from the left to right the indicies represent: step, chain, parameter
            ax.axhline(y= ml_rates.iloc[0,idx], c="C1", linestyle="--")
            ax.set_ylabel(labels[idx])
            ax.set_ylim(init_lb, init_ub)
            idx += 1
    plt.tight_layout()
    fig.savefig(f"{run_id}/trace_{run_id}.pdf", format="pdf")
    plt.show()

def sample_chain(sampler):
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    print(flat_samples)
    print(np.shape(flat_samples))
    flat_samples_df = pd.DataFrame(flat_samples, columns=list(range(flat_samples.shape[1])))
    print(flat_samples_df)
    flat_samples_df.to_csv(f"{run_id}/posteriors_{run_id}.csv",index=False)

def sampler_from_file(func):
    h_params_path = run_id + "/h_params_" + run_id + ".txt"
    with open(h_params_path, "r") as file:
        h_params = file.readlines()
        for line in h_params:
            print(line[:-1])
    reader_path = run_id + "/" + run_id + ".h5"
    reader = emcee.backends.HDFBackend(reader_path) 
    # tau = reader.get_autocorr_time()
    # print(f"No. steps until autocorrelation: {tau}")
    ml_rates = pd.read_csv(f"{run_id}/ML_{run_id}.csv")

    if func == 1:
        plot_trace(reader, ml_rates)
    elif func == 2:
        sample_chain(reader)
    # corner_plot(reader)

if __name__ == "__main__":
    sampler_from_file(func=2)
    # get_maximum_likelihood()
    # dfs = get_data()
    # get_transition_count_avg(dfs)
    # run_leaf_uncert_parallel_pool(run_id="MUT2_mcmc_04-12-24")
    # run_mcmc_leaf_uncert()
    # plot_chain_from_file()
    # combine_posteriors_from_file(directory="markov_fitter_reports/emcee/24chains_25000steps_15000burnin_thin100_09-10-24")
    # combine_posteriors_from_file(directory="mcmc_29-11-24")
    # get_maximum_likelihood(run_id=run_id)
    # samples, sampler = run_mcmc()
    # plot_posterior(samples, sampler)
    # plot_posterior_fromfile(
    #     "markov_fitter_reports/emcee/24chains_25000steps_15000burnin/emcee_run_log_24-04-24.csv"
    # )
    # plot_posterior_fromfile("emcee_run_log.csv")
    # print(get_transition_count(get_data()))
