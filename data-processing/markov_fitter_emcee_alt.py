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
import sys
import warnings
import shutil
from dataprocessing import first_cats

# set default parameters
func = 0 # default value is running the mcmc inference
n_processes = 10 # for parallelisation
init_lb, init_ub = 0, 0.1 # lb and ub of uniform distribution for initial values
ndim = 12 # no. rate parameters
nwalkers = 24 # no. markov chains run in parallel
nsteps = 5000 #25000 # no. steps for each markov chain
nshuffle = 1# 25 # no. times the leaf dataset is shuffled
burnin = 2500 #15000 # these first iterations are discarded from the chain
thin = 10 #100 # only every thin iteration is kept
t = 1 # the value of time for each timstep in P=exp[Q*t]
wd = "leaves_full_21-9-23_MUT2.2_CLEAN" # the simulation run to fit to
# wd = "leaves_full_15-9-23_MUT1_CLEAN"
cutoff = 59 # the simulation data is cutoff at this step number before being used to fit the CTMC model
run_id = "test" # the name of the run
# run_id = "MUT2_mcmc_03-12-24"
# run_id = "MUT2_mcmc_08-12-24"
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

def init_env():
    if os.path.exists(run_id):
        confirm = input(f"The directory '{run_id}' already exists. Do you want to replace it? (y/n): ")
        if confirm.lower() == 'y':
            shutil.rmtree(run_id)  # remove the existing directory and its contents
            os.mkdir(run_id)  # create a new directory
        else:
            print("Operation cancelled.")
            sys.exit() # terminate the program
    else:
        os.mkdir(run_id)  # create a new directory

def concatenator():
    dfs = []
    print(f"\n\nCurrent directory: {wd}\n\n")
    for leafdirectory in os.listdir(wd):
        # print(f"Current = {leafdirectory}")
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
                    df = df.reset_index().rename(columns={"index": "step"}) # reset index and rename to step
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
                if i > cutoff: # only fit to data from the first "cutoff" steps of each walk
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

    counts = leaf_sum.groupby(["transition"])["sum"].agg(["sum"]).rename(columns={"sum":"count"}).reset_index()

    return leaf_sum

def get_transition_counts():
    walks = concatenator()
    walks = pd.concat(walks)
    walks = pd.merge(walks, first_cats[["leafid", "first_cat"]], on="leafid")

    walks_sub = walks[["leafid","walkid","shape","step","first_cat"]]
    walks_sub.to_csv("MUT2.2_04-02-25.csv", index=False)

    # Drop rows where leafid is "pc4" and walkid is 3
    walks = walks[~((walks["leafid"] == "pc4") & (walks["walkid"] == 3))] # should only remove 1 row

    walks["prevshape"] = walks["shape"].shift(+1) # get transitions by shifting shape columns down by one and combining
    walks["transition"] = walks["prevshape"] + walks["shape"]
    walks.loc[walks["step"] == 0, "transition"] = walks["first_cat"] + walks["shape"] # replace 0th step with first_cat + shape
    # walks.loc[walks["step"] == 0, "transition"] = None # replace 0th step with first_cat + shape
    

    walks = walks.loc[walks["step"] <= cutoff] # only fit to data from the first "cutoff" steps of each walk

    counts = walks["transition"].value_counts().reset_index()

    return counts

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


def log_likelihood(params, counts):
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
    for i, transition in enumerate(counts["transition"]):
        log_likelihood +=  counts["count"][i] * np.log( # see Kalbfleisch 1985 eq 3.2
            Pt[transition_map_rates[transition]]
        )
        if np.isnan(log_likelihood):
            log_likelihood = -np.inf
            break
    return log_likelihood


def log_probability(params, t_mean):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, t_mean)

def get_maximum_likelihood():
    # dfs = get_data()
    # t_mean, t_std, t_sem, t_lb, t_ub = get_transition_count_avg(dfs)
    # print(t_mean)
    # counts = get_leaf_transitions(dfs)
    # print(counts)
    # counts = counts.groupby(["transition"])["sum"].agg(["sum"]).rename(columns={"sum":"count"}).reset_index()
    # print(counts)
    counts = get_transition_counts()

    nll = lambda Q: -log_likelihood(Q, counts)
    np.random.seed()
    init = np.random.uniform(init_lb, init_ub, ndim) # initialise 12 random numbers for Q matrix
    soln = optimize.minimize(nll, init)
    Q_ml = soln.x
    Q_ml_df = pd.DataFrame(Q_ml).T
    Q_ml_df.to_csv(f"{run_id}/ML_{run_id}.csv", index=False)
    print(f"Maximum likelihood rates: {Q_ml_df}")
    return Q_ml_df


def run_leaf_uncert_parallel_pool():
    # os.mkdir(run_id)
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
        file.write(f"cutoff = {cutoff}\n")
        file.write(f"wd = {wd}\n")

    print(f"Hyper parameters have been saved to {output_file}")

    # dfs = get_data()
    global transitions
    # t_mean, t_std, t_sem, t_lb, t_ub = get_transition_count_avg(dfs)
    # counts = get_leaf_transitions(dfs)
    # counts = counts.groupby(["transition"])["sum"].agg(["sum"]).rename(columns={"sum":"count"}).reset_index()
    # print(counts)
    counts = get_transition_counts()
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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(t_mean,),pool=pool, backend=backend)
        state = sampler.run_mcmc(
            init_params, nsteps, skip_initial_state_check=True, progress=True
        )
    Q_ml = get_maximum_likelihood()
    plot_trace(sampler, Q_ml)
    sample_chain(sampler)
    autocorrelation_analysis(sampler)
    
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
    fig.supxlabel("Iteration")
    plt.tight_layout()
    fig.savefig(f"{run_id}/trace_{run_id}.pdf", format="pdf")
    plt.show()

def sample_chain(sampler):
    """Reduce the size of the saved chain by discarding the burnin and rounding each step and recording only every thin step."""
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    flat_samples_df = pd.DataFrame(flat_samples, columns=list(range(flat_samples.shape[1])))
    flat_samples_df.to_csv(f"{run_id}/posteriors_{run_id}.csv",index=False)

def sampler_from_file():
    h_params_path = run_id + "/h_params_" + run_id + ".txt"
    with open(h_params_path, "r") as file:
        h_params = file.readlines()
        print("\n")
        for line in h_params:
            print(line[:-1])
        print("\n")
    reader_path = run_id + "/" + run_id + ".h5"
    reader = emcee.backends.HDFBackend(reader_path) 
    # tau = reader.get_autocorr_time()
    # print(f"No. steps until autocorrelation: {tau}")
    ml_rates = pd.read_csv(f"{run_id}/ML_{run_id}.csv")

    if func == 2:
        plot_trace(reader, ml_rates)
    elif func == 3:
        sample_chain(reader)
    # corner_plot(reader)

def autocorrelation_analysis(sampler):
    tau = sampler.get_autocorr_time()
    print(f"No. steps until autocorrelation: {tau}")

def print_hyperparams():
    print("\nHyper Parameters:\n")
    print(f"n_processes = {n_processes}")
    print(f"init_lb = {init_lb}, init_ub = {init_ub}")
    print(f"ndim = {ndim}")
    print(f"nwalkers = {nwalkers}")
    print(f"nsteps = {nsteps}")
    print(f"nshuffle = {nshuffle}")
    print(f"burnin = {burnin}")
    print(f"thin = {thin}")
    print(f"t = {t}")
    print(f"cutoff = {cutoff}")
    print(f"wd = {wd}")
    print(f"run_id = {run_id}\n\n")

def print_help():
    help_message = """
    Usage: python3 markov_fitter_emcee_alt.py [options]

    Options:
        -h              Show this help message and exit.
        -id [run id]    The name given to the mcmc/mle run.
        -d  [sim data]  Specify the simulation data to fit the model to
                        1   ...MUT1
                        2   ...MUT2
        -f  [function]  Pass function you want to perform:
                        0   ...run mcmc inference
                        1   ...run mle inference
                        2   ...plot mcmc trace from file
                        3   ...export mcmc posteriors from file
                        4   ...get transition counts from simulation data
    """
    print(help_message)

if __name__ == "__main__":
    args = sys.argv[1:]
    if "-h" in args:
        print_help()
    
    else:
        if "-id" in args:
            run_id = str(args[args.index("-id") + 1])
        else:
            warnings.warn(f"No run_id specified, defaulting to {run_id}")
        if "-d" in args:
            data = int(args[args.index("-d") + 1])
            if data == 1:
                wd = "leaves_full_15-9-23_MUT1_CLEAN"
            elif data == 2:
                wd = "leaves_full_21-9-23_MUT2.2_CLEAN"
        if "-f" in args:
            func = int(args[args.index("-f") + 1])
            if func  == 0:
                print_hyperparams()
                init_env()
                run_leaf_uncert_parallel_pool()
            if func == 1:
                print_hyperparams()
                init_env()
                get_maximum_likelihood()
            if func == 2 or func == 3:
                sampler_from_file()
            if func == 4:
                dfs = get_data()
                counts = get_leaf_transitions(dfs)
                print(counts)
                print(sum(counts["count"]))

