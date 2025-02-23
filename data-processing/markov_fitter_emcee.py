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
unif_lb, unif_ub = 0, 0.1 # lb and ub of uniform distribution for prior and initial values
ndim = 12 # no. rate parameters
nwalkers = 24 #24 # no. markov chains run in parallel
nsteps = 5000 #25000 # no. steps for each markov chain
nshuffle = 1# 25 # no. times the leaf dataset is shuffled
burnin = 2500 #15000 # these first iterations are discarded from the chain
thin = 10 #100 # only every thin iteration is kept
t = 1 # the value of time for each timstep in P=exp[Q*t]
wd = "leaves_full_21-9-23_MUT2.2_CLEAN" # the simulation run to fit to
# wd = "leaves_full_15-9-23_MUT1_CLEAN"
# wd = "leaves_full_10-02-25_MUT5_CLEAN"
cuton = 39 # the simulation data is removed before this step number and everything above is used to fit the CTMC
cutoff = 79 # the simulation data is cutoff at this step number before being used to fit the CTMC model
eq_init = False # whether to plot timeseries from equal numbers of each initial shape by dropping all leafids in excl
filt = ["u","l","d","c"] # only fit the ctmc to transitions from walks that started in these states
run_id = "test" # the name of the run WARNING - test will be overwritten by default

transition_map_rates = {
    "uu": (0, 0),"ul": (0, 1),"ud": (0, 2),"uc": (0, 3),
    "lu": (1, 0),"ll": (1, 1),"ld": (1, 2),"lc": (1, 3),
    "du": (2, 0),"dl": (2, 1),"dd": (2, 2),"dc": (2, 3),
    "cu": (3, 0),"cl": (3, 1),"cd": (3, 2),"cc": (3, 3),
}

labels = ["ul","ud","uc","lu","ld","lc","du","dl","dc","cu","cl","cd"]

# exclude these in concatenator if you want to infer from equal numbers of each initshape
excl = ["p0_121","p1_82","p2_195","p9_129","p5_249","pu3","p2_78_alt","p3_60", #unlobed
        "p8_1235","p1_35","p12b", # dissected
        "pc4","p12de","p7_437","p2_346_alt","p6_1155"] # compound

def init_env():
    """Create a run directory."""
    if os.path.exists(run_id):
        if not run_id == "test": # don't ask for confirmation if the run_id is "test"
            confirm = input(f"The directory '{run_id}' already exists. Do you want to replace it? (y/n): ")
            if confirm.lower() == 'y':
                shutil.rmtree(run_id)  # remove the existing directory and its contents
                os.mkdir(run_id)  # create a new directory
            else:
                print("Operation cancelled.")
                sys.exit() # terminate the program
        shutil.rmtree(run_id)
        os.mkdir(run_id)
    else:
        os.mkdir(run_id)  
    shutil.copy("markov_fitter_emcee.py", f"{run_id}/markov_fitter_emcee_{run_id}.txt") # save copy of code to run dir for reference


def concatenator():
    """Concatenate all the shape_report files from the random walks into a single dataframe."""
    dfs = []
    print(f"\nCurrent directory: {wd}\n")
    for leafdirectory in os.listdir(wd):
        if eq_init and leafdirectory in excl: # only fit to equal number of initial leaves
            continue
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
                    if not df.empty: # only append non-empty shape reports
                        df.insert(0, "leafid", leafdirectory)
                        df.insert(1, "walkid", int(re.findall(r"\d+", walkdirectory)[0]))
                        df = df.reset_index().rename(columns={"index": "step"}) # reset index and rename to step
                        dfs.append(df)
                    else:
                        print(f"Found empty shape report: {os.path.join(walkdirectory_path, file)} ...excluding from dataset.")
    return dfs


def get_transition_counts():
    """Get the total no. each transition type across the all walks in the dataset."""

    walks = concatenator()
    walks = pd.concat(walks)
    walks = pd.merge(walks, first_cats[["leafid", "first_cat"]], on="leafid")

    walks_sub = walks[["leafid","walkid","shape","step","first_cat"]]
    walks_sub.to_csv(f"{run_id}.csv", index=False)

    # Drop rows where leafid is "pc4" and walkid is 3
    walks = walks[~((walks["leafid"] == "pc4") & (walks["walkid"] == 3))] # should only remove 1 row

    walks["prevshape"] = walks["shape"].shift(+1) # get transitions by shifting shape columns down by one and combining
    walks["transition"] = walks["prevshape"] + walks["shape"]
    walks.loc[walks["step"] == 0, "transition"] = walks["first_cat"] + walks["shape"] # replace 0th step with first_cat + shape
    # walks.loc[walks["step"] == 0, "transition"] = None # replace 0th step with None
    
    walks = walks.loc[walks["step"] <= cutoff] # only fit to data from the first "cutoff" steps of each walk
    walks = walks.loc[walks["step"] >= cuton] # only fit to data after the first "cuton" steps of each walk
    walks = walks[walks["first_cat"].isin(filt)] # only fit to transitions from walks that started in these states
    
    counts = walks["transition"].value_counts().reset_index() # count no. transitions of each type

    # make sure all transitions are included in the counts DataFrame even if they are 0
    all_transitions = pd.DataFrame(
        [(a + b) for a in "uldc" for b in "uldc"], columns=["transition"] # df of all possible transitions
    )
    all_transitions["count"] = 0

    # merge with the existing counts DataFrame
    counts = all_transitions.merge(counts, on="transition", how="left").fillna(1) # fill missing counts with 1 for numerical stability
    counts["count"] = (counts["count_x"] + counts["count_y"]).astype(int)
    counts = counts[["transition", "count"]]
    print(counts)

    return counts


def log_prior(params):  # define a uniform prior from 0 to 0.1 for every transition rate
    """Uniform prior between unif_lb and unif_ub. Return -np.inf if any parameter is outside the prior range."""

    if all(unif_lb <= q <= unif_ub for q in params): # all q parameters must be within the prior range to return 0
        return 0
    return -np.inf


def log_likelihood(params, counts):
    """Calculate the log likelihood of the data given a set of transition rates Q."""

    Q = np.array(
        [
            [-(params[0] + params[1] + params[2]), params[0], params[1], params[2]],
            [params[3], -(params[3] + params[4] + params[5]), params[4], params[5]],
            [params[6], params[7], -(params[6] + params[7] + params[8]), params[8]],
            [params[9], params[10], params[11], -(params[9] + params[10] + params[11])],
        ]
    )
    log_likelihood = 0
    Pt = scipy.linalg.expm(Q * t)  # t is assumed to be the same for every transition
    for i, transition in enumerate(counts["transition"]):
        Pt_i = Pt[transition_map_rates[transition]] # get transition probabilty from Pt matrix
        counts_i = counts["count"][i] # get the count of the transition
        if Pt_i > 0:
            log_likelihood +=  counts_i * np.log(Pt_i) # see Kalbfleisch 1985 eq 3.2
        else:
            # log_likelihood = -np.inf # if the probability is 0, the log likelihood is -inf
            log_likelihood = -np.inf # if the probability is 0, the log likelihood is -inf
            break

    # print(log_likelihood)
    return log_likelihood


def log_probability(params, counts):
    """Calculate the log posterior probability of Q given the data by adding log prior and log likelihood."""
    
    lp = log_prior(params)
    if not np.isfinite(lp): # if any of the proposed parameters are outside the prior range, skip the likelihood calculation and return -np.inf
        return -np.inf
    
    # posterior is proportional to likelihood * prior therefore log(posterior) is proportional to log(likelihood) + log(prior)
    return lp + log_likelihood(params, counts)


def get_maximum_likelihood():
    """Get the maximum likelihood estimate of the transition rates Q given the data."""

    counts = get_transition_counts()

    nll = lambda Q: -log_likelihood(Q, counts)
    np.random.seed()
    init = np.random.uniform(unif_lb, unif_ub, ndim) # initialise 12 random numbers for Q matrix
    soln = optimize.minimize(nll, init)
    # soln = optimize.minimize(nll, init, method="trust-exact", tol=1e-5, options={"maxiter": 5000, "disp": True})
    Q_ml = soln.x
    Q_ml_df = pd.DataFrame(Q_ml).T
    Q_ml_df.to_csv(f"{run_id}/ML_{run_id}.csv", index=False)
    print(f"Maximum likelihood rates: {Q_ml_df}")
    return Q_ml_df


def run_leaf_uncert_parallel_pool():
    """Run parallelised MCMC inference to find the posterior distributions for all transition rate parameters given data."""

    counts = get_transition_counts()
    init_params = np.random.uniform(unif_lb, unif_ub, (nwalkers, ndim)) # generate initial values to fill Q matrix for each walker
    # set up file to save the run
    filename = f"{run_id}/{run_id}.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    with multiprocessing.Pool() as pool:
        np.random.seed() # different seed for each chain?
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(counts,),pool=pool, backend=backend)
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
    """Plot the trace of the MCMC chains for each parameter."""

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
            ax.set_ylim(unif_lb, unif_ub)
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
    print("\nHyper Parameters:")
    print(f"n_processes = {n_processes}")
    print(f"unif_lb = {unif_lb}, unif_ub = {unif_ub}")
    print(f"ndim = {ndim}")
    print(f"nwalkers = {nwalkers}")
    print(f"nsteps = {nsteps}")
    print(f"nshuffle = {nshuffle}")
    print(f"burnin = {burnin}")
    print(f"thin = {thin}")
    print(f"t = {t}")
    print(f"cuton = {cuton}")
    print(f"cutoff = {cutoff}")
    print(f"eq_init = {eq_init}")
    print(f"filt = {filt}")
    print(f"wd = {wd}")
    print(f"run_id = {run_id}")


def save_hyperparams():

    """Write hyperparameters to file in output directory."""
    
    print_hyperparams()
    output_file = f'{run_id}/h_params_{run_id}.txt'

    with open(output_file, 'w') as file:
        file.write(f"Run {run_id} MCMC Hyper Parameters:\n")
        file.write(f"n_processes = {n_processes}\n")
        file.write(f"unif_lb = {unif_lb}, unif_ub = {unif_ub}\n")
        file.write(f"ndim = {ndim}\n")
        file.write(f"nwalkers = {nwalkers}\n")
        file.write(f"nsteps = {nsteps}\n")
        file.write(f"nshuffle = {nshuffle}\n")
        file.write(f"burnin = {burnin}\n")
        file.write(f"thin = {thin}\n")
        file.write(f"t = {t}\n")
        file.write(f"cuton = {cuton}\n")
        file.write(f"cutoff = {cutoff}\n")
        file.write(f"eq_init = {eq_init}\n")
        file.write(f"filt = {filt}\n")
        file.write(f"wd = {wd}\n")

    print(f"Hyper parameters have been saved to {output_file}")


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
                init_env()
                save_hyperparams()
                run_leaf_uncert_parallel_pool()
            if func == 1:
                init_env()
                save_hyperparams()
                get_maximum_likelihood()
            if func == 2 or func == 3:
                sampler_from_file()
            if func == 4:
                counts = get_transition_counts()
                total = sum(counts["count"])
                print(f"total transitions {total}")

