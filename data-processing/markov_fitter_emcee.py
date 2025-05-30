import os
import shutil
import sys
import warnings
import multiprocessing
import re
import numpy as np
import pandas as pd
import scipy
from scipy import optimize
import emcee
import corner
from matplotlib import pyplot as plt
import scipy.linalg
from dataprocessing import first_cats
from dataprocessing import first_cats

# set default parameters
FUNC = 0  # default value is running the mcmc inference
N_PROCESSES = 10  # for parallelisation
# lb and ub of uniform distribution for prior and initial values
UNIF_LB, UNIF_UB = 0, 0.1
NDIM = 12  # no. rate parameters
NWALKERS = 24  # 24 # no. markov chains run in parallel
NSTEPS = 5000  # 25000 # no. steps for each markov chain
NSHUFFLE = 1  # 25 # no. times the leaf dataset is shuffled
BURNIN = 2500  # 15000 # these first iterations are discarded from the chain
THIN = 10  # 100 # only every THIN iteration is kept
T = 1  # the value of time for each timstep in P=exp[Q*t]
WD = "leaves_full_13-03-25_MUT2_CLEAN"  # the simulation run to fit to
# WD = "leaves_full_15-9-23_MUT1_CLEAN"
# WD = "leaves_full_10-02-25_MUT5_CLEAN"
CUTON = 0  # sim data is removed before this step before CTMC fitting
CUTOFF = 999  # sim data is CUTOFF at this step before fitting the CTMC
RESET_FIRST_CAT = False  # reset first_cat to the shape at step CUTON
EQ_INIT = False  # plot timeseries from equal numbers of each initial shape
# only fit the ctmc to transitions from walks that started in these states
FILT = ["u", "l", "d", "c"]
L_VER = 0  # version of likelihood function 0 - transition = prev + curr,
# 1 - transition = first_cat + curr (aka from0)
USE_PWALKS = False  # use pseudo walks instead of real walks to fit model
P_WALK_LEN = 160  # length of pseudo walks
N_PWS = 10  # number of pseudo walks to generate per walk
RUN_ID = "test"  # the name of the run - test will be overwritten by default

transition_map_rates = {
    "uu": (0, 0), "ul": (0, 1), "ud": (0, 2), "uc": (0, 3),
    "lu": (1, 0), "ll": (1, 1), "ld": (1, 2), "lc": (1, 3),
    "du": (2, 0), "dl": (2, 1), "dd": (2, 2), "dc": (2, 3),
    "cu": (3, 0), "cl": (3, 1), "cd": (3, 2), "cc": (3, 3),
}

labels = ["ul", "ud", "uc", "lu", "ld",
          "lc", "du", "dl", "dc", "cu", "cl", "cd"]

# exclude these in concatenator to infer from equal numbers of each initshape
excl = ["p0_121", "p1_82", "p2_195", "p9_129", "p5_249", "pu3", "p2_78_alt",
        "p3_60",  # unlobed
        "p8_1235", "p1_35", "p12b",  # dissected
        "pc4", "p12de", "p7_437", "p2_346_alt", "p6_1155"]  # compound


def init_env():
    """Create a run directory. Overwrite if it already exists."""
    if os.path.exists(RUN_ID):
        if not RUN_ID == "test":  # ignore warning if RUN_ID="test"
            confirm = input(
                f"The directory '{RUN_ID}' already exists. Do you want "
                "to replace it? (y/n): ")
            if confirm.lower() == 'y':
                shutil.rmtree(RUN_ID)
                os.mkdir(RUN_ID)
            else:
                print("Operation cancelled.")
                sys.exit()
        shutil.rmtree(RUN_ID)
        os.mkdir(RUN_ID)
    else:
        os.mkdir(RUN_ID)
    # save copy of code to run dir for reference
    shutil.copy("markov_fitter_emcee.py",
                f"{RUN_ID}/markov_fitter_emcee_{RUN_ID}.txt")


def concatenator():
    """Concatenate all the shape_report files from the random walks into a
    single dataframe."""
    dfs = []
    print(f"\nCurrent directory: {WD}\n")
    for leafdirectory in os.listdir(WD):
        if EQ_INIT and leafdirectory in excl:
            continue  # skip excluded leafids if EQ_INIT is True
        leafdirectory_path = os.path.join(WD, leafdirectory)
        if os.path.isdir(leafdirectory_path):
            for walkdirectory in os.listdir(leafdirectory_path):
                walkdirectory_path = os.path.join(
                    leafdirectory_path, walkdirectory)
                for file in os.listdir(walkdirectory_path):
                    if (
                        file.endswith(".csv")
                        and "shape_report" in file
                        and "shape_report1" not in file
                    ):
                        df = pd.read_csv(os.path.join(
                            walkdirectory_path, file))
                        if not df.empty:  # only append non-empty shape reports
                            df.insert(0, "leafid", leafdirectory)
                            df.insert(1, "walkid", int(
                                re.findall(r"\d+", walkdirectory)[0]))
                            # reset index and rename to step
                            df = df.reset_index().rename(
                                columns={"index": "step"})
                            dfs.append(df)
                        else:
                            print(
                                "Found empty shape report: "
                                f"{os.path.join(walkdirectory_path, file)}"
                                " ...excluding from dataset.")
    return dfs


def gen_pseudo_walks():
    """Generate extra pseudo walks by taking multiple different contiguous
    frames from each walk, with different start and end steps. Saves to
    .csv file."""

    dfs = concatenator()
    concat = pd.concat(dfs, ignore_index=True)
    concat = concat[["leafid", "walkid", "shape", "step"]]

    pws = []
    for _, leaf in concat.groupby("leafid"):
        for _, walk in leaf.groupby("walkid"):
            start = 0
            max_start = len(walk) - P_WALK_LEN
            step = max_start // N_PWS  # equal intervals start and max start
            for i in range(N_PWS):
                pw = walk.iloc[start:start + P_WALK_LEN].copy()
                pw["first_cat"] = pw["shape"].values[0]  # reset first cat
                pw["leafid"] = f"{pw['leafid'].values[0]}_" + \
                    f"{pw['walkid'].values[0]}_{i}"
                pw["walkid_real"] = pw["walkid"].values[0]
                pw["walkid"] = f"{pw['walkid'].values[0]}_{i}"  # pwalkid
                # pw["leafid"] = f"{pw['leafid'].values[0]}_{i}"  # pwleafid
                pw["step_real"] = pw["step"]
                pw["step"] = pw["step"] - pw["step"].min()  # start steps at 0
                pws.append(pw)
                start += step  # next pwalk start

    pws = pd.concat(pws, ignore_index=True)
    pws.to_csv(f"pwalks_{N_PWS}_{P_WALK_LEN}_{WD}.csv", index=False)
    print(f"Pseudo walks saved to pwalks_{N_PWS}_{P_WALK_LEN}_{WD}.csv")

    return pws


def get_transition_counts():
    """Get the total no. each transition type across the all walks in the
    dataset."""

    if USE_PWALKS:
        # if using pseudo walks, read in the pseudo walks
        walks = pd.read_csv(f"pwalks_{N_PWS}_{P_WALK_LEN}_{WD}.csv")
    else:
        walks = concatenator()
        walks = pd.concat(walks)
        walks = pd.merge(
            walks, first_cats[["leafid", "first_cat"]], on="leafid")

        walks_sub = walks[["leafid", "walkid", "shape", "step", "first_cat"]]
        walks_sub.to_csv(f"{RUN_ID}.csv", index=False)

        # Drop rows where leafid "pc4" and walkid 3 should only remove 1 row
        # walks = walks[~((walks["leafid"] == "pc4") & (walks["walkid"] == 3))]

    # get transitions by shifting shape columns down by one and combining
    walks["prevshape"] = walks["shape"].shift(+1)
    walks["transition"] = walks["prevshape"] + walks["shape"]
    walks.loc[walks["step"] == 0, "transition"] = walks["first_cat"] + \
        walks["shape"]  # replace 0th step with first_cat + shape

    # only fit to data from and including the first "CUTOFF" steps of each walk
    walks = walks.loc[walks["step"] <= CUTOFF]
    # only fit to data after and including the first "CUTON" steps of each walk
    walks = walks.loc[walks["step"] >= CUTON]
    # only fit to transitions from walks that started in these states
    walks = walks[walks["first_cat"].isin(FILT)]

    # redefine first_cat to shape at step CUTON
    if RESET_FIRST_CAT:
        walks["first_cat"] = walks.groupby(["leafid", "walkid"])[
            "shape"].transform("first")
        step_cuton = walks[walks["step"] == CUTON]
        mis = step_cuton[step_cuton["first_cat"] !=
                         step_cuton["shape"]]  # check for mismatches
        assert len(mis) == 0, f"Mismatch in first_cat and shape at {CUTON}"

    # count no. transitions of each type
    count = walks["transition"].value_counts().reset_index()

    # ensure all transitions included in counts even if they are 0
    all_transitions = pd.DataFrame(
        # df of all possible transitions
        [(a + b) for a in "uldc" for b in "uldc"], columns=["transition"]
    )
    all_transitions["count"] = 0

    # merge with the existing counts DataFrame
    count = all_transitions.merge(count, on="transition", how="left").fillna(
        1)  # fill missing counts with 1 for numerical stability
    count["count"] = (count["count_x"] + count["count_y"]).astype(int)
    count = count[["transition", "count"]]
    print(count)

    return count


def get_transition_counts_alt():
    """Get total no. each transition with initial state always being first_cat 
    and end state is for each time step. Returns no. each transition type for
    each time interval."""

    if USE_PWALKS:
        # if using pseudo walks, read in the pseudo walks
        walks = pd.read_csv(f"pwalks_{N_PWS}_{P_WALK_LEN}_{WD}.csv")
    else:
        walks = concatenator()
        walks = pd.concat(walks)
        walks = pd.merge(
            walks, first_cats[["leafid", "first_cat"]], on="leafid")

    walks_sub = walks[["leafid", "walkid", "shape", "step", "first_cat"]]

    def get_trans_group(gp):
        tr = []
        init_state = gp.iloc[0]["first_cat"]  # prev shape always first_cat
        for i in range(0, len(gp)):
            state_i = gp.iloc[i]["shape"]  # next shape is current
            # append transitions and time interval
            tr.append({"transition": init_state + state_i, "t": i + 1})
        return pd.DataFrame(tr)

    trans = walks_sub.groupby(  # count transitions in each walk
        ["leafid", "walkid", "first_cat"])[
            ["leafid", "walkid", "shape", "step", "first_cat"]
    ].apply(get_trans_group)
    trans = trans.reset_index(level=["leafid", "walkid", "first_cat"])
    # check for missing transitions
    assert len(trans) == len(walks_sub), "Mismatch in transition counts"
    # check first_cat always eqals from shape
    assert trans["first_cat"].equals(trans["transition"].str[0]), \
        "Mismatch in first_cat and prev state in transition"
    count = trans[["transition", "t"]].value_counts().reset_index()

    return count


def log_likelihood_alt(params, count):
    """Calculate log likelihood using transitions where intitial state is
    always at first_cat and end state is for each time step."""

    q = np.array(
        [
            [-(params[0] + params[1] + params[2]),
             params[0], params[1], params[2]],
            [params[3], -(params[3] + params[4] + params[5]),
             params[4], params[5]],
            [params[6], params[7], -
                (params[6] + params[7] + params[8]), params[8]],
            [params[9], params[10], params[11], -
                (params[9] + params[10] + params[11])],
        ]
    )
    log_l = 0

    for t in sorted(count["t"].unique()):  # calc pt for each unique t
        counts_sub = count[count["t"] == t].reset_index(drop=True)
        pt = scipy.linalg.expm(q * t)
        for j, transition in enumerate(counts_sub["transition"]):
            pt_j = pt[transition_map_rates[transition]]
            # get freq of each transition for this time length t
            count_j = counts_sub["count"][j]
            if pt_j > 0:
                log_l += count_j * np.log(pt_j)
            else:
                log_l = -np.inf
                break

    return log_l


def log_prior(params):
    """Uniform prior between UNIF_LB and UNIF_UB for all transition rates.
    Return -np.inf if any parameter is outside the prior range."""

    # all q parameters must be within the prior range to return 0
    if all(UNIF_LB <= q <= UNIF_UB for q in params):
        return 0
    return -np.inf


def log_likelihood(params, count):
    """Calculate the log likelihood of the data given a set of transition
    rates Q."""

    q = np.array(
        [
            [-(params[0] + params[1] + params[2]),
             params[0], params[1], params[2]],
            [params[3], -(params[3] + params[4] + params[5]),
             params[4], params[5]],
            [params[6], params[7], -
                (params[6] + params[7] + params[8]), params[8]],
            [params[9], params[10], params[11], -
                (params[9] + params[10] + params[11])],
        ]
    )
    log_l = 0
    # T is assumed to be the same for every transition
    pt = scipy.linalg.expm(q * T)
    for i, transition in enumerate(count["transition"]):
        # get transition probabilty from Pt matrix
        pt_i = pt[transition_map_rates[transition]]
        count_i = count["count"][i]  # get the count of the transition
        if pt_i > 0:
            log_l += count_i * np.log(pt_i)  # see Kalbfleisch 1985 eq 3.2
        else:
            log_l = -np.inf  # if prob is 0, log likelihood is -inf
            break

    return log_l


def log_probability(params, count):
    """Calculate the log posterior probability of Q given the data by adding
    log prior and log likelihood."""

    lp = log_prior(params)      # if any of the proposed parameters are outside
    if not np.isfinite(lp):     # the prior range, skip likelihood calculation
        return -np.inf          # and return -np.inf
    # posterior is proportional to likelihood * prior therefore log(posterior)
    # is proportional to log(likelihood) + log(prior)
    return lp + log_likelihood(params, count)


def get_maximum_likelihood():
    """Get the maximum likelihood estimate of the transition rates Q given
    the data."""

    count = get_transition_counts()
    if L_VER == 1:
        count = get_transition_counts_alt()

    def nll(q):  # negative log likelhood wrapper
        if L_VER == 0:
            return -log_likelihood(q, count)
        elif L_VER == 1:
            return -log_likelihood_alt(q, count)

    np.random.seed()
    # initialise 12 random numbers for q matrix
    init = np.random.uniform(UNIF_LB, UNIF_UB, NDIM)
    soln = optimize.minimize(nll, init)
    q_ml = soln.x
    q_ml_df = pd.DataFrame(q_ml).T
    q_ml_df.to_csv(f"{RUN_ID}/ML_{RUN_ID}.csv", index=False)
    print(f"Maximum likelihood rates: {q_ml_df}")
    return q_ml_df


def run_leaf_uncert_parallel_pool():
    """Run parallelised MCMC inference to find the posterior distributions for
    all transition rate parameters given data."""

    count = get_transition_counts()
    # generate initial values to fill Q matrix for each walker
    init_params = np.random.uniform(UNIF_LB, UNIF_UB, (NWALKERS, NDIM))
    filename = f"{RUN_ID}/{RUN_ID}.h5"     # set up file to save the chains
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(NWALKERS, NDIM)
    with multiprocessing.Pool() as pool:
        np.random.seed()  # different seed for each chain?
        sampler = emcee.EnsembleSampler(
            NWALKERS, NDIM, log_probability, args=(count,), pool=pool,
            backend=backend)
        _ = sampler.run_mcmc(
            init_params, NSTEPS, skip_initial_state_check=True, progress=True
        )
    q_ml = get_maximum_likelihood()
    plot_trace(sampler, q_ml)
    sample_chain(sampler)
    autocorrelation_analysis(sampler)


def corner_plot(sampler):
    """Visualise correlations between posterior distributions for each
    transition"""
    flat_samples = sampler.get_chain(flat=True)
    corner.corner(flat_samples, labels=labels)
    plt.tight_layout()
    plt.show()


def plot_trace(sampler, ml_rates):
    """Plot the trace of the MCMC chains for each parameter."""

    fig, axes = plt.subplots(4, 4, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    idx = 0
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i == j:
                ax.axis("off")
                continue
            # from left to right the indicies represent: step, chain, parameter
            ax.plot(samples[:, :, idx], c="C0", alpha=0.3)
            ax.axhline(y=ml_rates.iloc[0, idx], c="C1", linestyle="--")
            ax.set_ylabel(labels[idx])
            ax.set_ylim(UNIF_LB, UNIF_UB)
            idx += 1
    fig.supxlabel("Iteration")
    plt.tight_layout()
    fig.savefig(f"{RUN_ID}/trace_{RUN_ID}.pdf", format="pdf")
    plt.show()


def sample_chain(sampler):
    """Reduce the size of the saved chain by discarding the BURNIN and
    rounding each step and recording only every THIN step."""

    flat_samples = sampler.get_chain(discard=BURNIN, thin=THIN, flat=True)
    flat_samples_df = pd.DataFrame(
        flat_samples, columns=list(range(flat_samples.shape[1])))
    flat_samples_df.to_csv(f"{RUN_ID}/posteriors_{RUN_ID}.csv", index=False)


def sampler_from_file():
    """Read MCMC chains from file and plot the trace and corner plot."""
    h_params_path = RUN_ID + "/h_params_" + RUN_ID + ".txt"
    with open(h_params_path, "r", encoding="utf-8") as file:
        h_params = file.readlines()
        print("\n")
        for line in h_params:
            print(line[:-1])
        print("\n")
    reader_path = RUN_ID + "/" + RUN_ID + ".h5"
    reader = emcee.backends.HDFBackend(reader_path)
    # tau = reader.get_autocorr_time()
    # print(f"No. steps until autocorrelation: {tau}")
    ml_rates = pd.read_csv(f"{RUN_ID}/ML_{RUN_ID}.csv")

    if FUNC == 2:
        plot_trace(reader, ml_rates)
    elif FUNC == 3:
        sample_chain(reader)
    # corner_plot(reader)


def autocorrelation_analysis(sampler):
    """Calculate the autocorrelation time of the MCMC chains."""
    tau = sampler.get_autocorr_time()
    print(f"No. steps until autocorrelation: {tau}")


def print_hyperparams():
    """Print hyperparameters to the console."""
    print("\nHyper Parameters:")
    print(f"N_PROCESSES = {N_PROCESSES}")
    print(f"UNIF_LB = {UNIF_LB}, UNIF_UB = {UNIF_UB}")
    print(f"NDIM = {NDIM}")
    print(f"NWALKERS = {NWALKERS}")
    print(f"NSTEPS = {NSTEPS}")
    print(f"NSHUFFLE = {NSHUFFLE}")
    print(f"BURNIN = {BURNIN}")
    print(f"THIN = {THIN}")
    print(f"t = {T}")
    print(f"CUTON = {CUTON}")
    print(f"CUTOFF = {CUTOFF}")
    print(f"RESET_FIRST_CAT = {RESET_FIRST_CAT}")
    print(f"EQ_INIT = {EQ_INIT}")
    print(f"FILT = {FILT}")
    print(f"L_VER = {L_VER}")
    print(f"USE_PWALKS = {USE_PWALKS}")
    print(f"P_WALK_LEN = {P_WALK_LEN}")
    print(f"N_PWS = {N_PWS}")
    print(f"WD = {WD}")
    print(f"RUN_ID = {RUN_ID}")


def save_hyperparams():
    """Write hyperparameters to file in output directory."""

    print_hyperparams()
    output_file = f'{RUN_ID}/h_params_{RUN_ID}.txt'

    with open(output_file, 'w', encoding="utf-8") as file:
        file.write(f"Run {RUN_ID} MCMC Hyper Parameters:\n")
        file.write(f"N_PROCESSES = {N_PROCESSES}\n")
        file.write(f"UNIF_LB = {UNIF_LB}, UNIF_UB = {UNIF_UB}\n")
        file.write(f"NDIM = {NDIM}\n")
        file.write(f"NWALKERS = {NWALKERS}\n")
        file.write(f"NSTEPS = {NSTEPS}\n")
        file.write(f"NSHUFFLE = {NSHUFFLE}\n")
        file.write(f"BURNIN = {BURNIN}\n")
        file.write(f"THIN = {THIN}\n")
        file.write(f"t = {T}\n")
        file.write(f"CUTON = {CUTON}\n")
        file.write(f"CUTOFF = {CUTOFF}\n")
        file.write(f"RESET_FIRST_CAT = {RESET_FIRST_CAT}\n")
        file.write(f"EQ_INIT = {EQ_INIT}\n")
        file.write(f"FILT = {FILT}\n")
        file.write(f"L_VER = {L_VER}\n")
        file.write(f"USE_PWALKS = {USE_PWALKS}\n")
        file.write(f"P_WALK_LEN = {P_WALK_LEN}\n")
        file.write(f"N_PWS = {N_PWS}\n")
        file.write(f"WD = {WD}\n")

    print(f"Hyper parameters have been saved to {output_file}")


def print_help():
    """Print help message for the programme."""
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
                        5   ...generate pseudo walks from simulation data
    """
    print(help_message)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "-h" in args:
        print_help()

    else:
        if "-id" in args:
            RUN_ID = str(args[args.index("-id") + 1])
        else:
            warnings.warn(f"No RUN_ID specified, defaulting to {RUN_ID}")
        if "-d" in args:
            data = int(args[args.index("-d") + 1])
            if data == 1:
                WD = "leaves_full_15-9-23_MUT1_CLEAN"
            elif data == 2:
                WD = "leaves_full_21-9-23_MUT2.2_CLEAN"
        if "-f" in args:
            FUNC = int(args[args.index("-f") + 1])
            if FUNC == 0:
                init_env()
                save_hyperparams()
                run_leaf_uncert_parallel_pool()
            if FUNC == 1:
                init_env()
                save_hyperparams()
                get_maximum_likelihood()
            if FUNC == 2 or FUNC == 3:
                sampler_from_file()
            if FUNC == 4:
                counts = get_transition_counts()
                total = sum(counts["count"])
                print(f"total transitions {total}")
            if FUNC == 5:
                print_hyperparams()
                gen_pseudo_walks()
