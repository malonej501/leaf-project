import pandas as pd
import numpy as np
import scipy
import copy
from dataprocessing import concatenator, first_cats
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import cProfile


nsteps = 10000
nchains = 8
lb = 0  # lower bound of inferred rate parameters
ub = 0.1  # upper bound of inferred rate parameters
step_size = 0.05  # random numbers are picked uniformly between + and - this number which make the jump matrix
# for lb0 ub100, step_size10 seems to work well
# can lower to 5 if running for nsteps >= ~10000
# step parameters
mean = 0
stdev = 0.005
# for acceptance=0.99 stdev=0.005 seems to work well


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


dfs = get_data()


def likelihood_rates(Q):
    L_data = 1
    for walk in dfs:
        if not walk.empty:
            L_walk = 1
            t = 0
            initial_state = walk["first_cat"][0]
            steps = walk["shape"].tolist()
            for i, curr in enumerate(steps):
                if i == 0:
                    prev = initial_state
                else:
                    t += 1
                    prev = steps[i - 1]
                    transition = prev + curr
                    Pt = scipy.linalg.expm(Q * t)
                    # L_walk *= Pt[transition_map_rates[transition]]
                    L_walk += np.log(Pt[transition_map_rates[transition]])
        # rate_map = {}
        # L_data += np.log(L_walk)
        # LL_data += L_walk
        L_data += L_walk

    return L_data


def load_Pts(Q):
    Pts_length = 120
    Pts = []
    for t in range(1, Pts_length):
        Qt = Q * t
        Pt = scipy.linalg.expm(Qt)
        Pts.append(Pt)
        if len(Pts) >= 2 and np.allclose(
            Pts[-1], Pts[-2], atol=1e-10
        ):  # if the probabilities converge, don't bother computing remaining Pt matrices, just append equilibrium value until list is full
            current_len = len(Pts)
            while current_len < Pts_length:
                Pts.append(Pts[-1])
                current_len += 1
            break
    return Pts


def likelihood_rates_frominitial(Q):
    L_data = 1
    Pts = load_Pts(Q)
    for walk in dfs:
        if not walk.empty:
            L_walk = 1
            initial_state = walk["first_cat"][0]
            steps = walk["shape"].tolist()
            for i, curr in enumerate(steps):
                transition = initial_state + curr
                Pt = Pts[i]
                L_walk *= Pt[transition_map_rates[transition]]
        # rate_map = {}
        # L_data += np.log(L_walk)
        L_data += np.log(L_walk)

    return L_data


def likelihood_rates_t1(Q):
    t = 1
    L_data = 0
    Pt = scipy.linalg.expm(Q * t)
    for walk in dfs:
        if not walk.empty:
            L_walk = 0  # set to 1 if *= to get probability
            initial_state = walk["first_cat"][0]
            steps = walk["shape"].tolist()
            for i, curr in enumerate(steps):
                if i == 0:
                    prev = initial_state
                else:
                    prev = steps[i - 1]
                    transition = prev + curr
                    L_walk += np.log(Pt[transition_map_rates[transition]])
            L_data += L_walk

    return L_data


def metropolis_hastings_rates(chain_id):
    np.random.seed()
    report = []
    location_l = 0
    while np.isinf(location_l) or location_l == 0:
        init = np.random.uniform(lb, ub, 12)
        Q_init = np.array(
            [
                [-(init[0] + init[1] + init[2]), init[0], init[1], init[2]],
                [init[3], -(init[3] + init[4] + init[5]), init[4], init[5]],
                [init[6], init[7], -(init[6] + init[7] + init[8]), init[8]],
                [init[9], init[10], init[11], -(init[9] + init[10] + init[11])],
            ]
        )
        location = Q_init
        location_l = likelihood_rates_t1(location)
    proposal_l = 0
    step = 0
    for step in range(nsteps):
        proposal_l = 0
        proposal = None
        proposal_nondiag = None
        while (
            np.isinf(proposal_l)
            or proposal_l == 0
            or np.any(
                (proposal_nondiag < lb) | (proposal_nondiag > ub)
            )  # constrain non-diagonal rates to be between lb and ub
        ):
            # jump = np.random.uniform(-step_size, step_size, 12)
            jump = np.random.normal(mean, stdev, 12)
            Q_jump = np.array(
                [
                    [-(jump[0] + jump[1] + jump[2]), jump[0], jump[1], jump[2]],
                    [jump[3], -(jump[3] + jump[4] + jump[5]), jump[4], jump[5]],
                    [jump[6], jump[7], -(jump[6] + jump[7] + jump[8]), jump[8]],
                    [jump[9], jump[10], jump[11], -(jump[9] + jump[10] + jump[11])],
                ]
            )
            proposal = location + Q_jump
            proposal_nondiag = proposal[~np.eye(proposal.shape[0], dtype=bool)]
            proposal_l = likelihood_rates_t1(proposal)
        # location_l += 10000
        # proposal_l += 10000
        # acceptance_ratio = proposal_l / location_l  # for maximising likelihood
        # acceptance_ratio = location_l / proposal_l  # for maximising log_likelihood
        # acceptance_ratio = proposal_l / location_l
        acceptance_ratio = np.exp(proposal_l - location_l)
        # location_l -= 10000
        # proposal_l -= 10000
        # print(proposal_l / location_l)
        print(step, location_l, proposal_l, acceptance_ratio)
        # acceptance_threshold = np.random.uniform(0.99, 1)
        acceptance_threshold = np.random.uniform(0, 1)
        if acceptance_ratio > acceptance_threshold:
            location = copy.deepcopy(proposal)
            location_l = copy.deepcopy(proposal_l)
            # print(f"Step: {step}, Likelihood: {location_l}\n{location}")
        else:
            continue
        report.append([step, location_l, *location.flatten().tolist()])
        report_df = pd.DataFrame(
            report,
            columns=[
                "step",
                "location_LL",
                "q00",
                "q01",
                "q02",
                "q03",
                "q10",
                "q11",
                "q12",
                "q13",
                "q20",
                "q21",
                "q22",
                "q23",
                "q30",
                "q31",
                "q32",
                "q33",
            ],
        )
        report_df.to_csv(f"markov_fitter_reports/chain_{chain_id}.csv", index=False)
    print(f"ML params:\n{location}")


def parallel_search():
    processes = []
    for chain_id in range(nchains):  # [-39:]:
        process = multiprocessing.Process(
            target=metropolis_hastings_rates, args=(chain_id,)
        )
        processes.append(process)
        process.start()

    # this waits for each wid process to finish before moving onto the next leafid
    for process in processes:
        process.join()


def count_transitions():
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
    counts = ((pd.Series(alltransitions)).value_counts()).to_frame().reset_index()
    counts.columns = ["transition", "count"]
    counts["log_count"] = np.log(counts["count"])
    print(counts)
    sns.catplot(data=counts, x="transition", y="count", kind="bar")
    plt.show()


def count_shapes():
    allshapes = []
    for walk in dfs:
        if not walk.empty:
            initial_state = walk["first_cat"][0]
            steps = walk["shape"].tolist()
            for i, curr in enumerate(steps):
                if i == 0:
                    allshapes.append(initial_state)
                    allshapes.append(curr)
                else:
                    allshapes.append(curr)
    counts = ((pd.Series(allshapes)).value_counts()).to_frame().reset_index()
    counts.columns = ["shape", "count"]
    print(counts)
    sns.catplot(data=counts, x="shape", y="count", kind="bar")
    plt.show()


if __name__ == "__main__":
    parallel_search()

    # init = [ # MLE estimates
    #     0.0059100,
    #     0.0027669,
    #     0.0004471,
    #     0.0555743,
    #     0.0263329,
    #     0.0003601,
    #     0.0388942,
    #     0.0233140,
    #     0.0122715,
    #     0.0208720,
    #     0.0017733,
    #     0.0345245,
    # ]
    # Q = np.array(
    #     [
    #         [-(init[0] + init[1] + init[2]), init[0], init[1], init[2]],
    #         [init[3], -(init[3] + init[4] + init[5]), init[4], init[5]],
    #         [init[6], init[7], -(init[6] + init[7] + init[8]), init[8]],
    #         [init[9], init[10], init[11], -(init[9] + init[10] + init[11])],
    #     ]
    # )
    # L = likelihood_rates_t1(Q)
    # print(L)
    # count_shapes()
