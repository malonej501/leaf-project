import pandas as pd
import numpy as np
import sympy as sp
import scipy
import functools
import operator
import copy
from dataprocessing import concatenator, first_cats
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns


nsteps = 10000
nchains = 8
lb = 0  # lower bound of inferred rate parameters
ub = 100  # upper bound of inferred rate parameters
step_size = 10  # random numbers are picked uniformly between + and - this number which make the jump matrix
# for lb0 ub100, step_size10 seems to work well

p_labels = [
    "p00",
    "p01",
    "p02",
    "p03",
    "p10",
    "p11",
    "p12",
    "p13",
    "p20",
    "p21",
    "p22",
    "p23",
    "p30",
    "p31",
    "p32",
    "p33",
]
p = sp.symbols(p_labels)

transition_map = {
    "uu": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: p00
    * (1 - p01)
    * (1 - p02)
    * (1 - p03),
    "ul": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p00
    )
    * p01
    * (1 - p02)
    * (1 - p03),
    "ud": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p00
    )
    * (1 - p01)
    * p02
    * (1 - p03),
    "uc": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p00
    )
    * (1 - p01)
    * (1 - p02)
    * p03,
    "lu": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: p10
    * (1 - p11)
    * (1 - p12)
    * (1 - p13),
    "ll": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p10
    )
    * p11
    * (1 - p12)
    * (1 - p13),
    "ld": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p10
    )
    * (1 - p11)
    * p12
    * (1 - p13),
    "lc": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p10
    )
    * (1 - p11)
    * (1 - p12)
    * p13,
    "du": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: p20
    * (1 - p21)
    * (1 - p22)
    * (1 - p23),
    "dl": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p20
    )
    * p21
    * (1 - p22)
    * (1 - p23),
    "dd": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p20
    )
    * (1 - p21)
    * p22
    * (1 - p23),
    "dc": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p20
    )
    * (1 - p21)
    * (1 - p22)
    * p23,
    "cu": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: p30
    * (1 - p31)
    * (1 - p32)
    * (1 - p33),
    "cl": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p30
    )
    * p31
    * (1 - p32)
    * (1 - p33),
    "cd": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p30
    )
    * (1 - p31)
    * p32
    * (1 - p33),
    "cc": lambda p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33: (
        1 - p30
    )
    * (1 - p31)
    * (1 - p32)
    * p33,
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
                    L_walk *= Pt[transition_map_rates[transition]]
        # rate_map = {}
        # L_data += np.log(L_walk)
        L_data += np.log(L_walk)

    return L_data


def likelihood_rates_frominitial(Q):
    L_data = 1
    for walk in dfs:
        if not walk.empty:
            L_walk = 1
            t = 0
            initial_state = walk["first_cat"][0]
            steps = walk["shape"].tolist()
            for i, curr in enumerate(steps):
                t += 1
                transition = initial_state + curr
                Pt = scipy.linalg.expm(Q * t)
                L_walk *= Pt[transition_map_rates[transition]]
        # rate_map = {}
        # L_data += np.log(L_walk)
        L_data += np.log(L_walk)

    return L_data


def likelihood_rates_t1(Q):
    t = 1
    L_data = 1
    Pt = scipy.linalg.expm(Q * t)
    for walk in dfs:
        if not walk.empty:
            L_walk = 1
            initial_state = walk["first_cat"][0]
            steps = walk["shape"].tolist()
            for i, curr in enumerate(steps):
                if i == 0:
                    prev = initial_state
                else:
                    prev = steps[i - 1]
                    transition = prev + curr
                    L_walk *= Pt[transition_map_rates[transition]]
        L_data += np.log(L_walk)

    return L_data


def likelihood(
    p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33
):
    L_data = 1
    P = sp.Matrix(4, 4, p)
    for walk in dfs:
        if not walk.empty:
            L_walk = 1
            # create step sequence list with the initial shape on the front
            steps = walk["shape"].tolist()
            initial_state = walk["first_cat"][0]
            # build the likelihood function
            for i, curr in enumerate(steps):
                if i == 0:
                    prev = initial_state
                else:
                    prev = steps[i - 1]
                transition = prev + curr
                L_walk *= transition_map[transition](
                    p00,
                    p01,
                    p02,
                    p03,
                    p10,
                    p11,
                    p12,
                    p13,
                    p20,
                    p21,
                    p22,
                    p23,
                    p30,
                    p31,
                    p32,
                    p33,
                )
        # print(L_walk)
        # print(np.log(L_walk))
        # exit()
        L_data += np.log(L_walk)  # log of products = sum of the logs

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
            jump = np.random.uniform(-step_size, step_size, 12)
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
        # acceptance_ratio = proposal_l / location_l  # for maximising likelihood
        acceptance_ratio = location_l / proposal_l  # for maximising log_likelihood
        print(step, location_l, proposal_l, acceptance_ratio)
        acceptance_threshold = np.random.uniform(0.99, 1)
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


def metropolis_hastings():
    try:
        location_l = 0
        while np.isinf(location_l) or location_l == 0:
            init = np.random.uniform(0, 1, 16)
            location = init
            location_l = likelihood(*location)
            print(location_l)
        proposal_l = 0
        step = 0
        for step in range(nsteps):
            proposal_l = 0
            proposal = []
            while (
                np.isinf(proposal_l)
                or proposal_l == 0
                or any(i < 0 or i > 1 for i in proposal)
            ):
                jump = np.random.uniform(-0.05, 0.05, 16)
                proposal = location + jump
                proposal_l = likelihood(*proposal)
            # acceptance_ratio = proposal_l / location_l  # for maximising likelihood
            acceptance_ratio = location_l / proposal_l  # for maximising log_likelihood
            print(location_l, proposal_l, acceptance_ratio)
            acceptance_threshold = np.random.uniform(0.5, 1.5)
            if acceptance_ratio > acceptance_threshold:
                location = copy.deepcopy(proposal)
                location_l = copy.deepcopy(proposal_l)
                print(f"Step: {step}, Likelihood: {location_l}")
            else:
                continue
    finally:
        print(f"ML params: {location}")
        P = location.reshape(4, 4)
        Q = np.linalg.slogdet(P)
        print(f"Q estimate: {Q}")


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


# parallel_search()
count_shapes()

# ML params: [0.99826104 0.01398225 0.01326471 0.01950028 0.77214213 0.47574471
#  0.42267044 0.47365062 0.83230558 0.25730578 0.0937699  0.45734609
#  0.94638902 0.59232413 0.06093926 0.07854079]
# ML params: [0.99423626 0.00141407 0.00539718 0.01960178 0.75330801 0.87560217
#  0.71831958 0.37055998 0.56737443 0.62423334 0.56100007 0.38894312
#  0.84030609 0.26382123 0.60430812 0.00841867]
