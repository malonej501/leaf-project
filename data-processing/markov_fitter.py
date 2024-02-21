import pandas as pd
import numpy as np
import sympy as sp
import functools
import operator
import copy
from dataprocessing import concatenator, first_cats

nsteps = 500

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


def likelihood(
    p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33
):
    P = sp.Matrix(4, 4, p)
    for walk in dfs:
        if not walk.empty:
            likelihood = 1
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
                likelihood *= transition_map[transition](
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

    return likelihood


def metropolis_hastings():
    location_l = 0
    while np.isinf(location_l) or location_l == 0:
        init = np.random.uniform(0, 1, 16)
        location = init
        location_l = likelihood(*location)
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
        acceptance_ratio = proposal_l / location_l
        print(location_l, proposal_l, acceptance_ratio)
        acceptance_threshold = np.random.uniform(0, 1)
        if acceptance_ratio > acceptance_threshold:
            location = copy.deepcopy(proposal)
            location_l = copy.deepcopy(proposal_l)
            print(f"Step: {step}, Likelihood: {location_l}")
        else:
            continue
    print(f"ML params: {location}")
    P = location.reshape(4, 4)
    Q = np.linalg.slogdet(P)
    print(f"Q estimate: {Q}")


metropolis_hastings()

# ML params: [0.99826104 0.01398225 0.01326471 0.01950028 0.77214213 0.47574471
#  0.42267044 0.47365062 0.83230558 0.25730578 0.0937699  0.45734609
#  0.94638902 0.59232413 0.06093926 0.07854079]
# ML params: [0.99423626 0.00141407 0.00539718 0.01960178 0.75330801 0.87560217
#  0.71831958 0.37055998 0.56737443 0.62423334 0.56100007 0.38894312
#  0.84030609 0.26382123 0.60430812 0.00841867]
