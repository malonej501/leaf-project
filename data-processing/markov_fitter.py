import pandas as pd
import numpy as np
import sympy as sp
import functools
import operator
from dataprocessing import concatenator, first_cats

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


def log_likelihood(
    p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33
):
    dfs = get_data()
    P = sp.Matrix(4, 4, p)
    print(P)
    for walk in dfs:
        if not walk.empty:
            likelihood = 1
            # create step sequence list with the initial shape on the front
            steps = walk["shape"].tolist()
            initial_state = walk["first_cat"][0]
            print(walk)
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
            print(np.log(likelihood))
            exit()


log_likelihood(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
