import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import kruskal
import copy
import sympy as sp
from sympy import *

wd = "mcmc/unif_dated_21-1-24/"
filename = "mcmc_unif_dated_21-1-24.csv"

T = 0.1  # set value of T to enumerate probabilities

rates_full = pd.read_csv(wd + filename)
# print(rates_full)
prob_tab = pd.DataFrame(
    {
        "p00": [],
        "p01": [],
        "p02": [],
        "p03": [],
        "p10": [],
        "p11": [],
        "p12": [],
        "p13": [],
        "p20": [],
        "p21": [],
        "p22": [],
        "p23": [],
        "p30": [],
        "p31": [],
        "p32": [],
        "p33": [],
    }
)

#### Insert new columns for stationary rates

rates_full_wstat = copy.deepcopy(rates_full)
rates_full_wstat.insert(0, "q00", 0 - rates_full.iloc[:, 0:3].sum(axis=1))
rates_full_wstat.insert(5, "q11", 0 - rates_full.iloc[:, 3:6].sum(axis=1))
rates_full_wstat.insert(10, "q22", 0 - rates_full.iloc[:, 6:9].sum(axis=1))
rates_full_wstat.insert(15, "q33", 0 - rates_full.iloc[:, 9:12].sum(axis=1))
# print(rates_full_wstat)

#### Calculate Probabilities give t=0.1


def matrixfromrow(dataframe, i):
    row = dataframe.iloc[i].values
    rate_matrix = row.reshape(4, 4)
    return rate_matrix


def rowfrommatrix(matrix, prob_tab):
    row_arr = matrix.reshape(1, 16)
    row_df = pd.DataFrame(row_arr, columns=list(prob_tab.columns))
    return row_df


def getprobs(Q):

    ## Manual Way
    # t = sp.symbols("t")
    # # D - eigenvalues, C - eigenvectors
    # D, C = np.linalg.eig(Q)
    # Cinv = np.linalg.inv(C)
    # Ddiag = np.diagflat(np.array([sp.exp(val * t) for val in D]))
    # P = np.matmul(np.matmul(C, Ddiag), Cinv)
    # evaluate = np.vectorize(lambda expr: expr.subs(t, T))
    # Peval = evaluate(P)

    ## Quick Way
    Peval = scipy.linalg.expm(Q * T)
    # exit()
    return Peval


def rates_probs_mean(prob_tab):
    # Q1 = np.array(
    #     [
    #         [-7.33971033333333, 5.66013596969697, 0.265320606060606, 1.41425375757576],
    #         [99.9958611414141, -162.866899474747, 0.259923232323232, 62.6111151010101],
    #         [
    #             3.19065408080808,
    #             0.991705666666667,
    #             -4.24180003030303,
    #             0.0594402828282828,
    #         ],
    #         [19.7934216262626, 11.5333476161616, 5.61182216161616, -36.9385914040404],
    #     ]
    # )
    # print(Q1)
    # print(getprobs(Q1))
    # exit()

    for i in range(0, len(rates_full_wstat)):
        rates = matrixfromrow(rates_full_wstat, i)
        probs = getprobs(rates)
        row = rowfrommatrix(probs, prob_tab)
        prob_tab = pd.concat([prob_tab, row], ignore_index=True)

    print(prob_tab)
    prob_tab.to_csv(wd + f"probs_{filename}", index=false)


def rates_mean_probs(prob_tab):
    print(rates_full_wstat)
    print(rates_full_wstat.mean())
    means = rates_full_wstat.mean()
    means_reshape = np.reshape(means, (4, 4))
    probs = getprobs(means_reshape)
    print(probs)
    prob_tab = rowfrommatrix(probs, prob_tab)
    prob_tab.to_csv(wd + f"probs_{filename}", index=false)


rates_probs_mean(prob_tab)
