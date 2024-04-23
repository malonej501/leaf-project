import pymc as pm
from markov_fitter import get_data
from dataprocessing import first_cats, concatenator
import scipy
import pandas as pd
import numpy as np
import arviz as az
import pytensor.tensor as tt

# import emcee


print(f"running on PyMCv{pm.__version__}")


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


# def get_transition_count_matrix():
#     dfs = get_data()
#     alltransitions = []
#     count_matrix = np.zeros((4, 4))
#     for walk in dfs:
#         if not walk.empty:
#             initial_state = walk["first_cat"][0]
#             steps = walk["shape"].tolist()
#             for i, curr in enumerate(steps):
#                 if i == 0:
#                     prev = initial_state
#                 else:
#                     prev = steps[i - 1]
#                     transition = prev + curr
#                     alltransitions.append(transition)
#     count_df = ((pd.Series(alltransitions)).value_counts()).to_frame().reset_index()
#     count_df.columns = ["transition", "count"]
#     # organise the counts into a 4x4 matrix with same structure as P or Q
#     for key, (row, col) in transition_map_rates.items():
#         count_matrix[row, col] = count_df.loc[
#             count_df["transition"] == key, "count"
#         ].values[0]
#     return count_matrix, count_df


# def get_transition_count_matrix_list():
#     dfs = get_data()
#     matrix_list = []
#     for walk in dfs:
#         transitions = []
#         count_matrix = np.zeros((4, 4))
#         if not walk.empty:
#             initial_state = walk["first_cat"][0]
#             steps = walk["shape"].tolist()
#             for i, curr in enumerate(steps):
#                 if i == 0:
#                     prev = initial_state
#                 else:
#                     prev = steps[i - 1]
#                     transition = prev + curr
#                     transitions.append(transition)
#             count_df = (
#                 ((pd.Series(transitions)).value_counts()).to_frame().reset_index()
#             )
#             count_df.columns = ["transition", "count"]
#             # organise the counts into a 4x4 matrix with same structure as P or Q
#             for key, (row, col) in transition_map_rates.items():
#                 if count_df["transition"].str.contains(key).any():
#                     count_matrix[row, col] = count_df.loc[
#                         count_df["transition"] == key, "count"
#                     ].values[0]

#             matrix_list.append(count_matrix)

#     return matrix_list


# count_matrix, count_df = get_transition_count_matrix()
# # matrix_list = get_transition_count_matrix_list()


# data = pd.read_csv("MUT2.2_alt.csv")
# print(data)
# obs = data["state"].to_numpy()


def get_transitions(dfs):
    transitions = []
    for walk in dfs:
        if not walk.empty:
            initial_state = walk["first_cat"][0]
            steps = walk["shape"].tolist()
            for i, curr in enumerate(steps):
                if i == 0:
                    prev = initial_state
                else:
                    prev = steps[i - 1]
                    transitions.append(prev + curr)
    transitions = np.array(transitions)
    return transitions


def get_proportion_over_t(dfs):
    for walk in dfs:
        walk["step"] = walk.index.values

    concat = pd.concat(dfs, ignore_index=True)
    concat = pd.merge(concat, first_cats[["leafid", "first_cat"]], on="leafid")
    mapping = {"u": 0, "l": 1, "d": 2, "c": 3}
    concat["dummy"] = concat["shape"].map(mapping)
    print(concat)

    # no. each shape type for each step of each first_cat
    grouped_by_first_cat = (
        concat.groupby(["first_cat", "step", "shape"])
        .size()
        .reset_index(name="total_shape_firstcat")
    )
    # total no. leaves per step for every first_cat
    grouped_by_first_cat_total = (
        grouped_by_first_cat.groupby(["first_cat", "step"])
        .agg(total_firstcat=("total_shape_firstcat", "sum"))
        .reset_index()
    )

    grouped_by_first_cat = grouped_by_first_cat.merge(
        grouped_by_first_cat_total, on=["first_cat", "step"]
    )
    grouped_by_first_cat["proportion"] = (
        grouped_by_first_cat["total_shape_firstcat"]
        / grouped_by_first_cat["total_firstcat"]
    )
    return grouped_by_first_cat


def pymc_fit():
    dfs = concatenator()
    proportion_over_t = get_proportion_over_t(dfs)
    print(proportion_over_t)
    exit()

    markov_model = pm.Model()

    # help(pm.distributions)
    # help(pm.distributions.timeseries.PredefinedRandomWalk)
    # pm.distributions.continuous.Uniform

    # with markov_model:
    #     t = 1
    #     # obs_counts_tensor = pytensor.shared(count_matrix)
    #     Q = pm.Uniform("Q", lower=0, upper=100, shape=(4, 4))  # define prior Q matrix
    #     expected_counts = scipy.linalg.expm(Q * t) * count_df["count"].sum()
    #     sigma = pm.HalfNormal("sigma", sigma=1)
    #     Q_likelihood = pm.Normal(
    #         "Q_likelihood",
    #         shape=(4, 4),
    #         mu=expected_counts,
    #         sigma=sigma,
    #         observed=count_matrix,
    #     )
    #     idata = pm.sample()

    with markov_model:

        q01 = pm.Uniform("q01", lower=0, upper=1)
        q02 = pm.Uniform("q02", lower=0, upper=1)
        q03 = pm.Uniform("q03", lower=0, upper=1)
        q00 = pm.Deterministic("q00", -q01 - q02 - q03)
        q10 = pm.Uniform("q10", lower=0, upper=1)
        q12 = pm.Uniform("q12", lower=0, upper=1)
        q13 = pm.Uniform("q13", lower=0, upper=1)
        q11 = pm.Deterministic("q11", -q10 - q12 - q13)
        q20 = pm.Uniform("q20", lower=0, upper=1)
        q21 = pm.Uniform("q21", lower=0, upper=1)
        q23 = pm.Uniform("q23", lower=0, upper=1)
        q22 = pm.Deterministic("q22", -q20 - q21 - q23)
        q30 = pm.Uniform("q30", lower=0, upper=1)
        q31 = pm.Uniform("q31", lower=0, upper=1)
        q32 = pm.Uniform("q32", lower=0, upper=1)
        q33 = pm.Deterministic("q33", -q30 - q31 - q32)

        Q = tt.stacklists(
            [
                [q00, q01, q02, q03],
                [q10, q11, q12, q13],
                [q20, q21, q22, q23],
                [q30, q31, q32, q33],
            ]
        )
        sigma = pm.HalfNormal("sigma", sigma=1)

        # def logL():
        t = 1  # between each state
        Pt = tt.slinalg.expm(Q * t)
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
                        # likelihood = pm.Normal("likelihood", mu=tt.log(Pt[transition_map_rates[transition]]), sigma=sigma, observed=True)
                        # likelihood += tt.log(Pt[transition_map_rates[transition]])
                        likelihood += pm.Normal(
                            "L",
                            mu=tt.log(Pt[transition_map_rates[transition]]),
                            sigma=sigma,
                            observed=tt.log(Pt[transition_map_rates[transition]]),
                        )
        # return likelihood

        # likelihood = logL()
        # likelihood = pm.Normal("likelihood", mu=logL(), sigma=sigma, observed=True)
        # likelihood_obs = pm.Normal(
        #     "likelihood_obs", mu=likelihood, sigma=sigma, observed=likelihood
        # )

        idata = pm.sample(10000)

    summary = pm.summary(idata)
    print(summary)
    exit()
    pm.plot_trace(idata)

    graph = pm.model_to_graphviz(markov_model)
    graph.render("markov_model", format="pdf", cleanup=True)
    # idata.posterior["Q"]
    # az.plot_trace(idata, combined=True)

    # help(pytensor.tensor.slinalg.Expm)
    # expm = pytensor.tensor.slinalg.Expm

    # class ContinuousMarkov(pm.Categorical):
    #     def __init__(self, Q, init_P):
    #         super(pm.Discrete, self).__init__(*args, **kwargs)
    #         self.Q = Q
    #         self.init_P = init_P

    #     def logp(self, X):
    #         self.Q = Q
    #         t = 1
    #         state = X[:, 1]
    #         lik = pm.Categorical.dist(self.init_P).logp(state[0])
    #         P = scipy.linalg.expm(self.Q * t)
    #         for i in X:
    #             lik += pm.Categorical.dist(P[state[i]]).logp(state[i + 1])
    #         return lik


def emcee_fit():
    # dfs = get_data

    def logL(Q, dfs):
        t = 1  # between each state
        Pt = scipy.linalg.expm(Q * t)
        print(Pt)
        L_data = 0
        for walk in dfs:
            if not walk.empty:
                L_walk = 0
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


pymc_fit()
