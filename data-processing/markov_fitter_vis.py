import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

order = ["u", "l", "d", "c"]


def get_data():
    dfs = []
    for file in os.listdir("markov_fitter_reports"):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join("markov_fitter_reports", file))
            df.insert(0, "chain_id", file[6])
            dfs.append(df)
    concat = pd.concat(dfs, ignore_index=True)
    return concat


def plot_LL():
    data_full = get_data()
    sns.relplot(data=data_full, y="location_LL", x="step", hue="chain_id", kind="line")
    plt.show()


def get_data_post_burnin_long():
    data_full = get_data()
    data_post_burnin = data_full[data_full["step"] > 100]
    data_post_burnin_nondiag = data_post_burnin.drop(
        ["q00", "q11", "q22", "q33"], axis=1
    )
    data_post_burnin_nondiag_long = pd.melt(
        data_post_burnin_nondiag,
        id_vars=["chain_id", "step", "location_LL"],
        var_name="transition",
        value_name="rate",
    )

    # Create two variables for initial and final shape
    mapping = {"0": "u", "1": "l", "2": "d", "3": "c"}
    data_post_burnin_nondiag_long["initial_shape"] = (
        data_post_burnin_nondiag_long["transition"]
        .apply(lambda str: str[1])
        .replace(mapping)
    )
    data_post_burnin_nondiag_long["final_shape"] = (
        data_post_burnin_nondiag_long["transition"]
        .apply(lambda str: str[2])
        .replace(mapping)
    )

    return data_post_burnin_nondiag_long


def get_posterior_by_chain():
    data_post_burnin_nondiag_long = get_data_post_burnin_long()
    print(data_post_burnin_nondiag_long)

    sns.catplot(
        data=data_post_burnin_nondiag_long,
        y="rate",
        x="initial_shape",
        hue="final_shape",
        col="chain_id",
        kind="box",
        col_wrap=4,
        palette="colorblind",
        hue_order=order,
    )
    plt.show()


def get_posterior_overall():
    data_post_burnin_nondiag_long = get_data_post_burnin_long()
    print(data_post_burnin_nondiag_long)

    sns.catplot(
        data=data_post_burnin_nondiag_long,
        y="rate",
        x="initial_shape",
        hue="final_shape",
        kind="box",
        palette="colorblind",
        hue_order=order,
        col_order=order,
    )
    plt.show()


get_posterior_by_chain()
get_posterior_overall()