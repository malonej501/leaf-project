import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

order = ["u", "l", "d", "c"]
burnin = 2000  # proportion of steps to discard from the beginning of each chain when calculating posteriors
sns.set_palette("colorblind")


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
    data_post_burnin = data_full[data_full["step"] > burnin]
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
    data_post_burnin_nondiag_long

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
        kind="bar",
        palette="colorblind",
        hue_order=order,
        col_order=order,
    )
    plt.show()


def posterior_overall_pertransition_hist():
    data_post_burnin_nondiag_long = get_data_post_burnin_long()

    sns.displot(
        data=data_post_burnin_nondiag_long,
        x="rate",
        kind="hist",
        col="transition",
        col_wrap=4,
    )
    plt.show()


def rate_convergence():
    data_post_burnin_nondiag_long = get_data_post_burnin_long()
    chain_0 = data_post_burnin_nondiag_long[
        data_post_burnin_nondiag_long["chain_id"] == "0"
    ].reset_index(drop=True)
    print(chain_0)

    sns.relplot(
        data=chain_0,
        y="rate",
        x="step",
        col="transition",
        hue="transition",
        palette="colorblind",
        kind="line",
        col_wrap=4,
    )
    plt.show()


def rate_against_frequency():
    # each transition rate against the frequency of the initial state in the data
    data_post_burnin_nondiag_long = get_data_post_burnin_long()
    freq = pd.read_csv("MUT2.2_shape_counts.csv")
    print(data_post_burnin_nondiag_long)
    print(freq)
    long_merged = pd.merge(
        data_post_burnin_nondiag_long,
        freq,
        left_on="initial_shape",
        right_on="shape",
        how="inner",
    )
    print(long_merged)
    sns.catplot(
        data=long_merged,
        y="rate",
        x="count",
        hue="transition",
        col="chain_id",
        kind="box",
        col_wrap=4,
        palette="colorblind",
    )
    plt.show()


def relative_rates():
    data = get_data_post_burnin_long()
    data.groupby(["transition", "chain_id", "step"]).apply()

    # data["total_rate"] =

    print(data)


if __name__ == "__main__":
    # posterior_overall_pertransition_hist()
    # plot_LL()
    # get_posterior_by_chain()
    get_posterior_overall()
    # rate_convergence()
    # rate_against_frequency()

    # relative_rates()
