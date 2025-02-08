import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from datetime import date
from arrow_violin import import_phylo_and_sim_rates, import_phylo_ML_rates, import_sim_ML_rates, normalise_rates, rates_map2

norm_method="meanmean"
ML_data="ML6_genus_mean_rates_all" # ML data for the phylogeny
legend = False

plot_order = [
        "MUT1_simulation",
        "MUT2_simulation",
        # "jan_phylo_nat_class_uniform0-0.1_1",
        # "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_2",
        # "geeta_phylo_geeta_class_uniform0-100_4",
        "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
        "geeta_phylo_geeta_class_uniform0-100_genus_1",
    ]

def plot_phylo_and_sim_rates_with_leaf_icons():
    
    ML_phylo_rates_long = import_phylo_ML_rates(plot_order, calc_diff=False)
    ML_sim_rates_long = import_sim_ML_rates(calc_diff=False)
    phylo_sim_long = import_phylo_and_sim_rates(plot_order, calc_diff=False)
    phylo_sim_long, ML_phylo_rates_long, ML_sim_rates_long = normalise_rates(
        phylo_sim_long, ML_phylo_rates_long, ML_sim_rates_long
    )
    ML_sim_rates_long = ML_sim_rates_long.rename(columns={"Dataset":"dataset"})
    ML_all = pd.concat([ML_phylo_rates_long, ML_sim_rates_long], ignore_index=True) # combine ML rates

    #### plotting ####
    plt.rcParams["font.family"] = "CMU Serif"
    if legend:
        fig, axes = plt.subplots(
            nrows=4,
            ncols=4,
            figsize=(10, 8),  # sharey=True
        )  # , layout="constrained")
    else:
        fig, axes = plt.subplots(
            nrows=5,
            ncols=5,
            figsize=(9, 9),  # sharey=True
        )  # , layout="constrained")
    counter = -1
    legend_labels = []
    for i in range(1, 5):
        for j in range(0, 4):
            ax = axes[i, j]
            if i - 1 == j:
                ax.axis("off")
            if i - 1 != j:
                counter += 1
                transition = list(rates_map2.values())[counter]
                plot_data = phylo_sim_long[phylo_sim_long["transition"] == transition]
                ml_plot_data = ML_all[
                    ML_all["transition"] == transition
                ]
                rates = []
                ML_rates = []
                for k, dataset in enumerate(plot_order):
                    # dset.append(dataset)
                    rates.append(
                        plot_data["rate_norm"][
                            plot_data["Dataset"] == dataset
                        ].squeeze()
                    )

                    # get ml_plot_data in correct order
                    x = ml_plot_data[
                        ml_plot_data["dataset"].apply(lambda x: x in dataset)
                    ].reset_index(drop=True)
                    if not x.empty:
                        ML_rates.append(x.loc[0, "rate_norm"])
                    elif x.empty:
                        ML_rates.append(np.nan)
                    if dataset not in legend_labels:
                        legend_labels.append(dataset)
                ax.axvline(2.5, linestyle="--", color="grey", alpha=0.5)
                bp = ax.violinplot(rates, showextrema=False, showmeans=True)

                # for median in bp["medians"]:
                #     median.set_visible(False)
                # ax.set_title(transition)
                if norm_method == "zscore":
                    ax.set_ylim(-2.7, 8)  # for z-score normalisation
                elif norm_method == "zscore+2.7":
                    ax.set_ylim(0, 10.7)  # for z-score norm + 2.7
                elif norm_method == "zscore_global":
                    ax.set_ylim(-2, 5)
                elif norm_method == "meanmean":
                    ax.set_ylim(0, 10)  # for mean-mean normalisation
                elif norm_method == "minmax":
                    ax.set_ylim(-0.5, 3)  # for min-max normalisation

                # plot ML values
                if ML_data:
                    pos = list(range(1, len(plot_order) + 1))
                    ax.scatter(
                        pos, ML_rates, color="black", zorder=5, s=8, facecolors="white"
                    )  # , marker="D")

            xticklabs = ["S1", "S2"]
            xticklabs.extend([f"P{i}" for i in range(1, len(plot_order) - 1)])
            if i == 4:
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    ["MUT1", "MUT2", "Janssens", "Zuntini", "Geeta"],
                    fontsize=9,
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            if (i, j) == (3, 3):
                ax.set_xticks(
                    list(range(1, len(plot_order) + 1)),
                    ["MUT1", "MUT2", "Janssens", "Zuntini", "Geeta"],
                    fontsize=9,
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            if j != 0 and (i, j) != (1, 1):
                ax.set_yticklabels([])
            if i != 4 and (i, j) != (3, 3):
                ax.set_xticklabels([])

    ### plot leaf images ####
    icon_filenames = [
        "leaf_p7a_0_0.png",
        "leaf_p8ae_0_0.png",
        "leaf_pd1_0_0.png",
        "leaf_pc1_alt_0_0.png",
    ]
    icons = [os.path.join("uldc_model_icons", path) for path in icon_filenames]
    icon_imgs = [Image.open(path) for path in icons]
    img_width, img_height = icon_imgs[1].size
    scale_factor = 0.5
    shape_cats = ["Unlobed", "Lobed", "Dissected", "Compound"]

    for j in range(0, 4):
        ax = axes[0, j]
        ax.axis("off")
        ax.imshow(icon_imgs[j])
        ax.text(img_width / 2, img_height, shape_cats[j], ha="center", va="top")
        ax.set_xlim(img_width / scale_factor, (-img_width / 2) / scale_factor)
        ax.set_ylim(img_height, -(img_height / scale_factor))
    for i in range(1, 5):
        ax = axes[i, 4]
        ax.axis("off")
        ax.imshow(icon_imgs[i - 1])
        ax.text(img_width / 2, img_height, shape_cats[i - 1], ha="center", va="top")
        ax.set_xlim(0, (img_width / scale_factor) + ((img_width / 2) / scale_factor))
        ax.set_ylim(img_height / scale_factor, (-img_height / 2) / scale_factor)
    axes[0, 4].axis("off")

    xlab_pos = 0.43
    ylab_pos = 0.45
    fig.supxlabel("Dataset", x=xlab_pos, ha="center")
    fig.supylabel("Normalised rate", y=ylab_pos, ha="center", va="center")
    fig.text(xlab_pos, 0.9, "Final shape", ha="center", va="center", fontsize=12)
    fig.text(
        0.9,
        ylab_pos,
        "Initial shape",
        ha="center",
        va="center",
        rotation=270,
        fontsize=12,
    )
    plt.tight_layout()
    if legend:
        if (
            norm_method == "zscore"
            or norm_method == "zscore+2.7"
            or norm_method == "zscore_global"
        ):
            plt.subplots_adjust(
                hspace=0.2, wspace=0.2, right=0.745, left=0.064
            )  # for z-score-norm
        else:
            plt.subplots_adjust(
                hspace=0.2, wspace=0.2, right=0.745, left=0.044
            )  # for min-max-norm or mean-mean-norm
    else:
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
    
    plt.savefig(f"violin_{str(date.today())}.pdf", format="pdf", dpi=1200)
    plt.show()

if __name__ == "__main__":
    plot_phylo_and_sim_rates_with_leaf_icons()