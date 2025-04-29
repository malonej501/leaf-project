import pandas as pd
import numpy as np
from pdict import pdict
from dataprocessing import first_cats
from scipy import stats, linalg, spatial
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


LEAFIDS = ["p6af", "p6i", "p7a", "p7g", "p8ae", "p8i", "p9b", "p10c7", "p12b",
           "p12c7", "p12de", "p12f", "p1_414", "p4_510", "p6_163_alt",
           "p8_1235", "p7_277", "p6_1155", "p5_909", "p7_437", "p1_122_alt",
           "p2_149_alt", "p1_35", "p5_249", "p2_346_alt", "p6_81", "p7_43",
           "p1_82", "p2_78_alt", "p3_60", "p5_122", "p7_92", "p0_121",
           "p2_195", "p4_121", "p9_129", "pc1_alt", "pc3_alt", "pc4",
           "pc5_alt", "pd1", "pl1", "pl2", "pl3", "pl4", "pu1", "pu2", "pu3"]
SHOW_INITS = False  # whether to show initial leaves in PCA plot
SHOW_HULLS = False  # whether to show convex hulls in PCA plot
ORDER = ["u", "l", "d", "c"]
SUB_SAMPLE = False  # whether to sub-sample the data
SAMP_SIZE = 1000  # no. leaves to sample at random from each shape
P_TYPE = 3  # 0-scatter, 1-hist2d, 2-kdeplot matplotlib, 3-kdeplot seaborn
ALPHA = 0.005  # alpha in scatter plot, 0.05 for sub-sample, 0.005 for full
# def get_pdata():
#     dfs = []

#     for leafdirectory in os.listdir(wd):
#         print(f"Current = {leafdirectory}")
#         leafdirectory_path = os.path.join(wd, leafdirectory)
#         for walkdirectory in os.listdir(leafdirectory_path):
#             walkdirectory_path = os.path.join(
#                 leafdirectory_path, walkdirectory)
#             for file in os.listdir(walkdirectory_path):
#                 if file.endswith(".csv") and "report" in file and "shape" not in file:
#                     df = pd.read_csv(
#                         os.path.join(walkdirectory_path, file), header=None
#                     )
#                     df.insert(0, "leafid", leafdirectory)
#                     dfs.append(df)

#     # result = pd.concat(dfs, ignore_index=True)
#     # result.to_csv("/home/m/malone/leaf_storage/random_walks/result.csv", index=False)

#     return dfs


def do_pca():
    """
    Perform PCA on the walk parameter data and initial leaves parameter data. 
    Returns the PCA results for walks and inits, and explained variance ratio.
    """

    # need to update to 320
    pdata = pd.read_csv("MUT2.2_trajectories_param.csv")
    sdata = pd.read_csv("MUT2.2_trajectories_shape.csv")
    pinit = pd.DataFrame(pdict.values()).transpose()  # format pdit params
    # pinit.to_csv("pinit.csv", index=False)

    pdata = pdata[pdata.iloc[:, 3].str.contains("passed")]  # remove failed
    pdata = pdata.iloc[:, 5:-6]  # remove meta data and shape info
    pdata.columns = range(pdata.shape[1])  # rename cols

    # combine init and random walk leaves into 1 df for pca, separate later
    pdata = pd.concat([pinit, pdata], ignore_index=True)
    pdata = pdata.replace(
        {r".*true*.": 1, r".*false*.": 0, r".*nan*.": np.nan}, regex=True
    ).infer_objects(copy=False)
    filt = ["M_PI", "#define"]  # drop columns with M_PI or #define
    drop = [col for col in pdata.columns if any(
        f in str(value) for value in pdata[col] for f in filt)]
    pdata = pdata.drop(columns=drop)
    pdata = pdata.dropna(axis=1, how="any")  # drop columns with any NaN

    scaled_data = StandardScaler().fit_transform(pdata)  # scale data

    pca_params = PCA(n_components=2)  # PCA
    princip_params = pca_params.fit_transform(scaled_data)
    evr = pca_params.explained_variance_ratio_
    pdf = pd.DataFrame(data=princip_params, columns=["pc1", "pc2"])
    pdf_init = pdf.iloc[: len(pinit)]  # extract PCA of inits
    pdf_init.insert(0, "shape", first_cats["first_cat"])  # label inits
    pdf_walk = pdf.iloc[len(pinit):].reset_index(drop=True)  # drop inits
    assert len(pdf_walk) == len(sdata), "pdf_walk and sdata not same length"
    pdf_walk.insert(0, "shape", sdata["shape"])  # attach shape

    hulls = []  # Generate convex hulls
    for shape in ORDER:
        pca_sub = pdf_walk[pdf_walk["shape"] == shape]
        pca_sub = pca_sub[["pc1", "pc2"]]
        hull = spatial.ConvexHull(pca_sub)
        hulls.append(hull)

    return pdf_walk, pdf_init, evr, hulls


def paramspace():
    """Visualise walk leaves in PCA of parameter space."""

    pdf_walk, pdf_init, evr, hulls = do_pca()

    if SUB_SAMPLE:  # sub-sample the data
        pdf_walk = pdf_walk.groupby("shape").apply(
            lambda x: x.sample(SAMP_SIZE, random_state=1)
        ).reset_index(drop=True)

    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]

    fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
    if P_TYPE == 2:  # initialise grid for matplotlib kdeplot
        nbins = 100
        x = pdf_walk["pc1"]
        y = pdf_walk["pc2"]
        k = stats.gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():nbins*1j,
                          y.min():y.max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        vmin, vmax = zi.min(), zi.max()  # global density range

    for i, ax in enumerate(axs.flat):
        shape = ORDER[i]
        pld = pdf_walk[  # get walk data
            pdf_walk["shape"] == shape
        ].reset_index(drop=True)

        if P_TYPE == 0:  # plot walks
            ax.scatter(x=pld["pc1"], y=pld["pc2"], s=10, alpha=ALPHA, ec=None)
        elif P_TYPE == 1:
            ax.hist2d(x=pld["pc1"], y=pld["pc2"], bins=50, cmap="Blues")
        elif P_TYPE == 2:
            x = pld["pc1"]
            y = pld["pc2"]
            k = stats.gaussian_kde([x, y])
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            ax.contour(xi, yi, zi.reshape(xi.shape), alpha=0.5, vmin=vmin,
                       vmax=vmax)
        elif P_TYPE == 3:
            # https://seaborn.pydata.org/generated/seaborn.kdeplot.html
            sns.kdeplot(x=pld["pc1"], y=pld["pc2"], ax=ax, levels=5,
                        fill=True)
            ax.set(xlabel=None, ylabel=None)

        ax.set_title(fr"{order_full[i]}, $n={len(pld)}$")
        if SUB_SAMPLE:
            ax.set_title(f"{order_full[i]}")

        if SHOW_INITS:  # plot initial points
            ax.scatter(
                x=pdf_init["pc1"],
                y=pdf_init["pc2"],
                c=pdf_init["shape"].map(
                    {"u": "C0", "l": "C1", "d": "C2", "c": "C3", }
                ),
                edgecolor="white",
                linewidth=0.8,
            )

        if SHOW_HULLS:  # plot convex hulls for walk data
            unlobed_data = pdf_walk[  # get unlobed data
                pdf_walk["shape"] == "u"
            ].reset_index(drop=True)
            hull = hulls[i]
            hull_unlobed = hulls[0]
            for simplex in hull_unlobed.simplices:
                ax.plot(
                    unlobed_data["pc1"][simplex],
                    unlobed_data["pc2"][simplex],
                    color="grey",
                )
            for simplex in hull.simplices:
                print(simplex)
                ax.plot(
                    pld["pc1"][simplex],
                    pld["pc2"][simplex],
                    color="red",
                )
            ax.set_title(f"{order_full[i]} h-vol:{round(hull.volume, 2)}")

        fig.supxlabel(fr"PC1 (${(evr[0] * 100):.2f}\%$)")
        fig.supylabel(fr"PC2 (${(evr[1] * 100):.2f}\%$)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    paramspace()
