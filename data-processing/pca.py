import os
import pandas as pd
import numpy as np
from pdict import pdict, leafids
from scipy import stats, spatial
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from PIL import Image


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
SUB_SAMPLE = True  # whether to sub-sample the data
SAMP_SIZE = 1000  # no. leaves to sample at random from each shape
P_TYPE = 4  # 0-scatter, 1-hist2d, 2-kdeplot matplotlib, 3-kdeplot seaborn, 4-hexbin
ALPHA = 0.005  # alpha in scatter plot, 0.05 for sub-sample, 0.005 for full
N_BINS = 30  # number of bins for hist2d/hexbin/kdeplot
WD = "leaves_full_13-03-25_MUT2_CLEAN"  # walk directory
DATA = 0  # 0-len 80, 1-len 320


def get_p_and_s_data():
    """Get parameter and shape data of all leaves in a directory."""
    pdfs, sdfs = [], []

    for l_dir in os.listdir(WD):
        l_dir_path = os.path.join(WD, l_dir)
        if not os.path.isdir(l_dir_path):
            continue
        print(l_dir_path)
        for w_dir in os.listdir(l_dir_path):
            w_dir_path = os.path.join(l_dir_path, w_dir)
            for file in os.listdir(w_dir_path):
                if file.endswith(".csv") and \
                        "report" in file and "shape" not in file:
                    pdf = pd.read_csv(os.path.join(w_dir_path, file))
                    # no header if original MUT2
                    pdf.insert(0, "leafid", l_dir)
                    pdfs.append(pdf)
                elif file == "shape_report.csv":
                    sdf = pd.read_csv(os.path.join(w_dir_path, file))
                    sdf.insert(0, "leafid", l_dir)  # add leafid
                    w_num = int(w_dir.replace("walk", ""))
                    sdf.insert(1, "walkid", w_num)  # insert walkid to df
                    steps = sdf["leaf"].apply(lambda s: int(s.split("_")[-2]))
                    sdf.insert(2, "step", steps)  # insert step number to df
                    sdfs.append(sdf)

    pdata = pd.concat(pdfs, ignore_index=True)
    pdata.to_csv(f"{WD}/params.csv", index=False)
    sdata = pd.concat(sdfs, ignore_index=True)
    sdata.to_csv(f"{WD}/shapes.csv", index=False)

    return pdata, sdata


def sort_walk_shape_data(pdata, sdata):
    """Sort walk shape data to match parameter data according to leafid, 
    walkid and step."""
    idxs = ["leafid", "walkid", "step"]  # index columns for data sorting
    sdata_sort = sdata.set_index(idxs).reindex(pdata.set_index(idxs).index)
    sdata = sdata_sort.reset_index()
    match = (pdata[idxs] == sdata[idxs]).all().all()  # check idx columns match
    assert match, f"pdata and sdata {idxs} do not match"

    return sdata


def do_pca():
    """
    Perform PCA on the walk parameter data and initial leaves parameter data.
    Returns the PCA results for walks and inits, and explained variance ratio.
    """
    from dataprocessing import first_cats
    idxs = ["leafid", "walkid", "step"]  # index columns for data sorting
    psdata = pd.DataFrame()  # parameter data
    if DATA == 0:
        pdata = pd.read_csv("MUT2.2_trajectories_param.csv")
        pdata = pdata[pdata.iloc[:, 3].str.contains(
            "passed")].reset_index(drop=True)  # remove failed
        pdata = pdata.rename(columns={"0": "walkid", "1": "step"})
        pdata = pdata.iloc[:, :-6]  # remove shape info
        sdata = pd.read_csv("MUT2.2_trajectories_shape.csv")
        sdata = sdata[idxs + ["shape"]]  # filter sdata to relevant cols
        psdata = pd.merge(pdata, sdata, on=idxs, how="inner")  # merge shapes
        psdata = psdata.iloc[:, 5:]  # remove meta data

    elif DATA == 1:
        if os.path.isfile(f"{WD}/params.csv") and \
                os.path.isfile(f"{WD}/shapes.csv"):
            pdata = pd.read_csv(f"{WD}/params.csv")
            pdata = pdata[pdata["status"] ==  # remove failed
                          "leaf_check_passed"].reset_index(drop=True)
            pdata = pdata.rename(columns={"walk_id": "walkid"})
            sdata = pd.read_csv(f"{WD}/shapes.csv")
            sdata = sdata[idxs + ["shape"]]  # filter to relevant cols
            psdata = pd.merge(pdata, sdata, on=idxs, how="inner")

        else:
            print("Parameter and shape data not found. Generating...")
            pdata, sdata = get_p_and_s_data()
            print("Parameter and shape data generated.")
            pdata = pdata[pdata["status"] ==
                          "leaf_check_passed"].reset_index(drop=True)
            pdata = pdata.rename(columns={"walk_id": "walkid"})
            psdata = pd.merge(pdata, sdata, on=idxs, how="inner")

        # remove meta data and shape info
        psdata = psdata.drop(["leafid", "walkid", "step", "attempt", "status",
                              "target", "prop_weightdifference", "middle",
                              "leafwidth", "prop_overlappingmargin",
                              "prop_veinarea", "veinswidth",
                              "prop_veinsoutsidelamina"], axis=1)

    if SUB_SAMPLE:
        psdata = psdata.groupby("shape").apply(
            lambda x: x.sample(SAMP_SIZE, random_state=1)
        ).reset_index(drop=True)

    # Get params and shape data for initial leaves
    pinit = pd.DataFrame(pdict.values()).transpose()  # format pdict params
    # sort first_cats to match pdict order, to ensure correct labelling
    first_cats = first_cats.set_index("leafid").reindex(leafids).reset_index()
    assert list(first_cats["leafid"]) == leafids, (
        "first_cats and pdict leafid orders do not match"
    )
    pinit["leafid"] = leafids  # add leafid to pinit
    psinit = pd.merge(
        pinit, first_cats, on="leafid", how="inner")
    psinit = psinit.drop("leafid", axis=1)  # drop leafid
    psinit = psinit.rename(columns={"first_cat": "shape"})  # rename first_cat
    psinit.columns = range(psinit.shape[1])  # rename cols

    # pinit.to_csv("pinit.csv", index=False)

    psdata.columns = range(psdata.shape[1])  # rename cols

    # combine init and random walk leaves into 1 df for pca, separate later
    pdata = pd.concat([psinit, psdata], ignore_index=True)
    sdata = pdata.iloc[:, -1:]  # separate shape data
    pdata = pdata.iloc[:, :-1]
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
    pdf["shape"] = sdata  # reattach shape data
    pdf_init = pdf.iloc[: len(pinit)]  # extract PCA of inits
    pdf_walk = pdf.iloc[len(pinit):].reset_index(drop=True)  # drop inits
    assert pdf_init["shape"].equals(first_cats["first_cat"]), (
        "pdf_init and first_cats do not match"
    )

    hulls = []  # Generate convex hulls
    for shape in ORDER:
        pca_sub = pdf_walk[pdf_walk["shape"] == shape]
        pca_sub = pca_sub[["pc1", "pc2"]]
        hull = spatial.ConvexHull(pca_sub)
        hulls.append(hull)

    return pdf_walk, pdf_init, evr, hulls


def get_vmax_vmin(pdf_walk):
    """Get global min and max for 2d histogram, hexbin and kde plots."""
    glob = None
    if P_TYPE == 2:  # initialise grid for matplotlib kdeplot
        nbins = 100
        density = []
        for s in pdf_walk["shape"].unique():
            sub = pdf_walk[pdf_walk["shape"] == s]
            x = sub["pc1"]
            y = sub["pc2"]
            k = stats.gaussian_kde([x, y])
            xi, yi = np.mgrid[x.min():x.max():nbins*1j,
                              y.min():y.max():nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            density.append(zi)
        glob = np.concatenate(density)
    if P_TYPE == 4:  # calculate vim and vmax for hexbin
        all_counts = []
        fig_tmp, ax_tmp = plt.subplots()
        for s in pdf_walk['shape'].unique():
            sub = pdf_walk[pdf_walk['shape'] == s]
            hb = ax_tmp.hexbin(x=sub['pc1'], y=sub['pc2'], gridsize=N_BINS)
            all_counts.append(hb.get_array())
        plt.close(fig_tmp)  # Close the temporary figure
        glob = np.concatenate(all_counts)

    vmin = glob.min()
    vmax = glob.max()
    print(f"vmin: {vmin}, vmax: {vmax}")

    return vmin, vmax


def load_leaf_imgs():
    """Get leaf images for plotting"""
    icon_filenames = [
        "leaf_p7a_0_0.png",
        "leaf_p8ae_0_0.png",
        "leaf_pd1_0_0.png",
        "leaf_pc1_alt_0_0.png",
    ]
    icons = [os.path.join("uldc_model_icons", path) for path in icon_filenames]
    icon_imgs = [plt.imread(path) for path in icons]

    return icon_imgs


def paramspace():
    """Visualise walk leaves in PCA of parameter space."""

    pdf_walk, pdf_init, evr, hulls = do_pca()

    # if SUB_SAMPLE:  # sub-sample the data
    #     pdf_walk = pdf_walk.groupby("shape").apply(
    #         lambda x: x.sample(SAMP_SIZE, random_state=1)
    #     ).reset_index(drop=True)

    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]
    vmin, vmax = get_vmax_vmin(pdf_walk)
    icon_imgs = load_leaf_imgs()

    fig, axs = plt.subplots(2, 2, figsize=(6, 5), sharex=True, sharey=True,
                            layout="constrained")
    for i, ax in enumerate(axs.flat):
        shape = ORDER[i]
        pld = pdf_walk[  # get walk data
            pdf_walk["shape"] == shape
        ].reset_index(drop=True)

        if P_TYPE == 0:  # plot walks
            p = ax.scatter(x=pld["pc1"], y=pld["pc2"],
                           s=10, alpha=ALPHA, ec=None)
        elif P_TYPE == 1:
            p = ax.hist2d(x=pld["pc1"], y=pld["pc2"],
                          bins=N_BINS, cmap="viridis", cmin=1)
        elif P_TYPE == 2:
            x = pld["pc1"]
            y = pld["pc2"]
            xi, yi = np.mgrid[x.min():x.max():N_BINS*1j,
                              y.min():y.max():N_BINS*1j]
            k = stats.gaussian_kde([x, y])
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            zi[np.isclose(zi, 0, atol=1e-10)] = 0
            p = ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5, vmin=vmin,
                            vmax=vmax, antialiased=True)
        elif P_TYPE == 3:
            # https://seaborn.pydata.org/generated/seaborn.kdeplot.html
            p = sns.kdeplot(x=pld["pc1"], y=pld["pc2"], ax=ax, levels=5,
                            fill=True)
            ax.set(xlabel=None, ylabel=None)
        elif P_TYPE == 4:  # hexbin
            p = ax.hexbin(x=pld["pc1"], y=pld["pc2"], gridsize=N_BINS,
                          cmap="viridis", vmin=vmin, vmax=vmax, mincnt=1,
                          lw=0.2)
            ax.set(xlabel=None, ylabel=None)

        imbg_box = OffsetImage(icon_imgs[i], zoom=0.08, alpha=0.5)
        ab = AnnotationBbox(
            imbg_box,
            (1, 1),
            xycoords="axes fraction",
            box_alignment=(1, 1),  # upper right corner alignment
            frameon=False,
            pad=0.2,
        )
        ax.add_artist(ab)
        ax.grid(alpha=0.3)
        ax.set_title(fr"{order_full[i]}, $N={len(pld)}$")
        # if SUB_SAMPLE:
        #     ax.set_title(f"{order_full[i]}")

        if SHOW_INITS:  # plot initial points
            ax.scatter(
                x=pdf_init["pc1"],
                y=pdf_init["pc2"],
                c=pdf_init["shape"].map(
                    {"u": "C0", "l": "C1", "d": "C2", "c": "C3", }
                ),
                edgecolor="white",
                linewidth=0.8,
                alpha=1,
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

    if P_TYPE == 4:
        fig.colorbar(p, ax=axs, shrink=0.5, label="Frequency")
    fig.supxlabel(fr"PC1 (${(evr[0] * 100):.2f}\%$)")
    fig.supylabel(fr"PC2 (${(evr[1] * 100):.2f}\%$)")

    # plt.tight_layout()
    plt.savefig(f"pca_param_ptype{P_TYPE}_{WD}.pdf", dpi=1200)
    plt.show()


if __name__ == "__main__":
    paramspace()
    # get_p_and_s_data()
