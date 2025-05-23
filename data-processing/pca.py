import os
from textwrap import wrap
import pandas as pd
import numpy as np
from pdict import pdict, leafids
from scipy import stats, spatial
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Polygon, Patch
import matplotlib.colors as mcolors
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
BOOTSTRAP = True  # do "bootstrapping" on the final PCA distribution
SAMP_SIZE = 4000  # no. leaves to sample at random from each shape
SAMP_SEED = 1  # seed for sub sampling raw walks to equal size per shape
BOOSTRAP_SIZE = 1000  # no. leaves drawn per shape per bootstrap iteration
N_BOOTSTRAP = 1000  # no. bootstrap iterations
PLOT_BIN_COUNT_DIST = True  # show histogram of bin counts for 2D hist of PCA
P_TYPE = 7  # 0-scatter, 1-hist2d, 2-kdeplot matplotlib, 3-kdeplot seaborn, 4-hexbin
# 5-2dhist for mean bin counts, 6-2d hist mean bin counts with bin counts hist
# 7-2d hist mean bin counts single ax
ALPHA = 0.005  # alpha in scatter plot, 0.05 for sub-sample, 0.005 for full
N_BINS = 20  # number of bins for hist2d/hexbin/kdeplot
MINCT = 0  # method for minimum count for hexbin - 0 for 1, 1 for 5% of vmax
WD = "leaves_full_13-03-25_MUT2_CLEAN"  # walk directory
DATA = 1  # 0-len 80, 1-len 320
N_COMP = 4  # number of PCA components to keep
XVAR = "PC1"  # which component to plot on x-axis
YVAR = "PC2"  # which component to plot on y-axis
G_PARAMS = {name: val for name, val in globals().items()if name.isupper()}


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


def do_pca():
    """
    Perform PCA on the walk parameter data and initial leaves parameter data.
    Returns the PCA results for walks and inits, and explained variance ratio.
    """
    from dataprocessing import first_cats
    idxs = ["leafid", "walkid", "step"]  # index columns for data sorting

    # Get parameter and shape data for all walks
    psdata = pd.DataFrame()
    if DATA == 0:  # for wl-80
        pdata = pd.read_csv("MUT2.2_trajectories_param.csv")
        pdata = pdata[pdata.iloc[:, 3].str.contains(
            "passed")].reset_index(drop=True)  # remove failed
        pdata = pdata.rename(columns={"0": "walkid", "1": "step"})
        pdata = pdata.iloc[:, :-6]  # remove shape info
        sdata = pd.read_csv("MUT2.2_trajectories_shape.csv")
        sdata = sdata[idxs + ["shape"]]  # filter sdata to relevant cols
        psdata = pd.merge(pdata, sdata, on=idxs, how="inner")  # merge shapes
        psdata = psdata.iloc[:, 5:]  # remove meta data

    elif DATA == 1:  # for wl-320
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
            lambda x: x.sample(SAMP_SIZE, random_state=SAMP_SEED)
        ).reset_index(drop=True)

    print("psdata shape counts:\n",
          psdata.groupby("shape").size().reset_index(name='count'))

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

    # combine init and random walk leaves into 1 df for pca, separate later
    psdata.columns = range(psdata.shape[1])  # rename cols
    psinit.columns = range(psinit.shape[1])  # rename cols
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
    pca_params = PCA(n_components=N_COMP)  # PCA
    princip_params = pca_params.fit_transform(scaled_data)
    evr = pca_params.explained_variance_ratio_
    pdf = pd.DataFrame(data=princip_params, columns=[
                       f"PC{i+1}" for i in range(0, N_COMP)])
    pdf["shape"] = sdata  # reattach shape data
    pdf_init = pdf.iloc[: len(pinit)]  # extract PCA of inits
    pdf_walk = pdf.iloc[len(pinit):].reset_index(drop=True)  # drop inits
    assert pdf_init["shape"].equals(first_cats["first_cat"]), (
        "pdf_init and first_cats do not match"
    )

    hulls = []  # Generate convex hulls
    for shape in ORDER:
        pca_sub = pdf_walk[pdf_walk["shape"] == shape]
        pca_sub = pca_sub[[XVAR, YVAR]]
        hull = spatial.ConvexHull(pca_sub)
        hulls.append(hull)

    return pdf_walk, pdf_init, evr, hulls


def get_pca_lims(pdf_walk):
    """Get global min and max for PCA space to ensure consistent binning"""
    x_min = pdf_walk[XVAR].min()
    x_max = pdf_walk[XVAR].max()
    y_min = pdf_walk[YVAR].min()
    y_max = pdf_walk[YVAR].max()

    lims = [[x_min, x_max], [y_min, y_max]]
    return lims


def boostrap_sample(pdf_walk, lims):
    """Sub-sample the walk data many times to build a distribution of the PCA
    space. Counts no. bins occupied by each shape in a 2D histogram of the PCA
    space. Returns the bootstrap samples and the 2D hist bin counts."""

    bin_counts = []
    b_pdf_walks = []  # for storing bootstrap samples
    htmps_by_shape = {shape: []for shape in ORDER}  # Store hists by shape
    edgs = None  # for storing bin edges
    for i in range(N_BOOTSTRAP):
        print(f"Bootstrap {i}", end="\r")
        b_pdf_walk = pdf_walk.groupby("shape").sample(  # with replacement
            n=BOOSTRAP_SIZE, replace=True  # equally over shape
        ).reset_index(drop=True)  # total size = 4*BOOSTRAP_SIZE
        b_pdf_walk["bstrap"] = i  # add bootstrap column
        b_pdf_walks.append(b_pdf_walk)

        for shape in ORDER:
            pca_sub = b_pdf_walk[b_pdf_walk["shape"] == shape]
            h, xe, ye = np.histogram2d(
                pca_sub[XVAR], pca_sub[YVAR], bins=N_BINS,
                range=lims)  # global min and max for consistent binning
            bin_count = np.count_nonzero(h)
            bin_counts.append(
                {"bstrap": i, "shape": shape, "bin_count": bin_count})
            htmps_by_shape[shape].append(h)
            if edgs is None:
                edgs = [xe, ye]

    mean_htmps = {s: np.mean(htmps_by_shape[s], axis=0) for s in ORDER}

    b_pdf_walk = pd.concat(b_pdf_walks, ignore_index=True)
    bin_counts = pd.DataFrame(bin_counts)

    return b_pdf_walk, bin_counts, mean_htmps, edgs


def get_bin_count_dist(bin_counts):
    """Plot distribution of bin counts for 2D histogram of PCA space."""
    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for i, s in enumerate(ORDER):
        sub = bin_counts[bin_counts["shape"] == s]
        ax.hist(sub["bin_count"], alpha=0.3,
                color=f"C{i}", label=order_full[i])
    ax.grid(alpha=0.3)
    ax.set_xlabel("PCA bins occupied")
    ax.set_ylabel("Frequency")
    ax.set_title("\n".join(wrap(
        "No. PCA 2D histogram bins occupied by each shape over" +
        f" {N_BOOTSTRAP} bootstraps", width=40))
    )
    fig.legend(title="Shape", loc="outside right")
    if P_TYPE != 6:
        plt.savefig(f"pca_bin_count_dist_ptype{P_TYPE}_{WD}.pdf", dpi=1200,
                    metadata={"Keywords": str(G_PARAMS)})
        plt.show()


def get_vmax_vmin(pdf_walk, lims):
    """Get global min and max for 2d histogram, hexbin and kde plots. Make
    sure you implement the same binning for all plots."""
    glob = None
    if P_TYPE == 2:  # initialise grid for matplotlib kdeplot
        nbins = 100
        density = []
        for s in pdf_walk["shape"].unique():
            sub = pdf_walk[pdf_walk["shape"] == s]
            x = sub[XVAR]
            y = sub[YVAR]
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
            hb = ax_tmp.hexbin(x=sub[XVAR], y=sub[YVAR], gridsize=N_BINS)
            all_counts.append(hb.get_array())
        plt.close(fig_tmp)  # Close the temporary figure
        glob = np.concatenate(all_counts)
    if P_TYPE in [1, 5, 6]:  # calculate vmin and vmax for 2d histogram
        all_counts = []
        fig_tmp, ax_tmp = plt.subplots()
        for s in pdf_walk['shape'].unique():
            sub = pdf_walk[pdf_walk['shape'] == s]
            h, _, _ = np.histogram2d(
                sub[XVAR], sub[YVAR], bins=N_BINS, range=lims)
            all_counts.append(h)
        plt.close(fig_tmp)
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
    lims = get_pca_lims(pdf_walk)  # get global min and max for PCA space

    if BOOTSTRAP:
        b_pdf_walk, bin_counts, mean_htmps, _ = boostrap_sample(
            pdf_walk, lims)
        if PLOT_BIN_COUNT_DIST:
            get_bin_count_dist(bin_counts)

        pdf_walk = pdf_walk.groupby("shape").sample(
            n=BOOSTRAP_SIZE, replace=True
        ).reset_index(drop=True)  # reduce to BOOTSTRAP_SIZE

    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]
    if P_TYPE in [1, 2, 4, 5]:
        vmin, vmax = get_vmax_vmin(pdf_walk, lims)
    icon_imgs = load_leaf_imgs()

    fig, axs = plt.subplots(2, 2, figsize=(6, 5), sharex=True, sharey=True,
                            layout="constrained")

    for i, ax in enumerate(axs.flat):
        shape = ORDER[i]
        pld = pdf_walk[pdf_walk["shape"] == shape].reset_index(drop=True)

        if P_TYPE == 0:  # plot walks
            p = ax.scatter(x=pld[XVAR], y=pld[YVAR],
                           s=10, alpha=ALPHA, ec=None)
        elif P_TYPE == 1:
            h, _, _, p = ax.hist2d(x=pld[XVAR], y=pld[YVAR], range=lims,
                                   bins=N_BINS, cmin=1, vmin=vmin, vmax=vmax)
            print(np.nanmax(h))

        elif P_TYPE == 2:
            x = pld[XVAR]
            y = pld[YVAR]
            xi, yi = np.mgrid[x.min():x.max():N_BINS*1j,
                              y.min():y.max():N_BINS*1j]
            k = stats.gaussian_kde([x, y])
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            zi[np.isclose(zi, 0, atol=1e-10)] = 0
            p = ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5, vmin=vmin,
                            vmax=vmax, antialiased=True)
        elif P_TYPE == 3:
            # https://seaborn.pydata.org/generated/seaborn.kdeplot.html
            p = sns.kdeplot(x=pld[XVAR], y=pld[YVAR], ax=ax, levels=5,
                            fill=True)
            ax.set(xlabel=None, ylabel=None)
        elif P_TYPE == 4:  # hexbin
            mincnt = 1 if MINCT == 0 else 0.05*vmax
            print(f"mincnt: {mincnt}")
            p = ax.hexbin(x=pld[XVAR], y=pld[YVAR], gridsize=N_BINS,
                          cmap="viridis", vmin=vmin, vmax=vmax, lw=0.2,
                          mincnt=mincnt)

            bin_count = np.count_nonzero(p.get_array())
            print(f"Bin count {shape}: {bin_count}")
            ax.set(xlabel=None, ylabel=None)
            ax.text(1, 0, fr"Non-zero bins $={bin_count}$", ha="right",
                    va="bottom", transform=ax.transAxes)
        elif P_TYPE == 5:
            c_thresh = 1e-6  # threshold value for bins to be counted
            cmap = plt.get_cmap("viridis")
            cmap.set_bad(color="white")
            masked_heatmap = np.ma.masked_where(  # set white where count<1
                mean_htmps[shape] < c_thresh, mean_htmps[shape])
            p = ax.imshow(masked_heatmap.T, origin="lower",
                          cmap=cmap, extent=lims, vmin=vmin, vmax=vmax)
            bin_count = np.sum(~(mean_htmps[shape] < c_thresh))
            print(f"Bin count {shape}: {bin_count}")
            ax.text(1, 0, fr"Non-zero bins $={bin_count}$", ha="right",
                    va="bottom", transform=ax.transAxes)

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
                x=pdf_init[XVAR],
                y=pdf_init[YVAR],
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
                    unlobed_data[XVAR][simplex],
                    unlobed_data[YVAR][simplex],
                    color="grey",
                )
            for simplex in hull.simplices:
                print(simplex)
                ax.plot(
                    pld[XVAR][simplex],
                    pld[YVAR][simplex],
                    color="red",
                )
            ax.set_title(f"{order_full[i]} h-vol:{round(hull.volume, 2)}")

    if P_TYPE in [1, 4, 5]:
        fig.colorbar(p, ax=axs, shrink=0.5, label="Frequency")
    fig.supxlabel(fr"{XVAR} (${(evr[0] * 100):.2f}\%$)")
    fig.supylabel(fr"{YVAR} (${(evr[1] * 100):.2f}\%$)")

    # plt.tight_layout()
    plt.savefig(f"pca_param_ptype{P_TYPE}_{WD}.pdf", dpi=1200,
                metadata={"Keywords": str(G_PARAMS)})
    plt.show()


def get_vmin_vmax_htmp(mean_htmps):
    """Get global min and max for mean 2d histograms."""
    glob = None
    for s in mean_htmps:
        if glob is None:
            glob = mean_htmps[s]
        else:
            htmp = mean_htmps[s]
            glob = np.concatenate((glob, htmp))
    vmin = glob.min()
    vmax = glob.max()
    print(f"vmin: {vmin}, vmax: {vmax}")

    return vmin, vmax


def bincount_paramspace():
    """Plot histogram of bincounts and PCA paramspace on same figure."""

    c_thresh = 0.5  # threshold value for bins to be counted
    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]
    icon_imgs = load_leaf_imgs()
    pdf_walk, pdf_init, evr, hulls = do_pca()
    lims = get_pca_lims(pdf_walk)  # get global min and max for PCA space
    b_pdf_walk, bin_counts, mean_htmps, _ = boostrap_sample(pdf_walk, lims)
    vmin, vmax = get_vmin_vmax_htmp(mean_htmps)

    fig = plt.figure(figsize=(6, 8), layout="constrained")
    subfigs = fig.subfigures(2, 1, height_ratios=[2, 1])
    axs1 = subfigs[0].subplots(2, 2, sharex=True, sharey=True)
    axs2 = subfigs[1].subplots(1, 1)

    for i, ax in enumerate(axs1.flat):
        shape = ORDER[i]
        cmap = plt.get_cmap("viridis")
        cmap.set_bad(color="white")
        masked_heatmap = np.ma.masked_where(  # set white where count<1
            mean_htmps[shape] < c_thresh, mean_htmps[shape])
        p = ax.imshow(masked_heatmap.T, origin="lower",
                      cmap=cmap, extent=np.concatenate(lims).tolist(),
                      vmin=vmin, vmax=vmax)
        bin_count = np.sum(mean_htmps[shape] >= c_thresh)
        print(f"Bin count {shape}: {bin_count}")
        print(f"Max grid val {shape}: {masked_heatmap.max()}")
        print(mean_htmps[shape].sum())
        ax.text(1, 0, fr"#bins$\geq{c_thresh}$ $={bin_count}$", ha="right",
                va="bottom", transform=ax.transAxes)
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
        ax.set_title(fr"{order_full[i]}")
    subfigs[0].colorbar(p, ax=axs1, shrink=0.5, label=" Mean frequency")
    subfigs[0].supxlabel(fr"{XVAR} (${(evr[0] * 100):.2f}\%$)")
    subfigs[0].supylabel(fr"{YVAR} (${(evr[1] * 100):.2f}\%$)")
    h_rng = (bin_counts["bin_count"].min(), bin_counts["bin_count"].max())
    for i, s in enumerate(ORDER):
        sub = bin_counts[bin_counts["shape"] == s]
        axs2.hist(sub["bin_count"], alpha=0.3, range=h_rng,
                  color=f"C{i}", label=order_full[i], bins=h_rng[1]-h_rng[0])
    axs2.grid(alpha=0.3)
    subfigs[1].supxlabel("No. PCA bins occupied")
    subfigs[1].supylabel("Frequency")
    axs2.legend(loc="upper right")
    plt.savefig(f"pca_param_ptype{P_TYPE}_{WD}.pdf", dpi=1200,
                metadata={"Keywords": str(G_PARAMS)})
    plt.show()


def paramspace_combined():
    """Visualise PCA parameter space for different leaf shapes overlayed on
    top of unlobed."""

    c_thresh = 0.5  # threshold value for bins to be counted
    order_full = ["Unlobed", "Lobed", "Dissected", "Compound"]
    icon_imgs = load_leaf_imgs()
    pdf_walk, pdf_init, evr, hulls = do_pca()
    lims = get_pca_lims(pdf_walk)  # get global min and max for PCA space
    b_pdf_walk, bin_counts, mean_htmps, edgs = boostrap_sample(pdf_walk, lims)
    # vmin, vmax = get_vmin_vmax_htmp(mean_htmps)

    # fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    # handles = []
    # for i, shape in enumerate(ORDER):
    #     cmap = mcolors.ListedColormap([plt.get_cmap("tab10")(i)])
    #     # masked_heatmap = np.ma.masked_where(  # set white where count<1
    #     #     mean_htmps[shape] < c_thresh, mean_htmps[shape])
    #     msk = np.where(mean_htmps[shape] < c_thresh, 0, 1)  # binary mask
    #     htmp = mean_htmps[shape]
    #     # p = ax.imshow(masked_heatmap.T, origin="lower", alpha=0.3,
    #     #               cmap=cmap, extent=np.concatenate(lims).tolist())
    #     x, y = np.meshgrid(edgs[0][:-1], edgs[1][:-1])
    #     p1 = ax.contourf(x, y, msk.T, levels=[c_thresh, 999],
    #                      colors=f"C{i}", alpha=0.3,
    #                      extent=np.concatenate(lims).tolist())
    #     p2 = ax.contour(x, y, msk.T, levels=[c_thresh, 999],
    #                     colors=f"C{i}", alpha=1,
    #                     extent=np.concatenate(lims).tolist())
    #     handles.append(f"C{i}")
    # ax.legend(handles, labels=order_full, loc="upper right")
    # plt.show()
    fig, axs = plt.subplots(2, 2, figsize=(6, 4), layout="constrained",
                            sharex=True, sharey=True)
    msku = None
    for i, ax in enumerate(axs.flat):
        shape = ORDER[i]
        # cmap = mcolors.ListedColormap([
        #     (0, 0, 0, 0),           # RGBA for transparent (for 0)
        #     f"C{i}"  # Color for 1
        # ])
        cmap = [plt.get_cmap("Blues"), plt.get_cmap("Oranges"),
                plt.get_cmap("Greens"), plt.get_cmap("Reds")]
        msk = np.where(mean_htmps[shape] < c_thresh, 0, 1)  # binary mask
        # msk1 = np.ma.masked_where(  # set white where count<1
        #     mean_htmps[shape] < c_thresh, mean_htmps[shape])
        x, y = np.meshgrid(edgs[0][:-1], edgs[1][:-1])
        p1 = ax.contourf(x, y, msk.T,  levels=[c_thresh, 999],
                         colors=f"C{i}", alpha=0.3,
                         extent=np.concatenate(lims).tolist())
        # p1 = ax.contourf(x, y, msk1.T,  # levels=[c_thresh, 999],
        #                  cmap=cmap[i], alpha=1,
        #                  extent=np.concatenate(lims).tolist())

        p2 = ax.contour(x, y, msk.T, levels=[c_thresh, 999],
                        colors=f"C{i}", alpha=1, label=order_full[i],
                        extent=np.concatenate(lims).tolist())
        # ax.imshow(msk.T, origin="lower", alpha=0.5,
        #           cmap=cmap, extent=np.concatenate(lims).tolist())
        bin_count = np.sum(mean_htmps[shape] >= c_thresh)
        print(f"Bin count {shape}: {bin_count}")
        print(mean_htmps[shape].sum())
        # ax.text(1, 0, fr"#bins$\geq{c_thresh}$ $={bin_count}$", ha="right",
        #         va="bottom", transform=ax.transAxes)
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
        # ax.set_title(fr"{order_full[i]}")
        if i == 0:
            msku = msk
            # msku1 = msk1
        else:
            # cmapu = mcolors.ListedColormap([
            #     (0, 0, 0, 0),           # RGBA for transparent (for 0)
            #     "C0"  # Color for 1
            # ])
            p3 = ax.contourf(x, y, msku.T, levels=[c_thresh, 999],
                             colors="C0", alpha=0.3, zorder=0,
                             extent=np.concatenate(lims).tolist())
            p4 = ax.contour(x, y, msku.T, levels=[c_thresh, 999],
                            colors="C0", alpha=1, zorder=0,
                            extent=np.concatenate(lims).tolist())
            # ax.imshow(msku.T, origin="lower", alpha=0.2,
            #           cmap=cmapu, extent=np.concatenate(lims).tolist())
        ax.grid(alpha=0.3)

    fig.supxlabel(fr"{XVAR} (${(evr[0] * 100):.2f}\%$)")
    fig.supylabel(fr"{YVAR} (${(evr[1] * 100):.2f}\%$)")

    handles = [Patch(color=f"C{i}", label=s) for i, s in enumerate(order_full)]
    fig.legend(handles, order_full, loc="outside right")
    plt.show()


if __name__ == "__main__":
    for name, val in G_PARAMS.items():
        print(f"{name} {val}")
    if P_TYPE in [0, 1, 2, 3, 4, 5]:
        paramspace()
    elif P_TYPE == 6:
        bincount_paramspace()
    elif P_TYPE == 7:
        paramspace_combined()
    # get_p_and_s_data()
