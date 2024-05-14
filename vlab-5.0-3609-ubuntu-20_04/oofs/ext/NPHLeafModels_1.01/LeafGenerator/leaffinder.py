from pdict import *
import pwriter
import copy
import subprocess
import os
import shutil
import random
import time
import signal
import numpy as np
import cv2
import multiprocessing
import select
import re
from pyvirtualdisplay import Display

#################################### Initialisation ####################################

scheme = "mut3"  # model of mutation
testvals = [10, 1, 0.1]  # only relevant for scheme = "mut1"
startleaf = 0  # the index of the leaf in pdict that the simulation will start at, 0 for the beginning
# testvals = [0,0,0]
# nrounds = 100
ngen_thresh = 120  # threshold no. leaves which, if reached in at least 1 of the walks, will terminate the simulation
ncores = 10  # no. cores and also no. walks - performed in parallel
timeout = 160  # simulation will skip to next iteration if a leaf takes longer than this number of seconds to generate

# Walk constraint parameters
symmetry_thresh = (
    0.05  # 0.1 #the acceptable limit for the proportion of non_overlapping area
)
weightdifference_thresh = 0.06  # the absolute difference between the area of the left and right side of the leaf
primordium_thresh = 1.1  # this is how many times the width of the scale bar the maxwidth of the leaf must be greater than to pass the primordium check
overlappingmargin_thresh = 0.005  # 0.006 #0.08 #0.012 for 12b # the fraction of the lamina area that is permitted to be occupied by margin
veinarea_thresh = (
    0.02  # vein area must be greater than this fraction of the silhouette area
)
veinsoutsidelamina_thresh = 0.01  # 0.037 #0.75 for 12de and 12f 0.2 for everything else. # The proportion of vein area permitted to be non-overlapping with the lamina
veinswidth_thresh = (
    0.05  # veins must be greater than this fraction of the total width of the image
)

# paths
wd = "/home/m/malone/GitHub/leaf-project/vlab-5.0-3609-ubuntu-20_04/oofs/ext/NPHLeafModels_1.01"

nleaves = len(pdict["pspace1"])

# remove old leaf storage directory (if it exists) and create a new one
if os.path.exists(wd + "/leaves"):
    shutil.rmtree(wd + "/leaves")
os.makedirs(wd + "/leaves")

# make a directory for every leaf and within that a directory for every walk
for leafid in leafids:
    os.makedirs(wd + f"/leaves/{leafid}")
    for wid in range(ncores):
        os.makedirs(wd + f"/leaves/{leafid}/walk{wid}")
        os.makedirs(wd + f"/leaves/{leafid}/walk{wid}/rejected")

# remove any .png files in wd at the start of the simulation
for _, directories, _ in os.walk(wd):
    for name in directories:
        if "bin" in name:
            bin_path = wd + "/" + name
            for i in os.listdir(bin_path):
                if i.endswith(".png"):
                    os.remove(bin_path + "/" + i)

# retrieve list of parameters which will be held constant
prangelist = list(prange10_alt.values())


#################################### Main Functions ####################################


def runsim(step, templist, wid, leafid):
    """Runs lpfg with a given set of parameters, kills lpfg if "Error" occurs in the output or after timeout"""
    input_parameters = dict(zip(pdict.keys(), templist))
    pwriter.Input(input_parameters, wd + f"/bin{wid}", wid)  # + f"/parameters{wid}")
    process = subprocess.Popen(
        wd
        + f"/LeafGenerator/Start.sh bin{wid} plant{wid}.l leaf_{leafid}_{wid}_{step}.png",
        shell=True,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setpgrp,
    )
    # pwriter.Input(input_parameters, wd + "/bin0", 0) #+ f"/parameters{wid}")
    # process = subprocess.Popen(wd + f"/LeafGenerator/Start.sh bin0 plant0.l leaf_{leafid}_{wid}_{step}.png", shell=True, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, preexec_fn=os.setpgrp)
    start_time = time.monotonic()
    while True:
        if process.poll() is not None:
            # process has completed
            break
        # This next block will kill lpfg if more than "timeout" seconds elapses between lines output to the console
        # The readline function was where it was where the while loop was getting stuck previously, as there were no new lines being produced
        # We essentially implement a timeout for the readline function
        ready, _, _ = select.select([process.stdout], [], [], timeout)
        if ready:
            line = process.stdout.readline()
            if line:
                print(line)
                if "Error" in line:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    break
        elapsed_time = time.monotonic() - start_time
        # print(f"###### Elapsed time = {round(elapsed_time,2)} seconds")
        if elapsed_time > timeout:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            print("##### TIMED OUT")
            break


def testparams(plist, plist_i, step, wid, leafid, report):
    """Tries different parameter combinations for every element in testvals"""

    if scheme == "mut1":
        # p is the index of the parameter being varied in pdict, step is the iteration number of the random walk
        p = step % len(plist)
        # p = random.randint(0,len(plist))
        # While leaf.png is unable to generate, try running the simulation with each of the different testvals
        for n in range(len(testvals)):
            # save a copy of the latest version of plist_i
            # try different values in the copy
            # if it doesn't work, return to the last copy of plist_i and try the next value
            # if it works, overwrite plist_i with the successful templist and proceed to the next element
            print(
                f"#### Iteration: {leafid}_{wid}_{step} ####\n#### Parameter: {list(pdict.keys())[p]} ####"
            )
            metrics = []
            if prangelist[p] == 0:
                break
            else:
                templist = valchooser(plist, plist_i, n, p)  # testvals[n]
                print(f"#### testval = {templist[p]} ####")
                print(f"#### Current parameter values ####\n{templist}")
                runsim(step, templist, wid, leafid)
                if os.path.isfile(wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png"):
                    print(
                        f"#### leaf_{leafid}_{wid}_{step}.png successfully generated! ####"
                    )
                    (
                        margin,
                        lamina_margintest,
                        lamina_veinstest,
                        veins,
                        silhouette,
                    ) = leafcomponents(leafid, wid, step)
                    check, failed_conditions = leafchecker(
                        margin,
                        lamina_margintest,
                        lamina_veinstest,
                        veins,
                        silhouette,
                        metrics,
                    )
                    if check:
                        # categorise(leaf): # include this condition to restrict the walk to only particular shape categories
                        print(f"#### Leaf check passed!")
                        shutil.move(
                            wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png",
                            wd
                            + f"/leaves/{leafid}/walk{wid}/leaf_{leafid}_{wid}_{step}.png",
                        )
                        # report.append(f"{wid},{step},leaf_check_passed,{list(pdict.keys())[p]},{str(templist)[1:-1],{str(metrics)}}\n")
                        report.append(
                            [
                                [wid, step, "leaf_check_passed", list(pdict.keys())[p]]
                                + templist
                                + metrics
                            ]
                        )
                        plist_i = templist
                        break
                    else:
                        print(f"#### leaf check failed!")
                        shutil.move(
                            wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png",
                            wd
                            + f"/leaves/{leafid}/walk{wid}/rejected/leaf{leafid}_{wid}_{step}.png",
                        )
                        # report.append(f"{wid},{step},leaf_check_failed,{list(pdict.keys())[p]},{str(templist)[1:-1],{str(metrics)}}\n")
                        report.append(
                            [
                                [
                                    wid,
                                    step,
                                    "leaf_check_failed: "
                                    + ". ".join(failed_conditions),
                                    list(pdict.keys())[p],
                                ]
                                + templist
                                + metrics
                            ]
                        )
                        continue
                else:
                    # don't set plist_i to the templist
                    print(f"#### leaf_{leafid}_{wid}_{step}.png failed to generate!")
                    continue

    elif scheme == "mut2":
        p = random.randint(0, len(plist) - 1)  # select random parameter to vary
        print(
            f"#### Iteration: {leafid}_{wid}_{step} ####\n#### Parameter: {list(pdict.keys())[p]} ####"
        )
        metrics = []
        if prangelist[p] == 1:
            templist = valchooser(plist, plist_i, _, p)
            print(f"#### testval = {templist[p]} ####")
            print(f"#### Current parameter values ####\n{templist}")
            runsim(step, templist, wid, leafid)
            if os.path.isfile(wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png"):
                print(
                    f"#### leaf_{leafid}_{wid}_{step}.png successfully generated! ####"
                )
                (
                    margin,
                    lamina_margintest,
                    lamina_veinstest,
                    veins,
                    silhouette,
                ) = leafcomponents(leafid, wid, step)
                check, failed_conditions = leafchecker(
                    margin,
                    lamina_margintest,
                    lamina_veinstest,
                    veins,
                    silhouette,
                    metrics,
                )
                if check:
                    # categorise(leaf): # include this condition to restrict the walk to only particular shape categories
                    print(f"#### Leaf check passed!")
                    shutil.move(
                        wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png",
                        wd
                        + f"/leaves/{leafid}/walk{wid}/leaf_{leafid}_{wid}_{step}.png",
                    )
                    # report.append(f"{wid},{step},leaf_check_passed,{list(pdict.keys())[p]},{str(templist)[1:-1],{str(metrics)}}\n")
                    report.append(
                        [
                            [wid, step, "leaf_check_passed", list(pdict.keys())[p]]
                            + templist
                            + metrics
                        ]
                    )
                    plist_i = templist
                else:
                    print(f"#### leaf check failed!")
                    shutil.move(
                        wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png",
                        wd
                        + f"/leaves/{leafid}/walk{wid}/rejected/leaf{leafid}_{wid}_{step}.png",
                    )
                    # report.append(f"{wid},{step},leaf_check_failed,{list(pdict.keys())[p]},{str(templist)[1:-1],{str(metrics)}}\n")
                    report.append(
                        [
                            [
                                wid,
                                step,
                                "leaf_check_failed: " + ". ".join(failed_conditions),
                                list(pdict.keys())[p],
                            ]
                            + templist
                            + metrics
                        ]
                    )
            else:
                # don't set plist_i to the templist
                print(f"#### leaf_{leafid}_{wid}_{step}.png failed to generate!")

    elif scheme == "mut3":
        p = random.randint(7, 20)
        print(
            f"#### Iteration: {leafid}_{wid}_{step} ####\n#### Parameter: {list(pdict.keys())[p]} ####"
        )
        metrics = []
        if prangelist[p] == 1:
            templist = valchooser(plist, plist_i, _, p)
            print(f"#### testval = {templist[p]} ####")
            print(f"#### Current parameter values ####\n{templist}")
            runsim(step, templist, wid, leafid)
            if os.path.isfile(wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png"):
                print(
                    f"#### leaf_{leafid}_{wid}_{step}.png successfully generated! ####"
                )
                (
                    margin,
                    lamina_margintest,
                    lamina_veinstest,
                    veins,
                    silhouette,
                ) = leafcomponents(leafid, wid, step)
                check, failed_conditions = leafchecker(
                    margin,
                    lamina_margintest,
                    lamina_veinstest,
                    veins,
                    silhouette,
                    metrics,
                )
                if check:
                    # categorise(leaf): # include this condition to restrict the walk to only particular shape categories
                    print(f"#### Leaf check passed!")
                    shutil.move(
                        wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png",
                        wd
                        + f"/leaves/{leafid}/walk{wid}/leaf_{leafid}_{wid}_{step}.png",
                    )
                    # report.append(f"{wid},{step},leaf_check_passed,{list(pdict.keys())[p]},{str(templist)[1:-1],{str(metrics)}}\n")
                    report.append(
                        [
                            [wid, step, "leaf_check_passed", list(pdict.keys())[p]]
                            + templist
                            + metrics
                        ]
                    )
                    plist_i = templist
                else:
                    print(f"#### leaf check failed!")
                    shutil.move(
                        wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png",
                        wd
                        + f"/leaves/{leafid}/walk{wid}/rejected/leaf{leafid}_{wid}_{step}.png",
                    )
                    # report.append(f"{wid},{step},leaf_check_failed,{list(pdict.keys())[p]},{str(templist)[1:-1],{str(metrics)}}\n")
                    report.append(
                        [
                            [
                                wid,
                                step,
                                "leaf_check_failed: " + ". ".join(failed_conditions),
                                list(pdict.keys())[p],
                            ]
                            + templist
                            + metrics
                        ]
                    )
            else:
                # don't set plist_i to the templist
                print(f"#### leaf_{leafid}_{wid}_{step}.png failed to generate!")

    elif scheme == "mut4.1" or scheme == "mut4.2":
        #### For 4.1 vary just morphogen competence
        if scheme == "mut4.1":
            p = np.random.choice(
                [46, 59, 72, 85]
            )  # sample just the competence parameter for each of the morphogen blocks
        #### For 4.2 vary just
        elif scheme == "mut4.2":
            p = np.random.choice([47, 60, 73, 86])
        print(
            f"#### Iteration: {leafid}_{wid}_{step} ####\n#### Parameter: {list(pdict.keys())[p]} ####"
        )
        metrics = []
        if prangelist[p] == 1:
            templist = valchooser(plist, plist_i, _, p)
            print(f"#### testval = {templist[p]} ####")
            print(f"#### Current parameter values ####\n{templist}")
            runsim(step, templist, wid, leafid)
            if os.path.isfile(wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png"):
                print(
                    f"#### leaf_{leafid}_{wid}_{step}.png successfully generated! ####"
                )
                (
                    margin,
                    lamina_margintest,
                    lamina_veinstest,
                    veins,
                    silhouette,
                ) = leafcomponents(leafid, wid, step)
                check, failed_conditions = leafchecker(
                    margin,
                    lamina_margintest,
                    lamina_veinstest,
                    veins,
                    silhouette,
                    metrics,
                )
                if check:
                    # categorise(leaf): # include this condition to restrict the walk to only particular shape categories
                    print(f"#### Leaf check passed!")
                    shutil.move(
                        wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png",
                        wd
                        + f"/leaves/{leafid}/walk{wid}/leaf_{leafid}_{wid}_{step}.png",
                    )
                    # report.append(f"{wid},{step},leaf_check_passed,{list(pdict.keys())[p]},{str(templist)[1:-1],{str(metrics)}}\n")
                    report.append(
                        [
                            [wid, step, "leaf_check_passed", list(pdict.keys())[p]]
                            + templist
                            + metrics
                        ]
                    )
                    plist_i = templist
                else:
                    print(f"#### leaf check failed!")
                    shutil.move(
                        wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png",
                        wd
                        + f"/leaves/{leafid}/walk{wid}/rejected/leaf{leafid}_{wid}_{step}.png",
                    )
                    # report.append(f"{wid},{step},leaf_check_failed,{list(pdict.keys())[p]},{str(templist)[1:-1],{str(metrics)}}\n")
                    report.append(
                        [
                            [
                                wid,
                                step,
                                "leaf_check_failed: " + ". ".join(failed_conditions),
                                list(pdict.keys())[p],
                            ]
                            + templist
                            + metrics
                        ]
                    )
            else:
                # don't set plist_i to the templist
                print(f"#### leaf_{leafid}_{wid}_{step}.png failed to generate!")

    return plist_i


def valchooser(plist, plist_i, n, p):
    """Generates values for the random walk"""

    templist = copy.deepcopy(plist_i)

    if scheme == "mut1":
        if 7 < p < 21:
            if np.isnan(plist[p]):
                templist[p] = np.nan
            elif 7 < p < 14:
                # testval = templist[p] + np.random.choice([1,-1])*testvals[n]
                templist[p] = templist[p] + np.random.choice([1, -1])
                templist[p + (2 * (14 - p))] = templist[
                    p
                ]  # do the same to the opposite side of the array
            elif p == 14:
                templist[p] = 0
            elif 14 < p < 21:
                templist[p] = templist[p] + np.random.choice([1, -1])
                templist[p - (2 * (p - 14))] = templist[
                    p
                ]  # do the same to the opposite side of the array
        elif isinstance(plist[p], str) == True and "M_PI" in plist[p]:
            # testval = ("M_PI*" + str(round(random.uniform(0.05,0.5),5)))
            # Extact number and sample in same way
            currval = re.findall(r"\d+(\.\d+)?", templist[p])
            currval_float = float(currval[0])
            templist[p] = "M_PI*" + str(
                currval_float + round(np.random.choice([1, -1]) * testvals[n], 10)
            )
        elif isinstance(plist[p], str) and ("true" in plist[p] or "false" in plist[p]):
            templist[p] = random.choice(["true", "false"])
        else:
            # need to round to 10 decimal places as python does incorrect arithmetic when subtracting
            templist[p] = round(
                templist[p] + np.random.choice([1, -1]) * testvals[n], 10
            )

    elif scheme == "mut2":
        if 7 < p < 21:
            if np.isnan(plist[p]):
                templist[p] = np.nan
            elif 7 < p < 14:
                # testval = templist[p] + np.random.choice([1,-1])*testvals[n]
                templist[p] = templist[p] + np.random.choice([1, -1])
                templist[p + (2 * (14 - p))] = templist[
                    p
                ]  # do the same to the opposite side of the array
            elif p == 14:
                templist[p] = 0
            elif 14 < p < 21:
                templist[p] = templist[p] + np.random.choice([1, -1])
                templist[p - (2 * (p - 14))] = templist[p]
        elif isinstance(plist[p], str) and ("true" in plist[p] or "false" in plist[p]):
            templist[p] = random.choice(["true", "false"])
        else:
            ###### FOR MUT2.1 ######
            # mutations = [-0.3*currval_float,-0.1*currval_float,0.1*currval_float,0.3*currval_float]
            ###### FOR MUT2.2 ######
            mutations = [
                -np.round(np.random.uniform(10, 100), 1),
                -np.round(np.random.uniform(1, 10), 1),
                -np.round(np.random.uniform(0.1, 1), 1),
                np.round(np.random.uniform(0.1, 1), 1),
                np.round(np.random.uniform(1, 10), 1),
                np.round(np.random.uniform(10, 100), 1),
            ]
            if isinstance(plist[p], str) == True and "M_PI" in plist[p]:
                # testval = ("M_PI*" + str(round(random.uniform(0.05,0.5),5)))
                # Extact number and sample in same way
                currval = re.findall(r"\d+(\.\d+)?", templist[p])
                currval_float = float(currval[0])
                templist[p] = "M_PI*" + str(
                    round(currval_float + np.random.choice(mutations), 10)
                )
            else:
                # need to round to 10 decimal places as python does incorrect arithmetic when subtracting
                currval_float = float(templist[p])
                templist[p] = round(templist[p] + np.random.choice(mutations), 10)

    elif scheme == "mut3":
        if 7 < p < 21:
            if np.isnan(plist[p]):
                templist[p] = np.nan
            elif 7 < p < 14:
                # testval = templist[p] + np.random.choice([1,-1])*testvals[n]
                # Search space of values in the starting leaves
                templist[p] = templist[p] + np.random.choice([1, -1])
                templist[p + (2 * (14 - p))] = templist[
                    p
                ]  # do the same to the opposite side of the array
            elif p == 14:
                templist[p] = 0
            elif 14 < p < 21:
                templist[p] = templist[p] + np.random.choice([1, -1])
                templist[p - (2 * (p - 14))] = templist[p]

    elif scheme == "mut4.1":
        mutations = [
            np.random.uniform(0, 100),
            np.random.uniform(100, 1000),
            np.random.uniform(1000, 10000),
        ]  # , np.random.uniform(10000, 100000)]
        multiplier = np.random.choice([1, -1])
        templist[p] = abs(
            round(templist[p] + (multiplier * np.random.choice(mutations)), 10)
        )

    elif scheme == "mut4.2":
        mutations = [
            np.random.uniform(0, 10),
            np.random.uniform(10, 100),
            np.random.uniform(100, 1000),
        ]
        multiplier = np.random.choice([1, -1])
        templist[p] = round(
            templist[p] + (multiplier * np.random.choice(mutations)), 10
        )

    return templist


def leafchecker(
    margin, lamina_margintest, lamina_veinstest, veins, silhouette, metrics
):
    """Checks whether the leaf is valid or not"""
    failed_conditions = []

    ew = equalweightchecker(silhouette, veins, metrics)
    pr = primordiumchecker(silhouette, metrics)
    om = overlappingmarginchecker(margin, lamina_margintest, metrics)
    vw = veinswidthchecker(veins, metrics)
    vo = veinsioutsidelaminachecker(lamina_veinstest, veins, metrics)

    # N.B. these conditions will return false if some of the leaf components are empty matrices e.g. the eroded lamina is empty,
    # therefore, either the condition is violated or there were some missing leaf components

    if not pr:
        failed_conditions.append("Arrested development")
    if not om:
        failed_conditions.append("Overlapping margin")
    if not vw:
        failed_conditions.append("Veins too narrow")
    if not vo:
        failed_conditions.append("Veins outside lamina")
    if not ew:
        failed_conditions.append("Non-equal weight")

    if not failed_conditions:
        check = True
    else:
        check = False

    return check, failed_conditions


#################################### Leaf Features ####################################


def leafcomponents(leafid, wid, step):
    """Extracts key visual components from the leaf image"""
    # returns them as a np array - "black" has the greyscale value of 32 in this case
    leaf = cv2.imread(
        wd + f"/bin{wid}/leaf_{leafid}_{wid}_{step}.png", cv2.IMREAD_GRAYSCALE
    )  # Need greyscale so matrix is 2D

    silhouette = (leaf < 255).astype(np.uint8)
    margin = (leaf == 32).astype(np.uint8)
    veins = (np.logical_and(leaf > 135, leaf < 255)).astype(np.uint8)

    # to get the lamina, we take the green part of the leaf (no veins) plus the margin, dilate this to remove the vein gaps, then trim back to the silhouette minus the margin
    lamina_margin_noveins = (leaf == 134).astype(np.uint8) + margin
    # cv2.imwrite(wd + "/LeafGenerator/laminanoveins.png", lamina_margin_noveins)

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lamina_margin_noveins_dilated = cv2.dilate(
        lamina_margin_noveins, dilation_kernel, iterations=4
    )
    # cv2.imwrite(wd + "/LeafGenerator/laminanoveins_dilated.png", lamina_margin_noveins_dilated)

    # we trim off the excess lamina by taking the overlap with the slightly eroded silhouette
    lamina_trimmed = (np.logical_and(lamina_margin_noveins_dilated, silhouette)).astype(
        np.uint8
    )
    # cv2.imwrite(wd + "/LeafGenerator/lamina_trimmed.png", lamina_trimmed)
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )  # np.ones((2, 2), np.uint8)
    lamina_margintest = cv2.erode(
        lamina_trimmed, erosion_kernel, iterations=6
    )  # WAS 3 BEFORE 13-08-2023 WHEN I MADE THE MARGINS THICKER
    lamina_veinstest = cv2.erode(lamina_trimmed, erosion_kernel, iterations=1)

    return margin, lamina_margintest, lamina_veinstest, veins, silhouette


def barfinder(matrix):
    """Determines whether a scale bar is visible and if there is, determines its size and location"""
    matrixflipped = np.flip(matrix)
    # np.savetxt(wd + "/matrixflipped.csv", matrixflipped, delimiter=",")
    isbar = False
    row_number = 0
    for row in matrixflipped:
        if 1 in row:
            break
        row_number += 1
    # print(f"#### Bottom = {row_number}")
    matrixbottom = matrixflipped[: row_number + 1, :]  # includes the bar
    matrixtop = matrixflipped[row_number + 1 :, :]  # doesn't include the bar itself
    barrow = row_number  # gives the row of the bar in matrix flipped, in python format

    # extract the barrow in the non-flipped matrix to find the start and end of the bar
    bar = matrix[-(barrow + 1)]
    for counter, element in enumerate(list(bar)):
        if element == 1:
            firstcolumn = counter
            break
    bar_reversed = bar[::-1]
    for counter, element in enumerate(list(bar_reversed)):
        if element == 1:
            lastcolumn = len(bar_reversed) - counter
            break

    # np.savetxt(wd + "matrix_abovebar.csv", matrix_abovebar, delimiter=",")
    for i, row in enumerate(
        matrixtop[:-1]
    ):  # check if there is a row of 0s separating the bar from the rest of the leaf
        current_row = matrixtop[i]
        next_row = matrixtop[i + 1]

        if np.all(current_row == 0) and np.any(next_row == 1):
            isbar = True

    print(f"#### Bar Found: {isbar}")
    print(f"#### Barrow index: {barrow}")
    return isbar, firstcolumn, lastcolumn, matrixbottom


def middlefinder(isbar, veins, firstcolumn, lastcolumn):
    """Finds the middle column using the scale bar if it exists, or if not the midvein"""
    # only use the bar to find the midpoint if there are two parts to the silhouette, separated by at least one blank row
    if isbar:
        middle = int((firstcolumn + lastcolumn) / 2)
        nmaxlength = (
            lastcolumn - firstcolumn
        )  # return the length of the bar as nmaxlength
        print(f"#### Middle (bar): {middle}")
        return middle, nmaxlength
    # if there is no bar, instead choose the column with the longest uninterrupted stretch of 1s
    else:
        maxlength_veins, maxlength_i, nmaxlength = maxlength(veins)
        if (
            nmaxlength % 2 == 0
        ):  # if nmaxlength is even, set middle to the column right of the centre line
            middle = maxlength_i + int(
                nmaxlength / 2
            )  # N.B int() rounds down to nearest integer
        else:  # if odd, set middle to column in the centre group
            middle = maxlength_i + int(nmaxlength / 2)
        print(f"#### Middle (veins): {middle}")
        return middle, nmaxlength


def maxwidth(matrix):
    """Finds the length of the longest uninterrupted row of 1s"""
    max_length = 0
    for row in matrix:
        row_length = 0
        for value in row:
            if value == 1:
                row_length += 1
                if row_length > max_length:
                    max_length = row_length
            else:
                row_length = 0
    # print(f"#### Max width = {max_length}")
    return max_length


def maxwidth_interrupted(matrix):
    """Returns the column number of the furthest left and furthest right non-zero value in a matrix"""
    nonzero_cols = np.where(matrix.any(axis=0))[0]
    furthest_left = nonzero_cols[0]
    furthest_right = nonzero_cols[-1]
    maxwidth_interrupted = furthest_right - furthest_left
    return maxwidth_interrupted


def maxlength(matrix):
    """Finds the maximum uninterrupted vertical stretch of 1s and the associated column index"""
    maxlength = 0
    nmaxlength = 0
    maxlength_i = np.NAN
    for i, col in enumerate(range(len(matrix[0]))):
        collength = 0
        for row in matrix:
            if np.any(row[col] == 1):
                collength += 1
                if collength > maxlength:
                    maxlength = collength
                    maxlength_i = i
                    nmaxlength = 1  # set n to 1 if new maximum found
                elif collength == maxlength:
                    nmaxlength += (
                        1  # add 1 for every additional column equal to the current max
                    )
            else:
                collength = (
                    0  # reset collength to 0 if the stretch of 1s is interrupted
                )
    return (
        maxlength,
        maxlength_i,
        nmaxlength,
    )  # returns maxlength in matrix and the accompanying column index


#################################### Checks ####################################


def equalweightchecker(matrix, veins, metrics):
    """Checks if the difference in area between each side of the leaf, normalised by the total leaf area exceeds a threshold"""
    if (not (matrix == 1).any()) or (not (veins == 1).any()):
        return False
    else:
        isbar, firstcolumn, lastcolumn, _ = barfinder(matrix)
        middle, nmaxlength = middlefinder(isbar, veins, firstcolumn, lastcolumn)
        if nmaxlength % 2 == 0:
            matrix_left = matrix[:, 0:middle]
            matrix_right = matrix[:, middle:512]
        else:
            matrix_left = matrix[
                :, 0 : (middle + 1)
            ]  # if the nmaxlength is odd, we need the left side to also include the centre column
            matrix_right = matrix[:, middle:512]
        # cv2.imwrite(wd + "matrix_left.png", matrix_left)
        # cv2.imwrite(wd + "matrix_right.png", matrix_right)
        # to make sure the left and right matrices are the same size even when the middle of the leaf is not in the middle of the image, we add empty columns and then trim to equal sizes
        nrows = matrix_left.shape[0]
        buffermatrix = np.zeros([nrows, 300])
        matrix_left_aug = np.concatenate((buffermatrix, matrix_left), axis=1)[:, -300:]
        matrix_right_aug = np.concatenate((matrix_right, buffermatrix), axis=1)[:, :300]
        # cv2.imwrite(wd + "matrix_left_aug.png", matrix_left_aug)
        # np.savetxt(wd + "matrix_left_aug.csv", matrix_left_aug, delimiter=",")
        # cv2.imwrite(wd + "matrix_right_aug.png", matrix_right_aug)
        # np.savetxt(wd + "matrix_right_aug.csv", matrix_right_aug, delimiter=",")

        weightdifference = abs(np.sum(matrix_left_aug) - np.sum(matrix_right_aug))
        weightdifference_prop = weightdifference / np.sum(matrix)
        # print(np.sum(matrix_left), np.sum(matrix_right_aug), print(weightdifference))
        metrics.append(weightdifference_prop)
        metrics.append(middle)
        print(f"#### Weight difference prop. = {weightdifference_prop}")
        return weightdifference_prop < weightdifference_thresh


def primordiumchecker(matrix, metrics):
    """Checks if the width of the leaf exceeds a certain multiple of the scalebar"""
    if not (matrix == 1).any():
        return False
    else:
        _, _, _, matrixbottom = barfinder(matrix)
        barwidth = maxwidth(matrixbottom)
        leafwidth = maxwidth(matrix)
        print(f"#### Bar width = {barwidth}", f"#### Leaf width = {leafwidth}")
        metrics.append(leafwidth)
    return leafwidth > barwidth * primordium_thresh


def overlappingmarginchecker(margin, lamina, metrics):
    """Returns False if the proportion of lamina occupied by margin exceeds a threshold"""
    # we immediately return false if lamina is empty
    if (not (lamina == 1).any()) or (
        not (margin == 1).any()
    ):  # if this is true it means there are no 1s in the matrix
        return False
    else:
        overlap = (np.logical_and(margin, lamina)).astype(np.uint8)
        # cv2.imwrite(wd + "/LeafGenerator/overlap.png", overlap)
        prop_overlap = np.sum(overlap) / np.sum(lamina)
        # prop_overlap = np.sum(overlap)/np.sum(margin)
        print(f"#### Prop. margin inside lamina = {prop_overlap}")
        metrics.append(prop_overlap)
        return prop_overlap < overlappingmargin_thresh


def minveinschecker(silhouette, veins, metrics):
    """Returns False if the proportion of lamina occupied by veins is less than a threshold"""
    # we immediately return false if veins is empty
    if not (veins == 1).any():  # if this is true it means there are no 1s in the matrix
        return False
    else:
        veinarea_insideleaf = np.logical_and(silhouette, veins)
        prop_veinarea = np.sum(veinarea_insideleaf) / np.sum(silhouette)
        print(f"#### Prop. silhouette area that is veins = {prop_veinarea}")
        metrics.append(prop_veinarea)
        return prop_veinarea > veinarea_thresh


def veinswidthchecker(veins, metrics):
    """Returns False if the width of the veins structure is less than a threshold, or if veins is a blank matrix"""
    # we immediately return false if veins is empty
    if not (veins == 1).any():  # if this is true it means there are no 1s in the matrix
        return False
    else:
        veinswidth = maxwidth_interrupted(veins)
        imagewidth = np.shape(veins)[1]
        print(f"#### Vein maxwidth = {veinswidth}")
        metrics.append(veinswidth)
        return veinswidth > imagewidth * veinswidth_thresh


def veinsioutsidelaminachecker(lamina, veins, metrics):
    """Returns False if the proportion of the veins non-overlapping with the lamina exceeds a threshold, or if veins is blank"""
    # we immediately return false if veins is empty
    if not (veins == 1).any():  # if this is true it means there are no 1s in the matrix
        return False
    else:
        veinsoutsidelamina = veins - np.logical_and(veins, lamina)
        # np.savetxt(wd + "/LeafGenerator/veinsoutsidelamina.csv", veinsoutsidelamina, delimiter=",")
        # calculate as a proportion of the total vein area
        prop_veinsoutsidelamina = np.sum(veinsoutsidelamina) / np.sum(veins)
        print(f"#### Prop. vein area outside lamina = {prop_veinsoutsidelamina}")
        metrics.append(prop_veinsoutsidelamina)
        return prop_veinsoutsidelamina < veinsoutsidelamina_thresh


#################################### Process Management ####################################


def randomwalk(wid, leafid):
    """Runs the random walk on a given leafid until one of the 10 parallel walks generates a threshold amount"""
    # set random seed to ensure the random numbers occur independently
    np.random.seed(wid)

    leafid_index = leafids.index(leafid)
    plist = [ele[leafid_index] for ele in list(pdict.values())]
    plist_i = copy.deepcopy(plist)
    report = []
    ngenerated_thresh = False
    print(f"#### Starting parameter values ####\n{plist}")
    try:
        # for step in range(40): #nrounds*len(plist)):
        # 	plist_i = testparams(plist, plist_i, step, wid, leafid, report)
        step = 0
        # only moves onto the next leafid once a threshold number of leaves have been generated
        while ngenerated_thresh == False:
            plist_i = testparams(plist, plist_i, step, wid, leafid, report)
            step += 1
            n0 = len(os.listdir(wd + f"/leaves/{leafid}/walk0"))
            n1 = len(os.listdir(wd + f"/leaves/{leafid}/walk1"))
            n2 = len(os.listdir(wd + f"/leaves/{leafid}/walk2"))
            n3 = len(os.listdir(wd + f"/leaves/{leafid}/walk3"))
            n4 = len(os.listdir(wd + f"/leaves/{leafid}/walk4"))
            n5 = len(os.listdir(wd + f"/leaves/{leafid}/walk5"))
            n6 = len(os.listdir(wd + f"/leaves/{leafid}/walk6"))
            n7 = len(os.listdir(wd + f"/leaves/{leafid}/walk7"))
            n8 = len(os.listdir(wd + f"/leaves/{leafid}/walk8"))
            n9 = len(os.listdir(wd + f"/leaves/{leafid}/walk9"))

            if (
                n0 > ngen_thresh
                or n1 > ngen_thresh
                or n2 > ngen_thresh
                or n3 > ngen_thresh
                or n4 > ngen_thresh
                or n5 > ngen_thresh
                or n6 > ngen_thresh
                or n7 > ngen_thresh
                or n8 > ngen_thresh
                or n9 > ngen_thresh
            ):
                ngenerated_thresh = True
                print("#### n_generated threshold reached!")
    finally:
        with open(
            wd + f"/leaves/{leafid}/walk{wid}/report_{leafid}_{wid}.csv", "w"
        ) as csvfile:
            for step in report:
                csvfile.write(",".join(str(elem) for elem in step)[1:-1] + "\n")


def start():
    """For each leafid, starts [ncores] random walks, moving onto the next when all processes are finished"""
    # Start an Xcfb virtual screen on which the leaves can generate while remaining hidden
    display = Display(visible=0, size=(1366, 768))
    display.start()

    if __name__ == "__main__":
        processes = []
        for leafid in leafids[startleaf:]:  # [-39:]:
            for wid in range(ncores):
                process = multiprocessing.Process(target=randomwalk, args=(wid, leafid))
                processes.append(process)
                process.start()

            # this waits for each wid process to finish before moving onto the next leafid
            for process in processes:
                process.join()

    display.stop()


if __name__ == "__main__":
    # start()
    pass

######################################################################################################################
#################################### The Remaining Code is for debugging Purposes ####################################
######################################################################################################################


def generatedefaults():
    """Generates starting leaves from default parameter values"""
    for i, leafid in enumerate(leafids):
        print(f"############## LEAF {leafid} ##############")
        wid = 0
        step = 0

        leafid_index = leafids.index(leafid)
        templist = [ele[leafid_index] for ele in list(pdict.values())]
        print(templist)

        runsim(step, templist, wid, leafid)


def generatedefault():
    """Generates starting leaf from default parameter values"""

    leafid = "p12f"
    step = 0
    wid = 0
    print(f"############## LEAF {leafid} ##############")

    leafid_index = leafids.index(leafid)
    templist = [ele[leafid_index] for ele in list(pdict.values())]
    # set morphogens to visible
    templist[-7] = 0

    runsim(step, templist, wid, leafid)


generatedefault()
# generatedefaults()


def leafmeasure(mode):
    """Prints out the morphometrics by which leafs are assessed for validity"""

    keys = [
        "leafid",
        "prop_weightdifference",
        "middle",
        "leafwidth",
        "prop_overlappingmargin",
        "prop_veinarea",
        "veinswidth",
        "prop_veinsoutsidelamina",
    ]
    # keys = ["leafid", "prop_nonoverlapping", "leafwidth", "prop_overlappingmargin", "prop_veinarea", "veinswidth", "prop_veinsoutsidelamina"]
    # metrics = {key: [] for key in keys}
    metrics_list = []
    metrics_dict = {}

    if mode == "single":
        leafid = 0
        step = 0
        wid = 0
        metrics = []
        metrics.append(leafid)

        margin, lamina_margintest, lamina_veinstest, veins, silhouette = leafcomponents(
            leafid, wid, step
        )

        if (
            # symmetrychecker(silhouette,metrics) and \
            equalweightchecker(silhouette, veins, metrics)
            and primordiumchecker(silhouette, metrics)
            and overlappingmarginchecker(margin, lamina_margintest, metrics)
            and minveinschecker(  # overlappingmarginchecker1(hierarchies_subset,metrics)
                silhouette, veins, metrics
            )
            and veinswidthchecker(veins, metrics)
            and veinsioutsidelaminachecker(lamina_veinstest, veins, metrics)
        ):
            print("LEAF PASSES")
        else:
            print("LEAF FAILS")

    elif mode == "batch":
        for i, leafid in enumerate(leafids):
            print(f"############## LEAF {leafid} ##############")
            metrics = []
            metrics.append(leafid)
            wid = 0
            step = 0

            (
                margin,
                lamina_margintest,
                lamina_veinstest,
                veins,
                silhouette,
            ) = leafcomponents(leafid, wid, step)
            # symmetrychecker(silhouette,metrics)
            equalweightchecker(silhouette, veins, metrics)
            primordiumchecker(silhouette, metrics)
            overlappingmarginchecker(margin, lamina_margintest, metrics)
            # overlappingmarginchecker1(hierarchies_subset,metrics)
            minveinschecker(silhouette, veins, metrics)
            veinswidthchecker(veins, metrics)
            veinsioutsidelaminachecker(lamina_veinstest, veins, metrics)

            metrics_list.append(metrics)

        # for i in range(len(keys)):
        # 	metrics_dict[keys[i]] = metrics_list[i]

        with open("REPORT.csv", mode="w") as csv_file:
            # Write the header row
            csv_file.write(",".join(keys) + "\n")

            # Write the data rows
            for row in metrics_list:
                csv_file.write(",".join(map(str, row)) + "\n")

    else:
        print("ERROR: Invalid Mode")


# leafmeasure("batch")
