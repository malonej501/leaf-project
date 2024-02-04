import cv2
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks
from scipy import ndimage
import seaborn
from matplotlib import pyplot as plt
import multiprocessing

# Bigger means more blurring, needs to be a positive odd integer
kernel_size = (5, 5)
contourbuff_size = 40
wd = "/home/m/malone/vlab-5.0-ubuntu-20.04/oofs/ext/NPHLeafModels_1.01"
wd1 = "/home/m/malone/vlab-5.0-ubuntu-20.04/oofs/ext/NPHLeafModels_1.01/LeafGenerator/testleaves/"

randomwalkleaves = "/home/m/malone/leaf_storage/random_walks/leaves_full_11-10-23_MUT3"


def imageprocessing(leaf):
    # binarise the image
    ret1, thresh = cv2.threshold(leaf, 254, 255, cv2.THRESH_BINARY)

    # extract the contour for the leaf outline, SIMPLE means only the points that are not in a straight line from the previous are returned
    contours, heirarchies = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    contours_compressed, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    longest_contour_index = np.argmax([contour.size for contour in contours_compressed])

    contour = contours[longest_contour_index]
    contour_compressed = contours_compressed[longest_contour_index]

    # extract the veins
    veins = (np.logical_and(leaf > 134, leaf < 255)).astype(np.uint8)

    # create an image for overlaying points of interest on
    img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    return contour, contour_compressed, veins, img, thresh


def centreveinsfinder(veins):
    # Need to add the condition that the stretch of uninterrupted 1s is centred on the middle of the image
    # Or just the first branching point of the veins - start from the bottom work up until the width gets larger
    max_length = 0
    for row_index, row in enumerate(veins):
        row_length = 0
        for value in row:
            if value == 1:
                row_length += 1
                if row_length > max_length:
                    max_length = row_length
                    max_length_index = row_index
            else:
                row_length = 0
    # print(f"#### Max width = {max_length}")
    # print(f"#### Max length index = {max_length_index}")
    x = int(round(np.shape(veins)[1] / 2))
    y = max_length_index
    return (x, y)


def barfinder(rawmatrix):
    matrix = np.invert(np.array(rawmatrix, dtype=bool))
    matrixflipped = np.flip(matrix)
    row_number = 0
    for row in matrixflipped:
        if 1 in row:
            break
        row_number += 1
    # print(f"#### Bottom = {row_number}")
    matrixbottom = matrixflipped[: row_number + 1, :]
    barrow = row_number + 1

    return barrow, matrixbottom


def middlefinder(rawmatrix, barrow):
    matrix = np.invert(np.array(rawmatrix, dtype=bool))
    # extract the barrow in the non-flipped matrix
    bar = matrix[-barrow, :]
    for counter, element in enumerate(list(bar)):
        if element == 1:
            column = counter
            break
    middle = int(column + (np.sum(bar) / 2))
    # print(middle)

    return middle


def centreveinsfinder_alt(veins, centre_col):
    max_length_column_indices = []
    max_length_index = 0
    for row_index in range(len(veins) - 1, -1, -1):
        row_length = 0
        max_length_column_indices_i = []
        for i, value in enumerate(veins[row_index]):
            if value == 1:
                row_length += 1
                max_length_column_indices_i.append(i)

        longest_seq = []
        temp_seq = []
        for i in range(len(max_length_column_indices_i)):
            if (
                i == 0
                or max_length_column_indices_i[i]
                == max_length_column_indices_i[i - 1] + 1
            ):
                temp_seq.append(max_length_column_indices_i[i])
            else:
                # only accepts longest sequences that also contain the centre_col
                if len(temp_seq) > len(longest_seq) and centre_col in temp_seq:
                    longest_seq = temp_seq
                temp_seq = [max_length_column_indices_i[i]]
        if len(temp_seq) > len(longest_seq) and centre_col in temp_seq:
            longest_seq = temp_seq

        max_length_column_indices.append(longest_seq)

    startveinwidth = 0
    for startveinindex_inv, row in enumerate(max_length_column_indices):
        if any(row):
            startveinwidth = len(row)
            break

    for row_index_inv, row in enumerate(max_length_column_indices):
        if len(row) > (startveinwidth + 1):
            max_length_index = len(max_length_column_indices) - row_index_inv
            break
        # If there is no length greater than the startingveinwidth+1, take the index of startingveinwidth
        else:
            max_length_index = len(max_length_column_indices) - startveinindex_inv

    # #if the algorithm proposes a centreveins point very high up on the leaf we change it to the middle of the image
    if max_length_index < round(0.3 * 490):
        max_length_index = round(0.6 * 490)
    print(f"#### Max length index = {max_length_index}")

    x = centre_col
    y = max_length_index
    # print(x,y)

    return (x, y)


def dist_to_point(rawcontour, refpoint):
    contour = rawcontour[:, 0]

    refdist = []

    for coordinate in contour:
        # Pythagoras' Theorem
        refdist_n = np.sqrt(
            (coordinate[0] - refpoint[0]) ** 2 + (coordinate[1] - refpoint[1]) ** 2
        )
        refdist.append(refdist_n)

    # paste some of the contour at the end onto the start so that the middle lobe is identified as a maxima
    startbuffer = refdist[-contourbuff_size:]
    endbuffer = refdist[:contourbuff_size]
    refdist_buffered = np.concatenate((startbuffer, refdist))

    return refdist_buffered


def getextrema(contour, refdist):
    refdist_inv = [-1 * i for i in refdist]

    local_maxima_indices, _ = find_peaks(refdist, prominence=25, distance=10)
    local_minima_indices, _ = find_peaks(refdist_inv, prominence=25, distance=10)

    local_maxima_indices -= contourbuff_size
    local_minima_indices -= contourbuff_size

    local_maxima = contour[local_maxima_indices]
    local_minima = contour[local_minima_indices]

    return local_maxima, local_minima, local_maxima_indices, local_minima_indices


def morphometrics(
    rawcontour, local_maxima_indices, local_minima_indices, local_minima, refpoint
):
    contour = rawcontour[:, 0]

    minmax_dist_avg_i = []
    refmin_dist_i = []
    refmax_dist_i = []
    minmax_angle_i = []
    minmax_resultant_i = []
    minima_samerow_dist_i = []

    # print(local_minima_indices)
    # print(local_maxima_indices)

    if local_minima_indices.any() and local_maxima_indices.any():
        # add the last and first local_minima_indices to the start and end of local_minima indices respectively, so that the minima preceeding the first maxima and proceeding the last maxima can be returned later
        local_minima_indices_buffered = np.concatenate(
            (
                [local_minima_indices[-1]],
                local_minima_indices,
                [local_minima_indices[0]],
            )
        )
        # print(len(local_minima_indices))
        # print(local_minima_indices_buffered)
        # print(len(local_minima_indices_buffered))

        for i, value in enumerate(local_maxima_indices):
            # PROBLEM: THE BELOW AND ABOVE ARE NOT ALWAYS RIGHT - NEED TO MAKE IT ROBUST TO DIFFERENT NO. EXTREMA
            # THINK IT MIGHT HAVE BEEN SOLVED
            # For the first maxima, the adjacent minima are the 0th and 1st of the buffered list
            if i == 0:
                below = local_minima_indices_buffered[i]
                above = local_minima_indices_buffered[i + 1]
            # If the maxima is not the first or last and is greater than the 1st of the buffered minima list, take the closest value in the buffered list above and below
            if (
                value > local_minima_indices_buffered[1]
                and i != len(local_maxima_indices) - 1
            ):
                below = max([x for x in local_minima_indices if x < value])
                above = min([x for x in local_minima_indices if x > value])
            # For the last maxima, the adjacent minima are the last and penultimate of the buffered list
            if i == len(local_maxima_indices) - 1:
                below = local_minima_indices_buffered[-2]
                above = local_minima_indices_buffered[-1]

            local_maxima = contour[value]
            local_minima_below = contour[below]
            local_minima_above = contour[above]

            minmax_below_dist = np.sqrt(
                (local_maxima[0] - local_minima_below[0]) ** 2
                + (local_maxima[1] - local_minima_below[1]) ** 2
            )
            minmax_above_dist = np.sqrt(
                (local_maxima[0] - local_minima_above[0]) ** 2
                + (local_maxima[1] - local_minima_above[1]) ** 2
            )
            refmax_dist = np.sqrt(
                (local_maxima[0] - refpoint[0]) ** 2
                + (local_maxima[1] - refpoint[1]) ** 2
            )

            a = np.sqrt(
                (local_minima_above[0] - local_minima_below[0]) ** 2
                + (local_minima_above[1] - local_minima_below[1]) ** 2
            )
            b = minmax_above_dist
            c = minmax_below_dist
            minmax_angle = np.rad2deg(
                np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
            )

            v1 = [
                local_maxima[0] - local_minima_below[0],
                local_maxima[1] - local_minima_below[1],
            ]
            v2 = [
                local_maxima[0] - local_minima_above[0],
                local_maxima[1] - local_minima_above[1],
            ]
            minmax_resultant = np.subtract(v1, v2)
            minmax_resultant_mag = np.linalg.norm(minmax_resultant)

            minmax_dist_avg_i.append((minmax_above_dist + minmax_below_dist) / 2)
            minmax_dist_avg_i.append(
                min([minmax_above_dist, minmax_below_dist])
            )  # instead of averaging the two, we take the lower of the two values
            refmax_dist_i.append(refmax_dist)
            minmax_angle_i.append(minmax_angle)
            minmax_resultant_i.append(minmax_resultant_mag)

        local_minima_alt = local_minima[:, 0]

        rows = []
        for minima in local_minima_alt:
            samerow = (local_minima_alt[local_minima_alt[:, 1] == minima[1]]).tolist()
            if len(samerow) > 1 and samerow not in rows:
                rows.append(samerow)

        if len(rows) > 1:
            for row in rows:
                xvals = [sublist[0] for sublist in row]
                lhs = [x for x in xvals if x <= refpoint[0]]
                rhs = [x for x in xvals if x >= refpoint[0]]
                # check that there is a point both on the right and left of the midvein
                if lhs and rhs:
                    closestleftside = max(lhs)
                    closestrightside = min(rhs)
                    minima_samerow_dist = closestrightside - closestleftside
                    minima_samerow_dist_i.append(minima_samerow_dist)

    else:
        minmax_dist_avg_i.append(np.nan)
        refmax_dist_i.append(np.nan)
        minmax_angle_i.append(np.nan)
        minmax_resultant_i.append(np.nan)
        minima_samerow_dist_i.append(np.nan)

    minmax_dist_avg = np.mean(minmax_dist_avg_i)
    refmax_dist_avg = np.mean(refmax_dist_i)
    minmax_angle_avg = np.mean(minmax_angle_i)
    minmax_resultant_avg = np.mean(minmax_resultant_i)
    minima_samerow_dist_avg = np.mean(minima_samerow_dist_i)

    unlobed_ub = (
        0.15 * refmax_dist_avg
    )  # unlobed = where the maximum distance between the lobe tips and sinuses is less than 30% of the average distance from the lobe tips to the refpoint (centre of veins)
    lobed_ub = (
        0.4 * refmax_dist_avg
    )  # lobed = where the above measurement is greater than 30% but less than 60% of the average distance from the lobe tips to the refpoint
    dissected_ub = 0.8 * refmax_dist_avg

    all_extrema = np.concatenate((local_maxima_indices, local_minima_indices))

    return (
        minmax_dist_avg,
        refmax_dist_avg,
        minmax_angle_avg,
        minima_samerow_dist_avg,
        unlobed_ub,
        lobed_ub,
        dissected_ub,
        all_extrema,
    )  # , refmin_dist_4lowest_max


def classifier_old(
    minmax_dist_avg,
    minmax_angle_avg,
    minima_samerow_dist_avg,
    unlobed_ub,
    lobed_ub,
    dissected_ub,
    all_extrema,
):  # , refmin_dist_4lowest_max):
    if (
        minmax_dist_avg >= dissected_ub
        and len(all_extrema) > 5
        and minmax_angle_avg < 30
    ) or (minima_samerow_dist_avg < 25 and len(all_extrema) > 5):
        return "c"
    elif (
        minmax_dist_avg >= lobed_ub
        and minmax_dist_avg < dissected_ub
        and len(all_extrema) > 5
        and minmax_angle_avg < 30
    ):
        return "d"
    elif (
        minmax_dist_avg >= unlobed_ub
        and minmax_dist_avg < lobed_ub
        and len(all_extrema) > 5
        and minmax_angle_avg < 50
    ):
        return "l"
    elif minmax_dist_avg < unlobed_ub or len(all_extrema) <= 5 or minmax_angle_avg > 50:
        return "u"
    else:
        return "unclassified"


def classifier(
    minmax_dist_avg,
    minmax_angle_avg,
    minima_samerow_dist_avg,
    unlobed_ub,
    lobed_ub,
    dissected_ub,
    all_extrema,
):  # , refmin_dist_4lowest_max):
    if (len(all_extrema) > 5 and minmax_angle_avg < 8) or (
        minima_samerow_dist_avg < 25 and len(all_extrema) > 5
    ):
        return "c"
    if len(all_extrema) > 5 and minmax_angle_avg < 30:
        return "d"
    if len(all_extrema) > 5 and minmax_angle_avg < 60:
        return "l"
    if len(all_extrema) <= 5 or minmax_angle_avg > 60:
        return "u"
    else:
        return "unclassified"


def main_batch(directory):
    # remove all previous pertinent points images
    for filename in os.listdir(directory):
        if "-points" in filename or "-graph" in filename:
            os.remove(directory + "/" + filename)

    # load all the leaf image files
    files = os.listdir(directory)
    image_files = [f for f in files if f.endswith(".png")]
    image_files.sort()
    image_files.sort(key=lambda s: int("".join(filter(str.isdigit, s))))

    report = pd.DataFrame(
        columns=[
            "leaf",
            "shape",
            "no.extrema",
            "minmax_dist_avg",
            "minmax_angle_avg",
            "minima_samerow_dist_avg",
            "refmax_dist_avg",
            "unlobed_ub",
            "lobed_ub",
        ]
    )

    for file in image_files:
        print(f"#### Leaf: {file}")

        # import greyscale image
        leaf = cv2.imread(directory + "/" + file, cv2.IMREAD_GRAYSCALE)

        # extract the contours and key initial morphometrics
        contour, contour_compressed, veins, img, thresh = imageprocessing(leaf)
        barrow, _ = barfinder(thresh)
        centre_col = middlefinder(thresh, barrow)
        refpoint = centreveinsfinder_alt(veins, centre_col)
        refdist = dist_to_point(contour, refpoint)

        (
            local_maxima,
            local_minima,
            local_maxima_indices,
            local_minima_indices,
        ) = getextrema(contour, refdist)

        # annotate the silhouette of the leaf with the pertinent points and write to file
        for points in local_maxima:
            for x, y in points:
                cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
        for points in local_minima:
            for x, y in points:
                cv2.circle(img, (x, y), 10, (255, 191, 0), -1)
        cv2.circle(img, refpoint, 10, (0, 165, 255), -1)
        cv2.imwrite(directory + f"/{file[:-4]}-points.png", img)

        seaborn.scatterplot(refdist, color="black", edgecolor="none")
        seaborn.scatterplot(
            y=refdist[local_maxima_indices + 40],
            x=local_maxima_indices + 40,
            color="red",
            s=150,
            edgecolor="none",
        )
        seaborn.scatterplot(
            y=refdist[local_minima_indices + 40],
            x=local_minima_indices + 40,
            color="#00bfff",
            s=150,
            edgecolor="none",
        )
        plt.xlabel("Contour position", fontsize=16)
        plt.ylabel("Distance from ref. point", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
        plt.savefig(directory + f"/{file[:-4]}-graph.png")
        plt.clf()

        (
            minmax_dist_avg,
            refmax_dist_avg,
            minmax_angle_avg,
            minima_samerow_dist_avg,
            unlobed_ub,
            lobed_ub,
            dissected_ub,
            all_extrema,
        ) = morphometrics(
            contour, local_maxima_indices, local_minima_indices, local_minima, refpoint
        )
        category = classifier(
            minmax_dist_avg,
            minmax_angle_avg,
            minima_samerow_dist_avg,
            unlobed_ub,
            lobed_ub,
            dissected_ub,
            all_extrema,
        )
        # print(category)

        rowdf = pd.DataFrame.from_dict(
            {
                "leaf": file,
                "shape": category,
                "no.extrema": len(all_extrema),
                "minmax_dist_avg": minmax_dist_avg,
                "refmax_dist_avg": refmax_dist_avg,
                "minmax_angle_avg": minmax_angle_avg,
                "minima_samerow_dist_avg": minima_samerow_dist_avg,
                "unlobed_ub": unlobed_ub,
                "lobed_ub": lobed_ub,
                "dissected_ub": dissected_ub,
            },
            orient="index",
        ).T
        # "refmin_dist_4lowest_max": round(refmin_dist_4lowest_max)}, ignore_index=True)

        report = pd.concat([report, rowdf], ignore_index=True)

        # print(report)

    report.to_csv(directory + "/" + "shape_report.csv", index=False)


# def main(file):

#     # remove all previous pertinent points images
#     for filename in os.listdir(wd1):
#         if "-points" in filename or "-graph" in filename:
#             os.remove(wd1 + filename)

#     report = pd.DataFrame(columns=["leaf","shape","no.extrema",
#                                    "minmax_dist_avg","refmax_dist_avg",
#                                    "unlobed_ub","lobed_ub"])

#     print(f"#### Leaf: {file}")

#     leaf = cv2.imread(wd1 + file, cv2.IMREAD_GRAYSCALE)
#     print(np.shape(leaf))
#     #cv2.imwrite("grey.png", leaf)


#     contour, veins, img, thresh = imageprocessing(leaf)
#     barrow, _ = barfinder(thresh)
#     centre_col = middlefinder(thresh, barrow)
#     refpoint = centreveinsfinder_alt(veins, centre_col)
#     refdist = dist_to_point(contour, refpoint)
#     #exit()
#     local_maxima, local_minima, local_maxima_indices, local_minima_indices  = getextrema(contour, refdist)

#     #annotate the silhouette of the leaf with the pertinent points and write to file
#     for points in local_maxima:
#         for x,y in points:
#             cv2.circle(img, (x,y), 4, (0,0,255), -1)
#     for points in local_minima:
#         for x,y in points:
#             cv2.circle(img, (x,y), 4, (0,255,0), -1)
#     cv2.circle(img, refpoint, 4, (255,0,0), -1)
#     cv2.imwrite(wd1 + f"{file[:-4]}-points.png", img)

#     #print(local_maxima[1][0][0], local_maxima[1][0][1])

#     minmax_dist_avg, refmax_dist_avg, minmax_angle_avg, minmax_angle_min, unlobed_ub, lobed_ub, all_extrema = morphometrics(contour, local_maxima_indices, local_minima_indices, refpoint)
#     category = classifier(minmax_dist_avg, minmax_angle_avg, minmax_angle_min, unlobed_ub, lobed_ub, all_extrema)
#     print(category)

#     report = report.append({"leaf": file,
#                             "shape": category,
#                             "no.extrema": len(all_extrema),
#                             "minmax_dist_avg": minmax_dist_avg,
#                             "refmax_dist_avg": refmax_dist_avg,
#                             "unlobed_ub": unlobed_ub,
#                             "lobed_ub": lobed_ub}, ignore_index=True)

#     print(report)
#     report.to_csv(wd1 + "shape_report.csv", index=False)

#     seaborn.scatterplot(refdist, color="black", edgecolor="none")
#     seaborn.scatterplot(y=refdist[local_maxima_indices+20], x=local_maxima_indices+20, color="red", edgecolor="none")
#     seaborn.scatterplot(y=refdist[local_minima_indices+20], x=local_minima_indices+20, color="lightgreen", edgecolor="none")
#     plt.savefig(wd1 + f"{file[:-4]}-graph.png")
#     plt.clf()

# def main_batch(directory):
#     if "shape_report.csv" not in os.listdir(directory):
#         print(f"#### {directory} Missing Shape Report!")
#     # else:
#     #     print(f"#### {directory} Shape Report Found!")


def categorise(leaf):
    contour, contour_compressed, veins, img, thresh = imageprocessing(leaf)
    barrow, _ = barfinder(thresh)
    centre_col = middlefinder(thresh, barrow)
    refpoint = centreveinsfinder_alt(veins, centre_col)
    refdist = dist_to_point(contour, refpoint)
    # exit()
    local_maxima, local_minima, local_maxima_indices, local_minima_indices = getextrema(
        contour, refdist
    )

    (
        minmax_dist_avg,
        refmax_dist_avg,
        minmax_angle_avg,
        minmax_angle_min,
        unlobed_ub,
        lobed_ub,
        all_extrema,
    ) = morphometrics(contour, local_maxima_indices, local_minima_indices, refpoint)
    category = classifier(
        minmax_dist_avg,
        minmax_angle_avg,
        minmax_angle_min,
        unlobed_ub,
        lobed_ub,
        all_extrema,
    )

    return category == "l" or category == "c"


def refdistout(directory):
    # load all the leaf image files
    files = os.listdir(directory)
    image_files = [f for f in files if f.endswith(".png")]
    image_files.sort()
    image_files.sort(key=lambda s: int("".join(filter(str.isdigit, s))))

    leafidslist = []
    refdistlist = []

    for file in image_files:
        print(f"#### Leaf: {file}")

        leaf = cv2.imread(directory + "/" + file, cv2.IMREAD_GRAYSCALE)

        contour, contour_compressed, veins, img, thresh = imageprocessing(leaf)
        barrow, _ = barfinder(thresh)
        centre_col = middlefinder(thresh, barrow)
        refpoint = centreveinsfinder_alt(veins, centre_col)
        refdist = dist_to_point(contour, refpoint)

        # Pick 200 equally spaced points along the contour, to make contours all roughly the same length
        # round down to the nearest integer
        interval200 = int((len(refdist) / 200) // 1)
        refdist_compressed = refdist[::interval200]
        refdist_compressed200 = refdist_compressed[:200]
        # print(len(refdist_compressed200))

        leafidslist.append(file)
        refdistlist.append(refdist_compressed200)

    report = pd.DataFrame.from_dict(dict(zip(leafidslist, refdistlist)), orient="index")
    print(report)

    report.to_csv(directory + "/" + "refdist_report.csv", header=False)


# seaborn.set(style="ticks", rc={"figure.figsize":(10,7)})
# main("leaf1-1.png")


# main_batch(wd1)


def parallel(wd, function):
    if __name__ == "__main__":
        processes = []
        for leafdirectory in os.listdir(wd):
            leafdirectory_path = os.path.join(wd, leafdirectory)
            for walkdirectory in os.listdir(leafdirectory_path):
                walkdirectory_path = os.path.join(leafdirectory_path, walkdirectory)
                process = multiprocessing.Process(
                    target=function, args=(walkdirectory_path,)
                )
                processes.append(process)
                process.start()

            # this waits for each wid process to finish before moving onto the next leafid
            for process in processes:
                process.join()


# parallel(randomwalkleaves, main_batch)
# parallel(randomwalkleaves, refdistout)

# refdistout("/home/m/malone/vlab-5.0-ubuntu-20.04/oofs/ext/NPHLeafModels_1.01/LeafGenerator/trainingdata")

# path = randomwalkleaves + "/p7_277/walk9"

# if __name__ == "__main__":
#     for walkdirectory in os.listdir(path):
#         walkdirectory_path = os.path.join(path, walkdirectory)
#         process = multiprocessing.Process(target=main_batch, args=(walkdirectory_path,))
#         process.start()

# main_batch(wd1)
# main_batch("/home/m/malone/leaf_storage/accuracy_test_500")

# refdistout("/home/m/malone/leaf_storage/accuracy_test_200")
# main_batch("/home/m/malone/leaf_storage/Official Starting Sample/12-9-23")
# main_batch("/home/m/malone/leaf_storage/Official Starting Sample/more-symmetric_12-7-23")
main_batch(
    "/home/m/malone/vlab-5.0-ubuntu-20.04/oofs/ext/NPHLeafModels_1.01/LeafGenerator/starting_sample_12-9-23"
)