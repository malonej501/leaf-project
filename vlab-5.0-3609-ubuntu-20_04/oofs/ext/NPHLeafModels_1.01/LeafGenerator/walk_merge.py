import os
import shutil
import sys
from pdict import leafids

root = "../MUT5" # root directory containing separate leaffinder runs
merged_dir = root + "_merged" # new directory to store merged leaf directories
walk_len = None # expected walk_length for check_steps
mode = 1 # specify whether to copy leaf directories with walk_len pngs or steps in each walk directory 0 - pngs, 1 - steps


def initialise():

    """Create the new directory to store merged leaf directories"""

    if not os.path.exists(merged_dir):
        os.mkdir(merged_dir)
    else:
        print(f"Error: Directory {merged_dir} already exists.")
        sys.exit(1)


def walk_merge():

    """
    Iterates through separate leaffinder runs in a root directory and copies all leaf directories 
    that contain walk_len .pngs or steps in each walk directory to a new run_dir
    """

    for run_dir in os.listdir(root):  # loop through runs
        run_path = os.path.join(root, run_dir)
        if os.path.isdir(run_path):
            for leaf_dir in os.listdir(run_path): # loop through leaves
                leaf_path = os.path.join(run_path, leaf_dir)
                if os.path.isdir(leaf_path):

                    all_present = True # we assume each leaf has a complete set of walks until we find an incomplete walks

                    for walk_dir in os.listdir(leaf_path): # loop through walks
                        walk_path = os.path.join(leaf_path, walk_dir)
                        if os.path.isdir(walk_path):
                            if mode == 0:
                                png_files = [f for f in os.listdir(walk_path) if f.endswith('.png')]
                                if len(png_files) < walk_len: # stop looping through walks if fewer than 80 .pngs present
                                    print(f"No. pngs < {walk_len} in {walk_path}")
                                    all_present = False
                                    break
                            if mode == 1:
                                # get step numbers from .png file names
                                steps = set([int(file.split('_')[-2]) for file in os.listdir(walk_path) if file.endswith('.png')])
                                if walk_len - 1 not in steps: # stop looping through walks if last step is not present in .png filenames
                                    print(f"No. steps < {walk_len} in {walk_path}")
                                    all_present = False
                                    break

                    if all_present:
                        if mode == 0:
                            print(f"All {walk_len} pngs in {leaf_path}")
                        if mode == 1:
                            print(f"All {walk_len} steps in {leaf_path}")
                        dest_path = os.path.join(merged_dir, leaf_dir)
                        shutil.copytree(leaf_path, dest_path)
                        print(f"Copied {leaf_path} to {dest_path}")


def check_complete():
    
    """Check that all leaf directories have been copied to the new directory"""

    all_present = True

    for leafid in leafids:
        leaf_path = os.path.join(merged_dir, leafid)
        if not os.path.exists(leaf_path):
            print(f"Error: {leaf_path} not present in {merged_dir}")
            all_present = False
    
    if all_present:
        print(f"All {len(leafids)} leaf directories have been copied successfully into {merged_dir}")


def check_steps(dir):
    
    """
    Check that there is a png for each step in each walk directory of a multi-run and
    return the run/walk/step for which png is missing
    """

    expected_steps = set(range(0, walk_len))
    all_present = True

    for run_dir in os.listdir(dir): # loop through runs
        run_path = os.path.join(dir, run_dir)
        if os.path.isdir(run_path): 
            for leaf_dir in os.listdir(run_path): # loop through leaves
                leaf_path = os.path.join(run_path, leaf_dir)
                if os.path.isdir(leaf_path):
                    for walk_dir in os.listdir(leaf_path): # loop through walks
                        walk_path = os.path.join(leaf_path, walk_dir)
                        if os.path.isdir(walk_path):
                            steps = set([int(file.split('_')[-2]) for file in os.listdir(walk_path) if file.endswith('.png')])
                            if walk_len - 1 in steps: # if a file with last step is present, return the steps for which pngs are missing
                                missing = expected_steps.symmetric_difference(steps)
                                if missing:
                                    all_present = False
                                    print(f"Missing .png for {walk_path}: {missing}")
    
    if all_present:
        print("All steps are associated with a .png!")
    else:
        print("Some steps are not associated with a .png! This will result in not all leaf directories being copied or missing steps in the final merged dataset.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)                       


def check_empty(dir):

    """Check which directories contain leaf images."""

    no_empty_walks = 0
    grand_total_pngs = 0
    n_walks = len(os.listdir(f"{dir}/{leafids[0]}")) # get no. walks in a leaf dir
    exp_grand_total_pngs = walk_len * len(leafids) * n_walks
    incomplete_leafids = []
    complete_leafids = []
    for leaf_dir in os.listdir(dir):
        complete = True # assume each leaf directory is complete until we find an empty directory
        leaf_path = os.path.join(dir, leaf_dir)
        if os.path.isdir(leaf_path):
            for walk_dir in os.listdir(leaf_path):
                walk_path = os.path.join(leaf_path, walk_dir)
                if os.path.isdir(walk_path):
                    steps = set([int(file.split('_')[-2]) for file in os.listdir(walk_path) if file.endswith('.png')])
                    total_pngs = len(steps)
                    grand_total_pngs += total_pngs
                    if not steps: # check if completely empty
                        # print(f"{walk_path}")
                        complete = False
                        no_empty_walks += 1
                    elif walk_len - 1 not in steps: # check if partially empty
                        # print(f"Incomplete directory: {walk_path}")
                        print(f"{walk_path} {total_pngs} #") # print partially full walk directories
                        complete = False
                    else:
                        print(f"{walk_path} {total_pngs} ###") # print full walk directories
            if complete:
                complete_leafids.append(leaf_dir)
            else:
                incomplete_leafids.append(leaf_dir)
    
    print(f"Total pngs found: {grand_total_pngs} of {exp_grand_total_pngs} expected ({grand_total_pngs/exp_grand_total_pngs*100:.2f}%)")
    print(f"Complete leaf directories found in {dir}: {complete_leafids}")
    print(f"Empty walks found in {dir}: {incomplete_leafids}")
    print(f"Total incomplete leaf directories: {len(incomplete_leafids)} of {len(leafids)} expected ({len(incomplete_leafids)/len(leafids)*100:.2f}%)")
    print(f"Total empty walks: {no_empty_walks} of {len(leafids) * n_walks} expected ({no_empty_walks/(len(leafids) * n_walks)*100:.2f}%)")
    
def print_help():
    help_message = """
    Usage: python3 walk_merge.py [options]

    Options:
        -h              Show this help message and exit.
        -id [string]    Specify the root directory containing separate 
                        leaffinder runs, or separate walks if using -f 1.
        -wl [int]       Specify the expected walk length to check if walks are 
                        complete.
        -f  [function]  Pass function you want to perform:
                        0   ...merge leaf directories
                        1   ...print empty walk directories
    """
    print(help_message)

if __name__ == "__main__":
    args = sys.argv[1:]
    if "-h" in args:
        print_help()
    else:
        if "-id" in args:
            root = str(args[args.index("-id") + 1])
        else:
            print(f"WARNING: No run_id specified, defaulting to {root}")
        if "-wl" in args:
            walk_len = int(args[args.index("-wl") + 1])
        if "-f" in args:
            func = int(args[args.index("-f") + 1])
            if func == 0:
                print("Merge parameters:")
                print(f"root = {root}")
                print(f"merged_dir = {merged_dir}")
                print(f"walk_len = {walk_len}")
                print(f"mode = {mode}")
                check_steps(root)
                initialise()
                walk_merge()
                check_complete()
            if func == 1:
                print("Checking for empty walk directories in:")
                print(f"root = {root}")
                print(f"walk_len = {walk_len}")
                check_empty(root)
            