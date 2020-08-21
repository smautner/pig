import os
import sys
sys.path.append("../")
import input.basics as b
from collections import Counter, defaultdict
from operator import itemgetter
from itertools import combinations


def compare(a, b):
    """Compares 2 sequences and checks if their coordinates overlap.

    Args:
      a (str): coordinates of first sequence
      b (str): coordinates of second sequence

    Returns:
      True if overlap, else False
    """
    a_1, a_2 = a.split("-")
    b_1, b_2 = b.split("-")
    a_1, a_2, b_1, b_2 = int(a_1), int(a_2), int(b_1), int(b_2)
    if ((a_1 > a_2 and b_1 < b_2) or (a_1 < a_2 and b_1 > b_2)):
        # Make sure both sequences go into the same direction..
        return False
    elif a_1 > a_2:
            a_1, a_2 = a_2, a_1
            b_2, b_1 = b_1, b_2
    if ((a_1 <= b_1 <= a_2) or
        (a_1 <= b_2 <= a_2) or
        (b_1 <= a_1 <= b_2) or
        (b_1 <= a_2 <= b_2)):
        return True


def load(path):
    """
    Loads all files in pos, pos2 and neg, reads through them and orders them.

    Args:
      path (str): Path to the pos/neg directories 

    Returns:
      seqdict (dict): A dictionary that contains coordinates for every sequence
      filedict (dict): A dictionary that contains the size
                       (num(seq) * len(seq)) for each file
    """
    if not os.path.exists("tmp"):
        print("Creating tmp directory")
        os.makedirs("tmp")
    p1 = [ f"{path}pos/{f}" for f in  os.listdir(f"{path}pos")]
    p2 = [ f"{path}pos2/{f}" for f in  os.listdir(f"{path}pos2")]
    n = [ f"{path}neg/{f}" for f in  os.listdir(f"{path}neg")]
    seqdict = defaultdict(list)
    filedict = defaultdict(int)
    for i in p1+p2+n:
        i_name = i.split("/")[-1]
        with open(i) as f:
            for line in f.readlines()[2:]: # Skip first 2 lines of files
                if line.startswith("#"):
                    break # Last Sequence of File has been read
                else:
                    split = line.split(sep=" ", maxsplit=1)[0].split(sep="/")
                    seqdict[split[0]].append((split[1], i))
                    coord = split[1].split("-")
                    distance = abs(int(coord[0])-int(coord[1]))
                    filedict[i_name] += distance
                
    return seqdict, filedict


def find_collisions(seqdict, filedict):
    """
    Uses the dictionaries from load() to find and analzye overlaps.

    Returns:
      results (list): A list where each element contains information
                      about 2 conflicting files, like their size and location
    """
    cou = 0
    col = 0
    l = []
    overlap_file_counter = defaultdict(int)
    overlap_sequence_counter = defaultdict(int)
    
    for key in seqdict:
        for a, b in combinations(seqdict[key], 2):
            cou += 1
            if a[1] == b[1]:
                continue # Ignore collisions inside of the same files.
            elif compare(a[0], b[0]):
                if a[1] < b[1]: # Make sure the order is always the same.
                    l.append((a[1], b[1]))
                else:
                    l.append((b[1], a[1]))
                col += 1
                overlap_sequence_counter[key] += 1
    print (f"Total comparisons: {cou}, Overlaps: {col}")
    allcol = Counter(l).most_common()
    results = []
    for x in allcol:
        a = x[0][0]
        b = x[0][1]
        a_loc, a = a.split("/")[-2:] # a/b_loc will be pos/pos2 or neg
        b_loc, b = b.split("/")[-2:] # Meanwhile a and b is the filename
        a_name = a.split("-")
        b_name = b.split("-")
        if not a_name[:2] == b_name[:2]:
            if x[1] < 2:  # If the first 2 parts of the filename arent identical,
                continue  # Require at least 2 collisions to be added
        results.append(((a, filedict[a], a_loc), (b, filedict[b], b_loc), x[1]))
        overlap_file_counter[a] += 1
        overlap_file_counter[b] += 1
        if not a_loc == b_loc:
            print(a_loc, b_loc, a, b)

    # Files and sequences that cause most overlaps 
    top_files = sorted(overlap_file_counter.items(), key=itemgetter(1), reverse=True)
    top_sequences = sorted(overlap_sequence_counter.items(), key=itemgetter(1), reverse=True)
    # If there is files A and B with 10 overlaps this is only counted as 1 "file conflict"
    print(f"Different file conflicts: {len(results)}")
    # Files that are involved in the most different conflicts
    print(f"Top 5 files: {top_files[:5]}")
    # Sequences that are involved in the most total conflicts
    print(f"Top 5 sequences: {top_sequences[:5]}")
    return results


def create_blacklist(path=""):
    """
    Main blacklist function. Creates the actual blacklist and dumps it.

    Args:
      path (str): Path to the pos/neg directories. Also dumps blacklist here
    """
    if path:
        path += "/"
    seqdict, filedict = load(path)
    l = find_collisions(seqdict, filedict)
    blacklist = set()
    d = {"pos":0, "pos2":0, "neg":0}
    for x in l:
        if x[0][1] > x[1][1]:      # This picks which file has a higher "num(seq)*len(seq)"
            blacklist.add(x[1][0]) # and adds the smaller file to the blacklist
            d[x[1][2]] += 1
        else:
            blacklist.add(x[0][0])
            d[x[0][2]] += 1
    blacklist.add("416-60776-0-1.sto") # Incompatible with RNAz
    b.dumpfile(list(blacklist), f"{path}blacklist.json")
    print(f"{len(blacklist)} blacklisted files")
    print(f"{d['pos']} from pos, {d['pos2']} from pos2 and {d['neg']} from neg ")


def show(a, b=None, a_path="neg", b_path="neg", open_files=True):
    """Prints all the different overlaps between 2 given files a and b.
    If run on a Windows system and open_files is set to True this will also
    open given files directly to easily compare them.

    Args:
      a (str): Filename of the first file
      b (str): Filename of the second file
      a_path (str): Path to the directory that contains the first file
      b_path (str): Path to the directory that contains the second file
    """
    import subprocess
    with open(f"{a_path}/{a}") as af:
        for aline in af.readlines()[2:]:
            if aline.startswith("#"): # Last Sequence of a reached
                break
            asplit = aline.split(sep=" ", maxsplit=1)[0].split(sep="/")
            with open(f"{b_path}/{b}") as bf:
                for bline in bf.readlines()[2:]:
                    if bline.startswith("#"): # Last Sequence of b reached
                        break
                    bsplit = bline.split(sep=" ", maxsplit=1)[0].split(sep="/")
                    if asplit[0] == bsplit[0]:
                        #print(asplit, bsplit)
                        if compare(asplit[1], bsplit[1]):
                            print(f"{asplit[0]}: {asplit[1]}  {bsplit[1]}")

    if open_files and os.name == "nt": # Only for testing to make reading easier
        print(f"Opening {path}/{a} and {path}/{b}")
        subprocess.Popen(["notepad.exe", f"{a_path}/{a}"])
        subprocess.Popen(["notepad.exe", f"{b_path}/{b}"])
