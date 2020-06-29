import os
import other.help_functions as h
from collections import Counter, defaultdict
from itertools import combinations
from pprint import pprint


def compare(a, b):
    """Compares 2 sequences and checks if their coordinates overlap.
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
    if os.path.isfile("tmp/seq_dict.json"):
        print("Loading saved seq_dict.json")
        seqdict, filedict = h.loadfile("tmp/seq_dict.json")
    else:
        p1 = [ "%s/pos/%s" %(path,f) for f in  os.listdir("%s/pos" % path )]
        p2 = [ "%s/pos2/%s" %(path,f) for f in  os.listdir("%s/pos2" % path )]
        n = [ "%s/neg/%s" %(path,f) for f in  os.listdir("%s/neg" % path )]
        seqdict = defaultdict(list)
        filedict = defaultdict(int)
        for i in p1+p2+n:
            i_name = i.split("/")[2]
            with open(i) as f:
                for line in f.readlines()[2:]: # Skip first 2 lines of files
                    if line.startswith("#"):
                        break # Last Sequence of File has been read
                    else:
                        split = line.split(sep=" ", maxsplit=1)[0].split(sep="/")
                        seqdict[split[0]].append((split[1], i_name))
                        coord = split[1].split("-")
                        distance = abs(int(coord[0])-int(coord[1]))
                        filedict[i_name] += distance
            
        h.dumpfile((seqdict, filedict), "tmp/seq_dict.json")
                
    return seqdict, filedict


def find_collisions(seqdict, filedict):
    cou = 0
    col = 0
    l = []
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
    print (f"Total comparisons: {cou}, Collisions: {col}")
    allcol = Counter(l).most_common()
    results = []
    for x in allcol:
        a = x[0][0]
        b = x[0][1]
        a_name = a.split("-")
        b_name = b.split("-")
        if not a_name[:2] == b_name[:2]:
            if x[1] < 2:  # If the first 2 parts of the filename arent identical,
                continue  # Require at least 2 collisions to be added
        results.append(((a, filedict[a]), (b, filedict[b]), x[1]))
    print(f"Different collisions: {len(results)}")
    return results

def create_blacklist(l):
    blacklist = set()
    for x in l:
        if x[0][1] > x[1][1]:
            blacklist.add(x[1][0])
        else:
            blacklist.add(x[0][0])
    blacklist.add("416-60776-0-1.sto") # Incompatible with RNAz
    h.dumpfile(list(blacklist), "tmp/blacklist.json")
    print(f"{len(blacklist)} blacklisted files")

def show(a, b=None, path="data/neg", open_files=True):
    import subprocess
    if type(a) == tuple:
        b = a[1][0]
        a = a[0][0]
    with open(f"{path}/{a}") as af:
        for aline in af.readlines()[2:]:
            if aline.startswith("#"): # Last Sequence of a reached
                break
            asplit = aline.split(sep=" ", maxsplit=1)[0].split(sep="/")
            with open(f"{path}/{b}") as bf:
                for bline in bf.readlines()[2:]:
                    if bline.startswith("#"): # Last Sequence of b reached
                        break
                    bsplit = bline.split(sep=" ", maxsplit=1)[0].split(sep="/")
                    if asplit[0] == bsplit[0]:
                        #print(asplit, bsplit)
                        if compare(asplit[1], bsplit[1]):
                            print(f"{asplit[0]}: {asplit[1]}  {bsplit[1]}")

    if open_files and os.name == "nt": # Only for testing to make reading easier
        print(f"opening {path}/{a} and {path}/{b}")
        subprocess.Popen(["notepad.exe", f"{path}/{a}"])
        subprocess.Popen(["notepad.exe", f"{path}/{b}"])


if __name__ == "__main__":
    path = "data"
    d, f = load(path)
    col = find_collisions(d, f)
    create_blacklist(col)