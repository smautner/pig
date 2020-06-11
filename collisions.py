import os
import other.help_functions as h
from collections import Counter, defaultdict
from itertools import combinations


def load(path):
    if os.path.isfile("seq_dict.json"):
        print("Loading saved seq_dict.json")
        seqdict, filedict = h.loadfile("seq_dict.json")
    else:
        p1 = [ "%s/pos/%s" %(path,f) for f in  os.listdir("%s/pos" % path )]
        p2 = [ "%s/pos2/%s" %(path,f) for f in  os.listdir("%s/pos2" % path )]
        n = [ "%s/neg/%s" %(path,f) for f in  os.listdir("%s/neg" % path )]
        seqdict = defaultdict(list)
        filedict = defaultdict(int)
        for i in p1+p2+n:
            with open(i) as f:
                for line in f.readlines()[2:]: # Skip first 2 lines of files
                    if line.startswith("#"):
                        break # Last Sequence of File has been read
                    else:
                        split = line.split(sep=" ", maxsplit=1)[0].split(sep="/")
                        seqdict[split[0]].append((split[1], i))
                        filedict[i] += 1
            
        h.dumpfile((seqdict, filedict), "seq_dict.json") # Saves only around 8 seconds per execution
                
    return seqdict, filedict

def find_collisions(seqdict, filedict):
    cou = 0
    col = 0
    l = []
    for key in seqdict:
        for a, b in combinations(seqdict[key], 2):
            cou += 1
            if compare(a, b):
                if a[0] == b[0]:
                    if a[1] < b[1]:
                        l.append((a[1], b[1]))
                    else:
                        l.append((b[1], a[1]))
                    col += 1
    print (f"Total comparisons: {cou}, Collisions: {col}")
    allcol = Counter(l).most_common()
    print(f"Different file collisions: {len(allcol)}")
    subsetcount = 0
    for x in allcol:
        a = x[0][0]
        b = x[0][1]
        subset_a = a.split(sep="-")
        subset_b = b.split(sep="-")
        if subset_a[:2] == subset_b[:2]:
            subsetcount += 1
            #print(((a, filedict[a]), (b, filedict[b]), x[1]))
            #return
    print(f"Subsets: {subsetcount}")
    #print(Counter(l).most_common(5))
    return l

def compare(a, b):
    a_1, a_2 = a[0].split("-")
    b_1, b_2 = b[0].split("-")
    if a_1 < a_2:
        a_start, a_end = a_1, a_2
    else:
        a_start, a_end = a_2, a_1
    if b_1 < b_2:
        b_start, b_end = b_1, b_2
    else:
        b_start, b_end = b_2, b_1
    if ((a_start <= b_start <= a_end) or
        (a_start <= b_end <= a_end) or
        (b_start <= a_start <= b_end) or
        (b_start <= a_end <= b_end)):
        return True


if __name__ == "__main__":
    path = "data"
    d, f = load(path)
    l = find_collisions(d, f)
