import os
import other.help_functions as h
from collections import defaultdict
from itertools import combinations


def load(path):
    if os.path.isfile("seq_dict.json"):
        print("Loading saved seq_dict.json")
        dictionary = h.loadfile("seq_dict.json")
    else:
        p1 = [ "%s/pos/%s" %(path,f) for f in  os.listdir("%s/pos" % path )]
        p2 = [ "%s/pos2/%s" %(path,f) for f in  os.listdir("%s/pos2" % path )]
        n = [ "%s/neg/%s" %(path,f) for f in  os.listdir("%s/neg" % path )]
        dictionary = defaultdict(list)
        for i in p1+p2+n:
            with open(i) as f:
                for line in f.readlines()[2:]: # Skip first 2 lines of files
                    if line.startswith("#"):
                        break # Last Sequence of File has been read
                    else:
                        split = line.split(sep=" ", maxsplit=1)[0].split(sep="/")
                        dictionary[split[0]].append((split[1], i))
            
        h.dumpfile(dictionary, "seq_dict.json") # Saves only around 8 seconds per execution
                
    return dictionary

def find_collisions(dictionary):
    c = 0
    col = 0
    for key in dictionary:
        l = dictionary[key]
        for a, b in combinations(l, 2):
            c += 1
            if compare(a, b):
                if a[0] == b[0]:
                    #print(f"Collision between {a[1]} and {b[1]} in sequence {key}: {a[0]}, {b[0]}")
                    #return
                    col += 1
    print (f"Total comparisons: {c}, Collisions: {col}")
    return col, c

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
    d = load(path)
    find_collisions(d)
