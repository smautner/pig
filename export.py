import pandas as pd
import input.loadfiles as lf
import numpy as np

neg,pos =  lf.loaddata('data', blacklist_file = 'fuck')
neg = pd.DataFrame(neg)
pos = pd.DataFrame(pos)
neg.pop('name')
pos.pop('name')
negnp = neg.to_numpy()
posnp = pos.to_numpy()
X = np.vstack([posnp, negnp])
y= np.array([1]*len(posnp)+[0]*len(negnp))
np.savez ("Xy10k.npz", X,y)


