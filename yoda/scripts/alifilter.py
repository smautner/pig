
'''
read the alignments and dedensify so that near duplicates disappear :)
'''

import yoda.vectorizer as vv
import yoda.nearneigh as nn
import numpy as np
import ubergauss.tools as ut

'''
what needs to happen?
use vectorizer to load the data use nearneigh to sort stuff out
'''

limit = 0
X,y,files = vv.getXYFiles(path = '/home/stefan/WEINBERG/',limit = limit, discrete = True);# ut.dumpfile((X,y,files), f'{limit}delme.dmp')
# (X,y,files)  = ut.loadfile(f'{limit}delme.dmp')
print(f"vectorized")


dist, inst = nn.nearneigh(X,k = 400);#  ut.dumpfile((dist, inst), f'{limit}nndelme.dmp')
# (dist, inst) = ut.loadfile(f'{limit}nndelme.dmp')
print(f"got neighs")

nn.plot_NN(dist)
#nn.plotbythresh(dist,inst,files, thresh = .7, max = 2)
print('is k large enough?', sum(dist[:,-1] < .7 ))  # zero is good
# rmlist = nn.filter_thresh(dist,inst,thresh = .7)
rmlist = nn.filter_thresh(dist,inst,thresh = .7)


# confirm that we removed all the trash
dist, inst = nn.nearneigh([x for k,x in enumerate(X) if k not in rmlist],k = 2)
nn.plot_NN(dist)


nn.filter_dump(rmlist,files,out='okfiles.json')
print(f"dumped list")



