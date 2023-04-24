
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


#X,y,files = vv.getXYFiles(path = '/home/stefan/WEINBERG/',limit = 300, discrete = True); ut.dumpfile((X,y,files), '300delme.dmp')
(X,y,files)  = ut.loadfile('300delme.dmp')
print(f"vectorized")


# dist, inst = nn.nearneigh(X,k = 100); ut.dumpfile((dist, inst), '300nndelme.dmp')
(dist, inst) = ut.loadfile('300nndelme.dmp')
print(f"got neighs")

nn.plot_NN(dist)
nn.plotbythresh(dist,inst,files, thresh = .5, max = 2)
nn.filter_thresh(dist,inst,thresh = .7)
nn.plot_NN(dist)


