
'''
read the alignments and dedensify so that near duplicates disappear :)
'''

import yoda.filein as vv
import yoda.nearneigh as nn
import numpy as np
import ubergauss.tools as ut
import time

'''
INIT and vectorize
'''
start = time.time()
limit = 0
# X,y,files = vv.getXYFiles(path = '/home/stefan/WEINBERG/',limit = limit, discrete = True); ut.dumpfile((X,y,files), f'{limit}delme.dmp')
(X,y,files)  = ut.loadfile(f'{limit}delme.dmp')
print(f"vectorized   used so far:{(time.time()-start)/60}")


'''
get kneighbors
'''
# dist, inst = nn.neighbors(X,k = 400);  ut.dumpfile((dist, inst), f'{limit}nndelme.dmp')
(dist, inst) = ut.loadfile(f'{limit}nndelme.dmp')
print(f"we have neighbors now   used so far:{(time.time()-start)/60}")
nn.plot_NN(dist)



'''
inspect neighbors? all ok?
'''
nn.plotbythresh(dist,inst,files, thresh = .6, max = 2)
nn.plotbythresh(dist,inst,files, thresh = .65, max = 2)
nn.plotbythresh(dist,inst,files, thresh = .7, max = 2)
nn.plotbythresh(dist,inst,files, thresh = .75, max = 2)
print('is k large enough?', sum(dist[:,-1] < .7 ))  # zero is good

exit()

'''
filter
'''
rmlist = nn.filter_thresh(dist,inst,thresh = .7, checkmax = 400)
print(f"filtered  used so far:{(time.time()-start)/60}")

'''
checking the filter:
'''
dist2, inst2 = nn.neighbors([x for k,x in enumerate(X) if k not in rmlist],k = 2)
nn.plot_NN(dist2)
print(f"confirmationNNdone  used so far:{(time.time()-start)/60}")


'''
dump
'''
nn.filter_dump(rmlist,files,out='okfiles.json')
print(f"dumped list {(time.time() - start)/60}")



