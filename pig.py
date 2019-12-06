import sys
from sklearn.metrics import  make_scorer
from loadfiles import loaddata
import dill
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from collections import defaultdict
import pandas as pd
import copy
from sklearn.feature_selection import RFECV as rec
import pandas as pd
from sklearn.linear_model import Lasso
from skrebate import ReliefF as relief
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV as RSCV
import randomsearch as  rs
import basics as b
import time 

def clean(di,oklist):
    for k in list(di.keys()):
        if k not in oklist:
            di.pop(k)
    return di

def makeXY(featurelist):
    asd = [ clean(e,featurelist) for e in copy.deepcopy(p+n) ]
    df = pd.DataFrame(asd)
    X= df.to_numpy()
    y= [1]*len(p)+[0]*len(n)
    return X,y,df


debug=False

try:
    p,n = b.loadfile("pn")
except: 
    p,n = loaddata("/scratch/bi01/mautner/GRAPHLEARN/data",numneg=3000 if not debug else 200, pos='1' if debug else 'both', seed = 9)
    b.dumpfile((p,n), 'pn') 


allfeatures = list(p[1].keys()) # the filenames are the last one and we dont need that (for now)
allfeatures.remove("name")
X,y,df = makeXY(allfeatures)



############
#
#############

X = StandardScaler().fit_transform(X)
randseed = 41
testsize=.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randseed) # USE THE SAME SEED AS BELOW! 

def myscore(y,yh):
    
    y1 = np.count_nonzero(y==1)
    y0 = np.count_nonzero(y==0)
    tpr = sum([ i==j for i,j in zip(y,yh) if i==1  ])/np.count_nonzero(y==1) if y1 else 0
    tnr = sum([ i==j for i,j in zip(y,yh) if i==0  ])/np.count_nonzero(y==0) if y0 else 0
    return ((2*tnr)+tpr)/3
    
def scorer(esti,X,y):
    yh = esti.predict(X)
    return myscore(y,yh)

def lasso(X,y,alpha=.06):
    mod = Lasso(alpha=alpha)
    mod.fit(X_train,y_train)
    return [b for a,b in zip(mod.coef_, df.columns) if a!=0]

def runner(stuff):
    X,y,s = stuff
    f= dill.loads(s)
    return f(X,y)

def getfeaturelist(): 

    def myrelief(X,y,param,reli):
        #https://github.com/EpistasisLab/scikit-rebate
        return [ df.columns[top] for top in reli.top_features_[:param]]

    reli=relief(n_jobs=-1, n_neighbors =50) # 10 is clearly worse than 100
    reli2=relief(n_jobs=-1, n_neighbors =100)
    reli.fit(X,y)
    reli2.fit(X,y)


    selectors = [lambda x,y: lasso(X,y,alpha=.004),  
                 lambda x,y: lasso(X,y,alpha=.003),
                 lambda x,y: lasso(X,y,alpha=.005),
                 lambda x,y: lasso(X,y,alpha=.006),
                 lambda x,y: lasso(X,y,alpha=.007),
                 lambda x,y: lasso(X,y,alpha=.008),
                 lambda x,y: lasso(X,y,alpha=.009),
                 lambda x,y: myrelief(X,y,40,reli),
                 lambda x,y: myrelief(X,y,50,reli),
                 lambda x,y: myrelief(X,y,60,reli),
                 lambda x,y: myrelief(X,y,70,reli),
                 lambda x,y: myrelief(X,y,80,reli),
                 lambda x,y: myrelief(X,y,40,reli2),
                 lambda x,y: myrelief(X,y,50,reli2),
                 lambda x,y: myrelief(X,y,60,reli2),
                 lambda x,y: myrelief(X,y,70,reli2),
                 lambda x,y: myrelief(X,y,80,reli2)]
    #featurelists =  [ selector(X,y) for selector in selectors]

    featurelists = b.mpmap_prog( runner,  [( X,y,dill.dumps(s) )for s in selectors],chunksize=1,poolsize=5)  
    featurelists.append(df.columns)

    b.dumpfile(featurelists, "ftlist")
    

def gettasks():
    ftlist  = b.loadfile("ftlist")
    tasks = [(clf,param,ftli)for clf,param in zip(rs.classifiers,rs.param_lists) for ftli in ftlist]
    return tasks

def gettasks_annotated():
    ftlist  = b.loadfile("ftlist")
    tasks = [(clf,param,ftli, clfname,ftid)for clf,param,clfname in zip(rs.classifiers,rs.param_lists, rs.clfnames) for ftid,ftli in enumerate(ftlist)] 
    return tasks

def doajob(idd):
    tasks = gettasks()
    clf, param,FEATURELIST= tasks[idd]


    X,y,df = makeXY(FEATURELIST)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randseed) # USE THE SAME SEED AS BELOW! 

    def score(clf,param):
        searcher = RSCV(clf, 
                    param, 
                    n_iter=200 if not debug else 5, 
                    scoring='f1', # using my own doesnt work.. why?
                    n_jobs=24,
                    iid=False,
                    #fefit=True,
                    cv=5,
                    verbose=0,
                    pre_dispatch="2*n_jobs",
                    random_state=None,
                    refit=True, # enables best_score_
                    error_score=np.nan,
                    return_train_score=False)
        searcher.fit(X_train, y_train)
        return searcher 
    start=time.time()
    searcher = score(clf,param)
        
    #b.dumpfile( (searcher.best_params_, scorer(searcher.best_estimator_,X_test,np.array(y_test))) , "res%d" % idd)
    print( searcher.best_params_,searcher.best_score_, scorer(searcher.best_estimator_,X_test,np.array(y_test)) , idd, time.time()-start)
        

def readresult(fname):
    a= open(fname,"r").read()
    dic,numerics = a.split("}")
    params = eval(dic+"}")
    f1,perf,jid,time = [float(a) for a in numerics.split()]
    return f1,perf, dic , jid , time
    


if __name__ == "__main__":
    if sys.argv[1] == 'report':
        print("reporting")
        arrayjobid = sys.argv[2]
        jobstr= f"/home/mautner/JOBZ/pig_o/{arrayjobid}.o_%d"
        tasks = gettasks_annotated()
        tasksnum = len(tasks)
        print("number of tasks:", tasksnum)
        files = [jobstr %i  for i in range(1,tasksnum+1)]
        allresults = [readresult(f) for f in files]
        
        res = defaultdict(dict)
        for task, result in zip(tasks,allresults):
            res[task[-2]][task[-1]]= ("%.4f" % result[0], "%.1f"% result[-1])

        print(pd.DataFrame(res))



        res= [ (result[0],task[-2],result[2]) for task, result in zip(tasks,allresults)] 
        res.sort(reverse=True)
        for a,b,c in res[:3]:
            print (a,b,c)

    
    elif sys.argv[1] == 'maxtask':
        print(len( gettasks_annotated() )) # for use in qsub.. 

    elif sys.argv[1] == 'showft':
        features = b.loadfile("ftlist")
        ft= list(features[int(sys.argv[2])])
        import pprint
        pprint.pprint(ft)
        print("numft:",len(ft))
    elif sys.argv[1] == 'makeftlist':
        getfeaturelist()
    else:
        idd = int(sys.argv[1])-1
        doajob(idd)



