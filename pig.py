import sys
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
randseed = 42
testsize=.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randseed) # USE THE SAME SEED AS BELOW! 

def myscore(y,yh):
    tpr = sum([ i==j for i,j in zip(y,yh) if i==1  ])/np.count_nonzero(y==1)
    tnr = sum([ i==j for i,j in zip(y,yh) if i==0  ])/np.count_nonzero(y==0)
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

    def myrelief(X,y,param):
        #https://github.com/EpistasisLab/scikit-rebate
        return [ df.columns[top] for top in reli.top_features_[:param]]

    reli=relief()
    reli.fit(X,y)


    selectors = [lambda x,y: lasso(X,y,alpha=.05),  
                 lambda x,y: lasso(X,y,alpha=.01),
                 lambda x,y: myrelief(X,y,40),
                 lambda x,y: myrelief(X,y,60),
                 lambda x,y: myrelief(X,y,80)
                                                   ]
    #featurelists =  [ selector(X,y) for selector in selectors]

    featurelists = b.mpmap_prog( runner,  [( X,y,dill.dumps(s) )for s in selectors],chunksize=1,poolsize=5)  
    featurelists.append(df.columns)

    b.dumpfile(featurelists, "ftlist")
    

def doajob(idd):
    ftlist  = b.loadfile("ftlist")
    tasks = [(clf,param,ftli)for clf,param in zip(rs.classifiers,rs.param_lists) for ftli in ftlist]

    clf, param,FEATURELIST= tasks[idd]


    X,y,df = makeXY(FEATURELIST)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randseed) # USE THE SAME SEED AS BELOW! 

    def score(clf,param):
        searcher = RSCV(clf, 
                    param, 
                    n_iter=100 if not debug else 5, 
                    scoring=None,
                    n_jobs=10,
                    iid=False,
                    #fefit=True,
                    cv=5,
                    verbose=0,
                    pre_dispatch="2*n_jobs",
                    random_state=None,
                    error_score=np.nan,
                    return_train_score=False)
        searcher.fit(X_train, y_train)
        return searcher 
    start=time.time()
    searcher = score(clf,param)
        
    #b.dumpfile( (searcher.best_params_, scorer(searcher.best_estimator_,X_test,np.array(y_test))) , "res%d" % idd)
    print( searcher.best_params_, scorer(searcher.best_estimator_,X_test,np.array(y_test)) , idd, time.time()-start)
        

if __name__ == "__main__":
    if sys.argv[1] == 'report':
        pass
    if sys.argv[1] == 'makeftlist':
        getfeaturelist()
    else:
        idd = int(sys.argv[1])-1
        doajob(idd)



