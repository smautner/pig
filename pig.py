from loadfiles import loaddata
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


debug=True
p,n = loaddata("/home/pig/data",numneg=3000 if not debug else 200, pos='1' if debug else 'both', seed = 9)
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



def getfeaturelist(): 
    reli=relief()
    reli.fit(X,y)

    def relief(X,y,param):
        #https://github.com/EpistasisLab/scikit-rebate
        return [ df.columns[top] for top in reli.top_features_[:param]]



    selectors = [lambda x,y: lasso(X,y,alpha=.05),  
                 lambda x,y: lasso(X,y,alpha=.01),
                 lambda x,y: relief(X,y,40),
                 lambda x,y: relief(X,y,60),
                 lambda x,y: relief(X,y,80)
                                                   ]
    #featurelists =  [ selector(X,y) for selector in selectors]

    featurelist = b.mpmap_prog( lambda x: x(X,y), selectors,chunksize=1,poolsize=5)  
    featurelists.append(df.columns)

    ba.dumpfile(featurelist, "ftlist")
    

getfeaturelist()






'''

for FEATURELIST in featurelists:  # loop over all the selectors 
    
    # make some data 
    X,y,df = makeXY(FEATURELIST)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randseed) # USE THE SAME SEED AS BELOW! 

    def score(clf,param):
        searcher = RSCV(clf, 
                    param, 
                    n_iter=50 if not debug else 5, 
                    scoring=None,
                    n_jobs=4,
                    iid=False,
                    #fefit=True,
                    cv=4,
                    verbose=0,
                    pre_dispatch="2*n_jobs",
                    random_state=None,
                    error_score=np.nan,
                    return_train_score=False)
        searcher.fit(X_train, y_train)
        
        print(searcher.best_params_)
        return scorer(searcher.best_estimator_,X_test,np.array(y_test))
        
    res.append( [score(clf,param) for clf,param in zip(rs.classifiers,rs.param_lists)] )
'''
