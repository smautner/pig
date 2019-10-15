import numpy as np
from collections import defaultdict
import traceback as tb 



####
# CONSERVATION
####
def cons_targets(pos,ali):
    r=[]
    nu,le = ali.shape
    for i in pos: 
        a = ali[:,i]
        R = (a=='G').sum()+(a=='A').sum()
        Y = (a=='C').sum()+(a=='U').sum()
        d = (a=='.').sum()
        if R < Y:
            R = Y
        if d > R:
            r.append(0)
        else:
            if R/nu <= .5:
                r.append (0)
            else:
                r.append(R/nu)
    if np.isnan(np.mean(r)):
        print ('cons_targets has a problem')
        print (ali)
        print (pos)
        tb.print_stack()
    return np.array(r) 


def cons_targets_nuc(pos, ali):
    r=[]
    nu,le = ali.shape
    for i in pos: 
        a = ali[:,i]
        nucmax = max ( (a=='G').sum(), (a=='A').sum() ,(a=='C').sum(),(a=='U').sum() )
        delnum = (a=='.').sum()

        if delnum>nucmax:
            r.append(0)
        else:
            r.append(nucmax/nu)
            
    if np.isnan(np.mean(r)):
        print ('cons_targets nuc has a problem')
        print (ali)
        print (pos)
        tb.print_stack()
    return np.array(r) 



def conservation(ali):
    nu,le = ali.ali.shape
    r = cons_targets(range(le),ali.ali)
    r2 = cons_targets_nuc(range(le),ali.ali)
    ali.cons = r 
    ali.cons_nuc = r2
    return  {
            f"{ali.name}total conservation":np.mean(r), 
            f"{ali.name}total conservation +cov":np.mean([ 1 if b == '2' else a for a,b in zip(r,ali.covariance)  ]),
            f"{ali.name}total conservation_nuc":np.mean(r2), 
            f"{ali.name}total conservation_nuc +cov":np.mean([ 1 if b == '2' else a  for a,b in zip(r2,ali.covariance)  ]) 
    }




#####
# blocktype
#########
def blocktype(ali):
    '''we want features like: num of <-blocks: 10'''
    op='<([{'
    cl='>)]}'
    d={z:i for i,z in enumerate(op)} 
    d.update({z:i for i,z in enumerate(cl)})
    
    res = defaultdict(int)
    for a,e in ali.blockstartend: 
        n = ali.structure[a]
        res[d[n]]+=1
        
    return { f"{ali.name} number of %s blocks" % op[n]:res.get(d[op[n]],0) for n in range(len(op)) }

####
# stem length 
###
def stemlength(ali):
    s = [ b-a+1 for a,b in ali.blockstartend  ]
    s.sort()
    if len(s) ==2:
        s=[0,s[0],s[1]]
    elif len(s) ==1:
        s=[0,0,s[0]]
    elif len(s) == 0:
        s=[0,0,0]
    return {f"{ali.name} stem length smallest":s[0],
            f"{ali.name} stem length last-1":s[-2],
            f"{ali.name} stem length max":s[-1]}

###
# stem and flank conservation 
# percstem
# alignment lengthh  and count
# stemcov
#####
def stemflankconservation(ali):
    #scons, sconsn = cons_stem(ali.blockstartend,ali.ali)
    return {
            f"{ali.name} stem cons": np.mean(ali.cons[ali.blockmask]),
            f"{ali.name} stem cons nuc": np.mean(ali.cons_nuc[ali.blockmask]),
            f"{ali.name} flank cons": np.mean(ali.cons_nuc[ali.flankmask]),
            f"{ali.name} flank cons nuc": np.mean(ali.cons_nuc[ali.flankmask])
           }

def percstem(ali):
    a = len(ali.structure)
    allstacks = sum([ b-a+1 for a,b in ali.blockstartend ])
    return  {f"{ali.name}perc stem":allstacks/a}


def lencount(ali):
    return {
        f"{ali.name} length":ali.ali.shape[1],
        f"{ali.name} count":ali.ali.shape[0]
    }

def stemcov(ali):
    return {f"{ali.name} stem covariance": sum([a=='2' for a in ali.covariance])/len(ali.blockmask)  }
 
########
# sloppy
###########

def get_sloppy( ali):
    sloppy=0
    sloppy_all = 0
    ok = 0
    try:
        for p,z in ali.basepairs.f.items(): 
            for a,b in zip( ali.ali[:,p], ali.ali[:,z] ):
                z = [a,b] if b > a else [b,a]
                if z == ["G","U"]:
                    sloppy+=1
                elif z != ["C","G"] and z != ["A",'U']: # interestingly this performs worse  
                    sloppy_all +=1
                else:
                    ok +=1
        return  {
            f"{ali.name} sloppy gu":sloppy/(sloppy+ok+sloppy_all), 
            f"{ali.name} sloppy all ":(sloppy_all+sloppy)/(sloppy+ok+sloppy_all)
        }
    except: 
        print ("getsloppy: p con.both ali.name, ali.blocks", p, ali.basepairs.both, ali.name,ali.blockstartend )

       

             


#####
# UNIMPLEMENTED
######
def beststem(ali):
    pass

def yao_score(ali):
    #sqrt((cons_pos+0.2)*avg_bppr/avg_seq_id)*numSpecies*(1+log((double)(digital_msa->nseq)/numSpecies));
    # see mail
    pass

            
            
def getfeatures(ali):
    d={}
    for f in [conservation,blocktype,stemlength,stemflankconservation,
             percstem, lencount, stemcov, get_sloppy]:
        d.update(f(ali))
    return d
            
  