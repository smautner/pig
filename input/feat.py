import numpy as np
from collections import defaultdict
import traceback as tb 



####
# CONSERVATION
####
def cons_targets(pos,ali):
    r=[]
    nu,le = ali.ali.shape
    for i in pos: 
        a = ali.ali[:,i]
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
        print (ali.ali)
        print (ali.fname)
        print (pos)
        tb.print_stack()
    return np.array(r) 


def cons_targets_nuc(pos, ali):
    r=[]
    nu,le = ali.ali.shape
    for i in pos: 
        a = ali.ali[:,i]
        nucmax = max ( (a=='G').sum(), (a=='A').sum() ,(a=='C').sum(),(a=='U').sum() )
        delnum = (a=='.').sum()

        if delnum>nucmax:
            r.append(0)
        else:
            r.append(nucmax/nu)
            
    if np.isnan(np.mean(r)):
        print ('cons_targets nuc has a problem')
        print (ali.ali)
        print (pos)
        tb.print_stack()
    return np.array(r) 



def conservation(ali):
    nu,le = ali.ali.shape
    r = cons_targets(range(le),ali)
    r2 = cons_targets_nuc(range(le),ali)
    ali.cons = r 
    ali.cons_nuc = r2
    return  {
            f"total_conservation":np.mean(r), 
            f"total_conservation_+cov":np.mean([ 1 if b == '2' else a for a,b in zip(r,ali.covariance)  ]),
            f"total_conservation_nuc":np.mean(r2), 
            f"total_conservation_nuc_+cov":np.mean([ 1 if b == '2' else a  for a,b in zip(r2,ali.covariance)  ]) 
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
        
    return { f"number_of_{op[n]}_blocks":res.get(d[op[n]],0) for n in range(len(op)) }

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
    return {f"stem_length_smallest":s[0],
            f"stem_length_last-1":s[-2],
            f"stem_length_max":s[-1]}

###
# stem and flank conservation 
# percstem
# alignment lengthh  and count
# stemcov
#####
def stemflankconservation(ali):
    #scons, sconsn = cons_stem(ali.blockstartend,ali.ali)
    return {
            f"stem_cons": np.mean(ali.cons[ali.blockmask]) if len(ali.blockmask)> 0 else 0,
            f"stem_cons_nuc": np.mean(ali.cons_nuc[ali.blockmask]) if len(ali.blockmask)> 0 else 0,
            f"flank_cons": np.mean(ali.cons[ali.flankmask]) if len(ali.flankmask)> 0 else 0,
            f"flank_cons_nuc": np.mean(ali.cons_nuc[ali.flankmask]) if len(ali.flankmask)> 0 else 0
           }

def percstem(ali):
    length = len(ali.structure)
    allstacks = sum([ b-a+1 for a,b in ali.blockstartend ])
    return  {f"perc_stem":allstacks/length}


def lencount(ali):
    return {
        f"length":ali.ali.shape[1],
        f"count":ali.ali.shape[0]
    }

def stemcov(ali):
    return {f"stem_covariance": sum([a=='2' for a in ali.covariance])/len(ali.blockmask) if len(ali.blockmask)>0 else 0 }
 
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
            f"sloppy_gu":(sloppy/(sloppy+ok+sloppy_all)) if sloppy+ok+sloppy_all > 0 else 0, 
            f"sloppy_all":((sloppy_all+sloppy)/(sloppy+ok+sloppy_all) ) if sloppy+ok+sloppy_all > 0 else 0, 
        }
    except Exception as ex: 
        print ("getsloppy fehlt",ex,ali.ali, ali.structure)
        
        #print ("getsloppy: p con.both ali.name, ali.blocks", p, ali.basepairs.both, ali.name,ali.blockstartend )

       
#####
# Big mysterious hairpin
######

def bmm(ali):
    open_brackets = {"<", "(", "[", "{"}
    close_brackets = {">", ")", "]", "}"}
    counter = 0
    stem_counter = 0
    state = "no structure"
    for x in ali.structure:
        counter += 1
        if state == "no structure":
            if x in open_brackets:
                stem_counter += 1
                state = "brackets open"
        elif state == "brackets open":
            if x in close_brackets:
                state = "brackets closing"
            elif x in open_brackets:
                stem_counter += 1
        elif state == "brackets closing":
            if x in open_brackets:
                return {"big_mysterious_hairpin": 0} # Cannot be BMM
    #print(ali.structure)
    percentage = (stem_counter*2)/counter
    if percentage >= 0.7:
        return {"big_mysterious_hairpin": 1} # BMM found
    else:
        return {"big_mysterious_hairpin": 0} # Hairpin to small
             


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
             percstem, lencount, stemcov, get_sloppy, bmm]:
        d.update(f(ali))
    return d