import numpy as np
import traceback as tb 

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
    return r 

def conservation(ali):
    # account for cov or not
    covcnt =  ali.cov.count("2")
    nu,le = ali.ali.shape
    r = cons_targets(range(le),ali.ali)
    #print (ali.cov)
    #print([ 1 if b == '2' else a  for a,b in zip(r,ali.cov) ])
    return  {"total conservation":np.mean(r), "total conservation +cov":np.mean([ 1 if b == '2' else a  for a,b in zip(r,ali.cov)  ]) }

def cons_targets_rmgaps(pos,ali):
    allgaps = lambda x, ali: all(ali[:,x]=='-')
    pos = [x for x in pos if not allgaps(x,ali)]
    if len(pos)<5:
        return [0]
    return cons_targets(pos,ali)

def percstem(ali):
    a = len(ali.stru)
    allstacks = sum([ b-a+1 for a,b in ali.blocks ])
    bigstacks = sum([ b-a+1 for a,b in ali.blocks if b-a+1 > 2])
    return  {"perc stem":allstacks/a,'perc stem filtered':bigstacks/a}

def get_sloppy(pos, ali):
    sloppy=0
    ok = 0
    try:
        for p in pos: 
            for a,b in zip( ali.ali[:,p], ali.ali[:,ali.con.both[p]] ):
                z = [a,b] if b > a else [b,a]
                #if z != ["C","G"] and z != ["A",'U']: interestingly this performs worse  
                if z == ["G","U"]:
                    sloppy+=1
                else:
                    ok +=1
        return  sloppy/(sloppy+ok) 
    except: 
        print ("getsloppy: p con.both ali.name, ali.blocks", p, ali.con.both, ali.name,ali.blocks )

def weirdo_detection(ali):
    height, length = ali.shape
    points = [0]*height
    for a in range(length): # for every possition 
        items = ali[:,a]
        dots = (items == '.').sum()
        notdot = height - dots
        if dots > notdot: # dots are the norm 
            for i,e in enumerate(items):
                if e != '.':
                    points[i]+=1
        if notdot > dots: # not dots are the norm 
            for i,e in enumerate(items):
                if e == '.':  # if you have a dot you get a minus point
                    points[i]+=1
        else: # same number
            pass
    return np.argsort(points)
            

def checov(ali, pos, con, cov):
    # chhecking covariance manually, this is necessary after we remove the worst hit line 
    covn = cov.count("2")*2
    if covn == 0: return 0 
    loss = 0
    for p in pos:
        if cov[p]=='2':
            A=-1
            B=-1 
            try:
                for a,b in zip(ali[:,p], ali[:,con.both[p]]):
                    if a != '.' and b!='.':
                        if A==-1:
                            A=a
                            B=b
                        if a!=A and b!=B: # when we find a single alternative, all is good
                            break
                else:
                    loss+=1  # all are the sam
            except: 
                print ("cov check: ali p con.both", ali, p, con.both )
                
    return loss/covn

        

def get_lennogap(ali):
    allgaps = lambda x, ali: all(ali[:,x]=='-')
    pos = [x for x in range(ali.shape[1]) if not allgaps(x,ali)]
    r=  len(pos)
    return r
    
def cov_sloppycov_disturbance_instem(ali):
    # % covariance in stems in stems   # with or without shorts removed
    # sloppy
    # # optionally remove fail line 

    # get indices of stems (all and long)
    hbonds = [ range(a,b+1) for a,b in ali.blocks if b-a+1>2]
    hbonds = [a for b in hbonds for a in b]
    hbonds2 = [ range(a,b+1) for a,b in ali.blocks]
    hbonds2 = [a for b in hbonds2 for a in b]
    
    pcov_large = sum([1 for a in hbonds if ali.cov[a]=='2'])/len(hbonds) if hbonds else 0  # TODO
    pvoc_all   = sum([1 for a in hbonds2 if ali.cov[a]=='2'])/len(hbonds2)
    
    # now we get the sloppy covariance 
    sl_large = get_sloppy(hbonds, ali) if hbonds else 0  # TODO 
    sl_all = get_sloppy(hbonds2, ali)

    # now we filter the most horrible line and do the stuff up there again. 
    weird = weirdo_detection(ali.ali)
    
    
    aliX = ali.ali[[ a for a  in range(ali.ali.shape[0]) if a not in weird[-max(1,int(ali.ali.shape[0]/5)):]]] # hack off 20% 
    #print ("shapecheck:", ali.ali.shape, aliX.shape, -max(1,int(ali.ali.shape[0]/5)) )
    ali1 = ali.ali[[ a for a  in range(ali.ali.shape[0]) if a not in weird[-1:]]]
    ali2 = ali.ali[[ a for a  in range(ali.ali.shape[0]) if a not in weird[-2:]]]
    
    # ali.stems[stem][surround]
    # 1. cons after cleaning(aliX+ignore all gaps)
    # 2. cons in the flanks (in aliX and normal)
    
    
    
    stems = set()
    for _,surset in ali.stems:
        stems = stems.union(surset)
        
    lennogap = get_lennogap(aliX)  
    lennogap1 = get_lennogap(ali1)  
    lennogap2 = get_lennogap(ali2)  
    return {'ali len': ali.ali.shape[1],
            'ali count':ali.ali.shape[0],
            #'ali lennogap':lennogap if aliX.shape[0]>1 else 0 ,
            #'perc stem rmx': lennogap/sum([ b-a+1 for a,b in ali.blocks ]) if ali.ali.shape[0]>1 else 0, # TODO
            #'rm bad X cons': np.mean(cons_targets_rmgaps(hbonds2,aliX)) if ali.ali.shape[0]>1 else 0, #TODO
            #'cons flank': np.mean(cons_targets(stems, ali.ali)) if ali.ali.shape[0]>1 else 0 , #TODO 
            #'rm bad X cons flank': np.mean(cons_targets_rmgaps(stems, aliX))  if ali.ali.shape[0]>1 else 0, #TODO
            'aliX lennogap':lennogap ,
            'ali1 lennogap':lennogap1 ,
            'ali2 lennogap':lennogap2 ,
            #
            'perc stem aliX': sum([ b-a+1 for a,b in ali.blocks ])/lennogap,
            'perc stem ali1': sum([ b-a+1 for a,b in ali.blocks ])/lennogap1,
            'perc stem ali2': sum([ b-a+1 for a,b in ali.blocks ])/lennogap2,
            # SLOPPY
            "sloppy pair filtered":sl_large, 
            "sloppy pair":sl_all,
            # cons 
            'aliX cons': np.mean(cons_targets_rmgaps(hbonds2,aliX)),
            'ali1 cons': np.mean(cons_targets_rmgaps(hbonds2,ali1)),
            'ali2 cons': np.mean(cons_targets_rmgaps(hbonds2,ali2)),
            # FLANKS
            'cons flank': np.mean(cons_targets(stems, ali.ali)),
            'aliX cons flank': np.mean(cons_targets_rmgaps(stems, aliX))  ,
            'ali1 cons flank': np.mean(cons_targets_rmgaps(stems, ali1))  ,
            'ali2 cons flank': np.mean(cons_targets_rmgaps(stems, ali2))  ,
            
            # COVARIANCE
            'stem cov filtered':pcov_large,
            'stem cov':pvoc_all, 
            "aliX cov filtered":checov(aliX,hbonds,ali.con,ali.cov),
            "aliX cov":checov(aliX,hbonds2,ali.con,ali.cov),
            "ali1 cov filtered":checov(ali1,hbonds,ali.con,ali.cov),
            "ali2 cov filtered":checov(ali2,hbonds,ali.con,ali.cov),
            "ali2 cov":checov(ali2,hbonds2,ali.con,ali.cov)}

def stemlength(ali):
    s = [ b-a+1 for a,b in ali.blocks  ]
    s.sort()
    if len(s) ==2:
        s=[0,s[0],s[1]]
    elif len(s) ==1:
        s=[0,0,s[0]]
    elif len(s) == 0:
        s=[0,0,0]
    return {"stem length smallest":s[0],"stem length last-1":s[-2],"stem length max":s[-1]}


def stemconservation(ali):
    stacks2 = [ (a,b) for a,b in ali.blocks if b-a+1 > 2  ]
    return {"stem cons":cons_stem(ali.blocks,ali.ali), 'stem cons filtered':cons_stem(stacks2,ali.ali)}

def cons_stem(stacks,ali):

    targets = set()
    for a,b in stacks:
        for z in range(a-5,a):
            if z > 0:
                targets.add(z)
        for z in range(b+1,b+6):
            if z < ali.shape[1]:
                targets.add(z)
    
    if len(targets)==0:
        r= 0
    else:
        r= np.mean(cons_targets(targets,ali)) 
    #if np.isnan(r):  TODO
    #    return 0
    #else:
    return r


def beststem(ali):
    pass