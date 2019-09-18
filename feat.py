
def conservation(cov, ali):
    # account for cov or not
    covcnt =  cov.count("2")
    nu,le = ali.shape
    r= []
    for i in range(le): 
        a = ali[:,i]
        
        R = (a=='G').sum()+(a=='A').sum()
        Y = (a=='C').sum()+(a=='U').sum()
        d = (a=='.').sum()

        if R > Y:
            Y = R
        if d > R:
            r.append(0)
        else:
            r.append(R/le)
    
    
    return  np.mean(r), np.mean([ 1 if cov == '2' else a  for a,b in zip(r,cov)  ])
        

