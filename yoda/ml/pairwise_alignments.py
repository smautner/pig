import numpy as np

match = lambda x,y: 1 if x==y else -1





def smith_waterman_score(a,b, gap = -1, edgegap= -.03):

    # make some variables
    la = len(a)
    lb = len(b)
    matrix = np.zeros((la+1,lb+1))

    # # we also need to init oO
    # matrix[0] = np.arange(lb+1) * edgegap
    # matrix[:,0] = np.arange(la+1) * edgegap

    # lets fill the matrix
    for i in range(la):
        for j in range(lb):
            options = (matrix[i,j]+match(a[i], b[j]),0, # diagonal or zero ;)
                       matrix[i,j+1]+(gap if j < lb-1 else edgegap),             # down
                       matrix[i+1,j]+(gap if i < la-1 else edgegap))              # sideways
            matrix[i+1,j+1] = np.max(options)

    return np.max(matrix)

def needle(a,b, gap = -1, edgegap= -.03, get_score_only = False):

    # make some variables
    la = len(a)
    lb = len(b)
    matrix = np.zeros((la+1,lb+1))
    choices = np.zeros((la+1,lb+1))

    # we also need to init oO
    matrix[0] = np.arange(lb+1) * edgegap
    matrix[:,0] = np.arange(la+1) * edgegap
    choices [0] = 2
    choices [:,0] = 1

    # lets fill the matrix
    for i in range(la):
        for j in range(lb):
            options = (matrix[i,j]+match(a[i], b[j]), # diagonal
                       matrix[i,j+1]+(gap if j < lb-1 else edgegap),             # down
                       matrix[i+1,j]+(gap if i < la-1 else edgegap))              # sideways
            matrix[i+1,j+1] = max(options)
            choices[i+1,j+1] = np.argmax(options)

    if get_score_only:
        return matrix[la,lb]

    a = list(a)
    b = list(b)
    ca,cb = la,lb
    reta,retb = '',''
    while True:
        if ca == 0 and cb ==0:
            break

        elif choices[ca,cb] == 0:
            reta+=a.pop()
            retb+=b.pop()
            ca -=1
            cb -=1

        elif choices[ca,cb] == 1:
            reta+=a.pop()
            retb+='-'
            ca -=1

        elif choices[ca,cb] == 2:
            retb+=b.pop()
            reta+='-'
            cb -=1

    return reta[::-1], retb[::-1]









# https://github.com/pereperi/Gotoh-algorithm

def gotoh(seqI, seqJ):

    matrix = lambda x,y: 1 if x==y else -1
    lenI = len(seqI)
    lenJ = len(seqJ)

    gop = -7
    gep = -1

    m = [[0 for x in range(lenJ + 1)] for y in range(lenI + 1)]
    ix = [[0 for x in range(lenJ + 1)] for y in range(lenI + 1)]
    iy = [[0 for x in range(lenJ + 1)] for y in range(lenI + 1)]
    tbm = [[0 for x in range(lenJ + 1)] for y in range(lenI + 1)]
    tbx = [[0 for x in range(lenJ + 1)] for y in range(lenI + 1)]
    tby = [[0 for x in range(lenJ + 1)] for y in range(lenI + 1)]

    for i in range(1, lenI + 1):
        iy[i][0] = gop + (i-1) * gep
        ix[i][0] = float('-inf')
        m[i][0] = float('-inf')
        tbm[i][0] = 1
        tbx[i][0] = 1
        tby[i][0] = 1


    for j in range(1, lenJ + 1):
        ix[0][j] = gop + (j-1) * gep
        iy[0][j] = float('-inf')
        m[0][j] = float('-inf')
        tbm[0][j] = -1
        tbx[0][j] = -1
        tby[0][j] = -1

    for i in range(1, lenI + 1):
        for j in range(1, lenJ + 1):
            s = matrix(seqI[i-1], seqJ[j-1])
            #M
            sub = m[i-1][j-1] + s
            x = ix[i-1][j-1] + s
            y = iy[i-1][j-1] + s
            if sub >= x and sub >= y:
                m[i][j] = sub
                tbm[i][j] = 0
            elif x > y:
                m[i][j] = x
                tbm[i][j] = -1
            else:
                m[i][j] = y
                tbm[i][j] = 1
            #Ix
            sub = m[i][j-1] + gop
            x = ix[i][j-1] + gep
            if sub >= x:
                ix[i][j] = sub
                tbx[i][j] = 0
            else:
                ix[i][j] = x
                tbx[i][j] = -1
            #Iy
            sub = m[i-1][j] + gop
            y = iy[i-1][j] + gep
            if sub >= y:
                iy[i][j] = sub
                tby[i][j] = 0
            else:
                iy[i][j] = y
                tby[i][j] = 1


    i = lenI
    j = lenJ
    alnI = []
    alnJ = []

    if m[i][j] >= ix[i][j] and m[i][j] >= iy[i][j]:
        state = 0
    elif ix[i][j] > iy[i][j]:
        state = -1
    else:
        state = 1

    while i != 0 or j != 0:
        if state == 0:
            state = tbm[i][j]
            i += -1
            j += -1
            alnI.append(seqI[i])
            alnJ.append(seqJ[j])
        elif state == -1:
            state = tbx[i][j]
            j += -1
            alnI.append("-")
            alnJ.append(seqJ[j])
        else:
            state = tby[i][j]
            i += -1
            alnI.append(seqI[i])
            alnJ.append("-")

    seqI_aln = "".join(reversed(alnI))
    seqJ_aln = "".join(reversed(alnJ))
    return seqI_aln, seqJ_aln

