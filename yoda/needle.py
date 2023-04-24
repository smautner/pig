


import numpy as np


match = lambda x,y: 1 if x==y else -1

def needle(a,b, gap = -1, edgegap= -.1):

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




# if ca ==0:
#     rb = ''.join(b)[::-1]
#     retb+= rb
#     reta+= '-'*len(rb)
#     break
# if cb ==0:
#     ra = ''.join(a)[::-1]
#     reta+= ra
#     retb+= '-'*len(ra)
#     break
