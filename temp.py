import numpy as np
def find_last(arr,col):
    nzc = np.count_nonzero(arr!=col,axis=1)
    print(nzc)
    if nzc[0] ==0:
        last_idx =-1
    else:
        last_idx = arr.shape[0]-nzc[0]
    return last_idx


arr = np.array([[1,3],
                [4,6],
                [0,3],
                [0,0],
                [0,1],])



