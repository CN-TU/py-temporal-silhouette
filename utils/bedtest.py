import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestCentroid

def merge_labels(l,y1,y2):
    ystay = min(y1,y2)
    yleav = max(y1,y2)
    l[l==yleav] = ystay
    l[l>yleav] -= 1
    return l

def split_cluster(X,y,j):
    Xj = X[y==j,:]
    v = np.median(Xj[:,0])
    ny = np.zeros(len(Xj)).astype(int)
    ny[Xj[:,0]>v]=1
    ny[ny==1]=len(np.unique(y))
    ny[ny==0]=j
    y[y==j] = ny
    return y

def slice_cluster(y,j):
    m = len(y[y==j])
    ny = j * np.ones(m)
    th = np.random.uniform(0.4,0.6) 
    ny[int(m*th):]=len(np.unique(y))
    y[y==j] = ny
    return y

def corrupt_labels(X, y, corr_type, corr_lev, total_corr_levels, outliers):
    corr_rate = corr_lev/(3*total_corr_levels)
    cy = np.copy(y)
    l = X.shape[0]
    k = len(np.unique(y)) # number of classes
    ri = np.random.permutation(l) # random sample indices
    if outliers:
        rk = np.random.permutation(k-1)+1 # random cluster indices
    else:
        rk = np.random.permutation(k) # random cluster indices
    

    if corr_type == 'sp_overclustering':
        for e in range(corr_lev):
            k = len(np.unique(cy)) # number of classes
            _,clcard = np.unique(cy, return_counts=True)
            if outliers:
               t = np.argmax(clcard[1:])+1                
            else:
               t = np.argmax(clcard)                
            cy = split_cluster(X,cy,t)

    elif corr_type == 'sp_underclustering':
        for e in range(corr_lev):
            _,clcard = np.unique(cy, return_counts=True)
            if outliers:
                cl1 = np.argmin(clcard[1:])+1                
            else:
                cl1 = np.argmin(clcard)
            cts = NearestCentroid().fit(X,cy).centroids_
            DM = distance_matrix([cts[cl1]],cts).flatten()
            DM[DM==0] = np.inf
            #DM[np.diag_indices_from(DM)] = np.inf
            kk = np.argwhere(DM == np.min(DM)).flatten()
            cy = merge_labels(cy,kk[0],cl1)
            k = len(np.unique(cy)) # number of classes  
            if k <= 2:
                break          

    elif corr_type == 'slicing_clusters':
        for e in range(corr_lev):
            k = len(np.unique(cy)) # number of classes
            _,clcard = np.unique(cy, return_counts=True)
            if outliers:
                t = np.argmax(clcard[1:])+1                
            else:
                t = np.argmax(clcard)                
            cy = slice_cluster(cy,t)

    else: # 'impurity'
        els2corrupt = int(corr_rate*l)
        for e in range(els2corrupt):
            v = cy[ri[e]] # GT label of element 'e'
            p = np.ones(k)/(k-1)
            p[v] = 0 # prob. of choosing GT label is 0
            cy[ri[e]] = np.random.choice(k, 1, p=p)   
    return cy


def update_cls(x,y):
    A,B = np.copy(x),np.copy(y)
    ind = np.zeros(A.shape[0])
    M = distance_matrix(A,B)
    for i in range(A.shape[0]):
        a,b = np.unravel_index(np.nanargmin(M), np.array(M).shape)
        ind[b] = a
        M[a,:], M[:,b] = np.inf, np.inf
    return ind.astype(int)

def get_label(A,C):
    M = distance_matrix(A,C)
    return np.argmin(M, axis=1)

def get_centroids(x,l):
    k = np.unique(l)
    c = np.zeros((len(k),x.shape[1]))
    s = np.zeros((len(k),x.shape[1]))
    for i,label in enumerate(k):
        c[i,:] = np.nanmedian(x[l==label,:], axis=0)
        s[i,:] = np.std(x[l==label,:],axis=0)
    return c,s 

def add_oop_outliers(x,l,r):
    oop_outs_i = np.random.permutation(np.arange(len(l)))[:int(r*len(l))]
    a = np.zeros(len(l))
    cmed, cstd = get_centroids(x,l)
    k = np.unique(l)
    pos = np.ones((len(l),len(k)))
    for i,label in enumerate(k):
        pos[l==label,i]=0
        if label != 0:
            s = int(np.sum(pos[:,i])/20)
            pos[:,i] = pd.Series(pos[:,i]).rolling(s,min_periods=1, center=True).mean().to_numpy().astype(int)
        else:
            pos[:,i] = 0

    for i in oop_outs_i:
        old_label = l[i]
        if np.sum(pos[i,:]) > 0:
            p = pos[i,:]/np.sum(pos[i,:])
            new_label = np.random.choice(k, 1, p=p)  
            new_label = new_label-1 if not 0 in k else new_label  
            x[i,:] = cmed[new_label,:] + cstd[new_label,:]*(np.random.rand(1,cstd.shape[1])-0.5)
            l[i] = 0

    return x,l

