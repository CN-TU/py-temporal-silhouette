#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import median_abs_deviation

def temp_centroid_coherence(X):
    A,B = X[1:,:],X[:-1,:]   
    d = A-B
    d = np.linalg.norm(d, axis=1)
    if len(d):
        return mad(d,False)
    else:
        return 0

def dist2oc(x,M,l):
    d = distance.cdist(M,x)
    clus = np.unique(l)
    meandclus = np.inf * np.ones(len(clus))
    for i,c in enumerate(clus):
        meandclus[i] = np.mean(d[l==c])
    return np.min(meandclus)

def find_knearest(x, v, k, option='nearest'):
    x=x.flatten()
    if option=='random':
        i = np.random.permutation(len(x))
    else:
        i = np.argsort((np.abs(x - v)))
    ind = i[:k]
    return x[ind]

def mad(x, x_is_int=True):
    madx = median_abs_deviation(x, scale = "normal")
    if x_is_int:
        madx = 1 if madx<1 else madx 
    else:
        madx = 1 if madx==0 else madx 
    nmads = np.abs(x-np.median(x))/madx
    outs = len(nmads[nmads>3])
    return outs

def tempsil(t,x,l,s=200,kn=200,c=1):
    # Description: implementation of the Temporal Silhouette index for the
    # internal validation of streaming clustering, FIV, Jun 2022
    #
    # INPUTS
    # t: 1D-array with timestamps
    # x: XD-array with data vectors
    # l: 1D-array with labels
    # s: window-size of the simple-moving-average (SMA)
    # kn: number-of-neighbors of other clusters for calculating beta
    # c: sigma parameter to weight the penalization over contextual outliers [0...1]
    #
    # OUTPUTS
    # k: 1D-array with cluster-labels
    # ts2: 1D-array with quadratic cluster temporal silhouettes
    # TS: global Temporal Silhuette
 
    k = np.unique(l)
    ts = np.zeros(len(k))
    for i,label in enumerate(k):
        tl0 = t[l==label]
        tl1 = np.roll(tl0,-1)
        dtl = (tl1-tl0)[:-1]
        xk = x[l==label,:]
        IAD = 1 / (1 + c * mad(dtl,True)/xk.shape[0])
        wj = np.argwhere(l==label)
        wnt = np.argwhere(l!=label)
        SMA = np.zeros(xk.shape)
        tst = np.zeros(xk.shape)
        for a in range(xk.shape[1]):
            SMA[:,a] = pd.Series(xk[:,a]).rolling(s,min_periods=1, center=True).mean().to_numpy()
        a = np.zeros(len(SMA))
        b = np.zeros(len(SMA))
        for j in range(len(SMA)-1):
            a[j] =  distance.euclidean(xk[j],SMA[j])
            tst[j] = 0
            if len(wnt)>0:
                m = find_knearest(wnt,wj[j],kn,option='nearest')
                b[j] = dist2oc([xk[j]],x[m,:],l[m])
                tst[j] = (b[j] - a[j])/np.max([a[j],b[j]])
        TCD = temp_centroid_coherence(SMA)/xk.shape[0]
        ts[i] = (1 + np.mean(tst)) * IAD * (1 - TCD) - 1
    _,card = np.unique(l,return_counts=True)
    ts2 = np.power(ts,2) * np.sign(ts)
    TS2 = np.sum(card*ts2)/np.sum(card)
    TS = np.sqrt(np.abs(TS2)) * np.sign(TS2)
    return k,ts2,TS


