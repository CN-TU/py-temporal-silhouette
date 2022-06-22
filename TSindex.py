#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial import distance, distance_matrix
from scipy.stats import median_absolute_deviation

def dist2oc(x,M):
    d = distance.cdist(M,x)
    return np.median(d)

def find_knearest(x, v, k, option='nearest'):
    x=x.flatten()
    if option=='random':
        i = np.random.permutation(len(x))
    else:
        i = np.argsort((np.abs(x - v)))
    ind = i[:k]
    return x[ind]

def mad(x):
    try:
        madx = median_absolute_deviation(x, scale = np.quantile(x,0.75))
    except:
        madx = median_absolute_deviation(x)
    madx = 1 if madx<1 else madx 
    nmads = np.abs(x-np.median(x))/madx
    outs = len(nmads[nmads>3])
    return outs


def tempsil(t,x,l,s=200,kn=200,c=1):
    # Description: implementation of the Temporal Silhouette index for the
    # internal validation of streaming clustering, FIV, Jun 2022
    #
    # INPUTS
    # t: 1D-array with timestamps
    # x: 2D-array with data vectors
    # l: 1D-array with labels
    # s: window-size of the simple-moving-average (SMA)
    # kn: number-of-neighbors of other clusters for calculating beta
    # c: weight parameters for gamma
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
        IAD = xk.shape[0] / (xk.shape[0] + c * mad(dtl))
        wj = np.argwhere(l==label)
        wnt = np.argwhere(l!=label)
        SMA = np.zeros(xk.shape)
        tst = np.zeros(xk.shape)
        for a in range(xk.shape[1]):
            SMA[:,a] = pd.Series(xk[:,a]).rolling(s,min_periods=1, center=True).mean().to_numpy()
        a = np.zeros(len(SMA))
        b = np.zeros(len(SMA))
        for j in range(len(SMA)-1):
            a[j] = 2*distance.euclidean(xk[j],SMA[j])
            tst[j] = 0
            if len(wnt)>0:
                m = find_knearest(wnt,wj[j],kn,option='nearest')
                b[j] = dist2oc([xk[j]],x[m,:])
                tst[j] = (b[j] - a[j])/np.max([a[j],b[j]]) 
        ts[i] = np.mean(tst) * IAD
    _,card = np.unique(l,return_counts=True)
    ts2 = np.power(ts,2)
    TS = np.sqrt(np.sum(card*ts2)/np.sum(card))
    return k,ts2,TS


