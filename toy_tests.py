#!/usr/bin/env python3

# install dependencies to a local subdirectory
import dependencies
dependencies.assert_pkgs({
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'sklearn': 'scikit-learn',
})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TSindex import tempsil
from scipy.spatial import distance_matrix
from sklearn.datasets import make_blobs, make_classification
import sys

c  = int(sys.argv[1])

def get_label(A,C):
    M = distance_matrix(A,C)
    return np.argmin(M, axis=1)

def get_centroids(x,l):
    k = np.unique(l)
    c = np.zeros((len(k),x.shape[1]))
    s = np.zeros((len(k),x.shape[1]))
    for i,label in enumerate(k):
        c[i,:] = np.nanmedian(x[l==label,:])
        s[i,:] = np.std(x[l==label,:])
    return c,s 

def add_oop_outliers(x,l,r):
    oop_outs_i = np.random.permutation(np.arange(len(l)))[:int(r*len(l))]
    a=np.zeros(len(l))
    cmed, cstd = get_centroids(x,l)
    k = np.unique(l)
    pos = np.ones((len(l),len(k)))
    for i,label in enumerate(k):
        pos[l==label,i]=0
        if not label == 0:
            s = int(np.sum(pos[:,i])/20)
            pos[:,i] = pd.Series(pos[:,i]).rolling(s,min_periods=1, center=True).mean().to_numpy().astype(int)

    for i in oop_outs_i:
        old_label = l[i]
        if np.sum(pos[i,:]) > 0:
            p = pos[i,:]/np.sum(pos[i,:])
            new_label = np.random.choice(k, 1, p=p)       
            x[i,:] = cmed[new_label-1,:] + cstd[new_label-1,:]*(np.random.rand(1,cstd.shape[1])-0.5)*3
            l[i] = 0

    return x,l


def init_plots():
    fig = plt.figure(figsize=(20, 6))
    ax0 = plt.subplot(2,6,1)
    ax0.set_title("1. Sudden drift", fontsize=16)
    ax0.set_ylabel("A .",rotation=0, fontsize=20)
    ax1 = plt.subplot(2,6,2)
    ax1.set_title("2. Incremental drift", fontsize=16)
    ax2 = plt.subplot(2,6,3)
    ax2.set_title("3. Gradual drift", fontsize=16)
    ax3 = plt.subplot(2,6,4)
    ax3.set_title("4. Reocc. concepts", fontsize=16)
    ax4 = plt.subplot(2,6,5)
    ax4.set_title("5. Spatial outliers", fontsize=16)
    ax5 = plt.subplot(2,6,6)
    ax5.set_title("6. Temporal outliers", fontsize=16)
    ax6 = plt.subplot(2,6,7)
    ax6.set_ylabel("B .",rotation=0, fontsize=20)
    ax6.set_xlabel("time",rotation=0, fontsize=16)
    ax7 = plt.subplot(2,6,8)
    ax7.set_xlabel("time",rotation=0, fontsize=16)
    ax8 = plt.subplot(2,6,9)
    ax8.set_xlabel("time",rotation=0, fontsize=16)
    ax9 = plt.subplot(2,6,10)
    ax9.set_xlabel("time",rotation=0, fontsize=16)
    ax10 = plt.subplot(2,6,11)
    ax10.set_xlabel("time",rotation=0, fontsize=16)
    ax11 = plt.subplot(2,6,12)
    ax11.set_xlabel("time",rotation=0, fontsize=16)
    axs = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]
    return fig, axs

def plot_scatter(axs, pos, t, x, y):
    col = np.array(['black', 'blue', 'orange', 'green'])
    axs[pos].scatter(t,x, s=5, color=col[y], rasterized=True)

def make_data(case):
    if case==1:
        X, y = make_blobs(n_samples=1000, centers=2, cluster_std=0.2, n_features=1, random_state=0)
        y += 1 # 0s are for outliers if any
        ys = y[499:]
        ys[ys==2] += 1
        y[499:] = ys
        X[y==3] += 2

    elif case==2:
        X, y = make_blobs(n_samples=1000, centers=3, cluster_std=0.2, n_features=1, random_state=0)
        y += 1 # 0s are for outliers if any
        t = np.arange(len(y))/200
        X = X.flatten()
        X[y==1] = X[y==1] + t[y==1]  
        X[y==2] = X[y==2] + 4  
        X = X.reshape((len(X),1))

    elif case==3:
        X1, y1 = make_blobs(n_samples=250, centers=1, cluster_std=0.2, n_features=1, random_state=0)
        y1 += 2 # 0s are for outliers if any
        X2, y2 = X1+2, y1+1
        X1, X2 = X1.flatten(), X2.flatten()
        X,y = np.zeros(2*len(y1)),np.zeros(2*len(y1))
        t = np.arange(len(y))
        p = 1/(1 + np.exp(-(t-250)/100))
        c1, c2 = 0, 0
        for i in range(len(y)):
            v = np.random.choice([0,1], 1, p = [1-p[i], p[i]])
            if v:
                if c2 < len(y2):
                    y[i] = 3
                    X[i] = X2[c2] 
                    c2 += 1
                else:
                    y[i] = 2
                    X[i] = X1[c1] 
                    c1 += 1
            else:
                if c1 < len(y1):
                    y[i] = 2
                    X[i] = X1[c1] 
                    c1 += 1
                else:
                    y[i] = 3
                    X[i] = X2[c2] 
                    c2 += 1
        Xd = X.reshape((len(X),1))
        yd = y.astype(int)
        Xc, yc = make_blobs(n_samples=500, centers=1, cluster_std=0.2, n_features=1, random_state=0)
        Xc -= 2.5
        yc += 1 
        X,y = np.zeros(1000),np.zeros(1000)
        c1, c2 = 500, 500
        for i in range(1000):
            p = [c1/(c1+c2),1-c1/(c1+c2)]
            if np.random.choice([True, False], 1, p=p):
                X[i], y[i] = Xd[500-c1], yd[500-c1]
                c1 -=1
            else:
                X[i], y[i] = Xc[500-c2], yc[500-c2]
                c2 -=1

        X = X.reshape((len(X),1))
        y = y.astype(int)

    elif case==4:
        X1, y1 = make_blobs(n_samples=75, centers=1, cluster_std=0.2, n_features=1, random_state=0)
        y1 += 2 # 0s are for outliers if any
        X2, y2 = X1[:25,:]+2, y1[:25]+1
        Xd = np.concatenate((X1, X2), axis=0)
        yd = np.concatenate((y1, y2), axis=0)
        for i in range(4):
            #X = np.concatenate((X, X), axis=0)
            #y = np.concatenate((y, y), axis=0)
            Xd = np.concatenate((Xd, X1), axis=0)
            yd = np.concatenate((yd, y1), axis=0)
            Xd = np.concatenate((Xd, X2), axis=0)
            yd = np.concatenate((yd, y2), axis=0)
        Xc, yc = make_blobs(n_samples=500, centers=1, cluster_std=0.2, n_features=1, random_state=0)
        Xc -= 2.5
        yc += 1 

        X,y = np.zeros(1000),np.zeros(1000)
        c1, c2 = 500, 500
        for i in range(1000):
            p = [c1/(c1+c2),1-c1/(c1+c2)]
            if np.random.choice([True, False], 1, p=p):
                X[i], y[i] = Xd[500-c1], yd[500-c1]
                c1 -=1
            else:
                X[i], y[i] = Xc[500-c2], yc[500-c2]
                c2 -=1

        X = X.reshape((len(X),1))
        y = y.astype(int)

    elif case==5:
        X0, y0 = make_blobs(n_samples=50, centers=1, cluster_std=5, n_features=1, random_state=0)
        X1, y1 = make_blobs(n_samples=475, centers=1, cluster_std=0.2, n_features=1, random_state=0)
        X1, y1 = X1+2, y1+1
        X2, y2 = X1-5, y1+1
        X1mx, X2mx = np.max(X1)*1.1, np.max(X2)*1.1
        X1mn, X2mn = np.min(X1)*0.9, np.min(X2)*0.9
        for i in range(len(X0)):
            if X0[i,0] >= X1mn and X0[i,0] <= X1mx:
                X0[i,0] += 5
            elif X0[i,0] >= X2mn and X0[i,0] <= X2mx:
                X0[i,0] -= 5

        X = np.concatenate((X0, X1, X2), axis=0)
        y = np.concatenate((y0, y1, y2), axis=0)
        t = np.arange(len(y))
        i = np.random.permutation(len(y))
        y[t] = y[i]
        X[t] = X[i]

    else:
        X1, y1 = make_blobs(n_samples=150, centers=1, cluster_std=0.2, n_features=1, random_state=0)
        y1 += 1
        X2, y2 = X1+3, y1+1
        X3, y3 = make_blobs(n_samples=50, centers=1, cluster_std=0.2, n_features=1, random_state=0)
        X3, y3 = X3 + 1.5, y3+3
        X = np.concatenate((X1, X2, X3, X1), axis=0)
        y = np.concatenate((y1, y2, y3, y1), axis=0)
        for i in range(1):
            X = np.concatenate((X, X), axis=0)
            y = np.concatenate((y, y), axis=0)
        X,y = add_oop_outliers(X,y,0.01)

    t = np.arange(len(y))
    return X,y,t

def evaluate(t,X,y,c,text):
    _,c,ts = tempsil(t,X,y,s=100,kn=200,c=c)
    print("%s TS: %.3f" % (text, ts))


np.random.seed(0)
fig, axs = init_plots()

# 1. Sudden drift
print("1. Sudden drift")
X,y,t = make_data(1)
# case A
evaluate(t,X,y,c,"Case A (3 clusters) -")
plot_scatter(axs, 0, t, X, y)
# case B
y[y==3]=2
evaluate(t,X,y,c,"Case B (2 clusters) -")
plot_scatter(axs, 6, t, X, y)

# 2. Incremental drift
print("\n2. Incremental drift")
X,y,t = make_data(2)
# case A
evaluate(t,X,y,c,"Case A (3 clusters) -")
plot_scatter(axs, 1, t, X, y)
# case B
y[y==3]=1
evaluate(t,X,y,c,"Case B (2 clusters) -")
plot_scatter(axs, 7, t, X, y)

# 3. Gradual drift
print("\n3. Gradual drift")
X,y,t = make_data(3)
# case A
evaluate(t,X,y,c,"Case A (3 clusters) -")
plot_scatter(axs, 2, t, X, y)
# case B
y[y==3]=2
evaluate(t,X,y,c,"Case B (2 clusters) -")
plot_scatter(axs, 8, t, X, y)

# 4. Reoccuring concepts
print("\n4. Reoccuring concepts")
X,y,t = make_data(4)
# case A
evaluate(t,X,y,c,"Case A (3 clusters) -")
plot_scatter(axs, 3, t, X, y)
# case B
y[y==3]=2
evaluate(t,X,y,c,"Case B (2 clusters) -")
plot_scatter(axs, 9, t, X, y)

# 5. Outliers: local and extreme
print("\n5. Spatial outliers: local and extreme")
X,y,t = make_data(5)
# case A
evaluate(t,X,y,c,"Case A (2 clusters + outliers) -")
plot_scatter(axs, 4, t, X, y)
# case B
X1mu, X2mu = np.mean(X[y==1]), np.mean(X[y==2])
y = get_label(X,[[X1mu],[X2mu]]) + 1
evaluate(t,X,y,c,"Case A (2 clusters) -")
plot_scatter(axs, 10, t, X, y)


# 6. Outliers: out-of-phase
print("\n6. Temporal outliers: out-of-phase")
X,y,t = make_data(6)
# case A
evaluate(t,X,y,c,"Case A (3 clusters + outliers) -")
plot_scatter(axs, 5, t, X, y)
# case B
X1mu, X2mu, X3mu = np.mean(X[y==1]), np.mean(X[y==2]), np.mean(X[y==3])
y = get_label(X,[[X1mu],[X2mu],[X3mu]]) + 1
evaluate(t,X,y,c,"Case A (3 clusters) -")
plot_scatter(axs, 11, t, X, y)

print("\nTemporal Silhouette (TS) adjusted with sigma (c)=",c) 

#fig.suptitle('Different types of concept drift and outliers (x-axis: time,  y-axis: 1D-data)', fontsize=16)
plt.tight_layout()
#plt.savefig("toy_tests.pdf")
plt.show()







