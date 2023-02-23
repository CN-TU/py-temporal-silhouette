#!/usr/bin/env python3
# FIV, Jun 2022

# install dependencies to a local subdirectory
import dependencies
dependencies.assert_pkgs({
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
    'glob': 'glob3',
    're': 'regex'
})

from DenStream import DenStream
# Repo for DenStream: https://github.com/issamemari/DenStream
from clusopt_core.cluster import CluStream, Streamkm
# Repo for CluStream and StreamKM++: https://github.com/giuliano-oliveira/clusopt_core
from sklearn.cluster import Birch

from TSindex import tempsil

import numpy as np
import pandas as pd

import sys
import os
from os.path import exists
import subprocess
import glob
import re
import ntpath

from sklearn import metrics
from scipy.io.arff import loadarff 
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# if no parameters are provided, the script will be run with these configurations
DEFAULT_RUNS = '''
dataS/base/arff results/ base normal
dataS/base/arff results/ base-r remove
dataS/moving/arff results/ mov normal
dataS/moving/arff results/ mov-r remove
dataS/nonstat/arff results/ nonst normal
dataS/nonstat/arff results/ nonst-r remove
dataS/nonstat/arff results/ nonst-p phase
dataS/sequential/arff results/ seq normal
dataS/sequential/arff results/ seq-r remove
dataS/sequential/arff results/ seq-p phase
'''


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


if len(sys.argv) < 2:
    print ('No arguments given. Running default configurations.')
    runs = [ [sys.executable, __file__] + params.split(' ') for params in DEFAULT_RUNS.split('\n') if params ]
    for run in runs:
        print (' '.join(run))
        os.makedirs(run[3], exist_ok=True)
        subprocess.call(run)
    quit()

inpath  = sys.argv[1]
outpath = sys.argv[2]
group = sys.argv[3]
rm_outliers = sys.argv[4]

np.random.seed(0)

print("\nData folder:",inpath)
print("Result folder:",outpath)
print("Data group:",group)
print("Remove outliers:",rm_outliers)

os.makedirs(outpath, exist_ok=True)

res = []
df_columns=['filename','group','outliers','idf','algorithm','AMI','Sil','CH','DB','TS']
df = pd.DataFrame(columns=df_columns)
pd.set_option('display.float_format', '{:.10f}'.format)

outfile = outpath + "results_synthetic.csv"
if exists(outfile) == False:
    df.to_csv(outfile, sep=',')

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*.arff'))):
    print("\nData file (.arff)", filename)
    print("Data file index: ", idf)
    m = re.match('.*_([0-9]+).arff', filename)
    dataset_idx = m.group(1) if m else 0
    filename_short = ntpath.basename(filename)
    arffdata = loadarff(filename)
    df_data = pd.DataFrame(arffdata[0])

    if(df_data['class'].dtypes == 'object'):
        df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))

    labels = df_data['class'].to_numpy().astype(int)
    data = df_data.to_numpy()[:,:-1]
    data = np.array(data, dtype=np.float64)

    if rm_outliers=='remove':
        data = data[labels!=0,:]
        labels = labels[labels!=0]

    elif rm_outliers=='phase':
        data = data[labels!=0,:]
        labels = labels[labels!=0]
        data, labels = add_oop_outliers(data, labels,0.01)

    timestamps = np.arange(data.shape[0])
    print("Dataset shape:", data.shape)

    k = len(np.unique(labels))
    k=k-1 if 0 in labels else k # adjust k depending if outliers (label==0) in dataset 

    # Data-batch size
    blocksize = 200
    print("Blocksize:", blocksize)

    # Stream Clustering algorithms
    cls = CluStream(m=k*10, h=5000, t=2)
    stk = Streamkm(coresetsize=k * 10, length=5000, seed=42)
    den = DenStream(eps=0.2, lambd=0.1, beta=0.2, mu=11)
    bir = Birch(n_clusters=k, threshold=0.5)
    grt = []
    algorithms = (("CluStream", cls),("DenStream", den),("BIRCH", bir),("StreamKM", stk),("GT", grt))


    old_clusters = []
    for alg_name, alg in algorithms:

        #print("Algorithm:", alg_name)
        if alg_name == 'CluStream':
            alg.init_offline(data[:blocksize,:], seed=42)

        y = np.zeros(len(data)) 

        for i in range(0,data.shape[0],blocksize):

            chunk = data[i:(i+blocksize),:]

            if alg_name == 'CluStream':
                alg.partial_fit(chunk)
                clusters, _ = alg.get_macro_clusters(k, seed=42)
                if i!=0:
                    ind = update_cls(clusters, old_clusters)
                    clusters = clusters[ind,:]
                y[i:(i+blocksize)] = get_label(chunk,clusters)
                old_clusters = clusters

            elif alg_name == 'BIRCH':
                alg = alg.partial_fit(chunk)
                y[i:(i+blocksize)] = alg.predict(chunk)

            elif alg_name == 'StreamKM':
                alg.partial_fit(chunk)
                if i!=0:
                    ind = update_cls(clusters, old_clusters)
                    clusters = clusters[ind,:]
                clusters, _ = alg.get_final_clusters(k, seed=42)
                y[i:(i+blocksize)] = get_label(chunk,clusters)
                old_clusters = clusters

            elif alg_name == 'GT':
                y[i:(i+blocksize)] = labels[i:(i+blocksize)]
            else: #DenStream
                try: 
                    y[i:(i+blocksize)] = alg.fit_predict(chunk)
                except: # DenStream crashes sometimes when calling DBSCAN with a 1-data-point array
                    y[i:(i+blocksize)] = 0

        AMI = adjusted_mutual_info_score(labels, y)
        try: # Traditional validation can fail in cases with only one cluster
            Sil = metrics.silhouette_score(data, y, metric='euclidean')
            CH = metrics.calinski_harabasz_score(data, y)
            DB = metrics.davies_bouldin_score(data, y)
        except:
            Sil, CH, DB = 0, 0, np.inf

        _,coeff, TS = tempsil(timestamps,data,y,s=200,kn=500,c=1)

        df = df.append({'filename':filename, 'group':group, 'outliers': rm_outliers, 'idf':idf, 'algorithm':alg_name, 'AMI':AMI, 'Sil':Sil, 'CH':CH, 'DB':DB, 'TS':TS}, ignore_index=True)
        print(df.tail(1))

print("Summary file:",outfile,"\n")
df.to_csv(outfile, sep=',', mode='a', header=False)

