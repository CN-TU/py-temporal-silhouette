#!/usr/bin/env python3
# FIV, Jun 2023

# checking package dependencies
from dependencies import check_pkgs
check_pkgs()

from DenStream import DenStream
# Repo for DenStream: https://github.com/issamemari/DenStream
from clusopt_core.cluster import CluStream, Streamkm
# Repo for CluStream and StreamKM++: https://github.com/giuliano-oliveira/clusopt_core
from sklearn.cluster import Birch

import cvi
from TSindex import tempsil
import utils.bedtest as ub

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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# if no parameters are provided, the script will be run with these configurations
DEFAULT_RUNS = '''
dataS/sequential/arff results/ seq-p phase
dataS/base/arff results/ base normal
dataS/base/arff results/ base-r remove
dataS/moving/arff results/ mov normal
dataS/moving/arff results/ mov-r remove
dataS/nonstat/arff results/ nonst normal
dataS/nonstat/arff results/ nonst-r remove
dataS/nonstat/arff results/ nonst-p phase
dataS/sequential/arff results/ seq normal
dataS/sequential/arff results/ seq-r remove
'''



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
df_columns=['filename','group','outliers','idf','algorithm','AMI','Sil','CH','DB','iXB','iPS','irCIP','TS']
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
        data, labels = ub.add_oop_outliers(data, labels,0.01)

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

        if alg_name == 'CluStream':
            alg.init_offline(data[:blocksize,:], seed=42)

        y = np.zeros(len(data)) 

        for i in range(0,data.shape[0],blocksize):

            chunk = data[i:(i+blocksize),:]

            if alg_name == 'CluStream':
                alg.partial_fit(chunk)
                clusters, _ = alg.get_macro_clusters(k, seed=42)
                if i!=0:
                    ind = ub.update_cls(clusters, old_clusters)
                    clusters = clusters[ind,:]
                y[i:(i+blocksize)] = ub.get_label(chunk,clusters)
                old_clusters = clusters

            elif alg_name == 'BIRCH':
                alg = alg.partial_fit(chunk)
                y[i:(i+blocksize)] = alg.predict(chunk)

            elif alg_name == 'StreamKM':
                alg.partial_fit(chunk)
                if i!=0:
                    ind = ub.update_cls(clusters, old_clusters)
                    clusters = clusters[ind,:]
                clusters, _ = alg.get_final_clusters(k, seed=42)
                y[i:(i+blocksize)] = ub.get_label(chunk,clusters)
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

        _,coeff, TS = tempsil(timestamps,data,y,s=100,kn=1000,c=1)

        #incremental cvi
        ixb, ips, cip = cvi.XB(), cvi.PS(), cvi.rCIP()
        ixb_crit, ips_crit, cip_crit = np.zeros(len(y)),np.zeros(len(y)),np.zeros(len(y)) 
        for ix in range(len(y)):
            ixb_crit[ix] = ixb.get_cvi(data[ix, :], y[ix])   
            ips_crit[ix] = ips.get_cvi(data[ix, :], y[ix])   
            cip_crit[ix] = cip.get_cvi(data[ix, :], y[ix])   
        iXB = np.nanmean(ixb_crit)
        iPS = np.nanmean(ips_crit)
        rCIP = np.nanmean(cip_crit)

        df = df.append({'filename':filename, 'group':group, 'outliers': rm_outliers, 'idf':idf, 'algorithm':alg_name, 'AMI':AMI, 'Sil':Sil, 'CH':CH, 'DB':DB, 'iXB':iXB, 'iPS':iPS, 'irCIP':rCIP, 'TS':TS}, ignore_index=True)
        print(df.tail(1))

print("Summary file:",outfile,"\n")
df.to_csv(outfile, sep=',', mode='a', header=False)

