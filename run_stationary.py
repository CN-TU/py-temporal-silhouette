#!/usr/bin/env python3
# FIV, Jun 2023

# checking package dependencies
from dependencies import check_pkgs
check_pkgs()

from TSindex import tempsil
import cvi
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

#only for plotting purposes 
plot_datasets=1
if plot_datasets:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# if no parameters are provided, the script will be run with these configurations
DEFAULT_RUNS = '''
dataT results/ stationary
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
experiments = sys.argv[3]

np.random.seed(0)

print("\nData folder:",inpath)
print("Result folder:",outpath)
print("Experiments:",experiments)

os.makedirs(outpath, exist_ok=True)

res = []
df_columns=['filename','group','corr_level','total_corr_levels','idf','AMI','Sil','CH','DB','iXB','iPS','irCIP','TS']
df = pd.DataFrame(columns=df_columns)
pd.set_option('display.float_format', '{:.10f}'.format)

outfile = outpath + "results_stationary.csv"
if exists(outfile) == False:
    df.to_csv(outfile, sep=',')

tests = ['sp_overclustering','sp_underclustering','slicing_clusters','impurity']
corruption_levels = 8

if plot_datasets:
    fig = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.11, hspace=0.0, top=0.90, bottom=0.15, left=0.05, right=0.99) 

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*.arff'))):
    print("\nData file (.arff)", filename)
    print("Data file index: ", idf)
    filename_short = ntpath.basename(filename)

    arffdata = loadarff(filename)
    df_data = pd.DataFrame(arffdata[0])

    labels = df_data['class'].to_numpy().astype(int)
    data = df_data.to_numpy()[:,:-1]
    data = np.array(data, dtype=np.float64)
    data = (data-data.min())/(data.max()-data.min())

    new_pos = np.random.permutation(data.shape[0])
    data = data[new_pos,:]
    labels = labels[new_pos]

    outliers = 0
    k = len(np.unique(labels))
    if min(labels) == -1:
        labels = labels+1
        outliers=1

    timestamps = np.arange(data.shape[0])
    print("Dataset shape:", data.shape)

    if plot_datasets:
        df_data2d = df_data.iloc[:, [0, 1, -1]]
        df_data2d.columns = ['feat 1', 'feat 2', 'cluster']
        aux = df_data2d.drop('cluster', axis=1)
        aux = (aux-aux.min())/(aux.max()-aux.min())
        df_data2d = pd.concat((aux, df_data2d['cluster']), 1)
        df_data2d['cluster'] = df_data2d['cluster'].astype(int)
        df_data2d = df_data2d.sort_values(by=['cluster'], ascending=True)
        df_data2d['cluster'] = df_data2d['cluster'].astype(str)

        ax = plt.subplot(gs[idf])
        if outliers:
            sns.scatterplot(data=df_data2d, x="feat 1", y="feat 2", hue="cluster", ax=ax, s=10, legend=False,  alpha=0.7, linewidth=0.1)
        else:
            sns.scatterplot(data=df_data2d, x="feat 1", y="feat 2", hue="cluster", ax=ax, s=10, legend=False)
        ax.set_xlabel('feat 1', fontsize=18)
        ax.set_ylabel('feat 2', fontsize=18)
        if idf>0:
            ax.set_ylabel("")
        plt_title = filename_short.split('.')[0]
        plt.title(plt_title, fontsize=18, y=1)

    old_clusters = []
    for test in tests:
        for i in range(corruption_levels):
            y = ub.corrupt_labels(data, labels, test, i, corruption_levels, outliers) 

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
                try:   
                    cip_crit[ix] = cip.get_cvi(data[ix, :], y[ix])   
                except:
                    rCIP = 0
            iXB = np.nanmean(ixb_crit)
            iPS = np.nanmean(ips_crit)
            rCIP = np.nanmean(cip_crit)


            df = df.append({'filename':filename, 'group':test, 'corr_level': i, 'total_corr_levels': corruption_levels, 'idf':idf, 'AMI':AMI, 'Sil':Sil, 'CH':CH, 'DB':DB, 'iXB':iXB, 'iPS':iPS, 'irCIP':rCIP, 'TS':TS}, ignore_index=True)
            print(df.tail(1))

print("Summary file:",outfile,"\n")
df.to_csv(outfile, sep=',', mode='a', header=False)

if plot_datasets:
    plt.savefig("plots/stationary_datasets.pdf", rasterized=True, dpi=100)
    plt.close()


