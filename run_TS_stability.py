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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# if no parameters are provided, the script will be run with these configurations
DEFAULT_RUNS = '''
dataT results/ TSstab
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
df_columns=['filename','group','corr_level','total_corr_levels','idf','AMI','TS','s','kn']
df = pd.DataFrame(columns=df_columns)
pd.set_option('display.float_format', '{:.10f}'.format)

outfile = outpath + "results_TSstability.csv"
if exists(outfile) == False:
    df.to_csv(outfile, sep=',')

tests = ['sp_overclustering','sp_underclustering','slicing_clusters','impurity']
corruption_levels = 8

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*.arff'))):
    print("\nData file (.arff)", filename)
    print("Data file index: ", idf)
    filename_short = ntpath.basename(filename)

    arffdata = loadarff(filename)
    df_data = pd.DataFrame(arffdata[0])

    if(df_data['class'].dtypes == 'object'):
        df_data['class'] = df_data['class'].map(lambda x: x.decode("utf-8").lstrip('b').rstrip(''))

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

    ss = np.array([10,20,50,100,150,200,250,300,400,500]).astype(int)
    kns = np.linspace(200, 2000, num=10).astype(int)

    old_clusters = []
    for test in tests:
        for i in range(corruption_levels):
            y = ub.corrupt_labels(data, labels, test, i, corruption_levels, outliers) 
            AMI = adjusted_mutual_info_score(labels, y)
            for s in ss:
                for kn in kns:
                    _,coeff, TS = tempsil(timestamps,data,y,s=s,kn=kn,c=0)

                    newrow = {'filename':filename, 'group':test, 'corr_level': i, 'total_corr_levels': corruption_levels, 'idf':idf, 'AMI':AMI, 'TS':TS, 's':s, 'kn':kn}
                    df = pd.concat([df, pd.DataFrame([newrow])], ignore_index=True)
                    print(df.tail(1))

            print("Summary file:",outfile,"\n")
            df.to_csv(outfile, sep=',', mode='a', header=False)

