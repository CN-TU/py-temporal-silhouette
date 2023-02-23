#!/usr/bin/env python3

# install dependencies to a local subdirectory
import dependencies
dependencies.assert_pkgs({
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
})

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
from scipy.io.arff import loadarff
#import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

datafile  = sys.argv[1]
labelfile = sys.argv[2]

arffdata = loadarff(datafile)
dfd = pd.DataFrame(arffdata[0])
dfl = pd.read_csv(labelfile).astype(int)

dfl['estimated beds'] = dfd['ICU_Beds_Occupied_Estimated']
dfl['total beds'] = dfd['ICU_Beds_Occupied_Total']
#dfl['time (day-ID)'] = (np.arange(len(dfl))/51).astype(int)
dfl['time (day-ID)'] = np.arange(len(dfl))
algs=['CluStream','DenStream','BIRCH','StreamKM','GT']

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
g1 = sns.scatterplot(data=dfl, x="estimated beds", y='total beds', palette="gist_ncar", hue='GT', ax=ax[0], s=10, alpha=0.7, legend=False)
g2 = sns.scatterplot(data=dfl, x="estimated beds", y='total beds', hue='time (day-ID)', ax=ax[1], s=10, alpha=0.7, legend=False)
xlabels = ['{:,.0f}'.format(x) + 'K' for x in g1.get_xticks()/1000]
ylabels = ['{:,.0f}'.format(y) + 'K' for y in g1.get_yticks()/1000]
g1.set_xticklabels(xlabels), g2.set_xticklabels(xlabels)
g1.set_yticklabels(ylabels), g2.set_yticklabels(ylabels)
g2.set(ylabel=None)  
g1.text(1700, 700,'Colors are US States', fontsize=11)
g2.text(1700, 700,'Colors are time (days)', fontsize=11)
sns.set_theme(style='white')
plt.show()
#plt.savefig("covid1.pdf", dpi=200)
plt.close()

palette = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
fig, ax = plt.subplots(nrows=2, ncols=len(algs),figsize=(20, 8))
for i,alg in enumerate(algs):
    #g = sns.lineplot(data=dfl, x="time (day-ID)", y='estimated beds', hue=alg, palette='nipy_spectral', ax=ax[0,i], markers=True, alpha=0.7, legend=False, marker='o')
    if alg == 'GT':
        palette = 'nipy_spectral'
    g = sns.scatterplot(data=dfl, x="time (day-ID)", y='estimated beds', hue=alg, palette=palette, ax=ax[0,i], s=10, alpha=0.7, legend=False, linewidth=0)
    g.set_title(alg)
    g.set_yticklabels(ylabels)
    g.set(xticklabels=[]) 
    #g.set(xlabel=None)  
    if i:
        g.set(ylabel=None)  

palette = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
for i,alg in enumerate(algs):
    #g = sns.lineplot(data=dfl, x="time (day-ID)", y='total beds', hue=alg, palette='nipy_spectral', ax=ax[1,i], markers=True, alpha=0.7, legend=False, marker='o')
    if alg == 'GT':
        palette = 'nipy_spectral'
    g = sns.scatterplot(data=dfl, x="time (day-ID)", y='total beds', hue=alg, palette=palette, ax=ax[1,i], s=10, alpha=0.7, legend=False, linewidth=0)
    g.set_yticklabels(ylabels)
    g.set(xticklabels=[]) 
    if i:
        g.set(ylabel=None)  

plt.show()
#plt.savefig("covid2.pdf", dpi=200)
