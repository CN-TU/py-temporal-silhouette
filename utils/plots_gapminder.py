#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
from scipy.io.arff import loadarff

import warnings
warnings.filterwarnings("ignore")

datafile  = sys.argv[1]
labelfile = sys.argv[2]

arffdata = loadarff(datafile)
dfd = pd.DataFrame(arffdata[0])
dfl = pd.read_csv(labelfile).astype(int)

dfl['gdp'] = dfd['GDP_per_capita']
dfl['cpw'] = dfd['children_per_women']
dfl['time (year-ID)'] = (np.arange(len(dfl))/17).astype(int)+1800
algs=['CluStream','DenStream','BIRCH','StreamKM','GT']

sns.set_theme(style="white")
fig, ax = plt.subplots(nrows=2, ncols=len(algs),figsize=(20, 8))
for i,alg in enumerate(algs):
    g = sns.scatterplot(data=dfl, x="time (year-ID)", y='gdp', hue=alg, ax=ax[0,i], s=10, alpha=0.7, palette='tab10', linewidth=0)
    g.set_title(alg, fontsize=16)
    g.set_ylabel('Income', fontsize=16)
    g.set_xlabel('', fontsize=16)
    g.set(yticklabels=['0','0','10k','20k','30k','40k','50k','60k','70k']) 
    if i>0:
        g.set_ylabel('', fontsize=16)

for i,alg in enumerate(algs):
    g = sns.scatterplot(data=dfl, x="time (year-ID)", y='cpw', hue=alg, ax=ax[1,i], s=10, alpha=0.7, palette='tab10', linewidth=0)
    g.set_ylabel('Child. per woman', fontsize=16)
    g.set_xlabel('time (year)', fontsize=16)
    if i>0:
        g.set_ylabel('', fontsize=16)

plt.show()
plt.tight_layout()
#plt.savefig("plots/gapmind_clust.pdf", dpi=100, rasterized=True)
