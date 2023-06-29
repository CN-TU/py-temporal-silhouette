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

dfl['values'] = dfd['values']
dfl['time (day-ID)'] = np.arange(len(dfl))
algs=['CluStream','DenStream','BIRCH','StreamKM','GT']

fig, ax = plt.subplots(nrows=1, ncols=len(algs), figsize=(16, 4))

for i,alg in enumerate(algs):
    g = sns.scatterplot(data=dfl, x="time (day-ID)", y='values', palette='tab10', hue=alg, ax=ax[i], alpha=1, s=5, linewidth=0)
    ylabels = ['{:,.1f}'.format(y) + 'M' for y in g.get_yticks()/1000000]
    g.set_xlabel('time (day-ID)', fontsize=16)
    g.set_ylabel('feat 1', fontsize=16)
    ax[i].set_title(alg, fontsize=16, y=1)
    g.set_yticklabels(ylabels)
    if i:
        g.set(ylabel=None)  

plt.show()
plt.tight_layout()
#plt.savefig("plots/retail.pdf", dpi=100, rasterized=True)
