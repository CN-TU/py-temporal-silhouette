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

fig, ax = plt.subplots(nrows=1, ncols=len(algs), figsize=(20, 4))

palette = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
for i,alg in enumerate(algs):
    g = sns.scatterplot(data=dfl, x="time (day-ID)", y='values', hue=alg, palette=palette, ax=ax[i], alpha=1, s=5, linewidth=0)
    ylabels = ['{:,.1f}'.format(y) + 'M' for y in g.get_yticks()/1000000]
    g.set_yticklabels(ylabels)
    if i:
        #g.set(yticklabels=[]) 
        g.set(ylabel=None)  

plt.show()
#plt.savefig("retail.pdf", dpi=200)
