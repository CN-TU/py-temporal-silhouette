#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

import pandas as pd
import numpy as np
import sys
import os

import warnings
warnings.filterwarnings("ignore")

perffile  = sys.argv[1]

os.makedirs("plots", exist_ok=True)

df = pd.read_csv(perffile)
# Inverting DB and iXB to make them comparable
df['DB'] = df['DB'] * -1
df['iXB'] = df['iXB'] * -1
df['irCIP'] = df['irCIP'] * -1

# filename, group, corr_level, total_corr_levels, idf, AMI, Sil, CH, DB, iXB, iPS, irCIP, TS

fileset = pd.unique(df['filename'])
groups = ['sp_overclustering','sp_underclustering','slicing_clusters','impurity']

fig, ax = plt.subplots(figsize=(15, 4))

dfline = df[['AMI','Sil','TS']].sort_values(by=['AMI']).reset_index(drop=True)
sns.lineplot(data=dfline)
plt.xlabel("experiments (sorted by AMI values)", fontsize=16)
plt.ylabel("validation scores", fontsize=16)
plt.tight_layout()
plt.xlim([0,127])
plt.savefig("plots/ami-sil-ts.pdf", rasterized=True)
plt.close()

dfcorr = df[['AMI', 'Sil', 'CH', 'DB', 'iXB', 'iPS', 'irCIP', 'TS']].corr()
mask = np.triu(np.ones_like(dfcorr, dtype=bool))
f, ax = plt.subplots(figsize=(7, 5))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(dfcorr, mask=mask, annot=True, cmap=cmap, fmt=".2f", center=0, square=True, linewidths=.5)#, cbar_kws={"shrink": .5})
plt.tight_layout()
plt.savefig("plots/corrplot.pdf", rasterized=True)
plt.close()


for f in fileset:
    fcore = f.split('/')[1].split('.')[0]

    fig = plt.figure(figsize=(15,4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.05, hspace=0.0, top=0.92, bottom=0.12, left=0.01, right=0.99) 

    for i,g in enumerate(groups):
        dfaux = df.loc[(df["filename"] == f) & (df["group"] == g)]
        dfaux.set_index('corr_level')
        dfaux.drop(columns=['filename','group','total_corr_levels', 'idf','corr_level','Unnamed: 0'], inplace=True)
        dfaux.astype(float)
        dfaux = (dfaux-dfaux.min())/(dfaux.max()-dfaux.min())
        dfaux = dfaux.fillna(0)
        dfaux = dfaux.reset_index(drop=True)

        ax = plt.subplot(gs[i])
        sns.heatmap(data=dfaux, annot=True, fmt=".2f", cmap=cmap, center=0, square=True, linewidths=.5, cbar=False)
        plt.title(fcore+' '+g, fontsize=14, y=1)

    plt_title = fcore+'.pdf'
    plt.savefig("plots/"+plt_title, rasterized=True)
    plt.close()

