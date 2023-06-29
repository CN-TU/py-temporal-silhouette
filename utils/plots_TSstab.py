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

# filename, group, corr_level, total_corr_levels, idf, AMI, TS, s, kn

fileset = pd.unique(df['filename'])
corr_level = pd.unique(df['corr_level'])
groups = ['sp_overclustering','sp_underclustering','slicing_clusters','impurity']
ss = pd.unique(df['s'])
kns = pd.unique(df['kn'])
df['svar'] = 0
df['knvar'] = 0

for f in fileset:
    for g in groups:
        for c in corr_level:
            for k in kns:
                dfaux = df.loc[(df["filename"] == f) & (df["group"] == g) & (df["corr_level"] == c) & (df["kn"] == k)]
                dfaux_avg = np.nanmedian(dfaux['TS'])                
                df.iloc[dfaux.index, -2] = df.iloc[dfaux.index, -5] - dfaux_avg
                dfaux = df.loc[(df["filename"] == f) & (df["group"] == g) & (df["corr_level"] == c) & (df["kn"] == k)]
            for s in ss:
                dfaux = df.loc[(df["filename"] == f) & (df["group"] == g) & (df["corr_level"] == c) & (df["s"] == s)]
                dfaux_avg = np.nanmedian(dfaux['TS'])                
                df.iloc[dfaux.index, -1] = df.iloc[dfaux.index, -5] - dfaux_avg
                dfaux = df.loc[(df["filename"] == f) & (df["group"] == g) & (df["corr_level"] == c) & (df["s"] == s)]
            dfaux = df.loc[(df["filename"] == f) & (df["group"] == g) & (df["corr_level"] == c)]

fig = plt.figure(figsize=(10,4))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15, hspace=0.0, top=0.90, bottom=0.12, left=0.05, right=0.95) 

ax = plt.subplot(gs[0])
dfbox = df[['s','svar']]
sns.boxplot(data=df, x="s", y="svar", ax=ax, notch=True, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "coral"},whis=[5, 95])
ax.set_xlabel('w', fontsize=16)
plt.title('ΔTS_w', fontsize=16, y=1)
ax.set_ylabel('', fontsize=16)

ax = plt.subplot(gs[1])
dfbox = df[['kn','knvar']]
sns.boxplot(data=df, x="kn", y="knvar", ax=ax, notch=True, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "coral"},whis=[5, 95])
ax.set_xlabel('k', fontsize=16)
plt.title('ΔTS_k', fontsize=16, y=1)
ax.set_ylabel('', fontsize=16)

plt_title = 'TS_stab.pdf'
plt.savefig("plots/"+plt_title, rasterized=True, dpi=100)
plt.close()

