#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.preprocessing import StandardScaler

filename  = sys.argv[1]
best = sys.argv[2]

df = pd.read_csv(filename)

# Inverting DB and iXB to make them comparable
df['DB'] = df['DB'] * -1
df['iXB'] = df['iXB'] * -1
df['irCIP'] = df['irCIP'] * -1

datasets = pd.unique(df.idf) 
groups = pd.unique(df.group) 
algorithms = pd.unique(df.algorithm) 

df['AMI_best'] = 0  
df['Si_best'] = 0  
df['CH_best'] = 0    
df['DB_best'] = 0    
df['iXB_best'] = 0
df['iPS_best'] = 0
df['irCIP_best'] = 0
df['TS_best'] = 0
df['GT'] = 0

if best == 'gt':
    df.loc[df["algorithm"] == 'GT', 'GT'] = 1
else: 
    df = df.loc[(df["algorithm"] != "GT")]
    algorithms = pd.unique(df.algorithm) 

df_columns = df.columns
columns = [('AMI','AMI_best'),('Sil','Si_best'),('DB','DB_best'),('CH','CH_best'),('iXB','iXB_best'),('iPS','iPS_best'),('irCIP','irCIP_best'),('TS','TS_best')]
valinds = ['AMI', 'Sil', 'CH', 'DB', 'iXB', 'iPS', 'irCIP', 'TS']

df = df.round(15)
for dataset in datasets:
    for group in groups: 
        dfaux = df.loc[(df["idf"] == dataset) & (df["group"] == group)]
        for (cin,cout) in columns: 
            max_val = np.max(dfaux[cin].to_numpy())
            for index, row in dfaux.iterrows():
                if row[cin] == max_val:
                    df.at[index,cout] = 1


if best == 'gt':
    sbest = df['GT'].to_numpy()
    print("\n*** Best is GT ***")
else:
    sbest = df['AMI_best'].to_numpy()
    print("\n***Best is best AMI ***")

dfout = pd.DataFrame(columns=valinds, index=groups)

for group in groups: 
    for (cin,cout) in columns: 
        dfout.at[group,cin] = 0

for dataset in datasets:
    for group in groups: 
        dfaux = df.loc[(df["idf"] == dataset) & (df["group"] == group)]
        if best == 'gt':
            sbest = dfaux['GT'].to_numpy()
        else:
            sbest = dfaux['AMI_best'].to_numpy()
        for (cin,cout) in columns: 
            selec = dfaux[cout].to_numpy()
            if np.sum(sbest*selec==1) > 0:
                dfout.at[group,cin] = dfout.loc[group,cin] + 1

dfout.loc['all',:]= dfout.sum(axis=0)
print(dfout)

outfile_csv = '../results/res_comp_synth_'+best+'.csv'
print("Summary csv file:",outfile_csv)
dfout.to_csv(outfile_csv, sep=',', mode='w', header=False)

outfile_tex = '../results/res_comp_synth_'+best+'.tex'
print("Summary tex file:",outfile_tex,"\n")
dfout.to_latex(outfile_tex, float_format="%.2f" )


r=1 if best == 'gt' else 0 

fig = plt.figure(figsize=(15,5))
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.11, hspace=0.0, top=0.90, bottom=0.15, left=0.03, right=0.99) 

for i, alg in enumerate(algorithms):

    bp_data=np.zeros((20,len(groups)))         
    for g, group in enumerate(groups): 
        dfaux = df.loc[(df["algorithm"] == alg) & (df["group"] == group)]
        bp_data[:,g] = dfaux['AMI'].to_numpy()

    if i < len(algorithms)-r:
        ax = plt.subplot(gs[i])
        ax.boxplot(bp_data,patch_artist=True,labels=groups, showfliers=True)
        ax.set_ylim([-0.1,1.1])
        ax.set_xticklabels(labels=groups,rotation=45, fontsize=14)
        plt.title(alg, fontsize=18, y=1)

#plt.savefig("boxplots.pdf")
plt.show()


