# Temporal Silhouette for Stream Clustering Validation
## Instructions for experiment replication

**author:** Félix Iglesias

**contact:** felix.iglesias@tuwien.ac.at


The TS index and the experiments here implemented are from the paper:

*Temporal Silhouette: Validation of Stream Clustering Robust to Concept Drift* (under review...)

*Jun., 2022*


### 0. Requirements

Experiments require the following general-purpose Python packages:

- numpy
- pandas
- scipy
- sklearn
- glob
- re
- sys
- subprocess
- ntpath

Stream clustering algoriths used are:

- Birch from sklearn
- DenStream from [https://github.com/issamemari/DenStream](https://github.com/issamemari/DenStream)
- CluStream and Streamkm from [https://github.com/giuliano-oliveira/clusopt_core](https://github.com/giuliano-oliveira/clusopt_core)

### 1. Temporal Silhouette Index (TS)

Use TS in your codes by importing the "tempsil" function:

        > from TSindex import tempsil

        > tempsil(t,x,l,s,kn,c)
        # INPUTS
        # t: 1D-array with timestamps
        # x: 2D-array with data vectors
        # l: 1D-array with labels
        # s: window-size of the simple-moving-average (SMA), by default s=200 
        # kn: number-of-neighbors of other clusters for calculating beta, by default kn=200
        # c: weight parameters for gamma, by default c=1
        #
        # OUTPUTS
        # k: 1D-array with cluster-labels
        # ts2: 1D-array with quadratic cluster temporal silhouettes
        # TS: global Temporal Silhuette

For a simple example, open a terminal and run:

        > $ pyhton3 toy_tests.py

### 1. Datasets

**Datasets** are located within the [data] folder. They are also publicly available in **Mendeley**: 

“Data for Evaluation of Stream Data Analysis Algorithms”. Mendeley Data, V1, doi: 10.17632/c43kr4t7h8.1, [https://data.mendeley.com/datasets/c43kr4t7h8/1](https://data.mendeley.com/datasets/c43kr4t7h8/1)

The *clean* (without outliers) versions and the versions with *out-of-phase* outliers are created online during the runing of the experiments

### 2. Replicating experiments

Open a terminal in the current folder. Run:

        > $ pyhton3 run_analysis.py

A *results.csv* file will be created in the [results/] folder.

Once experiments finish, to show a comparison among internal validity indices, run...

### 3. Comparing performances of internal validation indices

For a comparison with the Ground Truth (GT) as benchmark: 

        > $ pyhton3 compare_results.py results/results.csv gt

For a comparison with the best clustering according to external validation (AMI) as benchmark: 

        > $ pyhton3 compare_results.py results/results.csv ami


