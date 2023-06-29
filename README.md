## Temporal Silhouette for Stream Clustering Validation
### Instructions for experiment replication

**author:** Félix Iglesias

**contact:** felix.iglesias@tuwien.ac.at


The TS index and the experiments here implemented are from the paper:

*Temporal Silhouette: Validation of Stream Clustering Robust to Concept Drift* (under review...)

*Jun., 2023*

If you use Temporal Silhouette and/or the experiments described here, please cite the repository and the paper (when published).


### 0. Requirements

Experiments require the following general-purpose Python packages:

- numpy
- pandas
- scipy
- sklearn
- glob
- re
- sys

Stream clustering algoriths used are:

- Birch from sklearn
- DenStream from [https://github.com/issamemari/DenStream](https://github.com/issamemari/DenStream)
- CluStream and Streamkm from [https://github.com/giuliano-oliveira/clusopt_core](https://github.com/giuliano-oliveira/clusopt_core)

Incremental cvis are obtained from the package:

- cvi

### 1. Temporal Silhouette Index (TS)

Use TS in your codes by importing the "tempsil" function:

        from TSindex import tempsil

        tempsil(t,x,l,s,kn,c)
        # INPUTS
        # t: 1D-array with timestamps
        # x: 2D-array with data vectors
        # l: 1D-array with labels
        # s: window-size of the simple-moving-average (SMA), default s=200 
        # kn: number-of-neighbors of other clusters for calculating beta, default kn=200
        # c: sigma parameter to weight the penalization over contextual outliers [0...1], default=1
        #
        # OUTPUTS
        # k: 1D-array with cluster-labels
        # ts2: 1D-array with quadratic cluster temporal silhouettes
        # TS: global Temporal Silhuette

For a simple example, open a terminal and run:

        $ python3 toy_tests.py c

where "c" is a real number for the TS sigma parameter. In the paper, Section 4.6 uses this script for the examples, with c=1 and c=0.

### 2. Datasets

**Synthetic datasets** are located within the [dataS] folder. They are also publicly available in **Mendeley**: 

“Data for Evaluation of Stream Data Analysis Algorithms”. Mendeley Data, V1, doi: 10.17632/c43kr4t7h8.1, [https://data.mendeley.com/datasets/c43kr4t7h8/1](https://data.mendeley.com/datasets/c43kr4t7h8/1)

The *clean* (without outliers) versions and the versions with *out-of-phase* outliers are created online during the runing of the experiments

**Real datasets** are located within the [dataR] folder (they are preprocessed versions in arff format). Original data has been obtained from:

- kaggle - Retail and Retailers Sales Time Series Collection: 
[https://www.kaggle.com/datasets/census/retail-and-retailers-sales-time-series-collection](https://www.kaggle.com/datasets/census/retail-and-retailers-sales-time-series-collection)

- Gapminder data repository - Fertility vs Income:
[https://www.gapminder.org/data/](https://www.gapminder.org/data/)

**Stationary datasets** are located within the [dataT] folder (they are preprocessed versions in arff format). Original data has been obtained from:

- Clustering Basic Benchmark of the University of Eastern Finland (Fränti and Sieranoja, 2018):
[https://cs.joensuu.fi/sipu/datasets/](https://cs.joensuu.fi/sipu/datasets/)

    - *s1* (Fränti and Virmajoki, 2006)
    - *unbalance2b* (based on the *umbalance2* dataset (Rezaei and Fränti, 2020) 
    - *d64* (Fränti et al, 2006). 

- The *noise* dataset has been created with the MDCgen tool (Iglesias et al, 2019):
[https://github.com/CN-TU/mdcgen-matlab](https://github.com/CN-TU/mdcgen-matlab)

- Fränti P, Sieranoja S (2018) K-means properties on six clustering benchmark datasets. Applied Intelligence 48(12):4743–4759
- Fränti P, Virmajoki O (2006) Iterative shrinking method for clustering problems. Pattern Recognition 39(5):761–765
- Rezaei M, Fränti P (2020) Can the number of clusters be determined by external indices? IEEE Access 8:89,239–89,257
- Fränti P, Virmajoki O, Hautamäki V (2006) Fast agglomerative clustering using a k-nearest neighbor graph. IEEE Trans on Pattern Analysis and Machine Intelligence 28(11):1875–1881
- Iglesias F, Zseby T, Ferreira D, Zimek A (2019) Mdcgen: Multidimensional dataset generator for clustering. Jour of Classification 36(3):599–618


### 3. Replicating experiments

For the synthetic data, open a terminal in the current folder. Run:

        $ python3 run_analysis_synthetic.py

A *results_synthetic.csv* file will be created in the [results/] folder. 

For the real data, open a terminal in the current folder. Run:

        $ python3 run_analysis_real.py

Three files (*fert_vs_gdp_labels.csv*, *results_real.csv* and *retail_labels.csv*) will be created in the [results/] folder.

**Warning!** The [results/] folder already contain files with the results published in the paper. Remove, rename or copy these files in a different folder before running the scripts; otherwise, results will be appended in the synthetic case and overwriten in the real case.

### 4. Comparing performances of CVIs/iCVIs of stream clustering performances over datasets with concept drift 

For a comparison with the Ground Truth (GT) as benchmark: 

        $ python3 utils/compare_results.py results/results_synthetic.csv gt

For a comparison with the best clustering according to external validation (AMI) as benchmark: 

        $ python3 utils/compare_results.py results/results_synthetic.csv ami

### 5. Plotting peformances of stream clustering for real data cases

To visualize the performances of the stream clustering algorithms used in real data cases, run:

        $ python3 utils/plots_retail.py dataR/retail.arff results/retail_labels.csv 

        $ python3 utils/plots_gapminder.py dataR/fert_vs_gdp.arff results/fert_vs_gdp_labels.csv


### 6. Comparing performances of CVIs/iCVIs in stationary datasets 

Open a terminal in the current folder. Run:

        $ python3 run_stationary.py

A *results_stationary.csv* file will be created in the [results/] folder. You can plot these results as heatmaps by running:

        $ python3 utils/plots_stationary.py results/results_stationary.csv

It will create the following figures in the [results/] folder: *noise.pdf*, *unbalanced2b.pdf*, *s1.pdf*, *d64.pdf*.

### 7. Sensitivity analysis on TS w and k parameters 

Open a terminal in the current folder. Run:

        $ python3 run_TS_stability.py

A *results_TSstability.csv* file will be created in the [results/] folder. You can plot these results as boxplots by running:

        $ python3 utils/plots_TSstab.py results/results_TSstability.csv

It will create the *TS_stab.pdf* figure in the [results/] folder.
