# Pharm-AutoML
Automatic machine learning pipeline for scientists.

# Get started

To install the running environment
```
conda env create -n autoMLenv -f environment.yml python=3.7
conda activate autoMLenv
```
We developed a software to accelerate the establishment of state-of-the-art machine learning classification models and extract scientific insights from these models. The first software is a python library and aimed at quickly establishing baseline model for the python coders. 

1. Use in jupyter notebook (available for all operating system):
  classification task of UCI heart dataset [(notebook)](https://github.com/gengbo-genentech/Pharm-AutoML/tree/main/AutoML_package/AutoML_notebook_breast_imputation.ipynb)

2. Use in linux terminal:
```
conda env create -n autoMLenv -f environment.yml python=3.7
source activate autoMLenv
cd ./AutoML_package/src_autoML
# Create the result folder you wanted to save all results
# please adjust the pharm-automl arguments in shell file (run_heart.sh), more details are listed in the following section (arguments of pharm-automl)
bash run_heart.sh
```

# Input dataset example
All the example data is saved in [here](https://github.com/gengbo-genentech/Pharm-AutoML/tree/main/AutoML_package/src_autoML/examples/data)

x: DataFrame, patients in rows and features in columns, missing values and categorical variables are allowed
  * eg: all the columns in the following dataframe except predicting target (column 'target' in heart dataset and column 'Biopsy' in cervical cancer dataset)

y: DataFrame, patients in rows and features in columns, binary target, missing values are allowed
  * eg: predicting target

<div align=center><img src="/imgs/dataExample.png" width="100%"></div>

more detail about heart dataset is [here](https://www.kaggle.com/ronitf/heart-disease-uci)

<div align=center><img src="/imgs/risk_factors_cervical_cancer1.png" width="100%"></div>
<div align=center><img src="/imgs/risk_factors_cervical_cancer2.png" width="100%"></div>

more detail about cervical cancer dataset is [here](https://www.kaggle.com/loveall/cervical-cancer-risk-classification)

# Methods or Algorithms used in each steps of machine learning pipeline
1. Feature selection: remove column with missing value by threshold, remove redundant features (pearson correlation)
2. Imputation: missing labeling, mean value, median value, one_hot_encoder
3. Preprocessing: standard_scaler with or without PCA
4. Classifier: all classifiers in sklearn, including support vector machine (SVC), k-nearest neighbors (KNN), random forest (RF), extra trees (ET), adaboost classifier (AB), gradient boosting classifier (GB), stochatic gradient descent classifier (SGD), xgboost classifier (XGboost)

<div align=center><img src="/imgs/flow_chart_1.jpg" width="80%"></div>

# Arguments of Pharm-AutoML
Please find code example in run_heart.sh or AutoML notebook.
* `data` [String] : The dataset need to be analyzed, data csv file directory.
* `target` [String] :  Predicting target name.
* `result_path` [String] : The path that you want to save your result, please create this result folder before using it.
* `allowMissing` [Boolean] : Allow missing value in classifiers, if allowMissing is false, missing value will be imputed; if allowMissing is true, add_nan_label, impute_category_strategy, impute_numerical_strategy, PCA_featureSelection_orNot, PCA_feature_num will be ignored.
* `drop_features` [String] : Features that need to remove, put none if there is no feature to be removed.
* `categorical_features` [String] : Categorical variables require one hot encoding, put none if there is no ordinal features. 
* `missing_threshold_rows` [Float] : If the missing fraction of a feature is higher than missing_threshold_rows, this feature will be removed.
* `missing_threshold_columns` [Float] : If the missing fraction of a feature is higher than missing_threshold_columns, this feature will be removed. 
* `correlation_threshold` [Float] : Remove the second feature if the correlation of two features is higher than correlation_threshold. 
* `check_diff_categorical_threshold` [Float] : Remove the categorical features which the ratio of the number of unique value subject over the number of total subject is higher than check_diff_categorical_threshold. 
* `add_nan_label` [Boolean] : Generate new feature columns by labeling the nan subjects. 
* `impute_category_strategy` [String] : The strategy to impute category features, options: {"most_frequent", "constant"}. 
* `impute_numerical_strategy` [String] : The strategy to impute numerical features
{"mean", "median"}.
* `PCA_featureReduction_orNot` [Boolean] : If use PCA feature reduction,PCA_featureReduction_orNot is true. 
* `PCA_featureReduction_num` [Integer] : Number of PCA features. 
* `clfs` [String] : The classifiers to use, eg: "SVC,KNN,RF,ET,AB,GB,SGD,XGboost,LRC". 
* `n_fold` [Integer] : Number of folds for cross validation. 
* `n_repeats` [Integer] : If n_repeats is None, non-repeated n fold cross validation will be used, eg: n_repeats=8. 
* `useDropColFeaImport` [Boolean] : If true, use drop column feature importances method. 