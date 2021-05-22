# Pharm-AutoML
Automatic machine learning pipeline for scientists. PharmAutoML is a python package to accelerate the establishment of state-of-the-art machine learning classification models and extract scientific insights from these models.

If you found this library useful in your research, please consider citing [this paper](https://pubmed.ncbi.nlm.nih.gov/33793093/).
```
Gengbo Liu, Dan Lu, and James Lu. "Pharm‐AutoML: an open‐source, end‐to‐end automated machine learning package for clinical outcome prediction." CPT: Pharmacometrics & Systems Pharmacology (2021 Apr 1).
```

# Get started
To install the running environment
```
conda env create -n autoMLenv python=3.7
conda activate autoMLenv
```

1. Install pharmAutoML and use as a python package
```
cd PharmAutoML-mlflow
python setup.py install
```
Now you can use autoML_pipeline by simply importing package in python script
```
from pharmAutoML import autoML_pipeline
```

2. Use in linux terminal:
```
cd pharmAutoML
bash run.sh
```

# Example
x: DataFrame, patients in rows and features in columns, missing values and categorical variables are allowed
  * eg: all the columns in the following dataframe except predicting target (column 'Biopsy' in cervical cancer dataset)

y: DataFrame, patients in rows and features in columns, binary target, missing values are allowed
  * eg: predicting target

<div align=center><img src="/imgs/risk_factors_cervical_cancer1.png" width="100%"></div>
<div align=center><img src="/imgs/risk_factors_cervical_cancer2.png" width="100%"></div>

more detail about cervical cancer dataset is [here](https://www.kaggle.com/loveall/cervical-cancer-risk-classification)

# Methods or Algorithms used in each steps of machine learning pipeline
1. Feature selection: remove column with missing value by threshold, remove redundant features (pearson correlation)
2. Imputation: missing labeling, mean value, median value, one_hot_encoder
3. Preprocessing: standard_scaler with or without PCA
4. Classifier: all classifiers in sklearn, including support vector machine (SVC), k-nearest neighbors (KNN), random forest (RF), extra trees (ET), adaboost classifier (AB), gradient boosting classifier (GB), stochatic gradient descent classifier (SGD), xgboost classifier (XGboost)

<div align=center><img src="/imgs/flow_chart_1.jpg" width="80%"></div>

# Option 1: Use Auto_ML library
1. Use in jupyter notebook (available for all operating system):

  classification task of UCI cervical cancer dataset [(notebook)](https://github.com/gengbo-genentech/Pharm-AutoML/blob/main/AutoML_notebook_mlflow.ipynb)

2. Use in terminal (mac os):
```
cd ./AutoML_package/src_autoML
bash run.sh
```

# Use mlflow to compare over different experiments

To get a mlflow type of user interface, go to the result folder located at and use the following command in terminal
```
source activate autoMLenv
cd Pharm-AutoML
mlflow server --backend-store-uri automl_result
```
Terminal will return a server uri (http), please open it in browser.
<div align=center><img src="/imgs/mlflow_ui.png" width="100%"></div>

# Arguments of API in Pharm-AutoML
This [(section)](https://github.com/gengbo-genentech/Pharm-AutoML/blob/main/pharmAutoML/default_parameters.json) defines the default parameter used in function AutoML.
Please find code example in run.sh or [(Jupyter notebook)](https://github.com/gengbo-genentech/Pharm-AutoML/blob/main/AutoML_notebook_mlflow.ipynb).
* `data` [String] : The dataset need to be analyzed, data csv file directory.
* `target` [String] :  Predicting target name.
* `result_path` [String] : The path that you want to save your result, please create this result folder before using it.
* `impute_missing_mode` [Boolean] : Allow missing value in classifiers, if allowMissing is false, missing value will be imputed; if allowMissing is true, add_nan_label, impute_category_strategy, impute_numerical_strategy, PCA_featureSelection_orNot, PCA_feature_num will be ignored.
* `tuning_mode` [Boolean] : if tuning_mode is true, the automl will implement all possible parameter grids set by user and output all results; if tuning_mode is false, the automl will search across all possible parameters and return the result of the best performed parameter
* `loss_function` [String]: the loss functions for binary classification are neg_log_loss, accuracy, roc_auc, f1, average_precision, precision, recall; the loss functions for multi classification are neg_log_loss, accuracy
* `drop_features` [String] : Features that need to remove, put none if there is no feature to be removed.
* `categorical_features` [String] : Categorical variables require one hot encoding, put none if there is no ordinal features. 
* `missing_threshold_rows` [Float] : If the missing fraction of a feature is higher than missing_threshold_rows, this feature will be removed.
* `missing_threshold_columns` [Float] : If the missing fraction of a feature is higher than missing_threshold_columns, this feature will be removed. 
* `correlation_threshold` [Float] : Remove the second feature if the correlation of two features is higher than correlation_threshold. 
* `check_diff_categorical_threshold` [Float] : Remove the categorical features which the ratio of the number of unique value subject over the number of total subject is higher than check_diff_categorical_threshold. 
* `impute_category_strategy` [String] : The strategy to impute category features, options: {"most_frequent", "constant"}. 
* `impute_numerical_strategy` [String] : The strategy to impute numerical features, options: {"mean", "median"}.
* `add_nan_label` [Boolean] : Generate new feature columns by labeling the nan subjects. 
* `feature_selection_method` [String] : Options include {"shapRFEcv", "RFEcv", None}
* `tts_random_seed` [Integer] : Random seed of train and test data splitting
* `test_size` [Float]: the proportion size of test size, when test_size is 0, only training and validation dataset, not test dataset
* `PCA` [Boolean] : If use PCA feature reduction,PCA_featureReduction_orNot is true. 
* `PCA_feature_num` [Integer] : Number of PCA features. 
* `clfs` [String] : The classifiers to use, eg: "SVC,KNN,RF,ET,AB,GB,SGD,XGboost,LRC". 
* `n_fold` [Integer] : Number of folds for cross validation. 
* `n_repeats` [Integer] : If n_repeats is None, non-repeated n fold cross validation will be used, eg: n_repeats=8. 
* `useDropColFeaImport` [Boolean] : If true, use drop column feature importances method.
