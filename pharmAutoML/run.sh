# https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29
# dataset directory
# examples/data/risk_factors_cervical_cancer.csv
# the current directory is file src_autoML
# the dataset need to be analyzed, data csv file directory
data=/Users/liug28/Downloads/PharmAutoML-mlflow/data/risk_factors_cervical_cancer.csv
# target name, your prediction target
target=Biopsy
# the path that you want to save your result, please create this result folder before using it
result_path=automl_result
# allow missing value in classifiers, 
# if allowMissing is false, missing value will be imputed
# if allowMissing is true, add_nan_label, impute_category_strategy, impute_numerical_strategy, PCA_featureSelection_orNot, PCA_feature_num will be ignored
impute_missing_mode=true
# if tuning_mode is true, the automl will implement all possible parameter grids set by user and output all results
# if tuning_mode is false, the automl will search across all possible parameters and return the result of the best performed parameter
tuning_mode=true
# the loss functions for binary classification are neg_log_loss, accuracy, roc_auc, f1, average_precision, precision, recall
# the loss functions for multi classification are neg_log_loss, accuracy
loss_function="roc_auc"
##################################################################################
# preprocessing parameters
##################################################################################
# features that need to remove, put none if there is no feature to be removed
drop_features=None
# if the missing fraction of a feature is higher than this missing threshold, this feature will be removed
# missing_threshold = number of missing subjects/number of total subjects in that column
missing_threshold_rows="0.2"
missing_threshold_columns="0.2"
# remove the second feature if the correlation of two features is higher than correlation_threshold
correlation_threshold="0.98"
##################################################################################
# imputation parameters
##################################################################################
# categorical variables require one hot encoding, put none if there is no ordinal features
categorical_features=None
# remove the categorical features which the ratio of the number of unique value subject 
# over the number of total subject is higher than check_diff_categorical_threshold
check_diff_categorical_threshold="0.1"
# the strategy to impute category features
# "most_frequent" or "constant"
impute_category_strategy="most_frequent"
# the strategy to impute numerical features
# "mean", "median"
impute_numerical_strategy="mean"
# generate new feature columns by labeling the nan subjects
add_nan_label="false"
##################################################################################
# feature selection parameters
##################################################################################
feature_selection_method=None
##################################################################################
# train test split parameters
##################################################################################
# random seed of train and test data splitting
tts_random_seed="0"
# the proportion size of test data
test_size="0"
##################################################################################
# classifier
##################################################################################
# if use PCA feature reduction, PCA_featureReduction_orNot=true
PCA="false"
PCA_feature_num="0"
# the classifiers to use, 
# eg: "SVC,KNN,RF,ET,AB,GB,SGD,XGboost,LRC"
clfs="KNN,RF,ET,AB,GB,SGD,XGboost,LRC"
# number of folds for cross validation
n_fold=5
# if n_repeats is None, non-repeated n fold cross validation will be used, eg: n_repeats=8
n_repeats=None
# if true, use drop column feature importances method, this method takes very long time to compute
useDropColFeaImport=false
# impute missing
python autoML_pipeline.py --data $data --target $target --result_path $result_path --categorical_features $categorical_features --impute_missing_mode $impute_missing_mode --tuning_mode $tuning_mode --loss_function $loss_function --drop_features $drop_features --missing_threshold_rows $missing_threshold_rows --missing_threshold_columns $missing_threshold_columns --correlation_threshold $correlation_threshold --check_diff_categorical_threshold $check_diff_categorical_threshold --impute_category_strategy $impute_category_strategy --impute_numerical_strategy $impute_numerical_strategy --add_nan_label $add_nan_label --feature_selection_method $feature_selection_method --tts_random_seed $tts_random_seed --test_size $test_size --PCA $PCA --PCA_feature_num $PCA_feature_num --clfs $clfs --n_fold $n_fold --n_repeats $n_repeats --useDropColFeaImport $useDropColFeaImport













