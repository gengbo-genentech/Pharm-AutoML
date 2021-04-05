import pandas as pd
import numpy as np
import os
import sys
from sklearn import linear_model
from sklearn.metrics import classification_report, roc_auc_score
import autoML_util
import imputer
import modelSelector_imputeMissing
import modelSelector_allowMissing
import resultAnalysis
from sklearn.model_selection import train_test_split
from featurePreprocessors import FeaturePreprocessors
from imputer import Imputer
from sklearn.model_selection import StratifiedShuffleSplit
import statistics
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

class PharmAutoML():
    """
    PharmAutoML main class, user can call pharm automl pipeline with and without missing values from here
    Attributes:
        X: input of auto ml classifier, a pandas dataframe with features in columns and subjects in row
        y: the predicting target of auto ml classifier, a pandas series
        random_seed: a random integer which controll the data split
    """
    def __init__(self, X, y, random_seed = np.random.RandomState(0)):
        self.X = X
        self.y = y
        self.num_unique_y = len(y.unique())
        self.random_seed = random_seed

        # preprocessing parameter grid
        self.para_grid_imputeMissing = None
        self.result_path_imputeMissing = None
        self.result_logger_imputeMissing = None
        self.useDropColFeaImport = None
        self.handler_imputeMissing = None
        self.featureDimReductionNumber = None
        self.clfs = None
        self.n_fold = None
        self.n_repeats = None
        self.result = None
        # missing allow parameters
        self.para_grid_missingAllow = None
        self.clfs_allowMissing = None
        self.useDropColFeaImport_allowMissing = None
        self.result_path_missingAllow = None
        self.result_logger_missingAllow = None
        self.handler_missingAllow = None
        self.allowMissingResult = None

        self.isDisplay = False
        self.test_size = 0.2

    def setUp_resultFolder_preproGrid_imputeMissing(self, result_path, preprocessing_paras):
        """set up result directory, initialize result path, preprocessing parameter grid and logger file
        Args:
            result_path: str
            preprocessing_paras: dictionary of preprocessing parameters
        Returns:
            none
        """
        self.result_path_imputeMissing = result_path
        autoML_util.setup_result_dir(result_path)
        print('preprocessing parameter dictionary is')
        print(preprocessing_paras)
        self.para_grid_imputeMissing = autoML_util.get_grid_complement_missing_threshold(self.X, preprocessing_paras)
        print('parameter grid is')
        print(self.para_grid_imputeMissing)
        # set up temp directory to save preprocessed data
        autoML_util.setup_temp_dir(len(self.para_grid_imputeMissing), self.result_path_imputeMissing)
        result_log_path = self.result_path_imputeMissing + '/resultImputeMissing.log'
        # set up logger
        result_logger, handler = autoML_util.setup_logger('logger', result_log_path)
        self.result_logger_imputeMissing=result_logger
        self.handler_imputeMissing = handler

    def feature_prepro_imputeMissing(self):
        """
        feature preprocessing pipeline, and save the preprocessed data into save folder with csv format
        """
        for i in range (len(self.para_grid_imputeMissing)):
            self.result_logger_imputeMissing.info('\n')
            self.result_logger_imputeMissing.info('For each feature preprocessing loop')
            self.result_logger_imputeMissing.info(i)
            self.result_logger_imputeMissing.info(self.para_grid_imputeMissing[i])
            param = self.para_grid_imputeMissing[i]
            print ('param')
            print (param)
            fp = FeaturePreprocessors(self.X, self.y, self.result_path_imputeMissing, i, drop_features = param['drop_features'])
            X_important, y_important = fp.feature_preprocessor_main(selection_params = {'missing_threshold_rows': param['missing_threshold_rows'], 
                'missing_threshold_columns': param['missing_threshold_columns'], 'correlation_threshold': param['correlation_threshold']})
            print ('imputation for training dataset')
            self.result_logger_imputeMissing.info('imputation for training dataset')
            imp = Imputer(X_important, grid_index = i, 
                    manual_categ_features=param['categorical_features'], logger = self.result_logger_imputeMissing, 
                    allDiff_threshold = param['check_diff_categorical_threshold'])
            if param['missing_threshold_columns'] != 0:
                X_important_imputed = imp.impute_missings_pipeline(num_strategy = param['impute_numerical_strategy'], 
                    cat_strategy = param['impute_category_strategy'], 
                    add_nan_label=param['add_nan_label'])
            else:
                X_important_imputed = imp.impute_allowMissing_pipeline()

            if self.num_unique_y == 2 and pd.api.types.is_string_dtype(y_important):
                y_one_hotted_df = pd.get_dummies(y_important)
                y_important = y_one_hotted_df.iloc[:,:1]

            X_train_important, X_test_important, y_train, y_test = train_test_split(X_important_imputed, y_important, random_state=self.random_seed, test_size=self.test_size, stratify=y_important)
            save_folder = self.result_path_imputeMissing + '/' + str(i)

            X_train_important.to_csv(save_folder + '/X_train_important.csv')
            y_train.to_csv(save_folder + '/y_train.csv')
            # save X test
            X_test_important.to_csv(save_folder + '/X_test_important.csv')
            y_test.to_csv(save_folder + '/y_test.csv')

    def hyperopt_pipeline_imputeMissing(self, modelSelector_imputeMissing):
        """
        classification pipeline with impution, including searching optimal hyperparameter and generate prediction result of testing data
        Args:
            modelSelector_imputeMissing: object of class modelSelector_imputeMissing
        Returns:
            result: dict, result[fold][para_index] = bm, preds, preds_probs, auc_score, pr_score
        """
        result = dict()
        for para_index in range(len(self.para_grid_imputeMissing)):
            result['para_index_'+str(para_index)]=dict()
            print ('for each preprocessing loop the classifier tuning record is')
            print (para_index)
            if self.result_logger_imputeMissing != None:
                self.result_logger_imputeMissing.info('for each preprocessing loop the classifier tuning record is')
                self.result_logger_imputeMissing.info(para_index)
            X_train, y_train, X_test, y_test= autoML_util.get_preprocessed_data(self.result_path_imputeMissing, para_index)
            bm = modelSelector_imputeMissing.all_classifier_nfold(X_train, y_train)
            result['para_index_'+str(para_index)]['bm'] = bm

        for para_index in range(len(self.para_grid_imputeMissing)):
            X_train, y_train, X_test, y_test= autoML_util.get_preprocessed_data(self.result_path_imputeMissing, para_index)
            temp_bm = result['para_index_'+str(para_index)]['bm']

            refit_bm, y_classes_n_fold, y_test_n_fold, preds, preds_probs = modelSelector_imputeMissing.bm_results(X_train, y_train, temp_bm)
            result['para_index_'+str(para_index)]['bm'] = refit_bm
            result['para_index_'+str(para_index)]['y_classes_n_fold'] = y_classes_n_fold
            result['para_index_'+str(para_index)]['y_test_n_fold'] = y_test_n_fold
            result['para_index_'+str(para_index)]['preds'] = preds
            result['para_index_'+str(para_index)]['preds_probs'] = preds_probs
        self.result = result
        autoML_util.save_model(len(self.para_grid_imputeMissing), self.clfs, self.result, self.result_path_imputeMissing)
        return result

    def clf_selection(self, clfs, n_fold, n_repeats, isFeatureReduction=False, featureDimReductionNumber = None, useDropColFeaImport = None):
        """calculate classifier prediction result for different metrics and generate analysis plots
        Args:
            clfs: list of classifiers, ['SVC', 'KNN', 'RF', 'ET', 'AB', 'GB', 'SGD', 'XGboost']
            n_fold: int
            n_repeats: int
            isFeatureReduction: true when using PCA feature reduction method
            featureDimReductionNumber: the number of reduced feature by PCA
            useDropColFeaImport: boolean, use feature drop method or not
        Returns:
            result: result[fold][para_index] = bm, preds, preds_probs, auc_score, pr_score
            metrics: dictionary of sorted result metrics
        """
        self.clfs = clfs
        self.n_fold = n_fold
        self.n_repeats = n_repeats
        self.isFeatureReduction = isFeatureReduction
        self.featureDimReductionNumber = featureDimReductionNumber
        self.useDropColFeaImport = useDropColFeaImport
        # initialize 
        model_imputeMissing = modelSelector_imputeMissing.ModelSelector_imputeMissing(len(self.para_grid_imputeMissing),
            self.clfs,
            self.result_path_imputeMissing,
            self.result_logger_imputeMissing,
            self.n_fold,
            self.n_repeats,
            max_evals = 10,
            isFeatureReduction = self.isFeatureReduction,
            featureDimReductionNumber = self.featureDimReductionNumber)
        self.result = self.hyperopt_pipeline_imputeMissing(model_imputeMissing)

        prepro_average_roc, prepro_stdev_roc, metrics = resultAnalysis.get_roc_curve_allFolds(self.result, 
            self.para_grid_imputeMissing, 
            self.clfs, 
            self.n_fold, 
            self.result_path_imputeMissing, 
            self.result_logger_imputeMissing)
        resultAnalysis.get_sorted_model_each_grid(self.para_grid_imputeMissing, 
            self.n_fold, 
            self.clfs, 
            self.result, 
            self.result_path_imputeMissing, 
            self.isFeatureReduction, 
            self.result_logger_imputeMissing, 
            prepro_average_roc, 
            prepro_stdev_roc, 
            self.useDropColFeaImport)
        self.result_logger_imputeMissing.removeHandler(self.handler_imputeMissing)
        return self.result, metrics

    def AutoML_imputeMissing(self, result_path, preprocessing_paras, classifier_paras):
        """
        pharm auto ml pipeline with missing imputation
        Args:
            result_path: string, directory to save results
            preprocessing_paras: dictionary, data preprocessing parameters
            classifier_paras: dictionary, data preprocessing parameters
        Returns:
            result: list of dict
        """
        self.setUp_resultFolder_preproGrid_imputeMissing(result_path, preprocessing_paras)
        self.feature_prepro_imputeMissing()
        result_imputeMissing = self.clf_selection(classifier_paras['clfs'],
            classifier_paras['n_fold'],
            classifier_paras['n_repeats'],
            classifier_paras['PCA_featureReduction_orNot'],
            classifier_paras['PCA_featureReduction_num'],
            classifier_paras['useDropColFeaImport'])
        return result_imputeMissing

######################################## allow missing automl start from here ########################################
    def setUp_resultFold_preproGrid_allowMissing(self, result_path, preprocessing_paras):
        """set up result directory, initialize result path, preprocessing parameter grid, logger
        Args:
            result_path: str
            config_file: str
        """
        self.result_path_missingAllow = result_path
        autoML_util.setup_result_dir(self.result_path_missingAllow)
        self.para_grid_missingAllow = autoML_util.get_grid_allow_missing(self.X, preprocessing_paras)
        autoML_util.setup_temp_dir(len(self.para_grid_missingAllow), self.result_path_missingAllow)
        result_log_path = self.result_path_missingAllow + '/resultMissingAllow.log'
        result_logger, handler = autoML_util.setup_logger('logger', result_log_path)
        self.result_logger_missingAllow=result_logger
        self.handler_missingAllow = handler

    def feature_prepro_allowMissing(self):
        """
        feature preprocessing pipeline without imputing missings, and save the preprocessed data into save folder with csv format
        """
        for i in range (len(self.para_grid_missingAllow)):
            self.result_logger_missingAllow.info('\n')
            self.result_logger_missingAllow.info('For each feature preprocessing loop')
            self.result_logger_missingAllow.info(i)
            self.result_logger_missingAllow.info(self.para_grid_missingAllow[i])
            param = self.para_grid_missingAllow[i]
            fp = FeaturePreprocessors(self.X, self.y, self.result_path_missingAllow, i, drop_features = param['drop_features'])
            X_important, y_important = fp.feature_preprocessor_main(selection_params = {'correlation_threshold': param['correlation_threshold']})
            print ('imputation for training dataset')
            self.result_logger_missingAllow.info('imputation for training dataset')
            imp = Imputer(X_important, grid_index = i, 
                    manual_categ_features=param['categorical_features'], logger = self.result_logger_missingAllow, 
                    allDiff_threshold = param['check_diff_categorical_threshold'])
            X_important_imputed = imp.impute_allowMissings_pipeline()

            if self.num_unique_y == 2 and pd.api.types.is_string_dtype(y_important):
                y_one_hotted_df = pd.get_dummies(y_important)
                y_important = y_one_hotted_df.iloc[:,:1]

            X_train_important, X_test_important, y_train, y_test = train_test_split(X_important_imputed, y_important, random_state=self.random_seed, test_size=self.test_size, stratify=y_important)
            save_fold = self.result_path_missingAllow + '/' + str(i)

            X_train_important.to_csv(save_fold + '/X_train_important.csv')
            y_train.to_csv(save_fold + '/y_train.csv')
            # save X test
            X_test_important.to_csv(save_fold + '/X_test_important.csv')
            y_test.to_csv(save_fold + '/y_test.csv')

    def hyperopt_pipeline_allowMissing(self, model_allowMissing):
        """
        classification pipeline without impution, including searching optimal hyperparameter and generate prediction result of testing data
        Args:
            modelSelector_allowMissing: object of class modelSelector_allowMissing
        Returns:
            result: dict, result[fold][para_index] = bm, preds, preds_probs, auc_score, pr_score
        """
        result = dict()
        for para_index in range(len(self.para_grid_missingAllow)):
            result['para_index_'+str(para_index)]=dict()
            print ('for each preprocessing loop the allow missing classifier tuning record is')
            print (para_index)
            self.result_logger_missingAllow.info('for each preprocessing loop the allow missing classifier tuning record is')
            self.result_logger_missingAllow.info(para_index)
            X_train, y_train, X_test, y_test= autoML_util.get_preprocessed_data(self.result_path_missingAllow, para_index)
            ss = StandardScaler()
            X_train_ss = ss.fit_transform(X_train)
            X_train[:] = X_train_ss
            bm = model_allowMissing.all_classifier_nfold_allowMissing(X_train, y_train)
            result['para_index_'+str(para_index)]['bm'] = bm

        for para_index in range(len(self.para_grid_missingAllow)):
            X_train, y_train, X_test, y_test= autoML_util.get_preprocessed_data(self.result_path_missingAllow, para_index)
            ss = StandardScaler()
            X_train_ss = ss.fit_transform(X_train)
            X_train[:] = X_train_ss
            temp_bm = result['para_index_'+str(para_index)]['bm']
            refit_bm, y_classes_n_fold, y_test_n_fold, preds, preds_probs = model_allowMissing.bm_results_allowMissing(X_train, y_train, temp_bm)
            result['para_index_'+str(para_index)]['bm'] = refit_bm
            result['para_index_'+str(para_index)]['y_classes_n_fold'] = y_classes_n_fold
            result['para_index_'+str(para_index)]['y_test_n_fold'] = y_test_n_fold
            result['para_index_'+str(para_index)]['preds'] = preds
            result['para_index_'+str(para_index)]['preds_probs'] = preds_probs
        self.allowMissingResult = result
        autoML_util.save_model(len(self.para_grid_missingAllow), self.clfs_allowMissing, self.allowMissingResult, self.result_path_missingAllow)
        return result

    def clf_allowMissing(self, n_fold, n_repeats, useDropColFeaImport, clfs=['XGboost']):
        """calculate classifier prediction result for different metrics and generate analysis plots
        Args:
            n_fold: int
            n_repeats: int
            clfs: only support ['XGboost'] so far
        Returns:
            result: result[fold][para_index] = bm, preds, preds_probs, auc_score, pr_score
        """
        self.clfs_allowMissing = clfs
        self.n_fold = n_fold
        self.n_repeats = n_repeats
        self.useDropColFeaImport_allowMissing = useDropColFeaImport

        model_allowMissing = modelSelector_allowMissing.ModelSelector_allowMissing(len(self.para_grid_missingAllow),
            self.clfs_allowMissing, 
            self.n_fold, 
            self.n_repeats, 
            self.result_path_missingAllow,
            self.result_logger_missingAllow, 
            max_evals = 10)
        self.allowMissingResult = self.hyperopt_pipeline_allowMissing(model_allowMissing)

        prepro_average_roc, prepro_stdev_roc, metrics = resultAnalysis.get_roc_curve_allFolds(self.allowMissingResult, 
            self.para_grid_missingAllow, 
            self.clfs_allowMissing, 
            self.n_fold, 
            self.result_path_missingAllow, 
            self.result_logger_missingAllow)
        resultAnalysis.get_sorted_model_each_grid_allowMissing(self.para_grid_missingAllow, 
            self.n_fold, 
            self.clfs_allowMissing, 
            self.allowMissingResult, 
            self.result_path_missingAllow, 
            self.result_logger_missingAllow, 
            prepro_average_roc, 
            prepro_stdev_roc, 
            self.useDropColFeaImport_allowMissing)
        self.result_logger_missingAllow.removeHandler(self.handler_missingAllow)
        return self.allowMissingResult

    def AutoML_allowMissing(self, result_path, preprocessing_paras, classifier_paras):
        """
        pharm auto ml pipeline for allow missing
        Args:
            result_path: string, directory to save results
            preprocessing_paras: dictionary, data preprocessing parameters
            classifier_paras: dictionary, data preprocessing parameters
        Returns:
            result: list of dict
        """
        self.setUp_resultFold_preproGrid_allowMissing(result_path, preprocessing_paras)
        self.feature_prepro_allowMissing()
        result_missingAllow = self.clf_allowMissing(classifier_paras['n_fold'], classifier_paras['n_repeats'], classifier_paras['useDropColFeaImport'])
        return result_missingAllow

if __name__ == "__main__":
    from args import args
    cwd = os.getcwd()
    print ('current working directory is')
    print (cwd)
    print (args.clfs)
# read data
    if args.data != None:
        allData=pd.read_csv(args.data)
        y = allData[args.target]
        x = allData.drop([args.target], axis=1)
    elif args.x_dir != None and args.y_dir != None:
        x=pd.read_csv(args.x_dir)
        y=pd.read_csv(args.y_dir)[args.target]
    elif args.train_dir!=None and args.test_dir != None:
        train = pd.read_csv(args.train_dir)
        test = pd.read_csv(args.test_dir)
        allData = pd.concat([train, test])
        allData=allData.set_index('PassengerId')
        y = allData[args.target]
        x = allData.drop([args.target], axis=1)
    preprocessing_paras = {}
    # features that need to drop, put [] if there is no feature to be removed
    if args.drop_features == None:
        preprocessing_paras['drop_features']=[[]]
    else:
        preprocessing_paras['drop_features']=[args.drop_features]
    # categorical variables require one hot encoding, put [] if there is no ordinal features
    if args.categorical_features == None:
        preprocessing_paras['categorical_features']=[[]]
    else:
        preprocessing_paras['categorical_features']=[args.categorical_features]   
    # if the missing fraction of a row or column is higher than missing threshold, this feature will be removed
    preprocessing_paras['missing_threshold_rows']=args.missing_threshold_rows
    preprocessing_paras['missing_threshold_columns']=args.missing_threshold_columns
    # remove the second feature if the correlation of two features is higher than correlation_threshold
    preprocessing_paras['correlation_threshold']=args.correlation_threshold
    # remove the categorical features which the ratio of the number of 
    # unique value subject over the number of total subject is higher than 
    # check_diff_categorical_threshold
    preprocessing_paras['check_diff_categorical_threshold']=args.check_diff_categorical_threshold
    # generate new feature columns by labeling the nan subjects
    preprocessing_paras['add_nan_label']=args.add_nan_label
    # the strategy to impute category features "most_frequent" or "constant"
    preprocessing_paras['impute_category_strategy']=args.impute_category_strategy
    # the strategy to impute numerical features "mean" or "median"
    preprocessing_paras['impute_numerical_strategy']=args.impute_numerical_strategy
    print ('preprocessing param')
    print (preprocessing_paras)
    # initialize classifier parameters as a dictionary
    classifier_paras = {}
    # if use PCA feature reduction, PCA_featureReduction_orNot=true
    classifier_paras['PCA_featureReduction_orNot']=args.PCA_featureReduction_orNot
    classifier_paras['PCA_featureReduction_num']=args.PCA_featureReduction_num
    # the classifiers will use in Pharm-AutoML 
    classifier_paras['clfs']=args.clfs
    # number of folds for cross validation
    classifier_paras['n_fold']=args.n_fold
    # if n_repeats is None, non-repeated n fold cross validation will be used n_repeats=None
    classifier_paras['n_repeats']=args.n_repeats
    # if true, use drop column feature importances method, this method takes very long time to compute
    classifier_paras['useDropColFeaImport']=args.useDropColFeaImport
    print ('classifier param')
    print (classifier_paras)

    if args.result_path == None:
        result_path = os.path.join(cwd, 'resultImputeMissing')
    elif args.result_path != None and args.allowMissing == False:
        result_path = os.path.join(args.result_path, 'resultImputeMissing')
    elif args.result_path != None and args.allowMissing == True:
        result_path = os.path.join(args.result_path, 'resultAllowMissing')
    print ('the result saved at')
    print (result_path)

    aml = PharmAutoML(x, y)
    if args.allowMissing == True:
        result_imputeMissing = aml.AutoML_allowMissing(result_path, preprocessing_paras, classifier_paras)
    elif args.allowMissing == False:
        result_missingAllow = aml.AutoML_imputeMissing(result_path, preprocessing_paras, classifier_paras)
