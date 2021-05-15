import pandas as pd
import numpy as np
import os
file_dir = os.path.dirname(__file__)
import sys
sys.path.append(file_dir)
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import classification_report, roc_auc_score
import autoML_util
import json
import modelSelector_allowMissing
import modelSelector_imputeMissing
from featurePreprocessors import FeaturePreprocessors
from imputation import Imputation
from featureSelection import FeatureSelection
from sklearn.model_selection import StratifiedShuffleSplit
import statistics
import shap
import mlflow
from hyperopt import Trials
import hyperopt
import warnings
import copy
warnings.filterwarnings("ignore")

class PharmAutoML():
    """
    PharmAutoML main class, user can call pharm automl pipeline with and without missing values from here

    Attributes:
        X: input of auto ml classifier, a pandas dataframe with features in columns and subjects in row
        y: the predicting target of auto ml classifier, a pandas series
        random_seed: a random integer which controll the data split
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num_unique_y = len(y.unique())

        self.x_prepro = None
        self.y_prepro = None
        self.x_imputed = None
        self.y_imputed = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.result_path = None        
        self.tuning_mode = None
        self.loss_function = None
        self.parameter = None
        self.multiple_option_parameter = None
        self.parameter_grid = None
        self.compare_model = False
        self.para_grid_missingAllow = None
        self.metric = None

        self.result_path_allowMissing = None
        self.ALLOW_MISSING_CLFS = ["XGboost"]
        self.OUTER_MAX_EVALS_OF_TUNING_MODE = 1
        self.OUTER_MAX_EVALS_OF_NORMAL_MODE = 9

    def preprocess(self, preprocessing_params):
        fp = FeaturePreprocessors(self.x, self.y, preprocessing_params)
        x_prepro, y_prepro = fp.feature_preprocessor_main()
        self.x_prepro = x_prepro
        self.y_prepro = y_prepro

    def imputation_imputeMissing(self, imputation_params):
        imp = Imputation(self.x_prepro, self.y_prepro, imputation_params)
        x_imputed, y_imputed = imp.impute_missings_pipeline()
        self.x_imputed = x_imputed
        self.y_imputed = y_imputed

    def feature_selection(self, feature_selection_params):
        fs = FeatureSelection(self.x_imputed, self.y_imputed, feature_selection_params)
        x_selected, y_selected = fs.feature_selection_main()
        self.x_imputed = x_selected
        self.y_imputed = y_selected

    def train_test(self, train_test_split_parameters):
        if train_test_split_parameters['test_size'] == 0:
            self.x_train = self.x_imputed
            self.y_train = self.y_imputed
        else:
            x_train, x_test, y_train, y_test = train_test_split(self.x_imputed, self.y_imputed, 
                random_state=train_test_split_parameters['tts_random_seed'], 
                test_size=train_test_split_parameters['test_size'], stratify=self.y_imputed)
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test

    def clf_imputeMissing(self, classifier_parameters, metric):
        ms = modelSelector_imputeMissing.ModelSelector_imputeMissing(self.x_train, self.y_train, classifier_parameters)
        best_model = ms.classifier_nfold(classifier_parameters["clfs"], fn_string = metric)
        ms.save_learner(best_model)
        mean_metrics, std_metrics = ms.bm_fit_train_validation_cv()
        if isinstance(self.x_test, pd.DataFrame):
            test_metrics = ms.bm_predict_test_data(self.x_test, self.y_test)
            ms.model_interpreter(self.x_test, self.y_test)
            return mean_metrics, std_metrics, test_metrics, best_model
        else:
            ms.model_interpreter(self.x_train, self.y_train)
            return mean_metrics, std_metrics, None, best_model

    def objective_imputeMissing(self, metric):
        def pipeline_imputeMissing(params):
            if self.tuning_mode:
                mlflow.log_param('impute_missing_mode', self.impute_missing_mode)
                mlflow.log_param('loss_function', self.loss_function)
                self.preprocess(params['preprocessing_parameters'])
                self.imputation_imputeMissing(params['imputation_parameters'])
                self.feature_selection(params['feature_selection_parameters'])
                mlflow.log_param('selected features', list(self.x_imputed.columns))
                self.train_test(params['train_test_split_parameters'])
                mean_metrics, std_metrics, test_metrics, best_model = self.clf_imputeMissing(params['classifier_parameters'], metric = metric)
                print ('best_model')
                print (best_model)
                for i in params.keys():
                    for j in params[i].keys():
                        if j in self.multiple_option_parameter:
                            if j == 'clfs':
                                mlflow.log_param(j, list(params[i][j].keys())[0])
                            else:
                                mlflow.log_param(j, params[i][j])
                for a in best_model.keys():
                    mlflow.log_param(a, str(best_model[a]))
                if isinstance(self.x_test, pd.DataFrame):
                    mlflow.log_metrics({**mean_metrics, **std_metrics, **test_metrics})
                else:
                    mlflow.log_metrics({**mean_metrics, **std_metrics})
                cv_metric = metric+'_cv_mean'
                return {'status': hyperopt.STATUS_OK, 'loss': -mean_metrics[cv_metric], 'best_model': best_model}
            else:
                with mlflow.start_run(nested=True) as run:
                    mlflow.log_param('impute_missing_mode', self.impute_missing_mode)
                    mlflow.log_param('loss_function', self.loss_function)
                    self.preprocess(params['preprocessing_parameters'])
                    if type(params['imputation_parameters']['categorical_features']) is tuple:
                        params['imputation_parameters']['categorical_features'] = params['imputation_parameters']['categorical_features'][0]
                    self.imputation_imputeMissing(params['imputation_parameters'])
                    self.feature_selection(params['feature_selection_parameters'])
                    mlflow.log_param('selected features', list(self.x_imputed.columns))
                    self.train_test(params['train_test_split_parameters'])
                    mean_metrics, std_metrics, test_metrics, best_model = self.clf_imputeMissing(params['classifier_parameters'], metric = metric)
                    print ('best_model')
                    print (best_model)
                    for i in params.keys():
                        for j in params[i].keys():
                            if j == 'clfs':
                                mlflow.log_param(j, list(params[i][j].keys())[0])
                            else:
                                mlflow.log_param(j, params[i][j])
                    for a in best_model.keys():
                        mlflow.log_param(a, str(best_model[a]))
                    if isinstance(self.x_test, pd.DataFrame):
                        mlflow.log_metrics({**mean_metrics, **std_metrics, **test_metrics})
                    else:
                        mlflow.log_metrics({**mean_metrics, **std_metrics})
                    cv_metric = metric+'_cv_mean'
                    return {'status': hyperopt.STATUS_OK, 'loss': -mean_metrics[cv_metric], 'best_model': best_model}
        return pipeline_imputeMissing

######################################## allow missing automl start from here ########################################
    def imputation_allowMissing(self, imputation_params):
        imp = Imputation(self.x_prepro, self.y_prepro, imputation_params)
        x_imputed, y_imputed = imp.impute_allowMissings_pipeline()
        self.x_imputed = x_imputed
        self.y_imputed = y_imputed

    def clf_allowMissing(self, classifier_parameters, metric):
        ms = modelSelector_allowMissing.ModelSelector_allowMissing(self.x_train, self.y_train, classifier_parameters)
        best_model = ms.classifier_nfold(classifier_parameters["clfs"], fn_string = metric)
        ms.save_learner(best_model)
        mean_metrics, std_metrics = ms.bm_fit_train_validation_cv()
        if isinstance(self.x_test, pd.DataFrame):
            test_metrics = ms.bm_predict_test_data(self.x_test, self.y_test)
            ms.model_interpreter(self.x_test, self.y_test)
            return mean_metrics, std_metrics, test_metrics, best_model
        else:
            ms.model_interpreter(self.x_train, self.y_train)
            return mean_metrics, std_metrics, None, best_model

    def objective_allowMissing(self, metric):
        def pipeline_allowMissing(params):
            if self.tuning_mode:
                mlflow.log_param('impute_missing_mode', self.impute_missing_mode)
                mlflow.log_param('loss_function', self.loss_function)
                self.preprocess(params['preprocessing_parameters'])
                self.imputation_allowMissing(params['imputation_parameters'])
                self.feature_selection(params['feature_selection_parameters'])
                mlflow.log_param('selected features', list(self.x_imputed.columns))
                self.train_test(params['train_test_split_parameters'])
                mean_metrics, std_metrics, test_metrics, best_model = self.clf_allowMissing(params['classifier_parameters'], metric = metric)
                print ('best_model')
                print (best_model)
                for i in params.keys():
                    for j in params[i].keys():
                        if j in self.multiple_option_parameter:
                            if j == 'clfs':
                                mlflow.log_param(j, list(params[i][j].keys())[0])
                            else:
                                mlflow.log_param(j, params[i][j])
                for a in best_model.keys():
                    mlflow.log_param(a, str(best_model[a]))
                if isinstance(self.x_test, pd.DataFrame):
                    mlflow.log_metrics({**mean_metrics, **std_metrics, **test_metrics})
                else:
                    mlflow.log_metrics({**mean_metrics, **std_metrics})
                cv_metric = metric+'_cv_mean'
                return {'status': hyperopt.STATUS_OK, 'loss': -mean_metrics[cv_metric], 'best_model': best_model}
            else:
                with mlflow.start_run(nested=True) as run:
                    mlflow.log_param('impute_missing_mode', self.impute_missing_mode)
                    mlflow.log_param('loss_function', self.loss_function)
                    self.preprocess(params['preprocessing_parameters'])
                    if type(params['imputation_parameters']['categorical_features']) is tuple:
                        params['imputation_parameters']['categorical_features'] = params['imputation_parameters']['categorical_features'][0]
                    self.imputation_allowMissing(params['imputation_parameters'])
                    self.feature_selection(params['feature_selection_parameters'])
                    mlflow.log_param('selected features', list(self.x_imputed.columns))
                    self.train_test(params['train_test_split_parameters'])
                    mean_metrics, std_metrics, test_metrics, best_model = self.clf_allowMissing(params['classifier_parameters'], metric = metric)
                    print ('best_model')
                    print (best_model)
                    for i in params.keys():
                        for j in params[i].keys():
                            if j == 'clfs':
                                mlflow.log_param(j, list(params[i][j].keys())[0])
                            else:
                                mlflow.log_param(j, params[i][j])
                    for a in best_model.keys():
                        mlflow.log_param(a, str(best_model[a]))
                    if isinstance(self.x_test, pd.DataFrame):
                        mlflow.log_metrics({**mean_metrics, **std_metrics, **test_metrics})
                    else:
                        mlflow.log_metrics({**mean_metrics, **std_metrics})
                    cv_metric = metric+'_cv_mean'
                    return {'status': hyperopt.STATUS_OK, 'loss': -mean_metrics[cv_metric], 'best_model': best_model}
        return pipeline_allowMissing

    def AutoML(self, result_path='./PharmAutoMLResult', impute_missing_mode = True, tuning_mode = True, loss_function = 'accuracy', parameter = None):
        """
        pharm auto ml pipeline with missing imputation
        Args:
            result_path: string, directory to save results
            preprocessing_paras: dictionary, data preprocessing parameters
            classifier_paras: dictionary, data preprocessing parameters
        Returns:
            result: list of dict
        """
        if self.num_unique_y > 2 and metric not in ['accuracy', 'neg_log_loss']:
            raise NotImplementedError("multiclassification task only support accuruacy or neg_log_loss loss function")

        self.tuning_mode = tuning_mode
        self.impute_missing_mode = impute_missing_mode
        self.loss_function = loss_function
        self.result_path = result_path
        if parameter != None:
            default_parameter = parameter
        else:
            default_para_json = open('./src_autoML/default_parameters.json')
            default_parameter = json.load(default_para_json)
        
        parameter_with_clf_list = copy.deepcopy(default_parameter)
        parameter_clf_dict = copy.deepcopy(default_parameter['classifier_parameters']['clfs'])
        if self.impute_missing_mode == False:
            parameter_clf_dict = {k:v for k, v in parameter_clf_dict.items() if k in self.ALLOW_MISSING_CLFS}
        parameter_with_clf_list["preprocessing_parameters"]["drop_features"] = [parameter_with_clf_list["preprocessing_parameters"]["drop_features"]]
        parameter_with_clf_list["imputation_parameters"]["categorical_features"] = [parameter_with_clf_list["imputation_parameters"]["categorical_features"]]
        clfs_parameter_grid = autoML_util.get_parameter_grid_for_clf(parameter_clf_dict,impute_missing_mode = self.impute_missing_mode)
        parameter_with_clf_list["classifier_parameters"]["clfs"] = clfs_parameter_grid

        if self.impute_missing_mode == False and 'RFEcv' in parameter_with_clf_list['feature_selection_parameters']['feature_selection_method']:
            temp = parameter_with_clf_list['feature_selection_parameters']['feature_selection_method']
            temp.remove('RFEcv')
            if temp == None:
                parameter_with_clf_list['feature_selection_parameters']['feature_selection_method'] = [None]
            else:
                parameter_with_clf_list['feature_selection_parameters']['feature_selection_method'] = list(temp)
        self.parameter = parameter_with_clf_list

        self.multiple_option_parameter = autoML_util.get_multiple_option_parameter(self.parameter)
        self.parameter_grid = autoML_util.get_parameter_grid(self.parameter)
        print ('parameter')
        print (self.parameter)
        print ('parameter_grid')

        for i in range(len(self.parameter_grid)):
            print (self.parameter_grid[i])
            print ("\n")

        mlflow.set_tracking_uri(self.result_path)
        mlflow.set_experiment(self.result_path)
        if self.tuning_mode:
            for grid_i in range (len(self.parameter_grid)):
                print ('########################################')
                print ('TUNING MODEL: the parameter of index ', grid_i, 'is')
                print (self.parameter_grid[grid_i])

                parameter_hp_i = autoML_util.hp_parameters(self.parameter_grid[grid_i])
                if self.impute_missing_mode == False:
                    train_objective = self.objective_allowMissing(self.loss_function)
                else:
                    train_objective = self.objective_imputeMissing(self.loss_function)
                trials = hyperopt.Trials()
                with mlflow.start_run() as run:
                    b_model_dict = hyperopt.fmin(fn=train_objective,
                                space=parameter_hp_i,
                                algo=hyperopt.tpe.suggest,
                                max_evals=self.OUTER_MAX_EVALS_OF_TUNING_MODE,
                                trials=trials)
                    search_run_id = run.info.run_id
                    experiment_id = run.info.experiment_id
                print ('the best model is :')
                print (hyperopt.space_eval(parameter_hp_i, b_model_dict))
                best_model = autoML_util.getBestModelfromTrials(trials)
                print (best_model)
        else:
            print ('########################################')
            print ('OPTIMIZATION MODEL: the parameter is ')
            print (self.parameter)

            parameter_hp = autoML_util.hp_parameters(self.parameter)
            if self.impute_missing_mode == False:
                train_objective = self.objective_allowMissing(self.loss_function)
            else:
                train_objective = self.objective_imputeMissing(self.loss_function)
            trials = hyperopt.Trials()
            mlflow.set_experiment('PharmAutoML_OPTIMIZATION_MODEL')
            with mlflow.start_run() as run:
                b_model_dict = hyperopt.fmin(fn=train_objective,
                            space=parameter_hp,
                            algo=hyperopt.tpe.suggest,
                            max_evals=self.OUTER_MAX_EVALS_OF_NORMAL_MODE,
                            trials=trials)
                search_run_id = run.info.run_id
                experiment_id = run.info.experiment_id
            print ('the best model is :')
            print (hyperopt.space_eval(parameter_hp, b_model_dict))
            best_model = autoML_util.getBestModelfromTrials(trials)
            print (best_model)

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

    params = {}

    params['preprocessing_parameters'] = {}
    params['preprocessing_parameters']['drop_features']=args.drop_features
    params['preprocessing_parameters']['missing_threshold_rows']=args.missing_threshold_rows
    params['preprocessing_parameters']['missing_threshold_columns']=args.missing_threshold_columns
    params['preprocessing_parameters']['correlation_threshold']=args.correlation_threshold

    params['imputation_parameters'] = {}
    params['imputation_parameters']['categorical_features']=args.categorical_features
    params['imputation_parameters']['check_diff_categorical_threshold']=args.check_diff_categorical_threshold
    params['imputation_parameters']['impute_category_strategy']=args.impute_category_strategy
    params['imputation_parameters']['impute_numerical_strategy']=args.impute_numerical_strategy
    params['imputation_parameters']['add_nan_label']=args.add_nan_label

    params['feature_selection_parameters'] = {}
    params['feature_selection_parameters']['feature_selection_method'] = args.feature_selection_method

    params['train_test_split_parameters'] = {}
    params['train_test_split_parameters']['tts_random_seed'] = args.tts_random_seed
    params['train_test_split_parameters']['test_size'] = args.test_size

    params['classifier_parameters'] = {}
    params['classifier_parameters']['PCA'] = args.PCA
    params['classifier_parameters']['PCA_feature_num'] = args.PCA_feature_num
    list_clfs = args.clfs
    dict_clfs = {}
    for i in list_clfs:
        dict_clfs[i] = {}
    params['classifier_parameters']['clfs'] = dict_clfs
    params['classifier_parameters']['n_fold'] = args.n_fold
    params['classifier_parameters']['n_repeats'] = args.n_repeats
    params['classifier_parameters']['useDropColFeaImport'] = args.useDropColFeaImport


    if args.result_path == None:
        result_path = os.path.join(cwd, 'result')
    else:
    	result_path = args.result_path
    print ('the result saved at')
    print (result_path)

    aml = PharmAutoML(x, y)
    result_imputeMissing = aml.AutoML(result_path, impute_missing_mode = args.impute_missing_mode, tuning_mode = args.tuning_mode, loss_function = args.loss_function, parameter = params)
