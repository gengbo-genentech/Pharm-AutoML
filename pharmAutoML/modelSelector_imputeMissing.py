import autoML_util
import os
import numpy as np
import pandas as pd
import importlib
from pharmAutoML import hpsklearn
from hyperopt import tpe
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from functools import partial
from sklearn.preprocessing import StandardScaler
from hyperopt import hp
import shap
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from statistics import mean, stdev
from sklearn import model_selection
from sklearn import metrics
from hyperopt import Trials
import modelInterpreter
import mlflow
import json

class ModelSelector_imputeMissing():
    def __init__(self, x_train, y_train, classifier_parameters):
        self.x_train = x_train
        self.y_train = y_train
        self.num_unique_y = len(y_train.unique())
        self.classifier_parameters = classifier_parameters
        self.random_state = np.random.RandomState(0)
        self.best_model = None
        self.max_evals = 3
        self.n_startup_jobs = 10
        self.n_EI_candidates = 12
        self.clf_name = None
        self.manual_clf_param = None

    def one_classifier_nfold(self, estim):
        """
        select the best hyperparameter for one classifier using hyperoptEstimator
        Args:
            estim, HyperoptEstimator
            X, pd dataframe
            y, pd dataframe
        Returns:
            bm, estim.best_model()
        """
        estim.fit(self.x_train.values, self.y_train.to_numpy().ravel(), n_folds = self.classifier_parameters["n_fold"], 
            n_repeats=self.classifier_parameters["n_repeats"], cv_shuffle=True, random_state=self.random_state)
        bm = estim.best_model()

        # print (bm)
        # self.result_logger.info('########################')
        # self.result_logger.info('the best model is')
        # self.result_logger.info(bm)
        return bm

    def binary_classification_train_validation_metrics(self, n_fold = 5):
        prepro1 = self.best_model['preprocs'][0]
        if self.classifier_parameters['PCA']:
            prepro2 = self.best_model['preprocs'][1]
        clf = self.best_model['learner']

        x_train_pp = prepro1.fit_transform(self.x_train.values)
        if self.classifier_parameters['PCA']:
            x_train_pp = prepro2.fit_transform(x_train_pp)
        
        y_pred = model_selection.cross_val_predict(clf, x_train_pp, self.y_train, cv=n_fold, method = 'predict_proba')[:,1]
        title = 'ROC Curves of cross validation'
        roc_fig = autoML_util.single_roc_plot(self.y_train.to_numpy(), y_pred, title=title)
        plt.tight_layout()
        mlflow.log_figure(roc_fig, "ROC_Curve_cross_validation.png")
        plt.close()

        log_loss_cv = model_selection.cross_val_score(clf, x_train_pp, self.y_train, scoring='neg_log_loss', cv=n_fold)
        acc_cv = model_selection.cross_val_score(clf, x_train_pp, self.y_train, scoring='accuracy', cv=n_fold)
        roc_auc_cv = model_selection.cross_val_score(clf, x_train_pp, self.y_train, scoring='roc_auc', cv=n_fold)
        f1_cv = model_selection.cross_val_score(clf, x_train_pp, self.y_train, scoring='f1', cv=n_fold)
        pr_auc = model_selection.cross_val_score(clf, x_train_pp, self.y_train, scoring='average_precision', cv=n_fold)
        precision_cv = model_selection.cross_val_score(clf, x_train_pp, self.y_train, scoring='precision', cv=n_fold)
        recall_cv = model_selection.cross_val_score(clf, x_train_pp, self.y_train, scoring='recall', cv=n_fold)
        mean_metrics = {"neg_log_loss_cv_mean": mean(log_loss_cv), "accuracy_cv_mean": mean(acc_cv), 
                        "roc_auc_cv_mean": mean(roc_auc_cv),"f1_cv_mean": mean(f1_cv),
                        "average_precision_cv_mean": mean(pr_auc), "precision_cv_mean": mean(precision_cv), "recall_cv_mean": mean(recall_cv)}
        std_metrics = {"neg_log_loss_cv_stdev": stdev(log_loss_cv), "accuracy_cv_stdev": stdev(acc_cv), 
                       "roc_auc_cv_stdev": stdev(roc_auc_cv),"f1_cv_stdev": stdev(f1_cv),
                        "average_precision_cv_stdev": stdev(pr_auc), "precision_cv_stdev": stdev(precision_cv), "recall_cv_stdev": stdev(recall_cv)}
        return mean_metrics, std_metrics

    def bm_fit_train_validation_cv(self, n_cv=5):
        mean_metrics, std_metrics = self.binary_classification_train_validation_metrics(n_fold = n_cv)
        print ('mean value of different metrics are ')
        print (mean_metrics)
        print ('std of different metrics are ')
        print (std_metrics)
        return mean_metrics, std_metrics

    def bm_predict_test_data(self, x_test, y_test):
        # bm = result['para_index_'+str(para_index)]['bm']['bm_'+clf_string]
        prepro1 = self.best_model['preprocs'][0]
        if self.classifier_parameters['PCA']:
            prepro2 = self.best_model['preprocs'][1]
        clf = self.best_model['learner']

        x_test = prepro1.transform(x_test.values)
        if self.classifier_parameters['PCA']:
            x_test = prepro2.fit_transform(x_test)
        
        preds_probs = clf.predict_proba(x_test)[:,1]
        title = 'ROC Curves of test data'
        roc_test_fig = autoML_util.single_roc_plot(y_test.to_numpy(), preds_probs, title=title)
        plt.tight_layout()
        mlflow.log_figure(roc_test_fig, "ROC_Curve_test_dataset.png")
        plt.close()

        log_loss = model_selection._validation._score(clf, x_test, y_test, scorer=metrics.check_scoring(clf, scoring = 'neg_log_loss'))
        acc = model_selection._validation._score(clf, x_test, y_test, scorer=metrics.check_scoring(clf, scoring = 'accuracy'))
        if self.num_unique_y == 2:
            roc_auc = model_selection._validation._score(clf, x_test, y_test, scorer=metrics.check_scoring(clf, scoring = 'roc_auc'))
            f1 = model_selection._validation._score(clf, x_test, y_test, scorer=metrics.check_scoring(clf, scoring = 'f1'))
            pr_auc = model_selection._validation._score(clf, x_test, y_test, scorer=metrics.check_scoring(clf, scoring = 'average_precision'))
            precision = model_selection._validation._score(clf, x_test, y_test, scorer=metrics.check_scoring(clf, scoring = 'precision'))
            recall = model_selection._validation._score(clf, x_test, y_test, scorer=metrics.check_scoring(clf, scoring = 'recall'))
            test_metrics = {"neg_log_loss_test": log_loss, "accuracy_test": acc, 
                    "roc_auc_test": roc_auc,"f1_test": f1,
                    "average_precision_test": pr_auc, "precision_test": precision, "recall_test": recall}
        else:
            test_metrics = {"neg_log_loss_test": log_loss, "accuracy_test": acc}
        return test_metrics

    def classifier_nfold(self, clf_name_and_param, fn_string='roc_auc'):
        """
        select the best hyperparameter for each imputation model and return the best models
        Args:
            X_train: pd dataframe
            y_train: pd dataframe
            loss_function: none, mean_absolute_error
        Returns:
            bm, dict of best models
        """
        isRefit = True
        
        clf_name = list(clf_name_and_param.keys())[0]
        self.clf_name = clf_name
        manual_clf_param = list(clf_name_and_param.values())[0]
        self.manual_clf_param = manual_clf_param

        if self.classifier_parameters['PCA'] == True:
            if self.classifier_parameters['PCA_feature_num'] != None:
                prep = [hpsklearn.standard_scaler('my_standard_scaler'), 
                    hpsklearn.pca('my_pca', n_components = self.classifier_parameters['PCA_feature_num'])]
            else:
                prep = [hpsklearn.standard_scaler('my_standard_scaler'), 
                    hpsklearn.pca('my_pca')]
        else:
            prep = [hpsklearn.standard_scaler('my_standard_scaler')]
        algo = partial(tpe.suggest, n_startup_jobs=self.n_startup_jobs, n_EI_candidates=self.n_EI_candidates)

        def loss_function(fn_string, num_unique_y):
            if fn_string == 'neg_log_loss':
                return lambda target, pred: -log_loss(target, pred)
            elif fn_string == 'accuracy':
                return lambda target, pred: -accuracy_score(target, pred)
            elif num_unique_y == 2 and fn_string == 'roc_auc':
                return lambda target, pred: -roc_auc_score(target, pred)
            elif num_unique_y == 2 and fn_string == 'f1':
                return lambda target, pred: -f1_score(target, pred)
            elif num_unique_y == 2 and fn_string == 'average_precision':
                return lambda target, pred: -average_precision_score(target, pred)
            elif num_unique_y == 2 and fn_string == 'precision':
                return lambda target, pred: -precision_score(target, pred)
            elif num_unique_y == 2 and fn_string == 'recall':
                return lambda target, pred: -recall_score(target, pred)
            else:
                raise NotImplementedError("this loss function is not implemented")

        # for LRC model
        if clf_name == 'LRC':
            estimLRC = hpsklearn.HyperoptEstimator(classifier=hpsklearn.logistic_regression_classifier('myLRC', **manual_clf_param, random_state=0),
                                                   preprocessing=prep, ex_preprocs=[], algo=algo, refit = isRefit, 
                                                   max_evals = self.max_evals, loss_fn = loss_function(fn_string, self.num_unique_y))
            if manual_clf_param != {}:
                for i in manual_clf_param.keys():
                    k = clf_name + '_' + i
                    mlflow.log_param(k, manual_clf_param[i])
            bm_LRC = self.one_classifier_nfold(estimLRC)
            self.best_model = bm_LRC
        # for SVC model
        if clf_name == 'SVC':
            estimSVC = hpsklearn.HyperoptEstimator(classifier=hpsklearn.svc('mySVCClf', probability=True, **manual_clf_param, random_state=0),
                                                   preprocessing=prep, ex_preprocs=[], algo=algo, refit = isRefit, 
                                                   max_evals = self.max_evals, loss_fn = loss_function(fn_string, self.num_unique_y))
            bm_SVC = self.one_classifier_nfold(estimSVC)
            if manual_clf_param != {}:
                for i in manual_clf_param.keys():
                    k = clf_name + '_' + i
                    mlflow.log_param(k, manual_clf_param[i])
            self.best_model = bm_SVC
        # for KNN model
        if clf_name == 'KNN':
            estimKNN = hpsklearn.HyperoptEstimator(classifier=hpsklearn.knn('myKNNClf', **manual_clf_param), 
                                                   preprocessing=prep, ex_preprocs=[], algo=algo, refit = isRefit, 
                                                   max_evals = self.max_evals, loss_fn = loss_function(fn_string, self.num_unique_y))
            bm_KNN = self.one_classifier_nfold(estimKNN)
            if manual_clf_param != {}:
                for i in manual_clf_param.keys():
                    k = clf_name + '_' + i
                    mlflow.log_param(k, manual_clf_param[i])
            self.best_model = bm_KNN
        # for random forest model
        if clf_name == 'RF':
            estimRF = hpsklearn.HyperoptEstimator(classifier=hpsklearn.random_forest('myRFClf', **manual_clf_param, random_state=0),
                                                  preprocessing=prep, ex_preprocs=[], algo=algo, refit = isRefit, 
                                                  max_evals = self.max_evals, loss_fn = loss_function(fn_string, self.num_unique_y))
            bm_RF = self.one_classifier_nfold(estimRF)
            if manual_clf_param != {}:
                for i in manual_clf_param.keys():
                    k = clf_name + '_' + i
                    mlflow.log_param(k, manual_clf_param[i])
            self.best_model = bm_RF
        # for extra tree model
        if clf_name == 'ET':
            estimET = hpsklearn.HyperoptEstimator(classifier=hpsklearn.extra_trees('myETClf', **manual_clf_param, random_state=0),
                                                  preprocessing=prep, ex_preprocs=[], algo=algo, refit = isRefit, 
                                                  max_evals = self.max_evals, loss_fn = loss_function(fn_string, self.num_unique_y))
            bm_ET = self.one_classifier_nfold(estimET)
            if manual_clf_param != {}:
                for i in manual_clf_param.keys():
                    k = clf_name + '_' + i
                    mlflow.log_param(k, manual_clf_param[i])
            self.best_model = bm_ET
        # for ada boosting model
        if clf_name == 'AB':
            estimAB = hpsklearn.HyperoptEstimator(classifier=hpsklearn.ada_boost('myABClf', **manual_clf_param, algorithm='SAMME.R', random_state=0),
                                                  preprocessing=prep, ex_preprocs=[], algo=algo, refit = isRefit, 
                                                  max_evals = self.max_evals, loss_fn = loss_function(fn_string, self.num_unique_y))
            bm_AB = self.one_classifier_nfold(estimAB)
            if manual_clf_param != {}:
                for i in manual_clf_param.keys():
                    k = clf_name + '_' + i
                    mlflow.log_param(k, manual_clf_param[i])
            self.best_model = bm_AB
        # for gradient boosting model
        if clf_name == 'GB':
            estimGB = hpsklearn.HyperoptEstimator(classifier=hpsklearn.gradient_boosting('myGBClf', **manual_clf_param, random_state=0), 
                                                  preprocessing=prep, ex_preprocs=[], algo=algo, refit = isRefit, 
                                                  max_evals = self.max_evals, loss_fn = loss_function(fn_string, self.num_unique_y))
            bm_GB = self.one_classifier_nfold(estimGB)
            if manual_clf_param != {}:
                for i in manual_clf_param.keys():
                    k = clf_name + '_' + i
                    mlflow.log_param(k, manual_clf_param[i])
            self.best_model = bm_GB
        # for sgd model
        if clf_name == 'SGD':
            loss_SGD = hp.pchoice('loss', [
                    (0.5, 'log'),
                    (0.5, 'modified_huber'),
                    ])
            estimSGD = hpsklearn.HyperoptEstimator(classifier=hpsklearn.sgd('mySGDClf', loss=loss_SGD, **manual_clf_param, random_state=0), 
                                         preprocessing=prep, ex_preprocs=[], algo=algo, refit = isRefit, 
                                         max_evals = self.max_evals, loss_fn = loss_function(fn_string, self.num_unique_y))
            bm_SGD = self.one_classifier_nfold(estimSGD)
            if manual_clf_param != {}:
                for i in manual_clf_param.keys():
                    k = clf_name + '_' + i
                    mlflow.log_param(k, manual_clf_param[i])
            self.best_model = bm_SGD
        # for xgboost model
        if clf_name == 'XGboost':
            estimXGboost = hpsklearn.HyperoptEstimator(classifier=hpsklearn.xgboost_classification('myXGboostClf', **manual_clf_param, random_state=0), 
                                             preprocessing=prep, ex_preprocs=[], algo=algo, refit = isRefit, 
                                             max_evals = self.max_evals, loss_fn = loss_function(fn_string, self.num_unique_y))
            bm_XGboost = self.one_classifier_nfold(estimXGboost)
            self.best_model = bm_XGboost
            if manual_clf_param != {}:
                for i in manual_clf_param.keys():
                    k = clf_name + '_' + i
                    mlflow.log_param(k, manual_clf_param[i])
        return self.best_model

    def model_interpreter(self, x_test, y_test):
        if self.clf_name in ['XGboost', 'RF', 'ET', 'GB']:
            shap_values = self.shap_analysis(x_test, y_test)
        features_importance = modelInterpreter.sklearn_feature_importance(self.x_train, self.y_train, 
            self.best_model, self.clf_name, self.classifier_parameters['PCA'])
        if self.classifier_parameters["useDropColFeaImport"]:
            features_importance = modelInterpreter.sklearn_feature_importance(self.x_train, self.y_train, 
                self.best_model, self.clf_name, self.classifier_parameters['PCA'])

    def shap_analysis(self, x_test, y_test):
        """ shap analyze the best model and save the feature importance figures in result_path
        Args:
            result: dictionary, selected model and prediction results
            result_path: string, result saving path
            isFeatureReduction: boolean
            clf_string: XGboost
        """
        prepro1 = self.best_model['preprocs'][0]
        if self.classifier_parameters['PCA']:
            prepro2 = self.best_model['preprocs'][1]
        clf = self.best_model['learner']

        x_train_pp = prepro1.fit_transform(self.x_train.values)
        if self.classifier_parameters['PCA']:
            x_train_pp = prepro2.fit_transform(x_train_pp)

        clf = clf.fit(x_train_pp, self.y_train.to_numpy().ravel())
        explainer = shap.TreeExplainer(clf)

        x_test_pp = prepro1.transform(x_test)
        if self.classifier_parameters['PCA']:
            x_test_pp = prepro2.transform(x_test_pp)
        x_test_pp = pd.DataFrame(data=x_test_pp, columns=x_test.columns)
        try:
            fig_shap = plt.figure(dpi=150)
            shap_values = explainer.shap_values(x_test_pp)
            shap_title = 'shap_plot_of_' + self.clf_name + '_on_test_data.png'
            # if clf_string != 'ET' or clf_string != 'RF':
            shap.summary_plot(shap_values, x_test_pp)
            plt.tight_layout()
            mlflow.log_figure(fig_shap, shap_title)

            # plt.savefig(result_path + '/shapPlot' + str(para_index) + 'prepro_' + clf_string + '.png', dpi=150)
            plt.close()

            fig_shap_bar = plt.figure(dpi=150)
            shap_bar_title = 'shap_bar_plot_of_' + self.clf_name + '_on_test_data.png'
            if self.clf_name != 'ET' or self.clf_name != 'RF':
                shap_bar_plot = shap.summary_plot(shap_values, x_test_pp, plot_type="bar")
                plt.tight_layout()
                mlflow.log_figure(fig_shap_bar, shap_bar_title)
                # plt.savefig(result_path + '/shapBarPlot' + str(para_index) + 'prepro_' + clf_string + '.png', dpi=150)
                plt.close()
            return shap_values
        except:
            print ('unsolved shap error, https://github.com/slundberg/shap/issues/941')

    def save_learner(self, best_model):
        clf = best_model['learner']
        # print ("dbg")
        print (clf.__class__.__name__)
        if clf.__class__.__name__ == "XGBClassifier":
        	mlflow.xgboost.log_model(clf, clf.__class__.__name__)
        else:
            mlflow.sklearn.log_model(clf, clf.__class__.__name__)
