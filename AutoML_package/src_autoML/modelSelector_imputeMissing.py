import autoML_util
import os
import numpy as np
import pandas as pd
from hpsklearn import HyperoptEstimator, logistic_regression_classifier, svc, knn, random_forest, extra_trees, ada_boost, gradient_boosting, sgd, xgboost_classification
from hyperopt import tpe
from hpsklearn import standard_scaler, min_max_scaler, pca
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from functools import partial
from sklearn.preprocessing import StandardScaler
from hyperopt import hp
import shap

class ModelSelector_imputeMissing():
    def __init__(self,para_grid_length,clfs_imputeMissing,result_path,result_logger=None,
        n_fold=5,n_repeats=None,max_evals=10,isFeatureReduction=None,featureDimReductionNumber=None):
        """
        ModelSelector_imputeMissing main class, call classifier with missing imputation from here
        Attributes:
            para_grid_length: number of preprocessing pipelines
            clfs_imputeMissing: the seleected classification methods
            result_path: string
            result_logger: logger file
            n_fold: number of cross validation folds
            n_repeats: number of repeated cross validation
            max_evals: max evaluation iterations
            isFeatureReduction: use feature reduction or not
            featureDimReductionNumber: number of feature dimension reduction
        """
        self.para_grid_length = para_grid_length
        self.clfs_imputeMissing = clfs_imputeMissing
        self.n_fold = n_fold
        self.n_repeats = n_repeats
        self.result_path = result_path
        self.result_logger = result_logger

        self.result = None
        self.isFeatureReduction=None
        self.featureDimReductionNumber=None
        self.random_state = np.random.RandomState(0)

        self.max_evals = max_evals
        self.n_startup_jobs = 20
        self.n_EI_candidates = 24

    def one_classifier_nfold(self, estim, X, y):
        """
        select the best hyperparameter for one classifier using hyperoptEstimator
        Args:
            estim, HyperoptEstimator
            X, pd dataframe
            y, pd dataframe
        Returns:
            bm, estim.best_model()
        """
        estim.fit(X.values, y.to_numpy().ravel(), n_folds = self.n_fold, n_repeats=self.n_repeats, cv_shuffle=True, random_state=self.random_state)
        bm = estim.best_model()
        print (bm)
        self.result_logger.info('########################')
        self.result_logger.info('the best model is')
        self.result_logger.info(bm)
        return bm

    def all_classifier_nfold(self, X_train, y_train):
        """
        select the best hyperparameter for each imputation model and return the best models
        Args:
            X_train: pd dataframe
            y_train: pd dataframe
        Returns:
            bm, dict of best models
        """
        isRefit = False
        if self.isFeatureReduction == True:
            if self.featureDimReductionNumber != None:
                prep = [standard_scaler('my_standard_scaler'), pca('my_pca', n_components = self.featureDimReductionNumber)]
            else:
                prep = [standard_scaler('my_standard_scaler'), pca('my_pca')]
        else:
            prep = [standard_scaler('my_standard_scaler')]
        bm = dict()
        algo = partial(tpe.suggest, n_startup_jobs=self.n_startup_jobs, n_EI_candidates=self.n_EI_candidates)
    # for LRC model
        if 'LRC' in self.clfs_imputeMissing:
            estimLRC = HyperoptEstimator(classifier=logistic_regression_classifier('myLRC', random_state=0),
                                         preprocessing=prep,
                                         ex_preprocs=[], 
                                         algo=algo, 
                                         refit = isRefit, 
                                         max_evals = self.max_evals)
            bm_LRC = self.one_classifier_nfold(estimLRC, X_train, y_train)
            bm['bm_LRC']=bm_LRC
    # for SVC model
        if 'SVC' in self.clfs_imputeMissing:
            estimSVC = HyperoptEstimator(classifier=svc('mySVCClf', probability=True, random_state=0),
                                         preprocessing=prep,
                                         ex_preprocs=[], 
                                         algo=algo, 
                                         refit = isRefit, 
                                         max_evals = self.max_evals)
            bm_SVC = self.one_classifier_nfold(estimSVC, X_train, y_train)
            bm['bm_SVC']=bm_SVC
    # for KNN model
        if 'KNN' in self.clfs_imputeMissing:
            estimKNN = HyperoptEstimator(classifier=knn('myKNNClf'),
                                         preprocessing=prep,
                                         ex_preprocs=[], 
                                         algo=algo, 
                                         refit = isRefit, 
                                         max_evals = self.max_evals)
            bm_KNN = self.one_classifier_nfold(estimKNN, X_train, y_train)
            bm['bm_KNN']=bm_KNN
    # for random forest model
        if 'RF' in self.clfs_imputeMissing:
            estimRF = HyperoptEstimator(classifier=random_forest('myRFClf', random_state=0),
                                        preprocessing=prep,
                                        ex_preprocs=[], 
                                        algo=algo, 
                                        refit = isRefit, 
                                        max_evals = self.max_evals)
            bm_RF = self.one_classifier_nfold(estimRF, X_train, y_train)
            bm['bm_RF']=bm_RF
    # for extra tree model
        if 'ET' in self.clfs_imputeMissing:
            estimET = HyperoptEstimator(classifier=extra_trees('myETClf', random_state=0),
                                        preprocessing=prep,
                                        ex_preprocs=[], 
                                        algo=algo, 
                                        refit = isRefit, 
                                        max_evals = self.max_evals)
            bm_ET = self.one_classifier_nfold(estimET, X_train, y_train)
            bm['bm_ET']=bm_ET
    # for ada boosting model
        if 'AB' in self.clfs_imputeMissing:
            estimAB = HyperoptEstimator(classifier=ada_boost('myABClf', algorithm='SAMME.R', random_state=0),
                                        preprocessing=prep,
                                        ex_preprocs=[], 
                                        algo=algo, 
                                        refit = isRefit, 
                                        max_evals = self.max_evals)
            bm_AB = self.one_classifier_nfold(estimAB, X_train, y_train)
            bm['bm_AB']=bm_AB
    # for gradient boosting model
        if 'GB' in self.clfs_imputeMissing:
            estimGB = HyperoptEstimator(classifier=gradient_boosting('myGBClf', random_state=0), 
                                        preprocessing=prep,
                                        ex_preprocs=[], 
                                        algo=algo, 
                                        refit = isRefit, 
                                        max_evals = self.max_evals)
            bm_GB = self.one_classifier_nfold(estimGB, X_train, y_train)
            bm['bm_GB']=bm_GB
    # for sgd model
        if 'SGD' in self.clfs_imputeMissing:
            loss_SGD = hp.pchoice('loss', [
                    (0.5, 'log'),
                    (0.5, 'modified_huber'),
                    ])
            estimSGD = HyperoptEstimator(classifier=sgd('mySGDClf', loss=loss_SGD, random_state=0), 
                                         preprocessing=prep,
                                         ex_preprocs=[], 
                                         algo=algo, 
                                         refit = isRefit, 
                                         max_evals = self.max_evals)
            bm_SGD = self.one_classifier_nfold(estimSGD, X_train, y_train)
            bm['bm_SGD']=bm_SGD
    # # for xgboost model
        if 'XGboost' in self.clfs_imputeMissing:
            estimXGboost = HyperoptEstimator(classifier=xgboost_classification('myXGboostClf', random_state=0), 
                                             preprocessing=prep,
                                             ex_preprocs=[],
                                             algo=algo, 
                                             refit = isRefit, 
                                             max_evals = self.max_evals)
            bm_XGboost = self.one_classifier_nfold(estimXGboost, X_train, y_train)
            bm['bm_XGboost']=bm_XGboost
        return bm

    def check_bm(self, bm, X, y):
        """refit the best models and calculate the prediction result for each cross validation fold
        Args:
            bm: dict of best models
            X: pd dataframe
            y: pd dataframe
        return:
            refit_bm: dictionary, refitted best model dictionary
            y_classes: dictionary, predicted classes
            y_test_n_fold: dictionary, y test label for each fold
            preds_n_fold: dictionary, predicted classes result
            preds_probs_n_fold: dictionary, predicted probability
        """
        prepro1 = bm['preprocs'][0]
        if self.isFeatureReduction:
            prepro2 = bm['preprocs'][1]
        clf = bm['learner']

        index = 1
        y_classes = {}
        y_test_n_fold = {}
        preds_n_fold = {}
        preds_probs_n_fold = {}
        skf = StratifiedKFold(n_splits=self.n_fold)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train = prepro1.fit_transform(X_train.values)
            if self.isFeatureReduction:
                X_train = prepro2.fit_transform(X_train)
            clf = clf.fit(X_train, y_train.to_numpy().ravel())
    # testing
            X_test = prepro1.transform(X_test.values)
            if self.isFeatureReduction:
                X_test = prepro2.transform(X_test)
            y_classes[index] = clf.classes_
            y_test_n_fold[index] = y_test
            preds = clf.predict(X_test)
            preds_n_fold[index] = preds
            preds_probs = clf.predict_proba(X_test)
            preds_probs_n_fold[index] = preds_probs
            index = index + 1
        refit_bm = bm
        X_train = refit_bm['preprocs'][0].fit_transform(X.values)
        if self.isFeatureReduction:
            X_train = refit_bm['preprocs'][1].fit_transform(X_train)
        refit_bm['learner'].fit(X_train, y.to_numpy().ravel())

        return refit_bm, y_classes, y_test_n_fold, preds_n_fold, preds_probs_n_fold

    def bm_results(self, X_train, y_train, bms):
        """
        Args:
            X_train: pd dataframe
            y_train: pd dataframe
            bms: dict of best models
        return:
            refit_bm: dictionary, refitted best model dictionary for all candidate classifiers
            y_classes_all: dictionary, predicted classes for all candidate classifiers
            y_test_all: dictionary, y test label for each fold for all candidate classifiers
            preds: dictionary, predicted classes result for all candidate classifiers
            preds_probs: dictionary, predicted probability for all candidate classifiers
        """
        refit_bm = dict(); y_classes_all = dict(); y_test_all = dict(); preds = dict(); preds_probs = dict()
    # for LRC model
        if 'LRC' in self.clfs_imputeMissing:
            refit_bm_LRC, y_classes_LRC, y_test_LRC, preds_LRC, preds_probs_LRC = self.check_bm(bms['bm_LRC'], X_train, y_train)
            refit_bm['bm_LRC'] = refit_bm_LRC
            y_classes_all['y_classes_LRC'] = y_classes_LRC
            y_test_all['y_test_LRC'] = y_test_LRC
            preds['preds_LRC']=preds_LRC
            preds_probs['preds_probs_LRC']=preds_probs_LRC
    # for SVC model
        if 'SVC' in self.clfs_imputeMissing:
            refit_bm_SVC, y_classes_SVC, y_test_SVC, preds_SVC, preds_probs_SVC = self.check_bm(bms['bm_SVC'], X_train, y_train)
            refit_bm['bm_SVC'] = refit_bm_SVC
            y_classes_all['y_classes_SVC'] = y_classes_SVC
            y_test_all['y_test_SVC'] = y_test_SVC
            preds['preds_SVC']=preds_SVC
            preds_probs['preds_probs_SVC']=preds_probs_SVC
    # for KNN model
        if 'KNN' in self.clfs_imputeMissing:
            refit_bm_KNN, y_classes_KNN, y_test_KNN, preds_KNN, preds_probs_KNN = self.check_bm(bms['bm_KNN'], X_train, y_train)
            refit_bm['bm_KNN'] = refit_bm_KNN
            y_classes_all['y_classes_KNN'] = y_classes_KNN
            y_test_all['y_test_KNN'] = y_test_KNN
            preds['preds_KNN']=preds_KNN
            preds_probs['preds_probs_KNN']=preds_probs_KNN
    # for random forest model
        if 'RF' in self.clfs_imputeMissing:
            refit_bm_RF, y_classes_RF, y_test_RF, preds_RF, preds_probs_RF = self.check_bm(bms['bm_RF'], X_train, y_train)
            refit_bm['bm_RF'] = refit_bm_RF
            y_classes_all['y_classes_RF'] = y_classes_RF
            y_test_all['y_test_RF'] = y_test_RF
            preds['preds_RF']=preds_RF
            preds_probs['preds_probs_RF']=preds_probs_RF
    # for extra tree model
        if 'ET' in self.clfs_imputeMissing:
            refit_bm_ET, y_classes_ET, y_test_ET, preds_ET, preds_probs_ET = self.check_bm(bms['bm_ET'], X_train, y_train)
            refit_bm['bm_ET'] = refit_bm_ET
            y_classes_all['y_classes_ET'] = y_classes_ET
            y_test_all['y_test_ET'] = y_test_ET
            preds['preds_ET']=preds_ET
            preds_probs['preds_probs_ET']=preds_probs_ET
    # for ada boosting model
        if 'AB' in self.clfs_imputeMissing:
            refit_bm_AB, y_classes_AB, y_test_AB, preds_AB, preds_probs_AB = self.check_bm(bms['bm_AB'], X_train, y_train)
            refit_bm['bm_AB'] = refit_bm_AB
            y_classes_all['y_classes_AB'] = y_classes_AB
            y_test_all['y_test_AB'] = y_test_AB
            preds['preds_AB'] = preds_AB
            preds_probs['preds_probs_AB'] = preds_probs_AB
    # for gradient boosting model
        if 'GB' in self.clfs_imputeMissing:
            refit_bm_GB, y_classes_GB, y_test_GB, preds_GB, preds_probs_GB = self.check_bm(bms['bm_GB'], X_train, y_train)
            refit_bm['bm_GB'] = refit_bm_GB
            y_classes_all['y_classes_GB'] = y_classes_GB
            y_test_all['y_test_GB'] = y_test_GB
            preds['preds_GB']=preds_GB
            preds_probs['preds_probs_GB']=preds_probs_GB
    # for sgd model
        if 'SGD' in self.clfs_imputeMissing:
            refit_bm_SGD, y_classes_SGD, y_test_SGD, preds_SGD, preds_probs_SGD = self.check_bm(bms['bm_SGD'], X_train, y_train)
            refit_bm['bm_SGD'] = refit_bm_SGD
            y_classes_all['y_classes_SGD'] = y_classes_SGD
            y_test_all['y_test_SGD'] = y_test_SGD
            preds['preds_SGD']=preds_SGD
            preds_probs['preds_probs_SGD']=preds_probs_SGD
    # # for xgboost model
        if 'XGboost' in self.clfs_imputeMissing:
            refit_bm_XGboost, y_classes_XGboost, y_test_XGboost, preds_XGboost, preds_probs_XGboost = self.check_bm(bms['bm_XGboost'], X_train, y_train)
            refit_bm['bm_XGboost'] = refit_bm_XGboost
            y_classes_all['y_classes_XGboost'] = y_classes_XGboost
            y_test_all['y_test_XGboost'] = y_test_XGboost
            preds['preds_XGboost']=preds_XGboost 
            preds_probs['preds_probs_XGboost']=preds_probs_XGboost

        return refit_bm, y_classes_all, y_test_all, preds, preds_probs
