import autoML_util
import numpy as np
import pandas as pd
from hpsklearn import HyperoptEstimator, xgboost_classification
from hyperopt import tpe
from hpsklearn import standard_scaler, min_max_scaler
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import StratifiedKFold
from functools import partial
from sklearn.preprocessing import StandardScaler

class ModelSelector_allowMissing():
    def __init__(self, para_grid_length, clfs_allowMissing, n_fold, n_repeats, result_path, result_logger, max_evals = 10):
        """
        ModelSelector_allowMissing main class, user can call classifier with missing allow from here

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
        self.clfs_allowMissing = clfs_allowMissing
        self.n_fold = n_fold
        self.n_repeats = n_repeats
        self.result_path = result_path
        self.result_logger = result_logger

        self.result = None
        self.random_state = np.random.RandomState(0)

        self.max_evals = max_evals
        self.n_startup_jobs = 20
        self.n_EI_candidates = 24

    def one_classifier_nfold_allowMissing(self, estim, X, y):
        """
        select the best hyperparameter for one allow missing classifier using hyperoptEstimator
        Args:
            estim, HyperoptEstimator
            X, pd dataframe
            y, pd dataframe
        Returns:
            bm, estim.best_model()
        """
        estim.fit(X.values, y.to_numpy().ravel(), n_folds = self.n_fold, n_repeats=self.n_repeats, random_state=self.random_state, cv_shuffle=True)
        bm = estim.best_model()
        self.result_logger.info('########################')
        print (bm)
        self.result_logger.info('the best allow missing model is')
        self.result_logger.info(bm)
        return bm

    def all_classifier_nfold_allowMissing(self, X_train, y_train):
        """
        select the best hyperparameter for allow missing model (xgboost) and return the best models
        Args:
            X_train: pd dataframe
            y_train: pd dataframe
        Returns:
            bm, dict of best models
        """
        bm = dict()
        isRefit = False
        algo = partial(tpe.suggest, n_startup_jobs=self.n_startup_jobs, n_EI_candidates=self.n_EI_candidates)
    # # for xgboost model
        if 'XGboost' in self.clfs_allowMissing:
            estimXGboost = HyperoptEstimator(classifier=xgboost_classification('XGboostClf_allowMissing', random_state=0), 
                                             preprocessing=[],
                                             ex_preprocs=[], 
                                             algo=algo, 
                                             refit = isRefit, 
                                             max_evals = self.max_evals)

            bm_XGboost = self.one_classifier_nfold_allowMissing(estimXGboost, X_train, y_train)
            bm['bm_XGboost']=bm_XGboost
        return bm

    def check_bm_allowMissing(self, bm, X, y):
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
            clf = clf.fit(X_train, y_train.to_numpy().ravel())
    # testing
            y_classes[index] = clf.classes_
            y_test_n_fold[index] = y_test
            preds = clf.predict(X_test)
            preds_n_fold[index] = preds
            # self.result_logger.info(type(clf))
            # self.result_logger.info(clf.get_params())
            preds_probs = clf.predict_proba(X_test)
            preds_probs_n_fold[index] = preds_probs
            index = index + 1
        refit_bm = bm
        refit_bm['learner'].fit(X, y.to_numpy().ravel())

        return refit_bm, y_classes, y_test_n_fold, preds_n_fold, preds_probs_n_fold

    def bm_results_allowMissing(self, X_train, y_train, bms):
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
    # # for xgboost model
        if 'XGboost' in self.clfs_allowMissing:
            refit_bm_XGboost, y_classes_XGboost, y_test_XGboost, preds_XGboost, preds_probs_XGboost = self.check_bm_allowMissing(bms['bm_XGboost'], X_train, y_train)
            refit_bm['bm_XGboost'] = refit_bm_XGboost
            y_classes_all['y_classes_XGboost'] = y_classes_XGboost
            y_test_all['y_test_XGboost'] = y_test_XGboost
            preds['preds_XGboost']=preds_XGboost
            preds_probs['preds_probs_XGboost']=preds_probs_XGboost
        return refit_bm, y_classes_all, y_test_all, preds, preds_probs
