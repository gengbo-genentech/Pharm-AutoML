import pandas as pd
import numpy as np
import xgboost as xgb
from probatus.feature_elimination import ShapRFECV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import mlflow
import matplotlib.pyplot as plt

class FeatureSelection():
    def __init__(self, x, y, featureSelection_params, metric = 'roc_auc', logger=None):
        # Dataset and optional training labels
        # metric: 'roc_auc', 'f1', 'precision', 'recall', 'average_precision'
        self.x = x
        self.y = y
        self.n_fold = 5
        self.scoring = metric
        self.featureSelection_params = featureSelection_params

        self.selected_x = None
        self.selected_y = None

    def feature_selection_main(self):
        if self.featureSelection_params["feature_selection_method"] == 'shapRFEcv':
            self.selected_x, self.selected_y = self.shapRFEcv_feature_selection(clf_name='xgboost')
            return self.selected_x, self.selected_y
        elif self.featureSelection_params["feature_selection_method"] == 'RFEcv':
            self.selected_x, self.selected_y = self.rfecv_feature_selection_with_grid_search(clf_name='logistic_regression')
            return self.selected_x, self.selected_y
        else:
            return self.x, self.y

    def choose_clf(self, clf_name='xgboost'):
        if clf_name == 'xgboost':
            xgb_clf = xgb.XGBClassifier()
            return xgb_clf
        elif clf_name == 'logistic_regression':
            lr_clf = LogisticRegression(solver='liblinear')
            return lr_clf
        elif clf_name == 'random_forest':
            rf_clf = RandomForestClassifier()
            return rf_clf
        elif clf_name == 'light_gbm':
            lgbm_clf = LGBMClassifier()
            return lgbm_clf

    def get_param_grid(self, clf_name='xgboost'):
        if clf_name == 'xgboost':
            xgboost_param_grid = {'n_estimators':[20,40,80], 
                                  'max_depth':[3,5,7],
                                  'learning_rate':[0.1,0.01,0.001]}
            return xgboost_param_grid
        elif clf_name == 'light_gbm':
            lightgbm_param_grid = {'n_estimators': [5, 7, 10], 
                                   'num_leaves': [3, 5, 7, 10]}
            return light_gbm_param_grid
        elif clf_name == 'random_forest':
            rf_param_grid = {"estimator__max_depth": [3, None],
                             "estimator__bootstrap": [True, False],  
                             "estimator__criterion": ["gini", "entropy"]}
            return rf_param_grid
        elif clf_name == 'logistic_regression':
            lr_param_grid = {"estimator__penalty": ["l1", "l2"],
                            "estimator__C": [2, 1, 0.5]
                            }
            return lr_param_grid

    def shapRFEcv_feature_selection(self, clf_name='xgboost'):
        clf = self.choose_clf(clf_name)
        param_grid = self.get_param_grid(clf_name)
        search = GridSearchCV(clf, param_grid, 
            cv=self.n_fold, 
            scoring=self.scoring, 
            refit=False)
        shap_elimination = ShapRFECV(search, step=0.2, 
            cv=StratifiedKFold(5), 
            scoring=self.scoring)
        report = shap_elimination.fit_compute(self.x, self.y)
        shapRFEcv_report_df = shap_elimination.report_df
        performance_plot_ax = self.shap_elimination_plot(shapRFEcv_report_df, shap_elimination.scorer.metric_name, show=False, figsize=(20, 15), dpi=60)
        row_n_of_highest_val_metric_mean = shapRFEcv_report_df['val_metric_mean'].argmax()
        feature_list_highest_metric = shapRFEcv_report_df.loc[row_n_of_highest_val_metric_mean,'features_set']
        print ('the selected features by shapRFEcv are ', feature_list_highest_metric)
        return self.x[feature_list_highest_metric], self.y

    def rfecv_plot(self, grid_score, show=False, **figure_kwargs):
        fig = plt.figure(**figure_kwargs)
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(grid_score) + 1), grid_score)
        if show:
            plt.show()
        mlflow.log_figure(fig, "RFEcv_report.png")

    def rfecv_feature_selection_with_grid_search(self, clf_name = 'logistic_regression'):
        estimator = self.choose_clf(clf_name)
        rfecv = RFECV(estimator, step=1, cv=5, scoring = self.scoring)
        clf = GridSearchCV(rfecv, param_grid=self.get_param_grid(clf_name))
        clf.fit(self.x, self.y)
        selector = clf.best_estimator_
        support = selector.support_
        validation_score = selector.grid_scores_
        selected_features = self.x.columns[support]
        print ('the best classifier is ')
        print (selector)
        print (str(len(selected_features)) + ' features are selected by rfecv')
        print (selected_features)
        print ('the best validation score is ' + str(validation_score[len(selected_features)-1]))
        self.rfecv_plot(validation_score)
        return self.x[selected_features], self.y

    def rfecv_feature_selection(self, clf_name = 'logistic_regression'):
        estimator = self.choose_clf(clf_name)
        selector = RFECV(estimator, step = 1, cv = 5, scoring = self.scoring)
        selector = selector.fit(self.x, self.y)
        support = selector.support_
        validation_score = selector.grid_scores_
        selected_features = self.x.columns[support]
        print (str(len(selected_features)) + ' features are selected by rfecv')
        print ('the best validation score is ' + str(validation_score[len(selected_features)-1]))
        self.rfecv_plot(validation_score)
        return self.x[selected_features], self.y

    def shap_elimination_plot(self, report_df, metric_name, show=False, **figure_kwargs):
        """
        Generates plot of the model performance for each iteration of feature elimination.
        Args:
            show (bool, optional):
                If True, the plots are showed to the user, otherwise they are not shown. Not showing plot can be useful,
                when you want to edit the returned axis, before showing it.
            **figure_kwargs:
                Keyword arguments that are passed to the plt.figure, at its initialization.
        Returns:
            (plt.axis):
                Axis containing the performance plot.
        """
        x_ticks = list(reversed(report_df["num_features"].tolist()))

        fig = plt.figure(**figure_kwargs)

        plt.plot(
            report_df["num_features"],
            report_df["train_metric_mean"],
            label="Train Score",
        )
        plt.fill_between(
            pd.to_numeric(report_df.num_features, errors="coerce"),
            report_df["train_metric_mean"] - report_df["train_metric_std"],
            report_df["train_metric_mean"] + report_df["train_metric_std"],
            alpha=0.3,
        )

        plt.plot(
            report_df["num_features"],
            report_df["val_metric_mean"],
            label="Validation Score",
        )
        plt.fill_between(
            pd.to_numeric(report_df.num_features, errors="coerce"),
            report_df["val_metric_mean"] - report_df["val_metric_std"],
            report_df["val_metric_mean"] + report_df["val_metric_std"],
            alpha=0.3,
        )

        plt.xlabel("Number of features")
        plt.ylabel(f"Performance {metric_name}")
        plt.title("Backwards Feature Elimination using SHAP & CV")
        plt.legend(loc="lower left")
        ax = plt.gca()
        ax.invert_xaxis()
        ax.set_xticks(x_ticks)

        mlflow.log_figure(fig, "shapRFEcv_report.png")

        if show:
            plt.show()
        else:
            plt.close()
        return ax
