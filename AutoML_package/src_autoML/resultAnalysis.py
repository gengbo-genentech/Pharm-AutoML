import autoML_util
import os
import numpy as np
import pandas as pd
from hpsklearn import HyperoptEstimator, logistic_regression_classifier, svc, knn, random_forest, extra_trees, ada_boost, gradient_boosting, sgd, xgboost_classification
from hyperopt import tpe
import sklearn
from hpsklearn import standard_scaler, min_max_scaler, pca
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import statistics
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hyperopt import hp
import modelInterpreter
import shap
from functools import wraps
import errno
import os
import pickle
import signal

def plot_roc_curve(result, n_fold, para_index, clf, file, rank_metrics='ROC'):
    """
    plot roc auc curve
    Args:
        result: dictionary of result
        n_fold: int, number of cross validation folds
        para_index: int, data preprocessing index
        clf: string, classifier
        file: save folder
        rank_metrics: string, ranking metrics
    """
    y_classes_clf = result['para_index_'+str(para_index)]['y_classes_n_fold']['y_classes_'+clf]
    y_test_clf = result['para_index_'+str(para_index)]['y_test_n_fold']['y_test_'+clf]
    preds_probs_clf = result['para_index_'+str(para_index)]['preds_probs']['preds_probs_'+clf]
    roc_title = 'ROC Curves of ' + clf + ' on validation data'
    classes = y_classes_clf[1]
    y_test = []
    preds_probs = []
    if len(np.unique(classes)) == 2:
        y_test_clf_all = []
        for fold in range(n_fold):
            temp_y_test = np.squeeze(y_test_clf[fold+1].values).tolist()
            temp_y_test2 = np.squeeze(np.swapaxes(label_binarize(temp_y_test, classes=y_classes_clf[fold+1]), 0, 1))
            y_test_clf_all = y_test_clf_all+[temp_y_test2]
        preds_probs_clf_all = []
        for fold in range(n_fold):
            temp_preds_probs_clf = np.swapaxes(np.squeeze(preds_probs_clf[fold+1]),0,1)
            preds_probs_clf_all = preds_probs_clf_all+[temp_preds_probs_clf[1]]
    else:
        y_test_clf_all = []
        for i in range(len(y_test_clf)):
            temp_y_test = np.squeeze(y_test_clf[i+1].values).tolist()
            temp_y_test2 = label_binarize(temp_y_test, classes=y_classes_clf[i+1]).tolist()
            y_test_clf_all = y_test_clf_all + temp_y_test2
        y_test_clf_all = np.swapaxes(np.array(y_test_clf_all), 0, 1)
        preds_probs_clf_all = []
        for i in range(len(y_test_clf)):
            preds_probs_clf_all = preds_probs_clf_all+np.squeeze(preds_probs_clf[i+1]).tolist()
        preds_probs_clf_all = np.swapaxes(preds_probs_clf_all, 0, 1)
    
    if len(np.unique(classes))==2:
        classes = None    
    ax, n_fold_roc_auc, average_roc_auc = autoML_util.plot_roc(y_test_clf_all, preds_probs_clf_all, classes = classes, title = roc_title)
    plt.tight_layout()
    plt.savefig(file + '/' + str(para_index) + 'prepro_' + str(n_fold) + 'folds_' + clf + '_roc.png', dpi=300)
    plt.close()
    pr_title = 'Precision Recall Curve of ' + clf  + ' on validation data'
    ax, n_fold_pr_score, all_average_precision = autoML_util.plot_precision_recall(y_test_clf_all, preds_probs_clf_all, classes = classes, title = pr_title)
    plt.tight_layout()
    plt.savefig(file + '/' + str(para_index) + 'prepro_' + str(n_fold) + 'folds_' + clf + '_pr.png', dpi=300)
    plt.close()
    if rank_metrics == 'ROC':
        return n_fold_roc_auc, average_roc_auc
    elif rank_metrics == 'PR':
        return n_fold_pr_score, all_average_precision

def metrics_table(result, n_fold, para_index, clf):
    """calculate predication results by different metrics
        acc, auroc, auprc, f1, sensitivity, precision
    Args:
        result: dict, result from hyperopt
        n_fold: int
        para_index: int, preprocessing parameter index
        clf: string, classifier
    Returns:
        metrics: dict, prediction result of all metrics
    """

    metrics = dict()
    y_classes_clf = result['para_index_'+str(para_index)]['y_classes_n_fold']['y_classes_'+clf]
    y_test_clf = result['para_index_'+str(para_index)]['y_test_n_fold']['y_test_'+clf]
    preds_clf = result['para_index_'+str(para_index)]['preds']['preds_'+clf]
    preds_probs_clf = result['para_index_'+str(para_index)]['preds_probs']['preds_probs_'+clf]
    if len(np.unique(y_classes_clf[1])) == 2:
        acc_nfold = []
        auroc_nfold = []
        auprc_nfold = []
        f1_nfold = []
        sensitivity_nfold = []
        precision_nfold = []
        for fold in range(n_fold):
            y_true = y_test_clf[fold+1]
            y_pred = preds_clf[fold+1]
            y_pred_prob = np.swapaxes(np.squeeze(preds_probs_clf[fold+1]),0,1)[1]
            acc=accuracy_score(y_true, y_pred)
            acc_nfold.append(acc)
            roc_auc = roc_auc_score(y_true, y_pred_prob)
            auroc_nfold.append(roc_auc)
            pr_auc = average_precision_score(y_true, y_pred_prob)
            auprc_nfold.append(pr_auc)
            f1 = f1_score(y_true, y_pred)
            f1_nfold.append(f1)
            precision = precision_score(y_true, y_pred)
            precision_nfold.append(precision)
            recall = recall_score(y_true, y_pred)
            sensitivity_nfold.append(recall)
        metrics['acc'] = acc_nfold
        metrics['auroc'] = auroc_nfold
        metrics['auprc'] = auprc_nfold
        metrics['f1'] = f1_nfold
        metrics['sensitivity'] = sensitivity_nfold
        metrics['precision'] = precision_nfold
    return metrics

def get_roc_curve_allFolds(result, para_grid, clfs, n_fold, result_path, logger):
    """get roc auc value and curve for all folds and save in result path
    Args:
        result: return dict from hyperopt
        para_grid: list of dict, parameter grids
        clfs: list of strings, candidate classifier names
        n_fold: int, number of folds
        result_path: string, result path
        logger: logger
    Returns:
        prepro_average_roc, dictionary of average rocs
        prepro_stdev_roc, dictionary of roc auc stdevs
        prepro_metrics, dictionary of all metrics
    """
    assert result != None
    prepro_average_roc = []
    prepro_stdev_roc = []
    prepro_metrics = []
    for para_index in range(len(para_grid)):
        print ('#################################')
        print ('for preprocess strategy number')
        print (para_index)
        logger.info('#################################')
        logger.info('for preprocess strategy number')
        logger.info(para_index)
        average_roc_aucs = dict()
        roc_auc_stdev = dict()
        metrics = dict()
        if clfs == None:
            logger.info('no classifers have been set')
            return
        for clf in clfs:
            n_fold_roc_auc, average_roc_auc = plot_roc_curve(result, n_fold, para_index, clf, file = result_path)
            metrics_n_fold = metrics_table(result, n_fold, para_index, clf)
            stdev_n_fold = statistics.stdev(n_fold_roc_auc)
            average_roc_aucs[clf] = average_roc_auc
            roc_auc_stdev[clf] = stdev_n_fold
            metrics[clf] = metrics_n_fold
        prepro_average_roc.append(average_roc_aucs)
        prepro_stdev_roc.append(roc_auc_stdev)
        prepro_metrics.append(metrics)
        print ('ROC AUCS')
        print (average_roc_aucs)
        print ('ROC AUC stdev')
        print (roc_auc_stdev)
        print ('all metrics')
        print (metrics)
        logger.info('result aucs for different preprocess methods')
        logger.info('ROC AUCS')
        logger.info(average_roc_aucs)
        logger.info('ROC AUC stdev')
        logger.info(roc_auc_stdev)
        logger.info('all metrics')
        logger.info(metrics)
    return prepro_average_roc, prepro_stdev_roc, prepro_metrics

def sort_models(average_roc_aucs, roc_auc_stdev, filename, prepro_index, sort_plot = True):
    """
    plot average roc auc values of model ranks on validation data
    Args:
        average_roc_aucs: list of average roc auc value across cross validations
        roc_auc_stdev: list of roc auc stdev value
        filename: string
        prepro_index: int
        sort_plot: boolean
    Returns:
        a: dict, sorted average roc auc
        sorted_model_name: list, sorted model name
    """
    a = sorted(average_roc_aucs.items(), key=lambda x: x[1], reverse = True)
    sorted_model_name = [e[0] for e in a]
    sorted_roc = [e[1] for e in a]
    print (sorted_model_name)
    sorted_stdev = []
    for x in sorted_model_name:
        sorted_stdev.append(roc_auc_stdev[x])
    if sort_plot:
        x_pos = np.arange(len(sorted_model_name))
        fig, ax = plt.subplots()
        ax.bar(x_pos, sorted_roc, yerr=sorted_stdev, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('mean ROC AUC value')
        print (x_pos)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_model_name)
        ax.set_title('Model rank on validation data of cross validation')
        ax.yaxis.grid(True)
        # Save the figure and show
        plt.tight_layout()
        for i, v in enumerate(sorted_roc):
            plt.text(x_pos[i]-0.25, v+0.01, str("%.2f"%v))
        save_name = filename + '/prepro_index_'+ str(prepro_index) + '_model_leaderboard_with_error_bars.png'
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.close()
    return a, sorted_model_name

def test_clf_on_testData(para_grid, n_fold, clfs, result, result_path, isFeatureReduction, result_logger):
    """ Test the fine-tuned impute missing classifiers on test data for each preprocessing pipeline and save plot of prediction result on test data
    Args:
        para_grid: dictionary of preprocessing parameters
        n_fold: int
        clfs: list of classifier strings
        result: dictionary, result
        result_path: string, result saving path
        isFeatureReduction: boolean, use feature reduction or not
        result_logger: log file
    """
    for para_index in range(len(para_grid)):
        print ('each preprocessing loop')
        print (para_index)
        result_logger.info('para_index')
        result_logger.info(para_index)
        X_train, y_train, X_test, y_test= autoML_util.get_preprocessed_data(result_path, para_index)
        for clf_string in clfs:
            print (clf_string)
            result_logger.info('clf')
            result_logger.info(clf_string)
            bm = result['para_index_'+str(para_index)]['bm']['bm_'+clf_string]
            prepro1 = bm['preprocs'][0]
            if isFeatureReduction:
                prepro2 = bm['preprocs'][1]
            clf = bm['learner']

            X_train = prepro1.fit_transform(X_train)
            if isFeatureReduction:
                X_train = prepro2.fit_transform(X_train)
            clf = clf.fit(X_train, y_train.to_numpy().ravel())

            X_test = prepro1.transform(X_test)
            if isFeatureReduction:
                X_test = prepro2.transform(X_test)
            preds_probs = clf.predict_proba(X_test)

            roc_title = 'ROC Curves of ' + clf_string + ' on para index ' + str(para_index) + ' test data'
            if len(np.unique(y_test)) == 2:
                auc_score = roc_auc_score(y_test.to_numpy(), preds_probs[:,1])
                autoML_util.single_roc_plot(y_test.to_numpy(), preds_probs[:,1], text=str(para_grid[para_index]), title=roc_title)
                plt.tight_layout()
                plt.savefig(result_path + '/'+ str(n_fold) + 'folds_' + str(para_index) + 'prepro_' + clf_string + '_roc.png', dpi=300)
                plt.close()
                print (auc_score)
                result_logger.info('auc_score on test dataset')
                result_logger.info(auc_score)
            else:
                temp_y_test = label_binarize(y_test, classes=np.unique(y_test)).tolist()                
                temp_y_test = np.swapaxes(np.array(temp_y_test), 0, 1)
                preds_probs = np.swapaxes(np.array(preds_probs), 0, 1)
                ax, n_fold_roc_auc, average_roc_auc = autoML_util.plot_roc(temp_y_test, preds_probs, np.unique(y_test), title = roc_title, average_plot = False)
                plt.savefig(result_path + '/'+ str(n_fold) + 'folds_' + str(para_index) + 'prepro_' + clf_string + '_roc.png', dpi=300)
                plt.close()
                result_logger.info('auc_score on test dataset')
                result_logger.info(n_fold_roc_auc)

def get_sorted_model_each_grid(para_grid, n_fold, clfs, result, result_path, isFeatureReduction, logger, prepro_average_roc, prepro_stdev_roc, useDropColFeaImport=False):
    """ get sorted impute missing model in each preprocessing parameter grid and evaluate feature importance rank
    Args:
        para_grid: dictionary of preprocessing parameters
        n_fold: int
        clfs: classifiers
        result: result
        result_path: result saving path
        isFeatureReduction: boolean, use feature reduction of not
        logger: log file 
        prepro_average_roc: list of average roc value
        prepro_stdev_roc: list of standard deviation value
        useDropColFeaImport: boolean
    """
    for para_index in range(len(para_grid)):
        print ('for para_index')
        print (para_index)
        logger.info('para_index')
        logger.info(para_index)
        average_roc_aucs = prepro_average_roc[para_index]
        roc_auc_stdev = prepro_stdev_roc[para_index]
        if average_roc_aucs != 0:
            sorted_model_dic, sorted_model_name=sort_models(average_roc_aucs, roc_auc_stdev, result_path, para_index)
            logger.info('para_index')
            logger.info(para_index)
            logger.info('sorted models')
            logger.info(sorted_model_dic)
        if sorted_model_name[0] == 'XGboost' or sorted_model_name[0] == 'RF' or sorted_model_name[0] == 'ET' or sorted_model_name[0] == 'AB' or sorted_model_name[0] == 'GB':
            clf_string = sorted_model_name[0]
            modelInterpreter.shap_analysis(para_index, result, result_path, isFeatureReduction, clf_string)

    test_clf_on_testData(para_grid, n_fold, clfs, result, result_path, isFeatureReduction, logger)
    for para_index in range(len(para_grid)):
        features_importance = modelInterpreter.sklearn_feature_importance(result, result_path, para_index, isFeatureReduction)
        logger.info('parameter index')
        logger.info(para_index)
        logger.info('sklearn feature importance')
        logger.info(features_importance)
# another way to rank feature importance
        if useDropColFeaImport:
            features_importance = modelInterpreter.drop_col_feature_importance_all_models(result, result_path, para_index, isFeatureReduction)
            logger.info('parameter index')
            logger.info(para_index)
            logger.info('drop column feature importance')
            logger.info(features_importance)

def test_clf_on_testData_allowMissing(para_grid, n_fold, clfs, result, result_path, result_logger):
    """ Test the fine-tuned classifier on test data for each preprocessing method and plot the roc auc score ranks on test data
    Args:
        para_grid: dictionary of preprocessing parameters
        n_fold: int
        clfs: list of classifiers (xgboost)
        result: result
        result_path: string, result saving path
        result_logger: log file
    """
    for para_index in range(len(para_grid)):
        print ('each preprocessing loop')
        print (para_index)
        result_logger.info('para_index')
        result_logger.info(para_index)
        X_train, y_train, X_test, y_test= autoML_util.get_preprocessed_data(result_path, para_index)
        for clf_string in clfs:
            print (clf_string)
            result_logger.info('clf')
            result_logger.info(clf_string)
            bm = result['para_index_'+str(para_index)]['bm']['bm_'+clf_string]
            ss = StandardScaler()
            X_train_ss = ss.fit_transform(X_train)
            preprocs = bm['preprocs']
            clf = bm['learner']
            clf = clf.fit(X_train_ss, y_train.to_numpy().ravel())
            X_test_ss = ss.transform(X_test)
            preds_probs = clf.predict_proba(X_test_ss)

            roc_title = 'ROC Curves of ' + clf_string + ' on para index ' + str(para_index) + ' test data'
            if len(np.unique(y_test)) == 2:
                auc_score = roc_auc_score(y_test.to_numpy(), preds_probs[:,1])                
                autoML_util.single_roc_plot(y_test.to_numpy(), preds_probs[:,1], text=str(para_grid[para_index]), title=roc_title)
                plt.tight_layout()
                plt.savefig(result_path + '/'+ str(n_fold) + 'folds_' + str(para_index) + 'prepro_' + clf_string + '_roc.png', dpi=300)
                plt.close()
                print (auc_score)
                result_logger.info('auc_score on test dataset')
                result_logger.info(auc_score)
            else:
                temp_y_test = label_binarize(y_test, classes=np.unique(y_test)).tolist()                
                temp_y_test = np.swapaxes(np.array(temp_y_test), 0, 1)
                preds_probs = np.swapaxes(np.array(preds_probs), 0, 1)
                ax, n_fold_roc_auc, average_roc_auc = autoML_util.plot_roc(temp_y_test, preds_probs, np.unique(y_test), title = roc_title, average_plot = False)
                plt.savefig(result_path + '/'+ str(n_fold) + 'folds_' + str(para_index) + 'prepro_' + clf_string + '_roc.png', dpi=300)
                plt.close()
                result_logger.info('auc_score on test dataset')
                result_logger.info(n_fold_roc_auc)

def get_sorted_model_each_grid_allowMissing(para_grid, n_fold, clfs, result, result_path, logger, prepro_average_roc, prepro_stdev_roc, useDropColFeaImport=False):
    """ get sorted allow missing model in each preprocessing parameter grid and evaluate feature importance rank
    Args:
        para_grid: dictionary of preprocessing parameters
        n_fold: int
        clfs: classifiers
        result: result
        result_path: result saving path
        logger: log file 
        prepro_average_roc: list of average roc value
        prepro_stdev_roc: list of standard deviation value
        useDropColFeaImport: boolean
    """
    for para_index in range(len(para_grid)):
        print ('for para_index')
        print (para_index)
        logger.info('para_index')
        logger.info(para_index)
        average_roc_aucs = prepro_average_roc[para_index]
        roc_auc_stdev = prepro_stdev_roc[para_index]
        if average_roc_aucs != 0:
            sorted_model_dic, sorted_model_name=sort_models(average_roc_aucs, roc_auc_stdev, result_path, para_index)
            logger.info('para_index')
            logger.info(para_index)
            logger.info('sorted models')
            logger.info(sorted_model_dic)
        if sorted_model_name[0] == 'XGboost':
            clf_string = sorted_model_name[0]
            modelInterpreter.shap_analysis_allowMissing(para_index, result, result_path, clf_string)
    test_clf_on_testData_allowMissing(para_grid, n_fold, clfs, result, result_path, logger)
    for para_index in range(len(para_grid)):
        features_importance = modelInterpreter.sklearn_feature_importance(result, result_path, para_index, False)
        logger.info('parameter index')
        logger.info(para_index)
        logger.info('features importance')
        logger.info(features_importance)
        # another way to rank feature importance
        if useDropColFeaImport:
            dc_features_importance = modelInterpreter.drop_col_feature_importance_all_models(result, result_path, para_index, False)
            logger.info('parameter index')
            logger.info(para_index)
            logger.info('features importance')
            logger.info(dc_features_importance)
