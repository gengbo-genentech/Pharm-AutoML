import os
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from hyperopt import hp
import pickle
from hyperopt import STATUS_OK
import mlflow

def get_preprocessed_data(result_path, para_index):
    """
    read in preprocessed data
    Args:
        result_path: str, result path.
        fold: int, fold index.
        para_index: int, parameter grid index
    Returns:
        X_train: pd dataframe
        X_test: pd dataframe
        y_train: pd dataframe
        y_test: pd dataframe
        logger_path: str, logger directory
    """
    para_path = result_path + '/' + str(para_index)
    X_train_dir = para_path + '/X_train_important.csv'
    y_train_dir = para_path + '/y_train.csv'
    X_test_dir = para_path + '/X_test_important.csv'
    y_test_dir = para_path + '/y_test.csv'
    X_train = pd.read_csv(X_train_dir, index_col = 0)
    y_train = pd.read_csv(y_train_dir, index_col = 0)
    X_test = pd.read_csv(X_test_dir, index_col = 0)
    y_test = pd.read_csv(y_test_dir, index_col = 0)
    return X_train, y_train, X_test, y_test

def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['best_model']

def get_best_model(paras_json, b_model_dict):
    bm_paras = paras_json
    for i in paras_json.keys():
        for j in paras_json[i].keys():
            if isinstance(paras_json[i][j], list):
                bm_paras[i][j] = bm_paras[i][j][b_model_dict[j]]
    return bm_paras

def hp_parameters(json_parameters):
    hp_paras = {}
    for i in json_parameters.keys():
        sub_paras = {}
        for j in json_parameters[i].keys():
            if j == 'drop_features' or j == 'categorical_features':
                sub_paras[j] = json_parameters[i][j]
            elif isinstance(json_parameters[i][j], list):
                sub_paras[j] = hp.choice(j, json_parameters[i][j])
            else:
                sub_paras[j] = hp.choice(j, [json_parameters[i][j]])
        hp_paras[i] = sub_paras
    return hp_paras

def get_parameter_grid(parameter_comparison):
    parameter_temp = parameter_comparison.copy()
    for i in parameter_comparison.keys():
        parameter_temp[i] = list(ParameterGrid(parameter_comparison[i]))
    parameter_grid = list(ParameterGrid(parameter_temp))
    return parameter_grid

def get_parameter_grid_for_clf(parameter_comparison, impute_missing_mode = True):
    parameter_temp = parameter_comparison.copy()
    for i in parameter_comparison.keys():
        parameter_temp[i] = list(ParameterGrid(parameter_comparison[i]))
    parameter_grid = list(ParameterGrid(parameter_temp))
    clf_param_list = []
    for d in parameter_grid:
        for k, v in d.items():
            if impute_missing_mode == False and k != "XGboost":
                continue
            clf_param = {k:v}
            if clf_param not in clf_param_list:
                clf_param_list.append(clf_param)
    return clf_param_list

def get_multiple_option_parameter(parameter):
    multiple_option_parameter = ["clfs"]
    for i in parameter.keys():
        for j in parameter[i].keys():
            if len(parameter[i][j]) > 1:
                multiple_option_parameter.append(j)
    return multiple_option_parameter

def save_model(para_grid_length, clfs, result, result_path):
    """
    save the classifier model
    Args:
        para_grid_length, 
        clfs, 
        result, 
        result_path
    """
    for para_index in range(para_grid_length):
        bm = result['para_index_'+str(para_index)]['bm']
        for clf_string in clfs:
            bm_name = 'bm_' + clf_string
            model = bm[bm_name]['learner']
            filename = result_path + '/' + str(para_index) + 'prepro_' + clf_string + '.sav'
            pickle.dump(model, open(filename, 'wb'))

def setup_temp_dir(prepro_num, directory):
    """set up data preprocessing directory
    Args:
        prepro_num: int, number of parameter grids
        directory: str, result path
    """
    for i in range (prepro_num):
        path = directory + '/' + str(i)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
        os.mkdir(path)

def get_fold_path(path, fold):
    """set up fold path
    Args:
        path: string, name of path
        fold: int, fold number
    """
    fold_i = fold + 1
    file_fold = path + '/fold_' + str(fold_i)
    return file_fold

def missing_stats(X, missing_threshold, axis=1):
    """
    Calculate and sort the fraction of missing in each column 
    Args:
        X: dataframe
        missing_threshold: float
    Returns:
        missing_threshold_rows_grid: list of float
    """
    a = 1-axis
    missing_series = X.isnull().sum(axis = a) / X.shape[a]
    # Calculate the fraction of missing in each column 
    missing_series = X.isnull().sum() / X.shape[0]
    if axis == 1:
        missing_stats_cols = pd.DataFrame(missing_series).rename(columns = {'index': 'feature', 0: 'missing_fraction'})
        # Sort with highest number of missing values on top
        missing_stats_cols = missing_stats_cols.sort_values('missing_fraction', ascending = False)
        missing_threshold_cols_grid = pd.DataFrame(missing_series[missing_series >= missing_threshold]).reset_index().rename(columns = {'index': 'cols', 0: 'missing_fraction'})
        return missing_threshold_cols_grid
    elif axis == 0:
        missing_stats_rows = pd.DataFrame(missing_series).rename(columns = {'index': 'feature', 0: 'missing_fraction'})
        # Sort with highest number of missing values on top
        missing_stats_rows = missing_stats_rows.sort_values('missing_fraction', ascending = False)
        missing_threshold_rows_grid = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns = {'index': 'rows', 0: 'missing_fraction'})
        return missing_threshold_rows_grid

def get_grid_complement_missing_threshold(x, d_list, missing_threshold_complement_mode=False):
    """
    get preprocessing parameter grid for missing imputation pipeline
    Args:
        X: dataframe
        d_list: dictionary of preprocessing parameters
        missing_threshold_complement_mode: complement the missing threshold or not
    Returns:
        grid: list of dictionary, parameter grid
    """
    if missing_threshold_complement_mode == True:
        min_threshold = min(d_list['missing_threshold'])
        output = missing_stats(x, min_threshold)
        if list(output['missing_fraction']) == []:
            d_list['missing_threshold'] = [0]
        else:
            d_list['missing_threshold'] = list(output['missing_fraction'])
    grid = list(ParameterGrid(d_list))
    return grid

def get_grid_allow_missing(x, d_list):
    """
    get preprocessing parameter grid for missing allow pipeline
    Args:
        X: dataframe
        d_list: dictionary of preprocessing parameters
    Returns:
        grid: list of dictionary, parameter grid
    """
    del d_list['impute_category_strategy']
    del d_list['impute_numerical_strategy']
    grid = list(ParameterGrid(d_list))
    return grid

def single_roc_plot(y_true, y_probas, text=None, title='ROC Curves', figsize=None, title_fontsize="large", text_fontsize="medium"):
    """
    generate single roc auc plot
    Args:
        y_true: numpy array, 1 * n.
        y_probas: dataframe, 1 * n.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=title_fontsize)
    fpr, tpr, _ = roc_curve(y_true, y_probas, pos_label=1)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label='ROC curve '
              '(area = {0:0.2f})'.format(roc_auc),
        color='blue', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return fig

def plot_roc(y_true, y_probas, classes = None, title='ROC Curves', average_plot = True,
                   ax=None, figsize=None, cmap='nipy_spectral',
                   title_fontsize="large", text_fontsize="medium"):
    """
    generate roc auc plots for each fold of cross validation experiments and average curve
    Args:
        y_true: list, list of np array.
        y_probas: list, list of np array.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=title_fontsize)
    n_fold_roc_auc = []
    for i in range(len(y_true)):
        fpr, tpr, _ = roc_curve(y_true[i], y_probas[i])
        roc_auc = auc(fpr, tpr)
        color = plt.cm.get_cmap(cmap)(float(i) / len(y_true))
        
        if classes is None:
            s = 'fold'
        else:
            s = classes[i]
        ax.plot(fpr, tpr, lw=2, color=color,
                label='ROC curve of {0} {1} (area = {2:0.2f})'
                      ''.format(s, i, roc_auc))
        n_fold_roc_auc.append(roc_auc)

    average_roc_auc = 0
    if classes is None:
        if average_plot:
            all_y_true = np.concatenate(y_true)
            all_probas = np.concatenate(y_probas)
            fpr_all, tpr_all, _ = roc_curve(all_y_true, all_probas)
            average_roc_auc = auc(fpr_all, tpr_all)
            ax.plot(fpr_all, tpr_all,
                    label='average ROC curve '
                          '(area = {0:0.2f})'.format(average_roc_auc),
                    color='blue', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return ax, n_fold_roc_auc, average_roc_auc

def plot_precision_recall(y_true, y_probas, classes = None, title='Precision-Recall Curve', 
                          average_plot = True, ax=None, figsize=None, cmap='nipy_spectral', 
                          title_fontsize="large", text_fontsize="medium"):
    """
    generate pr auc plots for each fold of cross validation experiments and average curve
    Args:
        y_true: list, list of np array.
        y_probas: list, list of np array.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=title_fontsize)
    n_fold_pr_score = []
    for i in range(len(y_true)):
        precision, recall, _ = precision_recall_curve(y_true[i], y_probas[i])
        average_precision = average_precision_score(y_true[i], y_probas[i])
        color = plt.cm.get_cmap(cmap)(float(i) / len(y_true))
        if classes is None:
            s = 'fold'
        else:
            s = classes[i]
        ax.plot(recall, precision, lw=2, color=color,
                label='Precision-recall curve of {0} {1} (area = {2:0.2f})'
                      ''.format(s, i, average_precision))
        n_fold_pr_score.append(average_precision)
    all_average_precision=0
    if classes is None:
        if average_plot:
            all_y_true = np.concatenate(y_true)
            all_probas = np.concatenate(y_probas)
            precision_all, recall_all, _ = precision_recall_curve(all_y_true, all_probas)
            all_average_precision = average_precision_score(all_y_true, all_probas)
            ax.plot(recall_all, precision_all,
                    label='average Precision-recall curve'
                          '(area = {0:0.2f})'.format(all_average_precision),
                    color='blue', linestyle=':', linewidth=4)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=text_fontsize)
    ax.set_ylabel('Precision', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='best', fontsize=text_fontsize)
    return ax, n_fold_pr_score, all_average_precision