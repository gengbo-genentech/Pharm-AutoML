## thank Christoph Molnar for his interpretable machine learning book
## https://github.com/christophM/interpretable-ml-book
import pandas as pd
import matplotlib.pyplot as plt
import autoML_util
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import plot_partial_dependence
import mlflow

def sort_feature_importances(coef, names, plot_name, n_features = 20):
    """
    sort the feature importance and save plots
    Args:
        coef: list of feature importances
        names: list of feature names
        plot_name: str, plot save name 
        n_features: int
    Returns:
        sorted feature names and importance pair
    """
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    if n_features > len(imp[-n_features:]):
        n_features = len(imp[-n_features:])
    fig = plt.figure(dpi=10*n_features)
    plt.barh(range(n_features), imp[-n_features:], align='center')
    plt.xticks(fontsize=80/n_features+4)
    plt.yticks(range(n_features), names[-n_features:], fontsize = 80/n_features)
    plt.tight_layout()
    mlflow.log_figure(fig, plot_name)
    # plt.savefig(plot_name, dpi=10*n_features)
    plt.close()
    return names, imp

def sklearn_feature_importance(x_train, y_train, best_model, clf_name, isFeatureReduction):
    """
    Evaluate feature importance on sklearn models
    Args:
        result: return dict from hyperopt_newpipeline
        result_path: string
        para_index: int
    Returns:
        feature importance: dict sorted features
    """
    if isFeatureReduction:
        features_names = [str(i+1) for i in range(len(x_train.columns))]
    else:
        features_names = x_train.columns
    features_importance = dict()

    if clf_name == 'LRC':
        clf = best_model['learner']
        lrc_fi = abs(clf.coef_[0])
        lrc_fi = 100.0 * (lrc_fi / lrc_fi.max())
        plot_name = 'sklearn_feature_importance_LRC.png'
        features_names, importance=sort_feature_importances(lrc_fi, list(features_names), plot_name)
        features_importance['LRC'] = features_names, importance

    if clf_name == 'SVC':
        clf = best_model['learner']
        plot_name = 'sklearn_feature_importance_SVC.png'
        if clf.get_params()['kernel'] == 'linear':
            svc=clf.fit(x_train,y_train)
            features_names, importance=sort_feature_importances(sum(abs(svc.coef_)).tolist(), list(features_names), plot_name)
            features_importance['SVC'] = features_names, importance
        else:
            print ('feature importance only works when the kernal of SVC is linear')

    if clf_name == 'RF':
        clf = best_model['learner']
        plot_name = 'sklearn_feature_importance_RF.png'
        features_names, importance=sort_feature_importances(clf.feature_importances_, list(features_names), plot_name)
        features_importance['RF'] = features_names, importance
    if clf_name == 'ET':
        clf = best_model['learner']
        plot_name = 'sklearn_feature_importance_ET.png'
        features_names, importance=sort_feature_importances(clf.feature_importances_, list(features_names), plot_name)
        features_importance['ET'] = features_names, importance
    if clf_name == 'AB':
        clf = best_model['learner']
        plot_name = 'sklearn_feature_importance_AB.png'
        features_names, importance=sort_feature_importances(clf.feature_importances_, list(features_names), plot_name)
        features_importance['AB'] = features_names, importance
    if clf_name == 'GB':
        clf = best_model['learner']
        plot_name = 'sklearn_feature_importance_GB.png'
        features_names, importance=sort_feature_importances(clf.feature_importances_, list(features_names), plot_name)
        features_importance['GB'] = features_names, importance
    if clf_name == 'XGboost':
        clf = best_model['learner']
        plot_name = 'sklearn_feature_importance_XGboost.png'
        features_names, importance=sort_feature_importances(clf.feature_importances_, list(features_names), plot_name)
        features_importance['XGboost'] = features_names, importance
    return features_importance

def drop_col_feature_importance(model, x_train, y_train):
    """
    get feature importance by dropping column method
    Args:
        model: sklearn model
        X_train: pd dataframe
        y_train: pd dataframe
    Returns:
        list of features names
        list of importances
    """
    model_clone = clone(model)
    # training and scoring the benchmark model
    model_clone.fit(x_train, y_train.values.ravel())
    benchmark_score = model_clone.score(x_train, y_train)
    # list for storing feature importances
    importances = []
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in x_train.columns:
        model_clone = clone(model)
        model_clone.fit(x_train.drop(col, axis = 1), y_train.values.ravel())
        drop_col_score = model_clone.score(x_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
    return list(x_train.columns), importances

def drop_col_feature_importance_all_models(x_train, y_train, best_model, clf_name, isFeatureReduction):
    """
    get feature importance for each model using dropping column method
    Args:
        result: return dict from hyperopt_newpipeline
        result_path: string
        para_index: int
        isFeatureReduction: boolean
    Returns:
        feature importance: dict
    """
    features_importance = dict()

    if clf_name == 'LRC':
        clf = best_model['learner']
        plot_name = 'dropCol_feature_importance_LRC.png'
        features_names, importance = drop_col_feature_importance(clf, x_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        LRC_sorted_names, LRC_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['LRC'] = LRC_sorted_names, LRC_sorted_imp
    if clf_name == 'SVC':
        clf = best_model['learner']
        plot_name = 'dropCol_feature_importance_SVC.png'
        features_names, importance = drop_col_feature_importance(clf, x_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        SVC_sorted_names, SVC_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['SVC'] = SVC_sorted_names, SVC_sorted_imp

    if clf_name == 'KNN':
        clf = best_model['learner']
        plot_name = 'dropCol_feature_importance_KNN.png'
        features_names, importance = drop_col_feature_importance(clf, x_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        KNN_sorted_names, KNN_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['KNN'] = KNN_sorted_names, KNN_sorted_imp

    if clf_name == 'RF':
        clf = best_model['learner']
        plot_name = 'dropCol_feature_importance_RF.png'
        features_names, importance = drop_col_feature_importance(clf, x_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        RF_sorted_names, RF_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['RF'] = RF_sorted_names, RF_sorted_imp

    if clf_name == 'ET':
        clf = best_model['learner']
        plot_name = 'dropCol_feature_importance_ET.png'
        features_names, importance = drop_col_feature_importance(clf, x_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        ET_sorted_names, ET_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['ET'] = ET_sorted_names, ET_sorted_imp

    if clf_name == 'AB':
        clf = best_model['learner']
        plot_name = 'dropCol_feature_importance_AB.png'
        features_names, importance = drop_col_feature_importance(clf, x_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        AB_sorted_names, AB_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['AB'] = AB_sorted_names, AB_sorted_imp

    if clf_name == 'GB':
        clf = best_model['learner']
        plot_name = 'dropCol_feature_importance_GB.png'
        features_names, importance = drop_col_feature_importance(clf, x_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        GB_sorted_names, GB_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['GB'] = GB_sorted_names, GB_sorted_imp

    if clf_name == 'SGD':
        clf = best_model['learner']
        plot_name = 'dropCol_feature_importance_SGD.png'
        features_names, importance = drop_col_feature_importance(clf, x_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        SGD_sorted_names, SGD_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['SGD'] = SGD_sorted_names, SGD_sorted_imp

    if clf_name == 'XGboost':
        clf = best_model['learner']
        plot_name = 'dropCol_feature_importance_XGboost.png'
        features_names, importance = drop_col_feature_importance(clf, x_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        XGboost_sorted_names, XGboost_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name)
        features_importance['XGboost'] = XGboost_sorted_names, XGboost_sorted_imp
    return features_importance
