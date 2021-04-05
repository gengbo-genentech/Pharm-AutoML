## thank Christoph Molnar for his interpretable machine learning book
## https://github.com/christophM/interpretable-ml-book
import pandas as pd
import matplotlib.pyplot as plt
import autoML_util
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import plot_partial_dependence

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
    plt.barh(range(n_features), imp[-n_features:], align='center')
    plt.xticks(fontsize=80/n_features+4)
    plt.yticks(range(n_features), names[-n_features:], fontsize = 80/n_features)
    plt.tight_layout()
    plt.savefig(plot_name, dpi=10*n_features)
    plt.close()
    return names, imp

def sklearn_feature_importance(result, result_path, para_index, isFeatureReduction):
    """
    Evaluate feature importance on sklearn models
    Args:
        result: return dict from hyperopt_newpipeline
        result_path: string
        para_index: int
    Returns:
        feature importance: dict sorted features
    """
    clfs_have_feature_imp = ['LRC', 'SVC', 'RF', 'ET', 'AB', 'GB', 'XGboost']
    X_train, y_train, X_test, y_test = autoML_util.get_preprocessed_data(result_path, para_index)
    if isFeatureReduction:
        features_names = [str(i+1) for i in range(len(X_train.columns))]
    else:
        features_names = X_train.columns
    features_importance = dict()

    if 'bm_LRC' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_LRC']['learner']
        lrc_fi = abs(clf.coef_[0])
        lrc_fi = 100.0 * (lrc_fi / lrc_fi.max())
        plot_name = result_path + '/skl_FI_para_' + str(para_index) + '_LRC' + '.png'
        features_names, importance=sort_feature_importances(lrc_fi, list(features_names), plot_name)
        features_importance['LRC'] = features_names, importance
    if 'bm_SVC' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_SVC']['learner']
        plot_name = result_path + '/skl_FI_para_' + str(para_index) + '_SVC' + '.png'
        if clf.get_params()['kernel'] == 'linear':
            svc=clf.fit(X_train,y_train)
            features_names, importance=sort_feature_importances(sum(abs(svc.coef_)).tolist(), list(features_names), plot_name)
        else:
            print ('feature importance only works when the kernal of SVC is linear')
        features_importance['SVC'] = features_names, importance
    if 'bm_RF' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_RF']['learner']
        plot_name = result_path + '/skl_FI_para_' + str(para_index) + '_RF' + '.png'
        features_names, importance=sort_feature_importances(clf.feature_importances_, list(features_names), plot_name)
        features_importance['RF'] = features_names, importance
    if 'bm_ET' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_ET']['learner']
        plot_name = result_path + '/skl_FI_para_' + str(para_index) + '_ET' + '.png'
        features_names, importance=sort_feature_importances(clf.feature_importances_, list(features_names), plot_name)
        features_importance['ET'] = features_names, importance
    if 'bm_AB' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_AB']['learner']
        plot_name = result_path + '/skl_FI_para_' + str(para_index) + '_AB' + '.png'
        features_names, importance=sort_feature_importances(clf.feature_importances_, list(features_names), plot_name)
        features_importance['AB'] = features_names, importance
    if 'bm_GB' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_GB']['learner']
        plot_name = result_path + '/skl_FI_para_' + str(para_index) + '_GB' + '.png'
        features_names, importance=sort_feature_importances(clf.feature_importances_, list(features_names), plot_name)
        features_importance['GB'] = features_names, importance
    if 'bm_XGboost' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_XGboost']['learner']
        plot_name = result_path + '/skl_FI_para_' + str(para_index) + '_XGboost' + '.png'
        features_names, importance=sort_feature_importances(clf.feature_importances_, list(features_names), plot_name)
        features_importance['bm_XGboost'] = features_names, importance
    return features_importance

def drop_col_feature_importance(model, X_train, y_train):
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
    model_clone.fit(X_train, y_train.values.ravel())
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.fit(X_train.drop(col, axis = 1), y_train.values.ravel())
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
    return list(X_train.columns), importances

def drop_col_feature_importance_all_models(result, result_path, para_index, isFeatureReduction):
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
    X_train, y_train, X_test, y_test, logger_path = autoML_util.get_preprocessed_data(result_path, para_index)
    features_importance = dict()

    if 'bm_LRC' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_LRC']['learner']
        plot_name = result_path + '/dropCol_FI_para_' + str(para_index) + '_LRC' + '.png'
        features_names, importance = drop_col_feature_importance(clf, X_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        LRC_sorted_names, LRC_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['LRC'] = LRC_sorted_names, LRC_sorted_imp
    if 'bm_SVC' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_SVC']['learner']
        plot_name = result_path + '/dropCol_FI_para_' + str(para_index) + '_SVC' + '.png'
        features_names, importance = drop_col_feature_importance(clf, X_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        SVC_sorted_names, SVC_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['SVC'] = SVC_sorted_names, SVC_sorted_imp

    if 'bm_KNN' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_KNN']['learner']
        plot_name = result_path + '/dropCol_FI_para_' + str(para_index) + '_KNN' + '.png'
        features_names, importance = drop_col_feature_importance(clf, X_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        KNN_sorted_names, KNN_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['KNN'] = KNN_sorted_names, KNN_sorted_imp

    if 'bm_RF' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_RF']['learner']
        plot_name = result_path + '/dropCol_FI_para_' + str(para_index) + '_RF' + '.png'
        features_names, importance = drop_col_feature_importance(clf, X_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        RF_sorted_names, RF_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['RF'] = RF_sorted_names, RF_sorted_imp

    if 'bm_ET' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_ET']['learner']
        plot_name = result_path + '/dropCol_FI_para_' + str(para_index) + '_ET' + '.png'
        features_names, importance = drop_col_feature_importance(clf, X_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        ET_sorted_names, ET_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['ET'] = ET_sorted_names, ET_sorted_imp

    if 'bm_AB' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_AB']['learner']
        plot_name = result_path + '/dropCol_FI_para_' + str(para_index) + '_AB' + '.png'
        features_names, importance = drop_col_feature_importance(clf, X_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        AB_sorted_names, AB_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['AB'] = AB_sorted_names, AB_sorted_imp

    if 'bm_GB' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_GB']['learner']
        plot_name = result_path + '/dropCol_FI_para_' + str(para_index) + '_GB' + '.png'
        features_names, importance = drop_col_feature_importance(clf, X_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        GB_sorted_names, GB_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['GB'] = GB_sorted_names, GB_sorted_imp

    if 'bm_SGD' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_SGD']['learner']
        plot_name = result_path + '/dropCol_FI_para_' + str(para_index) + '_SGD' + '.png'
        features_names, importance = drop_col_feature_importance(clf, X_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        SGD_sorted_names, SGD_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name, n_features = len(importance))
        features_importance['SGD'] = SGD_sorted_names, SGD_sorted_imp

    if 'bm_XGboost' in result['para_index_'+str(para_index)]['bm'].keys():
        clf = result['para_index_'+str(para_index)]['bm']['bm_XGboost']['learner']
        plot_name = result_path + '/dropCol_FI_para_' + str(para_index) + '_XGboost' + '.png'
        features_names, importance = drop_col_feature_importance(clf, X_train, y_train)
        if isFeatureReduction:
            features_names = [str(i+1) for i in range(len(features_names))]
        XGboost_sorted_names, XGboost_sorted_imp = sort_feature_importances(importance, list(features_names), plot_name)
        features_importance['XGboost'] = XGboost_sorted_names, XGboost_sorted_imp
    return features_importance

def shap_analysis(para_index, result, result_path, isFeatureReduction, clf_string = 'XGboost'):
    """ shap analyze the best model and save the feature importance figures in result_path
    Args:
        para_index: int, preprocessing index number
        result: dictionary, selected model and prediction results
        result_path: string, result saving path
        isFeatureReduction: boolean
        clf_string: XGboost
    """
    X_train, y_train, X_test, y_test= autoML_util.get_preprocessed_data(result_path, para_index)
    bm = result['para_index_'+str(para_index)]['bm']['bm_'+clf_string]
    prepro1 = bm['preprocs'][0]
    if isFeatureReduction:
        prepro2 = bm['preprocs'][1]
    clf = bm['learner']

    X_train_pp = prepro1.fit_transform(X_train)
    if isFeatureReduction:
        X_train_pp = prepro2.fit_transform(X_train_pp)
    clf = clf.fit(X_train_pp, y_train.to_numpy().ravel())
    print ('the following model for shap analysis')
    print (clf)
    explainer = shap.TreeExplainer(clf)

    X_test_pp = prepro1.transform(X_test)
    if isFeatureReduction:
        X_test_pp = prepro2.transform(X_test_pp)

    X_test_pp = pd.DataFrame(data=X_test_pp, columns=X_test.columns)
    try:
        shap_values = explainer.shap_values(X_test_pp)
        shap_title = 'shap plot of ' + clf_string + ' on para index ' + str(para_index) + ' test data'
        # if clf_string != 'ET' or clf_string != 'RF':
        shap.summary_plot(shap_values, X_test_pp)
        plt.tight_layout()
        plt.savefig(result_path + '/shapPlot' + str(para_index) + 'prepro_' + clf_string + '.png', dpi=150)
        plt.close()

        shap_bar_title = 'shap bar plot of ' + clf_string + ' on para index ' + str(para_index) + ' test data'
        if clf_string != 'ET' or clf_string != 'RF':
            shap.summary_plot(shap_values, X_test_pp, plot_type="bar")
            plt.tight_layout()
            plt.savefig(result_path + '/shapBarPlot' + str(para_index) + 'prepro_' + clf_string + '.png', dpi=150)
            plt.close()
    except:
        print ('unsolved shap error, https://github.com/slundberg/shap/issues/941')

def shap_analysis_allowMissing(para_index, result, result_path, clf_string = 'XGboost'):
    """ shap analyze the best model and save the feature importance figures in result_path
    Args:
        para_index: int, preprocessing index number
        result: dictionary, selected model and prediction results
        result_path: string, result saving path
        clf_string: XGboost
    """
    X_train, y_train, X_test, y_test= autoML_util.get_preprocessed_data(result_path, para_index)
    bm = result['para_index_'+str(para_index)]['bm']['bm_'+clf_string]
    ss = StandardScaler()

    X_train_ss = ss.fit_transform(X_train)
    clf = bm['learner']
    clf = clf.fit(X_train_ss, y_train.to_numpy().ravel())
    explainer = shap.TreeExplainer(clf)
    X_test_ss = ss.fit_transform(X_test)
    X_test_ss = pd.DataFrame(data=X_test_ss, columns=X_test.columns)
    shap_values = explainer.shap_values(X_test_ss)

    shap_title = 'shap plot of ' + clf_string + ' on para index ' + str(para_index) + ' test data'
    shap.summary_plot(shap_values, X_test_ss)
    plt.tight_layout()
    plt.savefig(result_path + '/shapPlot' + str(para_index) + 'prepro_' + clf_string + '.png', dpi=150)
    plt.close()
    shap_bar_title = 'shap bar plot of ' + clf_string + ' on para index ' + str(para_index) + ' test data'
    shap.summary_plot(shap_values, X_test_ss, plot_type="bar")
    plt.tight_layout()
    plt.savefig(result_path + '/shapBarPlot' + str(para_index) + 'prepro_' + clf_string + '.png', dpi=150)
    plt.close()
