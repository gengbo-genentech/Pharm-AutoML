import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import autoML_util
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeaturePreprocessors():
    '''
    Feature preprocessing class
    Args:
        X: pandas dataframe
        y: pandas series
        result_path: string
        grid_index: int
        drop_features: list of strings
    '''
    def __init__(self, X, y, preprocessing_params, result_path=None, result_logger=None, savePlot = False, isDisplay = False):
        self.X = X
        self.y = y
        self.base_features = list(X.columns)
        self.preprocessing_params = preprocessing_params
        self.preprocessing_params['drop_features'] = list(self.preprocessing_params['drop_features'])

        self.savePlot = savePlot
        self.isDisplay = isDisplay
        if self.isDisplay == False:
            matplotlib.use('Agg')

        # Dictionary to hold removed columns
        self.ops_cols = {}
        self.ops_rows = {}

        self.nan_y_index = None
        self.unique_stats = None
        self.single_unique_features = None

        # record missing cols or rows
        self.missing_threshold = None
        self.record_missing_cols = None
        self.record_missing_rows = None
        self.missing_stats_cols = None
        self.missing_stats_rows = None

        self.one_hot_correlated = False
        self.one_hot_features = None
        self.record_collinear = None
        self.corr_matrix = None

        self.all_identified = None

    def feature_preprocessor_main(self):
        """
        Args:
            params: dictionary of parameters
        return:
            X_important: pandas dataframe, selected dataframe after feature reduction
            y_important: pandas series, target
        """
        removed_features, removed_rows = self.find_removableFeatures(self.preprocessing_params)
        reserved_features = list(set(self.X.columns) - set(removed_features))
        reserved_rows = list(set(self.X.index) - set(removed_rows))
        X_important = self.X.loc[:, reserved_features]
        X_important = X_important.loc[reserved_rows, :]
        y_important = self.y.loc[reserved_rows]
        return X_important, y_important

    def find_removableFeatures(self, selection_params):
        """
        feature removal pipeline, remove all none values in column y
        find the columns have unique values and remove them
        remove columns which missing fraction is higher than threshold
        find columns that are corelated to each other
        Find the number of features identified to drop
        Args:
            selection_params: dict, preprocessing parameters
            eg:
                preprocessing_paras = {}
                preprocessing_paras['missing_threshold_rows']=0.2
                preprocessing_paras['missing_threshold_columns']=0.2
                preprocessing_paras['PCA_featureSelection_orNot']=False
                preprocessing_paras['PCA_feature_num']=None
                preprocessing_paras['correlation_threshold']=0.98
        """
        self.find_none_y()
        self.find_single_unique()
        if 'missing_threshold_rows' in selection_params.keys():
            self.find_missing(selection_params['missing_threshold_rows'], axis = 0)
        if 'missing_threshold_columns' in selection_params.keys():
            self.find_missing(selection_params['missing_threshold_columns'], axis = 1)
        if 'correlation_threshold' in selection_params.keys():
            self.find_collinear(selection_params['correlation_threshold'])
        else:
            self.find_collinear(0.98)
        if self.savePlot:
            if set(self.record_collinear['drop_feature']):
                self.plot_collinear()
            else:
                self.plot_collinear(plot_all = True)
        self.all_identified_cols = set(list(chain(*list(self.ops_cols.values()))))
        self.n_identified_cols = len(self.all_identified_cols)
        print('Data Preprocessor Summary')
        print('%d total features out of %d identified for removal.' % (self.n_identified_cols, self.X.shape[1]))
        print(self.ops_cols)
        self.all_identified_rows = set(list(chain(*list(self.ops_rows.values()))))
        self.n_identified_rows = len(self.all_identified_rows)
        print('%d total instances out of %d identified for removal.' % (self.n_identified_rows, self.X.shape[0]))
        print(self.ops_rows)
        return self.all_identified_cols, self.all_identified_rows

    def find_none_y(self, isRemove = True):
        """ check if there is nan value in y
        """
        self.nan_y_index = self.y.loc[np.array(pd.isna(self.y))].index
        if len(self.nan_y_index) != 0 and isRemove == True:
            self.X = self.X.drop(self.nan_y_index)
            self.y = self.y.drop(self.nan_y_index)

    def find_single_unique(self):
        """find the columns with single unique values"""
        unique_counts = self.X.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns = {'index': 'feature', 0: 'nunique'})
        self.unique_stats = self.unique_stats.sort_values('nunique', ascending = True)
        # Find the columns with only one unique count
        single_unique_features = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
        to_drop = list(single_unique_features['feature'])
        self.single_unique_features = single_unique_features
        self.ops_cols['drop_features'] = self.preprocessing_params['drop_features']
        if self.ops_cols['drop_features'] == [None]:
            length = 0
        else:
            length = len(self.ops_cols['drop_features'])
        drop_info = 'manually drop %d features.' % length
        print('Preprocessing Step 1: Manually Drop Features')
        print(drop_info)

        self.ops_cols['single_unique'] = to_drop
        single_unique_info = '%d features with a single unique value.' % len(self.ops_cols['single_unique'])
        print('Preprocessing Step 2: Find Single Unique Features')
        print(single_unique_info)

    def find_missing(self, missing_threshold, axis = 1):
        """save the missing columns higher than threshold in self.ops_cols or self.ops_rows
        Args:
            missing_threshold: float, missing tolerance
            axis: 0 or 1
        """
        if axis == 1:
            self.missing_threshold_columns = missing_threshold
        elif axis == 0:
            self.missing_threshold_rows = missing_threshold
        a = 1-axis
        missing_series = self.X.isnull().sum(axis = a) / self.X.shape[a]
        # Calculate the fraction of missing in each column 
        # missing_series = self.X.isnull().sum() / self.X.shape[0]
        if axis == 1:
            self.missing_stats_cols = pd.DataFrame(missing_series).rename(columns = {'index': 'feature', 0: 'missing_fraction'})
            # Sort with highest number of missing values on top
            self.missing_stats_cols = self.missing_stats_cols.sort_values('missing_fraction', ascending = False)
        elif axis == 0:
            self.missing_stats_rows = pd.DataFrame(missing_series).rename(columns = {'index': 'feature', 0: 'missing_fraction'})
            # Sort with highest number of missing values on top
            self.missing_stats_rows = self.missing_stats_rows.sort_values('missing_fraction', ascending = False)
        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns = {'index': 'cols_or_rows', 0: 'missing_fraction'})
        if axis == 1:
            to_drop = list(record_missing['cols_or_rows'])
            self.record_missing_cols = record_missing
            self.ops_cols['missing_features'] = to_drop
            missing_info = '%d features have more than %0.2f fraction of missing values.' % (len(self.ops_cols['missing_features']), missing_threshold)
        elif axis == 0:
            to_drop = list(record_missing['cols_or_rows'])
            self.record_missing_rows = record_missing
            self.ops_rows['missing_instances'] = to_drop
            missing_info = '%d instances have more than %0.2f fraction of missing values.' % (len(self.ops_rows['missing_instances']), missing_threshold)
        else:
            raise ValueError('axis in function find_missing should be 1 or 0!')
        print('Preprocessing Step 3: Find Missing')
        print(missing_info)

    def plot_missing(self, axis = 1, show=True, **figure_kwargs):
        """Histogram of missing fraction in each feature"""
        if axis == 1:
            if self.record_missing_cols is None:
                raise NotImplementedError("Columns missing values have not been calculated. Run `find_missing with axis = 1`")
        elif axis == 0:
            if self.record_missing_rows is None:
                raise NotImplementedError("Rows missing values have not been calculated. Run `find_missing with axis = 0`")
        self.reset_plot()
        # Histogram of missing values
        plt.style.use('seaborn-white')
        plt.figure(figsize = (7, 5))
        if axis == 1:
            plt.hist(self.missing_stats_cols['missing_fraction'], bins = np.linspace(0, 1, 11), edgecolor = 'k', linewidth = 0.5)
        elif axis == 0:
            plt.hist(self.missing_stats_rows['missing_fraction'], bins = np.linspace(0, 1, 11), edgecolor = 'k', linewidth = 0.5)
        plt.xticks(np.linspace(0, 1, 11))
        plt.xlabel('Missing Fraction', size = 14)
        plt.ylabel('Count of Features', size = 14)
        plt.title("Fraction of Missing Values Histogram", size = 16)
        missing_plot = self.prepro_dir + '/missing.png'
        plt.savefig(missing_plot, dpi=300)
        if self.isDisplay == False:
            plt.close()

    def find_collinear(self, correlation_threshold, one_hot=False):
        """find the columns that correlate to each other"""
        pd.options.mode.chained_assignment = None
        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot
         # Calculate the correlations between every column
        if one_hot:
            # One hot encoding
            features = pd.get_dummies(self.X)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]
            # Add one hot encoded data to original data
            self.data_all = pd.concat([features[self.one_hot_features], self.X], axis = 1)
            corr_matrix = pd.get_dummies(features).corr()
        else:
            corr_matrix = self.X.corr()
        self.corr_matrix = corr_matrix
        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])
        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:
            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])
            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]
            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features, 'corr_feature': corr_features, 'corr_value': corr_values})
            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index = True)
        self.record_collinear = record_collinear
        self.ops_cols['collinear'] = to_drop
        collinear_info = '%d features with a correlation magnitude greater than %0.2f.' % (len(self.ops_cols['collinear']), self.correlation_threshold)
        print('Preprocessing Step 4: Find Collinear Features by Pearson Correlations')
        print(collinear_info)

    def plot_collinear(self, plot_all = False):
        '''plot collinear features'''
        if self.record_collinear is None:
            raise NotImplementedError('Collinear features have not been idenfitied. Run `find_collinear`.')
        if plot_all:
          corr_matrix_plot = self.corr_matrix
          title = 'All Correlations'
        else:
            # Identify the correlations that were above the threshold
            # columns (x-axis) are features to drop and rows (y_axis) are correlated pairs
            corr_matrix_plot = self.corr_matrix.loc[list(set(self.record_collinear['corr_feature'])), list(set(self.record_collinear['drop_feature']))]
            title = "Correlations Above Threshold"
        f, ax = plt.subplots(figsize=(15, 12))
        # Diverging colormap
        cmap = sns.diverging_palette(300, 20, as_cmap=True)
        # Draw the heatmap with a color bar
        sns.heatmap(corr_matrix_plot, cmap=cmap, center=0, linewidths=.25, cbar_kws={"shrink": 0.6})
        # Set the ylabels 
        ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
        ax.set_yticklabels(list(corr_matrix_plot.index), size = int(150 / corr_matrix_plot.shape[0]));
        # Set the xlabels 
        ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
        ax.set_xticklabels(list(corr_matrix_plot.columns), size = int(150 / corr_matrix_plot.shape[1]));
        plt.title(title, size = 14)
        plt.tight_layout()
        collinear_plot = self.prepro_dir + '/collinear.png'
        plt.savefig(collinear_plot, dpi=500)
        if self.isDisplay == False:
            plt.close()

    def reset_plot(self):
        plt.rcParams = plt.rcParamsDefault