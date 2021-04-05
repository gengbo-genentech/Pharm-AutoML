import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
import autoML_util
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Imputer():
    """
    Imputation main class

    Attributes:
        data: input of auto ml classifier, a pandas dataframe with features in columns and subjects in row
        grid_index: int
    """
    def __init__(self, data, grid_index, manual_categ_features=[], logger=None, labels=None, allDiff_threshold = 0.1):
        # Dataset and optional training labels
        self.data = data
        self.labels = labels
        self.logger = logger
        self.data_imputed = data
        if labels != None and labels.isnull().sum() >= 1:
            print ('There are missing values in your predicting target, Please remove that record')

        # categorical variable and numerical variable
        self.col_types = {}
        self.col_types['categorical'] = []
        self.col_types['numerical'] = []

        for col in self.data.columns:
            if is_string_dtype(self.data[col]):
                if self.col_types['categorical'] == []:
                    self.col_types['categorical'] = [col]
                else:
                    self.col_types['categorical'].append(col)
            elif is_numeric_dtype(self.data[col]):
                if self.col_types['numerical'] == []:
                    self.col_types['numerical'] = [col]
                else:
                    self.col_types['numerical'].append(col)

        if manual_categ_features != []:
            print (manual_categ_features)
            self.col_types['categorical'] = list(set(self.col_types['categorical']+manual_categ_features))
            self.col_types['numerical'] = list(set(self.col_types['numerical'])-set(manual_categ_features))
        self.allDifferentCols = self.check_allDifferentData(threshold = allDiff_threshold)

        self.all_nan_cols = self.nan_num().index
        self.nan_cols = {}
        self.nan_cols['categorical'] = []
        self.nan_cols['numerical'] = []
        for c in self.col_types['categorical']:
            if c in self.all_nan_cols:
                self.nan_cols['categorical'].append(c)
        for n in self.col_types['numerical']:
            if n in self.all_nan_cols:
                self.nan_cols['numerical'].append(n)

        self.ops_PCA_cols = None

    def impute_missings_pipeline(self, num_strategy="mean", cat_strategy="most_frequent", add_nan_label=False):
        """
        1. find all different categorical columns (eg. names) and remove those
        2. add nan label and impute categorical columns
        3. add nan label and impute numerical columns
        4. one-hot encoding on all categorical columns
        Args:
            X: dataframe
            strategy: string, mean or median
            grid_index: dict, actual param grid
            check_diff_threshold: unique categorical value percentage for features (too many unique categorical value may cause generation of lots of one-hot features)
        Returns:
            X_important: pd dataframe
            y: pd dataframe
        """
        if self.nan_cols['categorical'] != []:
            if add_nan_label == True:
                self.add_nan_label(self.nan_cols['categorical'])
            self.impute_categorical_col(category_cols=self.nan_cols['categorical'], strategy = cat_strategy)

        if self.nan_cols['numerical'] != []:
            if add_nan_label == True:
                self.add_nan_label(self.nan_cols['numerical'])
            self.impute_numerical_col(numerical_cols=self.nan_cols['numerical'], strategy = num_strategy)
        print ('data imputation summary')
        print (self.nan_cols)
        if self.logger != None:
            self.logger.info('data imputation summary')
            self.logger.info(self.nan_cols)
        one_hotted_df = pd.get_dummies(self.data_imputed, columns=self.col_types['categorical'])
        return one_hotted_df

    def impute_allowMissings_pipeline(self):
        # impute pipeline for allow missing models
        one_hotted_df = pd.get_dummies(self.data_imputed, columns=self.col_types['categorical'])
        return one_hotted_df

    def check_allDifferentData(self, threshold = 0.1):
        # find all different categorical columns (eg. names) and remove those
        row_num = self.data.shape[0]
        allDiffCols = []
        for col in self.col_types['categorical']:
            unique_num = len(list(self.data[col].unique()))
            unique_rate = unique_num/row_num
            if unique_rate > threshold:
                allDiffCols.append(col)
        self.col_types['categorical'] = list(set(self.col_types['categorical'])-set(allDiffCols))
        self.data_imputed = self.data_imputed.drop(allDiffCols, axis =1)
        allDiff_info = '%d categorical features are dropped due to high number of different values in categorical feature.' % (len(allDiffCols))
        print (allDiff_info)
        print (allDiffCols)
        if self.logger != None:
            self.logger.info(allDiff_info)
            self.logger.info(allDiffCols)
        return allDiffCols

    def impute_categorical_col(self, category_cols = None, strategy='unknown'):
        '''
        if there is missing value in category column, impute "unknown" or "most_frequent" at missing first
        impute categorical columns
        Args:
            category_cols: list of strings, list of feature names
            strategy: string, unknown, most_frequent
        '''
        if category_cols == None:
            category_cols = self.nan_cols['categorical']
        else:
            category_cols = list(set(self.nan_cols['categorical'] + category_cols))

        print ('the imputed categorical features include ' + ', '.join(map(str, category_cols)))
        if self.logger != None:
            self.logger.info('the imputed categorical features include ' + ', '.join(map(str, category_cols)))
        for col in category_cols:
            if strategy == "most_frequent":
                imputer = SimpleImputer(strategy="most_frequent")
                self.data_imputed[col]=imputer.fit_transform(np.expand_dims(self.data[col].to_numpy(),axis=1))
            elif strategy == "unknown":
                imputer = SimpleImputer(strategy='constant', fill_value='unknown')
                self.data_imputed[col]=imputer.fit_transform(np.expand_dims(self.data[col].to_numpy(),axis=1))

    def impute_numerical_col(self, numerical_cols = None, strategy='mean'):
        ''' impute numerical columns
        Args:
            numerical_cols: list of strings, list of feature names
            strategy: string, mean or median
        '''
        if numerical_cols == None:
            numerical_cols = self.nan_cols['numerical']
        print ('the imputed numerical features include' + ', '.join(map(str, numerical_cols)))
        if self.logger != None:
            self.logger.info('the imputed numerical features include' + ', '.join(map(str, numerical_cols)))
        for col in numerical_cols:
            imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
            self.data_imputed[col]=imputer.fit_transform(self.data[col].values.reshape(-1, 1))

    def nan_num(self, axis = 1):
        # number of null in each columns
        a = 1-axis
        countNan=self.data.isna().sum(axis = a)
        nan_count = countNan.iloc[countNan.to_numpy().nonzero()[a]]
        return nan_count

    def add_nan_label(self, list_col):
        ''' add nan label on all columns have nan
        Args:
            list_col: list of strings, list of feature names
        '''
        out_df = self.data_imputed[:]
        for i in list_col:
            nan_label = i + '_nanLabel'
            nanLabel_df = out_df[i].isna().astype(int)
            out_df[nan_label]=nanLabel_df
        self.data_imputed = out_df

    def find_PCA_features(self, X, feature_num):
        """reduce feature by PCA
        Args:
            X: training data
            feature_num: number of selected features
        Returns:
            X_selected: training data after feature selection
            selected_features: list of string, selected features
            drop_features: list of string, dropped features
        """
        feature_num = int(feature_num)
        ss = StandardScaler()
        X_ss = ss.fit_transform(X)
        pca = PCA()
        print (X_ss.shape)
        pca.fit(X_ss)
        X_new = pca.transform(X_ss)
        X_rec = pca.inverse_transform(X_new)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loading = np.sum(np.abs(loadings), axis=1)
        feature_names = X.columns.values
        ind = sorted(range(len(loading)), key=lambda i: np.abs(loading[i]), reverse=True)[0:feature_num]

        mod_mat = np.concatenate((feature_names[ind].reshape((feature_num, 1)), loading[ind].reshape((feature_num, 1))), axis =1)
        mod_mat = pd.DataFrame(mod_mat, columns = ["feature", "correlation"])
        selected_features = mod_mat['feature'][:feature_num].values
        drop_features = list(set(feature_names).difference(set(selected_features)))
        X_selected = X[selected_features]

        self.ops_PCA_cols = drop_features
        pca_info = '%d features that are removed by PCA.' % len(self.ops_PCA_cols)
        print(pca_info)
        print(self.ops_PCA_cols)
        if self.logger != None:
            self.logger.info(pca_info)
            self.logger.info(self.ops_PCA_cols)

        return X_selected, selected_features, drop_features

    def plot_dists(self, col, title = "Distributions"):
        ''' numerical features 
        Args:
            col: string, column name
        '''
        sns.set(color_codes=True)
        sns.distplot(self.data_imputed[col], label='imputed_'+col, kde=True, rug=True)
        sns.distplot(self.data[col][~np.isnan(self.data[col])], label=col, kde=True, rug=True)
        plt.suptitle(title)
        plt.legend()
        plt.show()
