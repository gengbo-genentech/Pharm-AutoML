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

class Imputation():
    """
    Imputation main class

    Attributes:
        data: input of auto ml classifier, a pandas dataframe with features in columns and subjects in row
        grid_index: int
    """
    def __init__(self, x, y, imputation_params):
        # Dataset and optional training labels
        self.x = x
        self.data_imputed = x
        self.y = y
        self.num_unique_y = len(y.unique())
        self.imputation_params = imputation_params
        self.imputation_params["categorical_features"] = list(self.imputation_params["categorical_features"])

        # categorical variable and numerical variable
        self.col_types = {}
        self.col_types['categorical'] = []
        self.col_types['numerical'] = []
        self.col_type_describe()

        self.all_nan_cols = self.nan_num().index
        self.nan_cols = {}
        self.nan_cols['categorical'] = []
        self.nan_cols['numerical'] = []
        self.nan_col_describe()

        self.allDifferentCols = None

    def col_type_describe(self):
        for col in self.x.columns:
            if is_string_dtype(self.x[col]):
                if self.col_types['categorical'] == []:
                    self.col_types['categorical'] = [col]
                else:
                    self.col_types['categorical'].append(col)
            elif is_numeric_dtype(self.x[col]):
                if self.col_types['numerical'] == []:
                    self.col_types['numerical'] = [col]
                else:
                    self.col_types['numerical'].append(col)
        if self.imputation_params["categorical_features"] != [None] and self.imputation_params["categorical_features"] != None and self.imputation_params["categorical_features"] != []:
            self.col_types['categorical'] = list(set(self.col_types['categorical']+self.imputation_params["categorical_features"]))
            self.col_types['numerical'] = list(set(self.col_types['numerical'])-set(self.imputation_params["categorical_features"]))

    def nan_col_describe(self):
        for c in self.col_types['categorical']:
            if c in self.all_nan_cols:
                self.nan_cols['categorical'].append(c)
        for n in self.col_types['numerical']:
            if n in self.all_nan_cols:
                self.nan_cols['numerical'].append(n)

    def drop_allDifferentCols(self, threshold = 0.1):
        # find all different categorical columns (eg. names) and remove those
        row_num = self.x.shape[0]
        allDiffCols = []
        for col in self.col_types['categorical']:
            unique_num = len(list(self.x[col].unique()))
            unique_rate = unique_num/row_num
            if unique_rate > threshold:
                allDiffCols.append(col)
        self.col_types['categorical'] = list(set(self.col_types['categorical'])-set(allDiffCols))
        self.data_imputed = self.data_imputed.drop(allDiffCols, axis =1)
        allDiff_info = '%d categorical features are dropped due to high number of different values in categorical feature.' % (len(allDiffCols))
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

        print ('the imputed categorical features include: ' + ', '.join(map(str, category_cols)))
        for col in category_cols:
            if strategy == "most_frequent":
                imputer = SimpleImputer(strategy="most_frequent")
                self.data_imputed[col]=imputer.fit_transform(np.expand_dims(self.x[col].to_numpy(),axis=1))
            elif strategy == "unknown":
                imputer = SimpleImputer(strategy='constant', fill_value='unknown')
                self.data_imputed[col]=imputer.fit_transform(np.expand_dims(self.x[col].to_numpy(),axis=1))

    def impute_numerical_col(self, numerical_cols = None, strategy='mean'):
        ''' impute numerical columns
        Args:
            numerical_cols: list of strings, list of feature names
            strategy: string, mean or median
        '''
        if numerical_cols == None:
            numerical_cols = self.nan_cols['numerical']
        print ('the imputed numerical features include: ' + ', '.join(map(str, numerical_cols)))
        for col in numerical_cols:
            imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
            self.data_imputed[col]=imputer.fit_transform(self.x[col].values.reshape(-1, 1))

    def nan_num(self, axis = 1):
        # number of null in each columns
        a = 1-axis
        countNan=self.x.isna().sum(axis = a)
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

    def convert_y(self):
        if self.num_unique_y == 2 and pd.api.types.is_string_dtype(self.y):
            y_one_hotted_df = pd.get_dummies(self.y)
            y_important = y_one_hotted_df.iloc[:,:1]
            return y_important
        else:
            return self.y

    def impute_missings_pipeline(self):
        """
        1. find all different categorical columns (eg. names) and remove those
        2. add nan label and impute categorical columns
        3. add nan label and impute numerical columns
        4. one-hot encoding on all categorical columns
        Args:
            X: dataframe
            strategy: string, mean or median
            check_diff_threshold: unique categorical value percentage for features (too many unique categorical value may cause generation of lots of one-hot features)
        Returns:
            X_important: pd dataframe
            y: pd dataframe
        """
        self.allDifferentCols = self.drop_allDifferentCols(threshold = self.imputation_params['check_diff_categorical_threshold'])
        if self.nan_cols['categorical'] != []:
            if self.imputation_params["add_nan_label"] == True:
                self.add_nan_label(self.nan_cols['categorical'])
            self.impute_categorical_col(category_cols=self.nan_cols['categorical'], strategy = self.imputation_params['impute_category_strategy'])

        if self.nan_cols['numerical'] != []:
            if self.imputation_params["add_nan_label"] == True:
                self.add_nan_label(self.nan_cols['numerical'])
            self.impute_numerical_col(numerical_cols=self.nan_cols['numerical'], strategy = self.imputation_params['impute_numerical_strategy'])
        print ('data imputation summary')
        print (self.nan_cols)
        one_hotted_df = pd.get_dummies(self.data_imputed, columns=self.col_types['categorical'])
        y_one_hotted = self.convert_y()
        return one_hotted_df, y_one_hotted

    def impute_allowMissings_pipeline(self):
        # impute pipeline for allow missing models
        one_hotted_df = pd.get_dummies(self.data_imputed, columns=self.col_types['categorical'])
        y_one_hotted = self.convert_y()
        return one_hotted_df, y_one_hotted
