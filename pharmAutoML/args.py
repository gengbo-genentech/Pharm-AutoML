import argparse
import re

# convert string to boolean
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# convert list of string to list of boolean
def str2bool_list(s):
	if ',' not in s:
		return [str2bool(s)]
	else:
		l = [str2bool(i) for i in re.split(',', s)]
		return l

# convert string to int
def str2int(num):
	if num == 'None':
		return None
	elif isinstance(num, str):
		return int(num)
	elif isinstance(num, int):
		return num
	else:
		raise argparse.ArgumentTypeError('Number expected.')

# convert list of string to list of float
def str2int_list(s):
	if s == 'None' or s == None:
		return [None]
	elif ',' not in s:
		return [int(s)]
	else:
		l = [int(i) for i in re.split(',', s)]
		return l

# convert list of string to list of float
def str2float_list(s):
	if ',' not in s:
		return [float(s)]
	else:
		l = [float(i) for i in re.split(',', s)]
		return l

# convert string to list of string
def str2string_list(s):
	if s == 'None' or s == None:
		return [None]
	elif ',' not in s:
		return [str(s)]
	else:
		l = [str(i) for i in re.split(',', s)]
		return l

parser = argparse.ArgumentParser("PharmAutoML")

parser.add_argument("--data", type=str, help="data file for processing")
parser.add_argument("--target", type=str, help="y column")
parser.add_argument("--result_path", type=str, help="list of classifiers")
parser.add_argument("--impute_missing_mode", type=str2bool, help="allow data without imputation")
parser.add_argument("--tuning_mode", type=str2bool, help="using tuning mode")
parser.add_argument("--loss_function", type=str, help="using tuning mode")

# preprocessing parameters
parser.add_argument("--drop_features", type=str2string_list, help="features that need to be removed manually")
parser.add_argument("--missing_threshold_rows", type=str2float_list, required=True, help='list of missing threshold in rows')
parser.add_argument("--missing_threshold_columns", type=str2float_list, required=True, help='list of missing threshold in columns')
parser.add_argument("--correlation_threshold", type=str2float_list, required=True, help='list of correlation threshold')

# imputation parameters
parser.add_argument("--categorical_features", type=str2string_list, help="feature columns require one-hot encoding")
parser.add_argument("--check_diff_categorical_threshold", type=str2float_list, required=True, help='remove the categorical features that have all different value')
parser.add_argument("--impute_category_strategy", type=str2string_list, help="add new feature columns for columns have nan values")
parser.add_argument("--impute_numerical_strategy", type=str2string_list, help="add new feature columns for columns have nan values")
parser.add_argument("--add_nan_label", type=str2bool_list, help="add new feature columns for columns have nan values")

# feature selection parameters
parser.add_argument("--feature_selection_method", type=str2string_list, help='feature selection method')

# train_test_split_parameters
parser.add_argument("--tts_random_seed", type=str2int_list, help='train test split random seed')
parser.add_argument("--test_size", type=str2float_list, help='train test split size')

# classifier
parser.add_argument("--PCA", type=str2bool_list, help='use PCA to reduce feature dimension or not')
parser.add_argument("--PCA_feature_num", type=str2int_list, help='number of PCA features')
parser.add_argument("--clfs", type=lambda s: re.split(',', s), required=True, help='list of classifiers')
parser.add_argument("--n_fold", type=str2int_list, help="total fold number")
parser.add_argument("--n_repeats", type=str2int_list, help="number of repeat")
parser.add_argument("--useDropColFeaImport", type=str2bool_list, help='use drop column feature importance or not')

args = parser.parse_args()









