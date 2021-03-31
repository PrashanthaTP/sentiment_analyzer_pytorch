"""Module for location information"""
import os

from sentanalyzer.data import TRAIN,TEST,TRAIN_TEST
from sentanalyzer.utils.dev_utils import print_util
from sentanalyzer import version_manager

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def abs_path_wrapper(fn):
    def wrapper(*args,**kwargs):
        path = fn(*args,**kwargs)
        path = os.path.join(BASE_DIR,path)
        if os.path.exists(path):
            return path
        else:
            os.makedirs(os.path.dirname(path),exist_ok=True)
            return path
    return wrapper

#==============================================================================================#
original_dataset_path = r'E:/Users/VS_Code_Workspace/Python/NLP/Sentiment Analysis/twitter_dataset'
original_dataset_name = 'twitter_dataset.csv'


ORIGINAL_DATASET_LOCATION = os.path.join(original_dataset_path,original_dataset_name)

DATASET_DIR = r'data/src'
DATASET_NAME = r'twitter_dataset'

TRAIN_DATASET_NAME = 'sentiment_analysis_train.csv'
TEST_DATASET_NAME = 'sentiment_analysis_test.csv'
TRAIN_TEST_DATASET_NAME = 'sentiment_analysis_train_test.csv'

def get_raw_dataset_path():
    """returns the entire path for the dataset
    """
    
    return ORIGINAL_DATASET_LOCATION
   

@abs_path_wrapper
def get_dataset_path(dataset_type:str=TRAIN,version:str=version_manager.VERSION_1):
    """returns the entire path for the dataset

    Args:
        type(str) : one of TRAIN,TEST,TRAIN_TEST
        version (str, optional): version of the dataset to be returned. Defaults to None.

    Returns:
        str: if version is None Original Dataset is returned else the dataset of given version
    """
    file_name_part_1,file_name_part_2 = eval(dataset_type+'_DATASET_NAME').split('.')
    return os.path.join(DATASET_DIR,file_name_part_1 + version + '.' + file_name_part_2)
    
#===============================================================================================#
MODEL_NAME = 'LSTM.pt'
MODELS_DIR = 'experiments/'

@abs_path_wrapper
def get_model_path(version:str=version_manager.VERSION_1,add_to_name=''):
    """ returns model's path """
    if version is None:
        path = os.path.join(MODELS_DIR,MODEL_NAME)
        if os.path.exists(path):
            return path
        else:
            print_util.show_warning("get_model called without version!!! ")
            os.makedirs(os.path.dirname(path),exist_ok=True)
    first_part ,secondpart = MODEL_NAME.split('.')
    return os.path.join(MODELS_DIR,first_part + version + '_' + add_to_name +'.' + secondpart)

#===============================================================================================#

LOGS_DIR = 'logs'
LOGS_FILE = 'logs.txt'

@abs_path_wrapper
def get_log_dir():
    return LOGS_DIR

@abs_path_wrapper
def get_dataset_logs_path():
    return os.path.join(LOGS_DIR,LOGS_FILE)

@abs_path_wrapper
def get_logs_path(log_type,file_type="txt"):
    return os.path.join(LOGS_DIR,'logs_' + log_type+'.'+file_type)

#===============================================================================================#
VOCAB_DIR = 'data/vocab'
VOCAB_FILENAME = 'vocab.json'

@abs_path_wrapper
def get_vocab_path(version:str=version_manager.VERSION_1):
    """ returns vocabulary file path """
    first_part ,secondpart = VOCAB_FILENAME.split('.')
    return os.path.join(VOCAB_DIR,first_part + version + '.' + secondpart)

#===============================================================================================#
METRICS_NAME = 'metrics.pt'
METRICS_DIR = 'experiments/'

@abs_path_wrapper
def get_metrics_file_path(version:str=version_manager.VERSION_1,add_to_name=''):
    """ returns model's path """
    if version is None:
        path = os.path.join(METRICS_DIR,METRICS_NAME)
        if os.path.exists(path):
            return path
        else:
            print_util.show_warning("get_model called without version!!! ")
            os.makedirs(os.path.dirname(path),exist_ok=True)
    first_part ,secondpart = METRICS_NAME.split('.')
    return os.path.join(METRICS_DIR,first_part + version + '_' + add_to_name + '.' + secondpart)


#===============================================================================================#
#===============================================================================================#
#===============================================================================================#
#===============================================================================================#
#===============================================================================================#
