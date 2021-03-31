"""Loads the raw dataset,cleans it and saves the proocessed dataset"""
import os
from tqdm import trange
import pandas as pd

from sentanalyzer import locations, version_manager
from sentanalyzer.data import TEST, TRAIN, TRAIN_TEST
from sentanalyzer.data.preprocessing import (TextPreprocessPipeline,
                                             remove_non_word,
                                             remove_url_pattern,
                                             replace_chat_words,
                                             replace_emoticons,
                                             replace_short_forms,
                                             space_remover, tokenize)
from sentanalyzer.data.utils import load_dataset
from sentanalyzer.utils import logger



TRAIN_TEST_DATASET = locations.get_dataset_path(dataset_type=TRAIN_TEST,version=version_manager.VERSION_2)
TRAIN_DATASET = locations.get_dataset_path(dataset_type=TRAIN,version=version_manager.VERSION_2)
TEST_DATASET = locations.get_dataset_path(dataset_type=TEST,version=version_manager.VERSION_2)


TEST_PERCENT = 0.3
TOTAL_SAMPLES = 1_00_000#2_00_000
TRAIN_SIZE = int((1/2)*TOTAL_SAMPLES*(1-TEST_PERCENT)) #per each target
TEST_SIZE = TOTAL_SAMPLES - TRAIN_SIZE


def process_dataset(df:pd.DataFrame,pipeline:TextPreprocessPipeline):
    """Preprocesses using TextPreprocessPipeline
    
    Args:
        df (pd.DataFrame): dataframe to be preprocessed

    Returns:
        dict: contains 
        +   cleaned_df
        +   train_df
        +   test_df
    """
  

    pipeline = TextPreprocessPipeline(pipeline)
    cleaned_df = pd.DataFrame()
    for i in trange(len(df),desc="Processing tweets ",ascii=True):
        tweet = df.loc[i,'text']
        processed_tweet = pipeline.apply_on(tweet)
        if processed_tweet.strip():
            cleaned_df = cleaned_df.append([{'text':processed_tweet.strip(),'sentiment':df['sentiment'][i]}],ignore_index=True)

    # cleaned_df['sentiment'] = df['sentiment'].values

    cleaned_df.dropna(inplace=True)
    cleaned_df.reset_index(drop=True,inplace=True)
    # print(cleaned_df.isnull().any())
    # return
    
    train_df = cleaned_df[cleaned_df['sentiment'] == 4].iloc[:TRAIN_SIZE, :]
    train_df = train_df.append(cleaned_df[cleaned_df['sentiment'] == 0].iloc[:TRAIN_SIZE, :],ignore_index=True)
    test_df = cleaned_df[cleaned_df['sentiment'] == 4].iloc[TRAIN_SIZE:, :]
    test_df = test_df.append(cleaned_df[cleaned_df['sentiment'] == 0].iloc[TRAIN_SIZE:, :],ignore_index=True)
    
    return {
        "cleaned_df" :cleaned_df,
        "train_df":train_df,
        "test_df":test_df
    }
    
    
    
def setup_dataset_v1():
    """Creates the text preprocessed dataset"""
    
    print("Process started....")
    df = load_dataset(location=locations.get_raw_dataset_path(),
                      req_cols=['sentiment','text'],
                      total_len=TOTAL_SAMPLES,
                      targets=[4,0],
                      encoding='latin-1')
    

    print("Dataset loaded...Starting preprocesing...")
    tweet_cleaner_fns = [tokenize,
                        replace_short_forms,
                        replace_chat_words,
                        remove_url_pattern,
                        replace_emoticons,
                        remove_non_word,
                        space_remover]
    
    processed_dataset  = process_dataset(df,pipeline=tweet_cleaner_fns)
    cleaned_df,train_df,test_df =  (processed_dataset['cleaned_df'],
                                    processed_dataset['train_df'],
                                    processed_dataset['test_df'])
    
    

    cleaned_df.to_csv(TRAIN_TEST_DATASET)
    train_df.to_csv(TRAIN_DATASET)
    test_df.to_csv(TEST_DATASET)

    print("Dataset successfully created")
    log_data = "\nDataset details :"
    log_data += f"\n+ Total Dataset size : {len(cleaned_df)} | file_name : {os.path.basename(TRAIN_TEST_DATASET)}"
    log_data += f"\n+ Train Dataset size : {len(train_df)}   | file_name : {os.path.basename(TRAIN_DATASET)}"
    log_data += f"\n+ Test Dataset size  : {len(test_df)}    | file_name : {os.path.basename(TEST_DATASET)}\n"
    print(log_data)
    logger.log(log_data,locations.get_dataset_logs_path())
    
    

if __name__ == '__main__':
    setup_dataset_v1()
    
    