"""Module for data utilities"""
import pandas as pd 

    
def load_dataset(location:str,
                req_cols:list,
                total_len:int,
                targets:list,
                encoding='latin-1'):
    """Loads the csv dataset

    Args:
        location (str): Location where the csv dataset present
        req_cols (list): list of column names required
        total_len (int): length of the dataset to be returned
        targets (list): list of target classes
        encoding (str, optional): encoding format of the dataset. Defaults to 'latin-1'.

    Returns:
        pandas dataframe: dataset of length total_len
    """
    orig_df = pd.read_csv(location,encoding=encoding)
    COLS = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
    orig_df.columns = COLS

    orig_df = orig_df.loc[:,req_cols]
    orig_df.dropna(inplace=True)
    
    count_targets = len(targets)
    df = pd.DataFrame()
    for target in targets:
        df = df.append(orig_df[orig_df['sentiment'] ==  target].iloc[:total_len//count_targets, :], ignore_index=True)

    return df



