"""Module for Dataloaders for different datasets"""
import torch
from torch.utils.data import DataLoader

from sentanalyzer.data.datasets import TweetSentimentDataset
from sentanalyzer.data.collates import TweetPadCollate

def get_tweet_loader_v1(csv_file,vocab_file,transform=None,batch_size=32,shuffle=False,num_workers=0):
    """Return the dataloader

    Args:
        csv_file (str:path): dataset
        transform (fn): transform to be used per sample 
        batch_size (int, optional):  Defaults to 32.
        shuffle (bool, optional):  Defaults to False.
        num_workers (int, optional):  Defaults to 0.

    Returns:
        Dataloader
    """
    
    dataset = TweetSentimentDataset(csv_file,transform,vocab_file=vocab_file)
    pad_idx = dataset.vocab.str2idx["<PAD>"]
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=TweetPadCollate(pad_idx=pad_idx))
    return loader


def get_dataloader(dataset,vocab,batch_size,shuffle=False,num_workers=0,add_collate_fn=None):
    
    pad_idx = vocab.str2idx["<PAD>"]
    if add_collate_fn is True:
        collate_fn = TweetPadCollate(pad_idx)
        return DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=collate_fn,
                        num_workers=num_workers,
                        generator=torch.Generator().manual_seed(42))
    else:
        return DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        generator=torch.Generator().manual_seed(42))


#Testing get_tweet_loader
if __name__=='__main__':
    
    from sentanalyzer.locations import get_dataset_path,get_vocab_path
    
    path = get_dataset_path('TRAIN')
    print("Loading Dataset and building vocabulary....")
    loader = get_tweet_loader_v1(path,vocab_file=get_vocab_path(),shuffle=True)
    for i,(data,target) in enumerate(loader):
        if i==50:
            break