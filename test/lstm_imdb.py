"""Currently giving around 80% accuracy on 5000test data"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split
import os
from word_embeddings import create_emb_layer

from sentanalyzer.data.datasets import ImdbSentimentDataset
from sentanalyzer.data.dataloaders import get_dataloader

CURDIR = os.path.abspath(os.curdir)
lstm_checkpoint = os.path.join(CURDIR,'sentanalyzer','models','trained','lstm_v2_statedict.pt')

output_size = 1
############# LSTM 
hidden_dim=256
num_layers = 2
# seed = 42
seed = 256
class SentimentAnalyzerLSTM(nn.Module):
    """A LSTM based model 

    Args:
    + vocab_size
    + output_size
    + embedding_dim
    + hidden_dim
    + num_layers
    """
    def __init__(self,
                 output_size,
                 hidden_dim,
                 num_layers
                ):
        super(SentimentAnalyzerLSTM,self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim  = hidden_dim
        
    
        self.embedding,vocab_size , embedding_dim = create_emb_layer()
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.fc = nn.Sequential( nn.Linear(hidden_dim,64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(64,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,4),
                                nn.BatchNorm1d(4),
                                nn.ReLU(),
                                nn.Linear(4,output_size),
                                nn.Sigmoid())
       
        
    def forward(self,x):
        embeds = self.embedding(x)
        _,(hn,_) = self.lstm(embeds)
        return self.fc(hn[-1])
    
    
    def get_dataloaders(self,
                        dataset_locations_dict,
                        batch_size=32,
                        test_only=False):
        """returns train,test and validation datasets

        Args:
        + dataset_locations_dict (dict): should contain keys(TRAIN and TEST) with location 

        Returns:
        + tuple : train,val and test dataloaders
        """
        # if  test_only:
        #      test_dataset = TweetSentimentDataset(csv_path=dataset_locations_dict["TEST"],
        #                                         transform=None,
        #                                         freq_threshold=5,
        #                                         vocab_file=dataset_locations_dict["VOCAB"],
        #                                         create_vocab=False)
        #      return get_dataloader(test_dataset,
        #                             test_dataset.vocab,
        #                             batch_size=1,shuffle=False,num_workers=0,
        #                              add_collate_fn=True)
             
        train_val_dataset = ImdbSentimentDataset(csv_path=dataset_locations_dict["TRAIN_TEST"],
                                                transform=None,
                                                freq_threshold=5,
                                                vocab_file=dataset_locations_dict["VOCAB"],
                                                create_vocab=False)
            
        # test_dataset = TweetSentimentDataset(csv_path=dataset_locations_dict["TEST"],
        #                                         transform=None,
        #                                         freq_threshold=5,
        #                                         vocab_file=dataset_locations_dict["VOCAB"],
        #                                         create_vocab=False)
            
        train_ds_len = int(0.9*len(train_val_dataset))
      
        val_ds_len = int(0.05*len(train_val_dataset))
        
        test_ds_len = len(train_val_dataset)-train_ds_len-val_ds_len
        
        train_dataset,val_dataset,test_dataset = random_split(train_val_dataset,
                                                 lengths=[train_ds_len,val_ds_len,test_ds_len],
                                                 generator=torch.Generator().manual_seed(seed))
    
        train_dataloader = get_dataloader(train_dataset,
                                          train_val_dataset.vocab,
                                          batch_size=batch_size,shuffle=True,num_workers=0,
                                          add_collate_fn=True)
        val_dataloader = get_dataloader(val_dataset,
                                        train_val_dataset.vocab,
                                        batch_size=batch_size,shuffle=False,num_workers=0,
                                         add_collate_fn=True)
        test_dataloader = get_dataloader(train_val_dataset,
                                         train_val_dataset.vocab,
                                         batch_size=batch_size,shuffle=False,num_workers=0,
                                          add_collate_fn=True)
        
        # test_dataset.df.to_csv('sentiment_analysis_test_dataset_4990.csv')
        print(f"Training Dataset size : {len(train_dataset)}\n")
        print(f"Validation Dataset size : {len(val_dataset)}\n")
        print(f"Test Dataset size : {len(test_dataset)}\n")
        
        if test_only:
            return test_dataloader
        return train_dataloader,val_dataloader,test_dataloader

def get_model():
    lstm_model = SentimentAnalyzerLSTM(output_size,hidden_dim,num_layers)
    lstm_model.load_state_dict(torch.load(lstm_checkpoint))
    
    return lstm_model