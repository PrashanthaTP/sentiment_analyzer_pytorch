import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split
import os
from word_embeddings import create_emb_layer

from sentanalyzer.data.datasets import ImdbSentimentDataset
from sentanalyzer.data.dataloaders import get_dataloader

CURDIR = os.path.abspath(os.curdir)

hybrid_checkpoint = os.path.join(CURDIR,'sentanalyzer','models','trained','hybrid_v2_statedict.pt')

    


output_size = 1
###### CNN Params ###############
kernels = (2,3,4,5)
num_filters = 8 #8 ==> 16

###### LSTM Params ###############
hidden_dim= 256
num_layers = 1
# seed = 42
seed = 256
class SentimentAnalyzerHybrid(nn.Module):
    """A LSTM/CNN based model 

    Args:
    + output_size,
    + hidden_dim,
    + num_layers,
    + kernels,
    + num_filters
    """
    def __init__(self,
                 output_size,
                 hidden_dim,
                 num_layers,
                 kernels,
                 num_filters,
                ):
        super(SentimentAnalyzerHybrid,self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim  = hidden_dim
        self.kernels = kernels
        self.num_filters = num_filters
    
        ############################# Embedding Layer #############################################
        self.embedding,vocab_size , embedding_dim = create_emb_layer()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        ############################# LSTM #############################################
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,#Think twice before setting bidirectional to False
                            batch_first=True)
        
        self.lstm_fc = nn.Sequential( nn.Linear(hidden_dim,128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(128,64),
                                nn.BatchNorm1d(64),
                                nn.ReLU()
                                )
        
        ############################# CNN #############################################
        self.convs = nn.ModuleList([
            nn.Conv2d(1,num_filters,[window_size,embedding_dim],padding=(window_size-1,0))
            for window_size in kernels
        ])
  
        self.convs_fc = nn.Sequential( nn.Linear(num_filters*len(kernels),64),
                                nn.BatchNorm1d(64),
                                nn.ReLU())
       
       ############################# FC #############################################
      
        
        self.final_fc = nn.Sequential(
            nn.Linear(64+64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,output_size),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        embeds = self.embedding(x)
        
        _,(hn,_) = self.lstm(embeds)
        lstm_out = self.lstm_fc(hn[-1])
       
        embeds = embeds.unsqueeze(1)
        xs = []
        for conv in self.convs:
            x2 = torch.tanh(conv(embeds))
            x2 = torch.squeeze(x2,-1)
            x2 = F.max_pool1d(x2,x2.size(2))
    
            xs.append(x2)
            
        cnn_out = torch.cat(xs,2)
        cnn_out = cnn_out.view(cnn_out.size(0),-1)
        cnn_out = self.convs_fc(cnn_out)
        return self.final_fc(torch.cat([lstm_out,cnn_out],dim=1))
    
    
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
        #      test_dataset = ImdbSentimentDataset(csv_path=dataset_locations_dict["TEST"],
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
        test_dataloader = get_dataloader(test_dataset,
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
    hybrid_model = SentimentAnalyzerHybrid(output_size,hidden_dim,num_layers,kernels,num_filters)
    
    hybrid_model.load_state_dict(torch.load(hybrid_checkpoint))
    
    return hybrid_model