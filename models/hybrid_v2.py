import torch
from torch import nn
from torch.utils.data import random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from sentanalyzer.data.datasets import TweetSentimentDataset
from sentanalyzer.data.dataloaders import get_dataloader
from sentanalyzer import locations

from sentanalyzer.models.word_embedding import create_emb_layer


seed = 1

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
                 weights_matrix,
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
        self.embedding,vocab_size , embedding_dim = create_emb_layer(weights_matrix, 
                                                                non_trainable=False)
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
    
    def configure_parameters(self,add_lr_scheduler=False):
        """Configure optimizer and criterion 

        Returns:
        ---------------
        + tuple : optimizer,loss_function
        
        """
        optim = Adam(self.parameters(),lr=1e-3)
        if add_lr_scheduler:
             lr_scheduler = ReduceLROnPlateau(optim,mode='min',factor=0.1)
        criterion = nn.BCELoss()
        return (optim,criterion,lr_scheduler) if add_lr_scheduler else (optim,criterion)
    
    
    
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
     
             
        train_val_dataset = TweetSentimentDataset(csv_path=dataset_locations_dict["TRAIN_TEST"],
                                                transform=None,
                                                freq_threshold=5,
                                                vocab_file=dataset_locations_dict["VOCAB"],
                                                create_vocab=False)

            
        train_ds_len = int(0.9*len(train_val_dataset))
      
        val_ds_len = int(0.05*len(train_val_dataset))
        
        test_ds_len = len(train_val_dataset)-train_ds_len-val_ds_len
        
        train_dataset,val_dataset,test_dataset = random_split(train_val_dataset,
                                                 lengths=[train_ds_len,val_ds_len,test_ds_len],
                                                 generator=torch.Generator().manual_seed(seed))
    
        train_dataloader = get_dataloader(train_dataset,
                                          train_val_dataset.vocab,
                                          batch_size=batch_size,shuffle=True,num_workers=2,
                                          add_collate_fn=False)
        val_dataloader = get_dataloader(val_dataset,
                                        train_val_dataset.vocab,
                                        batch_size=batch_size,shuffle=False,num_workers=2,
                                         add_collate_fn=False)
        test_dataloader = get_dataloader(test_dataset,
                                         train_val_dataset.vocab,
                                         batch_size=batch_size,shuffle=False,num_workers=0,
                                          add_collate_fn=False)
        
        # test_dataset.df.to_csv('sentiment_analysis_test_dataset_4990.csv')
        print(f"Training Dataset size : {len(train_dataset)}\n")
        print(f"Validation Dataset size : {len(val_dataset)}\n")
        print(f"Test Dataset size : {len(test_dataset)}\n")
        
        if test_only:
            return test_dataloader
        return train_dataloader,val_dataloader,test_dataloader
    
    def get_description(self):
        optimizer,criterion,lr_scheduler = self.configure_parameters(add_lr_scheduler=True)
        description = f"vocab_size:{self.vocab_size}\n"\
                        + f"output_size:{self.output_size}\n"\
                        + f"embedding_dim:{self.embedding_dim}\n"\
                        +f"hidden_dim:{self.hidden_dim}\n"\
                        + f"num_layers:{self.num_layers}\n"\
                        + f"kernels:{self.kernels}\n"\
                        + f"filters:{self.num_filters}\n"\
                        + f"optimizer : \n"\
                        + f"{str(optimizer)}\n"\
                        + f"lr_scheduler : {lr_scheduler}\n"\
                        + f"loss function {str(criterion)}"\
                        + f'\n{str(self)}\n'\
                        + f'\nRandom seed seed for splitting dataset {seed}\n'
        return description

