import torch
from torch import nn
from torch.utils.data import random_split
from torch.optim import Adam
import torch.nn.functional as F

from sentanalyzer.data.datasets import TweetSentimentDataset
from sentanalyzer.data.dataloaders import get_dataloader
from sentanalyzer import locations

from sentanalyzer.models.word_embedding import create_emb_layer


seed = 42

class SentimentAnalyzer(nn.Module):
    """A LSTM based model 

    Args:
    + vocab_size
    + output_size
    + embedding_dim
    + hidden_dim
    + num_layers
    """
    def __init__(self,
                 weights_matrix,
                 output_size,
                 hidden_dim,
                 num_layers
                ):
        super(SentimentAnalyzer,self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim  = hidden_dim
        
    
        self.embedding,vocab_size , embedding_dim = create_emb_layer(weights_matrix, 
        
                                                                non_trainable=True)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,#Think twice before setting bidirectional to False
                            batch_first=True)
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        window_size = 3
        self.lstm_fc = nn.Sequential( nn.Linear(hidden_dim,128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                # nn.Linear(128,32),
                                # nn.BatchNorm1d(32),
                                # nn.ReLU(),
                                # nn.Linear(32,8),
                                # nn.BatchNorm1d(8),
                                # nn.ReLU(),
                                # nn.Linear(4,output_size),
                                # nn.ReLU()
                                )
       
       
       #CNN 
        self.cnn_1 = nn.Conv2d(1,4,(window_size,embedding_dim),padding=(window_size-1,0))
        # self.pool = nn.MaxPool2d(2,1)
        # self.cnn_2 = nn.Conv2d(16,32,(window_size,embedding_dim),padding=(window_size-1,0))
        self.cnn_fc = nn.Sequential(
            nn.Linear(4*102,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Linear(32,8),
            # nn.BatchNorm1d(8),
            # nn.ReLU()
        )
        
        
        self.final_fc = nn.Sequential(
            # nn.Linear(32 + 32,8),
            nn.Linear(128+32,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,output_size),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        embeds = self.embedding(x)
        
        _,(hn,_) = self.lstm(embeds)
        first_model_out = self.lstm_fc(hn[-1])
        # return self.fc(hn[-1])
        # print(f"Input shape {x.shape}")
        # out = self.embedding(x)
        # print(f"embedding out {out.shape}")
        out = embeds.unsqueeze(1)
        out = F.relu(self.cnn_1(out))
        out = out.reshape(out.shape[0],-1)
     
    
        second_model_out =  self.cnn_fc(out)

        # print(f"first model out shape {first_model_out.shape}")
        # print(f"second model out shape {second_model_out.shape}")
        return self.final_fc(torch.cat([first_model_out,second_model_out],dim=1))
    
    def configure_parameters(self):
        """Configure optimizer and criterion 

        Returns:
        ---------------
        + tuple : optimizer,loss_function
        
        """
        optim = Adam(self.parameters(),lr=1e-3)
        criterion = nn.BCELoss()
        return (optim,criterion)
    
    
    
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
                                          batch_size=batch_size,shuffle=True,num_workers=0,
                                          add_collate_fn=False)
        val_dataloader = get_dataloader(val_dataset,
                                        train_val_dataset.vocab,
                                        batch_size=batch_size,shuffle=False,num_workers=0,
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
        optimizer,criterion= self.configure_parameters()
        description = f"vocab_size:{self.vocab_size}\n"\
                        + f"output_size:{self.output_size}\n"\
                        + f"embedding_dim:{self.embedding_dim}\n"\
                        +f"hidden_dim:{self.hidden_dim}\n"\
                        + f"num_layers:{self.num_layers}\n"\
                        + f"optimizer : \n"\
                        + f"{str(optimizer)}"\
                        + f"loss function {str(criterion)}"\
                        + f'\n{str(self)}\n'\
                        + f'\nRandom seed seed for splitting dataset {seed}\n'
        return description

