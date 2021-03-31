import torch
from torch import nn
from torch.utils.data import random_split
from torch.optim import Adam
import torch.nn.functional as F

from sentanalyzer.data.datasets import TweetSentimentDataset
from sentanalyzer.data.dataloaders import get_dataloader
from sentanalyzer import locations



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
                 vocab_size,
                 output_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers
                ):
        super(SentimentAnalyzer,self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim  = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
 
        
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        
        # self.fc = nn.Linear(hidden_dim,output_size)
        self.fc = nn.Sequential( nn.Linear(hidden_dim,32),
                            
                                nn.ReLU(),
                                nn.Linear(32,4),
                                nn.Dropout(0.6),
                                nn.ReLU(),
                                nn.Linear(4,output_size))
        # self.fc1 = nn.Linear(hidden_dim,16)
        # self.fc2 = nn.Linear(16,output_size)
        # self.fc = nn.Linear(hidden_dim,output_size)
        # n_filters = 16
        # in_channels = 1
        # out_channels = n_filters
        # self.conv_net = nn.Sequential(
        #     nn.Conv1d(1,32,3),
        #     nn.MaxPool1d(2,2),
        #     nn.Conv1d()
        # )
        

    
    def forward(self,x):
        embeds = self.embedding(x)
        _,(hn,_) = self.lstm(embeds)
        return self.fc(hn[-1])
        
    
    def configure_parameters(self):
        """Configure optimizer and criterion 

        Returns:
        ---------------
        + tuple : optimizer,loss_function
        
        """
        optim = Adam(self.parameters(),lr=1e-4)
        criterion = nn.CrossEntropyLoss()
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
        if  test_only:
             test_dataset = TweetSentimentDataset(csv_path=dataset_locations_dict["TEST"],
                                                transform=None,
                                                freq_threshold=5,
                                                vocab_file=dataset_locations_dict["VOCAB"],
                                                create_vocab=False)
             return get_dataloader(test_dataset,
                                    test_dataset.vocab,
                                    batch_size=1,shuffle=False,num_workers=2)
             
        train_val_dataset = TweetSentimentDataset(csv_path=dataset_locations_dict["TRAIN"],
                                                transform=None,
                                                freq_threshold=3,
                                                vocab_file=dataset_locations_dict["VOCAB"],
                                                create_vocab=True)
            
        test_dataset = TweetSentimentDataset(csv_path=dataset_locations_dict["TEST"],
                                                transform=None,
                                                freq_threshold=3,
                                                vocab_file=dataset_locations_dict["VOCAB"],
                                                create_vocab=False)
            
        train_ds_len = int(0.9*len(train_val_dataset))
      
        val_ds_len = len(train_val_dataset)-train_ds_len
        
        train_dataset,val_dataset = random_split(train_val_dataset,
                                                 lengths=[train_ds_len,val_ds_len],
                                                 generator=torch.Generator().manual_seed(256))
    
        train_dataloader = get_dataloader(train_dataset,
                                          train_val_dataset.vocab,
                                          batch_size=batch_size,shuffle=True,num_workers=2)
        val_dataloader = get_dataloader(val_dataset,
                                        train_val_dataset.vocab,
                                        batch_size=batch_size,shuffle=False,num_workers=2)
        test_dataloader = get_dataloader(test_dataset,
                                         test_dataset.vocab,
                                         batch_size=1,shuffle=False,num_workers=2)
        
        print(f"Training Dataset size : {len(train_dataset)}\n")
        print(f"Validation Dataset size : {len(val_dataset)}\n")
        print(f"Test Dataset size : {len(test_dataset)}\n")
        
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
                        + f'\n{str(self)}'
        return description

