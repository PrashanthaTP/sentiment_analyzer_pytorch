"""Currently giving around 80% accuracy on 5000test data"""
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
class SentimentAnalyzerCNN(nn.Module):
    """A CNN based model 

    Args:
    + weights_matrix,
    + output_size,
    + kernels,
    + num_filters
    """
    def __init__(self,
                 weights_matrix,
                 output_size,
                 kernels,
                 num_filters,
                 
                ):
        super(SentimentAnalyzerCNN,self).__init__()
        self.output_size = output_size
        self.num_filters = num_filters
        self.kernels  = kernels
        
    
        self.embedding,vocab_size , embedding_dim = create_emb_layer(weights_matrix, non_trainable=True)
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1,num_filters,[window_size,embedding_dim],padding=(window_size-1,0))
            for window_size in kernels
        ])
  
        self.fc = nn.Sequential( nn.Linear(num_filters*len(kernels),64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(64,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,4),
                                nn.BatchNorm1d(4),
                                nn.ReLU(),
                                nn.Linear(4,output_size))
       
        
    def forward(self,x):
        """
        after embedding layer torch.Size([256, 1, 100, 728])
        after filter torch.Size([256, 4, 101, 1])
        after squeeze torch.Size([256, 4, 101])
        after filter torch.Size([256, 4, 102, 1])
        after squeeze torch.Size([256, 4, 102])
        after filter torch.Size([256, 4, 103, 1])
        after squeeze torch.Size([256, 4, 103])
        """
        embeds = self.embedding(x)
        
        x = torch.unsqueeze(embeds,1)
        # print('x',x.shape)
        xs = []
        for conv in self.convs:
            x2 = torch.tanh(conv(x))
            # print('after filter',x2.shape)
            x2 = torch.squeeze(x2,-1)
            # print('after squeeze',x2.shape)
            x2 = F.max_pool1d(x2,x2.size(2))
    
            xs.append(x2)
            
        x = torch.cat(xs,2)
        x = x.view(x.size(0),-1)
        logits = self.fc(x)
        return torch.sigmoid(logits)
        
    
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
             
        train_val_dataset = TweetSentimentDataset(csv_path=dataset_locations_dict["TRAIN_TEST"],
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
    
    def get_description(self):
        optimizer,criterion= self.configure_parameters()
        description = f"vocab_size:{self.vocab_size}\n"\
                        + f"output_size:{self.output_size}\n"\
                        + f"embedding_dim:{self.embedding_dim}\n"\
                        +f"kernels:{self.kernels}\n"\
                        + f"num_filters:{self.num_filters}\n"\
                        + f"optimizer : \n"\
                        + f"{str(optimizer)}\n"\
                        + f"loss function {str(criterion)}"\
                        + f'\n{str(self)}'\
                        + f'\nRandom seed seed for splitting dataset {seed}\n'
        return description

