import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim import Adam
import torch.nn.functional as F

from sentanalyzer.data.datasets import SentimentEmbeddingDataset
from sentanalyzer.data.dataloaders import get_dataloader
from sentanalyzer import locations,version_manager


class SentimentEmbedding(nn.Module):
    def __init__(self,
                 num_words,
                 vocab_size,
                 output_size,
                 embedding_dim,
                ):
        super(SentimentEmbedding,self).__init__()
        self.num_words = num_words
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        
        
        self.embedding = nn.Embedding(vocab_size,embedding_dim=embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim*num_words,64),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # nn.Linear(512,64),
            # nn.Dropout(0.6),
            # nn.ReLU(),
            
            nn.Linear(64,16),
            # nn.Dropout(0.6),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16,output_size),
            nn.Sigmoid()
            
        )
        
    def forward(self,x):
        embeds = self.embedding(x)
       
        embeds = embeds.reshape(x.shape[0],-1)
        # print(embeds.shape)
        out = self.fc(embeds).float()
        # print(out.shape)
        return out
    
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
        if  test_only:
             test_dataset = SentimentEmbeddingDataset(csv_path=dataset_locations_dict["TEST"],
                                                transform=None,
                                                freq_threshold=5,
                                                vocab_file=dataset_locations_dict["VOCAB"],
                                                create_vocab=False)
             return get_dataloader(test_dataset,
                                    test_dataset.vocab,
                                    batch_size=1,shuffle=False,num_workers=2)
        
        dataset = SentimentEmbeddingDataset(csv_path=locations.get_dataset_path(dataset_type="TRAIN_TEST",version=version_manager.VERSION_2),
                                                transform=None,
                                                freq_threshold=5,
                                                vocab_file=dataset_locations_dict["VOCAB"],
                                                create_vocab=True)
        
        
        
        train_val_dataset = SentimentEmbeddingDataset(csv_path=dataset_locations_dict["TRAIN"],
                                                transform=None,
                                                freq_threshold=5,
                                                vocab_file=dataset_locations_dict["VOCAB"],
                                                create_vocab=False)
            
        test_dataset = SentimentEmbeddingDataset(csv_path=dataset_locations_dict["TEST"],
                                                transform=None,
                                                freq_threshold=3,
                                                vocab_file=dataset_locations_dict["VOCAB"],
                                                create_vocab=False)
            
        train_ds_len = int(0.9*len(train_val_dataset))
      
        val_ds_len = len(train_val_dataset)-train_ds_len
        
        train_dataset,val_dataset = random_split(train_val_dataset,
                                                 lengths=[train_ds_len,val_ds_len],
                                                 generator=torch.Generator().manual_seed(256))
    
        train_dataloader = get_dataloader(dataset,
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
        description =   f"num_words : {self.num_words}\n"\
                        + f"vocab_size:{self.vocab_size}\n"\
                        + f"output_size:{self.output_size}\n"\
                        + f"embedding_dim:{self.embedding_dim}\n"\
                        + f"optimizer : \n"\
                        + f"{str(optimizer)}"\
                        + f"loss function {str(criterion)}"\
                        + f'\n{str(self)}'
        return description



def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim