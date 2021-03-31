import torch
import torch.nn as nn
import torch.nn.functional as F

import os 


CURDIR = (os.path.abspath(os.curdir))

embed_checkpoint = os.path.join(CURDIR,'sentanalyzer','models','trained','embedding_full_trained_weight_matrix.pt')
########### Embedding Size #############
num_words = 100
vocab_size = 10416
embedding_dim = 728
output_size = 1

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
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16,output_size),
            nn.Sigmoid()
            
        )
        
    def forward(self,x):
        embeds = self.embedding(x)
       
        embeds = embeds.reshape(x.shape[0],-1)
        out = self.fc(embeds).float()
        return out

def create_emb_layer():
    weights_matrix = get_weights_matrix()
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})

    emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


   

def get_weights_matrix():
    
    # embedding_model = SentimentEmbedding(num_words = num_words,
    #                                      vocab_size=vocab_size,
    #                                     output_size=output_size,
    #                                     embedding_dim=embedding_dim        )
 
    # embedding_model.load_state_dict(torch.load(embed_checkpoint)['model_state_dict'])
    # weights_matrix = None
    
    # for name,param in embedding_model.named_parameters():
    #     if name=='embedding.weight':
    #         weights_matrix = param 
    # return weights_matrix
    
    return torch.load(embed_checkpoint)