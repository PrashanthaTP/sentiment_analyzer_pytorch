from sentanalyzer.models.word_embedding import SentimentEmbedding
from sentanalyzer.models.hybrid_v2 import SentimentAnalyzerHybrid
from sentanalyzer.models.utils.trainer import Trainer


import train_helper

#### General Params ######
add_lr_scheduler = True

output_size = 1

batch_size = 256
n_epochs = 2

EXP_NAME  = 'Hybrid_v2'
model_file_name = 'hybrid_v2.pt'

######## Embedding params #######
num_words = 100
vocab_size = 10416
embedding_dim = 728

###### CNN Params ###############
kernels = (2,3,4,5)
num_filters = 8 #8 ==> 16

###### LSTM Params ###############
hidden_dim= 256
num_layers = 2



def get_weights_matrix(trainer):
    embed_checkpoint = r'E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\sentanalyzer\sentanalyzer\experiments\embedding_exp_15d_12m_20y_18H_40M\embedding_full_trained_.pt'
    embedding_model = SentimentEmbedding(num_words = num_words,
                                         vocab_size=vocab_size,
                                        output_size=output_size,
                                        embedding_dim=embedding_dim        )
 
    trainer.load_checkpoint(embedding_model,embed_checkpoint)
    return  train_helper.get_embeddings(embedding_model)


def main():
    trainer = Trainer()
   
    weights_matrix = get_weights_matrix(trainer)
    model = SentimentAnalyzerHybrid(weights_matrix=weights_matrix,
                                output_size=output_size,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                kernels=kernels,
                                num_filters=num_filters)
    
    train_helper.train(model,trainer,
                       n_epochs,batch_size,
                       EXP_NAME,model_file_name,add_lr_scheduler)
   

def test():
    trainer = Trainer()
   
    weights_matrix = get_weights_matrix(trainer)
    model = SentimentAnalyzerHybrid(weights_matrix=weights_matrix,
                                output_size=output_size,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                kernels=kernels,
                                num_filters=num_filters)
    # model_path = r"E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\sentanalyzer\sentanalyzer\experiments\exp_15d_12m_20y_20H_46M\lstm_v2_full_trained_.pt"
    model_path = r"E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\sentanalyzer\sentanalyzer\experiments\exp_Hybrid_v2_31d_12m_20y_18H_23M\hybrid_v2.pt"
    
    trainer = Trainer()
    trainer.load_checkpoint(model,model_path)
    # trainer.test(model,dataset_location,batch_size=1024)
    train_helper.test(model,trainer,batch_size=batch_size) 
    
if __name__ == '__main__':
    main()
    # test()