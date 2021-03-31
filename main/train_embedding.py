import os 
from datetime import datetime


from sentanalyzer.models.word_embedding import SentimentEmbedding
from sentanalyzer.models.utils import Trainer
from sentanalyzer.utils import logger
from sentanalyzer import locations,version_manager

def get_dir_and_basename(path):
    return os.path.dirname(path),os.path.basename(path)


def create_path_if_not(path):
    os.makedirs(path,exist_ok=True)




CURR_EXP_NO = "embedding_exp_" + datetime.now().strftime("%dd_%mm_%yy_%HH_%MM")
# LOG_DIR = os.path.join(locations.get_log_dir(),CURR_EXP_NO)
METRIC_FILE_PATH = locations.get_metrics_file_path(add_to_name=CURR_EXP_NO)
MODEL_FILE_PATH = locations.get_model_path(add_to_name=CURR_EXP_NO)

MODEL_DIR,MODEL_FILE_NAME = get_dir_and_basename(MODEL_FILE_PATH)
METRIC_DIR,METRIC_FILE_NAME = get_dir_and_basename(METRIC_FILE_PATH)


updated_model_file = os.path.join(MODEL_DIR,CURR_EXP_NO,'embedding.pt')
updated_metric_file = os.path.join(METRIC_DIR,CURR_EXP_NO,METRIC_FILE_NAME)


###############[ These needs to be set before running ]####################

LOG_DIR = os.path.join(MODEL_DIR,CURR_EXP_NO)
LOG_FILE = f'info_{CURR_EXP_NO}.txt'
LOG_PATH = os.path.join(LOG_DIR,LOG_FILE)

dataset_location = {
    'TRAIN':locations.get_dataset_path("TRAIN",version=version_manager.VERSION_2),
    'TEST':locations.get_dataset_path("TEST",version=version_manager.VERSION_2),
    'VOCAB':locations.get_vocab_path(version=version_manager.VERSION_4)
}
n_epochs = 5
vocab_size=10416
output_size=1
embedding_dim=728


# n_epochs = 5
# vocab_size=12791
# output_size=2
# embedding_dim=512
# hidden_dim=64
# num_layers = 1

batch_size = 512


###############################################################################


#TODO reduce embedding dim
def main():

    trainer = Trainer()
    model = SentimentEmbedding(num_words = 100,vocab_size=vocab_size,
                              output_size=output_size,
                              embedding_dim=embedding_dim,
                              )
    
    create_path_if_not(os.path.join(MODEL_DIR,CURR_EXP_NO))
    create_path_if_not(os.path.join(METRIC_DIR,CURR_EXP_NO))
    
    logger.log(model.get_description() ,LOG_PATH)
    
 
    _,train_log_str = trainer.fit(    model = model,
                        n_epochs=n_epochs,
                        dataset_location=dataset_location,
                        model_file=updated_model_file,
                        metrics_file = updated_metric_file,
                        validate_every_x_epoch=1 ,
                        batch_size = batch_size)
    
    test_log_str = trainer.test(model,
                                dataset_location)
    
    logger.log(train_log_str + '\n' + test_log_str,LOG_PATH)
    return

    



def test():
    trainer = Trainer()
    checkpoint = r'E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\sentanalyzer\sentanalyzer\experiments\embedding_exp_15d_12m_20y_18H_40M\embedding_full_trained_.pt'
    model = SentimentEmbedding(num_words = 100,vocab_size=vocab_size,
                              output_size=output_size,
                              embedding_dim=embedding_dim,
                              )
    trainer.load_checkpoint(model,checkpoint)
    for name,param in model.named_parameters():
        if name=='embedding.weight':
            print(param.shape)
            
            
if __name__=='__main__':
    main()
    # test()
    

    

