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





METRIC_FILE_PATH = locations.get_metrics_file_path()
MODEL_FILE_PATH = locations.get_model_path()

MODEL_DIR,MODEL_FILE_NAME = get_dir_and_basename(MODEL_FILE_PATH)
METRIC_DIR,METRIC_FILE_NAME = get_dir_and_basename(METRIC_FILE_PATH)





###############[ These needs to be set before running ]####################



dataset_location = {
    'TRAIN':locations.get_dataset_path("TRAIN",version=version_manager.VERSION_2),
    'TRAIN_TEST':locations.get_dataset_path("TRAIN_TEST",version=version_manager.VERSION_2),
    'TEST':locations.get_dataset_path("TEST",version=version_manager.VERSION_2),
    'VOCAB':locations.get_vocab_path(version=version_manager.VERSION_4)
}
n_epochs = 6
vocab_size=10416
output_size=1
embedding_dim=728
hidden_dim=256
num_layers = 2




batch_size = 256


###############################################################################
def get_embeddings(embedding_model):
    weights_matrix = None
    
    for name,param in embedding_model.named_parameters():
        if name=='embedding.weight':
            weights_matrix = param 
    return weights_matrix

            

def train(model,
          trainer,
          n_epochs,batch_size,
          exp_name,
          model_file_name,
          add_lr_scheduler = False
         ):
        

    CURR_EXP_NO = "exp_"+exp_name+'_' + datetime.now().strftime("%dd_%mm_%yy_%HH_%MM")
    LOG_DIR = os.path.join(MODEL_DIR,CURR_EXP_NO)
    LOG_FILE = f'info_{CURR_EXP_NO}.txt'
    LOG_PATH = os.path.join(LOG_DIR,LOG_FILE)
    updated_model_file = os.path.join(MODEL_DIR,CURR_EXP_NO,model_file_name)
    updated_metric_file = os.path.join(METRIC_DIR,CURR_EXP_NO,'metrics_'+ model_file_name)
    
    
    create_path_if_not(os.path.join(MODEL_DIR,CURR_EXP_NO))
    create_path_if_not(os.path.join(METRIC_DIR,CURR_EXP_NO))
    
    logger.log(model.get_description() + '\n' + f"batch_size : {batch_size}  epochs : {n_epochs}" ,LOG_PATH)
    
 
    train_history = trainer.fit(    model = model,
                        n_epochs=n_epochs,
                        dataset_location=dataset_location,
                        model_file=updated_model_file,
                        metrics_file = updated_metric_file,
                        validate_every_x_epoch=1 ,
                        batch_size = batch_size,
                        add_lr_scheduler=add_lr_scheduler,
                        apply_softmax=False)
    
    
   



    test_history = trainer.test(model,dataset_location,batch_size=batch_size)
    logger.log(train_history.history['log'] + '\n' + test_history.history['log'],LOG_PATH)
    
def test(model,trainer,batch_size):
    from sentanalyzer.utils.visualizer import display_confusion_matrix
    test_history = trainer.test(model,dataset_location,batch_size=batch_size) 
    display_confusion_matrix(test_history.history['y_preds'].flatten().numpy(),test_history.history['y_actual'].flatten().numpy())
    
if __name__=='__main__':
    pass
