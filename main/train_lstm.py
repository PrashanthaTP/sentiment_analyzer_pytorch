import os 
from datetime import datetime


from sentanalyzer.models.lstm import SentimentAnalyzer
from sentanalyzer.models.utils import Trainer
from sentanalyzer.utils import logger
from sentanalyzer import locations,version_manager

def get_dir_and_basename(path):
    return os.path.dirname(path),os.path.basename(path)


def create_path_if_not(path):
    os.makedirs(path,exist_ok=True)




CURR_EXP_NO = "exp_" + datetime.now().strftime("%dd_%mm_%yy_%HH_%MM")
# LOG_DIR = os.path.join(locations.get_log_dir(),CURR_EXP_NO)
METRIC_FILE_PATH = locations.get_metrics_file_path(add_to_name=CURR_EXP_NO)
MODEL_FILE_PATH = locations.get_model_path(add_to_name=CURR_EXP_NO)

MODEL_DIR,MODEL_FILE_NAME = get_dir_and_basename(MODEL_FILE_PATH)
METRIC_DIR,METRIC_FILE_NAME = get_dir_and_basename(METRIC_FILE_PATH)


updated_model_file = os.path.join(MODEL_DIR,CURR_EXP_NO,MODEL_FILE_NAME)
updated_metric_file = os.path.join(METRIC_DIR,CURR_EXP_NO,METRIC_FILE_NAME)


###############[ These needs to be set before running ]####################

LOG_DIR = os.path.join(MODEL_DIR,CURR_EXP_NO)
LOG_FILE = f'info_{CURR_EXP_NO}.txt'
LOG_PATH = os.path.join(LOG_DIR,LOG_FILE)

dataset_location = {
    'TRAIN':locations.get_dataset_path("TRAIN",version=version_manager.VERSION_2),
    'TEST':locations.get_dataset_path("TEST",version=version_manager.VERSION_2),
    'VOCAB':locations.get_vocab_path(version=version_manager.VERSION_3)
}
n_epochs = 15
vocab_size=12283
output_size=2
embedding_dim=728
hidden_dim=128
num_layers = 1


# n_epochs = 5
# vocab_size=12791
# output_size=2
# embedding_dim=512
# hidden_dim=64
# num_layers = 1

batch_size = 128


###############################################################################


#TODO reduce embedding dim
def main():

    trainer = Trainer()
    model = SentimentAnalyzer(vocab_size=vocab_size,
                              output_size=output_size,
                              embedding_dim=embedding_dim,
                              hidden_dim=hidden_dim,
                              num_layers=num_layers)
    
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
    # model_path = r"E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\sentanalyzer\sentanalyzer\experiments\exp_14d_12m_20y_12H_47M\LSTM_V1__exp_14d_12m_20y_12H_47M.pt"
    model_path = r"E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\sentanalyzer\sentanalyzer\experiments\exp_14d_12m_20y_14H_43M\LSTM_V1__exp_14d_12m_20y_14H_43M.pt"
    model = SentimentAnalyzer(vocab_size=vocab_size,
                              output_size=output_size,
                              embedding_dim=embedding_dim,
                              hidden_dim=hidden_dim,
                              num_layers=num_layers)
    trainer = Trainer()
    trainer.load_checkpoint(model,model_path)
    trainer.test(model,dataset_location)
    
if __name__=='__main__':
    main()
    # test()
    

    

# class TrainMeta:
#     def __init__(self,hparams,dims):
#         """saves hparams and dimension information 
#         Arguments
#         ---------------
#         + hparams(dict):
#             contains optimizer name,loss function,learning rate,epochs etc
#         + dims
#             input dim,output dim etc
            

#         Args:
#             hparams ([type]): [description]
#             dims ([type]): [description]
#         """
#         self.__hparams = hparams
#         self.__dims = dims 
    
#     def get_hparams(self):
#         return self.__hparams
#     def get_dims(self):
#         return self.__dims
    
