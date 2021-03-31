import os
import cnn_imdb as cnn
import lstm_imdb as lstm
import hybrid_imdb as hybrid
from sentanalyzer.models.utils import Trainer
from sentanalyzer.utils.visualizer import display_confusion_matrix
from sentanalyzer import locations,version_manager

dataset_location = {
    'TRAIN':r'sentanalyzer\\test\\movie_review_train.csv',
    'TEST':r'sentanalyzer\\test\\movie_review_test.csv',
    'TRAIN_TEST':r'sentanalyzer\\test\\movie_review_train_test.csv',
    'VOCAB':locations.get_vocab_path(version=version_manager.VERSION_4)
}

cnn_model = cnn.get_model()
lstm_model = lstm.get_model()
hybrid_model = hybrid.get_model()


version = 1
log = r'sentanalyzer\\test\\'
def test_and_plot_cm(model):
    
    trainer = Trainer()
    test_history = trainer.test(model,dataset_location,batch_size=256) 

    print(test_history.history['log'])
    log_path = os.path.join(log,'test_run_details_V_' + str(version) + '.txt')
    if os.path.exists(log_path):
        
        with open(log_path,'a+') as f:
            f.write("\n IMDB large moview review dataset : testing")
            f.write(str(model))
            f.write('\n')
            f.write(test_history.history['log'] + '\n')
    else:
        
        with open(log_path,'w') as f:
            f.write(str(model))
            f.write('\n')
            f.write(test_history.history['log'] + '\n')
        
    display_confusion_matrix(test_history.history['y_preds'].flatten().numpy(),test_history.history['y_actual'].flatten().numpy())
    
    

def main():
   test_and_plot_cm(cnn_model)
   test_and_plot_cm(lstm_model)
   test_and_plot_cm(hybrid_model) 
   
if __name__ =='__main__':
    main()
    