import numpy as npp
import torch
from torch.nn.functional import softmax
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm,trange

from sentanalyzer.models.utils import MetricManager,FilePathHandler
from datetime import datetime
import os
def get_model_checkpoint_name(model):
    timestamp =  datetime.now().strftime("%dd_%mm_%yy_%HH_%MM")
    return model.__class__.__name__ + '_' + timestamp + '.pt'
    

class Trainer():
    """Utility for training Neural networks built with nn.Module
    """
    __fns_required = [ 'forward',
                      'training_step',
                      'configure_parameters',
                      'get_dataloaders',
                      ]
     
    def __init__(self):
        # self.logger = SummaryWriter(log_dir=log_dir)
        pass
        
    def __is_function_implemented(self,model,f_name):
        f_op = getattr(model, f_name, None)
        return callable(f_op)
    
    def __check_implemented_properties(self,model):
        for fn in Trainer.__fns_required:
            if not self.__is_function_implemented(model,fn):
                raise NotImplementedError(f"[ERROR in {model.__class__.__name__} class] | {fn} not implemented.")
            
    def fit(self,
            model,
            n_epochs,
            dataset_location:dict,
            model_file,
            metrics_file,
            validate_every_x_epoch=None,
            batch_size=32,
            add_lr_scheduler = False,
            apply_softmax=False):
        """trains the given model

        Args:
        + `model` (nn.Module): Model to be trained
        + `n_epochs` (int): number of epochs
        + `dataset_location` (dict): location of TRAIN and TEST datasets
        + `model_file` (str): Name of the file where model state dict  will be saved
        + `metrics_file` (str): Name of the file where train metrics will be saved  
        + `validate_every_x_epoch` (int, optional): Decides after how many epochs the model has to be validated.
        + `batch_size`(int) : batch size for dataloaders. Defaults to 32.
        Defaults to 10% of total epochs.

        Returns:
        + MetricManager  : contains train metrics
            (For details see MetricManager class implementation)
        """
        # self.__check_implemented_properties(model)
        if add_lr_scheduler:
            optimizer,criterion,lr_scheduler = model.configure_parameters(add_lr_scheduler)
        else:
            optimizer,criterion = model.configure_parameters(add_lr_scheduler)
            
        train_dataloader,val_dataloader,_ = model.get_dataloaders(dataset_location,batch_size)
       
        if validate_every_x_epoch is None :
            validate_every_x_epoch = int(n_epochs*0.1) #Validate every 10% of total epochs
        
        validate_every_x_steps = None 
        
        metrics = MetricManager(metrics_file)
       
        best_val_loss = float('inf')
    
        global_steps = 0
        
        # tqdm_epoch_iterator = trange(n_epochs,desc="epochs",ascii=True,colour="green",position=0)
        for epoch in range(n_epochs):
            print()
            model.train()
            train_running_loss = 0
            train_running_acc = 0
          
            
            # tqdm_epoch_iterator.set_description(desc=f"epochs {epoch+1}/{n_epochs}")
            tqdm_train_iterator = tqdm(enumerate(train_dataloader),
                                       desc=f"[train]{epoch+1}/{n_epochs}",
                                       ascii=True,leave=True,
                                       total=len(train_dataloader),
                                       colour="green",position=0)
                        
            for batch_idx,(data,target) in tqdm_train_iterator:
    
                optimizer.zero_grad()
                y_pred = model(data)
                if not apply_softmax:
                    target = target.unsqueeze(1)
                loss = criterion(y_pred,target)
                loss.backward()
                optimizer.step()
                
                global_steps += 1
                train_running_loss += loss.item()
              
                if apply_softmax:
                    train_running_acc += self.get_accuracy_with_softmax(y_pred.detach(),target)
                else:
                    train_running_acc += self.get_accuracy_without_softmax(y_pred.detach(),target)
                    
                tqdm_train_iterator.set_postfix(avg_train_acc=f"{train_running_acc/(batch_idx+1):0.4f}",
                                                avg_train_loss=f"{(train_running_loss/(batch_idx+1)):0.4f}")
                
            #validation
            if  (epoch+1)%validate_every_x_epoch == 0:
                if validate_every_x_steps is None :
                    validate_every_x_steps = global_steps
                    
                model.eval()
                val_running_loss = 0
                val_running_acc = 0
                tqdm_val_iterator = tqdm(enumerate(val_dataloader),desc="[validation]",
                                         total=len(val_dataloader),leave=True,ascii=True,
                                         colour="yellow",position=0)
                with torch.no_grad():
                    for batch_idx,(data,target) in tqdm_val_iterator:
                        y_pred = model(data)
                        if not apply_softmax:
                            target = target.unsqueeze(1)
                        loss = criterion(y_pred,target)
                        val_running_loss += loss.item()
                        if apply_softmax:
                            val_running_acc += self.get_accuracy_with_softmax(y_pred,target)
                        else:
                            val_running_acc += self.get_accuracy_without_softmax(y_pred,target)
                            
                        tqdm_val_iterator.set_postfix(avg_val_acc=f"{val_running_acc/(batch_idx+1):0.4f}",
                                                      avg_val_loss=f"{val_running_loss/(batch_idx+1):0.4f}")         
             
                avg_train_loss = train_running_loss/(len(train_dataloader)*validate_every_x_epoch)
                avg_train_acc = train_running_acc/(len(train_dataloader)*validate_every_x_epoch)
                avg_val_loss = val_running_loss/(len(val_dataloader)*validate_every_x_epoch)
                avg_val_acc = val_running_acc/(len(val_dataloader)*validate_every_x_epoch)
                
                if add_lr_scheduler : lr_scheduler.step(avg_val_loss)
            
                metrics.update(avg_train_loss,avg_val_loss,avg_train_acc,avg_val_acc,epoch+1)
                # print(f"val_acc : {val_running_acc/(batch_idx+1):0.4f} val_loss : {val_running_loss/(batch_idx+1):0.4f}")
                # tqdm_train_iterator.set_postfix_str(metrics.get_metrics_string())
                
                if best_val_loss > avg_val_loss:    
                        best_val_loss = avg_val_loss
                        self.save_checkpoint(model,optimizer,
                                             val_acc=avg_val_acc,val_loss=avg_val_acc,
                                             model_file =model_file)
                        log = f"Train Dataset size {len(train_dataloader.dataset)}\n"
                        log+= f"Val Dataset Size {len(val_dataloader.dataset)}\n"
                        log += f"best_val_acc : {avg_val_acc:0.4f}\n"
                        log += f"best_val_loss : {avg_val_loss:0.4f}\n"
        
        
        dir_name,file_name    = os.path.dirname(model_file),os.path.basename(model_file) 
        first_part,second_part = file_name.split('.')
        self.save_checkpoint(model,optimizer,
                            val_acc=None,val_loss=None,
                             model_file =os.path.join(dir_name,first_part + '_full_trained_.' + second_part))
                      
        metrics.save_metrics()
        metrics.plot_metrics()
        history = History()
        history.history['name'] = 'Training History'
        history.history['metrics'] = metrics
        history.history['log'] = log
        return history
    
  
    def save_checkpoint(self,model, optimizer, val_loss,val_acc,model_file):

        state_dict = {'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc':val_acc,
                    'val_loss': val_loss}
        
        torch.save(state_dict, model_file)
        print(f'Model saved to ==> {model_file}')


    def load_checkpoint(self, model,model_file, optimizer=None):

    
        state_dict = torch.load(model_file)
        print(f'Model loaded from <== {model_file}')
        
        model.load_state_dict(state_dict['model_state_dict'])
        if optimizer is not None :
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        
        return state_dict['val_loss']


    def test(self,model,dataset_location_dict:dict,batch_size,apply_softmax=False):
        test_dataloader = model.get_dataloaders(dataset_locations_dict=dataset_location_dict,batch_size=batch_size,test_only=True)
        
        model.eval()
        test_running_acc = 0
        all_predictions = torch.tensor([])
        y_actual = torch.tensor([])
        tqdm_test_iterator = tqdm(enumerate(test_dataloader),desc="[TEST]",total=len(test_dataloader),ascii=True,colour="blue")
        with torch.no_grad():
            for batch_idx,(data,target) in tqdm_test_iterator:
                y_pred = model(data)
                all_predictions = torch.cat((all_predictions,torch.round(y_pred)),dim=0)
                y_actual = torch.cat((y_actual,target),dim=0)
                if apply_softmax:
                    # test_running_acc += (self.get_accuracy(softmax(y_pred,dim=1),target))
                    test_running_acc += self.get_accuracy_with_softmax(y_pred,target)
                else:
                    target = target.unsqueeze(1)
                    test_running_acc += self.get_accuracy_without_softmax(y_pred,target)
                    
                tqdm_test_iterator.set_postfix(avg_test_acc=f"{test_running_acc/(batch_idx+1):0.4f}")  
                # if (batch_idx+1)%10==0:
                #     print("===")
                #     print(data[0],y_pred[0],target[0])
                #     print("===")
        log = f"Test dataset size {len(test_dataloader.dataset)}\n"
        log += f"Test accuracy {(test_running_acc/len(test_dataloader))*100:4f} % "
        
        history = History()
        history.history['name'] = 'Testing History'
        history.history['y_preds'] = all_predictions
        history.history['y_actual'] = y_actual
        history.history['log'] = log
    
        
        return history 

    def dev_train(self,model,n_epochs:int,dataset_location:dict):
        """Trains the model on a singel batch to check if model is capable of overfitting
        
        @params
        
        + model : model to be trained
        
        + n_epochs(int) : Number of times the same batch has to be fed to the model.
        
        + dataset_location (dict): location of the TRAIN/TEST dataset
        """
        optimizer,criterion = model.configure_parameters()
        dataloader,_ = model.get_dataloaders(dataset_location)
        data,target = next(iter(dataloader))

        # tqdm_iterator = tqdm(enumerate(dataloader),desc="[DEV TRAIN]",total=len(dataloader))
        tqdm_iterator = trange(n_epochs,desc="[DEV TRAIN]")
        model.train()
        for _ in tqdm_iterator :
        
            optimizer.zero_grad()
            y_pred = model(data)
    
            loss = criterion(y_pred,target)
            loss.backward()
            optimizer.step()
            tqdm_iterator.set_postfix({"loss":f"{loss.item():.4f}"})
    
    
    def get_accuracy_with_softmax(self,y_pred,y_actual):
        """Calculates the accuracy (0 to 1)

        Args:
        + y_pred (tensor ): output from the model
        + y_actual (tensor): ground truth 

        Returns:
        + float: a value between 0 to 1
        """
        _, y_pred = torch.max(softmax(y_pred.detach(),dim=1) ,1)
        # print(y_pred,y_actual)
        # print(y_pred.shape,y_actual.shape,torch.sum(y_pred==y_actual),torch.sum(y_pred==y_actual).item())
        return (1/len(y_actual))*torch.sum(y_pred==y_actual)
    
    def get_accuracy_without_softmax(self,y_pred,y_actual):
        """Calculates the accuracy (0 to 1)

        Args:
        + y_pred (tensor ): output from the model
        + y_actual (tensor): ground truth 

        Returns:
        + float: a value between 0 to 1
        """
        
        # print(y_pred,y_actual)
        # print(y_pred.shape,y_actual.shape,torch.sum(y_pred==y_actual),torch.sum(y_pred==y_actual).item())
        # print(y_pred.shape,y_actual.shape)
        return (1/len(y_actual))*torch.sum(torch.round(y_pred)==y_actual)
    
    def get_num_correct(self,y_pred,y_actual):
        return torch.sum(torch.round(y_pred)==y_actual)
    
    
class History:
    def __init__(self):
        self.history = {}
        
        