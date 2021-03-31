from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os 


def get_date_time():
    '''
    -----------------------------
    Returns current date and time 
    ----------------------------
    
    Parameters
    ---------------
    None
    
    Returns 
    ---------------
    [date,time] : list of strings
    
    -----------------------------
    '''
   
    currDateTime = datetime.now()
    date = currDateTime.date()
    dateStr = f"{date.day}/{date.month}/{date.year}"
    time = currDateTime.time()
    timeStr = f"{time.hour}:{time.minute}:{time.second}"
    
    return [dateStr,timeStr]
        
        
def log(log_data,log_file_path):
 
    date,time = get_date_time()
    if not os.path.exists(log_file_path):
        with open(log_file_path,'w') as f:
            heading = "LOGS\n==============================================================\n"
            f.write(f"{heading} \n\n")
        print(f"[LOG] Created {log_file_path}")
    print(f"[LOG] Writing to {log_file_path}")
    with open(log_file_path, 'a+') as file:
        file.writelines(f"================================================================\n")
        file.writelines(f"Date : {date} | Time : {time}\n")
        file.writelines(log_data)
        
        
        
def log_train_meta(meta:dict):
    pass