from torch import tensor
from torch.nn.utils.rnn import pad_sequence

class TweetPadCollate:
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self,batch):
        data = [sample[0] for sample in batch]
        targets = [sample[1] for sample in batch]
        
        padded_data = pad_sequence(data,batch_first=True,padding_value=self.pad_idx) 
      
        return padded_data,tensor(targets)