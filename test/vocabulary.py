"""Module for Vocabulary"""
from collections import Counter

from nltk.tokenize import word_tokenize


import os
import json


class Vocabulary:
    """Class for dealing with constructing vocabulary for text dataset
    """
    def __init__(self,freq_threshold,file):
        self.freq_threshold = freq_threshold
       
        self.str2idx = self.__build_vocabulary_from_file(file)
    
    def __len__(self):
        return len(self.str2idx)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.lower() for tok in word_tokenize(text) if tok.strip()]
    
    
      
        
    def numericalize(self,text):
        """Assigns a number to each word in text and returns the list of those numbers"""
        tokens = self.tokenizer_eng(text)
        return [self.str2idx.get(word,self.str2idx["<UNK>"]) for word in tokens]
    
    def __build_vocabulary_from_file(self,file):
        return self.__load_vocab(file)
        
    
    def __load_vocab(self,file):
        with open(file,'r') as f:
           return json.load(f)
       
    def __save_vocab(self,file):
        with open(file,'w') as f:
            json.dump(self.str2idx,f)
    