"""
-----------------------------

classes
---------------
 
+ RegexHelper
    
    - containes various regex expressions for cleaning text
    
Functions
---------------

1. convert_to_lower
2. remove_stop_words
3. tokenize
4. remove_url_pattern
5. replace_neg_shortforms
6. replace_emoticons
7. replace_short_forms
8. replace_chat_words
9. remove_punctuations
10. correct_spelling
11. space_remover
12. lemmatize
    
       


"""
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk import WordNetLemmatizer
import string
import sentanalyzer_api.utils.shortform_dictionaries as shortforms
from bs4 import BeautifulSoup
#   from numpy import nan

tweet_tokenizer = TweetTokenizer(preserve_case=False,reduce_len=True,strip_handles=True)
stop_words = set(stopwords.words('english'))
new_stop_words = ['httpskeptical','go','undecided','annoy']
word_net_lemmatize = WordNetLemmatizer()

class RegexHelper:
    """
    -----------------------------
    Provides regex expression for various patterns
    ---------------
     
    `Patterns` 
    
    - Repeated letters in a single word (more than 2)
    - url pattern (http,https,www)
    - @ pattern 
    - non letters
    
    `Dictionaries`

    - negation shortforms
    - chat words
    - shortforms
    - emoticons
       
    -----------------------------
    """
  
    url_pattern_https = r'(https?://[^ ]+)|(https?://[A-Za-z0-9\./]+)'
    url_pattern_www = r'www.[^ ]+'
    url_pattern_1 = r'\w+.\w+'
    url_pattern = r'|'.join([url_pattern_https,url_pattern_www])
    
    at_pattern = r'@[A-Za-z0-9_]+'
    not_letters_pattern = r'[^a-zA-Z]'
    
    
    url_at_pattern = r'|'.join((url_pattern, at_pattern))
    
    negations_dic = shortforms.negations_dic
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
 
    
    shortforms_dic = shortforms.CONTRACTION_MAP
    shortforms_pattern =  re.compile(r'\b(' + '|'.join([k.strip() for k in shortforms_dic.keys()]) + r')\b')

    chat_words_dic = shortforms.SLANG_WORDS
    chat_words_pattern = re.compile(r'\b(' + '|'.join([k.strip().lower() for k in chat_words_dic.keys()]) + r')\b')
    
    emoticons_dic = shortforms.EMOTICONS
    emoticons_pattern = re.compile('|'.join([u'('+k+')' for k in emoticons_dic.keys()]) )
    # print(emoticons_dic.keys())
 

def convert_to_lower(text):
    return text.lower()
    
   
def remove_stop_words(text):
    return " ".join([t for t in text.split() if t not in stop_words])
  
  
def tokenize(text):
    """
    -----------------------------
    Tokenizes the  `text` with ``nltk.tokenize.TweetTokenizer()``
    
    Parameters
    -------------------------
    `text` : String to tokenized
        
    operations performed
    -------------------------
    - converts to lower case
    - strips user handles ex : @user
    - reduce length of string with repeated letters (above 3)
    - seperates emoticons
    
    Returns 
    ------------------------
    tokenized `text`  : str
    
    -----------------------------
    """
    return ' '.join(tweet_tokenizer.tokenize(text))


def remove_url_pattern(text):
    
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u'ufffd', '?')
    except:
        bom_removed = souped
    url_pattern_removed = re.sub( RegexHelper.url_pattern," ", bom_removed)
    return url_pattern_removed


def replace_neg_shortforms(text):
    neg_handled = RegexHelper.neg_pattern.sub( lambda x: RegexHelper.negations_dic[x.group()], text)
    return neg_handled
    
    
def replace_emoticons(text):
        # emoticons_replaced = RegexHelper.emoticons_pattern.sub(lambda x : print(type(x.group())),text)
        # return emoticons_replaced
        for emoticon in RegexHelper.emoticons_dic:
            text = re.sub(u'(' + emoticon + ')', "_".join(RegexHelper.emoticons_dic[emoticon].replace(",","").strip().lower().split()),text)
        return text 
    
    
def replace_short_forms(text):
        shorform_replaced = RegexHelper.shortforms_pattern.sub( lambda x: RegexHelper.shortforms_dic[x.group()].strip().lower(), text)
        return shorform_replaced
    
    
def replace_chat_words(text):
        chat_words_replaced = RegexHelper.chat_words_pattern.sub( lambda x: RegexHelper.chat_words_dic[x.group().upper().strip()].strip().lower(), text)
        return chat_words_replaced
    
    
def remove_punctuations(text):
        return text.translate(text.maketrans(string.punctuation," "*32,""))
    
def remove_non_word(text):
        return re.sub(RegexHelper.not_letters_pattern," ",text)
    
   
def space_remover(text):
        wordList = word_tokenize(text)
        return ' '.join([word.strip() for word in wordList if word.strip()])
    
     
def lemmatize(text):
        
        tokens = [word_net_lemmatize.lemmatize(word,pos='v') for word in text.split() if word]
        tokens = [word_net_lemmatize.lemmatize(word,pos='n') for word in tokens]
        return " ".join(tokens) 
    
    
def tweet_cleaner(tweet:str):
    """Applies series of preprocessing steps

    Args:
        text (str): string to be preprocessed

    Returns:
        str: text after performing following functions  on the input sequentially
        
        + replace_short_forms
        + replace_chat_words
        + remove_url_pattern
        + replace_emoticons
        + remove_non_word
        + space_remover
    """
    functions = [replace_short_forms,
                replace_chat_words,
                remove_url_pattern,
                replace_emoticons,
                remove_non_word,
                space_remover]
                 
    for preprocess_fn in functions:
        tweet = preprocess_fn(tweet)
    
    return tweet


# class TextPreprocessPipeline:
#     def __init__(self,fns:list):
#         self.__pipeline = fns
        
#     def add_fn(self,fn):
#         if not callable(fn):
#             raise TypeError("Argument to add_fn should be a function!!")
#         self.__pipeline.append(fn)
#         return self
    
#     def apply_on(self,text):
#         for fn in self.__pipeline:
#             text = fn(text)
#         return text
            
