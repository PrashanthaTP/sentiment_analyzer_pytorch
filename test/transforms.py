from torch import tensor
from sentanalyzer_api.utils.vocabulary import Vocabulary
from sentanalyzer_api.utils.preprocess import tweet_cleaner
# vocab_file = r'sentanalyzer_api/utils/vocab_V4_.json'
import os 
from dotenv import load_dotenv
load_dotenv()

vocab_file = r'sentanalyzer_api/utils/vocab_V4_.json'
if os.environ.get('MODE')== 'DEV':
    vocab_file = r'utils/vocab_V4_.json'
vocab = Vocabulary(freq_threshold=5,file=vocab_file)


def text_to_vector(tweet,num_words=100):
    tweet = tweet_cleaner(tweet)
    numericalized_data = vocab.numericalize(tweet)
        
    numericalized_data.extend([vocab.str2idx["<PAD>"] for _ in range(len(numericalized_data),num_words)])
    if len(numericalized_data)>num_words:
        numericalized_data = numericalized_data[:num_words]
    
    return tensor(numericalized_data)