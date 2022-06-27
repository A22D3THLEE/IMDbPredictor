import pandas as pd
from keras_preprocessing.text import Tokenizer

def load():
 train = pd.read_csv('train.csv', header=None, names=['Review'])
 reviews = train['Review']
 tokenizer = Tokenizer(num_words=10000)
 tokenizer.fit_on_texts(reviews)
 dictionary = tokenizer.word_index
 return dictionary
