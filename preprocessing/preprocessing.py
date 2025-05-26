import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import pandas as pd
from pre_processing import PreProcessing
import numpy as np
import spacy as sp

def remove_repetion_caracteres(string, max_repetition=2):
    if not string:
        return string
    
    result = string[0]
    count = 1
    
    for i in range(1, len(string)):
        if string[i] == string[i-1]:
            count += 1
            if count <= max_repetition:
                result += string[i]
        else:
            count = 1
            result += string[i]
    
    return result

def preprocess_text_pipeline(input_csv_path='./data/dataFrame.csv', 
                              output_csv_path='./data/dataFrame.csv',
                              stopwords_file='./txt_data/stopwords.txt',
                              text_column="comments"):
   
    stem = sp.load("pt_core_news_sm")
    pp = PreProcessing(language="pt")
    
    stopwords_dataset = [line.strip() for line in open(stopwords_file, 'r').readlines()]
    pp.append_stopwords_list(list(set(stopwords.words('portuguese')) - set(pp.stopwords)) + stopwords_dataset)

    def preprocessing(text):
        if pd.isna(text):
            return np.nan

        tokens = stem(text.lower())
        text = ' '.join([text for token in tokens for text in token.lemma_.strip().split()])
        text = pp.remove_stopwords(text)
        text = pp.lowercase_unidecode(text)
        text = pp.remove_stopwords(text)
        text = pp.remove_tweet_marking(text)
        text = remove_repetion_caracteres(text)
        text = pp.remove_urls(text)
        text = pp.remove_punctuation(text)
        text = pp.remove_numbers(text)
        text = pp.remove_n(text, n=2)
        
        return text

    df = pd.read_csv(input_csv_path)

    df['clean_text'] = df[text_column].apply(preprocessing)
    df.to_csv(output_csv_path, index=False)
