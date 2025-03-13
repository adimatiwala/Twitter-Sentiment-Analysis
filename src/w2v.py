import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    
    def on_epoch_end(self, model):
        self.epoch += 1

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['processed_tweet'].fillna('').astype(str)
    word_lists = [text.split() for text in texts]
    return word_lists

def train_word2vec(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=10):
    """Train Word2Vec model"""
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size, 
        window=window,  
        min_count=min_count,  
        workers=workers,  
        epochs=epochs,  
        callbacks=[EpochLogger()]
    )
    return model

def main():
    file_path = "../data/processed_twitter_training.csv"
    sentences = load_and_preprocess_data(file_path)
    
    model = train_word2vec(sentences)
    
    model.save("../models/word2vec.model")
    
    print("\nExample word vectors:")
    try:
        print("Vector for 'game':", model.wv['game'][:5], "...") 
        
        print("\nMost similar to 'game':")
        similar_words = model.wv.most_similar('game', topn=5)
        for word, score in similar_words:
            print(f"{word}: {score:.4f}")
            
    except KeyError as e:
        print(f"Word not found in vocabulary: {e}")

if __name__ == "__main__":
    main()
