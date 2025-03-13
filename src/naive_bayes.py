import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9 
            self.priors[idx] = X_c.shape[0] / float(n_samples)
    
    def predict(self, X):
        y_pred = []
        
        for x in X:
            posteriors = []
            
            for idx, _ in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                

                diff = x - self.mean[idx]
                precision = 1.0 / (self.var[idx] + 1e-6)  
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[idx] + 1e-6) + diff**2 * precision)
                
                posterior = prior + log_likelihood
                posteriors.append(posterior)
            
            y_pred.append(self.classes[np.argmax(posteriors)])
            
        return np.array(y_pred)

def get_tweet_vector(tweet, model):
    """Convert tweet to vector by averaging word vectors"""
    if pd.isna(tweet):
        return np.zeros(model.vector_size)
    
    tweet = str(tweet)
    
    words = tweet.split()
    word_vectors = []
    for word in words:
        try:
            word_vectors.append(model.wv[word])
        except KeyError:
            continue
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    return np.zeros(model.vector_size)

def main():
    w2v_model = Word2Vec.load("../models/word2vec.model")
    
    df = pd.read_csv("../data/processed_twitter_training.csv")
    
    logging.info("Converting tweets to vectors...")
    X = np.array([get_tweet_vector(tweet, w2v_model) for tweet in df['processed_tweet']])
    
    label_map = {
        'Positive': 3,
        'Neutral': 2,
        'Negative': 1,
        'Irrelevant': 0
    }
    y = np.array([label_map[label] for label in df['label']])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logging.info("Training Naive Bayes classifier...")
    model = NaiveBayes()
    model.fit(X_train, y_train)
    
    logging.info("Making predictions...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Irrelevant', 'Negative', 'Neutral', 'Positive']))

if __name__ == "__main__":
    main() 