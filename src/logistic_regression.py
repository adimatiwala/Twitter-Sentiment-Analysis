import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.classes = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)
        
        for idx, current_class in enumerate(self.classes):
            y_binary = (y == current_class).astype(int)
            
            for _ in range(self.num_iterations):
                # Forward pass
                linear_model = np.dot(X, self.weights[idx]) + self.bias[idx]
                y_predicted = self.sigmoid(linear_model)
                
                # gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_binary))
                db = (1 / n_samples) * np.sum(y_predicted - y_binary)
                
                self.weights[idx] -= self.learning_rate * dw
                self.bias[idx] -= self.learning_rate * db
    
    def predict(self, X):
        probabilities = np.zeros((X.shape[0], len(self.classes)))
        for idx, _ in enumerate(self.classes):
            linear_model = np.dot(X, self.weights[idx]) + self.bias[idx]
            probabilities[:, idx] = self.sigmoid(linear_model)
        
        # Return class with highest probability
        return self.classes[np.argmax(probabilities, axis=1)]

def get_tweet_vector(tweet, model):
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
    
    logging.info("Training logistic regression...")
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
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