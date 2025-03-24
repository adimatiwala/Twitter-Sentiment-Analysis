import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from gensim.models import Word2Vec
from logistic_regression import get_tweet_vector


df = pd.read_csv("../data/processed_twitter_training.csv")
w2v_model = Word2Vec.load("../models/word2vec.model")
X = np.array([get_tweet_vector(tweet, w2v_model) for tweet in df['processed_tweet']])


#X = data.drop('label', axis=1)
#y = data['label']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'saga']
}

lr_model = LogisticRegression(max_iter=1000)

# Grid Search 5 fold Cross Validation
grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5, scoring='f1_macro', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

y_pred = grid_search.best_estimator_.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(grid_search.best_estimator_, '../models/best_logistic_regression_model.pkl')
print("Best model saved to ../models/best_logistic_regression_model.pkl")
