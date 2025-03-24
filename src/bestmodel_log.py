import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from logistic_regression import get_tweet_vector
from gensim.models import Word2Vec

df = pd.read_csv("../data/processed_twitter_training.csv")
w2v_model = Word2Vec.load("../models/word2vec.model")
best_model = joblib.load("../models/best_logistic_regression_model.pkl")

X_test = np.array([get_tweet_vector(tweet, w2v_model) for tweet in df['processed_tweet']])

label_map = {
    'Positive': 3,
    'Neutral': 2,
    'Negative': 1,
    'Irrelevant': 0
}

y_test = np.array([label_map[label] for label in df['label']])
y_pred = best_model.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
display_labels = ['Irrelevant', 'Negative', 'Neutral', 'Positive']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Heatmap visualization
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_labels, yticklabels=display_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()