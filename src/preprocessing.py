import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)  

df = pd.read_csv("/Users/adityamatiwala/github/Twitter-Sentiment-Analysis/data/twitter_training.csv")

text_column = df.columns[-1]

df["processed_tweet"] = df[text_column].apply(preprocess_text)

processed_file_path = "/Users/adityamatiwala/github/Twitter-Sentiment-Analysis/data/processed_twitter_training.csv"
df.to_csv(processed_file_path, index=False)

