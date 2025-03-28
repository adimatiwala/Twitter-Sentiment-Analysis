import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('twitter_training.csv')
    print(df.head())
    print("--------------------------------")
    print(df.info())
    print("--------------------------------")
    unique_topic_count = df['topic'].nunique()
    print("Number of unique topics:", unique_topic_count)
    print("--------------------------------")
    tweets_per_topic = df.groupby('topic')['original_tweet'].count()
    print(tweets_per_topic)





if __name__ == '__main__':
    main()
