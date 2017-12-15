import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import math
import re

nltk.download('punkt', quiet=True)




def read_csv_data(csv_filename: str, location_column_idx: int, tweet_txt_column_idx: int):
    """
    Pre-process raw tweet data from a csv file. For each row, this function will:

        1. Tokenize the tweet text.
        2. Limit repeated characters to a maximum of 2. For example: 'Greeeeeetings' becomes 'Greetings'.
        3. Perform `Porter stemming  <https://en.wikipedia.org/wiki/Stemming>`_ on each token.
        4. Convert each token to lower case.

    :param csv_filename:
    :param location_column_idx: The zero-based index of the CSV column that contains the location information.
            The data itself must be a discrete value (a string or integer).
    :param tweet_txt_column_idx: The zero-based index of the CSV column that contains the tweet text.
    :return: Tuple (preprocessed_tweets, locations)
    """
    df = pd.read_csv(csv_filename)
    tweets = df.iloc[:, tweet_txt_column_idx+1].values
    locations = df.iloc[:, location_column_idx+1].values

    ps = PorterStemmer()

    total_lines = len(tweets)
    percent_pt = total_lines / 1000
    for i in range(0, len(tweets)):
        if (i % math.floor(percent_pt) == 0):
            print("\r{0:.2f}% complete...".format(i / percent_pt / 10), end='')
        tweet_txt = tweets[i]
        words = word_tokenize(tweet_txt)

        words = [ps.stem(re.sub('(.)\\1{2,}', '\\1\\1', w)).lower() for w in words]
        tweet_txt = ' '.join(words)
        tweets[i] = tweet_txt

    return tweets, locations
