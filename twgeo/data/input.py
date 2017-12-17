import math
import re
import time

import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


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

    print("Parsing data from {0}...".format(csv_filename))
    nltk.download('punkt', quiet=True)
    df = pd.read_csv(csv_filename)
    tweets = df.iloc[:, tweet_txt_column_idx + 1].values
    locations = df.iloc[:, location_column_idx + 1].values

    ps = PorterStemmer()

    total_lines = len(tweets)
    percent_pt = math.ceil(total_lines / 500)
    now = time.time()
    start = now

    for i in range(0, len(tweets)):
        if (i % percent_pt == 0):
            now = time.time()
            if i != 0:
                time_per_unit = (now - start) / i
            else:
                time_per_unit = 999
            eta = time_per_unit * (total_lines - i)
            if eta > 3600:
                eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta

            info = "\r{0:.2f}% complete. ({1:,}/{2:,}) ETA: {3}        ".format(i / percent_pt / 5, i, total_lines,
                                                                                eta_format)
            print(info, end='')

        tweet_txt = tweets[i]
        words = word_tokenize(tweet_txt)

        words = (ps.stem(re.sub('(.)\\1{2,}', '\\1\\1', w)).lower() for w in words)
        tweet_txt = ' '.join(words)
        tweets[i] = tweet_txt
    print("\r100% complete...")

    return tweets, locations
