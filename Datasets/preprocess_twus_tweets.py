import re
import pickle

TWEETS_TEST_DATA = "na/user_info.test"
TWEETS_DEV_DATA = "na/user_info.dev"
TWEETS_TRAIN_DATA = "na/user_info.train"


def extract_twitter_user_tweets(filepath, pickle_filename):
    regex_pattern = "(.+?)\t(-?\d+\.\d+)\t(-?\d+\.\d+)\t(.+)"
    data = []
    with open(filepath, 'rb') as f:
        data.extend(
            [(m.group(1), m.group(4)) for line in f for m in [re.search(regex_pattern, line.decode("utf-8"))] if m])

    tweets = {}
    for row in data:
        user_tweets = [tweet.strip() for tweet in row[1].split("|||")]
        tweets[row[0]] = user_tweets

    with open(pickle_filename + '.pickle', 'wb') as handle:
        pickle.dump(tweets, handle)

    return tweets


extract_twitter_user_tweets(TWEETS_DEV_DATA, "user_tweets_dev")
extract_twitter_user_tweets(TWEETS_TEST_DATA, "user_tweets_test")
extract_twitter_user_tweets(TWEETS_TRAIN_DATA, "user_tweets_train")
print("Hello World!")
