import re
import pickle
import os
import data.reverse_geocode as rg
import io
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

dirname = os.path.dirname(__file__)

TWITTER_TEST_DATA = os.path.join(dirname, 'na/user_info.test')
TWITTER_DEV_DATA = "na/user_info.dev"
TWITTER_TRAIN_DATA = "na/user_info.train"

STATES_TRAIN_DATA_FILE = "user_states_train2.pickle"
STATES_DEV_DATA_FILE = "user_states_dev2.pickle"
STATES_TEST_DATA_FILE = "user_states_test2.pickle"

TWEETS_TRAIN_DATA_FILE = "user_tweets_train3.pickle"
TWEETS_DEV_DATA_FILE = "user_tweets_dev2.pickle"
TWEETS_TEST_DATA_FILE = "user_tweets_test2.pickle"


def extract_twitter_data(filepath, pickle_filename):

    with open(os.path.join(dirname, "user_states_train.pickle"), 'rb') as handle:
        states_dev = pickle.load(handle)

    ps = PorterStemmer()


    regex_pattern = "([^\t]+)\t([-]?\d+\.\d+)\t([-]?\d+\.\d+)\t(.+)"
    data = []
    with io.open(filepath, 'r', encoding="utf-8") as file:
        data.extend(
            [[match.group(1), match.group(2), match.group(3), match.group(4)] for line in file for match in
             [re.search(regex_pattern, line.strip())] if match])

    parsed_data = []
    geocoder = rg.ReverseGeocode()

    for i in range(0, len(data)):
        username = data[i][0]
        stateStr = states_dev[username]
        if not stateStr:
            continue
        #stateStr = geocoder.reverse_geocode_state((data[i][1], data[i][2]))
        tweets = data[i][3]
        words = word_tokenize(tweets)
        words = [ps.stem(w) for w in words]
        tweets = ' '.join(words)


        state = geocoder.get_state_index(stateStr)
        region = geocoder.get_state_region(stateStr)
        regionStr = geocoder.get_state_region_name(stateStr)
        row = (username, tweets, state, region, stateStr, regionStr)
        parsed_data.append(row)


    with open(pickle_filename, 'wb') as handle:
        pickle.dump(parsed_data, handle)


def extract_twitter_user_states(filepath, pickle_filename):
    regex_pattern = "([^\t]+)\t([-]?\d+\.\d+)\t([-]?\d+\.\d+)"
    data = []
    with open(filepath, 'r') as file:
        data.extend(
            [[match.group(1), match.group(2), match.group(3)] for line in file for match in
             [re.search(regex_pattern, line.strip())] if match])

    states = {}
    geocoder = rg.ReverseGeocode()

    for i in range(0, len(data)):
        state = geocoder.reverse_geocode_state((data[i][1], data[i][2]))
        states[data[i][0]] = state

    with open(pickle_filename, 'wb') as handle:
        pickle.dump(states, handle)


def extract_twitter_user_tweets(filepath, pickle_filename):
    regex_pattern = "(.+?)\t(-?\d+\.\d+)\t(-?\d+\.\d+)\t(.+)"
    data = []
    with open(filepath, 'r') as f:
        data.extend(
            [(m.group(1), m.group(4)) for line in f for m in [re.search(regex_pattern, line)] if m])

    tweets = {}
    for row in data:
        user_tweets = [t for t in [re.sub("((@\w+))", "", tweet).strip() for tweet in row[1].split("|||")] if t]
        tweets[row[0]] = user_tweets

    with open(pickle_filename, 'wb') as handle:
        pickle.dump(tweets, handle)

    return tweets


if __name__ == '__main__':
    # extract_twitter_user_tweets(TWITTER_DEV_DATA, TWEETS_DEV_DATA_FILE)
    # extract_twitter_user_tweets(TWITTER_TEST_DATA, TWEETS_TEST_DATA_FILE)
    # extract_twitter_user_tweets(TWITTER_TRAIN_DATA, TWEETS_TRAIN_DATA_FILE)
    #
    # extract_twitter_user_states(TWITTER_DEV_DATA, STATES_DEV_DATA_FILE)
    # extract_twitter_user_states(TWITTER_TEST_DATA, STATES_TEST_DATA_FILE)
    #

    extract_twitter_data(TWITTER_TRAIN_DATA, TWEETS_TRAIN_DATA_FILE)
