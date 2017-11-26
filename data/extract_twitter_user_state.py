import re
import pickle

import data.reverse_geocode as rg

TWEETS_TEST_DATA = "na/user_info.test"
TWEETS_DEV_DATA = "na/user_info.dev"
TWEETS_TRAIN_DATA = "na/user_info.train"


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

    with open(pickle_filename + '.pickle', 'wb') as handle:
        pickle.dump(states, handle)


extract_twitter_user_states(TWEETS_DEV_DATA, 'user_states_dev')
extract_twitter_user_states(TWEETS_TEST_DATA, 'user_states_test')
extract_twitter_user_states(TWEETS_TRAIN_DATA, 'user_states_train')
