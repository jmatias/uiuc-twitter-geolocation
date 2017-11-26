import re
import pickle

import Datasets.reverse_geocode as rg

TWEETS_TEST_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.test"
TWEETS_DEV_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.dev"
TWEETS_TRAIN_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.train"


def extract_twitter_user_states(filepath, pickle_filename):
    regex_pattern = "([^\t]+)\t([-]?\d+\.\d+)\t([-]?\d+\.\d+)"
    data = []
    with open(filepath, 'r') as f:
        data.extend(
            [[m.group(1), m.group(2), m.group(3)] for line in f for m in
             [re.search(regex_pattern, line.strip())] if m])

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
