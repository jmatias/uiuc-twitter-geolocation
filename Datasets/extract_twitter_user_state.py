import re
import math
import pickle

import Datasets.reverse_geocode as rg

TWEETS_TEST_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.test"
TWEETS_DEV_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.dev"
TWEETS_TRAIN_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.train"

TWEETS_DATA_DIR = ""


def extract_twitter_user_states(filepath, pickle_filename):
    regex_pattern = "([^\t]+)\t([-]?\d+\.\d+)\t([-]?\d+\.\d+)"
    data = []
    with open(filepath, 'r') as f:
        data.extend(
            [[m.group(1), m.group(2), m.group(3)] for line in f for m in
             [re.search(regex_pattern, line.strip())] if m])

    states = {}
    batch_size = 90
    num_batches = math.ceil(len(data) / batch_size)

    geocoder = rg.ReverseGeocode()

    for i in range(0, len(data)):
        state = geocoder.reverse_geocode_state((data[i][1], data[i][2]))
        states[data[i][0]] = state

        # location_list = []
        # upper_bound = batch_size
        # if( i == num_batches-1 ):
        #     upper_bound = len(data) - i*batch_size
        #
        # for j in range(0, upper_bound):
        #     curr_index = i * upper_bound + j
        #     location_list.append((data[curr_index][1], data[curr_index][2]))
        #
        # state_list = rg.reverse_geocode_state(location_list)
        #
        # for j in range(0, upper_bound):
        #     curr_index = i * upper_bound + j
        #     states[data[curr_index][0]] = state_list[j]

    # location_list.append(("18.2", "-66.5"))
    with open(pickle_filename + '.pickle', 'wb') as handle:
        pickle.dump(states, handle)


extract_twitter_user_states(TWEETS_DEV_DATA,'user_states_dev')
extract_twitter_user_states(TWEETS_TEST_DATA,'user_states_test')
extract_twitter_user_states(TWEETS_TRAIN_DATA,'user_states_train')
print("Hello World!")
