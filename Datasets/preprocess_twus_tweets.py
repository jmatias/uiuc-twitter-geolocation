import re
import pickle

TWEETS_TEST_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.test"
TWEETS_DEV_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.dev"
TWEETS_TRAIN_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.train"

regex_pattern = "(-?\d+\.\d+)\s+?(-?\d+\.\d+)\t(.+)"
data = []
with open(TWEETS_DEV_DATA, 'rb') as f:
    data.extend([m.group(3) for line in f for m in [re.search(regex_pattern, line.decode("utf-8"))] if m])

with open('Datasets/user_states_train.pickle', 'rb') as handle:
    states_train = pickle.load(handle)

with open('Datasets/user_states_test.pickle', 'rb') as handle:
    states_test = pickle.load(handle)

with open('Datasets/user_states_test.pickle', 'rb') as handle:
    states_dev = pickle.load(handle)

