"""
Built-in dataset of ~450K US based users.

"""
import io
import math
import pickle
import re
import sys
import time
from os import path, makedirs

import keras
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

sys.path.insert(0, path.abspath('../../'))

import twgeo.data.reverse_geocode as rg
from twgeo.data import constants

_dirname = path.dirname(path.abspath(__file__))

_TWITTER_TEST_DATA = path.join(constants.DATASETS_DIR, 'twus/user_info.test')
_TWITTER_DEV_DATA = path.join(constants.DATASETS_DIR, 'twus/user_info.dev')
_TWITTER_TRAIN_DATA = path.join(constants.DATASETS_DIR, 'twus/user_info.train')
_TWITTER_CSV_DEV_DATA = 'twus_dev.csv'

_TWITTER_PARSED_TEST_DATA = 'twus_test.pickle'
_TWITTER_PARSED_DEV_DATA = 'twus_dev.pickle'
_TWITTER_PARSED_TRAIN_DATA = 'twus_train.pickle'

_TWITTER_PARSED_TEST_DATA_DROPBOX = "https://dl.dropbox.com/s/kg09i1z32n12o98/twus_dev.pickle"
_TWITTER_PARSED_DEV_DATA_DROPBOX = "https://dl.dropbox.com/s/ze4ov5j30u9rf5m/twus_test.pickle"
_TWITTER_PARSED_TRAIN_DATA_DROPBOX = "https://dl.dropbox.com/s/0d4l6jmgguzonou/twus_train.pickle"
_TWITTER_CSV_DEV_DATA_DROPBOX = "https://dl.dropbox.com/s/8drqqugn5fw7zbx/twus_dev.csv"
_TWITTER_CSV_DEV_DATA_MD5 = "4e60b193ae5f4232c80d6e5f27b8c94e"


def load_state_data():
    """
    Training samples labeled with the corresponding US State.

    :return: Tuple(x_train, y_train, x_dev, y_dev, x_test, y_test)
    """
    train_df, dev_df, test_df = _load_data()

    x_train = train_df['tweets'].values
    y_train = train_df['state'].values

    x_dev = dev_df['tweets'].values
    y_dev = dev_df['state'].values

    x_test = test_df['tweets'].values
    y_test = test_df['state'].values

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def load_region_data():
    """
    Training samples labeled with the corresponding US Census Region.

    :return: Tuple(x_train, y_train, x_dev, y_dev, x_test, y_test)
    """
    train_df, dev_df, test_df = _load_data()

    x_train = train_df['tweets'].values
    y_train = train_df['region'].values

    x_dev = dev_df['tweets'].values
    y_dev = dev_df['region'].values

    x_test = test_df['tweets'].values
    y_test = test_df['region'].values

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def _load_data():
    temp_dev_data = keras.utils.get_file(_TWITTER_PARSED_DEV_DATA, _TWITTER_PARSED_DEV_DATA_DROPBOX)
    temp_test_data = keras.utils.get_file(_TWITTER_PARSED_TEST_DATA, _TWITTER_PARSED_TEST_DATA_DROPBOX)
    temp_train_data = keras.utils.get_file(_TWITTER_PARSED_TRAIN_DATA, _TWITTER_PARSED_TRAIN_DATA_DROPBOX)

    with open(temp_dev_data, 'rb') as handle:
        dev_data = pickle.load(handle)

    with open(temp_test_data, 'rb') as handle:
        test_data = pickle.load(handle)

    with open(temp_train_data, 'rb') as handle:
        train_data = pickle.load(handle)

    dev_df = pd.DataFrame(dev_data, columns=['username', 'tweets', 'state', 'region', 'state_name', 'region_name'])
    test_df = pd.DataFrame(test_data, columns=['username', 'tweets', 'state', 'region', 'state_name', 'region_name'])
    train_df = pd.DataFrame(train_data, columns=['username', 'tweets', 'state', 'region', 'state_name', 'region_name'])
    return train_df, dev_df, test_df


def _extract_twitter_data(filepath, pickle_filename):
    print("Parsing data from {0} ...".format(filepath))

    ps = PorterStemmer()

    regex_pattern = "([^\t]+)\t([-]?\d+\.\d+)\t([-]?\d+\.\d+)\t(.+)"
    data = []
    with io.open(filepath, 'r', encoding="utf-8") as file:
        data.extend(
            [[match.group(1), match.group(2), match.group(3), match.group(4)] for line in file for match in
             [re.search(regex_pattern, line.strip())] if match])

    parsed_data = []
    geocoder = rg.ReverseGeocode()
    total_lines = len(data)
    percent_pt = math.ceil(total_lines / 1000)
    now = time.time()
    start = now

    for i in range(0, len(data)):
        if (i % math.floor(percent_pt) == 0):
            now = time.time()
            if i != 0:
                time_per_unit = (now - start) / i
            else:
                time_per_unit = 99
            eta = time_per_unit * (total_lines - i)
            if eta > 3600:
                eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta

            info = "\r{0:.2f}% complete ({1:,}/{2:,}). ETA: {3}           ".format(i / percent_pt / 10, i, total_lines,
                                                                                   eta_format)
            print(info, end='')

        username = data[i][0]
        state_str = geocoder.reverse_geocode_state((data[i][1], data[i][2]))
        if not state_str:
            continue

        tweets = data[i][3]
        words = word_tokenize(tweets)
        words = (ps.stem(re.sub('(.)\\1{2,}', '\\1\\1', w)) for w in words)
        tweets = ' '.join(words)

        state = geocoder.get_state_index(state_str)
        region = geocoder.get_state_region(state_str)
        region_str = geocoder.get_state_region_name(state_str)
        row = (username, tweets, state, region, state_str, region_str)
        parsed_data.append(row)

    if not path.exists(path.dirname(path.abspath(pickle_filename))):
        makedirs(path.dirname(path.abspath(pickle_filename)))

    with open(pickle_filename, 'wb') as handle:
        pickle.dump(parsed_data, handle)
    print("\r100% complete...")


if __name__ == '__main__':
    _extract_twitter_data(_TWITTER_DEV_DATA, _TWITTER_PARSED_DEV_DATA)
    _extract_twitter_data(_TWITTER_TEST_DATA, _TWITTER_PARSED_TEST_DATA)
    _extract_twitter_data(_TWITTER_TRAIN_DATA, _TWITTER_PARSED_TRAIN_DATA)
