import pandas as pd
import re
import pickle
from os import path, makedirs
import data.reverse_geocode as rg
import data.constants as constants
import io
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

_dirname = path.dirname(path.abspath(__file__))

_TWITTER_TEST_DATA = path.join(constants.DATASETS_DIR, 'twus/user_info.test')
_TWITTER_DEV_DATA = path.join(constants.DATASETS_DIR, 'twus/user_info.dev')
_TWITTER_TRAIN_DATA = path.join(constants.DATASETS_DIR, 'twus/user_info.train')

_TWITTER_PARSED_TEST_DATA = path.join(constants.DATACACHE_DIR, 'twus_test.pickle')
_TWITTER_PARSED_DEV_DATA = path.join(constants.DATACACHE_DIR, 'twus_dev.pickle')
_TWITTER_PARSED_TRAIN_DATA = path.join(constants.DATACACHE_DIR, 'twus_train.pickle')


def load_state_data():
    train_df, dev_df, test_df = _load_data()

    x_train = train_df['tweets'].values
    y_train = train_df['state'].values

    x_dev = dev_df['tweets'].values
    y_dev = dev_df['state'].values

    x_test = test_df['tweets'].values
    y_test = test_df['state'].values

    return (x_train, y_train, x_dev, y_dev, x_test, y_test)


def load_region_data():
    train_df, dev_df, test_df = _load_data()

    x_train = train_df['tweets'].values
    y_train = train_df['region'].values

    x_dev = dev_df['tweets'].values
    y_dev = dev_df['region'].values

    x_test = test_df['tweets'].values
    y_test = test_df['region'].values

    return (x_train, y_train, x_dev, y_dev, x_test, y_test)


def _load_data():
    if not path.exists(_TWITTER_PARSED_DEV_DATA):
        _extract_twitter_data(_TWITTER_DEV_DATA, _TWITTER_PARSED_DEV_DATA)

    if not path.exists(_TWITTER_PARSED_TEST_DATA):
        _extract_twitter_data(_TWITTER_TEST_DATA, _TWITTER_PARSED_TEST_DATA)

    if not path.exists(_TWITTER_PARSED_TRAIN_DATA):
        _extract_twitter_data(_TWITTER_TRAIN_DATA, _TWITTER_PARSED_TRAIN_DATA)

    with open(_TWITTER_PARSED_DEV_DATA, 'rb') as handle:
        dev_data = pickle.load(handle)

    with open(_TWITTER_PARSED_TEST_DATA, 'rb') as handle:
        test_data = pickle.load(handle)

    with open(_TWITTER_PARSED_TRAIN_DATA, 'rb') as handle:
        train_data = pickle.load(handle)

    dev_df = pd.DataFrame(dev_data, columns=['username', 'tweets', 'state', 'region', 'state_name', 'region_name'])
    test_df = pd.DataFrame(test_data, columns=['username', 'tweets', 'state', 'region', 'state_name', 'region_name'])
    train_df = pd.DataFrame(train_data, columns=['username', 'tweets', 'state', 'region', 'state_name', 'region_name'])
    return (train_df, dev_df, test_df)


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
    percent_pt = total_lines // 1000
    for i in range(0, len(data)):
        if (i % percent_pt == 0):
            print("\r{0}% complete...".format(i / percent_pt / 10), end='')

        username = data[i][0]
        stateStr = geocoder.reverse_geocode_state((data[i][1], data[i][2]))
        if not stateStr:
            continue

        tweets = data[i][3]
        words = word_tokenize(tweets)
        words = [ps.stem(w) for w in words]
        tweets = ' '.join(words)

        state = geocoder.get_state_index(stateStr)
        region = geocoder.get_state_region(stateStr)
        regionStr = geocoder.get_state_region_name(stateStr)
        row = (username, tweets, state, region, stateStr, regionStr)
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
