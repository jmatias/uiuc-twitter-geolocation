from typing import List
import data.extract_twus_data as twdata
import data.reverse_geocode as rg
from skip_thoughts import encoder_manager as em
import pickle
import numpy as np
import time
import keras
import math


class TwitterUser:
    def __init__(self, encoder: em.EncoderManager, geocoder: rg.ReverseGeocode):
        self._location_latitude = None
        self._location_longitude = None
        self._tweets = []
        self._state = None
        self._username = None
        self._encoder = encoder
        self._geocoder = geocoder

    @property
    def us_state(self):
        return self._state

    @us_state.setter
    def us_state(self, state):
        self._state = state

    @property
    def us_state_id(self):
        if self._state is None:
            raise ValueError("State is None")
        return self._geocoder.get_state_index(self._state)

    @property
    def us_region(self):
        return self._geocoder.get_state_region(self._state)

    @property
    def us_region_name(self):
        return self._geocoder.get_state_region_name(self._state)

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, username):
        self._username = username

    @property
    def tweets(self):
        return self._tweets

    @tweets.setter
    def tweets(self, tweets):
        self._tweets = tweets

    @property
    def encoder(self) -> em.EncoderManager:
        return self._encoder

    def to_thought_vectors(self):
        return self.encoder.encode(self.tweets, use_norm=False)

    def thought_vector_mean(self):
        try:
            return np.mean(self.to_thought_vectors(), axis=0)
        except ValueError as e:
            print("WTF!")
            print(e)


def load_twitter_users(encoder: em.EncoderManager, dataset='dev') -> List[TwitterUser]:
    geocoder = rg.ReverseGeocode()
    users = []

    ## Need to figure out how to make this work for all paths
    if (dataset == 'train'):
        states_data_file = twdata.STATES_TRAIN_DATA_FILE
        tweets_data_file = twdata.TWEETS_TRAIN_DATA_FILE
    elif (dataset == 'dev'):
        states_data_file = twdata.STATES_DEV_DATA_FILE
        tweets_data_file = twdata.TWEETS_DEV_DATA_FILE
    elif (dataset == 'test'):
        states_data_file = twdata.STATES_TEST_DATA_FILE
        tweets_data_file = twdata.TWEETS_TEST_DATA_FILE
    else:
        raise ValueError("Dataset value is not valid. Valid values: 'train','test', 'dev'", dataset)

    with open("data/" + states_data_file, 'rb') as handle:
        states_dev = pickle.load(handle)
    with open("data/" + tweets_data_file, 'rb') as handle:
        tweets_dev = pickle.load(handle)

    for username, state in states_dev.items():
        user = TwitterUser(encoder, geocoder)
        user.us_state = state
        user.username = username
        user.tweets = tweets_dev[username]
        if state:
            users.append(user)

    return users


def get_raw_tweet_list(twitter_users: List[TwitterUser]):
    tweet_list = []
    for user in twitter_users:
        tweet_list += user.tweets
    return tweet_list


def get_mean_thought_vectors(twitter_users: List[TwitterUser]):
    vectors = np.zeros(shape=(len(twitter_users), 2402))
    vector_users = {}
    i = 0

    start_time = time.time()
    for user in twitter_users:
        vectors[i] = np.hstack((user.thought_vector_mean(), np.array([user.us_region, user.us_state_id])))
        vector_users[user.username] = vectors[i]
        i += 1

        if (i % 10000 == 0):
            with open("data/user_vector_means.train", 'wb') as handle:
                pickle.dump(vector_users, handle)

        if (i % 100 == 0):
            end_time = time.time()

            print("Iteration {0} - {1}".format(i, time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))
            start_time = end_time

    return vectors


def get_all_data(twitter_users: List[TwitterUser]):
    vectors_x = np.zeros(shape=(len(twitter_users), 200, 2400))
    vectors_y = np.zeros(shape=(len(twitter_users), 2))

    i = 0
    start_time = time.time()
    for user in twitter_users:
        j = 1
        encoded_tweets = user.encoder.encode(user.tweets[:200], use_norm=False)

        for tweet in encoded_tweets:
            vectors_x[i, -j] = tweet
            j += 1

        vectors_y[i, 0] = user.us_state_id
        vectors_y[i, 1] = user.us_region
        i += 1

        if (i % 100 == 0):
            end_time = time.time()

            print("Iteration {0} - {1}".format(i, time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))
            start_time = end_time

    # with open("data/user_vectors_x.train", 'wb') as handle:
    #     pickle.dump(vectors_x, handle)
    #
    # with open("data/user_vectors_y.train", 'wb') as handle:
    #     pickle.dump(vectors_y, handle)


    return vectors_x, vectors_y


def get_all_data_generator(twitter_users: List[TwitterUser]):
    while True:

        for batch in range(math.ceil(len(twitter_users) / 500)):
            vectors_x = np.zeros((500, 200, 2400))
            vectors_y = np.zeros((500, 5))

            users = twitter_users[batch * 500: batch * 500 + 500]
            i = 0
            for user in users:
                encoded_tweets = user.encoder.encode(user.tweets[:200], use_norm=False)
                j = 1
                for tweet in encoded_tweets:
                    vectors_x[i, -j] = tweet
                    j += 1

                vectors_y[i] = keras.utils.to_categorical([user.us_region], num_classes=5)
                i += 1

            yield vectors_x, vectors_y


def get_max_tweet_count(twitter_users: List[TwitterUser]):
    max = 0

    for user in twitter_users:
        num_tweets = len(user.tweets)
        if (num_tweets > max):
            max = num_tweets
    return max
