import data.extract_twus_data as twdata
import pickle


class TwitterUser:
    def __init__(self):
        self._location_latitude = None
        self._location_longitude = None
        self._tweets = []
        self._state = None
        self._region = None
        self._username = None

    def _parse_rawdata(self):
        pass

    def get_tweets(self):
        return self._tweets

    def get_us_state(self):
        pass

    def set_us_state(self, state):
        self._state = state

    def get_us_region(self):
        pass

    def get_username(self):
        return self._username

    def set_username(self, username):
        self._username = username

    def set_tweets(self, tweets):
        self._tweets = tweets

    def get_tweets(self):
        return self._tweets


def load_twitter_users():
    users = []
    states_dev = {}
    tweets_dev = {}
    ## Need to figure out how to make this work for all paths

    with open("data/" + twdata.STATES_DEV_DATA_FILE, 'rb') as handle:
        states_dev = pickle.load(handle)
    with open("data/" + twdata.TWEETS_DEV_DATA_FILE, 'rb') as handle:
        tweets_dev = pickle.load(handle)

    for username, state in states_dev.items():
        user = TwitterUser()
        user.set_us_state(state)
        user.set_username(username)
        user.set_tweets(tweets_dev[username])
        users.append(user)

    return users
