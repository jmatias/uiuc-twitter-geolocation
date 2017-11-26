



class TwitterUser:
    def __init__(self, rawdata):
        self._rawdata = rawdata
        self._location_latitude = None
        self._location_longitude = None
        self._tweets = []
        self._state = None
        self._region = None


    def _parse_rawdata(self):
        pass


    def get_tweets(self):
        return self._tweets

    def get_us_state(self):
        pass

    def get_us_region(self):
        pass

