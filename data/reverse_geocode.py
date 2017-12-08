import sqlalchemy
import pandas as pd
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'state_table.csv')

state_table = pd.read_csv(filename)


class ReverseGeocode():
    def __init__(self):
        #self._db = sqlalchemy.create_engine("postgresql://postgres:admin@localhost/geo")
        self._db = None

    def reverse_geocode_state(self, location):
        """Find the corresponding US state of a given pair of coordinates.
        :param location: A tuple containing the (latitude, longitude)
        :return: The corresponding state. Example: 'FL'
        """
        point = "'POINT({1} {0})'".format(location[0], location[1])

        result_set = self._db.execute(
            "SELECT pprint_addy(r.addy[1]) As State FROM reverse_geocode(ST_GeomFromText({0},4269),true) As r;".format(
                point))

        for r in result_set:
            return r[0]

        raise ValueError("Could not find state for {0},{1}".format(location[0], location[1]), location)

    def get_state_region_name(self, state):
        if state is None:
            raise ValueError('state may not be None.')

        try:
            return state_table[state_table["abbreviation"] == state]['census_region_name'].values[0]
        except Exception as e:
            print(e)
            print(state)

    def get_state_region(self, state):
        if state is None:
            raise ValueError('state may not be None.')

        try:
            return state_table[state_table["abbreviation"] == state]['census_region'].values[0]
        except Exception as e:
            print(e)
            print(state)

    def get_state_index(self, state):
        if state is None:
            raise ValueError('state may not be None.')

        try:
            return state_table[state_table["abbreviation"] == state]['id'].values[0]
        except Exception as e:
            print(e)
            print(state)
