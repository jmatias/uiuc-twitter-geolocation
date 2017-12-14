import sqlalchemy
import pandas as pd
import os

_dirname = os.path.dirname(__file__)
_filename = os.path.join(_dirname, 'state_table.csv')

state_table = pd.read_csv(_filename)


class ReverseGeocode():
    """
    Optional module to find the US state and Census region from a pair of global coordinates.
    This class assumes you have a Postgres PostGIS database installed locally. To setup PostGIS with US Census data,
    follow the instructions `found here <http://postgis.net/docs/postgis_installation.html#loading_extras_tiger_geocoder/>`_.
    """

    def __init__(self):

        self._db = None

    def reverse_geocode_state(self, location) -> str:
        """
        Find the corresponding US state of a given pair of coordinates.

        :param location: A tuple containing the (latitude, longitude)
        :return: The corresponding state abbreviation. Example: 'FL, NY'
        """

        if self._db is None:
            self._db = sqlalchemy.create_engine("postgresql://postgres:admin@localhost/geo")

        point = "'POINT({1} {0})'".format(location[0], location[1])

        result_set = self._db.execute(
            "SELECT pprint_addy(r.addy[1]) As State FROM reverse_geocode(ST_GeomFromText({0},4269),true) As r;".format(
                point))

        for r in result_set:
            return r[0]

        raise ValueError("Could not find state for {0},{1}".format(location[0], location[1]), location)

    def get_state_region_name(self, state_abbrev) -> str:
        """
        Get the name of the Census region of a given state.

        :param state_abbrev: Example: 'FL, NY'
        :return:
        """
        if state_abbrev is None:
            raise ValueError('state may not be None.')

        try:
            return state_table[state_table["abbreviation"] == state_abbrev]['census_region_name'].values[0]
        except Exception as e:
            pass

    def get_state_region(self, state_abbrev) -> int:
        """
        Get the integer value of the Census region of the given state.

        :param state_abbrev: Example: 'FL, NY'
        :return:
        """
        if state_abbrev is None:
            raise ValueError('state may not be None.')

        try:
            return state_table[state_table["abbreviation"] == state_abbrev]['census_region'].values[0]
        except Exception as e:
            pass

    def get_state_index(self, state_abbrev) -> int:
        """
        Get the integer value of the given state.

        :param state_abbrev: Example: 'FL, NY'
        :return:
        """
        if state_abbrev is None:
            raise ValueError('state may not be None.')
        try:
            return state_table[state_table["abbreviation"] == state_abbrev]['id'].values[0]
        except Exception as e:
            pass
