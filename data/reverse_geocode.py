import sqlalchemy


class ReverseGeocode():
    def __init__(self):
        self._db = sqlalchemy.create_engine("postgresql://postgres:admin@localhost/geo")

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
