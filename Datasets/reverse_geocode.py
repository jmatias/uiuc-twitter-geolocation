import requests
import sys, re, sqlalchemy

'''SELECT pprint_addy(r.addy[1]) As State
        FROM reverse_geocode(ST_GeomFromText('POINT(-89.3985 40.63)',4269),true) As r;
'''

db = sqlalchemy.create_engine("postgresql://postgres:admin@localhost/geo")


def reverse_geocode_state(location):
    result_set = db.execute(
        "SELECT pprint_addy(r.addy[1]) As State FROM reverse_geocode(ST_GeomFromText('POINT(-89.3985 40.63)',4269),true) As r;")

    for r in result_set:
        return r[0]

    return None



    # url = 'http://open.mapquestapi.com/geocoding/v1/batch'
    # locations = ["{0},{1}".format(lat, long) for lat, long in location_list]
    #
    # params = {'location': locations, 'key': 'G0Mo8DLGi48aYGjFdHUlfKZIKsSNVqVK'}
    # r = requests.get(url, params=params)
    # results = r.json()['results']
    # return [loc['locations'][0]['adminArea3'] for loc in results]


print(reverse_geocode_state("lol"))
