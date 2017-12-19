
from __future__ import print_function

from os import path, environ

if __name__ == '__main__':
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from twgeo.models.geomodel import Model
    from twgeo.data import twus_dataset, constants

    x_train, y_train, x_dev, y_dev, x_test, y_test = twus_dataset.load_region_data()

    geoModel = Model(batch_size=650)

    geoModel.load_saved_model(path.join(constants.DATACACHE_DIR, 'geomodel_region'))
    predictions = geoModel.predict(x_test)
    evaluation = geoModel.evaluate(x_test, y_test)
    print(evaluation)
    print(predictions)

