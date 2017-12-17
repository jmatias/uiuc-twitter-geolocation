
from os import path, environ

if __name__ == '__main__':
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from twgeo.models.geomodel import Model
    from twgeo.data import twus_dataset, constants

    x_train, y_train, x_dev, y_dev, x_test, y_test = twus_dataset.load_state_data()

    geoModel = Model()

    geoModel.load_saved_model(path.join(constants.DATACACHE_DIR, 'geomodel_state'))
    lol = geoModel.predict(x_test)
    lol2 = geoModel.evaluate(x_test, y_test)
    print(lol2)

    print("Hello")
