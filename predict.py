from models.twitter_geomodel import Model
from data import twus

x_train, y_train, x_dev, y_dev, x_test, y_test = twus.load_state_data()

geoModel = Model(num_outputs=53,batch_size=192)
lol = geoModel.predict(x_test)
lol2 = geoModel.evaluate(x_test,y_test)
print(lol2)


geoModel.load_saved_model('geomodel_state.h5')
lol2 = geoModel.evaluate(x_test,y_test)
print(lol2)

print("Hello")
