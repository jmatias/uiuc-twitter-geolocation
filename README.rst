Twitter Geolocation Predictor
=============================

This is a deep-learning tool to predict the location of a Twitter user
based solely on the text content of his/her tweets without any other
form of metadata.


Overview
--------

The Twitter Geolocation Predictor is a Recurrent Neural Network
classifier. Every training sample is a collection of tweets labeled with
a location (e.g. country, state, city, etc.). The model will
tokenize all tweets into a sequence of words, and feed them into an
`Embedding Layer <https://en.wikipedia.org/wiki/Word_embedding>`__. The
embeddings will learn the meaning of words and use them as input for two
stacked `Long-Short Term
Memory <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__
layers. A `Softmax <https://en.wikipedia.org/wiki/Softmax_function>`__
fully-connected layer at the end yields the classification result.

    
.. image:: https://dl.dropbox.com/s/tvar2ccihtq0ijg/GeoModelGraph.png
   :width: 500px
   :align: center



Getting Started
---------------

Installation
~~~~~~~~~~~~

Clone the repository and install all the dependencies using pip.

.. code:: console

    $ sudo pip3 install -r requirements.txt

This will install the latest CPU version of Tensorflow. If you would
like to run on a GPU, follow the Tensorflow-GPU `installation
instructions <https://www.tensorflow.org/install/>`__.

Using A Pre-Processed Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tool comes with a built-in dataset of ~430K users located in the
U.S. (~410K for training, ~10K for development and ~10K for testing). To
train a model using this dataset, run the train.py sample script.

.. code-block:: bash

    $ python3 train.py --epochs 5 --batch_size 64 --vocab_size 20000

    Using TensorFlow backend.
    Downloading data from https://dl.dropbox.com/s/ze4ov5j30u9rf5m/twus_test.pickle
    55181312/55180071 [==============================] - 11s 0us/step
    Downloading data from https://dl.dropbox.com/s/kg09i1z32n12o98/twus_dev.pickle
    57229312/57227360 [==============================] - 12s 0us/step
    Downloading data from https://dl.dropbox.com/s/0d4l6jmgguzonou/twus_train.pickle
    2427592704/2427591168 [==============================] - 486s 0us/step

    Building model...
    Hidden layer size: 100
    Analyzing up to 500 words for each sample.
    Building tweet Tokenizer using a 20,000 word vocabulary. This may take a while...
    Tokenizing 419,869 tweets. This may take a while...
    Training model...
    Train on 410336 samples, validate on 9533 samples
    Epoch 1/5
      4608/410336 [..............................] - ETA: 39:11 - loss: 3.7145 - acc: 0.0911 - top_5_acc: 0.3092

You can also try using this data from your own source code.

.. code-block:: ipython

    In [1]: from data import twus
    Using TensorFlow backend.

    In [2]: x_train, y_train, x_dev, y_dev, x_test, y_test = twus.load_state_data()

    In [3]: x_train.shape
    Out[3]: (410336,)

    In [4]: y_train.shape
    Out[4]: (410336,)

Training the Model
------------------

.. code:: python

    from twgeo.models.geomodel import Model
    from twgeo.data import twus
    
    # x_train is an array of text. Each element contains all the tweets for a given user. 
    # y_train is an array of integer values, corresponding to each particular location we want to train against.
    x_train, y_train, x_dev, y_dev, x_test, y_test = twus.load_state_data()

    # num_outputs is the total number of possible classes (locations). In this example, 50 US states plus 3 territories.
    # time_steps is the total number of individual words to consider for each user.
    # Some users have more tweets then others. In this example, we are capping it at a total of 500 words per user.
    geoModel = Model(num_outputs=53, batch_size=64, time_steps=500,
                     vocab_size=20000)
                     
    geoModel.train(x_train, y_train, x_dev, y_dev, epochs=5)
    geoModel.save_model('mymodel.h5')

Making Predictions
------------------

.. code:: python

    from twgeo.models.geomodel import Model
    from twgeo.data import twus

    x_train, y_train, x_dev, y_dev, x_test, y_test = twus.load_state_data()

    geoModel.load_saved_model('mymodel.h5')
    results = geoModel.predict(x_test)

