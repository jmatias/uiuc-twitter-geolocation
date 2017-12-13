Twitter Geolocation Predictor
=============================

This is a deep-learning tool to predict the location of a Twitter user
based solely on the text content of his/her tweets without any other
form of metadata.

Contents
--------

-  `Overview <#overview>`__
-  `Getting Started <#getting-started>`__
    -  `Installation <#installation>`__
    -  `Using a Pre-Processed Dataset <#using-a-pre-processed-dataset>`__
    -  [Supplying your own data]
-  `Training the Model <#training-the-model>`__
-  `Making Predictions <#making-predictions>`__

Overview
--------

The Twitter Geolocation Predictor is a Recurrent Neural Network
classifier. Every training sample is a collection of tweets labeled with
a location (e.g. a country, a state, a region, etc.). The model will
tokenize all tweets into a sequence of words, and feed them into an
`Embedding Layer <https://en.wikipedia.org/wiki/Word_embedding>`__. The
embeddings will learn the meaning of words and use them as input for two
stacked `Long-Short Term
Memory <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__
layers. A `Softmax <https://en.wikipedia.org/wiki/Softmax_function>`__
fully-connected layer at the end yields the classification result.

.. raw:: html

    <img src="https://dl.dropbox.com/s/qxmkayuswz2hs04/GeoModelGraph.png" height="600px">

Getting Started
---------------

Installation
~~~~~~~~~~~~

Clone the repository, and install all the dependencies using pip.

.. code:: sh

    sudo pip3 install -r requirements.txt

This will install the latest CPU version of Tensorflow. If you would
like to run on a GPU, follow the Tensorflow-GPU `installation
instructions <https://www.tensorflow.org/install/>`__.

Using A Pre-Processed Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tool comes with a built-in dataset of ~430K users located in the
U.S. (~410K for training, ~10K for development and ~10K for testing). To
train a model using this dataset, run the train.py sample script.

::

    python3 train.py --epochs 5 --batch_size 64 --vocab_size 20000

    Using TensorFlow backend.
    Downloading data from https://dl.dropbox.com/s/ze4ov5j30u9rf5m/twus_test.pickle
    55181312/55180071 [==============================] - 11s 0us/step
    Downloading data from https://dl.dropbox.com/s/kg09i1z32n12o98/twus_dev.pickle
    57229312/57227360 [==============================] - 12s 0us/step
    Downloading data from https://dl.dropbox.com/s/0d4l6jmgguzonou/twus_train.pickle
    Downloading data from https://dl.dropbox.com/s/0d4l6jmgguzonou/twus_train.pickle
    2427592704/2427591168 [==============================] - 486s 0us/step
    Building tweet Tokenizer using a 20,000 word vocabulary. This may take a while...
    Tokenizing 419,869 tweets. This may take a while...
    Training model...
    Train on 410336 samples, validate on 9533 samples
    Epoch 1/5
      4608/410336 [..............................] - ETA: 39:11 - loss: 3.7145 - acc: 0.0911 - top_5_acc: 0.3092

You can also try using this data from your own source code.

.. code:: ipython

    In [1]: from data import twus
    Using TensorFlow backend.

    In [2]: x_train, y_train, x_dev, y_dev, x_test, y_test = twus.load_state_data()

    In [3]: x_train.shape
    Out[3]: (410336,)

    In [4]: y_train.shape
    Out[4]: (410336,)

Training the Model
------------------

Making Predictions
------------------
