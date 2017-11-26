from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import time

import re

start_time = time.time()

TWEETS_TEST_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.test"
TWEETS_DEV_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.dev"
TWEETS_TRAIN_DATA = "/home/javier/project/TwitterGeolocation/Datasets/na/user_info.train"

DATA_DIR = "/home/javier/harddrive/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/"
VOCAB_FILE = DATA_DIR + "vocab.txt"
EMBEDDING_MATRIX_FILE = DATA_DIR + "embeddings.npy"
CHECKPOINT_PATH = DATA_DIR + "model.ckpt-501424"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
MR_DATA_DIR = "/home/javier/harddrive/skip_thoughts/rt-polaritydata"

TWEETS_DATA_DIR = ""

encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)

data = []
# with open(os.path.join(MR_DATA_DIR, 'rt-polarity.neg'), 'rb') as f:
#   data.extend([line.decode('latin-1').strip() for line in f])
# with open(os.path.join(MR_DATA_DIR, 'rt-polarity.pos'), 'rb') as f:
#   data.extend([line.decode('latin-1').strip() for line in f])

# data.append("incredibly boring movie")


regex_pattern = "(-?\d+\.\d+)\s+?(-?\d+\.\d+)\t(.+)\|{3}"
data = []
with open(TWEETS_DEV_DATA, 'rb') as f:
    data.extend(
        [re.sub("((@\w+))", "", l).strip() for line in f for m in [re.search(regex_pattern, line.decode("utf-8").strip())]
         if m for l in m.group(3).split("|||") if re.sub("((@\w+))", "", l).strip()])

data[0] = "I am so tired. I could sleep for days."
encodings = encoder.encode(data[0:350000])


def get_nn(ind, num=10):
    encoding = encodings[ind]
    scores = sd.cdist([encoding], encodings, "cosine")[0]
    sorted_ids = np.argsort(scores)
    print("Sentence:")
    print("", data[ind])
    print("\nNearest neighbors:")
    for i in range(1, num + 1):
        print(" %d. %s (%.3f)" %
              (i, data[sorted_ids[i]], scores[sorted_ids[i]]))
    print("\n\n")


current_time = time.time()
duration = current_time - start_time


for i in range(0, 35000, 250):
    get_nn(i)

get_nn(0)
get_nn(5)
get_nn(6)
get_nn(10)
get_nn(500)
get_nn(600)
get_nn(700)
get_nn(800)
get_nn(900)
get_nn(1000)
get_nn(1100)
get_nn(1300)
get_nn(1500)
get_nn(15000)
get_nn(16000)
get_nn(17000)
get_nn(18000)

print("Duration {0:.3f} secs".format(duration))
