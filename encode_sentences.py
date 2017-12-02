from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import time
import data.twitter_user as twuser



start_time = time.time()


'''
Encode tweets as thought vectors and then find its ten closest neighbors.
'''

TWEETS_TEST_DATA = "data/na/user_info.test"
TWEETS_DEV_DATA = "data/na/user_info.dev"
TWEETS_TRAIN_DATA = "data/na/user_info.train"

DATA_DIR = "data/pretrained/skip_thoughts_uni_2017_02_02/"
VOCAB_FILE = DATA_DIR + "vocab.txt"
EMBEDDING_MATRIX_FILE = DATA_DIR + "embeddings.npy"
CHECKPOINT_PATH = DATA_DIR + "model.ckpt-501424"

encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)

twitter_users = twuser.load_twitter_users(encoder, dataset='train')
tweet_list = twuser.get_raw_tweet_list(twitter_users)
vector_list = twuser.get_mean_thought_vectors(twitter_users)

encodings = encoder.encode(tweet_list[0:35000],use_norm=False)




def get_nn(ind, num=10):
    encoding = encodings[ind]
    scores = sd.cdist([encoding], encodings, "cosine")[0]
    sorted_ids = np.argsort(scores)
    print("Sentence:")
    print("", tweet_list[ind])
    print("\nNearest neighbors:")
    for i in range(1, num + 1):
        print(" %d. %s (%.3f)" %
              (i, tweet_list[sorted_ids[i]], scores[sorted_ids[i]]))
    print("\n\n")


current_time = time.time()
duration = current_time - start_time

for i in range(0, 35000, 250):
    get_nn(i)


print("Duration {0:.3f} secs".format(duration))
