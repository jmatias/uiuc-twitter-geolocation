import argparse
import os
import pickle

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--training_data", help="Path to training data set.", default='./na/user_info.dev')
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("--max_words", type=int, help="Max number of words to analyze per user.", default=500)
    parser.add_argument("--vocab_size", type=int, help="Use the top N most frequent words.", default=50000)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size.", default=32)
    parser.add_argument("--tensorboard", action="store_true", help="Track training progress using Tensorboard.")

    args = parser.parse_args()

    from models.twitter_geomodel import Model

    geoModel = Model(num_outputs=53, epochs=args.epochs, batch_size=args.batch_size, time_steps=args.max_words,
                     train_datapath='data/user_tweets_train3.pickle', vocab_size=args.vocab_size,
                     use_tensorboard=args.tensorboard)

    dirname = os.path.dirname(__file__)

    with open(os.path.join(dirname, "data/user_tweets_train3.pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    train_df = pd.DataFrame(train_data, columns=['username', 'tweets', 'state', 'region', 'state_name', 'region_name'])
    X_train = train_df['tweets'].values
    Y_train = train_df['state'].values
    del train_df

    geoModel.train(X_train, Y_train)
    geoModel.save_model()
