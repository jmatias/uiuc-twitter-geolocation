import argparse

from os import path, environ

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("--max_words", type=int, help="Max number of words to analyze per user.", default=500)
    parser.add_argument("-v", "--vocab_size", type=int, help="Use the top N most frequent words.", default=20000)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size.", default=128)
    parser.add_argument("--hidden_size", type=int, help="Number of neurons in the hidden layers.", default=128)
    parser.add_argument("--tensorboard", action="store_true", help="Track training progress using Tensorboard.",
                        default=True)
    args = parser.parse_args()

    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from twgeo.data import twus_dataset, constants
    from twgeo.models.geomodel import Model

    x_train, y_train, x_dev, y_dev, x_test, y_test = twus_dataset.load_state_data(size='small')

    geoModel = Model(batch_size=args.batch_size, use_tensorboard=args.tensorboard)

    geoModel.build_model(num_outputs=53, time_steps=args.max_words, vocab_size=args.vocab_size,
                         hidden_layer_size=args.hidden_size)

    geomodel_state_model_file = path.join(constants.DATACACHE_DIR, 'geomodel_state')
    if path.exists(geomodel_state_model_file):
        print("Loading existing model at {0}".format(geomodel_state_model_file))
        geoModel.load_saved_model(geomodel_state_model_file)

    geoModel.train(x_train, y_train, x_dev, y_dev, epochs=args.epochs)
    geoModel.save_model(geomodel_state_model_file)
