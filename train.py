import argparse

from os import path, environ

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-d", "--dataset_size", type=str, help="Dataset size.", default='small',
                        choices=['micro', 'small', 'mid', 'large'])
    parser.add_argument("-c", "--classifier", type=str, help="Train a US State or US Census Region classifier.",
                        default='state',
                        choices=['region', 'state'])
    parser.add_argument("--max_words", type=int, help="Max number of words to analyze per user.", default=100)
    parser.add_argument("-v", "--vocab_size", type=int, help="Use the top N most frequent words.", default=20000)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size.", default=32)
    parser.add_argument("--hidden_size", type=int, help="Number of neurons in the hidden layers.", default=100)
    parser.add_argument("--tensorboard", action="store_true", help="Track training progress using Tensorboard.",
                        default=True)
    args = parser.parse_args()

    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from twgeo.data import twus_dataset, constants
    from twgeo.models.geomodel import Model

    if args.classifier == 'state':
        num_of_classes = 53
        x_train, y_train, x_dev, y_dev, x_test, y_test = twus_dataset.load_state_data(size=args.dataset_size)
        print("train.py: Training a US State classifier.")
    else:
        num_of_classes = 5
        x_train, y_train, x_dev, y_dev, x_test, y_test = twus_dataset.load_region_data(size=args.dataset_size)
        print("train.py: Training a US Census Region classifier.")

    geoModel = Model(batch_size=args.batch_size, use_tensorboard=args.tensorboard)

    geomodel_state_model_file = path.join(constants.DATACACHE_DIR, 'geomodel_' + args.classifier)
    if path.exists(geomodel_state_model_file + ".h5"):
        print("Loading existing model at {0}".format(geomodel_state_model_file))
        geoModel.load_saved_model(geomodel_state_model_file)
    else:
        geoModel.build_model(num_outputs=num_of_classes, time_steps=args.max_words, vocab_size=args.vocab_size,
                             hidden_layer_size=args.hidden_size)

    geoModel.train(x_train, y_train, x_dev, y_dev, epochs=args.epochs)
    geoModel.save_model(geomodel_state_model_file)

    #workaround for bug https://github.com/tensorflow/tensorflow/issues/3388
    import gc
    gc.collect()
