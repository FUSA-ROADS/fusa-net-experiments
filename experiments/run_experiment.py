import os.path
import yaml
import logging
import json
import argparse
from fusanet_utils.experiments.initializer import initialize_model, create_dataset, create_dataloaders
from fusanet_utils.experiments.trainer import train_model
from fusanet_utils.experiments.evaluator import evaluate_model


def dir_path(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--root_path', dest='root_path',
                        help='path to the root of this repo',
                        type=dir_path, default='../')
    parser.add_argument('--model_path', dest='model_path',
                        help='path to save/load model',
                        type=str, default='model.pt')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--verbose', help="Print info level logs",
                        action="store_true")
    parser.add_argument('--debug', help="Print debug level logs",
                        action="store_true")
    args = parser.parse_args()

    # Logging
    logging_level = logging.WARNING
    if args.verbose:
        logging_level = logging.INFO
    if args.debug:
        logging_level = logging.DEBUG
    main_logger = logging.getLogger()
    main_logger.setLevel(logging_level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    main_logger.addHandler(stream_handler)

    main_logger.info("Loading parameters, dataset and model")
    params = yaml.safe_load(open("params.yaml"))
    if args.train:
        dataset = create_dataset(args.root_path, params, stage='train')
        with open('index_to_name.json', 'w') as f:
            json.dump(dataset.label_dictionary(), f)
        main_logger.info("Creating dataloaders")
        loaders = create_dataloaders(dataset, params)
        initialize_model(args.model_path, params['train'],
                         len(dataset.categories))
        main_logger.info("Main: Training")
        train_model(loaders, params, args.model_path)
    if args.evaluate:
        main_logger.info(f"Main: Evaluating on {params['evaluate']['dataset']}")
        dataset = create_dataset(args.root_path, params, stage='evaluate')
        with open('index_to_name.json', 'r') as f:
            label_dictionary = json.load(f)
        evaluate_model(dataset, params, args.model_path, label_dictionary)
