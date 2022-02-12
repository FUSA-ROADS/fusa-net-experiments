import os.path
import yaml
import logging
import json
import argparse
import torch
import fusanet_utils.experiment_manager as manager

def dir_path(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--root_path', dest='root_path', help='path to the root of this repo', type=dir_path, default='../')
    parser.add_argument('--model_path', dest='model_path', help='path to save/load model', type=str, default='model.pt')  
    parser.add_argument('--train', action='store_true')  
    parser.add_argument('--evaluate', action='store_true')  
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', help="Print info level logs", action="store_true")
    parser.add_argument('--debug', help="Print debug level logs", action="store_true")
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
    
    dataset = manager.create_dataset(args.root_path, params)
    with open('index_to_name.json', 'w') as f:
        json.dump(dataset.label_dictionary(), f)
    main_logger.info("Creating dataloaders")
    loaders = manager.create_dataloaders(dataset, params)
    if args.train:
        # Save initial model
        manager.initialize_model(args.model_path, params['train'], len(dataset.categories), args.cuda)
        main_logger.info("Main: Training")
        manager.train(loaders, params, args.model_path, args.cuda)
    if args.evaluate:
        main_logger.info("Main: Evaluating")
        manager.evaluate_model(loaders, params, args.model_path)
    
