import os.path
import yaml
import logging
import json
import argparse
import torch
from torch.utils.data import ConcatDataset
from fusanet_utils.datasets.external import ESC, UrbanSound8K
from fusanet_utils.datasets.fusa import FUSA_dataset

import trainer
from model import NaiveModel

def dir_path(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--root_path', dest='root_path', help='path to the root of this repo', type=dir_path, default='../../')
    parser.add_argument('--model_path', dest='model_path', help='path to save/load model', type=str, default='model.pt')  
    parser.add_argument('--train', action='store_true')  
    parser.add_argument('--evaluate', action='store_true')  
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    # Logging
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
    
    dataset_param = params['train']['dataset']
    if dataset_param == 'ESC':
        train_dataset = [ESC(args.root_path)]
    elif dataset_param == 'US':
        train_dataset = [UrbanSound8K(args.root_path)]
    else:
        train_dataset = [ESC(args.root_path), UrbanSound8K(args.root_path)]
    # Create dataset for the experiment and save dictionary of classes index to names
    dataset = FUSA_dataset(ConcatDataset(train_dataset), feature_params=params["features"])
    with open('index_to_name.json', 'w') as f:
        json.dump(dataset.label_dictionary(), f)
    main_logger.info("Creating dataloaders")
    loaders = trainer.create_dataloaders(dataset, params)
    if args.train:
        # Save initial model
        model = NaiveModel(n_classes=len(dataset.categories))
        torch.save(model, args.model_path)
        
        main_logger.info("Main: Training")
        trainer.train(loaders, params, args.model_path, args.cuda)
    if args.evaluate:
        main_logger.info("Main: Evaluating")
        trainer.evaluate_model(loaders, params, args.model_path)
    
