import os.path
import yaml
import logging
import json
import argparse
import torch
from torch.utils.data import ConcatDataset, DataLoader
from fusanet_utils.datasets.external import ESC, UrbanSound8K, FolderDataset
from fusanet_utils.datasets.fusa import FUSA_dataset

import trainer
from model import Wavegram_Logmel_Cnn14

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
    if args.train:
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
    
        # Save initial model
        model = Wavegram_Logmel_Cnn14(
            n_classes=527,
            sampling_rate=32000,
            n_fft=1024,
            hop_length=320,
            n_mels=64,
            fmin=50,
            fmax=14000
            )
        if args.cuda:
            checkpoint = torch.load('Wavegram_Logmel_Cnn14_mAP=0.439.pth')
        else:
            checkpoint = torch.load('Wavegram_Logmel_Cnn14_mAP=0.439.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        torch.save(model, args.model_path)
        main_logger.info("Main: Training")
        trainer.train(loaders, params, args.model_path, args.cuda)

    if args.evaluate:
        dataset_param = params['evaluate']['dataset']
        main_logger.info(f"Main: Evaluating on {dataset_param}")
        if dataset_param == 'VitGlobal':
            evaluate_dataset = [FolderDataset(os.path.join(args.root_path, 'datasets', 'VitGlobal', 'audio', 'dataset'))]
        with open('index_to_name.json', 'r') as f:
            label_dictionary = json.load(f)
        dataset = FUSA_dataset(ConcatDataset(evaluate_dataset), feature_params=params["features"])
        trainer.evaluate_model(dataset, params, args.model_path, label_dictionary)
    
