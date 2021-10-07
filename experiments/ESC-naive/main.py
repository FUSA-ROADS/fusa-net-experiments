import os.path
import yaml
import json
import argparse
import torch
from torch.utils.data import ConcatDataset
from fusanet_utils.datasets.external import ESC
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
    args = parser.parse_args()

    print("Main: Loading parameters, dataset and model")
    params = yaml.safe_load(open("params.yaml"))
    print(params)
    # Create dataset for the experiment and save dictionary of classes index to names
    dataset = FUSA_dataset(ConcatDataset([ESC(args.root_path)]), feature_params=params["features"])
    with open('index_to_name.json', 'w') as f:
        json.dump(dataset.label_dictionary(), f)
    # Save initial model
    model = NaiveModel(n_classes=len(dataset.categories))
    torch.save(model, args.model_path)
    print("Main: Creating dataloaders")
    loaders = trainer.create_dataloaders(dataset, params)
    if args.train:
        print("Main: Training")
        trainer.train(loaders, params, args.model_path)
    if args.evaluate:
        print("Main: Evaluating")
        trainer.evaluate_model(loaders, params, args.model_path)
    
