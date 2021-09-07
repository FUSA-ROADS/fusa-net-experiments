import os.path
import yaml
import argparse
from esc import create_dataloaders, train, create_model_trace, evaluate_model


def dir_path(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--root_path', dest='root_path', help='path to the root of this repo', type=dir_path, default='../../datasets')
    parser.add_argument('--model_path', dest='model_path', help='path to save/load model', type=dir_path, default='model.pt')  
    parser.add_argument('--train', action='store_true')  
    parser.add_argument('--evaluate', action='store_true')  

    args = parser.parse_args()
    model_path = args.model_path

    params = yaml.safe_load(open("params.yaml"))
    print(params)
    
    train_loader, valid_loader = create_dataloaders(args.root_path, params)
    if args.train:
        train((train_loader, valid_loader), params, model_path)
        create_model_trace(model_path)
    elif args.evaluate:
        evaluate_model((train_loader, valid_loader), params, model_path)
    
