from typing import Dict, Tuple
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, f1_score
from dvclive import Live
from fusanet_utils.transforms import Collate_and_transform
import pandas as pd

live = Live()
logger = logging.getLogger(__name__)

def create_dataloaders(dataset, params: Dict):
    train_size = int(params["train"]["train_percent"]*len(dataset))
    valid_size = len(dataset) - train_size
    train_subset, valid_subset = random_split(dataset, (train_size, valid_size), generator=torch.Generator().manual_seed(params["train"]["random_seed"]))
    my_collate = Collate_and_transform(params['features'])
    train_loader = DataLoader(train_subset, shuffle=True, batch_size=params["train"]["batch_size"], collate_fn=my_collate)
    valid_loader = DataLoader(valid_subset, batch_size=256, collate_fn=my_collate)
    return train_loader, valid_loader

def train(loaders: Tuple, params: Dict, model_path: str, cuda: bool) -> None:
    """
    Make more abstract to other models
    """
    train_loader, valid_loader = loaders 
    n_train, n_valid = len(train_loader.dataset), len(valid_loader.dataset)    
    model = torch.load(model_path)    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['train']['learning_rate'])

    if cuda and torch.cuda.device_count() > 0:
        logger.info('GPU number: {}'.format(torch.cuda.device_count()))
        device = 'cuda'
    else:
        device = 'cpu'
    logger.info(f'Using {device}')  

    best_valid_loss = np.inf
    for epoch in range(params["train"]["nepochs"]):
        logger.info(f"Starting epoch {epoch}")
        global_loss = 0.0  
        global_accuracy = 0.0  
        model.to(device)
        model.train()
        for batch in train_loader:
            marshalled_batch = {}
            for key in batch:
                marshalled_batch[key] = batch[key].to(device)
            optimizer.zero_grad()
            y = model.forward(marshalled_batch)
            loss = criterion(y, marshalled_batch['label'])
            loss.backward()
            optimizer.step()
            global_loss += loss.item()
            accuracy = torch.sum(y.argmax(dim=1) == marshalled_batch['label'])
            global_accuracy += accuracy.item()
        logger.info(f"{epoch}, train/loss {global_loss/n_train:0.4f}")
        logger.info(f"{epoch}, train/accuracy {global_accuracy/n_train:0.4f}")
        live.log('train/loss', global_loss/n_train)
        live.log('train/accuracy', global_accuracy/n_train)
        
        global_loss = 0.0
        global_accuracy = 0.0  
        global_f1_score = 0.0
        model.eval()
        with torch.no_grad():
            for n_batch, batch in enumerate(valid_loader, start=1):
                marshalled_batch = {}
                for key in batch:
                    marshalled_batch[key] = batch[key].to(device)
                y = model.forward(marshalled_batch)
                loss = criterion(y, marshalled_batch['label'])
                global_loss += loss.item()
                accuracy = torch.sum(y.argmax(dim=1) == marshalled_batch['label'])
                global_accuracy += accuracy.item() 
                global_f1_score += f1_score(marshalled_batch['label'].cpu(), y.cpu().argmax(dim=1), average='macro')
        logger.info(f"{epoch}, valid/loss {global_loss/n_valid:0.4f}")
        logger.info(f"{epoch}, valid/accuracy {global_accuracy/n_valid:0.4f}")
        logger.info(f"{epoch}, f1_score macro {global_f1_score/n_batch:0.4f}")
        live.log('valid/loss', global_loss/n_valid)
        live.log('valid/accuracy', global_accuracy/n_valid)
        live.log('f1_score macro', global_f1_score/n_batch)
        live.next_step()

        if global_loss < best_valid_loss:
            if device == 'cuda': model.cpu()
            torch.save(model, model_path)
            model.create_trace()

def evaluate_model(loaders: Tuple, params: Dict, model_path: str) -> None:
    train_loader, valid_loader = loaders
    model = torch.load(model_path)
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in valid_loader:
            preds.append(model.forward(batch).argmax(dim=1).numpy())
            labels.append(batch['label'].numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    label_list = list(train_loader.dataset.dataset.label_dictionary().values())
    report = classification_report(labels, preds, target_names=label_list, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.to_csv('report.csv')
    logger.info('Reporte exportado con éxito')





