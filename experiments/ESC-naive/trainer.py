from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import dvclive    
from fusanet_utils.transforms import Collate_and_transform, RESIZER


def create_dataloaders(dataset, params: Dict):
    train_size = int(params["train"]["train_percent"]*len(dataset))
    valid_size = len(dataset) - train_size
    train_subset, valid_subset = random_split(dataset, (train_size, valid_size), generator=torch.Generator().manual_seed(params["train"]["random_seed"]))
    my_collate = Collate_and_transform(resizer=RESIZER.PAD)
    train_loader = DataLoader(train_subset, shuffle=True, batch_size=params["train"]["batch_size"], collate_fn=my_collate)
    valid_loader = DataLoader(valid_subset, batch_size=256, collate_fn=my_collate)    
    return train_loader, valid_loader

def train(loaders: Tuple, params: Dict, model_path: str) -> None:
    """
    Make more abstract to other models
    """
    train_loader, valid_loader = loaders 
    n_train, n_valid = len(train_loader.dataset), len(valid_loader.dataset)    
    model = torch.load(model_path)    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['train']['learning_rate'])

    best_valid_loss = np.inf
    for epoch in range(params["train"]["nepochs"]):
        global_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            y = model.forward(batch)
            loss = criterion(y, batch['label'])
            loss.backward()
            optimizer.step()
            global_loss += loss.item()
        print(f"{epoch}, train/loss {global_loss/n_train:0.4f}")
        dvclive.log('train/loss', global_loss/n_train)
        
        global_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                y = model.forward(batch)
                loss = criterion(y, batch['label'])
                global_loss += loss.item()            
        print(f"{epoch}, valid/loss {global_loss/n_valid:0.4f}")
        dvclive.log('valid/loss', global_loss/n_valid)
        dvclive.next_step()

        if global_loss < best_valid_loss: 
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
    print(classification_report(preds, labels, target_names=label_list))




