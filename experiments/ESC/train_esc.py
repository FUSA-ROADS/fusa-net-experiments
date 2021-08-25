import sys 
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import dvclive

sys.path.append('../..')
from transforms import Collate_and_transform, RESIZER
from datasets import FUSA_dataset, ESC

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train_esc.py data_path model_path\n")
    sys.exit(1)

data_path = sys.argv[1]
model_path = sys.argv[2]
"""
data_path = "../datasets"
model_path = "model.pt"
"""
params = yaml.safe_load(open("../../params.yaml"))
print(params)
dataset = FUSA_dataset(ConcatDataset([ESC(data_path)]), feature_params=params["features"])
train_size = int(params["train"]["train_percent"]*len(dataset))
valid_size = len(dataset) - train_size
train_subset, valid_subset = random_split(dataset, (train_size, valid_size), generator=torch.Generator().manual_seed(params["train"]["random_seed"]))
my_collate = Collate_and_transform(resizer=RESIZER.PAD)
train_loader = DataLoader(train_subset, shuffle=True, batch_size=params["train"]["batch_size"], collate_fn=my_collate)
valid_loader = DataLoader(valid_subset, batch_size=256, collate_fn=my_collate)

from model import NaiveModel
model = NaiveModel(n_classes=len(dataset.categories))
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=params['train']['learning_rate'])

best_valid_loss = np.inf
for epoch in range(params["train"]["nepochs"]):
    global_loss = 0.0
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        y = model.forward(batch['logmel'])
        loss = criterion(y, batch['label'])
        loss.backward()
        optimizer.step()
        global_loss += loss.item()
    dvclive.log('train/loss', global_loss/len(train_subset))
    
    global_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            y = model.forward(batch['logmel'])
            loss = criterion(y, batch['label'])
            global_loss += loss.item()            
    dvclive.log('valid/loss', global_loss/len(valid_subset))
    dvclive.next_step()

    if global_loss < best_valid_loss: 
        torch.save(model, model_path)