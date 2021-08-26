import sys 
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split


sys.path.append('../..')
from transforms import Collate_and_transform, RESIZER
from datasets import FUSA_dataset, ESC

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train_esc.py data_path model_path\n")
    sys.exit(1)

data_path = sys.argv[1]
model_path = sys.argv[2]

params = yaml.safe_load(open("../../params.yaml"))
params["features"]["overwrite"] = False
print(params)
dataset = FUSA_dataset(ConcatDataset([ESC(data_path)]), feature_params=params["features"])
train_size = int(params["train"]["train_percent"]*len(dataset))
valid_size = len(dataset) - train_size
train_subset, valid_subset = random_split(dataset, (train_size, valid_size), generator=torch.Generator().manual_seed(params["train"]["random_seed"]))
my_collate = Collate_and_transform(resizer=RESIZER.PAD)
train_loader = DataLoader(train_subset, shuffle=True, batch_size=params["train"]["batch_size"], collate_fn=my_collate)
valid_loader = DataLoader(valid_subset, batch_size=256, collate_fn=my_collate)

from model import NaiveModel
model = torch.load(model_path)
model.eval()
preds = []
labels = []
with torch.no_grad():
    for batch in valid_loader:
        preds.append(model.forward(batch['logmel']).argmax(dim=1).numpy())
        labels.append(batch['label'].numpy())

from sklearn.metrics import classification_report
preds = np.concatenate(preds)
labels = np.concatenate(labels)
print(classification_report(preds, labels, target_names=dataset.label_int2string(torch.arange(len(dataset.categories)))))


