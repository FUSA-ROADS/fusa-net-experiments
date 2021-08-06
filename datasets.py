from os.path import join
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torchaudio
from transforms import StereoToMono

datasets_path = "/datasets"

def get_label_transforms(dataset_name):
    a = pd.read_json("fusa_taxonomy.json").T[dataset_name].to_dict()
    transforms = {}
    for key, values in a.items():
        for value in values:
            transforms[value] = key
    return transforms

class ESC(Dataset):

    def __init__(self):
        df = pd.read_csv(join(datasets_path, "ESC-50/meta/esc50.csv"))
        label_transforms = get_label_transforms("ESC")
        self.audio_path = join(datasets_path, "ESC-50/audio")
        self.file_list = []
        self.labels = []
        self.categories = []
        for label in df["category"].unique():
            if label in label_transforms:
                self.categories += [label_transforms[label]]
                mask = df.category == label
                self.file_list += list(df["filename"].loc[mask])
                self.labels += [label_transforms[label]]*sum(mask)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(join(self.audio_path, self.file_list[idx]))
        return (waveform, self.labels[idx])
        
    def __len__(self) -> int:        
        return len(self.file_list)

class UrbanSound8K(Dataset):
    
    def __init__(self):
        df = pd.read_csv(join(datasets_path, "UrbanSound8K/metadata/UrbanSound8K.csv"))
        label_transforms = get_label_transforms("UrbanSound")
        self.audio_path = join(datasets_path, "UrbanSound8K/audio")
        self.file_list = []
        self.fold_list = []
        self.labels = []
        self.categories = []
        for label in df["class"].unique():
            if label in label_transforms:
                self.categories += [label_transforms[label]]
                mask = df["class"] == label
                self.file_list += list(df["slice_file_name"].loc[mask])
                self.fold_list += list(df["fold"].loc[mask])
                self.labels += [label_transforms[label]]*sum(mask)
        self.fold_list = ['fold' + str(fold) for fold in self.fold_list]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(
            join(self.audio_path, self.fold_list[idx], self.file_list[idx]))
        return (waveform, self.labels[idx])
        
    def __len__(self) -> int:        
        return len(self.file_list)

class FUSAv1(Dataset):

    def __init__(self, transform=None):
        self.dataset = ConcatDataset([ESC()])
        self.categories = []
        for d in self.dataset.datasets:
            self.categories += d.categories
        self.categories = sorted(list(set(self.categories)))
        self.le = LabelEncoder().fit(self.categories)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, torch.from_numpy(self.le.transform([y]))

    def __len__(self) -> int:
        return self.dataset.__len__()

    def label_int2string(self, label_batch: torch.Tensor) -> List[int]:
        return list(self.le.inverse_transform(label_batch.numpy().ravel()))
        
    
if __name__ == '__main__':

    dataset = FUSAv1(transform=StereoToMono())
    loader = DataLoader(dataset, shuffle=True, batch_size=5)
    for x, y in loader:
        break
    print(x.shape, y.shape)
    print(dataset.label_int2string(y))

