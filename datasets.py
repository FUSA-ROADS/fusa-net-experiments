from os.path import join, isfile, splitext
from typing import Dict, List, Tuple
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchaudio


datasets_path = "."

def get_label_transforms(dataset_name):
    a = pd.read_json("fusa_taxonomy.json").T[dataset_name].to_dict()
    transforms = {}
    for key, values in a.items():
        for value in values:
            transforms[value] = key
    return transforms

class ExternalDataset(Dataset):

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return (self._file_path(idx), self.labels[idx])
        
    def __len__(self) -> int:        
        return len(self.file_list)

class ESC(ExternalDataset):

    def __init__(self):
        
        # TODO: Abstract parts of init
        label_transforms = get_label_transforms("ESC")
        df = pd.read_csv(join(datasets_path, "ESC-50/meta/esc50.csv"))
        ESC_classes = df["category"].unique()
        # Verify that there are no typos in FUSA_taxonomy
        if not all([key in set(ESC_classes) for key in label_transforms.keys() if key != ""]):
            warnings.warn("Existen llaves de ESC que no calzan en fusa_taxonomy.json", UserWarning)
        
        self.audio_path = join(datasets_path, "ESC-50/audio")
        # Verify that files exist
        file_exist = df["filename"].apply(lambda x: isfile(join(join(datasets_path, "ESC-50/audio"), x)))
        if not file_exist.all():
            warnings.warn("Existen rutas incorrectas o archivos perdidos", UserWarning)
            df = df.loc[file_exist]
        
        self.file_list, self.labels, self.categories = [], [], []
        for label in ESC_classes:
            if label in label_transforms:
                self.categories += [label_transforms[label]]
                mask = df.category == label
                self.file_list += list(df["filename"].loc[mask])
                self.labels += [label_transforms[label]]*sum(mask)

    def _file_path(self, idx: int) -> str:
        return join(self.audio_path, self.file_list[idx])


class UrbanSound8K(ExternalDataset):
    
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


    def _file_path(self, idx: int) -> str:
        return join(self.audio_path, self.fold_list[idx], self.file_list[idx])


class FUSAv1(Dataset):

    def __init__(self, target_sample_rate: int=44100, overwrite_features: bool=False, waveform_transform=None, return_logmel=True, **kwargs):
        self.dataset = ConcatDataset([ESC(), UrbanSound8K()])
        self.categories = []
        for d in self.dataset.datasets:
            self.categories += d.categories
        self.categories = sorted(list(set(self.categories)))
        self.le = LabelEncoder().fit(self.categories)
        self.waveform_transform = waveform_transform
        self.target_sample_rate = target_sample_rate
        self.overwrite_features = overwrite_features
        self.return_logmel = return_logmel
        self.kwargs = kwargs        
        

    def __getitem__(self, idx: int) -> Dict:
        file_path, label = self.dataset[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        # Force resample
        waveform = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)(waveform)
        if self.waveform_transform is not None:
            waveform = self.waveform_transform(waveform)
        sample = {'waveform': waveform, 'label': torch.from_numpy(self.le.transform([label]))}
        # TODO: Implemente overwrite features
        if self.return_logmel:
            logmel_path = splitext(file_path)[0]+"_logmel.pt"
            if isfile(logmel_path):
                logmel = torch.load(logmel_path)
            else:
                mel_params = self.kwargs['mel_transform']
                mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.target_sample_rate, n_fft=mel_params['n_fft'], hop_length=mel_params['hop_length'], n_mels=mel_params['n_mels'])
                logmel = mel_transform(waveform).log()
                torch.save(logmel, logmel_path)
            sample['logmel'] = logmel
                
        return sample

    def __len__(self) -> int:
        return self.dataset.__len__()

    def label_int2string(self, label_batch: torch.Tensor) -> List[int]:
        return list(self.le.inverse_transform(label_batch.numpy().ravel()))
        
    
if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from transforms import StereoToMono, Collate_and_transform
    import yaml
    params = yaml.safe_load(open("params.yaml"))
    dataset = FUSAv1(target_sample_rate=params["sample_rate"], waveform_transform=StereoToMono(), **params)
    my_collate = Collate_and_transform(pad=True)
    loader = DataLoader(dataset, shuffle=True, batch_size=5, collate_fn=my_collate)
    for batch in loader:
        break
    print(batch['waveform'].shape)
    print(batch['logmel'].shape)
    print(dataset.label_int2string(batch['label']))    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(batch['logmel'].detach().numpy()[0, 0])

