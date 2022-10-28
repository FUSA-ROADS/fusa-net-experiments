from torch.utils.data import Dataset
from typing import Tuple
from pathlib import Path
import pandas as pd

translate_classes_aumilab = {'Musica': 'music', 
                     'Automovil_movimiento': 'car_moving', 
                     'Automovil': 'car_idling', 
                     'Ave': 'bird', 
                     'Perro': 'dog',
                     'Impacto': 'impact',
                     'Viento': 'wind',
                     'Grito': 'shouting',
                     'Alarma': 'alarm', 
                     'Motocicleta_movimiento': 'motorcycle_moving',
                     'Camion': 'truck_idling',
                     'Camion_movimiento': 'truck_moving',
                     'Conversacion': 'talk',
                     'Multitud': 'crowd',
                     'Bocina': 'horn',
                     'Sirena': 'siren',
                     'Lluvia': 'rain',
                     'Bus_movimiento': 'bus_moving',
                             'Pasos': 'steps',
                             'Bus': 'bus_idling',
                     'Motocicleta': 'motorcycle_idling',
                            'Vehiculo_aereo': 'airborne',
                            'Corte': 'cutting',
                            'Frenado': 'braking',
                            'Excavacion': 'drilling',
                             'Agua': 'water'
                            }

def annotations2dataframe(annotations):
    data = {}
    if len(annotations[0]['result']) == 0:
        return pd.DataFrame(columns=['start (s)', 'end (s)', 'class'])
    for k, event in enumerate(annotations[0]['result']):
        data[k] = {'start (s)': event['value']['start'], 
                   'end (s)': event['value']['end'], 
                   'class': translate_classes_aumilab[event['value']['labels'][0]]} 
    return pd.DataFrame(data).T

class Aumilab_labeled_dataset(Dataset):
    
    def __init__(self, fusa_merged, categories):
        data_path = Path('../../datasets/AUMILAB/imported/csv_file_61/')
        self.file_list = fusa_merged["_id"].apply(lambda x: data_path / (x+".wav") )
        assert all(self.file_list.apply(lambda x : x.exists())), "Missing files"
        self.label_list = fusa_merged["annotations"].apply(annotations2dataframe)
        self.categories = categories
            
    def __getitem__(self, idx: int) -> Tuple[Path, pd.DataFrame]:
        return (self.file_list[idx], self.label_list[idx])

    def __len__(self) -> int:
        return len(self.file_list)
    
    def listen(self, idx: int):
        data, sr = librosa.load(self.file_list[idx])        
        return Audio(data, rate=sr, autoplay=True)
    
import torch.nn.functional as F
from functools import partial

def scipy_snr(signal, dim):
    return signal.mean(dim=dim)/signal.std(dim=dim)

def torch_rms(signal, dim):
    return signal.pow(2).mean(dim=dim).sqrt()

def apply_to_windows(func, input, n_fft, hop_length, center=True, pad_mode='reflect'):
    if center:
        signal_dim = input.dim()
        extended_shape = [1] * (3 - signal_dim) + list(input.size())
        pad = int(n_fft // 2)
        input = F.pad(input.view(extended_shape), [pad, pad], pad_mode)
        input = input.view(input.shape[-signal_dim:])
    input = input.unsqueeze(-1)    
    return func(F.unfold(input, kernel_size=(n_fft, 1), stride=(hop_length, 1)))