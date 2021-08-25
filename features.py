from os.path import isfile, splitext
from typing import Dict
import torch
import torchaudio

def get_waveform(file_path: str, params: Dict) -> torch.Tensor:
    waveform, origin_sr = torchaudio.load(file_path)
    origin_ch = waveform.size()[0]
    target_sr = params["sampling_rate"]
    target_ch = params["number_of_channels"]
    if not origin_sr == target_sr:
        waveform = torchaudio.transforms.Resample(origin_sr, target_sr)(waveform)
    if target_ch == 1 and origin_ch == 2:
        how_to = params['combine_channels']
        if how_to == 'mean':
            return torch.mean(waveform, dim=0, keepdim=True)
        elif how_to == 'left':
            return waveform[0, :].view(1,-1)
        elif how_to == 'right':
            return waveform[1, :].view(1,-1)
    elif target_ch == 2 and origin_ch == 1:
        return waveform.repeat(2, 1)
    else:
        return waveform

class LogMelTransform:

    def __init__(self, waveform_path: str, params: Dict={}, overwrite: bool=False, eps: float=1e-3):
        self.logmel_path = splitext(waveform_path)[0]+"_logmel.pt"
        if not isfile(self.logmel_path) or overwrite:
            waveform =  get_waveform(waveform_path, params)
            sample_rate = params["sampling_rate"]
            mel_params = params["mel_transform"]
            mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=mel_params['n_fft'], hop_length=mel_params['hop_length'], n_mels=mel_params['n_mels'], normalized=mel_params["normalized"])
            logmel = (mel_transform(waveform)+eps).log()
            torch.save(logmel, self.logmel_path)

    def __call__(self):
        return torch.load(self.logmel_path)
        