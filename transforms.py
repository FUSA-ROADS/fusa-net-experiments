import torch
from torch import Tensor
from torch.nn.functional import pad
from typing import Dict, List, Optional

class StereoToMono(torch.nn.Module):
    """Convert stereo audio to mono."""
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of stereo audio of dimension (..., time).

        Returns:
            Tensor: Output mono signal of dimension (..., time).
        """
        if waveform.size()[0] == 1:
            return waveform
        return (torch.mean(waveform, 0)).view(1,-1)


class Collate_and_transform:
    
    def __init__(self, transforms: List=[], pad: bool=False, crop: bool=False):
        self.transforms = transforms
        self.pad = pad
        # TODO: Implement crop
        #.crop = crop

    def __call__(self, batch: List[Dict]):
        transformed_batch = []
        has_logmel = 'logmel' in batch[0] # If one has it everyone has it
        if self.pad:
            audio_lens = [sample['waveform'].size(-1) for sample in batch]  
            max_len_audio = max(audio_lens)  
            if has_logmel:
                logmel_lens = [sample['logmel'].size(-1) for sample in batch]  
                max_len_logmel = max(logmel_lens)
        for i, sample in enumerate(batch):
            if self.pad:
                sample['waveform'] = pad(sample['waveform'], (0, max_len_audio-audio_lens[i]))
                if has_logmel:
                    sample['logmel'] = pad(sample['logmel'], (0, max_len_logmel-logmel_lens[i]))                    
            #if self.crop:
            #    sample['waveform'] = sample['waveform'][-1, :min_len]
            #    if 'log_mel' in sample:
            #        sample['log_mel'] = sample['log_mel'][-1, :min_len]
            for transform in self.transforms:
                sample = transform(sample)
            transformed_batch.append(sample)
        mbatch = {}
        mbatch['waveform'] =  torch.stack([sample['waveform'] for sample in transformed_batch], dim=0)
        if has_logmel:
            mbatch['logmel'] =  torch.stack([sample['logmel'] for sample in transformed_batch], dim=0)
        mbatch['label'] = torch.LongTensor([sample['label'] for sample in transformed_batch])
        return mbatch

