import torch
from torch import Tensor

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