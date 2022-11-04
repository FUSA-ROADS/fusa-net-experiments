import numpy as np
import holoviews as hv
hv.extension('bokeh')
import panel as pn
import torch
import librosa
from torchaudio.transforms import MelSpectrogram
from functools import partial
from IPython.display import Audio

def create_sed_gui(sample, model, params, categories):
    
    mel_transform_pann = MelSpectrogram(sample_rate=params['features']["sampling_rate"],
                                    n_fft=1024, hop_length=320, n_mels=64, normalized=False)
    get_rms = partial(librosa.feature.rms, frame_length=1024, hop_length=320)

    with torch.no_grad():
        preds = model(sample).detach().numpy()

    logmel = (mel_transform_pann(sample['waveform']) + 1e-7).log10()
    rms = get_rms(sample['waveform'][0])

    n_windows, n_classes = preds[0].shape
    n_mels, _ = logmel[0].shape

    rms_plot = hv.Curve(rms[0], 'time', 'rms').opts(height=120, width=700)
    logmel_plot = hv.Image(logmel[0].numpy(), bounds=(0, 0, n_windows, n_mels), 
                           kdims=['time', 'frequency']).opts(height=150, width=700, cmap='Blues', invert_yaxis=True)
    preds_plot = hv.Image(preds[0].T, bounds=(0, 0, n_windows, n_classes),
                            kdims=['time', 'class']).opts(height=350, width=700, cmap='Blues', 
                                                          yticks=[(n_classes-int(tick)-0.5, label) for tick, label in zip(np.arange(33), categories)], 
                                                          clim=(0, 1), invert_yaxis=True)
    bg_plot = hv.Layout([rms_plot, logmel_plot, preds_plot]).cols(1)

    frame_slider = pn.widgets.IntSlider(name="Time", value=0, start=0, end=n_windows)
    length = 2*params['features']["sampling_rate"]

    @pn.depends(frame=frame_slider)
    def cross_hair(frame):
        return hv.VLine(frame).opts(color='red') * hv.VLine(frame+200).opts(color='red')

    cross_hair_dmap = hv.DynamicMap(cross_hair)
    plots = (bg_plot * cross_hair_dmap).cols(1)

    def prepare_wav_for_panel(waveform):
        max_val = np.amax(np.abs(waveform))
        return np.int16(waveform * 32767 / max_val)


    @pn.depends(frame=frame_slider)
    def listen(frame):
        idx_wav = int(frame*320)
        return pn.pane.Audio(prepare_wav_for_panel(sample['waveform'][0, idx_wav:idx_wav+length].numpy()), 
                             sample_rate=params['features']["sampling_rate"], name='Audio', autoplay=True)

    app = pn.Column(
        pn.Spacer(height=10),
        frame_slider,
        plots,
        listen,
    )
    return app