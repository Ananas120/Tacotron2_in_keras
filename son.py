# -*-config:utf-8 -* 

import scipy.signal
import numpy as np
import pandas as pd

from scipy.io import wavfile as wav
from scipy.fftpack import fft, rfft, ifft, irfft
from scipy.signal import stft, istft

from plot_utils import plot

def multiply_frequency(son, facteur):
    import parselmouth
    
    from parselmouth.praat import call
    
    manipulation = call(son, "To Manipulation", 0.001, 75, 600)
    
    pitchs = call(manipulation, "Extract pitch tier")
    call(pitchs, "Multiply frequencies", son.xmin, son.xmax, facteur)
    call([pitchs, manipulation], "Replace pitch tier")
    
    return call(manipulation, "Get resynthesis (overlap-add)")

def roll_factor(array, facteur_init, facteur_max=None, progression=None, keep_dim=True):
    if facteur_max is None: facteur_max = facteur_init
    if progression is None: progression = len(array)
    if keep_dim:
        new_array = np.zeros(len(array))
    else:
        raise notImplementedError()
        
    facteur_augm = (facteur_max - facteur_init) / progression
    facteur = facteur_init
    for i, val in enumerate(array):
        if int(i * facteur) >= len(new_array):
            break
        new_array[int(i*facteur)] = val
        facteur += facteur_augm
    return new_array


def build_son_from_fft(temporal_fft, rate=44100):
    array = irfft(temporal_fft[0])
    for ligne in temporal_fft[1:]:
        array = np.append(array, irfft(ligne))
    return Son(array, rate)

def build_son_from_stft(fourrier, hp):
    return istft(fourrier, fs=hp.sample_rate, noverlap=hp.hop_size, nperseg=hp.win_size, nfft=hp.n_fft)[-1]
    
class Son(object):
    def __init__(self, array, rate):
        self.array = array
        if (len(self.array.shape) > 1):
            self.array = self.array[0,:]
        self.rate = rate
        
    def __len__(self):
        return len(self.array)
    
    def duree(self):
        return len(self.array) / self.rate
                
    def play(self, output=True, start=0, time=None):
        import pyaudio
        format = pyaudio.paInt16
        
        array = self.array
        if array.dtype != np.int16:
            array = array.astype(np.int16)
        p = pyaudio.PyAudio()
        channels = array.shape[0] if len(array.shape) > 1 else 1
        print(self.rate, format, channels, output)
        stream = p.open(rate=self.rate, format=format, channels=channels, output=output)
        if time is None:
            fin = len(array)
        else:
            fin = int(time*self.rate + start)
        stream.write((array[start:fin]).tobytes())
        
        stream.close()
        p.terminate()
        
    def plot(self, start=0, time=None, **plot_kwargs):
        if time is None: 
            fin = len(self.array)
        else:
            fin = start + time * self.rate
        plot(self.array[start:fin], **plot_kwargs)
        
    def temporal_fft(self, window_length=1024):
        nombre = len(self.array) // window_length
        resultat = np.zeros((nombre,window_length))
        for i in range(nombre):
            resultat[i,:] = rfft(self.array[i * window_length : (i+1) * window_length])
        return resultat
    
    def _stft(self, nfft, win_size, hop_size):
        return stft(self.array, fs=self.rate, noverlap=hop_size, nperseg=win_size, nfft=nfft)
            
    def change_rate(self, new_rate):
        if new_rate == self.rate: return
        print("Resample from {} to {}".format(self.rate, new_rate))
        new_length = int(self.duree() * new_rate)
        self.array = scipy.signal.resample(self.array, new_length)
        self.rate = new_rate
        
    def normalize(self, amplitude_max):
        self.array *= amplitude_max / max(0.1, np.max(np.abs(self.array)))
        
    def save(self, filename):
        wav.write(filename, self.rate, self.array)
        
    def gen_from_wav(filename):
        rate, array = wav.read(filename)
        return Son(np.array(array), rate)
    gen_from_wav = staticmethod(gen_from_wav)
    
    def display(self):
        from IPython.display import Audio
        return Audio(data=self.array, rate=self.rate)
