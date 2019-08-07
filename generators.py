# -*-config:utf-8 -* 
import random

import numpy as np
import time

from keras.utils import Sequence
from sklearn.utils import shuffle
from threading import Thread, Lock

from utils import *

class BatchGenerator(Sequence):
    def __init__(self, 
                 data, 
                 vocab,
                 config, 
                 preprocess_audio,
                 preprocess_texte,
                 shuffle=False):
        self.data = data.sort_values(by='mel_frames').reset_index(drop=True)
        self.vocab = vocab
        self.hp = config['HPARAMS']
        
        self.preprocess_audio = preprocess_audio
        self.preprocess_texte = preprocess_texte

        self.batch_size     = config['BATCH_SIZE']

        self.max_len_phrase = config['MAX_LEN_PHRASE']
        self.vocab_size     = len(vocab) + 3

        self.outputs_per_step = self.hp.outputs_per_step
        self.audio_rate     = config['AUDIO_RATE']
        self.fft_len        = self.hp.n_fft
        self.num_freq       = self.hp.num_freq
        self.num_mels       = self.hp.num_mels
        self.win_size       = self.hp.win_size
        self.hop_size       = self.hp.hop_size
        
        #pad input sequences with the <pad_token> 0 ( _ )
        self._input_pad = 0
        if self.hp.symmetric_mels:
            self._target_pad = -self.hp.max_abs_value
        else:
            self._target_pad = 0.
        self._token_pad = 1.

        self.shuffle = shuffle
        self.on_epoch_end()
        
        
    def __str__(self):
        return """Informations générales :
        Nombre d'éléments dans le dataset : {}
        Batch_size : {} 
        Nombre de tour par epoch : {}
        \nInformations sur l'audio :
        Temps total : {}
        Audio rate : {} 
        FFT length : {} 
        \nInformations sur les phrases :
        Max length : {} 
        Vocab size : {}""".format(len(self.data), self.batch_size, self.__len__(), time_to_string(np.sum(self.data['audio_time'].values)), self.audio_rate, self.fft_len, self.max_len_phrase, self.vocab_size)
    
    def __len__(self):
        return len(self.batches)
    
    def size(self):
        return len(self.data)        
    
    def __getitem__(self, idx):
        if idx >= len(self.batches): idx = idx % len(self.batches)
        l_bound = self.batches[idx]
        r_bound = l_bound + self.batch_size
        if l_bound >= len(self.data):
            l_bound = l_bound % len(self.data)
            r_bound = l_bound + self.batch_size
        if r_bound >= len(self.data):
            r_bound = len(self.data)
            l_bound = r_bound - self.batch_size
        
        zero_mel = np.zeros((1, self.num_mels))
        
        liste_phrases   = []
        liste_mel       = []
        liste_next_mel  = []
        liste_linear    = []
        liste_stop      = []
        
        for b, (i, row) in enumerate(self.data.iloc[l_bound : r_bound].iterrows()):
            phrase = self.preprocess_texte(row['text'])
            
            mel_output, linear = self.preprocess_audio(row)

            mel_input = np.append(zero_mel, mel_output, axis=0)

            stop_token = np.zeros((len(mel_output),1))
            stop_token[-1,0] = 1.
            
            liste_phrases.append(phrase)
            liste_mel.append(mel_input)
            liste_next_mel.append(mel_output)
            liste_linear.append(linear)
            liste_stop.append(stop_token)
        
        if self.batch_size > 1:
            liste_phrases, liste_mel, liste_next_mel, liste_linear, liste_stop = self.build_batch(liste_phrases, liste_mel, liste_next_mel, liste_linear, liste_stop)
        else:
            liste_phrases   = np.array(liste_phrases)
            liste_mel       = np.array(liste_mel)
            liste_next_mel  = np.array(liste_next_mel)
            liste_linear    = np.array(liste_linear)
            liste_stop      = np.array(liste_stop)
            
        if self.outputs_per_step > 1:
            liste_mel, liste_next_mel, liste_linear, liste_stop = self.reshape_batch(liste_mel, liste_next_mel, liste_linear, liste_stop)
        else:
            liste_mel = liste_mel[:,:-1]
        
        return [liste_phrases, liste_mel], [liste_next_mel, liste_next_mel, liste_linear, liste_stop]
    
    def on_epoch_end(self):
        self.batches = np.arange(0, len(self.data), self.batch_size)
        if self.shuffle:
            np.random.shuffle(self.batches)
            
    def build_batch(self, liste_phrases, liste_mel, liste_next_mel, liste_linear, liste_stop):
        maxlen_phrases  = max([len(p) for p in liste_phrases])
        maxlen_frames   = max([len(f) for f in liste_mel])
        maxlen_frames_input = maxlen_frames + 1
        
        batch_phrases   = np.full((self.batch_size, maxlen_phrases), self._input_pad)
        batch_mel       = np.full((self.batch_size, maxlen_frames_input, self.num_mels), self._target_pad)
        batch_next_mel  = np.full((self.batch_size, maxlen_frames, self.num_mels), self._target_pad)
        batch_linear    = np.full((self.batch_size, maxlen_frames, self.num_freq), self._target_pad)
        batch_stop      = np.full((self.batch_size, maxlen_frames, 1), self._token_pad)
        
        for b in range(self.batch_size):
            phrase  = liste_phrases[b]
            mel     = liste_mel[b]
            next_mel= liste_next_mel[b]
            linear  = liste_linear[b]
            stop    = liste_stop[b]
            
            batch_phrases[b,:len(phrase)] = phrase
            batch_mel[b,:len(mel)] = mel
            batch_next_mel[b,:len(next_mel)] = next_mel
            batch_linear[b,:len(linear)] = linear
            batch_stop[b,:len(stop)] = stop
            
        return batch_phrases, batch_mel, batch_next_mel, batch_linear, batch_stop

    def reshape_batch(self, liste_mel, liste_next_mel, liste_linear, liste_stop):
        r = liste_next_mel.shape[1] % self.outputs_per_step
        if r != 0:
            pad_mel = np.full((self.batch_size, self.outputs_per_step - r, self.num_mels), self._input_pad)
            pad_linear = np.full((self.batch_size, self.outputs_per_step - r, self.num_freq), self._target_pad)
            pad_stop = np.full((self.batch_size, self.outputs_per_step - r, 1), self._token_pad)

            liste_mel = np.append(liste_mel, pad_mel, axis=1)
            liste_next_mel = np.append(liste_next_mel, pad_mel, axis=1)
            liste_linear = np.append(liste_linear, pad_linear, axis=1)
            liste_stop = np.append(liste_stop, pad_stop, axis=1)

        liste_mel = np.reshape(liste_mel[:,:-1,:], (self.batch_size, -1, self.num_mels * self.outputs_per_step))
        return liste_mel, liste_next_mel, liste_linear, liste_stop


class ParallelGenerator(Thread):
    def __init__(self, fn, *args, **kwargs):
        Thread.__init__(self)
        self.mutex = Lock()
        
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        
        self.result = None
        
    def run(self):
        try:
            self.mutex.acquire()
            self.result = self.fn(*self.args, **self.kwargs)
        finally:
            self.mutex.release()
        
    def get_result(self):
        with self.mutex:
            return self.result
        
    def clear(self):
        del self.result