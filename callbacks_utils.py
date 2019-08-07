import os
import random
import numpy as np

from keras.callbacks import *

import audio
from plot_utils import *


def get_callbacks(folder,
                  save_weights_name = "best_weights.h5",
                  with_early_stopping   = False,
                  patience_stop  = 3,
                  with_checkpoint   = True,
                  with_tensorboard  = True,
                  with_reduce_lr    = True,
                  reduce_factor     = 0.5,
                  patience_reduce   = 1,
                  monitor   = 'val_loss'
                 ):
    callbacks = []
        
    if with_early_stopping:
        early_stop = EarlyStopping(monitor=monitor, 
                                   min_delta=0.001, 
                                   patience=patience_stop, 
                                   mode='min', 
                                   verbose=1)
        callbacks.append(early_stop)
        
    if with_checkpoint:
        checkpoint = ModelCheckpoint(folder + "/" + save_weights_name, 
                                     monitor=monitor, 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min', 
                                     period=1)
        callbacks.append(checkpoint)
        
    if with_tensorboard:
        if not os.path.exists(folder + "/tensorboard"): 
            os.mkdir(folder + "/tensorboard")
        tensorboard = TensorBoard(log_dir=folder + "/tensorboard", 
                                  histogram_freq=0, 
                                  #write_batch_performance=True,
                                  write_graph=True, 
                                  write_images=False)
        callbacks.append(tensorboard)
    
    if with_reduce_lr:
        reduceLR = ReduceLROnPlateau(monitor=monitor,
                                     factor = reduce_factor,
                                     patience=patience_reduce,
                                     min_lr = 1e-5)
        callbacks.append(reduceLR)
        
    return callbacks

class PredictionCallback(Callback):
    def __init__(self, tacotron, generator, log_dir, steps_per_epoch):
        self.tacotron = tacotron
        self.generator = generator
        self.log_dir = log_dir
        self.steps_per_epoch = steps_per_epoch
        
        self.mel_dir = os.path.join(log_dir, "mels")
        self.linear_dir = os.path.join(log_dir, "linear")
        self.plot_dir = os.path.join(log_dir, "plots")
        self.wav_dir = os.path.join(log_dir, "wavs")
        
        os.makedirs(self.wav_dir, exist_ok=True)
        os.makedirs(self.mel_dir, exist_ok=True)
        os.makedirs(self.linear_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        print("Making prediction at epoch {} (step {})".format(epoch+1, (epoch+1)*self.steps_per_epoch))
        num = int(random.random() * len(self.generator))
        [p_in, mel_in], y = self.generator.__getitem__(num)
        p_in = p_in[:1]
        mel_in = mel_in[:1]
        self.make_prediction([p_in, mel_in], y, epoch)
        
    def make_prediction(self, x, y, epoch):
        _, target_mel, target_linear, _ = y
        _, mel, linear, _, alignment = self.tacotron._predict_with_target(*x)
        mel = mel[0]
        linear = linear[0]
        alignment = alignment[0]
            
        step = (epoch + 1) * self.steps_per_epoch
        mel_filename    = os.path.join(self.mel_dir, "step-{}.npy".format(step))
        linear_filename = os.path.join(self.linear_dir, "step-{}.npy".format(step))
        plot_mel_filename = os.path.join(self.plot_dir, "mel_step-{}.png".format(step))
        plot_linear_filename = os.path.join(self.plot_dir, "linear_step-{}.png".format(step))
        plot_align_filename = os.path.join(self.plot_dir, "align_step-{}.png".format(step))
        wav_mel_filename = os.path.join(self.wav_dir, "wav_from_mel_step-{}.wav".format(step))
        wav_linear_filename = os.path.join(self.wav_dir, "wav_from_linear_step-{}.wav".format(step))

        wav_from_linear = self.tacotron.wav_from_linear(linear)
        wav_from_mel = self.tacotron.wav_from_mel(mel)
            
        np.save(mel_filename, mel)
        np.save(linear_filename, linear)

        audio.save_wav(wav_from_mel, wav_mel_filename, self.tacotron.audio_rate)
        audio.save_wav(wav_from_linear, wav_linear_filename, self.tacotron.audio_rate)

        plot_alignment(alignment, title="Alignments", filename=plot_align_filename, show=False, fontsize=14)
        plot_spectrogram(mel, title="Mel spectrogram", filename=plot_mel_filename, show=False, target_spectrogram=target_mel[0])
        plot_spectrogram(linear, title="Linear spectrogram", filename=plot_linear_filename, show=False, target_spectrogram=target_linear[0])
