# -*-config:utf-8 -* 

import os
import cv2
import json
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K

from tqdm import tqdm
from functools import partial
from keras.models import *
from keras.layers import *
from keras.losses import *
from keras.callbacks import CallbackList
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
from concurrent.futures import ProcessPoolExecutor

import audio
from son import *
from utils import *
from plot_utils import *
from callbacks_utils import *
from generators import BatchGenerator, ParallelGenerator
from model import build_tacotron_model

class Tacotron(object):
    def __init__(self, 
                 hparams,
                 vocab, 
                 max_len_phrase,
                 max_audio_time,
                 nom        = "Tacotron",
                 pad_phrase = False,
                 batch_size = None,
                 metrics    = ["accuracy"],
                 backend_model      = "tacotron",
                 use_custom_linear  = True,
                 create_new_model   = False,
                ):
        
        self.nom = nom
        if not os.path.exists("modeles/" + self.nom):
            self.folder = "modeles/{}".format(nom)
            os.mkdir(self.folder)
        elif create_new_model:
            nombre = 1
            for model_nbr in os.listdir("modeles"):
                if "_" in model_nbr and model_nbr.split("_")[0] == nom:
                    nombre += 1
            self.folder = "modeles/{}_{}".format(nom, nombre)
            os.mkdir(self.folder)
        else:
            self.folder = "modeles/{}".format(nom)
        
        self.hparams    = hparams
        
        self.batch_size = batch_size
        self.is_compile = False
        
        self.vocab          = vocab
        self.vocab_size     = len(self.vocab) + 3
        self.max_len_phrase = max_len_phrase
        self.pad_phrase     = pad_phrase
        
        self.max_audio_time = max_audio_time
        self.outputs_per_step = self.hparams.outputs_per_step
        self.audio_rate     = self.hparams.sample_rate
        self.n_fft          = self.hparams.n_fft
        self.num_freq       = self.hparams.num_freq
        self.num_mels       = self.hparams.num_mels
        self.win_size       = self.hparams.win_size
        self.hop_size       = self.hparams.hop_size
        self.max_magnitude  = self.hparams.max_abs_value
        self.use_custom_linear = use_custom_linear
        
        self.backend_model  = backend_model
        self.model = None
        if not create_new_model:
            print("Chargement de l'ancien modele...")
            self._build_model()
            self.load_weights()
        if self.model is None:
            self._build_model()
            
        self.compile()
            
    def __str__(self):
        return """\n========== {} ==========\n
        \nInformations générales :
        Dossier : {}
        \nInformations sur l'audio :
        Max audio time : {} 
        Audio rate : {} 
        FFT length : {} 
        Win size : {}
        Hop size : {}
        \nInformations sur les phrases :
        Vocab size : {}
        Max length : {}""".format(self.nom, self.folder, self.max_audio_time, self.audio_rate, self.n_fft, self.win_size, self.hop_size, self.vocab_size, self.max_len_phrase)
        
            
    def _build_losses(self, loss):
        return [self._build_loss(l) for l in loss]
        
    def _build_loss(self, loss):
        if loss == 'custom':
            return [self.custom_mel_loss, self.custom_mel_loss, self.custom_linear_loss, self.custom_stop_loss]
        if loss == 'custom2':
            return [self.custom_mel_loss_2, self.custom_mel_loss_2, self.custom_linear_loss, self.custom_stop_loss]
        else:
            return loss
        
    def _build_metrics(self, metrics):
        return [self._build_metric(metric) for metric in metrics]
    
    def _build_metric(self, metric_name):
        return metric_name
    
    def _build_model(self):
        models = build_tacotron_model(self.hparams, self.vocab_size)
        self.encoder_model, self.decoder_model = models
        
    def _build_trainable_model(self):
        if self.model is not None: return self.model
        new_input_encoder = Input(shape=(None,))
        new_input_decoder = Input(shape=(None, self.num_mels * self.outputs_per_step))

        encoder_out = self.encoder_model(new_input_encoder)
        mel_before, mel_after, linear_out, stop_token_out, _ = self.decoder_model([encoder_out, new_input_decoder])

        trainable_model = Model(inputs=[new_input_encoder, new_input_decoder], 
                                outputs=[mel_before, mel_after, linear_out, stop_token_out])
        return trainable_model

    def _set_trainable(self, model, parts):
        for layer in model.layers:
            layer.trainable = False
            for partie in parts:
                if partie in layer.name or partie == "all":
                    layer.trainable = True
                    break
                    
    def custom_mel_loss_1(self, y_true, y_pred):
        err = K.abs(y_true - y_pred)
        no_pitch_mask = y_true + self.max_magnitude
        return K.mean(err) + K.mean(err * no_pitch_mask)
    
    def custom_mel_loss_2(self, y_true, y_pred):
        l1 = K.abs(y_true - y_pred)
        n_priority_freq = int(0.5 * self.num_mels)
        return 0.5 * K.mean(l1) + 0.5 * K.mean(l1[:,:,0:n_priority_freq])
    
    def custom_mel_loss(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
    def custom_stop_loss(self, y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred))
    
    def custom_linear_loss(self, y_true, y_pred):
        l1 = K.abs(y_true - y_pred)
        n_priority_freq = int(2000 / (self.audio_rate * 0.5) * self.num_freq)
        return 0.5 * K.mean(l1) + 0.5 * K.mean(l1[:,:,0:n_priority_freq])
                    
    def compile(self, optimizer='adam', loss='mse', metrics=['accuracy']):
        print("Model compilation...")
        print("Optimizer : {}\nLoss : {}\nMetrics : {}".format(optimizer, loss, metrics))
        self.encoder_model.compile(optimizer=optimizer, loss='mse', metrics=metrics)
        self.decoder_model.compile(optimizer=optimizer, loss='mse', metrics=metrics)
        self.is_compile = True
        
    def train(self, data_df,
              train_size,
              train_times,
              valid_size,
              valid_times,
              
              batch_size,
              learning_rate,
              
              pitch_scale,
              no_pitch_scale, 
              
              epochs    = 10,
              steps     = 5000,
              summary_step      = 500,
              prediction_step    = 500,
              evaluation_step    = 1000,
              
              with_original_optimizer = True,
              optimizer = 'adam',
              loss      = 'custom',
              layca     = False,
              metrics   = ["accuracy"],
              
              trainable_parts       = ["all"],
              save_config           = True,
              save_training_weights = False,
              save_weights_name     = "best_weights.h5",
              
              train_indices = None,
              valid_indices = None,
              
              shuffle   = True,
              debug     = True,
              print_train_infos = True,
              train_on_tpu  = False,
              
              with_early_stopping   = False,
              patience_stop  = 3,
              with_checkpoint   = True,
              with_tensorboard  = True,
              with_reduce_lr    = True,
              reduce_factor     = 0.5,
              patience_reduce   = 1,
              monitor   = 'val_loss',
              
              workers   = 3,
              max_queue_size    = 8,
              
              training_type = "custom_training"
             ):
        assert training_type in ("custom_training", "classic_training"), "training_type must be in (custom_training, classic_training) !"
        
        ######################################
        ########## Data preparation ##########
        ######################################
        
        self.pitch_scale = pitch_scale
        self.no_pitch_scale = no_pitch_scale
                
        if train_size < 1: train_size = int(train_size * len(data_df))
        if valid_size < 1: valid_size = int(valid_size * len(data_df))
            
        test_size = len(data_df) - train_size
        
        if train_indices is None or valid_indices is None:
            indices = np.arange(len(data_df))
            data_train, data_test = train_test_split(indices, test_size=test_size)
            if test_size != valid_size:
                data_valid, data_test = train_test_split(data_test, test_size=len(data_test) - valid_size)
            else:
                data_valid = data_test
        else:
            data_train = train_indices
            data_valid = valid_indices
        
        ##################################
        ########## Save Configs ##########
        ##################################
        
        train_config = {
              'train_size'      : train_size,
              'train_times'     : train_times,
              'valid_size'      : valid_size,
              'valid_times'     : valid_times,
              
              'epochs'          : epochs,
              'batch_size'      : batch_size,
              'learning_rate'   : learning_rate,
              
              'pitch_scale'     : pitch_scale,
              'no_pitch_scale'  : no_pitch_scale, 
              
              'with_original_optimizer' : with_original_optimizer,
              'optimizer' : optimizer,
              'loss'      : loss,
              'layca'     : layca,
              
              'trainable_parts'       : trainable_parts,
              'save_training_weights' : save_training_weights,
              'save_weights_name'     : save_weights_name,
              
              'shuffle'     : shuffle,
              'debug'       : debug,
              'print_train_infos' : print_train_infos,
              'train_on_tpu'    : train_on_tpu,
              
              'with_early_stopping'   : with_early_stopping,
              'patience_stop'     : patience_stop,
              'with_checkpoint'   : with_checkpoint,
              'with_tensorboard'  : with_tensorboard,
              'with_reduce_lr'    : with_reduce_lr,
              'reduce_factor'     : reduce_factor,
              'patience_reduce'   : patience_reduce,
              'monitor'   : monitor,
              
              'workers'   : workers,
              'max_queue_size'    : max_queue_size,
              'train_indices' : [int(i) for i in data_train],
              'valid_indices' : [int(i) for i in data_valid]
        }
        if print_train_infos:
            print("\n========== Training config ==========\n")
            print_dict(train_config)

        if save_config:
            with open(self.folder + "/train_config.json", "w") as fichier:
                fichier.write(json.dumps(train_config))
        
        data_train = data_df.iloc[data_train]
        data_valid = data_df.iloc[data_valid]
        
        ################################
        ########## Generators ##########
        ################################
                
        train_generator = self.get_generator(data_train, batch_size, shuffle)

        valid_generator = self.get_generator(data_valid, batch_size, shuffle)
            
        if print_train_infos:
            print("\n========== Generators ==========\n")
            print("Train generator :\n{}\n".format(train_generator))
            print("Valid generator :\n{}\n".format(valid_generator))
            #t0 = time.time()
            #for i in range(len(train_generator)):
            #    batch = train_generator.__getitem__(i)
            #print("Time to compute 1 epoch of data : {}".format(time_to_string(time.time() - t0)))
        
        ###############################
        ########## Callbacks ##########
        ###############################
        
        callbacks = get_callbacks(self.folder,
                                  save_weights_name = save_weights_name,
                                  with_early_stopping   = with_early_stopping,
                                  patience_stop     = patience_stop,
                                  with_checkpoint   = with_checkpoint,
                                  with_tensorboard  = with_tensorboard,
                                  with_reduce_lr    = with_reduce_lr,
                                  reduce_factor     = reduce_factor,
                                  patience_reduce   = patience_reduce,
                                  monitor   = monitor)
        


        #################################################
        ########## Trainable model preparation ##########
        #################################################
        clipvalue = 1. if self.hparams.tacotron_clip_gradients else None
        if with_original_optimizer and not layca:
            from keras.optimizers import SGD, Adam, RMSprop
            if optimizer == 'adam': optimizer = Adam(lr=learning_rate, clipvalue=clipvalue)
            elif optimizer == 'rmsprop': optimizer = RMSprop(lr=learning_rate, clipvalue=clipvalue)
            elif optimizer == 'sgd': optimizer = SGD(lr=learning_rate, clipvalue=clipvalue)
        else:
            from layer_rotation_control import SGD, Adam, RMSprop
            if optimizer == 'adam': optimizer = Adam(lr=learning_rate, layca=layca, clipvalue=clipvalue)
            elif optimizer == 'rmsprop': optimizer = RMSprop(lr=learning_rate, layca=layca, clipvalue=clipvalue)
            elif optimizer == 'sgd': optimizer = SGD(lr=learning_rate, layca=layca, clipvalue=clipvalue)
                
        loss = self._build_loss(loss) if type(loss) is str else self._build_losses(loss)
        metrics = self._build_metrics(metrics)
                
        print("Defining trainable model...")
        
        trainable_model = self._build_trainable_model()
        trainable_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        trainable_model.summary(250)
        
        if train_on_tpu and self.batch_size is not None:
            print("L'entrainement sur TPU n'est pas encore autorisé !")
            train_model = trainable_model
            #TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
            #batch_size = 8 * self.batch_size
            #train_model = tf.contrib.tpu.keras_to_tpu_model(self.model,
            #    strategy=tf.contrib.tpu.TPUDistributionStrategy(
            #        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
        else:
            train_model = trainable_model
                
        print("Prepairing {}...".format(training_type))        
        
        t = time.time()
        try:
            if training_type == 'custom_training':
                hist = self._custom_train(train_model,
                                          train_generator   = train_generator,
                                          valid_generator   = valid_generator,
                                          summary_step      = summary_step,
                                          prediction_step   = prediction_step,
                                          evaluation_step   = evaluation_step,
                                          steps             = steps,
                                          batch_size        = batch_size,
                                          callbacks         = callbacks,
                                          loss              = loss,
                                          optimizer         = optimizer,
                                          metrics           = metrics,
                                          shuffle           = shuffle,
                                          debug             = debug,
                                          workers           = workers,
                                         )
            elif training_type == "classic_training":
                hist = self._train(train_model,
                                   train_generator  = train_generator,
                                   valid_generator  = valid_generator,
                                   train_times      = train_times,
                                   valid_times      = valid_times,
                                   epochs           = epochs,
                                   batch_size       = batch_size,
                                   callbacks        = callbacks,
                                   shuffle          = shuffle,
                                   debug            = debug,
                                   workers          = workers,
                                   max_queue_size    = max_queue_size
                                  )
        except KeyboardInterrupt as e:
            print("Training stop after catching exception : {}".format(e))
            return None
        finally:
            train_model.save(os.path.join(self.folder, "full_model-{}.h5".format(steps)))
            self.encoder_model = train_model.layers[1]
            self.decoder_model = train_model.layers[-1]
        
        print("Temps total d'entrainement : {} ({} epochs)".format(time_to_string(time.time()-t), epochs))
        
        return hist
    
    def get_generator(self, data_df, batch_size=1, shuffle=True):
        config_generator = {
            'HPARAMS'       : self.hparams,
            'BATCH_SIZE'    : batch_size,
            
            'MAX_LEN_PHRASE': self.max_len_phrase,
            
            'OUTPUTS_PER_STEP' : self.outputs_per_step,
            'MAX_AUDIO_TIME': self.max_audio_time,
            'AUDIO_RATE'    : self.audio_rate,
            'FFT_LEN'       : self.n_fft,
            'NUM_FREQ'      : self.num_freq,
            'NUM_MELS'      : self.num_mels,
            'WIN_SIZE'      : self.win_size,
            'HOP_SIZE'      : self.hop_size
        }
        
        return BatchGenerator(data_df, 
                                self.vocab, 
                                config_generator, 
                                self.get_audio_data,
                                self.preprocess_phrase,
                                shuffle)
    
    def _train(self,
               train_model,
               train_generator,
               valid_generator,
              
               train_times,
               valid_times,
              
               epochs,
               batch_size,
              
               callbacks,
                            
               shuffle   = True,
               debug     = True,
               workers   = 3,
               max_queue_size    = 8,
              ):
        callbacks.append(PredictionCallback(self, 
                                            train_generator, 
                                            os.path.join(self.folder, "training-logs-test"), 
                                            len(train_generator) * train_times))
        return train_model.fit_generator(generator        = train_generator, 
                                         steps_per_epoch  = len(train_generator) * train_times, 
                                         epochs           = epochs, 
                                         verbose          = 1 if debug else 2,
                                         validation_data  = valid_generator,
                                         validation_steps = len(valid_generator) * valid_times,
                                         callbacks        = callbacks, 
                                         workers          = workers,
                                         max_queue_size   = max_queue_size,
                                         shuffle          = shuffle)
        
    def _custom_train(self,
                      train_model,
                      train_generator,
                      valid_generator,
                
                      steps,
                      batch_size,
                      callbacks,
                      
                      loss,
                      optimizer,
                      metrics,
                      
                      summary_step       = 500,
                      prediction_step    = 500,
                      evaluation_step    = 1000,
                            
                      shuffle   = True,
                      debug     = True,
                      workers   = 3,
                     ):
        #callback_list = CallbackList(callbacks=callbacks)
        #callback_list.set_model(train_model)
        
        def save_checkpoint(log_dir, checkpoint_dir, step, epoch, history, eval_history, eval_steps):
            print("\nWriting checkpoint at step {}".format(step))                    
            model_path = os.path.join(log_dir, checkpoint_dir, "training_model-{}.h5".format(step))
            
            logs = {"step":step, "loss":history["loss"].mean()}
            #model_checkpoint.on_epoch_end(epoch, logs=logs)
            
            history_dict = {}
            for k, v in history.items():
                history_dict[k] = [float(n) for n in v.get_data()]
                eval_history["val_" + k] = [float(n) for n in eval_history["val_" + k]]
            
            infos = {
                "step"  : step,
                "epoch" : epoch,
                "model_path"    : model_path,
                "history"       : history_dict
            }
            if len(eval_steps) > 0:
                infos["eval_history"] = eval_history
                infos["eval_steps"] = eval_steps
            #data_infos = [step, epoch, model_path, history_dict, eval_history, eval_steps]
            #summary_df = pd.DataFrame(data_infos, index=summary_df_index)
            #summary_df.to_csv(os.path.join(log_dir, "checkpoint.csv"))
            with open(os.path.join(log_dir, "checkpoint.json"), 'w') as fichier:
                json.dump(infos, fichier, indent=4)
            
        def load_checkpoint(log_dir):
            checkpoint_infos = os.path.join(log_dir, "checkpoint.json")
            
            #infos = pd.read_csv(checkpoint_infos, index_col=0, header=0)
            with open(checkpoint_infos, 'r') as fichier:
                infos = json.loads(fichier.read())
            step = infos["step"]
            epoch = infos["epoch"]
            model_path = infos["model_path"]
            history = infos["history"]
            if "eval_history" in infos and 'eval_steps' in infos:
                eval_history = infos["eval_history"]
                eval_steps = infos["eval_steps"]
            else:
                eval_history = {}
                for k in history.keys():
                    eval_history["val_"+k] = []
                eval_steps = []
            
            #self._load_weights(model_path, train_model)
            #train_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            
            for k, v in history.items():
                history[k] = DataWindow(data=v)
                
            return step, epoch, history, eval_history, eval_steps
                        
        def make_prediction(x, y, mel_dir, linear_dir, plot_dir, wav_dir, step):
            _, target_mel, target_linear, _ = y
            _, mel, linear, _, alignment = self._predict_with_target(*x)
            mel = mel[0]
            linear = linear[0]
            alignment = alignment[0]
            
            mel_filename    = os.path.join(mel_dir, "step-{}.npy".format(step))
            linear_filename = os.path.join(linear_dir, "step-{}.npy".format(step))
            plot_mel_filename = os.path.join(plot_dir, "mel_step-{}.png".format(step))
            plot_linear_filename = os.path.join(plot_dir, "linear_step-{}.png".format(step))
            plot_align_filename = os.path.join(plot_dir, "align_step-{}.png".format(step))
            wav_mel_filename = os.path.join(wav_dir, "wav_from_mel_step-{}.wav".format(step))
            wav_linear_filename = os.path.join(wav_dir, "wav_from_linear_step-{}.wav".format(step))

            wav_from_linear = self.wav_from_linear(linear)
            wav_from_mel = self.wav_from_mel(mel)
            
            np.save(mel_filename, mel)
            np.save(linear_filename, linear)

            audio.save_wav(wav_from_mel, wav_mel_filename, self.audio_rate)
            audio.save_wav(wav_from_linear, wav_linear_filename, self.audio_rate)

            plot_alignment(alignment, title="Alignments", filename=plot_align_filename, show=False, fontsize=14)
            plot_spectrogram(mel, title="Mel spectrogram", filename=plot_mel_filename, show=False, target_spectrogram=target_mel[0])
            plot_spectrogram(linear, title="Linear spectrogram", filename=plot_linear_filename, show=False, target_spectrogram=target_linear[0])
            
            
        def get_epoch_log(history):
            logs = {}
            for metric, data in history.items():
                logs[metric] = data.mean()
            return logs
        
        """ Creating directories """
        
        log_dir = os.path.join(self.folder, "training-logs")
        eval_dir = os.path.join(log_dir, "evals")
        checkpoint_dir = "checkpoints"
        mel_dir = "mels"
        linear_dir = "linear"
        plot_dir = "plots"
        wav_dir = "wavs"
        
        os.makedirs(os.path.join(log_dir, eval_dir), exist_ok=True)
        os.makedirs(os.path.join(log_dir, wav_dir), exist_ok=True)
        os.makedirs(os.path.join(log_dir, checkpoint_dir), exist_ok=True)
        os.makedirs(os.path.join(log_dir, mel_dir), exist_ok=True)
        os.makedirs(os.path.join(log_dir, linear_dir), exist_ok=True)
        os.makedirs(os.path.join(log_dir, plot_dir), exist_ok=True)
        
        os.makedirs(os.path.join(eval_dir, wav_dir), exist_ok=True)
        os.makedirs(os.path.join(eval_dir, mel_dir), exist_ok=True)
        os.makedirs(os.path.join(eval_dir, linear_dir), exist_ok=True)
        os.makedirs(os.path.join(eval_dir, plot_dir), exist_ok=True)
        
        
        step = 0
        epoch = 0
        
        metrics_names = ["loss", "mel_before_loss", "melafter__loss", "linear_loss", "stop_token_loss", "mel_before_acc", "mel_after_acc", "linear_acc", "stop_acc"]
        
        threads = []
        history = {}
        history_eval = {}
        steps_eval = []
        if os.path.exists(os.path.join(log_dir, "checkpoint.json")):
            step, epoch, history, history_eval, steps_eval = load_checkpoint(log_dir)
            print("Continue training from last checkpoint (step {})".format(step))
        else:
            for k in metrics_names: 
                history[k] = DataWindow()
                history_eval["val_{}".format(k)] = []
            save_checkpoint(log_dir, checkpoint_dir, step, epoch, history, history_eval, steps_eval)
            
        print("Training begin for {} steps".format(steps))
        
        while step < steps:
            """ Threads preparation for this epoch """
            threads = []
            for i in range(len(train_generator)):
                threads.append(ParallelGenerator(train_generator.__getitem__, i))
            for i in range(workers): threads[i].start()
                
            callback_list.on_epoch_begin(epoch, get_epoch_log(history))
                
            for i in range(len(train_generator)):
                """ Training step """
                t0 = time.time()
                
                x, y = threads[i].get_result()
                if i+workers < len(threads): threads[i+workers].start()
                loss_values = train_model.train_on_batch(x, y)
                threads[i].clear()
                t1 = time.time()
                
                """ Metric description """
                
                step += 1
                metrics_des = ""
                for metric_name, value in zip(metrics_names, loss_values):
                    history[metric_name].append(value)
                    if "loss" == metric_name:
                        metrics_des += " loss = {:.4f} Avg loss = {:.3f}".format(value, history[metric_name].mean())
                    elif debug:
                        metrics_des += " Avg {} = {:.3f}".format(metric_name, history[metric_name].mean())
                print("Step : {}\t[time : {:.2f}s{}]".format(step, t1-t0, metrics_des))
                
                
                if step % summary_step == 0:
                    save_checkpoint(log_dir, checkpoint_dir, step, epoch, history, history_eval, steps_eval)

                
                if step % prediction_step == 0:
                    print("\nMaking prediction at step {}".format(step))
                    num = int(random.random() * len(train_generator))
                    [p_in, mel_in], y = train_generator.__getitem__(num)
                    p_in = p_in[:1]
                    mel_in = mel_in[:1]
                    make_prediction([p_in, mel_in], y, os.path.join(eval_dir, mel_dir),
                                   os.path.join(eval_dir, linear_dir), os.path.join(eval_dir, plot_dir), os.path.join(eval_dir, wav_dir), step)
                
                if step % evaluation_step == 0:
                    print("\nEvaluation at step {}".format(step))
                    steps_eval.append(step)
                    eval_metrics = {}
                    for key in history_eval.keys(): eval_metrics[key] = []
                    for i in tqdm(range(len(valid_generator))):
                        x, y = valid_generator.__getitem__(i)
                        loss_values = train_model.evaluate(x, y, batch_size=batch_size, verbose=0)
                        
                        for metric_name, value in zip(metrics_names, loss_values):
                            eval_metrics["val_" + metric_name].append(value)
                        
                    for metric_name, value in eval_metrics.items():
                        moyenne = np.mean(np.array(value))
                        history_eval[metric_name].append(moyenne)
                        if "val_loss" == metric_name:
                            metrics_des += " val_loss = {:.4f}".format(moyenne)
                        elif debug:
                            metrics_des += " {} = {:.3f}".format(metric_name, moyenne)
                    print("Evaluation \t[{}]".format(metrics_des))        
                    
                    [p_in, mel_in], y = valid_generator.__getitem__(num)
                    p_in = p_in[:1]
                    mel_in = mel_in[:1]
                    make_prediction([p_in, mel_in], y, os.path.join(log_dir, mel_dir),
                                   os.path.join(log_dir, linear_dir), os.path.join(log_dir, plot_dir), os.path.join(log_dir, wav_dir), step)

                
                if step >= steps: break

            callback_list.on_epoch_end(epoch, get_epoch_log(history))
            epoch += 1
            print("\nEpoch {} finished".format(epoch))
            train_generator.on_epoch_end()
        
        save_checkpoint(log_dir, checkpoint_dir, step, epoch, history, history_eval, steps_eval)
        train_model.save(os.path.join(log_dir, "final_model.h5"))
        self.encoder_model = train_model.layers[1]
        self.decoder_model = train_model.layers[3]
        return history        
        
    def predict_with_target(self, data_df=None, out_dir="outputs", generator=None, batch_size=8):
        batch_size = min(len(data_df), batch_size)
        data_df = data_df.sort_values(by='mel_frames').reset_index(drop=True)
        mel_dir = os.path.join(out_dir, 'mels')
        linear_dir = os.path.join(out_dir, 'linear')
        plot_dir = os.path.join(out_dir, 'plots')
        wav_dir = os.path.join(out_dir, 'wavs')
        
        os.makedirs(mel_dir, exist_ok=True)
        os.makedirs(linear_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(wav_dir, exist_ok=True)
        
        outputs = []
        
        if generator is None:
            data_generator = self.get_generator(data_df, batch_size=batch_size, shuffle=False)
        else:
            data_generator = generator
                                    
        
        n = 0
        for i in tqdm(range(len(data_generator))):
            [phrases, mel_in], [_, mel_targets, linear_targets, stop_token] = data_generator.__getitem__(i)
            
            _, mel_predicts, linear_predicts, _, alignments = self._predict_with_target(phrases, mel_in)
            
            for b in range(batch_size):
                p = data_df.at[n, 'text']
                mel = mel_predicts[b]
                linear = linear_predicts[b]
                alignment = alignments[b]
                
                mel_filename    = os.path.join(mel_dir, "pred_{}.npy".format(n))
                linear_filename = os.path.join(linear_dir, "pred_{}.npy".format(n))
                plot_mel_filename = os.path.join(plot_dir, "mel_spectrogram_{}.png".format(n))
                plot_linear_filename = os.path.join(plot_dir, "linear_spectrogram_{}.png".format(n))
                plot_align_filename = os.path.join(plot_dir, "alignments_{}.png".format(n))
                wav_mel_filename = os.path.join(wav_dir, "wav_from_mel_{}.wav".format(n))
                wav_linear_filename = os.path.join(wav_dir, "wav_from_linear_{}.wav".format(n))

                wav_from_linear = self.wav_from_linear(linear)
                wav_from_mel = self.wav_from_mel(mel)
            
                np.save(mel_filename, mel)
                np.save(linear_filename, linear)

                audio.save_wav(wav_from_mel, wav_mel_filename, self.audio_rate)
                audio.save_wav(wav_from_linear, wav_linear_filename, self.audio_rate)

                plot_alignment(alignment, title="Alignments for :\n{}".format(p), filename=plot_align_filename, show=False, fontsize=14)
                plot_spectrogram(mel, title="Mel spectrogram", filename=plot_mel_filename, show=False, target_spectrogram=mel_targets[b])
                plot_spectrogram(linear, title="Linear spectrogram", filename=plot_linear_filename, show=False, target_spectrogram=linear_targets[b])
                
                outputs.append((p, mel, linear, alignment))
                n += 1
        return outputs
        
    def _predict_with_target(self, phrase_input, mel_input):
        encoder_out = self.encoder_model.predict(phrase_input)
        return self.decoder_model.predict([encoder_out, mel_input])        
        
    def _predict(self, phrase, min_iter=5, max_iter=1000):
        if min_iter >= max_iter:
            raise ValueError("min_iter must be smaller than max_iter !")
        processed_phrase = np.array([self.preprocess_phrase(phrase)])

        encoded_phrase = self.encoder_model.predict(processed_phrase)
        
        last_mel_outputs = np.zeros((1, self.outputs_per_step, self.num_mels))
        stop = False
        n_iter = 0
        while not stop:
            n_iter += 1
            
            mel_input = np.reshape(last_mel_outputs, (1, -1, self.num_mels * self.outputs_per_step))
            
            pred = self.decoder_model.predict([encoded_phrase, mel_input])
            
            _, mel, linear, stop_token, weights = pred
            
            last_mel = mel[:,-self.outputs_per_step:]
            
            last_mel_outputs = np.append(last_mel_outputs, last_mel, axis=1)
            
            if (stop_token[0][-1] > 0.5 or n_iter >= max_iter) and n_iter > min_iter:
                mel_prediction = mel[0]
                linear_prediction = linear[0]
                alignments = np.transpose(weights[0])
                stop = True
            
        return [mel_prediction, linear_prediction, alignments]
    
    def predict(self, liste_phrases, out_dir, min_iter=5, max_iter=100000):
        mel_dir = os.path.join(out_dir, 'mels')
        linear_dir = os.path.join(out_dir, 'linear')
        plot_dir = os.path.join(out_dir, 'plots')
        wav_dir = os.path.join(out_dir, 'wavs')
        
        os.makedirs(mel_dir, exist_ok=True)
        os.makedirs(linear_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(wav_dir, exist_ok=True)
        
        outputs = []
        
        for i, phrase in tqdm(enumerate(liste_phrases)):
            mel, linear, alignment = self._predict(phrase, min_iter=min_iter, max_iter=max_iter)
            
            mel_filename    = os.path.join(mel_dir, "pred_{}.npy".format(i))
            linear_filename = os.path.join(linear_dir, "pred_{}.npy".format(i))
            plot_mel_filename = os.path.join(plot_dir, "mel_spectrogram_{}.png".format(i))
            plot_linear_filename = os.path.join(plot_dir, "linear_spectrogram_{}.png".format(i))
            plot_align_filename = os.path.join(plot_dir, "alignments_{}.png".format(i))
            wav_mel_filename = os.path.join(wav_dir, "wav_from_mel_{}.wav".format(i))
            wav_linear_filename = os.path.join(wav_dir, "wav_from_linear_{}.wav".format(i))
            
            wav_from_linear = self.wav_from_linear(linear)
            wav_from_mel = self.wav_from_mel(mel)
            
            np.save(mel_filename, mel)
            np.save(linear_filename, linear)
            
            audio.save_wav(wav_from_mel, wav_mel_filename, self.audio_rate)
            audio.save_wav(wav_from_linear, wav_linear_filename, self.audio_rate)
            
            plot_alignment(alignment, title="Alignments for :\n{}".format(phrase), filename=plot_align_filename, show=False, fontsize=16)
            plot_spectrogram(mel, title="Mel spectrogram", filename=plot_mel_filename, show=False)
            plot_spectrogram(linear, title="Linear spectrogram", filename=plot_linear_filename, show=False)
            
            outputs.append((mel, linear, alignment))
            
        return outputs
        
        
    
    def evaluate(self, data_df):
        data = data.reset_index(drop=True)
        total = len(data)
        total_fourrier = 0
        total_mel = 0
        
        correct_time     = 0
        correct_fourrier = 0
        correct_audio    = 0
        
        err_stop        = []
        err_fourrier    = []
        err_audio       = []
        
        for i in range(total):
            encoded_phrase = string_to_code(data.at[i, 'text'], self.vocab)
            if self.backend_model == 'encoder_decoder_base':
                encoded_phrase = to_categorical(encoded_phrase, num_classes=self.vocab_size)
            encoded_phrase = np.array([encoded_phrase])

            state_h, state_c, time = self.encoder_model.predict(encoded_phrase)
            
            time = time[0,0]
            
            err_time_i = abs(time - data.at[i, 'duree'])
            if err_time_i < max_err_time:
                correct_time += 1
                
            err_time.append(err_time_i)
                
            true_fourrier = self.preprocess_audio(data.at[i, 'filename'])

            last_pred = np.zeros((1, 1, self.fft_pred_len))

            state = [state_h, state_c]

            states = [state_h, state_c]

            if with_initial_states and units == decoder_units:
                states = []
                for n in range(nb_decoder):
                    states += state
            else:
                states += [np.zeros((1, decoder_units)) for _ in range(nb_decoder*2)]

            if 'attention' in self.backend_model:
                if with_initial_states and units == decoder_units:
                    last_states = state
                else:
                    last_states = [np.zeros((1, units)) for _ in range(2)]
                states = state + states + last_states
            
            err_audio_i = 0
            
            for step in range(len(true_fourrier)):
                pred = self.decoder_model.predict([last_pred] + states)

                y_pred = pred[0]
                states = pred[1:]

                err_fourrier_i = np.sum(np.abs(true_fourrier[step] - y_pred))
                if err_fourrier_i < max_err_fourrier:
                    correct_fourrier += 1
                    
                err_fourrier.append(err_fourrier_i)
                err_audio_i += err_fourrier_i
                
                last_pred = np.array([np.array([true_fourrier[step]])])
            
            if err_audio_i < max_err_fourrier * len(true_fourrier):
                correct_audio += 1
            err_audio.append(err_audio_i)
            total_fourrier += len(true_fourrier)
            
        err_time        = np.array(err_time)
        err_fourrier    = np.array(err_fourrier)
        err_audio       = np.array(err_audio)
            
        index = ["Err totale", "Err moyenne", "Err max", "Nombre testé", "Nombre correct", "% correct"]
        col = ["Temps", "Audio", "Fourrier"]
        
        infos = [[np.sum(err_time), np.sum(err_audio), np.sum(err_fourrier)],
                 [np.mean(err_time), np.mean(err_audio), np.mean(err_fourrier)],
                 [np.max(err_time), np.max(err_audio), np.max(err_fourrier)],
                 [total, total, total_fourrier],
                 [correct_time, correct_audio, correct_fourrier],
                 ["{}%".format(arrondir(100 * correct_time / total, 2)), "{}%".format(arrondir(100 * correct_audio / total, 2)), "{}%".format(arrondir(100 * correct_fourrier / total_fourrier, 2))]
              ]
        infos = pd.DataFrame(infos, columns=col, index=index)
        return infos
    
    def get_audio_data(self, data_df):
        if self.use_custom_linear:
            filename = data_df['original_audio_path']
            son = Son.gen_from_wav(filename)
            son.change_rate(self.audio_rate)

            _, _, linear_spec = son._stft(nfft=self.n_fft, win_size=self.win_size, hop_size=self.hop_size)

            linear_spec = np.transpose(linear_spec)
            linear_spec = linear_spec.astype(np.float32)
            linear_spec = linear_spec / np.max(np.abs(linear_spec))
            linear_spec *= 2
            slinear_specpec = np.clip(linear_spec, -1, 1)
            linear_spec[np.where(np.abs(linear_spec) < 0.01)] = 0.
            linear_spec *= self.max_magnitude
        else:
            linear_spec = np.load(data_df['linear_path'])
        
        mel_spec = np.load(data_df['mel_path'])
        
        return mel_spec, linear_spec
    
    def wav_from_linear(self, S):
        S = np.transpose(S)
        if self.use_custom_linear:
            son = build_son_from_stft(S, self.hparams)
        else:
            son = audio.inv_linear_spectrogram(S, self.hparams)
        return son
    
    def wav_from_mel(self, S):
        S = np.transpose(S)
        return audio.inv_mel_spectrogram(S, self.hparams)
    
    def preprocess_phrase(self, phrase):
        if type(phrase) == str:
            encoded_phrase = string_to_code(phrase, self.vocab)
        else:
            encoded_phrase = phrase

        if not self.pad_phrase:
            return encoded_phrase
        p = np.zeros((self.max_len_phrase,))
        longueur = len(phrase)
        p[1:longueur+1] = encoded_phrase
        p[0] = 1
        p[longueur+1] = 2
        return p
        
    def summary(self, taille=100):
        print("\n========== Encoder part ==========\n")
        self.encoder_model.summary(taille)
        print("\n========== Decoder part ==========\n")
        self.decoder_model.summary(taille)
        
    def save_img_model(self, filename=None, show_shapes=True, show_layer_names=True, show=True, **plot_kwargs):
        if filename is None:
            filename = self.folder + "/image_model.png"
        plot_model(self.encoder_model, to_file="encoder_"+filename, show_shapes=show_shapes, show_layer_names=show_layer_names)
        plot_model(self.encoder_model, to_file="decoder_"+filename, show_shapes=show_shapes, show_layer_names=show_layer_names)
        if show:
            img = cv2.imread("encoder_"+filename, -1)
            plot(img[:,:,::-1], titre="Encoder architecture", type_graph="img", **plot_kwargs)
            img = cv2.imread("decoder_"+filename, -1)
            plot(img[:,:,::-1], titre="Decoder architecture", type_graph="img", **plot_kwargs)
        
    def plot_history(self, history, plot_val=None):
        if plot_val is None: plot_val = history.history.keys()
        acc = {}
        loss = {}
        for key in plot_val:
            if 'acc' in key:
                acc[key] = history.history[key]
            elif 'loss' in key:
                loss[key] = history.history[key]
        plot_multiple(acc, titre="Accuracy over epochs", xlabel="apochs", ylabel="accuracy")
        plot_multiple(loss, titre="Loss over epochs", xlabel="epochs", ylabel="loss")
        
    def save(self, model_format="json"):
        self.model.save(os.path.join(self.folder, "full_model.h5"))
        #self.save_model(self .folder + "/encoder_model." + model_format, model=self.encoder_model)
        self._save_weights(self.folder + "/encoder_weights.h5", self.encoder_model)
        #self.save_model(self .folder + "/decoder_model." + model_format, model=self.decoder_model)
        self._save_weights(self.folder + "/decoder_weights.h5", self.decoder_model)
        
    def load(self, model_format="json"):
        self.encoder_model = self.load_model(self .folder + "/encoder_model." + model_format)
            
        self.decoder_model = self.load_model(self .folder + "/decoder_model." + model_format)
        self.load_all_weights()
                    
    def load_weights(self):
        full_model = self._build_trainable_model()
        self._load_weights(os.path.join(self.folder, "full_model.h5"), full_model)
        self.encoder_model = full_model.layers[1]
        self.decoder_model = full_model.layers[-1]
        
        #self._load_weights(self.folder + "/encoder_weights.h5", model=self.encoder_model)
            
        #self._load_weights(self.folder + "/decoder_weights.h5", model=self.decoder_model)
        
    def save_model(self, filename, model=None):
        if model is None: 
            print("model is None !")
            return
        with open(filename, "w") as fichier:
            if ".json" in filename:
                fichier.write(model.to_json())
            elif ".yaml" in filename:
                fichier.write(model.to_yaml())
        
        
    def load_model(self, filename):
        model = None
        try:
            with open(filename, "r") as fichier:
                if ".json" in filename:
                    model = model_from_json(fichier.read())
                elif ".yaml" in filename:
                    model = model_from_yaml(fichier.read())
        except:
            print("Il n'y a pas de fichier de modèle !")
        return model
        
    def _save_weights(self, filename, model=None):
        if model is None: 
            print("model is None !")
            return
        print("Saving weights to {}".format(filename))
        model.save(filename, overwrite=True)
        
    def _load_weights(self, filename, model=None):
        if model is None: 
            print("model is None !")
            return
        try:
            model.load_weights(filename)
            print("Weights loaded from {}".format(filename))
        except:
            print("Le chargement des poids a échoué !")
        self.is_compile = False
