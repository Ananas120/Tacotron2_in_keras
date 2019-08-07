import os
import argparse
import numpy as np
import pandas as pd

from hparams import hparams
from tacotron_2 import Tacotron
from text_utils import *
from plot_utils import *

def get_data(folder):
    if not os.path.exists(folder):
        raise ValueError("The data folder does'nt exist, launch preprocessing to create data")
    data_df = pd.read_csv(os.path.join(folder, "train.csv"))
    mel_dir = "mels"
    linear_dir = "linear"
    audio_dir = "audio"
    for idx, row in data_df.iterrows():
        data_df.at[idx, 'mel_path'] = os.path.join(folder, mel_dir, row['mel_path'])
        data_df.at[idx, 'linear_path'] = os.path.join(folder, linear_dir, row['linear_path'])
        data_df.at[idx, 'processed_audio_path'] = os.path.join(folder, audio_dir, row['processed_audio_path'])
        
    vocab = get_vocab()
    return data_df, vocab

def create_model(args, hparams, vocab):
    return Tacotron(hparams = hparams,
                      vocab  = vocab,
                      max_audio_time = args.max_audio_time,
                      max_len_phrase = None,
                      nom   = args.name,
                      use_custom_linear = False,
                      create_new_model = args.new_model
                     )

def train(model, data_df, vocab, args, hparams):
    print("Prepairing training...")
    return model.train(data_df,
                       train_size  = args.train_size,
                       train_times = args.train_times,
                       valid_size  = args.valid_size,
                       valid_times = args.valid_times,
                
                       steps      = 500,
                       evaluation_step  = 500,
                       prediction_step  = 500,
                       summary_step     = 100,
                       batch_size       = args.batch_size,
                       learning_rate    = args.learning_rate,
                
                       pitch_scale      = args.pitch_scale,
                       no_pitch_scale   = args.no_pitch_scale,
                
                       optimizer   = args.optimizer,
                       loss        = args.loss,
                       layca       = args.layca,
                
                       save_config = True,
                       shuffle     = args.shuffle,
                       debug       = args.debug,
                
                       with_early_stopping   = False,
                       patience_stop  = 3,
                       with_checkpoint   = True,
                       with_tensorboard  = True,
                       with_reduce_lr    = True,
                       reduce_factor     = 0.5,
                       patience_reduce   = 1,
                       monitor   = 'val_loss',
                       
                       workers  = 3,
                       max_queue_size   = 8,
                       
                       training_type = "classic_training"
                      )
    
def predict(model, phrase, vocab, args, hparams):
    mel, linear, alignments = model.predict(phrase, out_dir="outputs", max_iter=10)[0]
    plot_alignment(alignments)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument("--train_size", type=float, default=0.01)
    parser.add_argument("--train_times", type=int, default=1)
    parser.add_argument("--valid_size", type=float, default=0.1)
    parser.add_argument("--valid_times", type=int, default=1)
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    parser.add_argument("--pitch_scale", type=float, default=1.5)
    parser.add_argument("--no_pitch_scale", type=float, default=0.75)
    
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--loss", default="custom")
    parser.add_argument("--layca", default=False, type=bool)
    
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument("--debug", default=True, type=bool)
    parser.add_argument("--max_audio_time", default=12., type=float)
    parser.add_argument("--name", default="Tacotron")
    parser.add_argument("--new_model", default=False, type=bool)
    
    args = parser.parse_args()

    modified_hp = hparams.parse(args.hparams)
    
    data_df, vocab = get_data(args.data_dir)
    
    model = create_model(args, modified_hp, vocab)
    
    model.summary()
    
        
    train(model, data_df[data_df['mel_frames'] < 500], vocab, args, modified_hp)
    print("Training achieved successfully after {} epochs !".format(args.epochs))
    model.predict_with_target(data_df.iloc[:4], out_dir="output_test")
    predict(model, phrase, vocab, args, modified_hp)
    
