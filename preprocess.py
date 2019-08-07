import os
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from multiprocessing import cpu_count

import preprocessor
from hparams import hparams
from utils import time_to_string


def preprocess(args, input_folders, out_dir, hparams):
    mel_dir = os.path.join(out_dir, 'mels')
    wav_dir = os.path.join(out_dir, 'audio')
    linear_dir = os.path.join(out_dir, 'linear')
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(linear_dir, exist_ok=True)

    if 'siwis' in input_folders.lower():
        metadata = preprocessor.build_from_siwis_path(hparams, input_folders, mel_dir,  linear_dir, wav_dir, args.n_jobs, tqdm=tqdm, multi_processing=False)
    else:
        metadata = preprocessor.build_from_path(hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
    write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
    df_columns = ["original_audio_path", "processed_audio_path", "mel_path", "linear_path", "timesteps", "mel_frames", "audio_time", "text", "text_length"]
    metadata_df = pd.DataFrame(metadata, columns=df_columns)
    metadata_df.to_csv(os.path.join(out_dir, 'train.csv'))

    mel_frames = np.sum(metadata_df['mel_frames'].values)
    timesteps = np.sum(metadata_df['timesteps'].values)
    sr = hparams.sample_rate
    hours = time_to_string(np.sum(metadata_df['audio_time'].values))
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({})'.format(
		len(metadata_df), mel_frames, timesteps, hours))
    print('Max input length (text chars): {}'.format(np.max(metadata_df['text_length'].values)))
    print('Max mel frames length: {}'.format(np.max(metadata_df['mel_frames'].values)))
    print('Max audio time : {}'.format(np.max(metadata_df['audio_time'].values)))

def norm_data(args):
    merge_books = (args.merge_books=='True')

    print('Selecting data folders..')
    
    supported_datasets = ['LJSpeech-1.0', 'LJSpeech-1.1', 'M-AILABS', 'SIWIS']
    
    if args.dataset not in supported_datasets:
        raise ValueError('dataset value entered {} does not belong to supported datasets: {}'.format(args.dataset, supported_datasets))
        
    if args.dataset.startswith('LJSpeech'):
        return [os.path.join(args.base_dir, args.dataset)]

    if 'siwis' in args.dataset.lower():
        return args.base_dir
    
    if args.dataset == 'M-AILABS':
        supported_languages = ['en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU','uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
        if args.language not in supported_languages:
            raise ValueError('Please enter a supported language to use from M-AILABS dataset! \n{}'.format(supported_languages))
            
        supported_voices = ['female', 'male', 'mix']
        if args.voice not in supported_voices:
            raise ValueError('Please enter a supported voice option to use from M-AILABS dataset! \n{}'.format(supported_voices))
            
        path = os.path.join(args.base_dir, args.language, 'by_book', args.voice)
        
        supported_readers = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
        if args.reader not in supported_readers:
            raise ValueError('Please enter a valid reader for your language and voice settings! \n{}'.format(supported_readers))
            
        path = os.path.join(path, args.reader)
        supported_books = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
        if merge_books:
            return [os.path.join(path, book) for book in supported_books]
        else:
            if args.book not in supported_books:
                raise ValueError('Please enter a valid book for your reader settings! \n{}'.format(supported_books))
                
            return [os.path.join(path, args.book)]


def run_preprocess(args, hparams):
    input_folders = norm_data(args)
    output_folder = os.path.join(args.base_dir, args.output)

    preprocess(args, input_folders, output_folder, hparams)


def main():
    print('initializing preprocessing...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--dataset', default='SIWIS')
    parser.add_argument('--language', default='en_US')
    parser.add_argument('--voice', default='female')
    parser.add_argument('--reader', default='mary_ann')
    parser.add_argument('--merge_books', default='False')
    parser.add_argument('--book', default='northandsouth')
    parser.add_argument('--output', default='training_data')
    parser.add_argument('--n_jobs', type=int, default=cpu_count())
    args = parser.parse_args()

    modified_hp = hparams.parse(args.hparams)
    
    assert args.merge_books in ('False', 'True')

    run_preprocess(args, modified_hp)


if __name__ == '__main__':    
    main()
