import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.io import loadmat

from podcast_encoding_permutation_utils import encoding_regression, load_header
from podcast_encoding_read_datum import read_datum

start_time = datetime.now()
print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

parser = argparse.ArgumentParser()
parser.add_argument('--word-value', type=str, default='all')
parser.add_argument('--window-size', type=int, default=50)
parser.add_argument('--shuffle', action='store_true', default=False)
parser.add_argument('--stim', type=str, default='Podcast')
parser.add_argument('--embeddings', type=str, default='gpt2xl-50d')
parser.add_argument('--pilot', type=str, default='')
parser.add_argument('--lags', nargs='+', type=int)
parser.add_argument('--outName', type=str, default='test')
parser.add_argument('--sig-elec-name',
                    type=str,
                    default='5000-sig-elec-50d-onethresh-01.csv')
parser.add_argument('--nonWords', action='store_false', default=True)
parser.add_argument('--datum-emb-fn',
                    type=str,
                    default='podcast-datum-glove-50d.csv')
parser.add_argument('--sid', type=int, default=None)
parser.add_argument('--gpt2', type=int, default=1)
parser.add_argument('--bert', type=int, default=None)
parser.add_argument('--bart', type=int, default=None)
parser.add_argument('--glove', type=int, default=1)
parser.add_argument('--electrodes', nargs='+', type=int)
parser.add_argument('--npermutations', type=int, default=5000)
args = parser.parse_args()

print(args)

hostname = os.environ['HOSTNAME']
if 'tiger' in hostname:
    tiger = 1
    PROJ_DIR = '/projects/HASSON/247/data/podcast'
    DATUM_DIR = PROJ_DIR
    CONV_DIR = PROJ_DIR
    if args.sid in [661, 662, 717, 723]:
        BRAIN_DIR_STR = 'preprocessed'
    else:
        BRAIN_DIR_STR = 'preprocessed-ica'
elif 'scotty' in hostname:
    tiger = 0
    PROJ_DIR = '/mnt/bucket/labs/hasson/ariel/247/'
    DATUM_DIR = os.path.join(PROJ_DIR, 'models/podcast-datums')
    CONV_DIR = os.path.join(PROJ_DIR,
                            'conversation_space/crude-conversations/Podcast')
    BRAIN_DIR_STR = 'preprocessed_all'
else:
    tiger = 0
    PROJ_DIR = None
    print("Could not find PROJ_DIR. Please specify it here.")
    sys.exit()

# Locate and read datum
datum = read_datum(args, DATUM_DIR)

if args.sid and not args.electrodes:
    print("Please enter atleast one electrode number")
    sys.exit()
elif not args.sid and args.electrodes:
    print('Enter a valid subject ID')
    sys.exit()
elif args.sid and args.electrodes:
    sid = 'NY' + str(args.sid) + '_111_Part1_conversation1'
    conv_dir = os.path.join(PROJ_DIR, str(args.sid))
    brain_dir = os.path.join(CONV_DIR, sid, BRAIN_DIR_STR)

    filesb = glob.glob(os.path.join(brain_dir, '*.mat'))
    filesb = sorted(filesb,
                    key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    electrode_list = args.electrodes
    labels = load_header(CONV_DIR, sid)

    # number of labels in header == number of electrode mat files
    assert len(filesb) <= len(labels)

    elecDir = ''.join([
        args.outName, '-', sid, '_', args.embeddings, '_160_200ms_',
        args.word_value, args.pilot, '/'
    ])
    elecDir = os.path.join(os.getcwd(), elecDir)
    os.makedirs(elecDir, exist_ok=True)

    for electrode in electrode_list:
        elec_signal = loadmat(filesb[electrode])['p1st']
        name = labels[electrode]

        encoding_regression(args, sid, datum, elec_signal, name)

if args.sig_elec_name:
    sig_elec_file = os.path.join(PROJ_DIR, 'prediction_presentation',
                                 args.sig_elec_name)
    sig_elec_list = pd.read_csv(sig_elec_file, header=None)[0].tolist()

    for sig_elec in sig_elec_list:
        sid, elec_name = sig_elec[:29], sig_elec[30:]
        print(sid, elec_name)

        labels = load_header(CONV_DIR, sid)
        if not labels:
            print(f'Header Missing')
        electrode_num = labels.index(elec_name)

        brain_dir = os.path.join(CONV_DIR, sid, BRAIN_DIR_STR)
        electrode_file = os.path.join(
            brain_dir, ''.join([
                sid, '_electrode_preprocess_file_',
                str(electrode_num), '.mat'
            ]))
        try:
            elec_signal = loadmat(electrode_file)['p1st']
        except FileNotFoundError as e:
            print(f'Missing: {electrode_file}')
            continue

        encoding_regression(args, sid, datum, elec_signal, elec_name)

end_time = datetime.now()

print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
