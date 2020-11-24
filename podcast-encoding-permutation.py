import argparse
import glob
import os
from pathlib import Path
import sys
from datetime import datetime

import mat73
import pandas as pd
from scipy.io import loadmat

from podcast_encoding_permutation_utils import build_XY, run_save_permutation, load_header
from podcast_encoding_read_datum import read_datum

# TODO: Line 69 from matlab fiel ask about this
# filesb = dir([brain_dir '/*_14.mat']);
start_time = datetime.now()
print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

parser = argparse.ArgumentParser()
parser.add_argument('--word-value', type=str, default='all')
parser.add_argument('--stim', type=str, default='Podcast')
parser.add_argument('--embeddings', type=str, default='gpt2xl-50d')
parser.add_argument('--pilot', type=str, default='')
parser.add_argument('--lags', nargs='+', type=int)
parser.add_argument('--outName', type=str, default='no-numba-test-tiger')
parser.add_argument('--sig-elec-name', type=str, default=None)
parser.add_argument('--nonWords', action='store_false', default=True)
parser.add_argument(
    '--datum-emb-fn',
    type=str,
    default='podcast-datum-gpt2-xl-c_1024-previous-pca_50d.csv')
parser.add_argument('--sid', type=int, default=None)
parser.add_argument('--gpt2', type=int, default=None)
parser.add_argument('--bert', type=int, default=None)
parser.add_argument('--bart', type=int, default=None)
parser.add_argument('--glove', type=int, default=1)
parser.add_argument('--electrodes', type=int, default=None)
parser.add_argument('--npermutations', type=int, default=5000)
args = parser.parse_args()

hostname = os.environ['HOSTNAME']
if 'tiger' in hostname:
    PROJ_DIR = '/projects/HASSON/247/data/podcast'
    DATUM_DIR = PROJ_DIR
    CONV_DIR = PROJ_DIR
    BRAIN_DIR_STR = 'preprocessed'
    tiger = 1
elif 'scotty' in hostname:
    PROJ_DIR = '/mnt/bucket/labs/hasson/ariel/247/'
    DATUM_DIR = os.path.join(PROJ_DIR, 'models/podcast-datums')
    CONV_DIR = os.path.join(PROJ_DIR,
                            'conversation_space/crude-conversations/Podcast')
    BRAIN_DIR_STR = 'preprocessed_all'
    tiger = 0
else:
    PROJ_DIR = None
    print("Could not find PROJ_DIR. Please specify it here.")
    sys.exit()
    tiger = 0

if args.sid and not args.electrodes:
    print("Please enter atleast one electrode number")
    sys.exit()
elif not args.sid and args.electrodes:
    print('Enter a valid subject ID')
    sys.exit()
else:
    sid = 'NY' + str(args.sid) + '_111_Part1_conversation1'
    conv_dir = os.path.join(PROJ_DIR, str(args.sid))
    brain_dir = os.path.join(CONV_DIR, sid, BRAIN_DIR_STR)
    filesb = glob.glob(os.path.join(brain_dir, '*.mat'))
    filesb = sorted(filesb,
                    key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    electrode_list = args.electrodes
    # Path('/root/dir/sub/file.ext').stem

    labels = load_header(CONV_DIR, sid)

    elecDir = ''.join([
        sid, '_', args.embeddings, '_160_200ms_', args.word_value, args.pilot, '_',
        args.outName, '/'
    ])
    elecDir = os.path.join(os.getcwd(), elecDir)
    os.makedirs(elecDir, exist_ok=True)

    # Locate and read datum
    datum = read_datum(args, DATUM_DIR)

    for electrode in electrode_list:
        elec_signal = loadmat(filesb[electrode])['p1st']
        name = filesb[electrode][31:]

        # Build design matrices
        X, Y = build_XY(datum, elec_signal, args.lags, 512)

        # Split into production and comprehension
        prod_X = X[datum.speaker == 'Speaker1', :]
        comp_X = X[datum.speaker != 'Speaker1', :]

        prod_Y = Y[datum.speaker == 'Speaker1', :]
        comp_Y = Y[datum.speaker != 'Speaker1', :]

        # Run permutation and save results
        filename = ''.join([elecDir, name, '_prod.csv'])
        run_save_permutation(args, prod_X, prod_Y, filename)

        filename = ''.join([elecDir, name, '_comp.csv'])
        run_save_permutation(args, comp_X, comp_Y, filename)


if args.sig_elec_name:
    sig_elec_file = os.path.join(PROJ_DIR, 'prediction_presentation',
                                 args.sig_elec_name)
    sig_elec_list = pd.read_csv(sig_elec_file, header=None)[0].tolist()

    for sig_elec in sig_elec_list:
        sid = sig_elec[:29]
        brain_dir = os.path.join(CONV_DIR, sid, BRAIN_DIR_STR)

        labels = load_header(CONV_DIR, sid)
        electrode_num = int(
            os.path.splitext(filesb[electrode])[0].split('_')[-1]) - 1
        # subtracted 1 to align with matlab indexing
        name = labels[electrode_num]
        f = labels.index(name)

        elec_signal = loadmat(
                os.path.join(
                    brain_dir,
                    ''.join([sid, '_electrode_preprocess_file_',
                            str(f), '.mat'])))['p1st']
        elecDir = ''.join([
            sid, '_', args.embeddings, '_160_200ms_', args.word_value, args.pilot, '_',
            args.outName, '/'
        ])
        elecDir = os.path.join(os.getcwd(), elecDir)
        os.makedirs(elecDir, exist_ok=True)

        # Build design matrices
        X, Y = build_XY(datum, elec_signal, args.lags, 512)

        # Split into production and comprehension
        prod_X = X[datum.speaker == 'Speaker1', :]
        comp_X = X[datum.speaker != 'Speaker1', :]

        prod_Y = Y[datum.speaker == 'Speaker1', :]
        comp_Y = Y[datum.speaker != 'Speaker1', :]

        # Run permutation and save results
        filename = ''.join([elecDir, name, '_prod.csv'])
        run_save_permutation(args, prod_X, prod_Y, filename)

        filename = ''.join([elecDir, name, '_comp.csv'])
        run_save_permutation(args, comp_X, comp_Y, filename)

end_time = datetime.now()

print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
