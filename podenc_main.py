import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd
from scipy.io import loadmat

from podenc_read_datum import read_datum
from podenc_utils import encoding_regression, load_header


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-value', type=str, default='all')
    parser.add_argument('--window-size', type=int, default=200)
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--shuffle', action='store_true', default=False)
    group1.add_argument('--phase-shuffle', action='store_true', default=False)
    parser.add_argument('--stim', type=str, default='Podcast')
    parser.add_argument('--pilot', type=str, default='')
    parser.add_argument('--lags', nargs='+', type=int)
    parser.add_argument('--outName', type=str, default='test')
    parser.add_argument('--nonWords', action='store_false', default=True)
    parser.add_argument('--datum-emb-fn',
                        type=str,
                        default='podcast-datum-glove-50d.csv')
    parser.add_argument('--gpt2', type=int, default=1)
    parser.add_argument('--bert', type=int, default=None)
    parser.add_argument('--bart', type=int, default=None)
    parser.add_argument('--glove', type=int, default=1)
    parser.add_argument('--electrodes', nargs='+', type=int)
    parser.add_argument('--npermutations', type=int, default=1)
    parser.add_argument('--min-word-freq', nargs='?', type=int, default=1)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sid', nargs='?', type=int, default=None)
    group.add_argument('--sig-elec-name', nargs='?', type=str, default=None)

    args = parser.parse_args()

    if args.sid and not args.electrodes:
        parser.error("--sid requires --electrodes")
    elif not args.sid and args.electrodes:
        parser.error("--electrodes requires --sid")

    return args


def setup_environ(args):
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
    elif 'tiger' not in hostname:
        tiger = 0
        PROJ_DIR = '/mnt/bucket/labs/hasson/ariel/247/'
        DATUM_DIR = os.path.join(PROJ_DIR, 'models/podcast-datums')
        CONV_DIR = os.path.join(
            PROJ_DIR, 'conversation_space/crude-conversations/Podcast')
        BRAIN_DIR_STR = 'preprocessed_all'
    else:
        tiger = 0
        PROJ_DIR = None
        print("Could not find PROJ_DIR. Please specify it here.")
        sys.exit()

    path_dict = dict(PROJ_DIR=PROJ_DIR,
                     DATUM_DIR=DATUM_DIR,
                     CONV_DIR=CONV_DIR,
                     BRAIN_DIR_STR=BRAIN_DIR_STR,
                     tiger=tiger)

    vars(args).update(path_dict)
    print(args)

    return args


def process_subjects(args):
    if args.sid and args.electrodes:
        sid = 'NY' + str(args.sid) + '_111_Part1_conversation1'
        brain_dir = os.path.join(args.CONV_DIR, sid, args.BRAIN_DIR_STR)

        filesb = glob.glob(os.path.join(brain_dir, '*.mat'))
        filesb = sorted(
            filesb, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

        electrode_list = args.electrodes
        labels = load_header(args.CONV_DIR, sid)

        # number of labels in header == number of electrode mat files
        assert len(filesb) <= len(labels)

        elecDir = ''.join([
            args.outName, '-', sid, '_160_200ms_', args.word_value, args.pilot,
            '/'
        ])
        elecDir = os.path.join(os.getcwd(), 'Results', elecDir)
        os.makedirs(elecDir, exist_ok=True)

        for electrode in electrode_list:
            elec_signal = loadmat(filesb[electrode])['p1st']
            name = labels[electrode]

            encoding_regression(args, sid, datum, elec_signal, name)

    return


def process_sig_electrodes(args):
    flag = 'prediction_presentation' if tiger else ''
    sig_elec_file = os.path.join(args.PROJ_DIR, flag, args.sig_elec_name)
    sig_elec_list = pd.read_csv(sig_elec_file, header=None)[0].tolist()

    for sig_elec in sig_elec_list:
        sid, elec_name = sig_elec[:29], sig_elec[30:]

        labels = load_header(args.CONV_DIR, sid)
        if not labels:
            print('Header Missing')
        electrode_num = labels.index(elec_name)

        brain_dir = os.path.join(args.CONV_DIR, sid, args.BRAIN_DIR_STR)
        electrode_file = os.path.join(
            brain_dir, ''.join([
                sid, '_electrode_preprocess_file_',
                str(electrode_num + 1), '.mat'
            ]))
        try:
            elec_signal = loadmat(electrode_file)['p1st']
        except FileNotFoundError:
            print(f'Missing: {electrode_file}')
            continue

        encoding_regression(args, sid, datum, elec_signal, elec_name)

    return


if __name__ == "__main__":
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)

    # Locate and read datum
    datum = read_datum(args)

    if args.sig_elec_name:
        process_sig_electrodes(args)
    else:
        process_subjects(args)

    end_time = datetime.now()

    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
