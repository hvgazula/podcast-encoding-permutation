import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd
from scipy.io import loadmat

from podenc_read_datum import read_datum
from podenc_utils import (create_output_directory, encoding_regression,
                          load_header)


def parse_arguments():
    """Read commandline arguments

    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-value', type=str, default='all')
    parser.add_argument('--window-size', type=int, default=200)
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--shuffle', action='store_true', default=False)
    group1.add_argument('--phase-shuffle', action='store_true', default=False)
    parser.add_argument('--stim', type=str, default='Podcast')
    parser.add_argument('--pilot', type=str, default='')
    parser.add_argument('--lags', nargs='+', type=int)
    parser.add_argument('--output-prefix', type=str, default='test')
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
    """Update args with project specific directories and other flags
    """
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
    else:
        tiger = 0
        PROJ_DIR = '/mnt/bucket/labs/hasson/ariel/247/'
        DATUM_DIR = os.path.join(PROJ_DIR, 'models/podcast-datums')
        CONV_DIR = os.path.join(
            PROJ_DIR, 'conversation_space/crude-conversations/Podcast')
        BRAIN_DIR_STR = 'preprocessed_all'

    path_dict = dict(PROJ_DIR=PROJ_DIR,
                     DATUM_DIR=DATUM_DIR,
                     CONV_DIR=CONV_DIR,
                     BRAIN_DIR_STR=BRAIN_DIR_STR,
                     tiger=tiger)

    vars(args).update(path_dict)
    print(args)

    return args


def process_subjects(args, datum):
    """Run encoding on particular subject (requires specifying electrodes)
    
    TODO: Run of all available electrodes without having to specify
    """
    sid = 'NY' + str(args.sid) + '_111_Part1_conversation1'
    brain_dir = os.path.join(args.CONV_DIR, sid, args.BRAIN_DIR_STR)

    filesb = glob.glob(os.path.join(brain_dir, '*.mat'))
    filesb = sorted(filesb,
                    key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    labels = load_header(args.CONV_DIR, sid)

    # number of labels in header == number of electrode mat files
    assert len(filesb) <= len(labels)

    # Loop over each electrode
    for electrode in args.electrodes:
        elec_signal = loadmat(filesb[electrode])['p1st']
        name = labels[electrode]

        # Perform encoding/regression
        encoding_regression(args, sid, datum, elec_signal, name)

    return


def process_sig_electrodes(args, datum):
    """Run encoding on select significant elctrodes specified by a file 
    """
    flag = 'prediction_presentation' if not args.tiger else ''

    # Read in the significant electrodes
    sig_elec_file = os.path.join(args.PROJ_DIR, flag, args.sig_elec_name)
    sig_elec_list = pd.read_csv(sig_elec_file, header=None)[0].tolist()

    # Loop over each electrode
    for sig_elec in sig_elec_list:
        subject_id, elec_name = sig_elec[:29], sig_elec[30:]

        # Read subject's header
        labels = load_header(args.CONV_DIR, subject_id)
        if not labels:
            print('Header Missing')
        electrode_num = labels.index(elec_name)

        # Read electrode data
        brain_dir = os.path.join(args.CONV_DIR, subject_id, args.BRAIN_DIR_STR)
        electrode_file = os.path.join(
            brain_dir, ''.join([
                subject_id, '_electrode_preprocess_file_',
                str(electrode_num + 1), '.mat'
            ]))
        try:
            elec_signal = loadmat(electrode_file)['p1st']
        except FileNotFoundError:
            print(f'Missing: {electrode_file}')
            continue

        # Perform encoding/regression
        encoding_regression(args, subject_id, datum, elec_signal, elec_name)

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

    # Processing significant electrodes or individual subjects
    if args.sig_elec_name:
        process_sig_electrodes(args, datum)
    else:
        process_subjects(args, datum)

    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
