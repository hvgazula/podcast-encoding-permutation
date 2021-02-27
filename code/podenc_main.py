import argparse
import csv
import glob
import os
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import numpy as np
from podenc_phase_shuffle import phase_randomize_1d
from podenc_read_datum import read_datum
from podenc_utils import (append_jobid_to_string, create_output_directory,
                          encoding_regression, encoding_regression_pr,
                          load_header)
from scipy.io import loadmat


def main_timer(func):
    def function_wrapper():
        start_time = datetime.now()
        print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

        func()

        end_time = datetime.now()
        print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')

    return function_wrapper


def parse_arguments():
    """Read commandline arguments

    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-value',
                        default='all',
                        choices=['all', 'top', 'bottom'])
    parser.add_argument('--window-size', type=int, default=200)
    parser.add_argument('--stim', type=str, default='Podcast')
    parser.add_argument('--pilot', type=str, default='')
    parser.add_argument('--lags', nargs='+', type=int)
    parser.add_argument('--output-prefix', type=str, default='test')
    parser.add_argument('--nonWords', action='store_true', default=False)
    parser.add_argument('--datum-emb-fn',
                        type=str,
                        default='podcast-datum-glove-50d.csv')
    parser.add_argument('--gpt2', type=int, default=1)
    parser.add_argument('--bert', type=int, default=None)
    parser.add_argument('--bart', type=int, default=None)
    parser.add_argument('--glove', type=int, default=1)
    parser.add_argument('--electrodes', nargs='*', type=int)
    parser.add_argument('--npermutations', type=int, default=1)
    parser.add_argument('--min-word-freq', nargs='?', type=int, default=1)
    parser.add_argument('--job-id', type=int, default=0)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sid', nargs='?', type=int, default=None)
    group.add_argument('--sig-elec-file', nargs='?', type=str, default=None)

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--shuffle', action='store_true', default=False)
    group1.add_argument('--phase-shuffle', action='store_true', default=False)

    args = parser.parse_args()

    if not args.sid and args.electrodes:
        parser.error("--electrodes requires --sid")

    if not args.shuffle and not args.phase_shuffle:
        args.npermutations = 1

    return args


def setup_environ(args):
    """Update args with project specific directories and other flags
    """
    hostname = os.environ['HOSTNAME']
    if any([item in hostname for item in ['tiger', 'della']]):
        tiger = 1
        PROJ_DIR = '/projects/HASSON/247/data/podcast'
        DATUM_DIR = PROJ_DIR
        CONV_DIR = PROJ_DIR
        BRAIN_DIR_STR = 'preprocessed_all'
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

    return args


def process_subjects(args):
    """Run encoding on particular subject (requires specifying electrodes)
    """
    sid = 'NY' + str(args.sid) + '_111_Part1_conversation1'
    brain_dir = os.path.join(args.CONV_DIR, sid, args.BRAIN_DIR_STR)

    labels = load_header(args.CONV_DIR, sid)

    # Load all mat files and sort them
    all_files = glob.glob(os.path.join(brain_dir, '*.mat'))
    all_files = sorted(
        all_files, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    # number of labels in header == number of electrode mat files
    assert len(all_files) <= len(labels)

    # If no electrodes are specified use all
    if not args.electrodes:
        args.electrodes = [
            int(os.path.splitext(file)[0].split('_')[-1]) for file in all_files
        ]
        select_files = all_files
    else:
        # if specified select corresponding files
        args.electrodes = [
            x for x in args.electrodes if 0 < x <= len(all_files)
        ]
        select_files = [
            file for file in all_files
            if any('_' + str(idx) + '.mat' in file for idx in args.electrodes)
        ]

    return select_files, labels


def dumdum1(iter_idx, args, datum, signal, name):
    np.random.seed(iter_idx)
    new_signal = phase_randomize_1d(signal)
    (prod_corr, comp_corr) = encoding_regression_pr(args, datum, new_signal,
                                                    name)

    return (prod_corr, comp_corr)


def write_output(args, output_mat, name, output_str):

    output_dir = create_output_directory(args)

    if all(output_mat):
        trial_str = append_jobid_to_string(args, output_str)
        filename = os.path.join(output_dir, name + trial_str + '.csv')
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(output_mat)


def this_is_where_you_perform_regression(args, select_files, labels, datum):

    for file, electrode in zip(select_files, args.electrodes):
        name = labels[electrode - 1]  # python indexing

        if 'EKG' in name:
            continue
        elec_signal = loadmat(file)['p1st']

        # Perform encoding/regression
        if args.phase_shuffle:
            with Pool(16) as pool:
                corr = pool.map(
                    partial(dumdum1,
                            args=args,
                            datum=datum,
                            signal=elec_signal,
                            name=name), range(args.npermutations))

            prod_corr, comp_corr = map(list, zip(*corr))

            write_output(args, prod_corr, name, 'prod')
            write_output(args, comp_corr, name, 'comp')
        else:
            encoding_regression(args, datum, elec_signal, name)
    return


@main_timer
def main():
    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)

    # Locate and read datum
    datum = read_datum(args)

    # Processing significant electrodes or individual subjects
    select_files, labels = process_subjects(args)
    this_is_where_you_perform_regression(args, select_files, labels, datum)


if __name__ == "__main__":
    main()
