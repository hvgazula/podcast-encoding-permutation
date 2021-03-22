import csv
import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from podenc_parser import parse_arguments
from podenc_phase_shuffle import phase_randomize_1d
from podenc_read_datum import read_datum
from podenc_utils import (append_jobid_to_string, create_output_directory,
                          encoding_regression, encoding_regression_pr,
                          load_header)
from scipy.io import loadmat
from utils import main_timer, write_config


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
        elecs = [i for i in args.electrodes]

        select_files = []
        for elec in elecs:
            flag = 0
            for file in all_files:
                if '_' + str(elec) + '.mat' in file:
                    select_files.append(file)
                    flag = 1
            if not flag:
                args.electrodes.remove(elec)

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

        if any([item in name for item in ['EKG', 'ECG', 'SG']]):
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


def process_sig_electrodes(args, datum):
    """Run encoding on select significant elctrodes specified by a file
    """
    # Read in the significant electrodes
    sig_elec_file = os.path.join(
        os.path.join(os.getcwd(), 'code', args.sig_elec_file))
    sig_elec_list = pd.read_csv(sig_elec_file)

    # Loop over each electrode
    for subject, elec_name in sig_elec_list.itertuples(index=False):

        if isinstance(subject, int):
            subject_id = glob.glob(
                os.path.join(args.CONV_DIR, 'NY' + str(subject) + '*'))[0]
            subject_id = os.path.basename(subject_id)

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

        if args.phase_shuffle:
            with Pool(16) as pool:
                corr = pool.map(
                    partial(dumdum1,
                            args=args,
                            datum=datum,
                            signal=elec_signal,
                            name=elec_name), range(args.npermutations))

            prod_corr, comp_corr = map(list, zip(*corr))

            write_output(args, prod_corr, elec_name, 'prod')
            write_output(args, comp_corr, elec_name, 'comp')
        else:
            # Perform encoding/regression
            encoding_regression(args, datum, elec_signal,
                                str(subject) + '_' + elec_name)

    return


@main_timer
def main():
    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)

    # Saving configuration to output directory
    write_config(vars(args))

    # Locate and read datum
    datum = read_datum(args)

    # Processing significant electrodes or individual subjects
    if args.sig_elec_file:
        process_sig_electrodes(args, datum)
    else:
        select_files, labels = process_subjects(args)
        this_is_where_you_perform_regression(args, select_files, labels, datum)


if __name__ == "__main__":
    main()
