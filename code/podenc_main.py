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
from utils import load_pickle, main_timer, write_config


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

    PICKLE_DIR = os.path.join(os.getcwd(), 'data', args.project_id,
                              str(args.sid), 'pickles')

    path_dict = dict(PROJ_DIR=PROJ_DIR,
                     DATUM_DIR=DATUM_DIR,
                     CONV_DIR=CONV_DIR,
                     BRAIN_DIR_STR=BRAIN_DIR_STR,
                     PICKLE_DIR=PICKLE_DIR,
                     tiger=tiger)

    args.electrode_file = '_'.join([str(args.sid), 'electrode_names.pkl'])

    vars(args).update(path_dict)

    return args


# def process_subjects(args):
#     """Run encoding on particular subject (requires specifying electrodes)
#     """
#     sid = 'NY' + str(args.sid) + '_111_Part1_conversation1'
#     brain_dir = os.path.join(args.CONV_DIR, sid, args.BRAIN_DIR_STR)

#     labels = load_header(args.CONV_DIR, sid)

#     # Load all mat files and sort them
#     all_files = glob.glob(os.path.join(brain_dir, '*.mat'))
#     all_files = sorted(
#         all_files, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

#     # number of labels in header == number of electrode mat files
#     assert len(all_files) <= len(labels)

#     # If no electrodes are specified use all
#     if not args.electrodes:
#         args.electrodes = [
#             int(os.path.splitext(file)[0].split('_')[-1]) for file in all_files
#         ]
#         select_files = all_files
#     else:
#         # if specified select corresponding files
#         elecs = [i for i in args.electrodes]

#         select_files = []
#         for elec in elecs:
#             flag = 0
#             for file in all_files:
#                 if '_' + str(elec) + '.mat' in file:
#                     select_files.append(file)
#                     flag = 1
#             if not flag:
#                 args.electrodes.remove(elec)

#     return select_files, labels


def process_subjects(args):
    """Run encoding on particular subject (requires specifying electrodes)
    """
    # trimmed_signal = trimmed_signal_dict['trimmed_signal']

    # if args.electrodes:
    #     indices = [electrode_ids.index(i) for i in args.electrodes]

    #     trimmed_signal = trimmed_signal[:, indices]
    #     electrode_names = [electrode_names[i] for i in indices]

    df = pd.DataFrame(
        load_pickle(os.path.join(args.PICKLE_DIR, args.electrode_file)))

    if args.electrodes:
        electrode_info = {
            key: next(
                iter(df.loc[(df.subject == str(args.sid)) &
                            (df.electrode_id == key), 'electrode_name']), None)
            for key in args.electrodes
        }

    # # Loop over each electrode
    # for elec_id, elec_name in electrode_info.items():

    #     if elec_name is None:
    #         print(f'Electrode ID {elec_id} does not exist')
    #         continue

    #     elec_signal = load_electrode_data(args, elec_id)
    #     # datum = load_processed_datum(args)

    #     encoding_regression(args, datum, elec_signal, elec_name)

    # # write_electrodes(args, electrode_names)

    return electrode_info


def dumdum1(iter_idx, args, datum, signal, name):

    seed = iter_idx + int(os.getenv("SLURM_ARRAY_TASK_ID", 0)) * 10000
    # seed = (1 + iter_idx) * int(np.random.randint(0, sys.maxsize) // 1e13)
    # seed = (1 + iter_idx) + 50000
    np.random.seed(seed)

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


# def this_is_where_you_perform_regression(args, select_files, labels, datum):

#     for file, electrode in zip(select_files, args.electrodes):
#         name = labels[electrode - 1]  # python indexing

#         if any([item in name for item in ['EKG', 'ECG', 'SG']]):
#             continue

#         elec_signal = loadmat(file)['p1st']
#         elec_signal = elec_signal.reshape(-1, 1)

#         # Perform encoding/regression
#         if args.phase_shuffle:
#             with Pool(16) as pool:
#                 corr = pool.map(
#                     partial(dumdum1,
#                             args=args,
#                             datum=datum,
#                             signal=elec_signal,
#                             name=name), range(args.npermutations))

#             prod_corr, comp_corr = map(list, zip(*corr))

#             write_output(args, prod_corr, name, 'prod')
#             write_output(args, comp_corr, name, 'comp')
#         else:
#             encoding_regression(args, datum, elec_signal, name)
#     return


def load_electrode_data(args, elec_id):
    '''Loads specific electrodes mat files
    '''
    if args.project_id == 'tfs':
        DATA_DIR = '/projects/HASSON/247/data/conversations-car'
        process_flag = 'preprocessed'
    elif args.project_id == 'podcast':
        DATA_DIR = '/projects/HASSON/247/data/podcast-data'
        process_flag = 'preprocessed_all'
    else:
        raise Exception('Invalid Project ID')

    convos = sorted(glob.glob(os.path.join(DATA_DIR, str(args.sid), '*')))

    all_signal = []
    for convo_id, convo in enumerate(convos, 1):

        if args.conversation_id != 0 and convo_id != args.conversation_id:
            continue

        file = glob.glob(
            os.path.join(convo, process_flag, '*_' + str(elec_id) + '.mat'))[0]

        mat_signal = loadmat(file)['p1st']
        mat_signal = mat_signal.reshape(-1, 1)

        # mat_signal = trim_signal(mat_signal)

        if mat_signal is None:
            continue
        all_signal.append(mat_signal)

    if args.project_id == 'tfs':
        elec_signal = np.vstack(all_signal)
    else:
        elec_signal = np.array(all_signal)
        elec_signal = np.squeeze(elec_signal, axis=0)

    return elec_signal


def this_is_where_you_perform_regression(args, electrode_info, datum):

    # Loop over each electrode
    for elec_id, elec_name in electrode_info.items():

        if elec_name is None:
            print(f'Electrode ID {elec_id} does not exist')
            continue

        elec_signal = load_electrode_data(args, elec_id)

        # Perform encoding/regression
        if args.phase_shuffle:
            if args.project_id == 'podcast':
                with Pool() as pool:
                    corr = pool.map(
                        partial(dumdum1,
                                args=args,
                                datum=datum,
                                signal=elec_signal,
                                name=elec_name), range(args.npermutations))
            else:
                corr = []
                for i in range(args.npermutations):
                    corr.append(dumdum1(i, args, datum, elec_signal,
                                        elec_name))

            prod_corr, comp_corr = map(list, zip(*corr))
            write_output(args, prod_corr, elec_name, 'prod')
            write_output(args, comp_corr, elec_name, 'comp')
        else:
            encoding_regression(args, datum, elec_signal, elec_name)

    return None


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
            elec_signal = elec_signal.reshape(-1, 1)
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


# @main_timer
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
        # select_files, labels = process_subjects(args)
        electrode_info = process_subjects(args)
        # this_is_where_you_perform_regression(args, select_files, labels, datum)
        this_is_where_you_perform_regression(args, electrode_info, datum)


if __name__ == "__main__":
    main()
