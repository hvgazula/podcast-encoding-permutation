import csv
import glob
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat
from statsmodels.stats import multitest


def plot_permutations(subject_id, person, pp):
    results_path = '/scratch/gpfs/hgazula/podcast-encoding/results/'

    if person == 'harsha':
        subject_path = os.path.join(results_path, '20210401-phase-shuffle',
                                    str(subject_id))
        subject_files = sorted(glob.glob(os.path.join(subject_path, '*.csv')))
    elif person == 'bobbi':
        subject_files = sorted(
            glob.glob(os.path.join(results_path, 'bobbi-matlab-perm',
                                   '*.csv')))
    else:
        print('invalid person')

    for file in subject_files:
        if person == 'harsha':
            elec_name = os.path.splitext(
                os.path.basename(file))[0].split('_comp')[0]
            print(elec_name)
        elif person == 'bobbi':
            pass
        else:
            print('invalid person')

        data = pd.read_csv(file, header=None)

        fig, ax = plt.subplots()
        plt.plot(data.T)
        ax.set(xlabel='lag (s)', ylabel='correlation', title=elec_name)

        pp.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        plt.plot(np.mean(data, axis=0))
        ax.set(xlabel='lag (s)',
               ylabel='correlation',
               title=elec_name + '(mean)')

        pp.savefig(fig)
        plt.close()


def return_electrode_files(subject_id, shuffle_flag='noshuffle'):
    parent_dir = '/scratch/gpfs/hgazula/podcast-encoding/results'

    if shuffle_flag == 'noshuffle':
        bobbi_661 = glob.glob(
            os.path.join(parent_dir, 'bobbi-matlab-no-perm',
                         'NY' + str(subject_id) + '*', '*'))
        harsha_661 = glob.glob(
            os.path.join(parent_dir, '20210401-no-shuffle', str(subject_id),
                         '*'))
    elif shuffle_flag == 'shuffle':
        bobbi_661 = glob.glob(
            os.path.join(parent_dir, 'bobbi-matlab-perm',
                         'NY' + str(subject_id) + '*', '*'))
        harsha_661 = glob.glob(
            os.path.join(parent_dir, '20210401-phase-shuffle', str(subject_id),
                         '*'))
    else:
        print('invalid shuffle flag')

    bobbi_661.sort()
    harsha_661.sort()

    bobbi_661 = [
        elec for elec in bobbi_661
        if not any([item in elec for item in ['EKG', 'ECG', 'SG']])
    ]
    harsha_661 = [
        elec for elec in harsha_661
        if not any([item in elec for item in ['EKG', 'ECG', 'SG']])
    ]

    if shuffle_flag == 'noshuffle':
        bobbi_661 = sorted(bobbi_661,
                           key=lambda x: os.path.splitext(os.path.basename(x))[
                               0].split('conversation1')[-1].split('.')[0])
        harsha_661 = sorted(harsha_661,
                            key=lambda x: os.path.splitext(os.path.basename(x))
                            [0].split('_comp')[0])
    elif shuffle_flag == 'shuffle':
        bobbi_661 = sorted(bobbi_661,
                           key=lambda x: os.path.splitext(os.path.basename(x))[
                               0].split('_perm')[0])
        harsha_661 = sorted(harsha_661,
                            key=lambda x: os.path.splitext(os.path.basename(x))
                            [0].split('_comp')[0])
    else:
        print('invalid shuffle flag')

    assert len(bobbi_661) == len(harsha_661)

    print(f'Returning {len(bobbi_661)} electrode files from Bobbi')
    print(f'Returning {len(bobbi_661)} electrode files from Harsha')

    return (bobbi_661, harsha_661)


def print_electrodes(bobbi_folder, harsha_folder):

    bobbi_list = []
    for bobbi in bobbi_folder:
        bobbi_elec = os.path.basename(bobbi)
        bobbi_elec_name = bobbi_elec.split('conversation1')[-1].split('.')[0]
        bobbi_list.append(bobbi_elec_name)
    print(bobbi_list)

    harsha_list = []
    for bobbi in harsha_folder:
        bobbi_elec = os.path.basename(bobbi)
        bobbi_elec_name = bobbi_elec.split('_comp')[0]
        harsha_list.append(bobbi_elec_name)
    print(harsha_list)


def plot_individual_electrodes(subject_id,
                               pp,
                               special_list=None,
                               shuffle_flag='noshuffle'):

    bobbi_folder, harsha_folder = return_electrode_files(
        subject_id, shuffle_flag)

    bobbi_fdr = pd.read_csv('/scratch/gpfs/hgazula/podcast-encoding/results/jupyter_bobbi.csv')
    harsha_fdr = pd.read_csv('/scratch/gpfs/hgazula/podcast-encoding/results/jupyter_harsha.csv')

    for bobbi, harsha in zip(bobbi_folder, harsha_folder):
        bobbi_elec = os.path.basename(bobbi)
        harsha_elec = os.path.basename(harsha)

        if shuffle_flag == 'noshuffle':
            bobbi_elec_name = bobbi_elec.split('conversation1')[-1].split(
                '.')[0]
        elif shuffle_flag == 'shuffle':
            bobbi_elec_name = bobbi_elec.split('_perm')[0]
        else:
            print('invalid shuffle flag')

        harsha_elec_name = harsha_elec.split('_comp')[0]

        # if not special_list or [subject, bobbi_elec_name] not in special_list:
        #     continue

        print(subject, bobbi_elec_name, harsha_elec_name)

        val1 = bobbi_fdr[(bobbi_fdr.electrode == bobbi_elec_name)
                         & (bobbi_fdr.subject == subject)]['pcor'].values[0]
        val2 = bobbi_fdr[(bobbi_fdr.electrode == bobbi_elec_name)
                         & (bobbi_fdr.subject == subject)]['score'].values[0]

        val3 = harsha_fdr[(harsha_fdr.electrode == bobbi_elec_name)
                          & (harsha_fdr.subject == subject)]['pcor'].values[0]
        val4 = harsha_fdr[(harsha_fdr.electrode == bobbi_elec_name)
                          & (harsha_fdr.subject == subject)]['score'].values[0]

        print(f'Bobbi (Pre): {val2}, Bobbi (Post): {val1}')
        print(f'Harsh (Pre): {val4}, Harsh (Post): {val3}')

        assert bobbi_elec_name == harsha_elec_name, "Bad Electrode Name"

        print(f'Reading {bobbi_elec_name}')
        if os.path.splitext(bobbi)[-1] == '.csv':
            bobbi = pd.read_csv(bobbi, header=None).values
        elif os.path.splitext(bobbi)[-1] == '.mat':
            bobbi = loadmat(bobbi)['rc'][0]
        else:
            print('invalid file format')

        if os.path.splitext(harsha)[-1] == '.csv':
            harsha = pd.read_csv(harsha, header=None).values
        elif os.path.splitext(harsha)[-1] == '.mat':
            harsha = loadmat(harsha)['rc'][0]
        else:
            print('invalid file format')

        if shuffle_flag == 'noshuffle':
            bobbi = bobbi.reshape(1, -1)
            harsha = harsha.reshape(1, -1)

            if bobbi.shape[1] == 801:
                bobbi_m = bobbi[:, 320:-320]

            if harsha.shape[1] == 801:
                harsha_m = harsha[:, 320:-320]
            else:
                harsha_m = harsha

        elif shuffle_flag == 'shuffle':
            bobbi = bobbi[:1000, :]
            harsha = harsha[:1000, :]

            bobbi_m = np.mean(bobbi, axis=0).reshape(1, -1)
            harsha_m = np.mean(harsha, axis=0).reshape(1, -1)
        else:
            print('invalid flag')

        assert bobbi_m.shape[1] == harsha_m.shape[
            1], "Mismatch: Number of Lags"

        if shuffle_flag == 'shuffle':
            title = bobbi_elec_name + ' (mean)'
        elif shuffle_flag == 'noshuffle':
            title = bobbi_elec_name
        else:
            print('invalid flag')

        fig, ax = plt.subplots()
        lags = np.arange(-2000, 2001, 25)
        ax.plot(lags, bobbi_m.reshape(-1, 1), 'k', label=f'bobbi')
        ax.plot(lags, harsha_m.reshape(-1, 1), 'r', label=f'harsha')
        ax.legend()
        ax.set(xlabel='lag (s)', ylabel='correlation', title=title)
        ax.grid()
        pp.savefig(fig)
        plt.close()

        if shuffle_flag == 'shuffle':
            fig = plt.figure()
            plt.title(bobbi_elec_name + ' correlation')

            ax1 = plt.subplot(211)
            lags = np.arange(-2000, 2001, 25)
            ax1.plot(bobbi.T, 'k', label=f'bobbi')
            ax1.set_xticklabels([])
            ax1.set(xlabel=f'Bobbi (Pre): {val2}, Bobbi (Post): {val1}')
            ax1.grid()

            ax2 = plt.subplot(212, sharex=ax1)
            lags = np.arange(-2000, 2001, 25)
            ax2.plot(harsha.T, 'r', label=f'harsha')
            ax2.set(xlabel=f'Harsha (Pre): {val4}, Harsha (Post): {val3}')
            ax2.grid()
            
            pp.savefig(fig)
            plt.close()


if __name__ == '__main__':
    not_in_harsha = pd.read_csv(
        '/scratch/gpfs/hgazula/podcast-encoding/code/new_not_in_harsha_by.csv'
    ).values.tolist()
    not_in_bobbi = pd.read_csv(
        '/scratch/gpfs/hgazula/podcast-encoding/code/new_not_in_bobbi_by.csv'
    ).values.tolist()

    for subject in [661, 662, 717, 723, 741, 742, 743, 763, 798]:
        pp = PdfPages(str(subject) + '_noshuffle.pdf')
        plot_individual_electrodes(subject, pp, None, 'noshuffle')
        pp.close()
