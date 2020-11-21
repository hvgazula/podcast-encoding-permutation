import argparse
import csv
import glob
import os
import statistics
import sys
from datetime import datetime

import mat73
import numpy as np
import pandas as pd
from scipy.io import loadmat

from podcast_encoding_permutation_utils import build_XY, encode_lags_numba1

np.random.seed(0)
start_time = datetime.now()
print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

hostname = os.environ['HOSTNAME']

if 'tiger' in hostname:
    PROJ_DIR = '/projects/HASSON/247/data/podcast'
    EMB_DIR = os.path.join(PROJ_DIR, os.pardir, 'embeddings')
    LOG_DIR = '/scratch/gpfs/hgazula/perm_test'
    tiger = 1
elif 'scotty' in hostname:
    PROJ_DIR = '/mnt/bucket/labs/hasson/ariel/247/'
    # LOG_DIR = os.path.join(PROJ_DIR, 'models/encoding/bobbi/perm_tests/')
    LOG_DIR = os.environ['HOME']
    EMB_DIR = os.path.join(PROJ_DIR, 'models/embeddings/')
    tiger = 0

parser = argparse.ArgumentParser()
parser.add_argument('--fs-clin', type=int, default=512)
parser.add_argument('--word-value', type=str, default='all')
parser.add_argument('--stim', type=str, default='Podcast')
parser.add_argument('--embeddings', type=str, default='gpt2xl-50d')
parser.add_argument('--pilot', type=str, default='')
parser.add_argument('--outName', type=str, default='no-numba-test-tiger')
parser.add_argument('--sig-elec-name', type=str, default=None)
parser.add_argument('--nonWords', action='store_false', default=True)
parser.add_argument(
    '--datum-emb-fn',
    type=str,
    default='podcast-datum-gpt2-xl-c_1024-previous-pca_50d.csv')
parser.add_argument('--sid', type=int, default=661)
parser.add_argument('--gpt2', type=int, default=None)
parser.add_argument('--bert', type=int, default=None)
parser.add_argument('--bart', type=int, default=None)
parser.add_argument('--glove', type=int, default=1)
parser.add_argument('--electrode', type=int, default=None)
args = parser.parse_args()

if not args.sid:
    print('Enter a valid subject ID')
else:
    sid = 'NY' + str(args.sid) + '_111_Part1_conversation1'

if args.sig_elec_name:
    sig_elec_file = os.path.join(PROJ_DIR, 'prediction_presentation',
                                 args.sig_elec_name)
    sig_elec = pd.read_csv(sig_elec_file, header=None)[0].tolist()
else:
    sig_elec = 0

if tiger:
    conv_dir = os.path.join(PROJ_DIR, str(args.sid))
else:
    conv_dir = os.path.join(PROJ_DIR,
                            'conversation_space/crude-conversations/Podcast',
                            str(args.sid), '/')

lags = np.arange(-2000, 2001, 100)

i = args.electrode

if isinstance(sig_elec, int):
    if tiger:
        conv_dir = os.path.join(PROJ_DIR, sid)
        brain_dir = os.path.join(conv_dir, 'preprocessed-highgamma')
    else:
        conv_dir = os.path.join(
            PROJ_DIR, 'conversation_space/crude-conversations/Podcast', sid)
        brain_dir = os.path.join(conv_dir, 'preprocessed_all')
    filesb = glob.glob(os.path.join(brain_dir, '*.mat'))
    # filesb = sorted(filesb,
    #                 key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    filesb = sorted(filesb)
else:
    sid = sig_elec[i][:29]
    if tiger:
        conv_dir = os.path.join(PROJ_DIR, sid)
        brain_dir = os.path.join(conv_dir, 'preprocessed-highgamma')
    else:
        conv_dir = os.path.join(
            PROJ_DIR, 'conversation_space/crude-conversations/Podcast', sid)
        brain_dir = os.path.join(conv_dir, 'preprocessed_all')

if i > len(filesb):
    sys.exit()

misc_dir = os.path.join(conv_dir, 'misc')
header = mat73.loadmat(os.path.join(misc_dir, sid + '_header.mat'))
labels = header.header.label

if sig_elec:
    name = filesb[i][31:]
else:
    electrode_num = int(os.path.splitext(filesb[i])[0].split('_')[-1]) - 1
    # subtracted 1 to align with matlab indexing
    name = labels[electrode_num]

print(i, name)
elecDir = ''.join([
    sid, '_', args.embeddings, '_160_200ms_', args.word_value, args.pilot, '_',
    args.outName, '/'
])
elecDir = os.path.join(LOG_DIR, elecDir)

if not os.path.exists(elecDir):
    os.makedirs(elecDir, exist_ok=True)

if not os.path.isfile(elecDir + name + '_perm.csv'):
    f = labels.index(name)

    # Load electrode signal
    if isinstance(sig_elec, int):
        elec_signal = loadmat(filesb[i])['p1st']
    else:
        elec_signal = loadmat(
            os.path.join(
                brain_dir,
                ''.join([sid, '_electrode_preprocess_file_',
                         str(f), '.mat'])))['p1st']

    # Locate and read datum
    if tiger:
        datumName = os.path.join(PROJ_DIR, args.datum_emb_fn)
    else:
        datumName = os.path.join(
            '/mnt/bucket/labs/hasson/ariel/247/models/podcast-datums/',
            args.datum_emb_fn)

    df = pd.read_csv(datumName, header=0)
    d2 = df

    # print(d2.shape)
    if args.nonWords:
        d2 = d2[d2.is_nonword == 0]
    if args.gpt2:
        d2 = d2[d2.in_gpt2 == 1]
    if args.bert:
        d2 = d2[d2.in_bert == 1]
    if args.bart:
        d2 = d2[d2.in_bart == 1]
    if args.glove:
        d2 = d2[d2.in_glove == 1]

    # d2 = d2[d2.in_roberta == 1]

    df_cols = d2.columns.tolist()
    embedding_columns = df_cols[df_cols.index('0'):]
    d2 = d2[~d2['word'].isin(['sp', '{lg}', '{ns}', '{inaudible}'])]
    d2 = d2.dropna()

    d2['embeddings'] = d2[embedding_columns].values.tolist()
    d2 = d2.drop(columns=embedding_columns)

    if args.word_value == 'bottom':
        d2 = d2.dropna(subset=['gpt2_xl_target_prob', 'human_target_prob'])
        denom = 3
        if args.pilot == 'GPT2':
            pred = d2.gpt2_xl_target_prob
        elif args.pilot == 'mturk':
            pred = d2.human_target_prob
        m = sorted(pred)
        med = statistics.median(m)
        datum = d2[
            pred <= m[np.ceil(len(m) / denom)],
            ['word', 'onset', 'offset', 'accuracy', 'speaker', 'embeddings']]
    elif args.word_value == 'all':
        datum = d2[[
            'word', 'onset', 'offset', 'accuracy', 'speaker', 'embeddings'
        ]]
    else:
        d2 = d2.dropna(subset=['gpt2_xl_target_prob', 'human_target_prob'])
        denom = 3
        if args.pilot == 'GPT2':
            pred = d2.gpt2_xl_target_prob
        elif args.pilot == 'mturk':
            pred = d2.human_target_prob
        m = sorted(pred)
        med = statistics.median(m)
        datum = d2[
            pred >= m[len(m) - np.ceil(len(m) / denom)],
            ['word', 'onset', 'offset', 'accuracy', 'speaker', 'embeddings']]

    X, Y = build_XY(datum, elec_signal, lags, 512)

    prod_X = X[datum.speaker == 'Speaker1', :]
    comp_X = X[datum.speaker == 'Speaker2', :]

    prod_Y = Y[datum.speaker == 'Speaker1', :]
    comp_Y = Y[datum.speaker == 'Speaker2', :]

    # run permutation
    if prod_X.shape[0]:
        permx = np.stack(
            [encode_lags_numba1(prod_X, prod_Y) for _ in range(2500)])
    else:
        print('Not encoding production due to lack of examples')

    if comp_X.shape[0]:
        permx = np.stack(
            [encode_lags_numba1(comp_X, comp_Y) for _ in range(1000)])
    else:
        print('Not encoding comprehension due to lack of examples')

    filename = ''.join([elecDir, name, '_perm.csv'])
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(permx)

end_time = datetime.now()
print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
