import glob
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

mainDir = '/mnt/bucket/labs/hasson/ariel/247/'
logdir = os.path.join(mainDir, 'models/encoding/log/50d_removeNonWords/')
convDir = os.path.join(mainDir, 'conversation_space/crude-conversations/Podcast/')

logdir_subdir = '_50d_160_200ms_all_removeNonWord_rng1'
convDir_subdir = 'encoding_files_50d_all_160_200ms_removedNonWords_alignedGPT2-XL-GLoVe_freq5'

lags = np.arange(-2000, 2001, 25)
names = []
pVals = []

# % Find patients

# %All Elec
subjects = glob.glob(os.path.join(logdir, '*' + logdir_subdir))
print(len(subjects))

for subject in subjects:
    sub = os.path.split(subject)[-1][:29]
    print(sub)

    logdir_sub = os.path.join(logdir, sub + logdir_subdir)
    print(logdir_sub)

    elec = glob.glob(os.path.join(convDir, sub, convDir_subdir, '*.mat'))
    print(os.path.join(convDir, sub, convDir_subdir, '*.mat'))
    print(len(elec))

    for x in elec:
        elecname = os.path.split(os.path.splitext(x)[0])[-1][29:]
        print(elecname)

        if elecname.startswith(('SG', 'ECGEKG', 'EEGSG')):
            continue

        logdir_csv = os.path.join(logdir_sub, elecname + '_perm.csv')
        print(logdir_csv)
    
        rc = loadmat(x)['rc']
        print(rc.shape)
        names.append('_'.join([sub, elecname]))

        if not os.path.exists(logdir_csv):
            print(f'Perm for {elecname} does not exist') 
        else:
            tops = pd.read_csv(logdir_csv, header=None).values
            print(tops.shape)

            if tops.shape[1] != len(lags):
                print('perm is wrong length')
            else:
                omaxs = np.max(tops, axis=1)

                ss = sum(np.max(rc) > Omaxs)
                s = ss/5000
                s = 1 - s
                pVals.append(s)
        break
    break