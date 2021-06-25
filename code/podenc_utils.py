import csv
import os
from functools import partial
from multiprocessing import Pool

import mat73
import numpy as np
from numba import jit, prange
from podenc_phase_shuffle import phase_randomize
from scipy import stats
from sklearn.model_selection import KFold


def encColCorr(CA, CB):
    """[summary]

    Args:
        CA ([type]): [description]
        CB ([type]): [description]

    Returns:
        [type]: [description]
    """
    df = np.shape(CA)[0] - 2

    CA -= np.mean(CA, axis=0)
    CB -= np.mean(CB, axis=0)

    r = np.sum(CA * CB, 0) / np.sqrt(np.sum(CA * CA, 0) * np.sum(CB * CB, 0))

    t = r / np.sqrt((1 - np.square(r)) / df)
    p = stats.t.sf(t, df)

    r = r.squeeze()

    if r.size > 1:
        r = r.tolist()
    else:
        r = float(r)

    return r, p, t


def cv_lm_003(X, Y, kfolds):
    """Cross-validated predictions from a regression model using sequential
        block partitions with nuisance regressors included in the training
        folds

    Args:
        X ([type]): [description]
        Y ([type]): [description]
        kfolds ([type]): [description]

    Returns:
        [type]: [description]
    """
    skf = KFold(n_splits=kfolds, shuffle=False)

    # Data size
    nSamps = X.shape[0]
    nChans = Y.shape[1] if Y.shape[1:] else 1

    # Extract only test folds
    folds = [t[1] for t in skf.split(np.arange(nSamps))]

    YHAT = np.zeros((nSamps, nChans))
    # Go through each fold, and split
    for i in range(kfolds):
        # Shift the number of folds for this iteration
        # [0 1 2 3 4] -> [1 2 3 4 0] -> [2 3 4 0 1]
        #                       ^ dev fold
        #                         ^ test fold
        #                 | - | <- train folds
        folds_ixs = np.roll(range(kfolds), i)
        test_fold = folds_ixs[-1]
        train_folds = folds_ixs[:-1]
        # print(f'\nFold {i}. Training on {train_folds}, '
        #       f'test on {test_fold}.')

        test_index = folds[test_fold]
        # print(test_index)
        train_index = np.concatenate([folds[j] for j in train_folds])

        # Extract each set out of the big matricies
        Xtra, Xtes = X[train_index], X[test_index]
        Ytra, Ytes = Y[train_index], Y[test_index]

        # Mean-center
        Xtra -= np.mean(Xtra, axis=0)
        Xtes -= np.mean(Xtes, axis=0)
        Ytra -= np.mean(Ytra, axis=0)
        Ytes -= np.mean(Ytes, axis=0)

        # Fit model
        B = np.linalg.pinv(Xtra) @ Ytra

        # Predict
        foldYhat = Xtes @ B

        # Add to data matrices
        YHAT[test_index, :] = foldYhat.reshape(-1, nChans)

    return YHAT


@jit(nopython=True)
def fit_model(X, y):
    """Calculate weight vector using normal form of regression.

    Returns:
        [type]: (X'X)^-1 * (X'y)
    """
    beta = np.linalg.solve(X.T.dot(X), X.T.dot(y))
    return beta


@jit(nopython=True)
def build_Y(onsets, brain_signal, lags, window_size):
    """[summary]

    Args:
        onsets ([type]): [description]
        brain_signal ([type]): [description]
        lags ([type]): [description]
        window_size ([type]): [description]

    Returns:
        [type]: [description]
    """

    half_window = round((window_size / 1000) * 512 / 2)
    t = len(brain_signal)

    Y1 = np.zeros((len(onsets), len(lags), 2 * half_window + 1))

    for lag in prange(len(lags)):
        lag_amount = int(lags[lag] / 1000 * 512)

        index_onsets = np.minimum(
            t - half_window - 1,
            np.maximum(half_window + 1,
                       np.round_(onsets, 0, onsets) + lag_amount))

        # subtracting 1 from starts to account for 0-indexing
        starts = index_onsets - half_window - 1
        stops = index_onsets + half_window

        for i, (start, stop) in enumerate(zip(starts, stops)):
            Y1[i, lag, :] = brain_signal[start:stop].reshape(-1)

    return Y1


def build_XY(args, datum, brain_signal):
    """[summary]

    Args:
        args ([type]): [description]
        datum ([type]): [description]
        brain_signal ([type]): [description]

    Returns:
        [type]: [description]
    """
    X = np.stack(datum.embeddings)

    onsets = datum.onset.values

    lags = np.array(args.lags)
    brain_signal = brain_signal.reshape(-1, 1)

    Y = build_Y(onsets, brain_signal, lags, args.window_size)

    return X, Y


def encode_lags_numba(args, X, Y):
    """[summary]

    Args:
        X ([type]): [description]
        Y ([type]): [description]

    Returns:
        [type]: [description]
    """
    if args.shuffle:
        np.random.shuffle(Y)

    Y = np.mean(Y, axis=-1)

    PY_hat = cv_lm_003(X, Y, 10)
    rp, _, _ = encColCorr(Y, PY_hat)

    return rp


def encode_lags_numba_pr(args, X, Y):
    """[summary]

    Args:
        X ([type]): [description]
        Y ([type]): [description]

    Returns:
        [type]: [description]
    """
    if args.shuffle:
        np.random.shuffle(Y)

    Y = np.mean(Y, axis=-1)

    PY_hat = cv_lm_003(X, Y, 10)
    rp, _, _ = encColCorr(Y, PY_hat)

    return rp


def encoding_mp(i, args, prod_X, prod_Y):
    np.random.seed(i)
    perm_rc = encode_lags_numba(args, prod_X, prod_Y)
    return perm_rc


def run_save_permutation_pr(args, prod_X, prod_Y, filename):
    """[summary]

    Args:
        args ([type]): [description]
        prod_X ([type]): [description]
        prod_Y ([type]): [description]
        filename ([type]): [description]
    """
    if prod_X.shape[0]:
        perm_rc = encode_lags_numba(args, prod_X, prod_Y)
    else:
        perm_rc = None

    return perm_rc


def run_save_permutation(args, prod_X, prod_Y, filename):
    """[summary]

    Args:
        args ([type]): [description]
        prod_X ([type]): [description]
        prod_Y ([type]): [description]
        filename ([type]): [description]
    """
    if prod_X.shape[0]:
        with Pool(16) as pool:
            perm_prod = pool.map(
                partial(encoding_mp, args=args, prod_X=prod_X, prod_Y=prod_Y),
                range(args.npermutations))

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(perm_prod)


def load_header(conversation_dir, subject_id):
    """[summary]

    Args:
        conversation_dir ([type]): [description]
        subject_id (string): Subject ID

    Returns:
        list: labels
    """
    misc_dir = os.path.join(conversation_dir, subject_id, 'misc')
    header_file = os.path.join(misc_dir, subject_id + '_header.mat')
    if not os.path.exists(header_file):
        return
    header = mat73.loadmat(header_file)
    labels = header.header.label

    return labels


def create_output_directory(args):
    folder_name = '-'.join([args.output_prefix, str(args.sid)])
    folder_name = folder_name.strip('-')
    if not args.replication:
        full_output_dir = os.path.join(os.getcwd(), 'results',
                                       args.output_parent_dir, folder_name)
    else:
        full_output_dir = os.path.join(os.getcwd(), 'results',
                                       args.output_parent_dir, folder_name,
                                       'fold' + str(args.fold_idx))
    os.makedirs(full_output_dir, exist_ok=True)
    return full_output_dir


def encoding_regression_pr(args, datum, elec_signal, name):
    """[summary]

    Args:
        args (Namespace): Command-line inputs and other configuration
        sid (str): Subject ID
        datum (DataFrame): ['word', 'onset', 'offset', 'speaker', 'accuracy']
        elec_signal (numpy.ndarray): of shape (num_samples, 1)
        name (str): electrode name
    """
    # Build design matrices
    X, Y = build_XY(args, datum, elec_signal)

    # Split into production and comprehension
    prod_X = X[datum.speaker == 'Speaker1', :]
    comp_X = X[datum.speaker != 'Speaker1', :]

    prod_Y = Y[datum.speaker == 'Speaker1', :]
    comp_Y = Y[datum.speaker != 'Speaker1', :]

    # Run permutation and save results
    prod_corr = run_save_permutation_pr(args, prod_X, prod_Y, None)
    comp_corr = run_save_permutation_pr(args, comp_X, comp_Y, None)

    return (prod_corr, comp_corr)


def encoding_regression(args, datum, elec_signal, name):
    """[summary]

    Args:
        args (Namespace): Command-line inputs and other configuration
        sid (str): Subject ID
        datum (DataFrame): ['word', 'onset', 'offset', 'speaker', 'accuracy']
        elec_signal (numpy.ndarray): of shape (num_samples, 1)
        name (str): electrode name
    """
    # Build design matrices
    X, Y = build_XY(args, datum, elec_signal)

    # Split into production and comprehension
    prod_X = X[datum.speaker == 'Speaker1', :]
    comp_X = X[datum.speaker != 'Speaker1', :]

    prod_Y = Y[datum.speaker == 'Speaker1', :]
    comp_Y = Y[datum.speaker != 'Speaker1', :]

    print(f'{args.sid} {name} Prod: {len(prod_X)} Comp: {len(comp_X)}')

    output_dir = create_output_directory(args)

    # Run permutation and save results
    trial_str = append_jobid_to_string(args, 'prod')
    filename = os.path.join(output_dir, name + trial_str + '.csv')
    run_save_permutation(args, prod_X, prod_Y, filename)

    trial_str = append_jobid_to_string(args, 'comp')
    filename = os.path.join(output_dir, name + trial_str + '.csv')
    run_save_permutation(args, comp_X, comp_Y, filename)

    return


def append_jobid_to_string(args, speech_str):
    speech_str = '_' + speech_str

    if args.job_id:
        trial_str = '_'.join([speech_str, f'{args.job_id:02d}'])
    else:
        trial_str = speech_str

    return trial_str
