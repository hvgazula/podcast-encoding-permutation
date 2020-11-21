import numpy as np

from scipy import signal, stats
from sklearn.model_selection import KFold


def encColCorr(CA, CB):

    assert CA.shape == CB.shape
    df = np.shape(CA)[0] - 2

    CA = signal.detrend(CA, axis=0, type='constant')
    CB = signal.detrend(CB, axis=0, type='constant')

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
    """
    skf = KFold(n_splits=kfolds, shuffle=False)

    # Data size
    nSamps = X.shape[0]
    try:
        nChans = Y.shape[1]
    except E:
        nChans = 1

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
        Xtra = signal.detrend(Xtra, axis=0, type='constant')
        Xtes = signal.detrend(Xtes, axis=0, type='constant')
        Ytra = signal.detrend(Ytra, axis=0, type='constant')
        Ytes = signal.detrend(Ytes, axis=0, type='constant')

        # Fit model
        B = fit_model(Xtra, Ytra)

        # Predict
        foldYhat = Xtes @ B

        # Add to data matrices
        YHAT[test_index, :] = foldYhat.reshape(-1, nChans)

    return YHAT


# @jit(nopython=True)
def fit_model(Xtra, Ytra):
    lamb = 1
    XtX_lamb = Xtra.T.dot(Xtra) + lamb * np.eye(Xtra.shape[1])
    XtY = Xtra.T.dot(Ytra)
    B = np.linalg.solve(XtX_lamb, XtY)
    return B


def encode_lags(datum, brain_signal, lags, fs_clin):

    half_window = round((100 / 1000) * fs_clin)  # time based on sampling rate

    onsets = np.array(datum.onset).astype(int)
    X = np.stack(datum.embeddings)

    t = len(brain_signal)
    rp_lags = []

    for lag in lags:
        lag_amount = int(lag / 1000 * fs_clin)
        index_onsets = np.minimum(
            t - half_window - 1,
            np.maximum(half_window + 1, onsets + lag_amount))
        starts = index_onsets - half_window
        stops = index_onsets + half_window + 1

        cols = [range(*i) for i in zip(starts, stops)]

        Y = np.array([np.mean(brain_signal[col_range]) for col_range in cols])

        PY_hat = cv_lm_003(X, Y, 10)
        rp, _, _ = encColCorr(Y.reshape(-1, 1), PY_hat)

        rp_lags.append(rp)

    return rp_lags


# def encode_lags_numba(datum, brain_signal, lags, fs_clin):

#     half_window = round((100 / 1000) * fs_clin)

#     onsets = np.array(datum.onset).astype(int)
#     X = np.stack(datum.embeddings)

#     t = len(brain_signal)
#     rp_lags = []

#     Y = np.zeros((len(onsets), len(lags)))

#     for lag in range(len(lags)):
#         lag_amount = int(lags[lag] / 1000 * fs_clin)
#         index_onsets = np.minimum(
#             t - half_window - 1,
#             np.maximum(half_window + 1, onsets + lag_amount))
#         starts = index_onsets - half_window
#         stops = index_onsets + half_window + 1

#         for i, (start, stop) in enumerate(zip(starts, stops)):
#             Y[i, lag] = np.mean(brain_signal[start:stop])

#         # PY_hat = cv_lm_003(X, Y, 10)
#         # rp, _, _ = encColCorr(Y.reshape(-1, 1), PY_hat)

#         # rp_lags.append(rp)
#     sys.exit()
#     return rp_lags


def build_XY(datum, brain_signal, lags, fs_clin):
    half_window = round((100 / 1000) * fs_clin)

    onsets = np.array(datum.onset).astype(int)
    X = np.stack(datum.embeddings)
    t = len(brain_signal)

    Y = inner_function(lags, t, fs_clin, half_window, onsets, brain_signal)
    return X, Y


def encode_lags_numba1(X, Y):
    np.random.shuffle(Y)
    PY_hat = cv_lm_003(X, Y, 10)
    rp, _, _ = encColCorr(Y, PY_hat)
    return rp


# def encode_lags_numba0(X, Y):
#     rp_lags = []
#     for column in Y.T:
#         np.random.shuffle(column)
#         PY_hat = cv_lm_003(X, column, 10)
#         rp, _, _ = encColCorr(column.reshape(-1, 1), PY_hat)
#         rp_lags.append(rp)
#     print(rp_lags)
#     return rp_lags

def inner_function(lags, t, fs_clin, half_window, onsets, brain_signal):
    Y = np.zeros((len(onsets), len(lags)))
    for lag in range(len(lags)):
        lag_amount = int(lags[lag] / 1000 * fs_clin)

        index_onsets = np.minimum(
            t - half_window - 1,
            np.maximum(half_window + 1, onsets + lag_amount))

        starts = index_onsets - half_window
        stops = index_onsets + half_window + 1

        for i, (start, stop) in enumerate(zip(starts, stops)):
            Y[i, lag] = np.mean(brain_signal[start:stop])

    return Y