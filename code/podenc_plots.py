import glob
import os

import matplotlib.pyplot as plt
import numpy as np

# TODO: this file is work in progress


def extract_correlations(directory_list):
    all_corrs = []
    for dir in directory_list:
        file_list = glob.glob(os.path.join(dir, '*.csv'))
        for file in file_list:
            with open(file, 'r') as csv_file:
                ha = list(map(float, csv_file.readline().strip().split(',')))
            all_corrs.append(ha)

    hat = np.stack(all_corrs)
    mean_corr = np.mean(hat, axis=0)
    return mean_corr


# tfs-glove50-pca50d-pressure

# sub_list = [661, 662, 717, 723, 741, 742, 743, 763, 798, 'sig-elec']
# # sub_list = ['sig-elec']
# # python_dir_list = glob.glob(os.path.join(os.getcwd(), 'test-NY*'))
# for subject in sub_list:
#     print('/scratch/gpfs/hgazula/podcast-encoding/results/' + str(subject) +
#           '-test/*')

python_dir_list1 = glob.glob(
    '/scratch/gpfs/hgazula/podcast-encoding/results/20210526-0747-podcast-full--glove50/*'
)
python_dir_list2 = glob.glob(
    '/scratch/gpfs/hgazula/podcast-encoding/results/20210526--test-glove/*')
# python_dir_list3 = glob.glob(
#     '/scratch/gpfs/hgazula/podcast-encoding/results/' + str(subject) +
#     '-test-tfs/*')

py_mean_corr1 = extract_correlations(python_dir_list1)
py_mean_corr2 = extract_correlations(python_dir_list2)
# py_mean_corr3 = extract_correlations(python_dir_list3)

# matlab_dir_list = glob.glob(os.path.join(os.getcwd(), 'NY*'))
# m_mean_corr = extract_correlations(matlab_dir_list)

fig, ax = plt.subplots()
lags = np.arange(-2000, 2001, 25)
ax.plot(lags, py_mean_corr1, 'k', label='tfs-enc')
ax.plot(lags, py_mean_corr2, 'r', label='pod-enc')
# ax.plot(lags, py_mean_corr3, 'b', label='tfs')
ax.legend()
ax.set(xlabel='lag (s)', ylabel='correlation', title='Here it is')
ax.grid()

fig.savefig("20210526-glove-encoding-podcast-vs-tfs-normalization.png")
plt.show()
