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


python_dir_list = glob.glob(os.path.join(os.getcwd(), 'test-NY*'))
py_mean_corr = extract_correlations(python_dir_list)

matlab_dir_list = glob.glob(os.path.join(os.getcwd(), 'NY*'))
m_mean_corr = extract_correlations(matlab_dir_list)

fig, ax = plt.subplots()
lags = np.arange(-2000, 2001, 25)
ax.plot(lags, py_mean_corr, 'k', label='python')
ax.plot(lags, m_mean_corr, 'r', label='matlab')
ax.legend()
ax.set(xlabel='lag (s)', ylabel='correlation', title='Here it is')
ax.grid()

fig.savefig("comparison_final.png")
plt.show()
