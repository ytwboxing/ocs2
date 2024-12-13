import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
import numpy as np
import pandas as pd
import sys

def read_data(file_path, skip_row_val):
	column_names = ['index',
					'x', 'vel', 'theta', 'theta-dot', 
                    'force',
					'not_useful'
                    ]
	data = pd.read_csv(file_path, header = None, names = column_names, skiprows=skip_row_val)

	return data 

#‘b’ blue 
#‘g’ green 
#‘r’ red 
#‘c’ cyan 
#‘m’ magenta 
#‘y’ yellow 
#‘k’ black 
#‘w’ white  
def func():
    # plt.figure("cartpole(ocs2)")
    # plt.title("cartpole(ocs2)")

    # plt.subplot(5, 1, 1)
    # plt.plot(dataset['index'].values, dataset['x'].values, '.g', label='x')
    # plt.legend()
    # plt.subplot(5, 1, 2)
    # plt.plot(dataset['index'].values, dataset['vel'].values, '.g', label='vel')
    # plt.legend()
    # plt.subplot(5, 1, 3)
    # plt.plot(dataset['index'].values, dataset['theta'].values, '.g', label='theta')
    # plt.legend()
    # plt.subplot(5, 1, 4)
    # plt.plot(dataset['index'].values, dataset['theta-dot'].values, '.g', label='theta-dot')
    # plt.legend()
    # plt.subplot(5, 1, 5)
    # plt.plot(dataset['index'].values, dataset['force'].values, '.g', label='force')
    # plt.legend()
    
    # plt.subplots_adjust(hspace=0.5)
    # plt.tight_layout()
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(20, 15))
    plt.suptitle("cartpole(ocs2)")
    markersize_ = 4.0
    epsilon_ = 0.2
    N = len(dataset['index'].values)
    u_max = 5.
    u_min = -10.
    x_max = 1.5
    x_min = -1.5

    axs[0].plot(dataset['index'].values, dataset['x'].values, '.r', label='x', markersize = markersize_)
    axs[0].plot(dataset['index'].values, x_max * np.ones(N), '-b')
    axs[0].plot(dataset['index'].values, x_min * np.ones(N), '-b')
    axs[0].legend()
    axs[0].set_ylim(min(dataset['x'].values) - epsilon_, max(dataset['x'].values) + epsilon_)

    axs[1].plot(dataset['index'].values, dataset['vel'].values, '.r', label='vel', markersize = markersize_)
    axs[1].legend()
    axs[1].set_ylim(min(dataset['vel'].values) - epsilon_, max(dataset['vel'].values) + epsilon_)

    axs[2].plot(dataset['index'].values, dataset['theta'].values, '.r', label='theta', markersize = markersize_)
    axs[2].legend()
    axs[2].set_ylim(min(dataset['theta'].values) - epsilon_, max(dataset['theta'].values) + epsilon_)

    axs[3].plot(dataset['index'].values, dataset['theta-dot'].values, '.r', label='theta-dot', markersize = markersize_)
    axs[3].legend()
    axs[3].set_ylim(min(dataset['theta-dot'].values) - epsilon_, max(dataset['theta-dot'].values) + epsilon_)

    axs[4].plot(dataset['index'].values, dataset['force'].values, '.r', label='force', markersize = markersize_)
    axs[4].plot(dataset['index'].values, u_max * np.ones(N), '-b')
    axs[4].plot(dataset['index'].values, u_min * np.ones(N), '-b')
    axs[4].legend()
    axs[4].set_ylim(min(dataset['force'].values) - epsilon_, max(dataset['force'].values) + epsilon_)

    plt.show(block=False)
    plt.pause(1)
    input("press ctrl+c to quit...")
    plt.close(all)

dataset = read_data('/home/ytw/ocs2_ws/log.csv', 0)

while True:
	func()