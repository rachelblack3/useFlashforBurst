# other numerical tools
import os
os.environ["CDF_LIB"] = "/data/hpcdata/users/rablack75/cdf37_1-dist/lib"
import sys

# numerical essentials 
import numpy as np
import matplotlib.pyplot as plt
# for plotting
import matplotlib.colors as mcolors
import matplotlib.dates as mdates


# for creating cdfs
import pandas as pd


# for cdf reading
from spacepy import toolbox
#spacepy.toolbox.update(leapsecs=True)
from spacepy import pycdf
import h5py

# other numerical tools
import os
from datetime import datetime,timedelta, date

# my own misc. functions
import global_bug as gl
import funcs_stats as stats
import find_properties as fp
import funcs_analysis as fa

import glob

import xarray as xr
        

# do plot checks
T_window = 1024*(1/35000)
duration = 208896*(1/35000)
n_bins = 406

t_array = np.linspace(T_window,duration, n_bins).tolist()
t_array.insert(0,0.)
t_array = np.array(t_array)

burst_1 = dataset["burst_power"][k,:,:]

print(burst_1)
print(t_array)
f05_1 = np.zeros(406)
f01_05 = np.zeros(406)
f_all = np.zeros(406)
for i in range(406):
    f01_05[i] = np.sum(burst_1[:5,i])
    f05_1[i] = np.sum(burst_1[5:,i])
    f_all[i] = np.sum(burst_1[:,i])

# make figure for psd and integrated power
fig,axes = plt.subplots(2,1,figsize =(12,10))
# Name plot axis

ax1 = axes[0]
plot = FFT_s
# plot survey PSD
psd_reduced = fft_reduced_s
colorbar_norm = mcolors.LogNorm(vmin=10**(-9), vmax=10**(-5))
# append final frequency value
Frequency = plot["Frequencies"][:]
Frequency = np.append(Frequency, plot["Frequencies"][-1]+(plot["Frequencies"][-1]-plot["Frequencies"][-2]))
burst_samps = ax1.pcolormesh(plot["Time"], np.asarray(Frequency) , np.array(psd_reduced).T, norm=colorbar_norm, cmap="viridis")
fig.colorbar(burst_samps,label=r'$nT^2/Hz$',ax= ax1)
ax1.set_ylim(0,gyro)
# plot integrated power
ax2=axes[1]
non_zero_indices = np.nonzero(f01_05)[0]
ax2.plot(t_array[non_zero_indices],f01_05[f01_05 != 0],label = f'lower')#, linestyle = 'dashed', marker = '*')
non_zero_indices = np.nonzero(f05_1)[0]
ax2.plot(t_array[non_zero_indices],f05_1[f05_1 != 0],label = f'upper')
ax2.set_yscale('log')
ax2.legend()

plt.show()