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
from datetime import datetime,timedelta

# my own misc. functions
import global_use as gl
import funcs_stats as stats
import find_properties as fp
import funcs_analysis as fa

import glob

import xarray as xr

date_string = sys.argv[1]
single_day = datetime.strptime(date_string, "%Y%m%d")
print(single_day)

""" Defining all global variables 
"""

day_files = gl.DataFiles(single_day)

# String version of dates for filename  
date_string,year,month,day =gl.get_date_string(single_day)


""" 
Acess OMNI data from folder
"""
# Create the OMNI dataset
omni_dataset = gl.omni_dataset(single_day,single_day+timedelta(days=1))
AE, epoch_omni = omni_dataset.omni_stats
Kp, Dst, epoch_omni_low = omni_dataset.omni_stats_low_res

date_params = {"year": year,
                "month": month,
                "day": day,
                "single_day": single_day}


# Getting the LANL data
lanl_file = h5py.File(day_files.lanl_data)
lanl_data = gl.AccessLANLAttrs(lanl_file)

# Getting survey file and accessing survey frequencies, epoch and magnitude
survey_file = pycdf.CDF(day_files.survey_data)
survey_data = gl.AccessSurveyAttrs(survey_file)
survey_freq = survey_data.frequency
survey_epoch = survey_data.epoch_convert()
survey_bins = survey_data.bin_edges
Btotal = survey_data.Bmagnitude
Breduced = fa.BackReduction(Btotal, 'B', False)
survey_binwidths = survey_data.frequency_bin_widths
surv = {"Bmagnitude":Breduced, "Frequency":survey_freq, "Epoch":survey_epoch}

# get all gyro's/flhr

mag_dat, mag_check = day_files.magnetic_data()
if mag_check == False:
    print("No magnetometer data on this day")

else:
    mag_file = pycdf.CDF(mag_dat)
    mag_data = gl.AccessL3Attrs(mag_file)
    # Getting gyrofrequencies and plasma frequency for the full day
    fce, fce_05, fce_005,f_lhr = mag_data.f_ce
    fce_epoch = mag_data.epoch_convert()

# Getting LANL attributes
Lstar = lanl_data.L_star
MLT = lanl_data.MLT
MLAT_N, MLAT_S = lanl_data.MLAT_N_S
lanl_epoch = lanl_data.epoch
fnm_Kletzing = 'PSD_Kletzing_'
fnm_sliding = 'PSD_sliding_'


PSD_folder = '/data/hpcflash/users/rablack75/data/Burst'
# What is the date and time of this burst? 
FFT_file_path = PSD_folder + '/' + month + '/output/' + day
# FFT Full filepath
FFT_files_K = os.path.join(FFT_file_path, fnm_Kletzing + "*.cdf")
# files are all CDFs for a given day
FFT_files_K = glob.glob(FFT_files_K)
print("the list of ffts",FFT_files_K)

if not FFT_files_K:
    sys.exit()


FFT_files_s = os.path.join(FFT_file_path, fnm_sliding + "*.cdf")
FFT_files_s = glob.glob(FFT_files_s)

# Create Z dimension based on number of  bursts 
nbursts = len(FFT_files_s)

# Create x dimension based on number of  frequency bands (flhr - fce in 0.1 bands) 
nbands = 10

# Create y dimension based on number of  frequency bands (flhr - fce in 0.1 bands) 
test_cdf = pycdf.CDF(FFT_files_s[1])
ntimes = np.shape(test_cdf["PSD"])[0]


# Create empty 3D DataArray
data_array = xr.DataArray(
    data=np.full((nbursts, nbands, ntimes), np.nan),  # Initialize with NaNs
    dims=("x", "y", "z"),
    name="burst_power"
)

# Create empty 2D DataArrays for statistics along x-axis (mean, sum, std)
mean_power = xr.DataArray(
    data=np.full((nbursts, nbands), np.nan),  # 2D: (x, y)
    dims=("x", "y"),
    name="mean_pow"
)

median_power = xr.DataArray(
    data=np.full((nbursts, nbands), np.nan),  # 2D: (x, y)
    dims=("x", "y"),
    name="median_power"
)

std_power = xr.DataArray(
    data=np.full((nbursts, nbands), np.nan),  # 2D: (x, y)
    dims=("x", "y"),
    name="std_power"
)

max_amp = xr.DataArray(
    data=np.full((nbursts, nbands), np.nan),  # 2D: (x, y)
    dims=("x", "y"),
    name="max_amp"
)

mean_amp = xr.DataArray(
    data=np.full((nbursts, nbands), np.nan),  # 2D: (x, y)
    dims=("x", "y"),
    name="mean_amp"
)

timestamp = xr.DataArray(
    data=np.full(nbursts, np.nan),  # 1D: (x)
    dims="x",
    name="timestamp"
)

survey_power = xr.DataArray(
    data=np.full((nbursts,nbands), np.nan),  # 2D: (x, y)
    dims=("x", "y"),
    name="survey_power"
)

frequency_bands = xr.DataArray(
    data=np.full((nbursts,nbands), np.nan),  # 2D: (x, y)
    dims=("x", "y"),
    name="frequency_bands"
)

# Create a Dataset to hold all DataArrays
dataset = xr.Dataset({
    "burst_power": data_array,
    "mean_power": mean_power,
    "median_power": median_power,
    "std_power": std_power,
    "max_amp": max_amp,
    "mean_amp": mean_amp,
    "timestamp": timestamp,
    "survey_power": survey_power,
    "frequency_bands": frequency_bands
})


# Add attributes (optional)
dataset["burst_power"].attrs["description"] = "integrated power for given (burst, frequency band) coordinate"
dataset["survey_power"].attrs["description"] = "integrated power for survey at given (burst, frequency band) coordinate"
dataset["mean_power"].attrs["description"] = "Mean power for each (burst, frequency band) coordinate"
dataset["median_power"].attrs["description"] = "Median power for each (burst, frequency band) coordinate"
dataset["std_power"].attrs["description"] = "Standard deviation of power for each (burst, frequency band) coordinate"
dataset["timestamp"].attrs["description"] = "Timestamp of burst record"
dataset["max_amp"].attrs["description"] = "Maximum amplitude for each (burst, frequency band) coordinate"
dataset["mean_amp"].attrs["description"] = "Mean amplitude for each (burst, frequency band) coordinate"
dataset["frequency_bands"].attrs["description"] = "Frequency bands from f_lhr to fce"

# start loop through all FFT files
k = 0    
for FFT_Kletzing, FFT_sliding in zip(FFT_files_K, FFT_files_s):
    
    # 0.468s in CDF format 
    FFT_K = pycdf.CDF(FFT_Kletzing)
    # 0.029s in CDF format

    FFT_s = pycdf.CDF(FFT_sliding)

    fft_datetime = FFT_s["BurstDatetime"][...]
    time_min = datetime(year=2015,month=1,day=2,hour=7,minute=17,second=20)
    time_max = time_min+ timedelta(minutes=1)


    dataset["timestamp"].loc[k] = fft_datetime

    gyro = FFT_s["fce"][...]
    findLHR = fp.FindLHR(fft_datetime,fce_epoch,f_lhr)
    flhr_stamp = findLHR.get_lhr

    freq_bandwidth = FFT_s["Frequencies"][1] - FFT_s["Frequencies"][0]
    # do background reduction and then integrate power
    fft_reduced_s = fa.BackReduction(FFT_s["PSD"], 'B', 's')

    

    # Save integration statistics for burst sample (power integral and related stats for each timestamp in burst record) - all frequency bands
    integral_stats_s = stats.FrequencyIntegralFull(fft_reduced_s, FFT_s,FFT_s["Frequencies"], flhr_stamp, gyro, freq_bandwidth).integrate_in_frequency()

    # Save integration for survey sample (power integral for given survey time) - all frequency bands 
    survey_integral,frequency_bands = stats.FrequencyIntegralSurvey(surv,fft_datetime,flhr_stamp, gyro, survey_binwidths).integrate_in_frequency()

    # Assign power to xarray dataset - both burst and survey
    power_array = integral_stats_s["frequency integrated power"]
    dataset["burst_power"].loc[k, :, :] = power_array
    dataset["survey_power"].loc[k, :] = survey_integral

    dataset["mean_power"].loc[k, :] = integral_stats_s["mean power"]
    dataset["median_power"].loc[k, :] = integral_stats_s["median power"]
    dataset["std_power"].loc[k, :] = integral_stats_s["power std"]
    dataset["max_amp"].loc[k, :] = integral_stats_s["maximum amplitude"]
    dataset["frequency_bands"].loc[k,:] = frequency_bands
    #dataset["mean_amp"].loc[k, :] = "Mean amplitude for each (burst, frequency band) coordinate"

        # do plot checks
   

    
    k=k+1

# Save the dataset to a netCDF file
if not os.path.exists(f'/data/hpcflash/users/rablack75/power_netCDFs/{month}'): 
    # if the directory is not present  
    # then create it. 
    os.makedirs(f'/data/hpcflash/users/rablack75/power_netCDFs/{month}')

dataset.to_netcdf(f'/data/hpcflash/users/rablack75/power_netCDFs/{month}/power_{date_string}.nc')
print(str(single_day), "is done")

""" get power statistics out """
