# script to plot observed GPH SD at 50hPa against model GPH SD obtained through a permutation
# procedure
import os
from datetime import *
import random
import multiprocessing as mp
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def std_real(i, ds, group):
    dates = ds.time.values
    ds = ds.isel(time=xr.DataArray(range(len(ds.time.values)), dims='time',
                                   coords={'time': dates}),
                 number=xr.DataArray(random.choices(range(len(ds.number.values)), k=11), dims='time'))
    mod_50_sd = ds.groupby(group).std("time")
    return mod_50_sd


IN_PATH = "/home/osman/s2s_snap_biases/outputs/DATA/"
OUT_PATH_FIG = "/home/osman/s2s_snap_biases/outputs/FIGURES/"


models = ["BOM", "CMA", "CNRM", "ECCC", "ECMWF", "HMCR", "ISAC", "JMA", "KMA", "NCEP", "UKMO"]
IC = dict(BOM=['08_01', '08_16', '09_01', '09_16', '10_01', '10_16', '11_01', '11_16'],
          CMA=['08_01', '08_15', '09_01', '09_15', '10_01', '10_15', '11_01', '11_15'],
          CNRM=['0801', '0815', '0901', '0915', '1001', '1015', '1101', '1115'],
          ECCC=['08_01', '08_15', '08_29', '09_12', '10_03', '10_17', '10_31', '11_14'],
          ECMWF=['08_01', '08_15', '09_02', '09_16', '09_30', '10_14', '10_31', '11_14'],
          HMCR=['08_01', '08_15', '08_29', '09_12', '10_03', '10_17', '10_31', '11_14'],
          ISAC=['07_30', '08_14', '08_29', '09_13', '09_28', '10_13', '11_02', '11_17'],
          JMA=['07_31', '08_10', '08_31', '09_10', '09_30', '10_10', '10_31', '11_10'],
          KMA=['08_01', '08_17', '09_01', '09_17', '10_01', '10_17', '11_01', '11_17'],
          NCEP=['08_01', '08_15', '09_01', '09_15', '10_01', '10_15', '11_01', '11_15'],
          UKMO=['08_01', '08_17', '09_01', '09_17', '10_01', '10_17', '11_01', '11_17'])

for i in models:
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(19, 8))
    plt.suptitle(i + ' Z50 standard deviation (1999-2010, excl 2002)')
    j = 0
    for k in IC[i]:
        print(i, k)
        if not os.path.isfile(IN_PATH + i + 'sd_realizations_IC_' + k + '.nc4'):
            # open model file
            if os.path.isfile(IN_PATH + i + "_SH_polar_cap_1999_2010_IC_" + k + ".nc4"):
                pool = mp.Pool(mp.cpu_count())
                ds_mod = xr.open_dataset(IN_PATH + i + "_SH_polar_cap_1999_2010_IC_" +
                                         k + ".nc4")
                # select 50hPa level
                ds_mod = ds_mod.sel(isobaricInhPa=50, time=ds_mod['time.year'] != 2002)
                month_day_str = xr.DataArray(ds_mod.indexes['time'].strftime("%m-%d"),
                                             coords=[ds_mod.coords["time"]], name="month_day_str")
                mod_50_sd = [pool.apply(std_real, args=(j, ds_mod, month_day_str)) for j in range(10000)]
                pool.close()
                mod_50_sd = xr.concat(mod_50_sd, dim='iteration')
                mod_50_sd.to_netcdf(OUT_PATH_DATA + i + 'sd_realizations_IC_' + k + '.nc4')
# plot results in panels
ds_obs = xr.open_dataset(IN_PATH + "SH_polar_cap_1999_2010.nc4")
# select 50hPa level and remove 2002
ds_obs = ds_obs.sel(level=50, time=ds_obs['time.year'] != 2002)
# groupby month and day and compute standard deviation
month_day_str = xr.DataArray(ds_obs.indexes['time'].strftime("%m-%d"),
                             coords=[ds_obs.coords["time"]], name="month_day_str")
obs_50_sd = ds_obs.groupby(month_day_str).std("time")
fechas = [datetime.strptime(obs_50_sd.month_day_str.values[j],
                            "%m-%d").strftime("%b-%d") for j in
          range(len(obs_50_sd.month_day_str.values))]
for i in models:
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(19, 8))
    plt.suptitle(i + ' Z50 standard deviation (1999-2010, excl 2002)')
    j = 0
    for k in IC[i]:
        mod_50_sd = xr.open_dataset(IN_PATH + i + 'sd_realizations_IC_' + k + '.nc4')
        try:
            begin = np.int(np.where(obs_50_sd.month_day_str.values == mod_50_sd.month_day_str.values[0])[0])
            end = [begin + len(mod_50_sd.step.values) if (
                        (begin + len(mod_50_sd.step.values)) <= len(obs_50_sd.month_day_str.values)) else len(
                obs_50_sd.month_day_str.values)]
            step = [len(mod_50_sd.step.values) if (end[0] != len(obs_50_sd.month_day_str.values)) else end[0] - begin]
            begin_mod = 0
        except:
            begin = 0
            end = [begin + len(mod_50_sd.step.values) - 1 if (
                        (begin + len(mod_50_sd.step.values) - 1) <= len(obs_50_sd.month_day_str.values)) else len(
                obs_50_sd.month_day_str.values)]
            step = [len(mod_50_sd.step.values) if (end[0] != len(obs_50_sd.month_day_str.values)) else end[0] - begin]
            begin_mod = 1
        if j < 4:
            ii = 0
            jj = j
        else:
            ii = 1
            jj = j - 4
        CS = ax[ii, jj].plot(obs_50_sd.month_day_str.values, obs_50_sd.z.values / 9.8, color='red',
                             linewidth=2)
        ax[ii, jj].fill_between(obs_50_sd.month_day_str.values[begin: end[0]],
                                np.min(mod_50_sd.gh.values[:, 0, begin_mod:step[0]], axis=0),
                                np.max(mod_50_sd.gh.values[:, 0, begin_mod:step[0]], axis=0), alpha=0.3,
                                facecolor="black")
        ax[ii, jj].fill_between(obs_50_sd.month_day_str.values[begin: end[0]],
                                mod_50_sd.gh[:, 0, begin_mod:step[0]].quantile(0.25, dim='iteration'),
                                mod_50_sd.gh[:, 0, begin_mod:step[0]].quantile(0.75, dim='iteration'),
                                alpha=0.5, facecolor="black")
        ax[ii, jj].fill_between(obs_50_sd.month_day_str.values[begin: end[0]],
                                mod_50_sd.gh[:, 0, begin_mod:step[0]].quantile(0.1, dim='iteration'),
                                mod_50_sd.gh[:, 0, begin_mod:step[0]].quantile(0.9, dim='iteration'), alpha=0.4,
                                facecolor="black")
        ax[ii, jj].plot(obs_50_sd.month_day_str.values[begin: end[0]],
                        mod_50_sd.gh[:, 0, begin_mod:step[0]].mean(dim='iteration'), color='black')
        ax[ii, jj].set_ylabel('Amplitude (m)', fontsize=12)
        ax[ii, jj].set_xlabel('Date', fontsize=12)
        ax[ii, jj].set_title('IC ' + mod_50_sd.month_day_str.values[0])
        ax[ii, jj].set_xticks(obs_50_sd.month_day_str.values[::15])
        ax[ii, jj].set_xticklabels(fechas[::15], fontsize=8, rotation=45)
        ax[ii, jj].set_ylim([0, 500])
        j += 1
    fig.subplots_adjust(top=0.9, wspace=0.3, hspace=0.4)
    plt.savefig(OUT_PATH_FIG + 'sd50_' + i + ".jpg", format="jpg",
                dpi=300, papertype="A4")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
