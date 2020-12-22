# code to plot synthesize figures based on suggestions of partners
# Fig1: GPH SD at 50hPa as a function of leadtime

import os
from datetime import *
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

OUT_PATH_FIG = "/home/osman/s2s_snap_biases/outputs/FIGURES/"
PATH_DATA = "/home/osman/s2s_snap_biases/outputs/DATA/"
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
if not os.path.isfile(PATH_DATA + 'ERAi_GPH_SD_50.nc4'):
    ds_obs = xr.open_dataset(PATH_DATA + "SH_polar_cap_1999_2010.nc4")
    # select 50hPa level and remove 2002
    ds_obs = ds_obs.sel(level=50, time=ds_obs['time.year'] != 2002)
    # groupby month and day and compute standard deviation
    month_day_str = xr.DataArray(ds_obs.indexes['time'].strftime("%m-%d"),
                             coords=[ds_obs.coords["time"]], name="month_day_str")
    obs_50_sd = ds_obs.groupby(month_day_str).std("time")
    obs_50_sd.to_netcdf(PATH_DATA + 'ERAi_GPH_SD_50.nc4')
else:
    obs_50_sd = xr.open_dataset(PATH_DATA + 'ERAi_GPH_SD_50.nc4')
titles = ['Beginning of Aug', 'mid Aug', 'Beginning of Sep', 'mid Sep', 'Beginning of Oct',
          'mid Oct', 'Beginning of Nov', 'mid Nov']
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(21, 8))
plt.suptitle(' Z50 standard deviation (1999-2010, excl 2002)')
for k in range(8):
    color = iter(cm.rainbow(np.linspace(0, 1, len(models) * 2)))
    if k<4:
        ii = 0
        jj = k
    else:
        ii = 1
        jj = k-4
    CS = ax[ii, jj].plot(obs_50_sd.month_day_str.values, obs_50_sd.z.values / 9.8, color='black',
                         linewidth=2)
    ax[ii, jj].set_ylabel('Amplitude (m)', fontsize=12)
    ax[ii, jj].set_xlabel('Date', fontsize=12)
    ax[ii, jj].set_title('IC: ' + titles[k])
    fechas = [datetime.strptime(obs_50_sd.month_day_str.values[j],
                                "%m-%d").strftime("%b-%d") for j in range(len(obs_50_sd.month_day_str.values))]
    ax[ii, jj].set_xticks(obs_50_sd.month_day_str.values[::15])
    ax[ii, jj].set_xticklabels(fechas[::15], fontsize=8, rotation=45)
    ax[ii, jj].set_ylim([0, 400])
    for i in models:
        IC_model = IC[i][k]
        mod_50_sd = xr.open_dataset(PATH_DATA + i + 'sd_realizations_IC_' + IC_model + '.nc4')
        mod_50_sd = mod_50_sd.mean(dim='iteration')
        try:
            begin = np.int(np.where(obs_50_sd.month_day_str.values == mod_50_sd.month_day_str.values[0])[0])
            end = [begin + len(mod_50_sd.step.values) if ((begin + len(mod_50_sd.step.values)) <= len(obs_50_sd.month_day_str.values)) else len(obs_50_sd.month_day_str.values)]
            step = [len(mod_50_sd.step.values) if (end[0] != len(obs_50_sd.month_day_str.values)) else end[0] - begin]
            begin_mod = 0
        except:
            begin = 0
            end = [begin + len(mod_50_sd.step.values) - 1 if ((begin + len(mod_50_sd.step.values) - 1) <= len(obs_50_sd.month_day_str.values)) else len(obs_50_sd.month_day_str.values)]
            step = [len(mod_50_sd.step.values) if (end[0] != len(obs_50_sd.month_day_str.values)) else end[0] - begin]
            begin_mod = 1
        c = next(color)
        ax[ii, jj].plot(obs_50_sd.month_day_str.values[begin: end[0]],
                       mod_50_sd.gh[0, begin_mod:step[0]], color=c,
                        label=i, lw=1.2)
        c = next(color)
    k += 1
fig.subplots_adjust(top=0.9, bottom=0.2, wspace=0.3, hspace=0.45)
plt.legend(bbox_to_anchor=(-3.2, -0.5), loc='lower left', ncol=len(models))
plt.savefig(OUT_PATH_FIG + 'sd50_all.jpg', format="jpg", dpi=300, papertype="A4")
plt.show()
plt.clf()
plt.cla()
plt.close()

