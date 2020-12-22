# scrip to compute and plot u bias at different levels for different IC
# procedure
import os
from datetime import *

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as CM
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
ds_obs = xr.open_dataset(IN_PATH + "u_zonalmean_SH_1999_2010.nc4")
# select 10, 200, 500 and 850 hPa level and remove 2002
levels = [10, 200, 500, 850]
ds_obs = ds_obs.sel(level=levels, time=ds_obs['time.year'] != 2002)
# groupby month and day and compute mean
month_day_str = xr.DataArray(ds_obs.indexes['time'].strftime("%m-%d"), coords=[ds_obs.coords["time"]],
                             name="month_day_str")
obs_u_mean = ds_obs.groupby(month_day_str).mean("time")
for i in models:
        for k in IC[i]:
            if i != 'BOM':
                if not os.path.isfile(IN_PATH + i + '_u_bias_IC_' + k + '.nc4') and os.path.isfile(
                        IN_PATH + i + "_u_zonalmean_1999_2010_IC_" + k + ".nc4"):
                    ds_mod = xr.open_dataset(IN_PATH + i + '_u_zonalmean_1999_2010_IC_' + k + ".nc4")
                    # select levels
                    if i == 'HMCR':
                        ds_mod = ds_mod.sel(isobaricInhPa=[200, 500, 850], time=ds_mod['time.year'] != 2002)
                    else:
                        ds_mod = ds_mod.sel(isobaricInhPa=levels, time=ds_mod['time.year'] != 2002)
                    ds_mod['step'] = [datetime.utcfromtimestamp(j.tolist() / 1e9).strftime("%m-%d") for j in
                                      ds_mod.valid_time.values[0, :]]
                    ds_mod = ds_mod.mean(dim=['number', 'time'], skipna=True)
                    try:
                        begin = np.int(np.where(obs_u_mean.month_day_str.values == ds_mod.step.values[0])[0])
                        end = begin + len(ds_mod.step.values) if (
                                    (begin + len(ds_mod.step.values)) <= len(obs_u_mean.month_day_str.values)) else len(
                            obs_u_mean.month_day_str.values)
                        step = len(ds_mod.step.values) if (end != len(obs_u_mean.month_day_str.values)) else end - begin
                        begin_mod = 0
                    except:
                        begin = 0
                        end = begin + len(ds_mod.step.values) - 1 if ((begin + len(ds_mod.step.values) - 1) <= len(
                            obs_u_mean.month_day_str.values)) else len(obs_u_mean.month_day_str.values)
                        step = len(ds_mod.step.values) if (end != len(obs_u_mean.month_day_str.values)) else end - begin
                        begin_mod = 1
                    ds_mod = ds_mod.rename({"step": "month_day_str", "isobaricInhPa": 'level', "latitude": 'lat'})
                    ds_obs_s = obs_u_mean.isel(month_day_str=np.arange(begin, end))
                    ds_mod = ds_mod.isel(month_day_str=np.arange(begin_mod, step))
                    ds_obs_i = ds_obs_s.interp_like(ds_mod)
                    bias = ds_mod - ds_obs_i
                    bias.to_netcdf(IN_PATH + i + '_u_bias_IC_' + k + '.nc4')
            else:
                if not os.path.isfile(IN_PATH + i + '_u_bias_IC_' + k + '.nc4'):
                    ds_mod = xr.open_dataset(IN_PATH + i + "_u_zonalmean_1999_2010_IC_" + k + ".nc4")
                    ds_mod = ds_mod.sel(level=levels, hdate=ds_mod['hdate.year'] != 2002)
                    ds_mod['step'] = pd.date_range(datetime.utcfromtimestamp(ds_mod.hdate.values[0].tolist() / 1e9),
                                                   periods=len(ds_mod.step.values) + 1)[1:]
                    ds_mod['step'] = [datetime.utcfromtimestamp(jj.tolist() / 1e9).strftime("%m-%d") for jj in ds_mod.step.values]
                    ds_mod = ds_mod.mean(dim=['number', 'hdate'], skipna=True)
                    begin = np.int(np.where(obs_u_mean.month_day_str.values == ds_mod.step.values[0])[0])
                    end = begin + len(ds_mod.step.values) if (
                          (begin + len(ds_mod.step.values)) <= len(obs_u_mean.month_day_str.values)) else len(obs_u_mean.month_day_str.values)
                    step = len(ds_mod.step.values) if (end != len(obs_u_mean.month_day_str.values)) else end - begin
                    begin_mod = 0
                    ds_mod = ds_mod.rename({"step": "month_day_str", "latitude": 'lat'})
                    ds_obs_s = obs_u_mean.isel(month_day_str=np.arange(begin, end))
                    ds_mod = ds_mod.isel(month_day_str=np.arange(begin_mod, step))
                    ds_obs_i = ds_obs_s.interp_like(ds_mod)
                    bias = ds_mod - ds_obs_i
                    bias.to_netcdf(IN_PATH + i + '_u_bias_IC_' + k + '.nc4')

levs = np.arange(-7, 8, 1)
cmap = CM.RdBu_r
fechas = pd.date_range(start="1999-08-01", end="1999-12-31")
xlabel = [ff.strftime("%m-%d") for ff in fechas[::15]]
niveles = ["-80", "-70", "-60", "-50", "-40", "-30"]
for i in models:
        for k in IC[i]:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(19, 5))
            plt.suptitle(i + ' u bias (1999-2010, excl 2002) - IC: ' + k)
            ds = xr.open_dataset(IN_PATH + i + '_u_bias_IC_' + k + '.nc4')
            xs = [datetime(1999, np.int(i[0:2]), np.int(i[3:5])) for i in ds.month_day_str.values]
            jj = 0
            for j in levels[1: len(levels)]:
                CS = ax[jj].contourf(xs, ds.lat.values, np.transpose(ds.u.sel(level=j).values),
                                    vmin=-7, levels=levs, vmax=7, cmap=cmap, extend='both')
                cmap.set_under(cmap(0))
                cmap.set_over(cmap(cmap.N - 1))
                ax[jj].set_ylabel('Latitude', fontsize=10)
                ax[jj].set_xlabel('Date', fontsize=10)
                ax[jj].set_title(str(j) + ' hPa')
                ax[jj].set_yticklabels(niveles)
                ax[jj].set_xticks(fechas[::15])
                ax[jj].set_xticklabels(xlabel, fontsize=8, rotation=45)
                ax[jj].set_ylim([-80, -30])
                ax[jj].set_xlim(datetime.strptime("1999-08-01", "%Y-%m-%d"),
                                datetime.strptime("1999-12-31", "%Y-%m-%d"))
                jj += 1
            fig.tight_layout()
            fig.subplots_adjust(top=0.9, right=0.91, wspace=0.3, hspace=0.4)
            cbar_ax = fig.add_axes([0.94, 0.15, 0.03, 0.7])
            fig.colorbar(CS, cax=cbar_ax)
            plt.savefig(OUT_PATH_FIG + 'u_bias_' + i + "_IC_" + k + ".jpg", format="jpg",
                        dpi=300, papertype="A4")
            plt.show()
            plt.clf()
            plt.cla()
            plt.close()
