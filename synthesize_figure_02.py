# code to plot synthesize figures based on suggestions of partners
# Fig2: zonal mean zonal wind integrated bias
import os
from datetime import *
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

OUT_PATH_FIG = "/home/osman/s2s_snap_biases/outputs/FIGURES/"
IN_PATH = "/home/osman/s2s_snap_biases/outputs/DATA/"
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

#compute integrated mean bias and integrated mean zonal wind
ds_obs = xr.open_dataset(IN_PATH + 'u_zonalmean_SH_1999_2010.nc4')
# select 10, 200, 500 and 850 hPa level and remove 2002
levels = [200, 300, 500, 700, 850, 1000]
ds_obs = ds_obs.sel(level=levels, time=ds_obs['time.year'] != 2002)
# groupby month and day and compute mean
month_day_str = xr.DataArray(ds_obs.indexes['time'].strftime("%m-%d"), coords=[ds_obs.coords["time"]],
                             name="month_day_str")
obs_u_mean = ds_obs.groupby(month_day_str).mean("time")
# integrate vertically
obs_int_u_mean = obs_u_mean.integrate('level')
obs_int_u_mean.u.values = obs_int_u_mean.u.values / (levels[-1] - levels[0])

for i in models:
        for k in IC[i]:
            if not os.path.isfile(IN_PATH + i + '_u_int_bias_IC_' + k + '.nc4') and os.path.isfile(
                IN_PATH + i + '_u_zonalmean_1999_2010_IC_' + k + ".nc4"):
                if i != 'BOM':
                    ds_mod = xr.open_dataset(IN_PATH + i + '_u_zonalmean_1999_2010_IC_' + k + ".nc4")
                    ds_mod = ds_mod.sel(isobaricInhPa=levels[2:], time=ds_mod['time.year'] != 2002)
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
                else:
                    ds_mod = xr.open_dataset(IN_PATH + i + '_u_zonalmean_1999_2010_IC_' + k + ".nc4")
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
                limits = ds_mod.level.values[-1] - ds_mod.level.values[0]
                ds_mod = ds_mod.integrate('level')
                ds_mod.u.values = ds_mod.u.values / limits
                obs_int_u_mean_i = obs_int_u_mean.interp_like(ds_mod)
                bias = ds_mod - obs_int_u_mean_i
                bias.to_netcdf(IN_PATH + i + '_u_int_bias_IC_' + k + '.nc4')
titles = ['Beginning of Aug', 'Beginning of Sep', 'Beginning of Oct', 'Beginning of Nov']
levs = np.arange(-7, 8, 1)
levs_obs = np.arange(8, 23, 3)
cmap = cm.RdBu_r
fechas = pd.date_range(start="1999-08-01", end="1999-12-31")
models_low = ["BOM", "CMA", "ECCC", "HMCR", "ISAC"]
models_high = ["CNRM", "ECMWF", "JMA", "KMA", "NCEP", "UKMO"]
niveles = ["-80", "-70", "-60", "-50", "-40", "-30"]
xlabel = [ff.strftime("%m-%d") for ff in fechas[::15]]
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 20))
plt.suptitle(' <[u]> bias (1999-2010, excl 2002)')
jj = 0
obs_int_u_mean = obs_int_u_mean.rolling(month_day_str=7, center=True).mean().dropna('month_day_str')
for k in range(0, 8, 2):
    ubias = []
    for i in models_low:
            ds = xr.open_dataset(IN_PATH + i + '_u_int_bias_IC_' + IC[i][k] + '.nc4')
            ubias.append(ds)
    ubias = xr.concat(ubias, dim='model')
    meanB = ubias.mean(dim='model', skipna=True)
    xs = [datetime(1999, np.int(i[0:2]), np.int(i[3:5])) for i in meanB.month_day_str.values]
    ax[jj, 0].contourf(xs, meanB.lat.values, np.transpose(meanB.u.values), vmin=-7, levels=levs,
                       vmax=7, cmap=cmap, extend='both')
    xs_obs = [datetime(1999, np.int(i[0:2]), np.int(i[3:5])) for i in obs_int_u_mean.month_day_str.values]
    ax[jj, 0].contour(xs_obs, obs_int_u_mean.lat.values, np.transpose(obs_int_u_mean.u.values), levels=levs_obs,
                      cmap=cm.Greys)
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(cmap.N - 1))
    ax[jj, 0].set_ylabel('Latitude', fontsize=10)
    ax[jj, 0].set_xlabel('Date', fontsize=10)
    ax[jj, 0].set_title('Low-top models - ''IC: ' + titles[jj])
    ax[jj, 0].set_yticklabels(niveles)
    ax[jj, 0].set_xticks(fechas[::15])
    ax[jj, 0].set_xticklabels(xlabel, fontsize=8, rotation=45)
    ax[jj, 0].set_ylim([-80, -30])
    ax[jj, 0].set_xlim(datetime.strptime("1999-08-01", "%Y-%m-%d"), datetime.strptime("1999-12-31", "%Y-%m-%d"))
    ubias = []
    for i in models_high:
            ds = xr.open_dataset(IN_PATH + i + '_u_int_bias_IC_' + IC[i][k] + '.nc4')
            ubias.append(ds)
    ubias = xr.concat(ubias, dim='model')
    meanB = ubias.mean(dim='model', skipna=True)
    xs = [datetime(1999, np.int(i[0:2]), np.int(i[3:5])) for i in meanB.month_day_str.values]
    CS = ax[jj, 1].contourf(xs, meanB.lat.values, np.transpose(meanB.u.values), vmin=-7, levels=levs,
                       vmax=7, cmap=cmap, extend='both')
    ax[jj, 1].contour(xs_obs, obs_int_u_mean.lat.values, np.transpose(obs_int_u_mean.u.values), levels=levs_obs,
                      cmap=cm.Greys)
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(cmap.N - 1))
    ax[jj, 1].set_ylabel('Latitude', fontsize=10)
    ax[jj, 1].set_xlabel('Date', fontsize=10)
    ax[jj, 1].set_title('High-top models - IC: ' + titles[jj])
    ax[jj, 1].set_yticklabels(niveles)
    ax[jj, 1].set_xticks(fechas[::15])
    ax[jj, 1].set_xticklabels(xlabel, fontsize=8, rotation=45)
    ax[jj, 1].set_ylim([-80, -30])
    ax[jj, 1].set_xlim(datetime.strptime("1999-08-01", "%Y-%m-%d"), datetime.strptime("1999-12-31", "%Y-%m-%d"))
    jj += 1
fig.tight_layout()
fig.subplots_adjust(top=0.94, right=0.87, wspace=0.3, hspace=0.4)
cbar_ax = fig.add_axes([0.94, 0.23, 0.03, 0.5])
fig.colorbar(CS, cax=cbar_ax)
plt.savefig(OUT_PATH_FIG + 'u_int_bias.jpg', format="jpg",
            dpi=300)
plt.show()
plt.clf()
plt.cla()
plt.close()
