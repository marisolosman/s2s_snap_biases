# compute correlations of observed GPH averaged over polar cap against models

import os
from datetime import *
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors

def p_cor(x, y):
    """
    Uses the scipy stats module to calculate a correlation test
    :x vector
    :y vector
    """
    # Check NA values
    co = np.count_nonzero(~np.isnan(x))
    nas = np.sum(np.isnan(x)) + np.sum(np.isnan(y))
    if (co < 4) | (nas != 0):  # If fewer than 4 observations return -9999
        return -9999
    # Run the pearson r test
    tau, pvalue = stats.pearsonr(x, y)
    return tau

# The function we are going to use for applying our pearson test per gridpoint
def pearson_correlation(x, y, dim='time'):
    return xr.apply_ufunc(
        p_cor, x, y,
        input_core_dims=[[dim], [dim]],
        vectorize=True,  # !Important!
        output_dtypes=[float]
    )

def correlations(ds_mod, ds_obs):
    begin = np.int(np.where(ds_obs.fec.values == pd.to_datetime(ds_mod.valid_time.values[0, 0]).strftime("%m-%d"))[0])
    end = begin + len(ds_mod.step.values) if ((begin + len(ds_mod.step.values)) <= len(ds_obs.fec.values)) else len(ds_obs.fec.values)
    step = len(ds_mod.step.values) if (end != len(ds_obs.fec.values)) else (end - begin)
    ds_obs = ds_obs.isel(fec=np.arange(begin, end)).sel(level=ds_mod.isobaricInhPa.values)
    ds_mod = ds_mod.isel(step=np.arange(0, step))
    ds_mod = ds_mod.transpose('number', 'isobaricInhPa', 'time', 'step')
    ds_obs = ds_obs.rename({'z': 'gh', 'level': 'isobaricInhPa', 'anios': 'time', 'fec': 'step'})
    ds_obs['step'] = ds_mod['step']
    ds_obs['time'] = ds_mod['time']
    r = pearson_correlation(ds_obs, ds_mod, 'time')
    r = r.where(r != -9999).mean(dim='number', skipna=True)
    r['valid_time'] = ds_mod['valid_time'][0, np.arange(0, step)]
    return r

OUT_PATH = "/home/osman/s2s_snap_biases/outputs/DATA/"
OUT_PATH_BOM = "/datos/osman/"
FIG_PATH = "/home/osman/s2s_snap_biases/outputs/FIGURES/"
ds_obs = xr.open_dataset(OUT_PATH + 'SH_polar_cap_1999_2010.nc4')
ds_obs = ds_obs.sel(time=ds_obs['time.year'] != 2002)
# groupby month and day and compute standard deviation
month_day_str = xr.DataArray(ds_obs.indexes['time'].strftime("%m-%d"), coords=[ds_obs.coords["time"]],
                             name="month_day_str")
y = np.unique(month_day_str.values)
x = np.unique(ds_obs['time.year'].values)
ind = pd.MultiIndex.from_product((x, y), names=('anios', 'fec'))
ds_obs = ds_obs.assign(time=ind).unstack('time')

models = ["BOM", "CMA", "CNRM", "ECCC", "ECMWF", "HMCR", "ISAC", "JMA", "KMA",
          "NCEP", "UKMO"]
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
    for j in IC[i]:
        if not (os.path.isfile(OUT_PATH + i + "_SH_Rvserai_1999_2010_IC_" + j + ".nc4")):
            if i =="BOM":
                ds = xr.open_dataset(OUT_PATH_BOM + i + "_gh_IC_" + j + ".nc4")
            else:
                ds = xr.open_dataset(OUT_PATH + i + "_gh_IC_" + j + ".nc4")
            if pd.to_datetime(ds.valid_time.values[0, 0]) <= datetime.strptime("1999-11-18", "%Y-%m-%d"):
                # discard 2002
                ds = ds.sel(time=ds['time.year'] != 2002, isobaricInhPa=ds['isobaricInhPa'] != 925)
                ds = ds.sel(latitude=slice(-60, -90)).mean(dim=['latitude', 'longitude'])
                month_day_str_mod = xr.DataArray(ds.indexes['time'].strftime("%m-%d"), coords=[ds.coords["time"]],
                                                 name="month_day_str")
                ds_correlations = correlations(ds, ds_obs)
                ds_correlations.to_netcdf(OUT_PATH + i + "_SH_Rvserai_1999_2010_IC_" + j + ".nc4")
cmap = colors.ListedColormap(['w', '#fff5f0', '#fddcc9', '#ee9677', '#db6b55', '#c53e3d', '#ab162a'])
levels = np.arange(0, 1.2, 0.2)
fechas = pd.date_range(start="1999-08-01", end="1999-12-31")
niveles = ["1000", "850", "500", "200", "100", "50", "10"]
xlabel = [ff.strftime("%m-%d") for ff in fechas[::15]]
for i in models:
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
    plt.suptitle(i + ' Correlation against observations (1999-2010, excl 2002)')
    k = 0
    for j in IC[i]:
        ds = xr.open_dataset(OUT_PATH + i + "_SH_Rvserai_1999_2010_IC_" + j + ".nc4")
        if k<4:
            ii = 0
            jj = k
        else:
            ii = 1
            jj = k-4
        CS = ax[ii, jj].contourf(pd.to_datetime(ds.valid_time.values), ds.isobaricInhPa.values, ds.gh.values, vmin=0,
                                 levels=levels, vmax=1, cmap=cmap)
        CS1 = ax[ii, jj].contour(pd.to_datetime(ds.valid_time.values), ds.isobaricInhPa.values, ds.gh.values, vmin=0,
                                 levels=levels, colors="k", linewidths=0.5)
        ax[ii, jj].clabel(CS1, CS1.levels[::2], fontsize=8, inline=1, fmt='%2.1f')
        ax[ii, jj].set_title('IC: ' + pd.to_datetime(ds.valid_time.values[0]).strftime("%m-%d"))
        ax[ii, jj].set_ylabel('Level', fontsize=20)
        ax[ii, jj].set_xlabel('Date', fontsize=20)
        ax[ii, jj].set_yscale('log')
        ax[ii, jj].invert_yaxis()
        ax[ii, jj].set_xlim(datetime.strptime("1999-08-01", "%Y-%m-%d"), datetime.strptime("1999-12-31",
                                                                                           "%Y-%m-%d"))
        ax[ii, jj].set_yticks([1000, 850, 500, 200, 100, 50, 10])
        ax[ii, jj].set_yticklabels(niveles)
        ax[ii, jj].set_xticks(fechas[::15])
        ax[ii, jj].set_xticklabels(xlabel, fontsize=8, rotation=45)
        k += 1
    fig.subplots_adjust(top=0.9, right=0.85, wspace=0.3, hspace=0.4)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    fig.colorbar(CS, cax=cbar_ax)
    plt.savefig(FIG_PATH + 'Rvserai_' + i + ".jpg", format="jpg", dpi=300, papertype="A4")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

