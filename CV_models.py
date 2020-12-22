# scrip to compute the gph anomalies predicted with values from IC at 10hPa
# calculus done for each model with IC from august to mid october
# then CV predictions are averaged and correlation computed against observations and plotted
import os
from datetime import *
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
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
    begin_mod = np.where(datetime(1999, 8, 1).strftime("%m-%d") <= pd.to_datetime(ds_mod.time.values[0] + ds_mod.step.values[:]).strftime("%m-%d"))[0][0]
    begin = np.int(np.where(ds_obs.fec.values == pd.to_datetime(ds_mod.time.values[0] + ds_mod.step.values[begin_mod]).strftime("%m-%d"))[0])
    end = (begin + len(ds_mod.step.values) - begin_mod) if ((begin + len(ds_mod.step.values) - begin_mod) <= len(ds_obs.fec.values)) else len(ds_obs.fec.values)
    step = len(ds_mod.step.values) if (end != len(ds_obs.fec.values)) else (end - begin)
    print(begin_mod, begin, end, step)
    ds_obs = ds_obs.isel(fec=np.arange(begin, end)).sel(level=ds_mod.isobaricInhPa.values)
    ds_mod = ds_mod.isel(step=np.arange(begin_mod, step))
    ds_mod = ds_mod.transpose('isobaricInhPa', 'time', 'step')
    ds_obs = ds_obs.rename({'z': 'gh', 'level': 'isobaricInhPa', 'anios': 'time', 'fec': 'step'})
    ds_obs['step'] = ds_mod['step']
    ds_obs['time'] = ds_mod['time']
    r = pearson_correlation(ds_obs, ds_mod, 'time')
    r = r.where(r != -9999)
    r['valid_time'] = pd.to_datetime(ds_mod.time.values[0] + ds_mod.step.values)
    return r

def l_mod(x, y):
    """
    Uses the scikit learn module to calculate the CV prediction
    :x vector
    :y vector
    """
    lr = linear_model.LinearRegression()
    predicted = cross_val_predict(lr, pd.DataFrame(x), y, cv=10)
    return predicted

# The function we are going to use for applying our kendal test per pixel
def lin_mod(x, y, dim='time'):
    return xr.apply_ufunc(
        l_mod, x, y,
        input_core_dims=[[dim], [dim]],
        vectorize=True,  # !Important!
        dask='parallelized',
        output_core_dims=[[dim]],
        output_dtypes=[float]
        )

def CV_predic(ds):
    ds_ref = ds.sel(isobaricInhPa=10).isel(step=0)
    ds = ds.transpose('isobaricInhPa', 'step', 'number', 'time')
    ds_ref = ds_ref.transpose('number', 'time')
    ds_predic = lin_mod(ds_ref, ds, 'time')
    return ds_predic

OUT_PATH = "/home/osman/s2s_snap_biases/outputs/DATA/"
OUT_PATH_BOM = "/datos/osman/"
FIG_PATH = "/home/osman/s2s_snap_biases/outputs/FIGURES/"
ds_obs = xr.open_dataset(OUT_PATH + 'SH_polar_cap_1999_2010.nc4')
ds_obs = ds_obs.sel(time=ds_obs['time.year'] != 2002)
# groupby month and day and compute standard deviation
month_day_str = xr.DataArray(ds_obs.indexes['time'].strftime("%m-%d"),
                             coords=[ds_obs.coords["time"]], name="month_day_str")
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
cmap = colors.ListedColormap(['w', '#fff5f0', '#fddcc9', '#ee9677', '#db6b55', '#c53e3d', '#ab162a'])
levels = np.arange(0, 1.2, 0.2)
fechas = pd.date_range(start="1999-08-01", end="1999-12-31")
niveles = ["1000", "850", "500", "200", "100", "50", "10"]
xlabel = [ff.strftime("%m-%d") for ff in fechas[::15]]
for i in models:
   if i != "HMCR":
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
        plt.suptitle(i + ' stat model - Correlation against observations (1999-2010, excl 2002)')
        k = 0
        for j in IC[i]:
            if not (os.path.isfile(OUT_PATH + i + '_SH_predic_1999_2010_IC_' + j + ".nc4")):
                if i == "BOM":
                    ds = xr.open_dataset(OUT_PATH_BOM + i + "_gh_IC_" + j + ".nc4", chunks={'step': 10})
                else:
                    ds = xr.open_dataset(OUT_PATH + i + "_gh_IC_" + j + ".nc4", chunks={'step': 10})
                if pd.to_datetime(ds.valid_time.values[0, 0]) <= datetime.strptime("1999-11-18", "%Y-%m-%d"):
                    # discard 2002
                    ds = ds.sel(time=ds['time.year'] != 2002)
                    ds = ds.sel(latitude=slice(-60, -90)).mean(dim=['latitude', 'longitude'], skipna=True)
                    ds = ds.fillna(ds.mean('time', skipna=True))
                    month_day_str = xr.DataArray(ds.indexes['time'].strftime("%m-%d"),
                                                 coords=[ds.coords["time"]], name="month_day_str")
                    try:
                        ds_predic = CV_predic(ds)
                        ds_predic = ds_predic.compute()
                        ds_predic.to_netcdf(OUT_PATH + i + '_SH_predic_1999_2010_IC_' + j + ".nc4")
                    except:
                        print("can't compute CV predictions", i, j)
            else:
                ds = xr.open_dataset(OUT_PATH + i + '_SH_predic_1999_2010_IC_' + j + ".nc4")
                ds = ds.mean(dim='number', skipna=True)
                ds = ds.sel(isobaricInhPa=ds['isobaricInhPa'] != 925)
                ds_correlations = correlations(ds, ds_obs)
                ds_correlations.to_netcdf(OUT_PATH + i + '_stat_model_Rvserai_1999_2010_IC_' + j + ".nc4")
                ds = xr.open_dataset(OUT_PATH + i + '_stat_model_Rvserai_1999_2010_IC_' + j + ".nc4")
                ds = ds_correlations
                if k < 4:
                    ii = 0
                    jj = k
                else:
                    ii = 1
                    jj = k - 4
                CS = ax[ii, jj].contourf(pd.to_datetime(ds.valid_time.values), ds.isobaricInhPa.values,
                                         ds.gh.values, vmin=0, levels=levels, vmax=1, cmap=cmap)
                CS1 = ax[ii, jj].contour(pd.to_datetime(ds.valid_time.values), ds.isobaricInhPa.values,
                                         ds.gh.values, vmin=0, levels=levels, colors="k", linewidths=0.5)
                ax[ii, jj].clabel(CS1, CS1.levels[::2], fontsize=8, inline=1, fmt='%2.1f')
                ax[ii, jj].set_title('IC: ' + j)
                ax[ii, jj].set_ylabel('Level', fontsize=12)
                ax[ii, jj].set_xlabel('Date', fontsize=12)
                ax[ii, jj].set_yscale('log')
                ax[ii, jj].invert_yaxis()
                ax[ii, jj].set_xlim(datetime.strptime("1999-08-01", "%Y-%m-%d"),
                                    datetime.strptime("1999-12-31", "%Y-%m-%d"))
                ax[ii, jj].set_yticks([1000, 850, 500, 200, 100, 50, 10])
                ax[ii, jj].set_yticklabels(niveles)
                ax[ii, jj].set_xticks(fechas[::15])
                ax[ii, jj].set_xticklabels(xlabel, fontsize=8, rotation=45)
                k += 1
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, right=0.91, wspace=0.3, hspace=0.4)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.03, 0.7])
        fig.colorbar(CS, cax=cbar_ax)
        plt.savefig(FIG_PATH + 'stat_model_RvsERAi_' + i + ".jpg", format="jpg")
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()


