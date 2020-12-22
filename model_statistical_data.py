# compute correlations of model GPH averaged over polar cap against value at 10hPa
# correlation is computed using a 10000 permutation procedure
import os
from datetime import *
import random
import multiprocessing as mp
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
# The function we are going to use for applying our kendal test per pixel
def pearson_correlation(x, y, dim='time'):
    return xr.apply_ufunc(
        p_cor, x, y,
        input_core_dims=[[dim], [dim]],
        vectorize=True,  # !Important!
        output_dtypes=[float]
    )

def correlations(i, ds):
    dates = ds.time.values
    rand_members = random.choices(range(len(ds.number.values)), k=11)
    ds_ref = ds.isel(time=xr.DataArray(range(len(ds.time.values)), dims='time',
                                       coords={'time': dates}),
                     number=xr.DataArray(rand_members, dims='time'))
    ds_ref = ds_ref.sel(isobaricInhPa=10).isel(step=0)
    ds = ds.transpose('step', 'isobaricInhPa', 'time', 'number')
    gh_m = np.empty(ds.gh.shape[0:3])
    for ii in range(len(dates)):
        gh_m[:, :, ii] = np.mean(ds.gh.values[:, :, ii, rand_members[ii] != ds.number.values], axis=2)
    ds_mean = xr.DataArray(gh_m, coords=[ds.step.values, ds.isobaricInhPa.values,
                                         ds.time.values],
                           dims=['step', 'isobaricInhPa', 'time'], name="gh")
    r = pearson_correlation(ds_ref, ds_mean, 'time')
    return r

FILE = "/home/osman/descarga_era5/gh_erai_day.nc"
OUT_PATH = "/home/osman/s2s_snap_biases/outputs/DATA/"
OUT_PATH_BOM = "/datos/osman/"
FIG_PATH = "/home/osman/s2s_snap_biases/outputs/FIGURES/"
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
    if i != "HMCR":
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
        plt.suptitle(i + ' Correlation (1999-2010, excl 2002) ')
        k = 0
        for j in IC[i]:
            if not (os.path.isfile(OUT_PATH + i + "_SH_pc_1999_2010_IC_" + j + ".nc4")):
                if i == "BOM":
                    ds = xr.open_dataset(OUT_PATH_BOM + i + "_gh_IC_" + j + ".nc4")
                else:
                    ds = xr.open_dataset(OUT_PATH + i + "_gh_IC_" + j + ".nc4")
                if pd.to_datetime(ds.valid_time.values[0, 0]) <= datetime.strptime("1999-11-18", "%Y-%m-%d"):
                    # discard 2002
                    ds = ds.sel(time=ds['time.year'] != 2002)
                    ds = ds.sel(latitude=slice(-60, -90)).mean(dim=['latitude', 'longitude'])
                    month_day_str = xr.DataArray(ds.indexes['time'].strftime("%m-%d"),
                                                 coords=[ds.coords["time"]], name="month_day_str")
                    pool = mp.Pool(mp.cpu_count())
                    ds_correlations = [pool.apply(correlations, args=(j, ds)) for j in range(10000)]
                    pool.close()
                    ds_correlations = xr.concat(ds_correlations, dim='iteration')
                    ds_correlations = ds_correlations.where(ds_correlations != -9999)
                    ds_correlations = ds_correlations.mean(dim='iteration', skipna=True)
                    ds_correlations.to_netcdf(OUT_PATH + i + "_SH_pc_1999_2010_IC_" + j + ".nc4")
niveles = ["1000", "850", "500", "200", "100", "50", "10"]
cmap = colors.ListedColormap(['w', '#fff5f0', '#fddcc9', '#ee9677', '#db6b55', '#c53e3d', '#ab162a'])
levels = np.arange(0, 1.2, 0.2)
fechas = pd.date_range(start="1999-08-01", end="1999-12-31")
xlabel = [ff.strftime("%m-%d") for ff in fechas[::15]]
for i in models:
    if i == "HMCR":
        pass
    else:
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
        plt.suptitle(i + ' Correlation (1999-2010, excl 2002) ')
        k = 0
        for j in IC[i]:
            ds = xr.open_dataset(OUT_PATH + i + "_SH_pc_1999_2010_IC_" + j + ".nc4")
            if k < 4:
                ii = 0
                jj = k
            else:
                ii = 1
                jj = k-4
            startd = datetime.strptime("1999-" + j[0:2] + "-" + j[3:5], "%Y-%m-%d") if i != "CNRM" else datetime.strptime("1999-" + j[0:2] + "-" + j[2:4], "%Y-%m-%d")
            xaxis = [startd + pd.to_timedelta(ff.astype(timedelta)) for ff in ds.step.values]
            CS = ax[ii, jj].contourf(xaxis, ds.isobaricInhPa.values, np.transpose(ds.gh.values),
                                     vmin=0, levels=levels, vmax=1, cmap=cmap)
            CS1 = ax[ii, jj].contour(xaxis, ds.isobaricInhPa.values, np.transpose(ds.gh.values),
                                     vmin=0, levels=levels, colors="k", linewidths=0.5)
            ax[ii, jj].clabel(CS1, CS1.levels[::2], fontsize=8, inline=1, fmt='%2.1f')
            ax[ii, jj].set_title('IC: ' + startd.strftime("%m-%d"))
            ax[ii, jj].set_ylabel('Level', fontsize=20)
            ax[ii, jj].set_xlabel('Date', fontsize=20)
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
        plt.savefig(FIG_PATH + 'correlation_' + i + ".jpg", format="jpg",
                    dpi=300)
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()