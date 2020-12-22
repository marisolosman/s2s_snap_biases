# compute correlations of observed GPH averaged over polar cap against value at 10hPa

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
def pearson_correlation(x, y, dim='anios'):
    return xr.apply_ufunc(
        p_cor, x, y,
        input_core_dims=[[dim], [dim]],
        vectorize=True,  # !Important!
        output_dtypes=[float]
    )
def correlations(ds_mod, ds_obs):
    ds_mod = ds_mod.transpose('fec', 'anios')
    ds_obs = ds_obs.transpose('level', 'fec', 'anios')
    r = pearson_correlation(ds_obs, ds_mod, 'anios')
    r = r.where(r != -9999)
    r['valid_time'] = ds_mod.fec.values[0]
    return r

FILE = "/home/osman/descarga_era5/gh_erai_day.nc"
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

IC = ['08_01', '08_15', '09_01', '09_15', '10_01', '10_15', '11_01', '11_15']
cmap = colors.ListedColormap(['w', '#fff5f0', '#fddcc9', '#ee9677', '#db6b55', '#c53e3d', '#ab162a'])
levels = np.arange(0, 1.2, 0.2)
fechas = pd.date_range(start="1999-08-01", end="1999-12-31")
xlabel = [ff.strftime("%m-%d") for ff in fechas[::15]]
niveles = ["1000", "850", "500", "200", "100", "50", "10"]
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
plt.suptitle(' ERAi stat model - Correlation against observations (1999-2010, excl 2002)')
k = 0
for j in IC:
        begin = np.int(np.where(ds_obs.fec.values == j[0:2] +"-"+j[3:5])[0])
        ds_ref = ds_obs.isel(fec=np.arange(begin, len(ds_obs.fec.values))).sel(level=10)
        ds_predic = ds_obs.isel(fec=np.arange(begin, len(ds_obs.fec.values)))
        ds_correlations = correlations(ds_ref, ds_predic)
        ds_correlations.to_netcdf(OUT_PATH + 'erai_SH_pc_1999_2010_IC_' + j + ".nc4")
        ds = ds_correlations
        if k < 4:
            ii = 0
            jj = k
        else:
            ii = 1
            jj = k - 4
        xs = [datetime(1999, np.int(i[0:2]), np.int(i[3:5])) for i in ds.fec.values]
        CS = ax[ii, jj].contourf(xs, ds.level.values, ds.z.values, vmin=0, levels=levels, vmax=1, cmap=cmap)
        CS1 = ax[ii, jj].contour(xs, ds.level.values, ds.z.values, vmin=0, levels=levels, colors="k", linewidths=0.5)
        ax[ii, jj].clabel(CS1, CS1.levels[::2], fontsize=8, inline=1, fmt='%2.1f')
        ax[ii, jj].set_title('IC: ' + j)
        ax[ii, jj].set_ylabel('Level', fontsize=12)
        ax[ii, jj].set_xlabel('Date', fontsize=12)
        ax[ii, jj].set_yscale('log')
        ax[ii, jj].set_ylim([10, 1000])
        ax[ii, jj].invert_yaxis()
        ax[ii, jj].set_xlim(datetime.strptime("1999-08-01", "%Y-%m-%d"), datetime.strptime("1999-12-31", "%Y-%m-%d"))
        ax[ii, jj].set_yticks([1000, 850, 500, 200, 100, 50, 10])
        ax[ii, jj].set_yticklabels(niveles)
        ax[ii, jj].set_xticks(fechas[::15])
        ax[ii, jj].set_xticklabels(xlabel, fontsize=8, rotation=45)
        k += 1
fig.tight_layout()
fig.subplots_adjust(top=0.9, right=0.91, wspace=0.3, hspace=0.4)
cbar_ax = fig.add_axes([0.94, 0.15, 0.03, 0.7])
fig.colorbar(CS, cax=cbar_ax)
plt.savefig(FIG_PATH + 'RvsERAi_obs.jpg', format="jpg")
plt.show()
plt.clf()
plt.cla()
plt.close()


