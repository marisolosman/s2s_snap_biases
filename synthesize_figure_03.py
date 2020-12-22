# code to plot synthesize figures based on suggestions of partners
# Fig3: correlation as a function of leadtime at 500hPa

from datetime import *
import numpy as np
import xarray as xr
import pandas as pd
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

IC_obs = ['08_01', '08_15', '09_01', '09_15', '10_01', '10_15', '11_01', '11_15']
titles = ['Beginning of Aug', 'mid Aug', 'Beginning of Sep', 'mid Sep', 'Beginning of Oct',
          'mid Oct', 'Beginning of Nov', 'mid Nov']
fechas = pd.date_range(start="1999-08-01", end="1999-12-31")
xlabel = [ff.strftime("%m-%d") for ff in fechas[::15]]
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(21, 8))
plt.suptitle('Correlation btw model value at 850hPa with model value at 10hPa for diff IC (1999-2010, excl 2002)')
for k in range(8):
    color = iter(cm.rainbow(np.linspace(0, 1, len(models) * 2)))
    if k<4:
        ii = 0
        jj = k
    else:
        ii = 1
        jj = k-4
    ds_obs = xr.open_dataset(PATH_DATA + 'erai_SH_pc_1999_2010_IC_' + IC_obs[k] + '.nc4')
    ds_obs = ds_obs.rolling(fec=7, center=True).mean().dropna('fec')
    xs = [datetime(1999, np.int(i[0:2]), np.int(i[3:5])) for i in ds_obs.fec.values]
    CS = ax[ii, jj].plot(xs, ds_obs.z.sel(level=850).values, color='black',
                         linewidth=2)
    ax[ii, jj].set_ylabel('Correlation', fontsize=12)
    ax[ii, jj].set_xlabel('Date', fontsize=12)
    ax[ii, jj].set_title('IC: ' + titles[k])
    ax[ii, jj].set_xticks(fechas[::15])
    ax[ii, jj].set_xticklabels(xlabel, fontsize=8, rotation=45)
    ax[ii, jj].set_ylim([-0.5, 1])
    ax[ii, jj].set_xlim(datetime.strptime("1999-08-01", "%Y-%m-%d"),
                        datetime.strptime("1999-12-31", "%Y-%m-%d"))
    for i in models:
        if i != 'HMCR':
            IC_model = IC[i][k]
            ds_mod = xr.open_dataset(PATH_DATA + i + '_SH_pc_1999_2010_IC_' + IC_model + '.nc4')
            ds_mod = ds_mod.rolling(step=7, center=True).mean().dropna('step')
            c = next(color)
            startd = datetime.strptime("1999-" + IC_model[0:2] + "-" + IC_model[3:5], "%Y-%m-%d") if i != "CNRM" else datetime.strptime("1999-" + IC_model[0:2] + "-" + IC_model[2:4], "%Y-%m-%d")
            xs_mod = [startd + pd.to_timedelta(ff.astype(timedelta)) for ff in ds_mod.step.values]
            ax[ii, jj].plot(xs_mod, ds_mod.gh.sel(isobaricInhPa=850).values, color=c, label=i, lw=1.2)
            c = next(color)
        else:
            c = next(color)
            c = next(color)
    k += 1
fig.subplots_adjust(top=0.9, bottom=0.2, wspace=0.3, hspace=0.45)
plt.legend(bbox_to_anchor=(-3.2, -0.5), loc='lower left', ncol=len(models))
plt.savefig(OUT_PATH_FIG + 'corr_ic_10_850.jpg', format="jpg", dpi=300)
plt.show()
plt.clf()
plt.cla()
plt.close()
