# Select ERA interim data and S2S model data for the period 1999-2010 and compute the mean geopotential height over the
# polar cap 60S-90S
import subprocess
import xarray as xr
import numpy as np
import os

if not(os.path.isfile('SH_polar_cap_1999_2010.nc4')):
    FILE = "/home/osman/descarga_era5/gh_erai_day.nc"
    ds = xr.open_dataset(FILE, chunks={'lat': 10, 'lon': 10, 'time': 10})
    # select period of study
    ds = ds.sel(time=np.logical_and(ds['time.year'] >= 1999, ds['time.year'] <= 2010))
    # select polar cap and compute mean
    ds = ds.sel(lat=slice(-60, -90), time=ds['time.month'] >= 8).mean(dim=['lat', 'lon'])
    # save data
    ds.to_netcdf('SH_polar_cap_1999_2010.nc4')

OUT_PATH = "/home/osman/s2s_snap_biases/outputs/DATA/"
models = ["BOM", "CMA", "CNRM", "ECCC", "ECMWF", "HMCR", "ISAC", "JMA", "KMA",
          "NCEP", "UKMO"]
MAIN_DIR = "/datos/S2S/"
PATH = {'BOM': "/datos/S2S/BOM/gh/", 'CMA': "/pikachu/datos2/S2S/CMA/", 'CNRM': "/pikachu/datos2/S2S/CNRM/",
        "ECCC": "/datos/S2S/ECCC/", "ECMWF": "/datos/S2S/ECMWF/", "HMCR": "/datos/S2S/HMCR/",
        "ISAC": "/datos/S2S/ISAC/", "JMA": "/datos/S2S/JMA/", "KMA": "/datos/S2S/KMA/",
        "NCEP": "/datos/S2S/NCEP/", "UKMO": "/datos/S2S/UKMO/"}

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
        if not (os.path.isfile(OUT_PATH + i + "_SH_polar_cap_1999_2010_IC_" + j + ".nc4")):
            if i == "BOM":
                try:
                    ds = xr.open_mfdataset(PATH[i] + i + "_gh_SHPolar_cf_reforecast_*_" + j + ".grib", engine="cfgrib",
                                           combine="nested", concat_dim=["time"])
                    ds = ds.sel(time=np.logical_and(ds['time.year'] >= 1999, ds['time.year'] <= 2010))
                    ds.to_netcdf(i + "_gh_cf_" + j + ".nc4")
                    ds = xr.open_mfdataset(PATH[i] + i + "_gh_SHPolar_pf_reforecast_*_" + j + ".grib", engine="cfgrib",
                                           combine="nested", concat_dim=["time"])
                    ds = ds.sel(time=np.logical_and(ds['time.year'] >= 1999, ds['time.year'] <= 2010))
                    ds.to_netcdf(i + "_gh_pf_" + j + ".nc4")
                    # concatenate cf with pf and compute mean over polar cap
                    ds_cf = xr.open_dataset(i + "_gh_cf_" + j + ".nc4", chunks={'latitude': 10})
                    ds_pf = xr.open_dataset(i + "_gh_pf_" + j + ".nc4", chunks={'latitude': 10})
                    # concate along realization
                    ds = xr.concat([ds_cf, ds_pf], dim="number")
                    ds.to_netcdf(OUT_PATH + i + "_gh_IC_" + j + ".nc4")
                    ds = xr.open_dataset(OUT_PATH + i + "_gh_IC_" + j + ".nc4", chunks={'latitude': 10})
                    # select polar cap
                    ds_pc = ds.sel(latitude=slice(-60, -90)).mean(dim=["latitude", "longitude"])
                    ds_pc.to_netcdf(OUT_PATH + i + "_SH_polar_cap_1999_2010_IC_" + j + ".nc4")
                finally:
                    pass
            else:
                try:
                    subprocess.call("rm " + PATH[i] + "*.idx")
                except IOError:
                    print("No idx file")
                try:
                    ds_cf = xr.open_mfdataset(PATH[i] + i + "_gh_SH_cf_reforecast_*_" + j + ".grib", engine="cfgrib",
                                              combine="nested", concat_dim=["time"])
                    ds_pf = xr.open_mfdataset(PATH[i] + i + "_gh_SH_pf_reforecast_*_" + j + ".grib",
                                              engine="cfgrib", combine="nested", concat_dim=["time"])  # ,
                except OSError:
                    print(PATH[i] + i + "_gh_SH_cf_reforecast_*_" + j + ".grib")
                    # concatenate cf with pf and compute mean over polar cap
                    # concatenate along realization
                try:
                    ds = xr.concat([ds_cf, ds_pf], dim="number")
                    ds = ds.sel(time=np.logical_and(ds['time.year'] >= 1999, ds['time.year'] <= 2010))
                    ds.to_netcdf(OUT_PATH + i + "_gh_IC_" + j + ".nc4")
                    ds = xr.open_dataset(OUT_PATH + i + "_gh_IC_" + j + ".nc4", chunks={'latitude': 10})
                    # select polar cap
                    ds_pc = ds.sel(latitude=slice(-60, -90)).mean(dim=["latitude", "longitude"])
                    ds_pc.to_netcdf(OUT_PATH + i + "_SH_polar_cap_1999_2010_IC_" + j + ".nc4")
                except:
                    print("no xarray file")


