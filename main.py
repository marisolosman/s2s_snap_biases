# This is a sample Python script.
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    ds = xr.open_dataset('/home/osman/snap_s2s_biases/SH_polar_cap.nc4')
    ndays = 31 + 30 + 31 + 30 + 31
    nlevs = 15
    nyears = 28
    pc_series = ds.z.values.reshape([nyears, ndays, nlevs])

    tl_cor = np.empty([ndays, nlevs - 1])

    for i in np.arange(ndays):
        for j in np.arange(nlevs - 1):
            tl_cor[i, j] = np.corrcoef(pc_series[:, 0, 1], pc_series[:, i, j + 1])[0, 1]

    tl_cor[tl_cor < 0] = np.nan
    plt.figure
    plt.contourf(np.arange(ndays - 51), ds.level.values[1:nlevs], np.transpose(tl_cor[0:ndays - 51]),
                 levels=[0, 0.2, 0.4, 0.6, 0.8, 1], vmin=0, vmax=1)
    plt.yscale('log')
    plt.gca().invert_yaxis()
    plt.show()
    plt.savefig('ejemplo.png')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#file = xr.open_dataset('~/Dropbox/docencia/seminario_computacion/archivos_ejemplos/air.mon.mean.nc')
#print(file)