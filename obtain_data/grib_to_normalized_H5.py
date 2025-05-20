import os
import json
import numpy as np
import xarray as xr
import torch
import h5py

# ─── Settings ───
precip_dir = 'tmp_single'
pressure_dir = 'tmp_pressure_500_850'

tp_files = sorted(os.listdir(precip_dir))
pres_files = sorted(os.listdir(pressure_dir))

print(tp_files)
print(pres_files)

# pressure variables and levels in order
pressure_vars = ['t','u','v','z','r','w']
pressure_levels = [850, 500]  # will produce channels in var-order × level-order

# backend_kwargs to avoid .idx
backend_kwargs_tp = {"filter_by_keys": {"shortName":"tp"},
                     "indexpath":"", "decode_timedelta":False}
backend_kwargs_pres = {"filter_by_keys": {"typeOfLevel":"isobaricInhPa"},
                       "indexpath":"", "decode_timedelta":False}

# dimensions
lat_dim, lon_dim = 289, 289
tp_crop = slice(5, -7)  # for flattening

# total number of channels
n_channels = 1 + len(pressure_vars)*len(pressure_levels)

# ─── PASS 1: Compute global mean & std ───
sum_ = np.zeros(n_channels, dtype=np.float64)
sum_sq = np.zeros(n_channels, dtype=np.float64)
count = 0
total_steps = 0

for tp_fn, pres_fn in zip(tp_files, pres_files):
    # 1) Precipitation
    with xr.open_dataset(os.path.join(precip_dir, tp_fn),
                         engine="cfgrib", backend_kwargs=backend_kwargs_tp) as ds_tp:
        da = ds_tp['tp']  # (time,step,lat,lon)
        arr = da.values.reshape(-1,lat_dim,lon_dim)[tp_crop]  # (t,289,289)
    # accumulate for channel 0
    sum_[0] += arr.sum()
    sum_sq[0] += (arr**2).sum()
    count += arr.size
    total_steps += arr.shape[0]


    # 2) Pressure channels
    with xr.open_dataset(os.path.join(pressure_dir, pres_fn),
                         engine="cfgrib", backend_kwargs=backend_kwargs_pres) as ds_pres:
        # stack into (var, time, level, lat, lon)
        da_all = ds_pres.to_array(dim='variable')
        da_chan= da_all.stack(channel=('variable','isobaricInhPa'))
        arr_all= da_chan.transpose('channel','time','latitude','longitude').values
        # iterate channels 1..12

        for i in range(arr_all.shape[0]):
            chan = arr_all[i]  # (time,289,289)
            sum_[i+1] += chan.sum()
            sum_sq[i+1] += (chan**2).sum()
    # time count adds for each pressure channel but already counted via arr.size

# compute global mean/std per channel
mean = sum_ / count
std = np.sqrt(sum_sq/count - mean**2)

# save stats
with open('stats.json','w') as f:
    json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f, indent=2)

# ─── PASS 2: Normalize & write HDF5 ───
# reopen stats
with open('stats.json') as f:
    stats = json.load(f)
mean = np.array(stats['mean'], dtype=np.float32)
std = np.array(stats['std'],  dtype=np.float32)

# estimate total timesteps across all files
# (assuming consistent tp length)
with xr.open_dataset(os.path.join(precip_dir, tp_files[0]),
                     engine="cfgrib", backend_kwargs=backend_kwargs_tp) as ds0:
    n0 = ds0['tp'].values.reshape(-1,lat_dim,lon_dim)[tp_crop].shape[0]

# create HDF5 for full dataset
h5 = h5py.File('dataset_normalized_2018-2021.h5','w')
dset = h5.create_dataset('data',
    shape=(total_steps, n_channels, 288, 288),
    dtype='float16', chunks=(n0, n_channels, 288, 288))

idx = 0
for tp_fn, pres_fn in zip(tp_files, pres_files):
    # load & normalize tp
    with xr.open_dataset(os.path.join(precip_dir,tp_fn),
                         engine="cfgrib", backend_kwargs=backend_kwargs_tp) as ds_tp:
        arr_tp = ds_tp['tp'].values.reshape(-1,lat_dim,lon_dim)[tp_crop]
    arr_tp = (arr_tp - mean[0]) / std[0]
    arr_tp = arr_tp[:,1:,1:]  # drop first row/col → 288×288

    # load & normalize pressure channels
    with xr.open_dataset(os.path.join(pressure_dir,pres_fn),
                         engine="cfgrib", backend_kwargs=backend_kwargs_pres) as ds_pres:
        da_all = ds_pres.to_array(dim='variable')
        da_chan= da_all.stack(channel=('variable','isobaricInhPa'))
        arr_all= da_chan.transpose('channel','time','latitude','longitude').values
    # normalize & crop spatial
    arr_all = (arr_all - mean[1:,None,None,None]) / std[1:,None,None,None]
    arr_all = arr_all[:,:,1:,1:]  # now (12, t, 288,288)

    # stack tp + pressure → (13, t,288,288) then transpose to (t,13,288,288)
    bloc = np.concatenate([arr_tp[None], arr_all], axis=0)
    bloc = np.transpose(bloc, (1,0,2,3))

    # write into HDF5
    n = bloc.shape[0]
    dset[idx:idx+n] = bloc.astype('float16')
    idx += n

h5.close()
print("Done: dataset_normalized.h5 created.")

'''
[0] tp – total precipitation

[1] t @ 850 hPa – temperature at 850 hPa

[2] t @ 500 hPa – temperature at 500 hPa

[3] u @ 850 hPa – zonal wind at 850 hPa

[4] u @ 500 hPa – zonal wind at 500 hPa

[5] v @ 850 hPa – meridional wind at 850 hPa

[6] v @ 500 hPa – meridional wind at 500 hPa

[7] z @ 850 hPa – geopotential height at 850 hPa

[8] z @ 500 hPa – geopotential height at 500 hPa

[9] r @ 850 hPa – relative humidity at 850 hPa

[10] r @ 500 hPa – relative humidity at 500 hPa

[11] w @ 850 hPa – vertical velocity at 850 hPa

[12] w @ 500 hPa – vertical velocity at 500 hPa
'''
