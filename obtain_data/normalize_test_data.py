import os
import json
import numpy as np
import xarray as xr
import torch
import h5py


precip_dir = 'tmp_single'
pressure_dir = 'tmp_pressure'

tp_files = sorted(os.listdir(precip_dir))
pres_files = sorted(os.listdir(pressure_dir))

# Ensure file correspondence
assert len(tp_files) == len(pres_files), "Mismatched file counts"

# Load normalization statistics
with open('stats.json', 'r') as f:
    stats = json.load(f)
mean = np.array(stats['mean'], dtype=np.float32)
std = np.array(stats['std'], dtype=np.float32)

# Original configuration from training
pressure_vars = ['t', 'u', 'v', 'z', 'r', 'w']
pressure_levels = [850, 500]
n_channels = 1 + len(pressure_vars) * len(pressure_levels)
assert len(mean) == n_channels, "Stats-channel mismatch"

lat_dim, lon_dim = 289, 289
tp_crop = slice(5, -7)
backend_kwargs_tp = {
    "filter_by_keys": {"shortName": "tp"},
    "indexpath": "",
    "decode_timedelta": False
}
backend_kwargs_pres = {
    "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
    "indexpath": "",
    "decode_timedelta": False
}

# ─── Calculate Total Time Steps ────────────────────────────────────────────
total_steps = 0
for tp_fn in tp_files:
    with xr.open_dataset(os.path.join(precip_dir, tp_fn),
                         engine="cfgrib",
                         backend_kwargs=backend_kwargs_tp) as ds_tp:
        arr = ds_tp['tp'].values.reshape(-1, lat_dim, lon_dim)[tp_crop]
        total_steps += arr.shape[0]

# ─── Create HDF5 Container ─────────────────────────────────────────────────
h5 = h5py.File('test_dataset_normalized.h5', 'w')
dset = h5.create_dataset(
    'data',
    shape=(total_steps, n_channels, 288, 288),
    dtype='float16',
    chunks=True
)

# ─── Process and Normalize Data ────────────────────────────────────────────
idx = 0
for tp_fn, pres_fn in zip(tp_files, pres_files):
    # Process precipitation data
    with xr.open_dataset(os.path.join(precip_dir, tp_fn),
                         engine="cfgrib",
                         backend_kwargs=backend_kwargs_tp) as ds_tp:
        tp_data = ds_tp['tp'].values.reshape(-1, lat_dim, lon_dim)[tp_crop]

    # Normalize and crop spatial dims
    tp_norm = (tp_data - mean[0]) / std[0]
    tp_norm = tp_norm[:, 1:, 1:]  # (t, 288, 288)

    # Process pressure variables
    with xr.open_dataset(os.path.join(pressure_dir, pres_fn),
                         engine="cfgrib",
                         backend_kwargs=backend_kwargs_pres) as ds_pres:
        ds_pres = ds_pres[pressure_vars]  # Ensure variable order
        da_all = ds_pres.to_array(dim='variable')
        da_chan = da_all.stack(channel=('variable', 'isobaricInhPa'))
        pres_data = da_chan.transpose('channel', 'time', 'latitude', 'longitude').values

    # Normalize and crop
    pres_norm = (pres_data - mean[1:, None, None, None]) / std[1:, None, None, None]
    pres_norm = pres_norm[:, :, 1:, 1:]  # (12, t, 288, 288)

    # Combine data and write
    combined = np.concatenate([tp_norm[None], pres_norm], axis=0)  # (13, t, 288, 288)
    combined = combined.transpose(1, 0, 2, 3)  # (t, 13, 288, 288)

    dset[idx:idx + combined.shape[0]] = combined.astype('float16')
    idx += combined.shape[0]

h5.close()
print("Successfully created normalized test dataset.")