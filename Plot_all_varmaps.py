import os
import json
import numpy as np
import xarray as xr
import torch
import h5py
import matplotlib.pyplot as plt

# ─── Settings ───
precip_dir = 'tmp_single'


tp_files = sorted(os.listdir(precip_dir))[:2]


# backend_kwargs to avoid .idx
backend_kwargs_tp = {"filter_by_keys": {"shortName":"tp"},
                     "indexpath":"", "decode_timedelta":False}

lat_dim, lon_dim = 289, 289
tp_crop = slice(5, -7)  # for flattening


for tp_fn in tp_files:
    # 1) Precipitation
    with xr.open_dataset(os.path.join(precip_dir, tp_fn),
                         engine="cfgrib", backend_kwargs=backend_kwargs_tp) as ds_tp:
        da = ds_tp['tp']  # (time,step,lat,lon)
        arr = da.values.reshape(-1, lat_dim, lon_dim)[tp_crop]  # (t,289,289)

        # Select first 5 samples
        samples = arr[:5]

        # Create figure
        fig, axs = plt.subplots(5, 2, figsize=(15, 25))

        # Plot each sample
        stats = []
        for i, sample in enumerate(samples):
            # Spatial plot
            im = axs[i, 0].imshow(sample, cmap='viridis', origin='lower')
            plt.colorbar(im, ax=axs[i, 0], label='TP (m)')
            axs[i, 0].set_title(f'Sample {i + 1} - Spatial Distribution')

            # Histogram
            flat_data = sample.flatten()
            axs[i, 1].hist(flat_data, bins=50, log=True)
            axs[i, 1].set_title(f'Sample {i + 1} - Value Distribution')
            axs[i, 1].set_xlabel('TP Value (m)')
            axs[i, 1].set_ylabel('Count (log scale)')

            # Calculate statistics
            stats.append({
                'min': np.min(sample),
                'max': np.max(sample),
                'mean': np.mean(sample),
                'std': np.std(sample),
                '95th_percentile': np.percentile(sample, 95),
                'zeros_pct': np.mean(sample == 0) * 100,
                '>1mm_pct': np.mean(sample > 0.001) * 100  # 1mm = 0.001m
            })

        plt.tight_layout()
        plt.savefig('precipitation_samples_stats.png')
        plt.close()

        # Print statistics
        print("\nSample Statistics:")
        for i, s in enumerate(stats):
            print(f"\nSample {i + 1}:")
            print(f"Range: {s['min']:.2e} - {s['max']:.2e} m")
            print(f"Mean: {s['mean']:.2e} ± {s['std']:.2e} m")
            print(f"95th percentile: {s['95th_percentile']:.2e} m")
            print(f"Zero values: {s['zeros_pct']:.1f}%")
            print(f"Values >1mm: {s['>1mm_pct']:.1f}%")

        # Print global stats
        all_data = arr.flatten()
        print("\nGlobal Statistics (all samples in first file):")
        print(f"Global max: {np.max(arr):.2e} m")
        print(f"Global 99.9th percentile: {np.percentile(arr, 99.9):.2e} m")
        print(f"Global zero percentage: {np.mean(arr == 0) * 100:.1f}%")
