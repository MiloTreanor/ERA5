import h5py
import numpy as np
from tqdm import tqdm

def add_labels_to_original(
    h5_path: str,
    dataset_key: str = "data",
    labels_key: str = "labels",
):
    with h5py.File(h5_path, "a") as f:
        raw = f[dataset_key]   # [T, C, H, W]
        T, C, H, W = raw.shape
        if labels_key in f:
            print(f"'{labels_key}' already exists—deleting & recreating.")
            del f[labels_key]
        labels_ds = f.create_dataset(
            labels_key,
            shape=(T - 1, H, W),
            dtype=raw.dtype,
            compression="gzip",
            compression_opts=5,
        )
        for t in tqdm(range(T - 1), desc="Writing labels"):
            # grab channel 0 at t+1
            labels_ds[t] = raw[t + 1, 0]

    print(f"Added '/{labels_key}' with shape {(T-1, H, W)} to {h5_path}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--h5_path", help="Your normalized [T,13,288,288] file")
    p.add_argument("--dataset_key", default="data")
    p.add_argument("--labels_key", default="labels")
    args = p.parse_args()
    add_labels_to_original('/scratch-shared/tmp.Udl4HYbZtd/dataset_normalized_2018-2021_labelled.h5', args.dataset_key, args.labels_key)