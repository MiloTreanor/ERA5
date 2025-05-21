import h5py
import numpy as np
from tqdm import tqdm

def add_labels_to_original(
    h5_path: str,
    dataset_key: str = "data",
    labels_key: str = "labels",
    chunk_size: int = 500,
):
    with h5py.File(h5_path, "a") as f:
        raw = f[dataset_key]           # [T, C, H, W]
        T, C, H, W = raw.shape

        # Remove existing labels ds if present
        if labels_key in f:
            print(f"'{labels_key}' already exists—deleting & recreating.")
            del f[labels_key]

        # Create labels dataset with chunking
        labels_ds = f.create_dataset(
            labels_key,
            shape=(T - 1, H, W),
            dtype=raw.dtype,
            compression="gzip",
            compression_opts=5,
            chunks=(min(chunk_size, T-1), H, W)
        )

        # Write in chunks of frames
        for start in range(0, T - 1, chunk_size):
            end = min(start + chunk_size, T - 1)
            # raw[start+1 : end+1, channel=0, :, :]
            labels_ds[start:end] = raw[start+1:end+1, 0]
            tqdm.write(f"Wrote frames {start:d}–{end-1:d}")
        print(f"Added '/{labels_key}' with shape {(T-1, H, W)} to {h5_path}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--h5_path", help="Path to your [T,13,288,288] HDF5 file", default='test_dataset_normalized.h5')
    p.add_argument("--dataset_key", default="data")
    p.add_argument("--labels_key", default="labels")
    p.add_argument("--chunk_size", type=int, default=500,
                   help="Number of frames per write")
    args = p.parse_args()

    add_labels_to_original(
        args.h5_path,
        dataset_key=args.dataset_key,
        labels_key=args.labels_key,
        chunk_size=args.chunk_size
    )