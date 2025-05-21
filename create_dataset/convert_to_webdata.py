import h5py
import torch
import os
from tqdm import tqdm
from webdataset import ShardWriter


def convert_h5_to_wds(h5_path, output_dir, shard_size=10000, prefix="train"):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        # Get dataset dimensions
        data = f["data"]
        labels = f["labels"] if "labels" in f else None
        total_samples = data.shape[0]

        # Create shard pattern
        pattern = os.path.join(output_dir, f"{prefix}-%06d.tar")

        with ShardWriter(pattern, maxcount=shard_size) as writer:
            for idx in tqdm(range(total_samples), desc=f"Converting {prefix}"):
                sample = {
                    "__key__": f"sample{idx:09d}",
                    "input.pth": torch.from_numpy(data[idx]),
                }

                if labels is not None:
                    sample["label.pth"] = torch.from_numpy(labels[idx])

                writer.write(sample)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="train")
    parser.add_argument("--shard-size", type=int, default=10000)
    args = parser.parse_args()

    convert_h5_to_wds(args.input, args.output, args.shard_size, args.prefix)