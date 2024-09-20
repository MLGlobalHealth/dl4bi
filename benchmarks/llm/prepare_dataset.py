#!/usr/bin/env python3
"""Build datasets based on Karpathy's [scripts](https://github.com/karpathy/nanoGPT/tree/master/data)."""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import requests
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def openwebtext(hf_cache_dir: str):
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(
        "openwebtext",
        cache_dir=hf_cache_dir,
        num_proc=os.cpu_count() // 2,
    )
    ds = ds["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    ds["valid"] = ds.pop("test")  # rename test -> valid

    def f_enc(x):
        ids = enc.encode_ordinary(x["text"]) + [enc.eot_token]
        return {"ids": ids, "len": len(ids)}

    tokenized = ds.map(
        f_enc,
        remove_columns=["text"],
        desc="Tokenizing the splits",
        num_proc=os.cpu_count() // 2,
    )

    path = Path("cache/openwebtext")
    path.mkdir(parents=True, exist_ok=True)
    for name, part in tokenized.items():
        n = np.sum(part["len"], dtype=np.uint64)
        arr = np.zeros((n,), np.uint16)
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"Writing to {path}"):
            batch = part.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("pandas")  # As of 2024-07-22, doesn't work with NumPy 2.0
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        np.save(path / f"{name}.npy", arr)


def shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = Path("cache/shakespeare")
    path.mkdir(parents=True, exist_ok=True)
    input_path = path / "input.txt"
    data = fetch(url, input_path)
    _shakespeare_full(data, path / "full")
    _shakespeare_char(data, path / "char")


def _shakespeare_full(data, path: Path):
    n_train = int(0.9 * len(data))
    train_data, valid_data = data[:n_train], data[n_train:]
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    valid_ids = enc.encode_ordinary(valid_data)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "train.npy", train_ids)
    np.save(path / "valid.npy", valid_ids)
    print("\n === Shakespeare Full Dataset === \n")
    print(f"Train data has {len(train_ids):,} tokens")
    print(f"Valid data has {len(valid_ids):,} tokens")


def _shakespeare_char(data, path: Path):
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    enc = lambda x: [stoi[c] for c in x]
    n_train = int(0.9 * len(data))
    train_data, valid_data = data[:n_train], data[n_train:]
    train_ids, valid_ids = enc(train_data), enc(valid_data)
    meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "train.npy", train_ids)
    np.save(path / "valid.npy", valid_ids)
    with open(path / "char.pkl", "wb") as f:
        pickle.dump(meta, f)
    print("\n === Shakespeare Char Dataset === \n")
    print(f"Unique chars: {''.join(chars)}")
    print(f"Vocab size: {vocab_size}")
    print(f"Train data has {len(train_ids):,} tokens")
    print(f"Valid data has {len(valid_ids):,} tokens")


def fetch(url: str, path: Path):
    if path.exists():
        with open(path) as f:
            return f.read()
    with open(path, "w") as f:
        data = requests.get(url).text
        f.write(data)
    return data


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--hf_cache_dir",
        default=os.path.expanduser("~/hf"),
        help="Where to cache HuggingFace datasets.",
    )
    parser.add_argument(
        "-o",
        "--openwebtext",
        action="store_true",
        help="Process Openwebtext dataset.",
    )
    parser.add_argument(
        "-s",
        "--shakespeare",
        action="store_true",
        help="Process Shakespeare dataset.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    if args.openwebtext:
        openwebtext(args.hf_cache_dir)
    if args.shakespeare:
        shakespeare()
