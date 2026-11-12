#!/usr/bin/env python3
"""
KidLM-5M Offline Training Script
Trains Llama-style decoder-only Transformer on KidLM-corpus Parquet shards.
Runs 100% offline with zero external Hugging Face / internet calls.
"""

import os
import glob
import math
import time
import argparse
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset

from model import ModelArgs, Transformer, export_llama2_c_q8


class ShardedKidLMDataset(Dataset):
    def __init__(self, data_files: List[str], tokenizer: Tokenizer, seq_len: int = 256):
        self.seq_len = seq_len
        print(f"Loading {len(data_files)} Parquet shards offline...")
        ds = load_dataset("parquet", data_files=data_files, split="train")

        print("Tokenizing and packing tokens into sequences...")
        all_token_ids = []
        # Pre-tokenize all texts
        batch_size = 5000
        for i in range(0, len(ds), batch_size):
            batch_texts = ds[i : i + batch_size]["text"]
            encodings = tokenizer.encode_batch(batch_texts)
            for enc in encodings:
                all_token_ids.extend(enc.ids)
                all_token_ids.append(2) # </s> EOS token

        total_tokens = len(all_token_ids)
        print(f"Total dataset tokens: {total_tokens:,}")

        # Chunk into (seq_len + 1) blocks for next-token prediction
        num_chunks = total_tokens // (seq_len + 1)
        truncated_len = num_chunks * (seq_len + 1)
        self.data = torch.tensor(all_token_ids[:truncated_len], dtype=torch.long).view(num_chunks, seq_len + 1)
        print(f"Packed into {num_chunks:,} training sequences of length {seq_len}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def get_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Train KidLM-5M Offline")
    parser.add_argument("--data_dir", type=str, default=os.path.join(script_dir, "kidlm_shards"))
    parser.add_argument("--tokenizer_path", type=str, default=os.path.join(script_dir, "kidlm_tokenizer", "tokenizer.json"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "kidlm_checkpoints"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_accum", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on compute device: {device}")

    # 1. Load Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # 2. Load Shards
    shard_files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    if not shard_files:
        raise FileNotFoundError(f"No parquet shards found in {args.data_dir}")

    dataset = ShardedKidLMDataset(shard_files, tokenizer, seq_len=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 3. Model
    model_args = ModelArgs(
        dim=256,
        n_layers=6,
        n_heads=8,
        vocab_size=4096,
        hidden_dim=688,
        max_seq_len=256,
        tie_embeddings=True
    )
    model = Transformer(model_args).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # 4. Optimizer & Schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = (len(loader) // args.grad_accum) * args.epochs

    print(f"\nStarting training: {args.epochs} epoch(s), {total_steps} optimizer steps...")
    step = 0
    start_time = time.time()

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            lr = get_lr(step, total_steps, args.warmup_steps, args.max_lr, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            logits, loss = model(x, y)
            loss = loss / args.grad_accum
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % 100 == 0:
                    elapsed = time.time() - start_time
                    tok_per_sec = (step * args.batch_size * args.grad_accum * 256) / elapsed
                    print(f"Step {step}/{total_steps} | Loss: {loss.item() * args.grad_accum:.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:.0f} tok/s")

        print(f"Epoch {epoch + 1}/{args.epochs} completed. Average Loss: {epoch_loss / len(loader):.4f}")

    # 5. Export Final Weights
    pt_path = os.path.join(args.output_dir, "kidlm_5m.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"\nSaved PyTorch checkpoint to {pt_path}")

    # Export INT8 binary for ESP32-S3 Flash partition
    q8_path = os.path.join(args.output_dir, "kidlm_5m_q8.bin")
    export_llama2_c_q8(model, q8_path)
    q8_size_mb = os.path.getsize(q8_path) / (1024 * 1024)
    print(f"Exported INT8 binary to {q8_path} ({q8_size_mb:.2f} MB)")
    print("Ready to flash directly to XIAO ESP32-S3 at offset 0x190000!")


if __name__ == "__main__":
    train()
