import numpy as np
import torch
from transformers import GPT2LMHeadModel
from pathlib import Path

OUT_DIR = str(Path(__file__).parent.parent / 'data')

model = GPT2LMHeadModel.from_pretrained("gpt2")  # 124M
model.eval()


def export(arr, path):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    with open(path, "wb") as f:
        f.write(np.int32(arr.ndim).tobytes())
        f.write(np.array(arr.shape, dtype=np.int32).tobytes())
        f.write(arr.tobytes())


def make_seq(T):
    # Must match makeSeq() in the C++ test exactly: (i * 7919) % 50257.
    # 7919 is prime, so ids spread across the vocab without repeating.
    return [(i * 7919) % 50257 for i in range(T)]


# Sequences vary T, which is what the causal mask, position embeddings, and
# attention shapes depend on. Token content doesn't affect control flow.
cases = {
    "prompt8":  [15496, 11, 314, 1101, 257, 3303, 2746, 11],  # "Hello, I'm a language model,"
    "single":   [15496],           # T=1: degenerate mask, first wpe row only
    "long300":  make_seq(300),     # large T×T mask, deep into wpe
    "max1024":  make_seq(1024),    # block_size boundary (wpe has exactly 1024 rows)
}

for name, tokens in cases.items():
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # (1, T)
    with torch.no_grad():
        logits = model(x)[0]        # (1, T, vocab_size)
        logits = logits[0]          # (T, vocab_size) — drop batch dim only
    export(logits.numpy(), f"{OUT_DIR}/logits_{name}.bin")
    path = f"{OUT_DIR}/logits_{name}.bin"

    print(f"{name:10s} T={len(tokens):5d}  logits{tuple(logits.shape)}  -> {path}")