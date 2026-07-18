import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

torch.manual_seed(0)  # reproducible across runs

DATA = Path(__file__).parent.parent / 'data'
def export(arr, name):
    p = DATA / Path(name).name
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr.astype(np.float32))
    with open(p, 'wb') as f:
        f.write(np.int32(arr.ndim).tobytes())
        f.write(np.array(arr.shape, dtype=np.int32).tobytes())
        f.write(arr.tobytes())

# --- existing small hand-picked cases -------------------------------------
t = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32)
export(t.detach().numpy(), 'tensor3d.bin')
export(F.gelu(t, approximate='tanh').detach().numpy(), 'tensor3d_gelu.bin')

neg = torch.tensor([[[-2.0, -0.5, -1.0], [-0.75, -3.0, -0.1]],
                    [[-5.0, -0.25, -1.5], [-4.0, -0.8, -2.5]]], dtype=torch.float32)
zero = torch.zeros(2, 2, 3, dtype=torch.float32)
large = torch.tensor([[[8.0, 10.0, 20.0], [50.0, 100.0, 15.0]],
                      [[6.0, 30.0, 12.0], [9.0, 40.0, 25.0]]], dtype=torch.float32)

for name, tt in [("neg", neg), ("zero", zero), ("large", large)]:
    out = F.gelu(tt, approximate='tanh')
    export(tt.detach().numpy(), f'gelu_{name}_in.bin')
    export(out.detach().numpy(), f'gelu_{name}_out.bin')

# --- large random GELU cases, GPT-2 dims ----------------------------------
gelu_shapes = [
    (3072,),
    (64, 768),
    (128, 3072),
    (16, 197, 2304),
]
for i, shape in enumerate(gelu_shapes):
    x = torch.randn(*shape) * 3.0
    out = F.gelu(x, approximate='tanh')
    export(x.numpy(), f"gelu_rand_{i}_in.bin")
    export(out.numpy(), f'gelu_rand_{i}_out.bin')

# --- large random SOFTMAX cases (over last dim) ---------------------------
softmax_shapes = [
    (768,),
    (2304,),
    (64, 768),
    (12, 256, 256),   # attention-scores shaped (nh, T, T)
    (16, 512, 3072),
]
for i, shape in enumerate(softmax_shapes):
    x = torch.randn(*shape) * 5.0   # wide range -> exercises subtract-max
    out = F.softmax(x, dim=-1)
    export(x.numpy(), f'softmax_rand_{i}_in.bin')
    export(out.numpy(), f'softmax_rand_{i}_out.bin')

# --- large random LAYERNORM cases (over last dim) -------------------------
# weight/bias are learned per-feature; export them too so the C++ side loads
# real values, not identity.
layernorm_shapes = [
    (64, 768),
    (128, 768),
    (4, 512, 768),
    (16, 197, 3072),
]
for i, shape in enumerate(layernorm_shapes):
    C = shape[-1]
    x = torch.randn(*shape) * 2.0
    w = torch.randn(C)
    b = torch.randn(C)
    out = F.layer_norm(x, (C,), weight=w, bias=b, eps=1e-5)
    export(x.numpy(), f'layernorm_rand_{i}_in.bin')
    export(w.numpy(), f'layernorm_rand_{i}_w.bin')
    export(b.numpy(), f'layernorm_rand_{i}_b.bin')
    export(out.numpy(), f'layernorm_rand_{i}_out.bin')

# --- large random MATMUL cases, GPT-2 dims --------------------------------
# 2D and batched. Includes the packed-QKV projection (768 -> 2304) and the
# MLP up/down (768 -> 3072 -> 768) contraction dims.
matmul_shapes = [  # (M, K, N)
    (64, 768, 2304),
    (128, 768, 3072),
    (128, 3072, 768),
    (197, 768, 768),
    (512, 768, 2304),
]
for i, (M, K, N) in enumerate(matmul_shapes):
    a = torch.randn(M, K) * 0.1
    b = torch.randn(K, N) * 0.1
    out = a @ b
    export(a.numpy(), f'matmul_rand_{i}_a.bin')
    export(b.numpy(), f'matmul_rand_{i}_b.bin')
    export(out.numpy(), f'matmul_rand_{i}_out.bin')

bmm_shapes = [  # (B, M, K, N) — attention-shaped
    (12, 197, 64, 197),   # scores: Q @ K^T per head (hs=64)
    (12, 197, 197, 64),   # context: probs @ V
    (16, 512, 768, 768),
]
for i, (B, M, K, N) in enumerate(bmm_shapes):
    a = torch.randn(B, M, K) * 0.1
    b = torch.randn(B, K, N) * 0.1
    out = torch.bmm(a, b)
    export(a.numpy(), f'bmm_rand_{i}_a.bin')
    export(b.numpy(), f'bmm_rand_{i}_b.bin')
    export(out.numpy(), f'bmm_rand_{i}_out.bin')

print("done")