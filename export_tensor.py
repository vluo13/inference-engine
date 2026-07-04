import torch
import torch.nn.functional as F
import numpy as np

def export(arr, path):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    with open(path, 'wb') as f:
        f.write(np.int32(arr.ndim).tobytes())
        f.write(np.array(arr.shape, dtype=np.int32).tobytes())
        f.write(arr.tobytes())

t = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32)
out = F.gelu(t, approximate='tanh')

export(t.detach().numpy(), 'tests/tensors/tensor3d.bin')
export(out.detach().numpy(), 'tests/tensors/tensor3d_gelu.bin')

neg = torch.tensor([[[-2.0, -0.5, -1.0], [-0.75, -3.0, -0.1]],
                    [[-5.0, -0.25, -1.5], [-4.0, -0.8, -2.5]]], dtype=torch.float32)

# zeros — GELU(0) = 0
zero = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)

# large values — tanh saturates, GELU(x) ≈ x
large = torch.tensor([[[8.0, 10.0, 20.0], [50.0, 100.0, 15.0]],
                      [[6.0, 30.0, 12.0], [9.0, 40.0, 25.0]]], dtype=torch.float32)

for name, t in [("neg", neg), ("zero", zero), ("large", large)]:
    out = F.gelu(t, approximate='tanh')
    export(t.detach().numpy(), f'tests/tensors/gelu_{name}_in.bin')
    export(out.detach().numpy(), f'tests/tensors/gelu_{name}_out.bin')