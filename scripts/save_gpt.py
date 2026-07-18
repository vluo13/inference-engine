import numpy as np
import os
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()

size = 48
tensors = []
for k, v in sd_hf.items():
    print(k, v.shape)
    name = k.encode('utf-8')
    size += 8 # length of the name in bytes
    size += len(name) # each is 1 byte
    size += 8 # ndims
    size += 8 * len(v.shape) # each gpt weight dimension should fit within size_t
    size += 8 # int64 offset
    v_numpy = v.detach().numpy()
    v_contiguous = np.ascontiguousarray(v_numpy.astype(np.float32))
    tensors.append((name, v_contiguous.shape, v_contiguous)) # name, shape, data

# write the stuff at size offset
offset = size
print(size)
with open("weights/weights.bin", "wb") as f:
    f.write(np.int64(size).tobytes())
    f.write(np.int64(12).tobytes())
    f.write(np.int64(12).tobytes())
    f.write(np.int64(768).tobytes())
    f.write(np.int64(1024).tobytes())
    f.write(np.int64(50257).tobytes())
    for name, shape, data in tensors:
        # name
        f.write(np.int64(len(name)).tobytes())
        print(len(name), name)
        f.write(name)

        # shape
        f.write(np.int64(len(shape)).tobytes())
        f.write(np.array(shape, dtype=np.int64).tobytes())
        
        # offset
        f.write(np.int64(offset).tobytes())
        offset += data.nbytes
    for _, _, data in tensors:
        f.write(data.tobytes())

assert os.path.getsize("weights/weights.bin")==offset

print("Done")