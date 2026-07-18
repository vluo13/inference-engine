import torch
from torch.nn import functional as F
import tiktoken
from transformers import GPT2LMHeadModel

# model = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
# model.eval()
# torch.manual_seed(42)
# tokens = [15496, 11, 314, 1101, 257, 3303, 2746, 11] # "Hello, I'm a language model,"
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(5, 1) # (5, 8)
# x = tokens
# # print(x)

# # generate!
# # while x.size(1) < 30: # max_length=30
# #     # forward the model to get the logitsf
# #     with torch.no_grad():
# #         logits = model(x)[0] # (B, T, vocab_size)
# #         # take the logits at the last position
# #         logits = logits[:, -1, :] # (B, vocab_size)
# #         # get the probabilities
# #         probs = F.softmax(logits, dim=-1)
# #         # do top-k sampling of 50 (huggingface pipeline default)
# #         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
# #         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
# #         # select a token from the top-k probabilities
# #         # note: multinomial does not demand the input to sum to 1
# #         ix = torch.multinomial(topk_probs, 1) # (B, 1)
# #         # gather the corresponding indices
# #         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
# #         # append to the sequence
# #         x = torch.cat((x, xcol), dim=1)
# while x.size(1) < 30: # max_length=30
#     with torch.no_grad():
#         logits = model(x)[0] # (B, T, vocab_size)
#         logits = logits[:, -1, :] # (B, vocab_size)
#         top = torch.topk(logits, 5)
#         # print("top5 values:", top.values.tolist())
#         # print("top5 indices:", top.indices.tolist())
#         xcol = torch.argmax(logits, dim=-1, keepdim=True) # (B, 1)
#         x = torch.cat((x, xcol), dim=1)

# # print the generated text
enc = tiktoken.get_encoding('gpt2')
print(enc.encode("Hello, I'm a language model"))
print(enc.decode([407,257,8300,3303,13,314,1101]))
# for i in range(1):
#     tokens = x[i, :30].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)

# print("\n")
# print(enc.decode([407,257,8300,3303,13,314,1101,257,3303,2746,13,314,1101,257,3303,2746,13,314,1101,257,3303,2746,13]))
