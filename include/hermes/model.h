#ifndef HERMES_MODEL_H
#define HERMES_MODEL_H

#include <vector>
#include <string>
#include <unordered_map>

#include "hermes/Tensor.h"

namespace hermes {

struct GPT2Config {
    size_t layers;
    size_t heads;
    size_t embed;
    size_t block_size;
    size_t vocab_size;
};

struct LayerNormWeights {
    Tensor ln_weight;
    Tensor ln_bias;
};

struct AttentionWeights {
    Tensor attn_weight;
    Tensor attn_bias;
    Tensor proj_weight;
    Tensor proj_bias;
};

struct MLPWeights {
    Tensor fc_weight;
    Tensor fc_bias;
    Tensor proj_weight;
    Tensor proj_bias;
};

struct TransformerWeights {
    LayerNormWeights ln1;
    AttentionWeights attn;
    LayerNormWeights ln2;
    MLPWeights mlp;
};

class MLP 
{
public:
    MLP(size_t in, size_t out, MLPWeights weights);
    Tensor forward(const Tensor& x) const;
private:
    size_t in_;
    size_t out_;
    MLPWeights weights_;
};

class CausalSelfAttention 
{
public:
    CausalSelfAttention(size_t heads, size_t embed, AttentionWeights weights);
    Tensor forward(const Tensor& x) const;
private:
    size_t heads_;
    size_t embed_;
    AttentionWeights weights_;
};

class Transformer
{
public:
    Transformer(size_t heads, size_t embed, TransformerWeights weights);
    Tensor forward(const Tensor& x) const;
private:
    size_t heads_;
    size_t embed_;
    LayerNormWeights ln1_;
    CausalSelfAttention attn_;
    LayerNormWeights ln2_;
    MLP mlp_;
};

class GPT2 
{
public:
    static GPT2 load(const std::string& weights);
    static GPT2 load(const std::unordered_map<std::string, Tensor>& state_dict, const GPT2Config& config);
    Tensor forward(const Tensor& x) const;
    std::vector<int> generate(const std::vector<int>& tokens, size_t max_length=1024) const;
private:
    size_t block_size_;
    size_t vocab_size_;
    Tensor wte_;
    Tensor wpe_;
    Tensor ln_f_weight_;
    Tensor ln_f_bias_;
    std::vector<Transformer> transformers_;

    GPT2(size_t block_size, size_t vocab_size, Tensor wte, Tensor wpe, Tensor ln_f_weight, Tensor ln_f_bias, std::vector<Transformer> transformers);
};

}

#endif

// https://github.com/karpathy/llm.c/blob/master/train_gpt2.py

// vocab_size is the total number of tokens
// T is number of tokens in input
// n_embed is embedding dimension
// n_heads is number of heads
// 12 blocks in small gpt2

// Tokens: (T)
// Embeddings: (T) -> (T, n_embed)

// Block

// Layer Norm: (T, n_embed) -> (T, n_embed)

// Attention
// qkv projection: (T, n_embed) -> (T, 3 * n_embed)
// qkv split: (T, 3 * n_embed) ->
// q = (T, n_embed)
// k = (T, n_embed)
// v = (T, n_embed)
// Reshape:
// q = (T, n_embed) -> (T, n_heads, n_embed / n_heads)
// k = (T, n_embed) -> (T, n_heads, n_embed / n_heads)
// v = (T, n_embed) -> (T, n_heads, n_embed / n_heads)
// Transpose:
// q = (T, n_heads, n_embed / n_heads) -> (n_heads, T, n_embed / n_heads)
// k = (T, n_heads, n_embed / n_heads) -> (n_heads, T, n_embed / n_heads)
// v = (T, n_heads, n_embed / n_heads) -> (n_heads, T, n_embed / n_heads)
// q * k^T = (n_heads, T, n_embed / n_heads) * (n_heads, n_embed / n_heads, T)
// q * k^T = (n_heads, T, n_embed / n_heads) * (n_heads, n_embed / n_heads, T) -> (n_heads, T, T)
// Divide by dimensionality factor: (n_heads, T, T) -> (n_heads, T, T)
// Mask along dim 2: (n_heads, T, T) -> (n_heads, T, T)
// Softmax along dim 2: (n_heads, T, T) -> (n_heads, T, T)
// A = (q * k^T) * v = (n_heads, T, T) * (n_heads, T, n_embed / n_heads) -> (n_heads, T, n_embed / n_heads)
// Transpose A: (n_heads, T, n_embed / n_heads) -> (T, n_heads, n_embed / n_heads)
// Make A row major in memory, contiguous
// Reshape A: (T, n_heads, n_embed / n_heads) -> (T, n_embed)
// Projection A: (T, n_embed) -> (T, n_embed)

// Residual: (T, n_embed) = (T, n_embed) + (T, n_embed)

// Layer Norm: (T, n_embed) -> (T, n_embed)
// MLP:
// (T, n_embed) -> (T, 4 * n_embed)
// Gelu: (T, 4 * n_embed) -> (T, 4 * n_embed)
// (T, 4 * n_embed) -> (T, n_embed)

// Residual: (T, n_embed) = (T, n_embed) + (T, n_embed)

// Layer Norm: (T, n_embed) -> (T, n_embed)

// LM_Head: (n_embed) -> (vocab_size), not a projection, its computing similarity with each vector in the embedding table to get the logit
// Only need the last token logits

// lets say q k v are all dim
// Then, we get input some vector which is (T, n_embed)
// (T_nembed) * Wq -> (T, dim) = Q, projection to dim
// (T_nembed) * Wk -> (T, dim) = K, projection to dim
// (T_nembed) * Wv -> (T, dim) = V, projection to dim
// Q * K^T = (T,T)
// query * key measures how close they are
// Then, scalar divide by dimensionality constant to get matrix B. First row of B is just how token 0 attends to every other token. In general, each row is how token i attends to the other tokens.
// Then do masking and softmax so that each token will only aggregate information from itself and tokens before it
// Then, do A = B * V to get attention scores. Each row of V is the value for a token, column is a single feature across tokens. so when you do a dot product of a row from B with a column from V, 
// thats the weighted sum of a single feature across each token. 
// The weighting is based on how token i cares about each other tokens. Then each row of A corresponds to a single token where each element is a weighted sum of that feature