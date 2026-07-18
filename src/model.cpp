#include "hermes/model.h"

#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <limits>
#include <fstream>
#include <utility>
#include <numeric>
#include <cassert>
#include <iostream>

#include "hermes/Tensor.h"
#include "hermes/kernels.h"
#include "hermes/ops.h"
#include "hermes/utils.h"

namespace hermes {

MLP::MLP(size_t in, size_t out, MLPWeights weights)
    : in_{in}, out_{out}, weights_{std::move(weights)} {}

Tensor MLP::forward(const Tensor& x) const {
    // except if dims don't align
    if (x.shape().back() != in_) {
        throw std::runtime_error("Incompatible dimensions with MLP forward");
    }

    Tensor out = matmul(x, weights_.fc_weight); // (T, in) * (in, out) -> (T, out)
    out = add(out, weights_.fc_bias); // (T, out) + (out,) -> (T, out)
    out = gelu(out);
    out = matmul(out, weights_.proj_weight); // (T, out) * (out, in) -> (T, in)
    return add(out, weights_.proj_bias);
}

CausalSelfAttention::CausalSelfAttention(size_t heads, size_t embed, AttentionWeights weights)
    : heads_{heads}, embed_{embed}, weights_{std::move(weights)} {}

Tensor CausalSelfAttention::forward(const Tensor& x) const {
    if (x.shape().back() != embed_) {
        throw std::runtime_error("Incompatible dimensions with Attention forward");
    }

    Tensor out = matmul(x, weights_.attn_weight); // (T, embed) * (embed, 3 * embed) -> (T, 3 * embed)
    out = add(out, weights_.attn_bias);
    size_t sliceDim = out.shape().size() - 1;
    Tensor q = out.slice(sliceDim, 0, embed_); // (T, n_embed)
    Tensor k = out.slice(sliceDim, embed_, 2 * embed_); // (T, n_embed)
    Tensor v = out.slice(sliceDim, 2 * embed_, 3 * embed_); // (T, n_embed)
    std::vector<size_t> newShape {x.shape()[0], heads_, embed_ / heads_};
    q = q.reshape(newShape).transpose(0, 1); // (n_heads, T, n_embed / n_heads)
    k = k.reshape(newShape).transpose(0, 1); // (n_heads, T, n_embed / n_heads)
    v = v.reshape(newShape).transpose(0, 1); // (n_heads, T, n_embed / n_heads)
    Tensor qkt = bmm(q, k.transpose(k.shape().size() - 1,  k.shape().size() - 2)); // (n_heads, T, T)
    qkt = div(qkt, static_cast<float>(std::sqrt(embed_ / heads_)));
    // mask each head
    size_t head_stride = Tensor::numel(qkt.shape()) / heads_;
    size_t tokens = qkt.shape().back();
    for (size_t i = 0; i < heads_; ++i) {
        float* scores = qkt.dataPtr() + (i * head_stride);
        for (size_t row = 0; row < tokens; ++row) {
            for (size_t col = 0; col < tokens; ++col) {
                if (col > row) {
                    scores[row*tokens + col] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }
    Tensor attn_weights = softmax(qkt); // (n_heads, T, T)
    Tensor attn_output = bmm(attn_weights, v); // (n_heads, T, T) * (n_heads, T, n_embed / n_heads) -> (n_heads, T, n_embed / n_heads)
    attn_output = attn_output.transpose(0, 1); // (T, n_heads, n_embed / n_heads)
    attn_output = attn_output.reshape(std::vector<size_t>{tokens, embed_}); // (T, n_embed)
    attn_output = matmul(attn_output, weights_.proj_weight); // verify that proj weight is in out (T, n_embed)
    attn_output = add(attn_output, weights_.proj_bias);
    assert(attn_output.shape() == x.shape());
    return attn_output; // (T, n_embed)
}

Transformer::Transformer(size_t heads, size_t embed, TransformerWeights weights)
    : heads_{heads}, embed_{embed} 
    , ln1_{std::move(weights.ln1)}
    , attn_{heads_, embed, std::move(weights.attn)}
    , ln2_{std::move(weights.ln2)}
    , mlp_{embed, embed, std::move(weights.mlp)} {}

Tensor Transformer::forward(const Tensor& x) const {
    if (x.shape().back() != embed_) {
        throw std::runtime_error("Incompatible dimensions with transformer forward");
    }

    Tensor out = layerNorm(x, ln1_.ln_weight, ln1_.ln_bias);

    assert(out.shape() == x.shape());
    out = add(x, attn_.forward(out));
    assert(out.shape() == x.shape());
    out = add(out, mlp_.forward(layerNorm(out, ln2_.ln_weight, ln2_.ln_bias)));

    return out;
}

GPT2 GPT2::load(const std::string& weights) {
    std::ifstream reader(weights, std::ios_base::binary);
    if (!reader) throw std::runtime_error("cannot open " + weights);

    size_t header_size;
    reader.read(reinterpret_cast<char*>(&header_size), sizeof(size_t));
    size_t layers;
    reader.read(reinterpret_cast<char*>(&layers), sizeof(size_t));
    size_t heads;
    reader.read(reinterpret_cast<char*>(&heads), sizeof(size_t));
    size_t embed;
    reader.read(reinterpret_cast<char*>(&embed), sizeof(size_t));
    size_t block_size;
    reader.read(reinterpret_cast<char*>(&block_size), sizeof(size_t));
    size_t vocab_size;
    reader.read(reinterpret_cast<char*>(&vocab_size), sizeof(size_t));

    std::unordered_map<std::string, Tensor> state_dict;
    std::vector<std::pair<std::string, size_t>> tensor_sizes;
    size_t fpos = 48;
    while (fpos < header_size) {
        size_t name_length;
        reader.read(reinterpret_cast<char*>(&name_length), sizeof(size_t));
        fpos += sizeof(size_t);

        std::string name(name_length, '\0');
        reader.read(name.data(), static_cast<std::streamsize>(name_length));
        fpos += name_length;

        size_t ndims;
        reader.read(reinterpret_cast<char*>(&ndims), sizeof(size_t));
        fpos += sizeof(size_t);

        std::vector<size_t> shape(ndims);
        reader.read(reinterpret_cast<char*>(shape.data()), static_cast<std::streamsize>(ndims * sizeof(size_t)));
        fpos += ndims * sizeof(size_t);

        size_t offset;
        reader.read(reinterpret_cast<char*>(&offset), sizeof(size_t));
        fpos += sizeof(size_t);

        state_dict.emplace(name, Tensor{shape}); // try empalce/piecewise ctor better?
        tensor_sizes.push_back(std::make_pair(name, sizeof(float) * Tensor::numel(shape)));
    }

    for (const auto& [name, nbytes] : tensor_sizes) {
        reader.read(reinterpret_cast<char*>(state_dict.at(name).dataPtr()), static_cast<std::streamsize>(nbytes));
    }
    
    return load(state_dict, GPT2Config {layers, heads, embed, block_size, vocab_size});
}

GPT2 GPT2::load(const std::unordered_map<std::string, Tensor>& state_dict, const GPT2Config& config) {
    std::vector<Transformer> transformers;
    for (size_t i = 0; i < config.layers; ++i) {
        LayerNormWeights ln1 {state_dict.at("transformer.h." + std::to_string(i) + ".ln_1.weight"), state_dict.at("transformer.h." + std::to_string(i) + ".ln_1.bias")};
        AttentionWeights attn {state_dict.at("transformer.h." + std::to_string(i) + ".attn.c_attn.weight"), state_dict.at("transformer.h." + std::to_string(i) + ".attn.c_attn.bias"), 
            state_dict.at("transformer.h." + std::to_string(i) + ".attn.c_proj.weight"), state_dict.at("transformer.h." + std::to_string(i) + ".attn.c_proj.bias")};
        LayerNormWeights ln2 {state_dict.at("transformer.h." + std::to_string(i) + ".ln_2.weight"), state_dict.at("transformer.h." + std::to_string(i) + ".ln_2.bias")};
        MLPWeights mlp {state_dict.at("transformer.h." + std::to_string(i) + ".mlp.c_fc.weight"), state_dict.at("transformer.h." + std::to_string(i) + ".mlp.c_fc.bias"), 
            state_dict.at("transformer.h." + std::to_string(i) + ".mlp.c_proj.weight"), state_dict.at("transformer.h." + std::to_string(i) + ".mlp.c_proj.bias")};

        transformers.push_back(Transformer(config.heads, config.embed, TransformerWeights {ln1, attn, ln2, mlp}));
    }
    return GPT2(config.block_size, config.vocab_size, std::move(state_dict.at("transformer.wte.weight")), std::move(state_dict.at("transformer.wpe.weight")), 
                std::move(state_dict.at("transformer.ln_f.weight")), std::move(state_dict.at("transformer.ln_f.bias")), std::move(transformers));
}

Tensor GPT2::forward(const Tensor& x) const {
    if (x.shape()[0] > 1024) {
        throw std::runtime_error("Number of tokens exceeds block size");
    }
    Tensor tok_emb = embedding(x, wte_);
    Tensor pos {x.shape()};
    std::iota(pos.dataPtr(), pos.dataPtr() + x.shape()[0], 0);
    Tensor pos_emb = embedding(pos, wpe_);
    Tensor out = add(tok_emb, pos_emb);

    for (Transformer t : transformers_) {
        out = t.forward(out);
    }
    out = layerNorm(out, ln_f_weight_, ln_f_bias_);
    return matmul(out, wte_.transpose(0,1)); // (T, 768) * (768, 50257) -> (T, 50257)
}

std::vector<int> GPT2::generate(const std::vector<int>& tokens, size_t max_length) const{
    std::vector<int> output = tokens;
    while (output.size() < block_size_ && output.size() < max_length) {
        // convert tokens to a float Tensor 
        Tensor tokTensor {std::vector<size_t>{output.size()}};
        for (size_t i = 0; i < output.size(); ++i) {
            tokTensor.dataPtr()[i] = static_cast<float>(output[i]);
        }
        // pass to forward
        Tensor out = forward(tokTensor);

        // take the last dim
        Tensor lastDim = out.slice(0, output.size()-1, output.size());

        // take the max logit as the answer and append to output
        size_t maxIdx = 0;
        float maxLogit = lastDim.dataPtr()[0];
        for (size_t i = 1; i < vocab_size_; ++i) {
            if (lastDim.dataPtr()[i] > maxLogit) {
                maxIdx = i;
                maxLogit = lastDim.dataPtr()[i];
            }
        }
        output.push_back(static_cast<int>(maxIdx));
        std::cout << maxIdx << ",";
        std::cout.flush(); 
    }
    return output;
}

GPT2::GPT2(size_t block_size, size_t vocab_size, Tensor wte, Tensor wpe, Tensor ln_f_weight, Tensor ln_f_bias, std::vector<Transformer> transformers) 
    : block_size_{block_size}
    , vocab_size_{vocab_size}
    , wte_{std::move(wte)} 
    , wpe_{std::move(wpe)}
    , ln_f_weight_{std::move(ln_f_weight)}
    , ln_f_bias_{std::move(ln_f_bias)}
    , transformers_{std::move(transformers)} {}

}

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
// A = softmax(q * k^T) * v = (n_heads, T, T) * (n_heads, T, n_embed / n_heads) -> (n_heads, T, n_embed / n_heads)
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