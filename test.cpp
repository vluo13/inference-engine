#include <iostream>

#include "hermes/model.h"

int main() {
    hermes::GPT2 model = hermes::GPT2::load("/Users/admin/Desktop/inference-engine/weights/weights.bin");
    std::vector<int> tokens {15496, 11, 314, 1101, 257, 3303, 2746, 11};
    model.generate(tokens, 15);
    return 0;
}