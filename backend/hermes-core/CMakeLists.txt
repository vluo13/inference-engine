# CMake version
cmake_minimum_required(VERSION 3.28)

# Define the project
project(Hermes_Core LANGUAGES CXX)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the library target
add_library(hermes_core STATIC
    src/Tensor.cpp
    #src/engine.cpp
    #src/KVcache.cpp
    #src/kernels.cpp
)

# Add include directories
target_include_directories(hermes_core PUBLIC include)