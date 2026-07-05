# CMake version
cmake_minimum_required(VERSION 4.1)

# project name
project(inference-engine LANGUAGES CXX)

# set to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-pedantic)

# create tensor library
add_library(hermes src/Tensor.cpp include/hermes/Tensor.h src/ops.cpp include/hermes/ops.h src/utils.cpp include/hermes/utils.h src/kernels.cpp include/hermes/kernels.h)
target_include_directories(hermes PUBLIC include)

# Download GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Testing
enable_testing()

add_executable(
  hermes_test
  tests/tensor_test.cpp
  tests/ops_test.cpp
  tests/utils_test.cpp
  tests/kernels_test.cpp
  tests/test_utils.h
)
target_link_libraries(
  hermes_test
  GTest::gtest_main
  hermes
)

include(GoogleTest)
gtest_discover_tests(hermes_test)