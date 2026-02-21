# CMake version
cmake_minimum_required(VERSION 4.1)

# project name
project(inference-engine LANGUAGES CXX)

# set to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# create tensor library
# add_library(tensor src/Tensor.cpp include/tensor/Tensor.h)
# target_include_directories(tensor PUBLIC include)

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
  hello_test
  test/hello_test.cpp
)
target_link_libraries(
  hello_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(hello_test)