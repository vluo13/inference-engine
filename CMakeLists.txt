# CMake version
cmake_minimum_required(VERSION 4.1)

# project name
project(inference-engine LANGUAGES CXX)

# set to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_SANITIZERS "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
if(ENABLE_SANITIZERS)
  set(SANITIZER_FLAGS
    -fsanitize=address,undefined
    -fsanitize-address-use-after-scope
    -fno-omit-frame-pointer
    -fno-sanitize-recover=all  
    -g)
  add_compile_options(${SANITIZER_FLAGS})
  add_link_options(-fsanitize=address,undefined)
endif()

set(HERMES_WARNINGS
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wconversion
  -Wsign-conversion
  -Wnull-dereference
  -Wdouble-promotion
  -Wformat=2
  -Wcast-align
  -Wunused
  -Wold-style-cast
  -Woverloaded-virtual
  -Wnon-virtual-dtor)

# create tensor library
add_library(hermes src/Tensor.cpp include/hermes/Tensor.h src/ops.cpp include/hermes/ops.h src/utils.cpp include/hermes/utils.h src/kernels.cpp include/hermes/kernels.h include/hermes/model.h src/model.cpp)
target_include_directories(hermes PUBLIC include)
target_compile_options(hermes PRIVATE ${HERMES_WARNINGS})

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
  tests/kernels_test.cpp
  tests/test_utils.h
  tests/model_test.cpp
)
target_link_libraries(
  hermes_test
  GTest::gtest_main
  hermes
)
target_compile_definitions(hermes_test PRIVATE TENSOR_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/")
target_compile_options(hermes_test PRIVATE ${HERMES_WARNINGS})

add_executable(
  generate
  test.cpp
)
target_link_libraries(
  generate
  hermes
)
target_compile_options(generate PRIVATE ${HERMES_WARNINGS})

include(GoogleTest)
gtest_discover_tests(hermes_test)