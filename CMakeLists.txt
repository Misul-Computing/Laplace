cmake_minimum_required(VERSION 3.16)
project(laplace CXX OBJCXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    message(FATAL_ERROR "Laplace requires a native Apple Silicon macOS toolchain")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OSX_ARCHITECTURES arm64)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LAPLACE_NATIVE "Build for the Apple Silicon host" ON)
option(LAPLACE_LTO "Enable link-time optimization" OFF)

add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
if(LAPLACE_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=native)
endif()
if(LAPLACE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

set_source_files_properties(src/matmul_simd.cpp PROPERTIES
    COMPILE_OPTIONS "-march=armv8.5-a+dotprod+fp16+i8mm+bf16")
set_source_files_properties(src/metal.mm PROPERTIES LANGUAGE OBJCXX)

set(LAPLACE_CORE_SOURCES
    src/artifact_index.cpp
    src/artifact_set.cpp
    src/canonical_metal.cpp
    src/column_grouped_q4.cpp
    src/compat_rule.cpp
    src/compatibility_report.cpp
    src/execution_plan.cpp
    src/generation_loop.cpp
    src/gguf.cpp
    src/gguf_import.cpp
    src/matmul.cpp
    src/matmul_simd.cpp
    src/metal.mm
    src/mmap.cpp
    src/mlx_package.cpp
    src/safetensors.cpp
    src/safetensors_adapter.cpp
    src/sampler.cpp
    src/semantic_model.cpp
    src/sparse_selector_coeff.cpp
    src/tokenizer.cpp
)

set(LAPLACE_APPLE_FRAMEWORKS
    "-framework IOKit"
    "-framework CoreFoundation"
    "-framework Metal"
    "-framework Foundation")

add_library(laplace_core STATIC ${LAPLACE_CORE_SOURCES})
target_include_directories(laplace_core PUBLIC src)
target_link_libraries(laplace_core PUBLIC ${LAPLACE_APPLE_FRAMEWORKS})

add_executable(laplace src/main.cpp)
target_link_libraries(laplace PRIVATE laplace_core)

enable_testing()

function(laplace_contract_test target)
    add_executable(${target} tests/${target}.cpp)
    target_include_directories(${target} PRIVATE tests)
    target_link_libraries(${target} PRIVATE laplace_core)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

laplace_contract_test(test_artifact_index)
laplace_contract_test(test_compatibility_report)
laplace_contract_test(test_semantic_model)
laplace_contract_test(test_compat_rule)
laplace_contract_test(test_execution_plan)
laplace_contract_test(test_gguf_import)
laplace_contract_test(test_safetensors_adapter)
laplace_contract_test(test_mlx_package)
laplace_contract_test(test_generation_loop)

add_executable(test_artifact_set tests/test_artifact_set.cpp
    src/artifact_set.cpp src/compatibility_report.cpp src/mmap.cpp)
target_compile_definitions(test_artifact_set PRIVATE LAPLACE_ARTIFACT_SET_TESTING=1)
target_include_directories(test_artifact_set PRIVATE src tests)
target_link_libraries(test_artifact_set PRIVATE ${LAPLACE_APPLE_FRAMEWORKS})
add_test(NAME test_artifact_set COMMAND test_artifact_set)

add_executable(test_cli tests/test_cli.cpp)
target_include_directories(test_cli PRIVATE tests)
target_compile_definitions(test_cli PRIVATE
    LAPLACE_BINARY_PATH="${CMAKE_BINARY_DIR}/laplace")
add_dependencies(test_cli laplace)
add_test(NAME test_cli COMMAND test_cli)

# This test target is a device witness. It is not in CTest because sandboxed
# macOS runners can compile it without having access to a Metal device.
add_executable(test_canonical_metal EXCLUDE_FROM_ALL
    tests/test_canonical_metal.cpp
    tests/reference_fp32.cpp
    ${LAPLACE_CORE_SOURCES})
target_include_directories(test_canonical_metal PRIVATE src tests)
target_compile_definitions(test_canonical_metal PRIVATE LAPLACE_METAL_TESTING=1)
target_link_libraries(test_canonical_metal PRIVATE ${LAPLACE_APPLE_FRAMEWORKS})
