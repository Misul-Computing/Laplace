cmake_minimum_required(VERSION 3.16)
# Project: Laplace - the Apple Silicon inference engine.
# See README.md, AGENTS.md, research/REPORT.md.
project(laplace CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LAPLACE_NATIVE "Build with -march=native (use only on the machine that will run it)" OFF)
option(LAPLACE_OPENMP "Use OpenMP for parallel matmul" ON)
option(LAPLACE_KV_CAPTURE "Build the opt-in LaplaceKV research trace hook" OFF)
if(LAPLACE_KV_CAPTURE)
    add_compile_definitions(LAPLACE_KV_CAPTURE=1)
endif()

# Warnings
add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)

# SIMD kernels are compiled with Apple Silicon dot-product extensions.
set(LAPLACE_SIMD_FLAGS -march=armv8.5-a+dotprod+fp16+i8mm+bf16)
set_source_files_properties(src/matmul_simd.cpp PROPERTIES COMPILE_OPTIONS "${LAPLACE_SIMD_FLAGS}")
set_source_files_properties(src/laplace_kv.cpp PROPERTIES COMPILE_OPTIONS "${LAPLACE_SIMD_FLAGS}")

if(LAPLACE_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=native)
endif()

# LTO: lets the optimizer see across translation units, which helps inline
# the matmul dispatch and NEON kernels into the arch hot paths.
option(LAPLACE_LTO "Enable link-time optimization" OFF)
if(LAPLACE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# OpenMP
set(LAPLACE_OMP_TARGET "")
if(LAPLACE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(LAPLACE_OMP_TARGET OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP requested but not found")
    endif()
endif()

set(LAPLACE_SOURCES
    src/main.cpp
    src/mmap.cpp
    src/gguf.cpp
    src/safetensors.cpp
    src/kvcache.cpp
    src/matmul.cpp
    src/matmul_simd.cpp
    src/model.cpp
    src/ops.cpp
    src/laplace_arch.cpp
    src/arch_qwen3next.cpp
    src/arch_llama.cpp
    src/arch_phi3.cpp
    src/arch_gemma4.cpp
    src/ssm.cpp
    src/tokenizer.cpp
    src/sampler.cpp
    src/laplace_kv.cpp
    src/laplace_moe.cpp
)

add_executable(laplace ${LAPLACE_SOURCES})
target_include_directories(laplace PRIVATE src)
if(LAPLACE_OMP_TARGET)
    target_link_libraries(laplace PRIVATE ${LAPLACE_OMP_TARGET})
endif()

# Targets that pull in threadpool.h need IOKit + CoreFoundation on Apple
# for power-aware scheduling (battery detection) and os_sync_wait_on_address.
if(APPLE)
    set(LAPLACE_ACCELERATE_FW "-framework Accelerate")
    set(LAPLACE_APPLE_FW "-framework IOKit" "-framework CoreFoundation" "-framework Metal" "-framework Foundation" ${LAPLACE_ACCELERATE_FW})
    target_link_libraries(laplace PRIVATE ${LAPLACE_APPLE_FW})
    set_source_files_properties(src/metal.mm PROPERTIES LANGUAGE OBJCXX)
    target_sources(laplace PRIVATE src/metal.mm)
endif()

# Tests (self-contained; no model file required)
enable_testing()

add_executable(test_gguf tests/test_gguf.cpp src/gguf.cpp src/mmap.cpp)
target_include_directories(test_gguf PRIVATE src)
add_test(NAME test_gguf COMMAND test_gguf)

add_executable(test_safetensors tests/test_safetensors.cpp src/safetensors.cpp src/mmap.cpp)
target_include_directories(test_safetensors PRIVATE src)
add_test(NAME test_safetensors COMMAND test_safetensors)

add_executable(test_matmul tests/test_matmul.cpp src/matmul.cpp src/matmul_simd.cpp)
target_include_directories(test_matmul PRIVATE src)
if(LAPLACE_OMP_TARGET)
    target_link_libraries(test_matmul PRIVATE ${LAPLACE_OMP_TARGET})
endif()
if(APPLE)
    target_link_libraries(test_matmul PRIVATE ${LAPLACE_APPLE_FW})
    target_sources(test_matmul PRIVATE src/metal.mm)
    set_source_files_properties(src/metal.mm PROPERTIES LANGUAGE OBJCXX)
endif()
add_test(NAME test_matmul COMMAND test_matmul)
set_tests_properties(test_matmul PROPERTIES ENVIRONMENT "LAPLACE_NOMETAL=1")
# Same kernel tests, scalar fallback path forced.
add_test(NAME test_matmul_scalar COMMAND test_matmul)
set_tests_properties(test_matmul_scalar PROPERTIES ENVIRONMENT "LAPLACE_NOSIMD=1;LAPLACE_NOMETAL=1")

if(APPLE)
    add_executable(test_metal tests/test_metal.cpp src/matmul.cpp src/matmul_simd.cpp src/metal.mm)
    target_include_directories(test_metal PRIVATE src)
    target_link_libraries(test_metal PRIVATE "-framework Metal" "-framework Foundation" "-framework IOKit" "-framework CoreFoundation")
    add_test(NAME test_metal COMMAND test_metal)

    add_executable(bench_cpu_gpu tests/bench_cpu_gpu.cpp src/matmul.cpp src/matmul_simd.cpp src/metal.mm)
    target_include_directories(bench_cpu_gpu PRIVATE src)
    target_link_libraries(bench_cpu_gpu PRIVATE "-framework Metal" "-framework Foundation" "-framework IOKit" "-framework CoreFoundation")
endif()

add_executable(test_kvcache tests/test_kvcache.cpp
    src/kvcache.cpp src/laplace_kv.cpp src/ops.cpp)
target_include_directories(test_kvcache PRIVATE src)
if(APPLE)
    target_link_libraries(test_kvcache PRIVATE ${LAPLACE_ACCELERATE_FW})
endif()
add_test(NAME test_kvcache COMMAND test_kvcache)

add_executable(test_sampler tests/test_sampler.cpp src/sampler.cpp)
target_include_directories(test_sampler PRIVATE src)
add_test(NAME test_sampler COMMAND test_sampler)

add_executable(test_tokenizer tests/test_tokenizer.cpp src/tokenizer.cpp src/gguf.cpp src/mmap.cpp)
target_include_directories(test_tokenizer PRIVATE src)
add_test(NAME test_tokenizer COMMAND test_tokenizer)

add_executable(test_arch tests/test_arch.cpp
    src/ops.cpp
    src/laplace_arch.cpp
    src/arch_qwen3next.cpp
    src/arch_llama.cpp
    src/arch_phi3.cpp
    src/arch_gemma4.cpp
    src/laplace_moe.cpp
    src/matmul.cpp
    src/matmul_simd.cpp
    src/ssm.cpp
    src/kvcache.cpp
    src/laplace_kv.cpp
    src/gguf.cpp
    src/mmap.cpp)
target_include_directories(test_arch PRIVATE src)
if(LAPLACE_OMP_TARGET)
    target_link_libraries(test_arch PRIVATE ${LAPLACE_OMP_TARGET})
endif()
if(APPLE)
    target_link_libraries(test_arch PRIVATE ${LAPLACE_APPLE_FW})
    target_sources(test_arch PRIVATE src/metal.mm)
    set_source_files_properties(src/metal.mm PROPERTIES LANGUAGE OBJCXX)
endif()
add_test(NAME test_arch COMMAND test_arch)

add_executable(test_laplace_kv tests/test_laplace_kv.cpp
    src/laplace_kv.cpp src/ops.cpp)
target_include_directories(test_laplace_kv PRIVATE src tests)
if(APPLE)
    target_link_libraries(test_laplace_kv PRIVATE ${LAPLACE_ACCELERATE_FW})
endif()
add_test(NAME test_laplace_kv COMMAND test_laplace_kv)

add_executable(bench_kvcache tests/bench_kvcache.cpp
    src/laplace_kv.cpp src/kvcache.cpp src/ops.cpp)
target_include_directories(bench_kvcache PRIVATE src)
if(APPLE)
    target_link_libraries(bench_kvcache PRIVATE ${LAPLACE_ACCELERATE_FW})
endif()

# Tools
add_executable(dump_meta tools/dump_meta.cpp src/gguf.cpp src/mmap.cpp)
target_include_directories(dump_meta PRIVATE src)
