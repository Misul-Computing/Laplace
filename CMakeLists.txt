cmake_minimum_required(VERSION 3.20)
project(laplace LANGUAGES CXX OBJCXX)

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

option(LAPLACE_NATIVE "Build for the active Apple Silicon Mac" ON)
option(LAPLACE_LTO "Enable link-time optimization" ON)

add_compile_options(
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-missing-field-initializers)

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
    src/closed_v1_source_schema.cpp
    src/column_grouped_affine_uint2_skip.cpp
    src/column_grouped_u2_atlas.cpp
    src/compat_rule.cpp
    src/compatibility_report.cpp
    src/execution_plan.cpp
    src/generation_loop.cpp
    src/gguf.cpp
    src/gguf_import.cpp
    src/gguf_index.cpp
    src/gguf_product_compiler.cpp
    src/matmul.cpp
    src/matmul_simd.cpp
    src/metal.mm
    src/mlx_package.cpp
    src/mmap.cpp
    src/product_authority.cpp
    src/product_package.cpp
    src/runtime_session.cpp
    src/safetensors.cpp
    src/safetensors_adapter.cpp
    src/sampler.cpp
    src/semantic_manifest.cpp
    src/semantic_model.cpp
    src/session_resources.mm
    src/source_compiler_graph_proof.cpp
    src/source_schema.cpp
    src/sparse_selector_coeff.cpp
    src/state_abi.cpp
    src/token_contract.cpp
    src/token_program.cpp)

set(LAPLACE_APPLE_FRAMEWORKS
    "-framework Accelerate"
    "-framework CoreFoundation"
    "-framework Foundation"
    "-framework IOKit"
    "-framework Metal")

add_library(laplace_core STATIC ${LAPLACE_CORE_SOURCES})
target_include_directories(laplace_core PUBLIC src)
target_link_libraries(laplace_core PUBLIC ${LAPLACE_APPLE_FRAMEWORKS})

add_executable(laplace src/main.cpp)
target_link_libraries(laplace PRIVATE laplace_core)

include(CTest)

if(BUILD_TESTING)
    function(laplace_contract_test target)
        add_executable(${target} tests/${target}.cpp)
        target_include_directories(${target} PRIVATE src tests)
        target_link_libraries(${target} PRIVATE laplace_core)
        add_test(NAME ${target} COMMAND ${target})
    endfunction()

    laplace_contract_test(test_compatibility_report)
    laplace_contract_test(test_semantic_model)
    laplace_contract_test(test_source_compiler_graph_proof)
    laplace_contract_test(test_source_schema)
    laplace_contract_test(test_closed_v1_source_schema)
    laplace_contract_test(test_execution_plan)
    laplace_contract_test(test_token_contract)
    laplace_contract_test(test_token_program)

    add_executable(test_cli tests/test_cli.cpp)
    target_include_directories(test_cli PRIVATE src tests)
    target_compile_definitions(test_cli PRIVATE
        LAPLACE_BINARY_PATH="${CMAKE_BINARY_DIR}/laplace")
    target_link_libraries(test_cli PRIVATE laplace_core)
    add_dependencies(test_cli laplace)
    add_test(NAME test_cli COMMAND test_cli)
endif()
