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
option(LAPLACE_LTO "Enable link-time optimization" ON)
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

include(cmake/LaplaceProductSources.cmake)
set(LAPLACE_PRODUCT_SOURCE_CLOSURE_SNAPSHOT
    "${CMAKE_BINARY_DIR}/laplace-product-source-closure.txt")

set(LAPLACE_SOURCES
    src/main.cpp
    src/codec_binding.cpp
    src/generation_loop.cpp
    src/product_package.cpp
    src/gguf_product_compiler.cpp
    src/normalized_gguf_product_compiler.cpp
    src/mlx_package.cpp
    src/normalized_source_evidence.cpp
    src/gguf_normalized_source_adapter.cpp
    src/mlx_normalized_source_adapter.cpp
    src/safetensors_adapter.cpp
    src/safetensors.cpp
    src/artifact_index.cpp
    src/artifact_set.cpp
    src/canonical_metal.cpp
    src/compatibility_report.cpp
    src/compat_rule.cpp
    src/product_authority.cpp
    src/execution_plan.cpp
    src/physical_codec.cpp
    src/gguf_index.cpp
    src/mmap.cpp
    src/gguf.cpp
    src/matmul.cpp
    src/matmul_simd.cpp
    src/column_grouped_u2_atlas.cpp
    src/column_grouped_affine_uint2_skip.cpp
    src/token_program.cpp
    src/sampler.cpp
    src/semantic_model.cpp
    src/semantic_dispatch_program.cpp
    src/bound_dispatch_requirements.cpp
    src/closed_v1_source_schema.cpp
    src/source_schema.cpp
    src/source_compiler_graph_proof.cpp
    src/program_ir.cpp
    src/program_metal.mm
    src/physical_program.cpp
    src/physical_program_package.cpp
    src/program_package.cpp
    src/program_state.cpp
    src/container_schema_program.cpp
    src/source_evidence.cpp
    src/source_program_solver.cpp
    src/semantic_manifest.cpp
    src/token_contract.cpp
    src/sparse_selector_coeff.cpp
    src/runtime_session.cpp
    src/session_resources.mm
    src/state_abi.cpp
)

add_executable(laplace ${LAPLACE_SOURCES})
target_include_directories(laplace PRIVATE src)
if(LAPLACE_OMP_TARGET)
    target_link_libraries(laplace PRIVATE ${LAPLACE_OMP_TARGET})
endif()

# Targets that pull in threadpool.h need IOKit + CoreFoundation on Apple
# for power-aware scheduling (battery detection) and os_sync_wait_on_address.
if(APPLE)
    add_library(laplace_metal_pipeline_transaction STATIC
        src/metal_pipeline_transaction.mm)
    target_include_directories(laplace_metal_pipeline_transaction PUBLIC src)
    target_link_libraries(laplace_metal_pipeline_transaction PUBLIC
        "-framework Metal" "-framework Foundation")

    add_library(laplace_product_metal_execution STATIC
        src/product_metal_execution.cpp
        src/product_metal_codec_capabilities.cpp
        src/structural_metal_compiler.cpp
        src/semantic_dispatch_program.cpp
        src/bound_dispatch_requirements.cpp
        src/normalized_codec_program.cpp
        src/metal_codec_capability.cpp
        src/metal_library_source_catalog.mm
        src/codec_certificate.cpp)
    target_include_directories(laplace_product_metal_execution PUBLIC src)
    target_link_libraries(laplace_product_metal_execution PUBLIC
        laplace_metal_pipeline_transaction
        "-framework Metal" "-framework Foundation")

    set(LAPLACE_ACCELERATE_FW "-framework Accelerate")
    set(LAPLACE_APPLE_FW "-framework IOKit" "-framework CoreFoundation"
        "-framework Metal" "-framework Foundation" ${LAPLACE_ACCELERATE_FW}
        laplace_product_metal_execution)
    target_link_libraries(laplace PRIVATE ${LAPLACE_APPLE_FW})
    set_source_files_properties(src/metal.mm PROPERTIES LANGUAGE OBJCXX)
    target_sources(laplace PRIVATE src/metal.mm)
endif()

# Tests (self-contained; no model file required)
enable_testing()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(test_universal_source_policy_example
    tests/test_universal_source_policy.cpp)
add_test(NAME test_universal_source_policy_example
    COMMAND test_universal_source_policy_example)
add_test(NAME test_universal_source_policy_scanner_self
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tools/check_universal_product.py --self-test)
add_test(NAME test_universal_source_policy_target_closure
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tools/check_universal_product.py
        --target-closure
        ${CMAKE_SOURCE_DIR}/cmake/LaplaceProductSources.cmake
        ${LAPLACE_PRODUCT_SOURCE_CLOSURE_SNAPSHOT})
add_test(NAME test_universal_source_policy
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tools/check_universal_product.py
        ${CMAKE_SOURCE_DIR}/cmake/LaplaceProductSources.cmake)

add_executable(test_container_schema_program
    tests/test_container_schema_program.cpp
    src/container_schema_program.cpp src/compatibility_report.cpp)
target_include_directories(test_container_schema_program PRIVATE src tests)
add_test(NAME test_container_schema_program COMMAND test_container_schema_program)

add_executable(test_source_program_solver
    tests/test_source_program_solver.cpp
    src/source_program_solver.cpp src/compatibility_report.cpp)
target_include_directories(test_source_program_solver PRIVATE src tests)
add_test(NAME test_source_program_solver COMMAND test_source_program_solver)

add_executable(test_program_ir tests/test_program_ir.cpp
    src/program_ir.cpp src/compatibility_report.cpp)
target_include_directories(test_program_ir PRIVATE src tests)
add_test(NAME test_program_ir COMMAND test_program_ir)

add_executable(test_program_state tests/test_program_state.cpp
    src/program_state.cpp src/program_ir.cpp src/compatibility_report.cpp)
target_include_directories(test_program_state PRIVATE src tests)
target_compile_definitions(test_program_state PRIVATE LAPLACE_TESTING=1)
add_test(NAME test_program_state COMMAND test_program_state)

add_executable(test_physical_program tests/test_physical_program.cpp
    src/physical_program.cpp src/physical_program_interpreter.cpp
    src/compatibility_report.cpp)
target_include_directories(test_physical_program PRIVATE src tests)
add_test(NAME test_physical_program COMMAND test_physical_program)

add_executable(test_physical_program_package tests/test_physical_program_package.cpp
    src/physical_program_package.cpp src/physical_program.cpp
    src/physical_program_interpreter.cpp src/program_ir.cpp
    src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp
    src/physical_codec.cpp src/codec_certificate.cpp src/token_contract.cpp src/semantic_model.cpp
    src/artifact_index.cpp src/artifact_set.cpp src/compatibility_report.cpp
    src/mmap.cpp)
target_include_directories(test_physical_program_package PRIVATE src tests)
add_test(NAME test_physical_program_package COMMAND test_physical_program_package)

add_executable(test_program_package tests/test_program_package.cpp
    src/program_package.cpp src/container_schema_program.cpp
    src/physical_program_package.cpp
    src/physical_program.cpp src/physical_program_interpreter.cpp
    src/program_state.cpp src/program_ir.cpp src/token_program.cpp
    src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp
    src/physical_codec.cpp src/codec_certificate.cpp src/token_contract.cpp
    src/semantic_model.cpp src/artifact_index.cpp src/artifact_set.cpp
    src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_program_package PRIVATE src tests)
add_test(NAME test_program_package COMMAND test_program_package)
add_test(NAME test_unseen_container_schema
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tests/run_unseen_container_schema.py
        $<TARGET_FILE:test_program_package>)

add_executable(test_program_reference tests/test_program_reference.cpp
    src/program_reference.cpp src/physical_program_package.cpp src/physical_program.cpp
    src/physical_program_interpreter.cpp
    src/program_ir.cpp src/semantic_model.cpp src/semantic_manifest.cpp
    src/source_compiler_graph_proof.cpp src/physical_codec.cpp
    src/codec_certificate.cpp src/token_contract.cpp src/artifact_index.cpp
    src/artifact_set.cpp src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_program_reference PRIVATE src tests)
add_test(NAME test_program_reference COMMAND test_program_reference)

if(APPLE)
    add_executable(test_program_metal tests/test_program_metal.cpp
        src/program_metal.mm src/program_reference.cpp
        src/physical_program_package.cpp src/physical_program.cpp
        src/physical_program_interpreter.cpp src/program_ir.cpp
        src/semantic_model.cpp src/semantic_manifest.cpp
        src/source_compiler_graph_proof.cpp src/physical_codec.cpp
        src/codec_certificate.cpp src/token_contract.cpp src/artifact_index.cpp
        src/artifact_set.cpp src/compatibility_report.cpp src/mmap.cpp)
    target_include_directories(test_program_metal PRIVATE src tests)
    target_link_libraries(test_program_metal PRIVATE "-framework Metal"
        "-framework Foundation")
    add_test(NAME test_program_metal COMMAND test_program_metal)
endif()
add_test(NAME test_physical_program_unseen
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tests/run_unseen_physical_program.py
        $<TARGET_FILE:test_physical_program>
        ${CMAKE_BINARY_DIR}/unseen-physical-program)

add_executable(test_codec_program tests/test_codec_program.cpp src/codec_program.cpp)
target_include_directories(test_codec_program PRIVATE src tests)
add_test(NAME test_codec_program COMMAND test_codec_program)

add_executable(test_codec_certificate tests/test_codec_certificate.cpp
    src/codec_certificate.cpp src/physical_codec.cpp src/semantic_model.cpp
    src/compatibility_report.cpp)
target_include_directories(test_codec_certificate PRIVATE src tests)
add_test(NAME test_codec_certificate COMMAND test_codec_certificate)

add_executable(test_codec_binding tests/test_codec_binding.cpp
    src/codec_binding.cpp src/physical_codec.cpp
    src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp
    src/semantic_model.cpp src/token_contract.cpp src/artifact_index.cpp
    src/artifact_set.cpp src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_codec_binding PRIVATE src tests)
target_compile_definitions(test_codec_binding PRIVATE LAPLACE_RUNTIME_PACKAGE_TESTING=1)
add_test(NAME test_codec_binding COMMAND test_codec_binding)

add_executable(test_sparse_ffn_bank tests/test_sparse_ffn_bank.cpp src/sparse_ffn_bank.cpp)
target_include_directories(test_sparse_ffn_bank PRIVATE src tests)
add_test(NAME test_sparse_ffn_bank COMMAND test_sparse_ffn_bank)

add_executable(test_sparse_selector_coeff tests/test_sparse_selector_coeff.cpp
    src/sparse_selector_coeff.cpp src/matmul.cpp src/matmul_simd.cpp src/ops.cpp)
target_include_directories(test_sparse_selector_coeff PRIVATE src tests)
if(LAPLACE_OMP_TARGET)
    target_link_libraries(test_sparse_selector_coeff PRIVATE ${LAPLACE_OMP_TARGET})
endif()
if(APPLE)
    target_link_libraries(test_sparse_selector_coeff PRIVATE ${LAPLACE_APPLE_FW})
    target_sources(test_sparse_selector_coeff PRIVATE src/metal.mm
        src/column_grouped_affine_uint2_skip.cpp)
    set_source_files_properties(src/metal.mm PROPERTIES LANGUAGE OBJCXX)
endif()
add_test(NAME test_sparse_selector_coeff COMMAND test_sparse_selector_coeff)

add_executable(test_compatibility_report tests/test_compatibility_report.cpp src/compatibility_report.cpp)
target_include_directories(test_compatibility_report PRIVATE src tests)
add_test(NAME test_compatibility_report COMMAND test_compatibility_report)

add_executable(test_gguf tests/test_gguf.cpp src/gguf.cpp src/mmap.cpp
    src/artifact_set.cpp src/compatibility_report.cpp)
target_include_directories(test_gguf PRIVATE src)
add_test(NAME test_gguf COMMAND test_gguf)

add_executable(test_artifact_set tests/test_artifact_set.cpp
    src/artifact_set.cpp src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_artifact_set PRIVATE src tests)
target_compile_definitions(test_artifact_set PRIVATE LAPLACE_ARTIFACT_SET_TESTING=1)
add_test(NAME test_artifact_set COMMAND test_artifact_set)

add_executable(test_artifact_index tests/test_artifact_index.cpp
    src/artifact_index.cpp src/artifact_set.cpp src/semantic_model.cpp
    src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_artifact_index PRIVATE src tests)
add_test(NAME test_artifact_index COMMAND test_artifact_index)

add_executable(test_normalized_source_evidence
    tests/test_normalized_source_evidence.cpp
    src/normalized_source_evidence.cpp
    src/gguf_normalized_source_adapter.cpp
    src/mlx_normalized_source_adapter.cpp
    src/artifact_index.cpp src/artifact_set.cpp src/semantic_model.cpp
    src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_normalized_source_evidence PRIVATE src tests)
add_test(NAME test_normalized_source_evidence
    COMMAND test_normalized_source_evidence)

add_executable(test_semantic_model tests/test_semantic_model.cpp
    src/semantic_model.cpp src/compatibility_report.cpp)
target_include_directories(test_semantic_model PRIVATE src tests)
add_test(NAME test_semantic_model COMMAND test_semantic_model)

add_executable(test_semantic_dispatch_program tests/test_semantic_dispatch_program.cpp
    src/semantic_dispatch_program.cpp src/semantic_model.cpp
    src/compatibility_report.cpp)
target_include_directories(test_semantic_dispatch_program PRIVATE src tests)
add_test(NAME test_semantic_dispatch_program COMMAND test_semantic_dispatch_program)

add_executable(test_bound_dispatch_requirements
    tests/test_bound_dispatch_requirements.cpp
    src/bound_dispatch_requirements.cpp src/semantic_dispatch_program.cpp
    src/codec_binding.cpp src/codec_certificate.cpp src/physical_codec.cpp
    src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp
    src/token_contract.cpp src/semantic_model.cpp src/artifact_index.cpp
    src/artifact_set.cpp src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_bound_dispatch_requirements PRIVATE src tests)
target_compile_definitions(test_bound_dispatch_requirements PRIVATE
    LAPLACE_RUNTIME_PACKAGE_TESTING=1)
add_test(NAME test_bound_dispatch_requirements COMMAND test_bound_dispatch_requirements)

add_executable(test_structural_metal_compiler
    tests/test_structural_metal_compiler.cpp
    src/structural_metal_compiler.cpp src/bound_dispatch_requirements.cpp
    src/semantic_dispatch_program.cpp src/codec_binding.cpp
    src/codec_certificate.cpp src/normalized_codec_program.cpp
    src/metal_codec_capability.cpp src/physical_codec.cpp
    src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp
    src/token_contract.cpp src/semantic_model.cpp src/artifact_index.cpp
    src/artifact_set.cpp src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_structural_metal_compiler PRIVATE src tests)
target_compile_definitions(test_structural_metal_compiler PRIVATE
    LAPLACE_RUNTIME_PACKAGE_TESTING=1)
add_test(NAME test_structural_metal_compiler
    COMMAND test_structural_metal_compiler)

if(APPLE)
    add_executable(test_metal_library_source_catalog
        tests/test_metal_library_source_catalog.cpp
        src/metal_library_source_catalog.mm)
    target_include_directories(test_metal_library_source_catalog PRIVATE src tests)
    target_compile_definitions(test_metal_library_source_catalog PRIVATE
        LAPLACE_TESTING=1)
    target_link_libraries(test_metal_library_source_catalog PRIVATE
        "-framework Metal" "-framework Foundation"
        laplace_metal_pipeline_transaction)
    add_test(NAME test_metal_library_source_catalog
        COMMAND test_metal_library_source_catalog)
endif()

add_executable(test_metal_codec_capability
    tests/test_metal_codec_capability.cpp
    src/normalized_codec_program.cpp src/metal_codec_capability.cpp
    src/codec_certificate.cpp src/compatibility_report.cpp)
target_include_directories(test_metal_codec_capability PRIVATE src tests)
add_test(NAME test_metal_codec_capability
    COMMAND test_metal_codec_capability)

add_executable(test_source_compiler_graph_proof tests/test_source_compiler_graph_proof.cpp
    src/source_compiler_graph_proof.cpp src/semantic_model.cpp src/compatibility_report.cpp)
target_include_directories(test_source_compiler_graph_proof PRIVATE src tests)
add_test(NAME test_source_compiler_graph_proof COMMAND test_source_compiler_graph_proof)

add_executable(test_token_contract tests/test_token_contract.cpp src/token_contract.cpp)
target_include_directories(test_token_contract PRIVATE src tests)
add_test(NAME test_token_contract COMMAND test_token_contract)

add_executable(test_semantic_manifest tests/test_semantic_manifest.cpp
    src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp src/semantic_model.cpp src/artifact_index.cpp
    src/token_contract.cpp src/artifact_set.cpp src/compatibility_report.cpp src/mmap.cpp src/physical_codec.cpp)
target_include_directories(test_semantic_manifest PRIVATE src tests)
add_test(NAME test_semantic_manifest COMMAND test_semantic_manifest)

add_executable(test_source_schema tests/test_source_schema.cpp src/source_schema.cpp)
target_include_directories(test_source_schema PRIVATE src tests)
add_test(NAME test_source_schema COMMAND test_source_schema)

add_executable(test_closed_v1_source_schema tests/test_closed_v1_source_schema.cpp
    src/closed_v1_source_schema.cpp src/source_schema.cpp src/source_compiler_graph_proof.cpp
    src/semantic_model.cpp src/compatibility_report.cpp src/artifact_index.cpp src/artifact_set.cpp src/mmap.cpp)
target_include_directories(test_closed_v1_source_schema PRIVATE src tests)
add_test(NAME test_closed_v1_source_schema COMMAND test_closed_v1_source_schema)

add_executable(test_source_schema_authority tests/test_source_schema_authority.cpp
    src/compat_rule.cpp src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp
    src/physical_codec.cpp
    src/token_contract.cpp src/semantic_model.cpp src/artifact_index.cpp src/artifact_set.cpp
    src/compatibility_report.cpp src/mmap.cpp src/source_schema.cpp)
target_include_directories(test_source_schema_authority PRIVATE src tests)
add_test(NAME test_source_schema_authority COMMAND test_source_schema_authority)

add_executable(test_runtime_package tests/test_runtime_package.cpp
    src/compat_rule.cpp src/product_authority.cpp src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp src/semantic_model.cpp
    src/physical_codec.cpp
    src/token_contract.cpp src/artifact_index.cpp src/artifact_set.cpp src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_runtime_package PRIVATE src tests)
add_test(NAME test_runtime_package COMMAND test_runtime_package)

add_executable(test_product_package tests/test_product_package.cpp
    src/product_package.cpp src/gguf_product_compiler.cpp src/normalized_gguf_product_compiler.cpp
    src/normalized_source_evidence.cpp src/gguf_normalized_source_adapter.cpp
    src/codec_certificate.cpp
    src/mlx_normalized_source_adapter.cpp src/mlx_package.cpp src/safetensors.cpp src/safetensors_adapter.cpp
    src/gguf_import.cpp src/gguf_index.cpp src/artifact_index.cpp src/compat_rule.cpp
    src/product_authority.cpp src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp src/semantic_model.cpp
    src/physical_codec.cpp
    src/token_contract.cpp src/token_program.cpp src/artifact_set.cpp src/compatibility_report.cpp src/mmap.cpp src/gguf.cpp
    src/closed_v1_source_schema.cpp src/source_schema.cpp
    src/physical_program.cpp src/physical_program_package.cpp src/program_ir.cpp)
target_include_directories(test_product_package PRIVATE src tests)
target_compile_definitions(test_product_package PRIVATE LAPLACE_ARTIFACT_SET_TESTING=1)
add_test(NAME test_product_package COMMAND test_product_package)

add_executable(test_gguf_product_compiler tests/test_gguf_product_compiler.cpp
    src/gguf_product_compiler.cpp src/normalized_gguf_product_compiler.cpp
    src/normalized_source_evidence.cpp src/gguf_normalized_source_adapter.cpp
    src/codec_certificate.cpp src/gguf_index.cpp src/gguf.cpp
    src/artifact_index.cpp src/artifact_set.cpp src/compat_rule.cpp
    src/product_authority.cpp src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp
    src/physical_codec.cpp
    src/semantic_model.cpp src/token_contract.cpp src/token_program.cpp
    src/compatibility_report.cpp src/mmap.cpp
    src/closed_v1_source_schema.cpp src/source_schema.cpp)
target_include_directories(test_gguf_product_compiler PRIVATE src tests)
add_test(NAME test_gguf_product_compiler COMMAND test_gguf_product_compiler)

add_executable(test_product_authority tests/test_product_authority.cpp
    src/product_authority.cpp src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp src/semantic_model.cpp
    src/physical_codec.cpp
    src/token_contract.cpp src/artifact_index.cpp src/artifact_set.cpp
    src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_product_authority PRIVATE src tests)
add_test(NAME test_product_authority COMMAND test_product_authority)

add_executable(test_compat_rule tests/test_compat_rule.cpp
    src/compat_rule.cpp src/semantic_model.cpp src/compatibility_report.cpp src/artifact_set.cpp src/mmap.cpp)
target_include_directories(test_compat_rule PRIVATE src tests)
add_test(NAME test_compat_rule COMMAND test_compat_rule)

add_executable(test_gguf_import tests/test_gguf_import.cpp
    src/gguf_import.cpp src/gguf_index.cpp src/artifact_index.cpp src/compat_rule.cpp src/semantic_model.cpp src/compatibility_report.cpp
    src/semantic_manifest.cpp src/physical_codec.cpp src/source_compiler_graph_proof.cpp src/token_contract.cpp src/artifact_set.cpp src/mmap.cpp src/gguf.cpp)
target_include_directories(test_gguf_import PRIVATE src tests)
add_test(NAME test_gguf_import COMMAND test_gguf_import)

add_executable(test_gguf_index tests/test_gguf_index.cpp
    src/gguf_index.cpp src/artifact_index.cpp src/compat_rule.cpp src/semantic_model.cpp
    src/semantic_manifest.cpp src/physical_codec.cpp src/source_compiler_graph_proof.cpp src/token_contract.cpp src/compatibility_report.cpp src/artifact_set.cpp src/mmap.cpp src/gguf.cpp)
target_include_directories(test_gguf_index PRIVATE src tests)
add_test(NAME test_gguf_index COMMAND test_gguf_index)

option(LAPLACE_ENABLE_QUALIFICATION_TESTS "Build external Qwen qualification gates" OFF)
set(LAPLACE_QUALIFICATION_GGUF "" CACHE FILEPATH "Pinned Qwen2.5-0.5B F16 GGUF qualification artifact")
set(LAPLACE_HAS_QUALIFICATION_GGUF 0)
if(LAPLACE_ENABLE_QUALIFICATION_TESTS)
    if(NOT EXISTS "${LAPLACE_QUALIFICATION_GGUF}")
        message(FATAL_ERROR "LAPLACE_QUALIFICATION_GGUF must name the pinned Qwen2.5-0.5B F16 GGUF")
    endif()
    file(SHA256 "${LAPLACE_QUALIFICATION_GGUF}" LAPLACE_QUALIFICATION_GGUF_SHA256)
    if(NOT LAPLACE_QUALIFICATION_GGUF_SHA256 STREQUAL "73f932c0a3a3daf1122545473973b4189ef0811cd6f2f14d73afe9fa1edd8efd")
        message(FATAL_ERROR "LAPLACE_QUALIFICATION_GGUF does not have the pinned SHA-256")
    endif()
    set(LAPLACE_HAS_QUALIFICATION_GGUF 1)

    add_executable(test_external_golden tests/test_external_golden.cpp
        tests/reference_fp32.cpp src/gguf_import.cpp src/gguf_index.cpp src/artifact_index.cpp src/compat_rule.cpp src/semantic_model.cpp
        src/semantic_manifest.cpp src/physical_codec.cpp src/source_compiler_graph_proof.cpp src/token_contract.cpp src/compatibility_report.cpp src/artifact_set.cpp src/mmap.cpp src/gguf.cpp)
    target_include_directories(test_external_golden PRIVATE src tests)
    target_compile_definitions(test_external_golden PRIVATE LAPLACE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        LAPLACE_QUALIFICATION_GGUF="${LAPLACE_QUALIFICATION_GGUF}")
    foreach(prefix RANGE 1 8)
        add_test(NAME qualification_gate_a_prefix_${prefix} COMMAND test_external_golden ${prefix})
    endforeach()
endif()

add_executable(test_execution_plan tests/test_execution_plan.cpp
    src/execution_plan.cpp src/semantic_dispatch_program.cpp
    src/codec_certificate.cpp
    src/physical_codec.cpp src/semantic_manifest.cpp
    src/source_compiler_graph_proof.cpp src/token_contract.cpp
    src/semantic_model.cpp src/artifact_index.cpp src/artifact_set.cpp
    src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_execution_plan PRIVATE src tests)
if(APPLE)
    target_link_libraries(test_execution_plan PRIVATE ${LAPLACE_APPLE_FW})
endif()
add_test(NAME test_execution_plan COMMAND test_execution_plan)

if(LAPLACE_ENABLE_QUALIFICATION_TESTS)
    add_executable(test_runtime_session tests/test_runtime_session.cpp tests/reference_fp32.cpp
        src/runtime_session.cpp src/session_resources.mm src/state_abi.cpp src/scalar_executor.cpp
        src/canonical_metal.cpp src/sparse_selector_coeff.cpp src/tokenizer.cpp
        src/matmul.cpp src/matmul_simd.cpp src/column_grouped_q4.cpp
        src/column_grouped_u2_atlas.cpp
        src/column_grouped_affine_uint2_skip.cpp src/metal.mm
        src/gguf_import.cpp src/gguf_index.cpp src/artifact_index.cpp src/compat_rule.cpp src/semantic_model.cpp src/execution_plan.cpp
        src/semantic_manifest.cpp src/physical_codec.cpp src/source_compiler_graph_proof.cpp src/token_contract.cpp src/compatibility_report.cpp src/artifact_set.cpp src/mmap.cpp src/gguf.cpp)
    target_include_directories(test_runtime_session PRIVATE src tests)
    target_compile_definitions(test_runtime_session PRIVATE LAPLACE_TESTING=1
        LAPLACE_QUALIFICATION_RUNTIME=1
        LAPLACE_RUNTIME_PACKAGE_TESTING=1
        LAPLACE_QUALIFICATION_GGUF="${LAPLACE_QUALIFICATION_GGUF}")
    if(APPLE)
        target_link_libraries(test_runtime_session PRIVATE ${LAPLACE_APPLE_FW})
    endif()
    add_test(NAME qualification_gate_b_scalar_production COMMAND test_runtime_session)

endif()

if(APPLE)
    add_executable(test_metal_pipeline_transaction
        tests/test_metal_pipeline_transaction.cpp
        src/metal_pipeline_transaction.mm
        src/metal.mm)
    target_include_directories(test_metal_pipeline_transaction PRIVATE src tests)
    target_compile_definitions(test_metal_pipeline_transaction PRIVATE
        LAPLACE_TESTING=1)
    target_link_libraries(test_metal_pipeline_transaction PRIVATE
        ${LAPLACE_APPLE_FW})
    add_test(NAME test_metal_pipeline_transaction
        COMMAND test_metal_pipeline_transaction)
    set_tests_properties(test_metal_pipeline_transaction PROPERTIES
        SKIP_REGULAR_EXPRESSION "SKIP: no Metal device")

    add_executable(test_metal_capability_snapshot
        tests/test_metal_capability_snapshot.cpp
        src/metal_capability_snapshot.mm
        src/codec_program.cpp)
    target_include_directories(test_metal_capability_snapshot PRIVATE src tests)
    target_compile_definitions(test_metal_capability_snapshot PRIVATE
        LAPLACE_TESTING=1
        LAPLACE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(test_metal_capability_snapshot PRIVATE ${LAPLACE_APPLE_FW})
    add_test(NAME test_metal_capability_snapshot COMMAND test_metal_capability_snapshot)
    set_tests_properties(test_metal_capability_snapshot PROPERTIES
        SKIP_REGULAR_EXPRESSION "SKIP: no Metal device")

    add_executable(test_canonical_metal tests/test_canonical_metal.cpp tests/reference_fp32.cpp
        src/canonical_metal.cpp src/sparse_selector_coeff.cpp src/gguf_import.cpp src/gguf_index.cpp src/artifact_index.cpp src/compat_rule.cpp src/semantic_model.cpp src/semantic_manifest.cpp src/token_contract.cpp
        src/physical_codec.cpp
        src/source_compiler_graph_proof.cpp
        src/compatibility_report.cpp src/artifact_set.cpp src/mmap.cpp src/gguf.cpp
        src/tokenizer.cpp src/execution_plan.cpp src/column_grouped_q4.cpp
        src/matmul.cpp src/matmul_simd.cpp src/column_grouped_u2_atlas.cpp
        src/column_grouped_affine_uint2_skip.cpp
        src/trellis_codec.cpp src/metal.mm)
    target_include_directories(test_canonical_metal PRIVATE src tests)
    target_compile_definitions(test_canonical_metal PRIVATE
        LAPLACE_METAL_TESTING=1
        LAPLACE_RUNTIME_PACKAGE_TESTING=1
        LAPLACE_HAS_QUALIFICATION_GGUF=${LAPLACE_HAS_QUALIFICATION_GGUF}
        LAPLACE_QUALIFICATION_GGUF="${LAPLACE_QUALIFICATION_GGUF}")
    target_link_libraries(test_canonical_metal PRIVATE ${LAPLACE_APPLE_FW})
    add_test(NAME test_canonical_metal_attention_scale
        COMMAND test_canonical_metal --attention-scale)
    add_test(NAME test_canonical_metal_moe
        COMMAND test_canonical_metal --canonical-moe)
    add_test(NAME test_canonical_metal_grouped_affine_lowering
        COMMAND test_canonical_metal --affine-u2-256-lowering)
    add_test(NAME test_canonical_metal_grouped_affine_physical_boundary
        COMMAND test_canonical_metal --affine-u2-256-e2e)
    set_tests_properties(test_canonical_metal_attention_scale test_canonical_metal_moe
        test_canonical_metal_grouped_affine_lowering
        test_canonical_metal_grouped_affine_physical_boundary PROPERTIES
        SKIP_REGULAR_EXPRESSION "SKIP: no Metal device")
    if(LAPLACE_ENABLE_QUALIFICATION_TESTS)
        add_test(NAME qualification_canonical_metal COMMAND test_canonical_metal)
    endif()

    add_executable(bench_canonical_metal tests/bench_canonical_metal.cpp
        src/canonical_metal.cpp src/sparse_selector_coeff.cpp src/gguf_import.cpp src/gguf_index.cpp src/artifact_index.cpp src/compat_rule.cpp src/semantic_model.cpp src/semantic_manifest.cpp src/token_contract.cpp
        src/physical_codec.cpp
        src/source_compiler_graph_proof.cpp
        src/compatibility_report.cpp src/artifact_set.cpp src/mmap.cpp src/gguf.cpp src/tokenizer.cpp
        src/execution_plan.cpp src/matmul.cpp src/matmul_simd.cpp
        src/column_grouped_u2_atlas.cpp
        src/column_grouped_affine_uint2_skip.cpp src/metal.mm)
    target_include_directories(bench_canonical_metal PRIVATE src tests)
    target_compile_definitions(bench_canonical_metal PRIVATE
        LAPLACE_QUALIFICATION_RUNTIME=1
        LAPLACE_HAS_QUALIFICATION_GGUF=${LAPLACE_HAS_QUALIFICATION_GGUF}
        LAPLACE_QUALIFICATION_GGUF="${LAPLACE_QUALIFICATION_GGUF}")
    target_link_libraries(bench_canonical_metal PRIVATE ${LAPLACE_APPLE_FW})

    add_executable(test_qwen_canonical_parity tests/test_qwen_canonical_parity.cpp
        src/canonical_metal.cpp src/sparse_selector_coeff.cpp src/gguf_import.cpp src/gguf_index.cpp src/artifact_index.cpp src/compat_rule.cpp src/semantic_model.cpp src/semantic_manifest.cpp src/token_contract.cpp
        src/physical_codec.cpp
        src/source_compiler_graph_proof.cpp
        src/compatibility_report.cpp src/artifact_set.cpp src/mmap.cpp src/gguf.cpp
        src/execution_plan.cpp src/model.cpp src/token_graph_backend.cpp src/topology.cpp src/ops.cpp
        src/laplace_arch.cpp src/arch_qwen3next.cpp src/arch_llama.cpp src/arch_phi3.cpp src/arch_adaptive.cpp
        src/laplace_moe.cpp src/matmul.cpp src/matmul_simd.cpp src/ssm.cpp src/kvcache.cpp
        src/laplace_kv.cpp src/laplace_kv_adaptive.cpp src/laplace_kv_q4.cpp src/tokenizer.cpp
        src/column_grouped_u2_atlas.cpp
        src/column_grouped_affine_uint2_skip.cpp src/metal.mm)
    target_include_directories(test_qwen_canonical_parity PRIVATE src tests)
    target_compile_definitions(test_qwen_canonical_parity PRIVATE
        LAPLACE_QUALIFICATION_RUNTIME=1)
    if(LAPLACE_OMP_TARGET)
        target_link_libraries(test_qwen_canonical_parity PRIVATE ${LAPLACE_OMP_TARGET})
    endif()
    target_link_libraries(test_qwen_canonical_parity PRIVATE ${LAPLACE_APPLE_FW})

    # Artifact-free product-route witness.  This deliberately uses an
    # in-memory carried manifest fixture, so product-session routing is tested
    # independently of the optional external qualification artifact.
    add_executable(test_runtime_session_product tests/test_runtime_session_product.cpp
        src/runtime_session.cpp src/session_resources.mm src/state_abi.cpp
        src/canonical_metal.cpp src/sparse_selector_coeff.cpp src/execution_plan.cpp
        src/codec_binding.cpp
        src/physical_codec.cpp
        src/matmul.cpp src/matmul_simd.cpp src/column_grouped_q4.cpp
        src/column_grouped_u2_atlas.cpp
        src/column_grouped_affine_uint2_skip.cpp src/trellis_codec.cpp src/metal.mm
        src/compatibility_report.cpp src/artifact_set.cpp src/artifact_index.cpp
        src/semantic_model.cpp src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp
        src/token_contract.cpp src/product_authority.cpp src/mmap.cpp
        src/physical_program.cpp src/physical_program_interpreter.cpp
        src/physical_program_package.cpp src/program_ir.cpp src/program_metal.mm
        src/program_package.cpp src/program_state.cpp src/token_program.cpp
        src/container_schema_program.cpp)
    target_include_directories(test_runtime_session_product PRIVATE src tests)
    target_compile_definitions(test_runtime_session_product PRIVATE
        LAPLACE_TESTING=1 LAPLACE_METAL_TESTING=1 LAPLACE_RUNTIME_PACKAGE_TESTING=1)
    target_link_libraries(test_runtime_session_product PRIVATE ${LAPLACE_APPLE_FW})
    add_test(NAME test_runtime_session_product COMMAND test_runtime_session_product)
    add_test(NAME test_runtime_program_session
        COMMAND test_runtime_session_product --program-session)
    set_tests_properties(test_runtime_session_product PROPERTIES
        SKIP_REGULAR_EXPRESSION "SKIP: product Metal device unavailable")
    set_tests_properties(test_runtime_program_session PROPERTIES
        SKIP_REGULAR_EXPRESSION "SKIP: product Metal device unavailable")

    add_executable(test_product_metal_codec_capabilities
        tests/test_product_metal_codec_capabilities.cpp
        src/product_metal_execution.cpp
        src/product_metal_codec_capabilities.cpp
        src/structural_metal_compiler.cpp src/metal_library_source_catalog.mm
        src/normalized_codec_program.cpp src/metal_codec_capability.cpp
        src/codec_certificate.cpp src/semantic_dispatch_program.cpp
        src/bound_dispatch_requirements.cpp
        src/runtime_session.cpp src/session_resources.mm src/state_abi.cpp
        src/canonical_metal.cpp src/sparse_selector_coeff.cpp src/execution_plan.cpp
        src/codec_binding.cpp src/physical_codec.cpp
        src/matmul.cpp src/matmul_simd.cpp src/column_grouped_q4.cpp
        src/column_grouped_u2_atlas.cpp
        src/column_grouped_affine_uint2_skip.cpp src/trellis_codec.cpp src/metal.mm
        src/compatibility_report.cpp src/artifact_set.cpp src/artifact_index.cpp
        src/semantic_model.cpp src/semantic_manifest.cpp src/source_compiler_graph_proof.cpp
        src/token_contract.cpp src/product_authority.cpp src/mmap.cpp)
    target_include_directories(test_product_metal_codec_capabilities PRIVATE src tests)
    target_compile_definitions(test_product_metal_codec_capabilities PRIVATE
        LAPLACE_TESTING=1 LAPLACE_METAL_TESTING=1 LAPLACE_RUNTIME_PACKAGE_TESTING=1)
    target_link_libraries(test_product_metal_codec_capabilities PRIVATE ${LAPLACE_APPLE_FW})
    add_test(NAME test_product_metal_codec_capabilities
        COMMAND test_product_metal_codec_capabilities)
endif()

add_executable(test_state_abi tests/test_state_abi.cpp src/state_abi.cpp src/compatibility_report.cpp)
target_include_directories(test_state_abi PRIVATE src tests)
add_test(NAME test_state_abi COMMAND test_state_abi)

add_executable(test_rule_qualification tests/test_rule_qualification.cpp
    src/compat_rule.cpp src/semantic_model.cpp src/compatibility_report.cpp src/artifact_set.cpp src/mmap.cpp)
target_include_directories(test_rule_qualification PRIVATE src tests)
add_test(NAME test_rule_qualification COMMAND test_rule_qualification)

add_executable(test_safetensors tests/test_safetensors.cpp src/safetensors.cpp src/mmap.cpp)
target_include_directories(test_safetensors PRIVATE src)
add_test(NAME test_safetensors COMMAND test_safetensors)

add_executable(test_safetensors_adapter tests/test_safetensors_adapter.cpp
    src/safetensors.cpp src/safetensors_adapter.cpp src/artifact_index.cpp src/artifact_set.cpp
    src/semantic_model.cpp src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_safetensors_adapter PRIVATE src tests)
add_test(NAME test_safetensors_adapter COMMAND test_safetensors_adapter)

add_executable(test_mlx_package tests/test_mlx_package.cpp
    src/mlx_package.cpp src/safetensors.cpp src/safetensors_adapter.cpp src/artifact_index.cpp
    src/artifact_set.cpp src/semantic_model.cpp src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_mlx_package PRIVATE src tests)
add_test(NAME test_mlx_package COMMAND test_mlx_package)

add_executable(test_matmul tests/test_matmul.cpp src/column_grouped_q4.cpp
    src/column_grouped_affine_uint2_skip.cpp
    src/matmul.cpp src/matmul_simd.cpp src/ops.cpp)
target_include_directories(test_matmul PRIVATE src)
target_compile_definitions(test_matmul PRIVATE LAPLACE_TESTING=1)
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

add_executable(test_column_grouped_q4 tests/test_column_grouped_q4.cpp src/column_grouped_q4.cpp)
target_include_directories(test_column_grouped_q4 PRIVATE src tests)
add_test(NAME test_column_grouped_q4 COMMAND test_column_grouped_q4)

add_executable(test_column_grouped_affine_lowbit
    tests/test_column_grouped_affine_lowbit.cpp
    src/column_grouped_affine_lowbit.cpp)
target_include_directories(test_column_grouped_affine_lowbit PRIVATE src tests)
add_test(NAME test_column_grouped_affine_lowbit COMMAND test_column_grouped_affine_lowbit)

add_executable(test_column_grouped_affine_uint2_skip
    tests/test_column_grouped_affine_uint2_skip.cpp
    src/column_grouped_affine_uint2_skip.cpp)
target_include_directories(test_column_grouped_affine_uint2_skip PRIVATE src tests)
add_test(NAME test_column_grouped_affine_uint2_skip
    COMMAND test_column_grouped_affine_uint2_skip)

add_executable(test_column_grouped_u2_atlas
    tests/test_column_grouped_u2_atlas.cpp
    src/column_grouped_u2_atlas.cpp
    src/column_grouped_affine_uint2_skip.cpp
    src/column_grouped_q4.cpp
    src/matmul.cpp src/matmul_simd.cpp src/ops.cpp)
target_include_directories(test_column_grouped_u2_atlas PRIVATE src tests)
target_compile_definitions(test_column_grouped_u2_atlas PRIVATE LAPLACE_TESTING=1)
if(LAPLACE_OMP_TARGET)
    target_link_libraries(test_column_grouped_u2_atlas PRIVATE ${LAPLACE_OMP_TARGET})
endif()
if(APPLE)
    target_link_libraries(test_column_grouped_u2_atlas PRIVATE ${LAPLACE_APPLE_FW})
    target_sources(test_column_grouped_u2_atlas PRIVATE src/metal.mm)
endif()
add_test(NAME test_column_grouped_u2_atlas COMMAND test_column_grouped_u2_atlas)

if(APPLE)
    add_executable(test_metal tests/test_metal.cpp
        src/token_graph_backend.cpp
        src/model.cpp
        src/topology.cpp
        src/ops.cpp
        src/laplace_arch.cpp
        src/arch_qwen3next.cpp
        src/arch_llama.cpp
        src/arch_phi3.cpp
        src/arch_adaptive.cpp
        src/laplace_moe.cpp
        src/sparse_ffn_bank.cpp
        src/column_grouped_q4.cpp
        src/column_grouped_affine_uint2_skip.cpp
        src/matmul.cpp
        src/matmul_simd.cpp
        src/ssm.cpp
        src/kvcache.cpp
        src/laplace_kv.cpp
        src/laplace_kv_adaptive.cpp
        src/laplace_kv_q4.cpp
        src/trellis_codec.cpp
        src/gguf.cpp
        src/mmap.cpp
        src/semantic_model.cpp
        src/compatibility_report.cpp
        src/metal.mm)
    target_include_directories(test_metal PRIVATE src tests)
    target_compile_definitions(test_metal PRIVATE LAPLACE_TESTING=1)
    if(LAPLACE_OMP_TARGET)
        target_link_libraries(test_metal PRIVATE ${LAPLACE_OMP_TARGET})
    endif()
    target_link_libraries(test_metal PRIVATE ${LAPLACE_APPLE_FW})
    add_test(NAME test_metal COMMAND test_metal)
    add_test(NAME test_metal_trellis COMMAND test_metal --trellis)
    add_test(NAME test_metal_trellis_real_shape COMMAND test_metal --trellis-real-shape)
    add_test(NAME test_metal_column_grouped_q4 COMMAND test_metal --column-grouped-q4)
    add_test(NAME test_metal_column_grouped_affine_lowbit
        COMMAND test_metal --column-grouped-affine-lowbit)
    add_test(NAME test_metal_column_grouped_affine_lowbit_real_shape
        COMMAND test_metal --column-grouped-affine-lowbit-real-shape)
    add_test(NAME test_metal_column_grouped_affine_uint2_skip
        COMMAND test_metal --column-grouped-affine-uint2-skip)
    add_test(NAME test_metal_column_grouped_affine_uint2_skip_real_shape
        COMMAND test_metal --column-grouped-affine-uint2-skip-real-shape)
    add_test(NAME test_metal_column_grouped_affine_uint2_skip_session
        COMMAND test_metal --column-grouped-affine-uint2-skip-session)
    set_tests_properties(test_metal test_metal_trellis test_metal_trellis_real_shape
        test_metal_column_grouped_q4
        test_metal_column_grouped_affine_lowbit
        test_metal_column_grouped_affine_lowbit_real_shape
        test_metal_column_grouped_affine_uint2_skip
        test_metal_column_grouped_affine_uint2_skip_real_shape
        test_metal_column_grouped_affine_uint2_skip_session PROPERTIES
        SKIP_REGULAR_EXPRESSION "SKIP: no Metal device")

    add_executable(bench_cpu_gpu tests/bench_cpu_gpu.cpp src/matmul.cpp src/matmul_simd.cpp
        src/column_grouped_affine_uint2_skip.cpp src/metal.mm)
    target_include_directories(bench_cpu_gpu PRIVATE src)
    target_link_libraries(bench_cpu_gpu PRIVATE "-framework Metal" "-framework Foundation" "-framework IOKit" "-framework CoreFoundation" laplace_metal_pipeline_transaction)

    add_executable(check_gguf_gemv tests/check_gguf_gemv.cpp src/gguf.cpp src/mmap.cpp
        src/matmul.cpp src/matmul_simd.cpp src/ops.cpp
        src/column_grouped_affine_uint2_skip.cpp src/metal.mm)
    target_include_directories(check_gguf_gemv PRIVATE src)
    target_link_libraries(check_gguf_gemv PRIVATE "-framework Metal" "-framework Foundation" "-framework IOKit" "-framework CoreFoundation" laplace_metal_pipeline_transaction)
endif()

add_executable(test_kvcache tests/test_kvcache.cpp
    src/kvcache.cpp src/laplace_kv.cpp src/laplace_kv_adaptive.cpp
    src/laplace_kv_q4.cpp src/ops.cpp)
target_include_directories(test_kvcache PRIVATE src)
if(APPLE)
    target_link_libraries(test_kvcache PRIVATE ${LAPLACE_ACCELERATE_FW})
endif()
add_test(NAME test_kvcache COMMAND test_kvcache)

add_executable(test_trellis_codec tests/test_trellis_codec.cpp src/trellis_codec.cpp)
target_include_directories(test_trellis_codec PRIVATE src tests)
add_test(NAME test_trellis_codec COMMAND test_trellis_codec)

add_executable(test_sampler tests/test_sampler.cpp src/sampler.cpp)
target_include_directories(test_sampler PRIVATE src)
add_test(NAME test_sampler COMMAND test_sampler)

add_executable(test_tokenizer tests/test_tokenizer.cpp src/tokenizer.cpp src/gguf.cpp src/mmap.cpp)
target_include_directories(test_tokenizer PRIVATE src)
add_test(NAME test_tokenizer COMMAND test_tokenizer)

add_executable(test_token_program tests/test_token_program.cpp src/token_program.cpp)
target_include_directories(test_token_program PRIVATE src tests)
add_test(NAME test_token_program COMMAND test_token_program)

add_executable(test_arch tests/test_arch.cpp
    src/token_graph_backend.cpp
    src/topology.cpp
    src/ops.cpp
    src/laplace_arch.cpp
    src/arch_qwen3next.cpp
    src/arch_llama.cpp
    src/arch_phi3.cpp
    src/arch_adaptive.cpp
    src/laplace_moe.cpp
    src/matmul.cpp
    src/matmul_simd.cpp
    src/ssm.cpp
    src/kvcache.cpp
    src/laplace_kv.cpp
    src/laplace_kv_adaptive.cpp
    src/laplace_kv_q4.cpp
    src/gguf.cpp
    src/mmap.cpp)
target_include_directories(test_arch PRIVATE src tests)
if(LAPLACE_OMP_TARGET)
    target_link_libraries(test_arch PRIVATE ${LAPLACE_OMP_TARGET})
endif()
if(APPLE)
    target_link_libraries(test_arch PRIVATE ${LAPLACE_APPLE_FW})
    target_sources(test_arch PRIVATE src/metal.mm src/column_grouped_affine_uint2_skip.cpp)
    set_source_files_properties(src/metal.mm PROPERTIES LANGUAGE OBJCXX)
endif()
add_test(NAME test_arch COMMAND test_arch)

add_executable(test_model_init tests/test_model_init.cpp
    src/token_graph_backend.cpp
    src/model.cpp
    src/topology.cpp
    src/ops.cpp
    src/laplace_arch.cpp
    src/arch_qwen3next.cpp
    src/arch_llama.cpp
    src/arch_phi3.cpp
    src/arch_adaptive.cpp
    src/laplace_moe.cpp
    src/matmul.cpp
    src/matmul_simd.cpp
    src/ssm.cpp
    src/kvcache.cpp
    src/laplace_kv.cpp
    src/laplace_kv_adaptive.cpp
    src/laplace_kv_q4.cpp
    src/gguf.cpp
    src/mmap.cpp)
target_include_directories(test_model_init PRIVATE src tests)
if(LAPLACE_OMP_TARGET)
    target_link_libraries(test_model_init PRIVATE ${LAPLACE_OMP_TARGET})
endif()
if(APPLE)
    target_link_libraries(test_model_init PRIVATE ${LAPLACE_APPLE_FW})
    target_sources(test_model_init PRIVATE src/metal.mm src/column_grouped_affine_uint2_skip.cpp)
    set_source_files_properties(src/metal.mm PROPERTIES LANGUAGE OBJCXX)
endif()
add_test(NAME test_model_init COMMAND test_model_init)

add_executable(test_cli tests/test_cli.cpp
    src/gguf_import.cpp src/gguf_index.cpp src/artifact_index.cpp src/compat_rule.cpp
    src/semantic_model.cpp src/compatibility_report.cpp src/semantic_manifest.cpp src/physical_codec.cpp
    src/source_compiler_graph_proof.cpp src/token_contract.cpp src/artifact_set.cpp src/mmap.cpp src/gguf.cpp src/token_program.cpp)
target_include_directories(test_cli PRIVATE src tests)
target_compile_definitions(test_cli PRIVATE
    LAPLACE_BINARY_PATH="${CMAKE_BINARY_DIR}/laplace")
add_dependencies(test_cli laplace)
add_test(NAME test_cli COMMAND test_cli)

add_executable(test_generation_loop tests/test_generation_loop.cpp
    src/generation_loop.cpp)
target_include_directories(test_generation_loop PRIVATE src tests)
add_test(NAME test_generation_loop COMMAND test_generation_loop)

add_executable(test_laplace_kv tests/test_laplace_kv.cpp
    src/laplace_kv.cpp src/ops.cpp)
target_include_directories(test_laplace_kv PRIVATE src tests)
if(APPLE)
    target_link_libraries(test_laplace_kv PRIVATE ${LAPLACE_ACCELERATE_FW})
endif()
add_test(NAME test_laplace_kv COMMAND test_laplace_kv)

add_executable(test_laplace_kv_q4
    tests/test_laplace_kv_q4.cpp
    src/laplace_kv_q4.cpp
    src/ops.cpp)
target_include_directories(test_laplace_kv_q4 PRIVATE src)
if(APPLE)
    target_link_libraries(test_laplace_kv_q4 PRIVATE ${LAPLACE_APPLE_FW})
endif()
add_test(NAME test_laplace_kv_q4 COMMAND test_laplace_kv_q4)

add_executable(test_laplace_kv_adaptive
    tests/test_laplace_kv_adaptive.cpp
    src/laplace_kv_adaptive.cpp
    src/laplace_kv.cpp
    src/laplace_kv_q4.cpp
    src/ops.cpp)
target_include_directories(test_laplace_kv_adaptive PRIVATE src tests)
if(APPLE)
    target_link_libraries(test_laplace_kv_adaptive PRIVATE
        ${LAPLACE_ACCELERATE_FW})
endif()
add_test(NAME test_laplace_kv_adaptive COMMAND test_laplace_kv_adaptive)

add_executable(test_laplace_moe tests/test_laplace_moe.cpp
    src/laplace_moe.cpp)
target_include_directories(test_laplace_moe PRIVATE src tests)
add_test(NAME test_laplace_moe COMMAND test_laplace_moe)

add_executable(test_moe_worklist tests/test_moe_worklist.cpp
    src/moe_worklist.cpp src/physical_codec.cpp src/semantic_model.cpp
    src/artifact_set.cpp src/compatibility_report.cpp src/mmap.cpp)
target_include_directories(test_moe_worklist PRIVATE src tests)
add_test(NAME test_moe_worklist COMMAND test_moe_worklist)

add_executable(bench_kvcache tests/bench_kvcache.cpp
    src/laplace_kv.cpp src/laplace_kv_adaptive.cpp src/laplace_kv_q4.cpp
    src/kvcache.cpp src/ops.cpp)
target_include_directories(bench_kvcache PRIVATE src)
if(APPLE)
    target_link_libraries(bench_kvcache PRIVATE ${LAPLACE_ACCELERATE_FW})
endif()

# Tools
add_executable(dump_meta tools/dump_meta.cpp src/gguf.cpp src/mmap.cpp)
target_include_directories(dump_meta PRIVATE src)

# Semantic manifests and the transitional CodecProgram adapter both consume
# the same bounded certificate implementation. Keep the shared source list in
# one place so focused test targets cannot accidentally link an older authority.
set(LAPLACE_CODEC_CERTIFICATE_CONSUMERS
    laplace
    test_codec_program
    test_codec_binding
    test_semantic_manifest
    test_source_schema_authority
    test_runtime_package
    test_product_package
    test_gguf_product_compiler
    test_product_authority
    test_gguf_import
    test_gguf_index
    test_external_golden
    test_runtime_session
    test_metal_capability_snapshot
    test_canonical_metal
    bench_canonical_metal
    test_qwen_canonical_parity
    test_runtime_session_product
    test_cli)
foreach(target IN LISTS LAPLACE_CODEC_CERTIFICATE_CONSUMERS)
    if(TARGET ${target})
        target_sources(${target} PRIVATE src/codec_certificate.cpp)
    endif()
endforeach()

set(LAPLACE_SEMANTIC_DISPATCH_CONSUMERS
    test_runtime_session
    test_canonical_metal
    bench_canonical_metal
    test_qwen_canonical_parity
    test_runtime_session_product)
foreach(target IN LISTS LAPLACE_SEMANTIC_DISPATCH_CONSUMERS)
    if(TARGET ${target})
        target_sources(${target} PRIVATE src/semantic_dispatch_program.cpp)
    endif()
endforeach()

function(laplace_assert_product_source_boundary_at_marker)
    set(actual_sources)
    set(linked_product_targets)
    foreach(product_target IN LISTS LAPLACE_PRODUCT_TARGETS)
        if(NOT TARGET "${product_target}")
            message(FATAL_ERROR "declared product target does not exist: ${product_target}")
        endif()
        get_target_property(target_sources "${product_target}" SOURCES)
        foreach(source IN LISTS target_sources)
            if(source MATCHES "\\$<")
                message(FATAL_ERROR "unresolved generator-expression product source: ${source}")
            endif()
            get_filename_component(absolute_source "${source}" ABSOLUTE
                BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
            file(RELATIVE_PATH relative_source "${CMAKE_CURRENT_SOURCE_DIR}"
                "${absolute_source}")
            list(APPEND actual_sources "${relative_source}")
        endforeach()
        get_target_property(target_links "${product_target}" LINK_LIBRARIES)
        if(target_links)
            foreach(link IN LISTS target_links)
                if(link MATCHES "\\$<")
                    message(FATAL_ERROR "unresolved generator-expression product link: ${link}")
                endif()
                if(TARGET "${link}")
                    get_target_property(resolved_target "${link}" ALIASED_TARGET)
                    if(NOT resolved_target)
                        set(resolved_target "${link}")
                    endif()
                    get_target_property(imported_target "${resolved_target}" IMPORTED)
                    if(NOT imported_target)
                        list(APPEND linked_product_targets "${link}")
                    endif()
                endif()
            endforeach()
        endif()
    endforeach()

    set(unique_product_targets ${LAPLACE_PRODUCT_TARGETS})
    list(REMOVE_DUPLICATES unique_product_targets)
    if(NOT unique_product_targets STREQUAL LAPLACE_PRODUCT_TARGETS)
        message(FATAL_ERROR "LAPLACE_PRODUCT_TARGETS contains duplicate targets")
    endif()

    list(REMOVE_DUPLICATES actual_sources)
    list(SORT actual_sources)
    set(declared_sources ${LAPLACE_PRODUCT_SOURCES})
    set(unique_declared_sources ${declared_sources})
    list(REMOVE_DUPLICATES unique_declared_sources)
    if(NOT unique_declared_sources STREQUAL declared_sources)
        message(FATAL_ERROR "LAPLACE_PRODUCT_SOURCES contains duplicate sources")
    endif()
    list(REMOVE_DUPLICATES declared_sources)
    list(SORT declared_sources)
    if(NOT actual_sources STREQUAL declared_sources)
        message(FATAL_ERROR "LAPLACE_PRODUCT_SOURCES does not match the product target closure")
    endif()
    foreach(source IN LISTS LAPLACE_QUALIFICATION_ONLY_SOURCES)
        if(source IN_LIST actual_sources)
            message(FATAL_ERROR "qualification-only source linked by product: ${source}")
        endif()
    endforeach()

    set(unique_qualification_sources ${LAPLACE_QUALIFICATION_ONLY_SOURCES})
    list(REMOVE_DUPLICATES unique_qualification_sources)
    if(NOT unique_qualification_sources STREQUAL LAPLACE_QUALIFICATION_ONLY_SOURCES)
        message(FATAL_ERROR "LAPLACE_QUALIFICATION_ONLY_SOURCES contains duplicate sources")
    endif()

    list(REMOVE_DUPLICATES linked_product_targets)
    list(SORT linked_product_targets)
    set(declared_linked_targets ${LAPLACE_PRODUCT_TARGETS})
    list(REMOVE_ITEM declared_linked_targets laplace)
    list(SORT declared_linked_targets)
    if(NOT linked_product_targets STREQUAL declared_linked_targets)
        message(FATAL_ERROR "LAPLACE_PRODUCT_TARGETS does not match linked product libraries")
    endif()

    set(reachable_product_targets laplace)
    set(pending_product_targets laplace)
    while(pending_product_targets)
        list(POP_FRONT pending_product_targets current_product_target)
        get_target_property(current_product_links "${current_product_target}" LINK_LIBRARIES)
        foreach(link IN LISTS current_product_links)
            if("${link}" IN_LIST LAPLACE_PRODUCT_TARGETS AND
                    NOT "${link}" IN_LIST reachable_product_targets)
                list(APPEND reachable_product_targets "${link}")
                list(APPEND pending_product_targets "${link}")
            endif()
        endforeach()
    endwhile()
    list(SORT reachable_product_targets)
    set(declared_product_targets ${LAPLACE_PRODUCT_TARGETS})
    list(SORT declared_product_targets)
    if(NOT reachable_product_targets STREQUAL declared_product_targets)
        message(FATAL_ERROR "declared product target is not linked from laplace")
    endif()
endfunction()

# This assertion covers all target mutations above this marker. file(GENERATE)
# below evaluates final target properties and is the authoritative later-drift
# check run by test_universal_source_policy_target_closure.
laplace_assert_product_source_boundary_at_marker()

set(product_source_closure_content "")
foreach(product_target IN LISTS LAPLACE_PRODUCT_TARGETS)
    string(APPEND product_source_closure_content "target\t${product_target}\n")
    get_target_property(marker_target_links "${product_target}" LINK_LIBRARIES)
    foreach(marker_link IN LISTS marker_target_links)
        if(TARGET "${marker_link}")
            get_target_property(resolved_link_target "${marker_link}" ALIASED_TARGET)
            if(NOT resolved_link_target)
                set(resolved_link_target "${marker_link}")
            endif()
            get_target_property(imported_link_target "${resolved_link_target}" IMPORTED)
            if(imported_link_target)
                set(link_authority imported)
            else()
                set(link_authority local)
            endif()
        else()
            set(resolved_link_target "${marker_link}")
            set(link_authority item)
        endif()
        string(APPEND product_source_closure_content
            "linkmeta\t${marker_link}\t${resolved_link_target}\t${link_authority}\n")
    endforeach()
    string(APPEND product_source_closure_content
        "source\t$<JOIN:$<TARGET_PROPERTY:${product_target},SOURCES>,\nsource\t>\n")
    string(APPEND product_source_closure_content
        "link\t$<JOIN:$<TARGET_PROPERTY:${product_target},LINK_LIBRARIES>,\nlink\t>\n")
endforeach()
file(GENERATE OUTPUT "${LAPLACE_PRODUCT_SOURCE_CLOSURE_SNAPSHOT}"
    CONTENT "${product_source_closure_content}")
