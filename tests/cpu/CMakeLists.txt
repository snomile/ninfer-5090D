cmake_minimum_required(VERSION 3.28)
project(ninfer_cpu_regression LANGUAGES C CXX)

# Build the actual platform-independent implementation and its regular tests.
# GPU execution and OS-specific allocation are deliberately outside this entry.
get_filename_component(NINFER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
enable_testing()
set(NVTX3_INCLUDE_DIR "" CACHE PATH "NVTX c/include directory for runtime CPU tests")
set(CUDA_RUNTIME_INCLUDE_DIR "" CACHE PATH "Official CUDA runtime headers, optional; no GPU execution")
set(CUDA_CRT_INCLUDE_DIR "" CACHE PATH "Official CUDA compiler runtime headers, optional")

add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE "${NINFER_SOURCE_DIR}/third_party")
add_subdirectory("${NINFER_SOURCE_DIR}/third_party/xgrammar" xgrammar)
add_subdirectory("${NINFER_SOURCE_DIR}/third_party/llama-jinja" llama-jinja EXCLUDE_FROM_ALL)

function(cpu_includes target)
  target_include_directories(${target} PRIVATE
    "${NINFER_SOURCE_DIR}/include" "${NINFER_SOURCE_DIR}/src"
    "${NINFER_SOURCE_DIR}/third_party")
  if(NVTX3_INCLUDE_DIR)
    target_include_directories(${target} PRIVATE "${NVTX3_INCLUDE_DIR}")
  endif()
endfunction()

add_library(ninfer_cpu_text STATIC
  "${NINFER_SOURCE_DIR}/src/text/unicode.cpp"
  "${NINFER_SOURCE_DIR}/third_party/utf8proc/utf8proc.c"
  "${NINFER_SOURCE_DIR}/src/text/grammar.cpp"
  "${NINFER_SOURCE_DIR}/src/text/json_schema.cpp"
  "${NINFER_SOURCE_DIR}/src/text/schema_composition.cpp"
  "${NINFER_SOURCE_DIR}/src/text/json_input.cpp"
  "${NINFER_SOURCE_DIR}/src/runtime/engine/public_types.cpp")
cpu_includes(ninfer_cpu_text)
target_include_directories(ninfer_cpu_text PRIVATE
  "${NINFER_SOURCE_DIR}/third_party/xgrammar/cpp"
  "${NINFER_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson")
target_link_libraries(ninfer_cpu_text PUBLIC ninfer_xgrammar ninfer::json)

function(cpu_test name source)
  add_executable(${name} "${NINFER_SOURCE_DIR}/${source}")
  cpu_includes(${name})
  target_link_libraries(${name} PRIVATE ninfer_cpu_text ${ARGN})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

cpu_test(ninfer_grammar_test tests/test_grammar.cpp)
cpu_test(ninfer_json_schema_test tests/text/test_json_schema.cpp)
target_include_directories(ninfer_json_schema_test PRIVATE
  "${NINFER_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson")
cpu_test(ninfer_regex_choice_test tests/text/test_regex_choice.cpp)
cpu_test(ninfer_public_api_test tests/test_public_api.cpp)
add_test(NAME ninfer_json_schema_oracle_test
  COMMAND "${CMAKE_COMMAND}" -E env
    "NINFER_SCHEMA_PROBE=$<TARGET_FILE:ninfer_json_schema_test>"
    "${Python3_EXECUTABLE}" -B "${NINFER_SOURCE_DIR}/tests/text/test_json_schema.py")
add_test(NAME ninfer_regex_choice_oracle_test
  COMMAND "${CMAKE_COMMAND}" -E env
    "NINFER_REGEX_PROBE=$<TARGET_FILE:ninfer_regex_choice_test>"
    "${Python3_EXECUTABLE}" -B "${NINFER_SOURCE_DIR}/tests/text/test_regex_choice.py")

# Pure protocol implementations have no Engine, CUDA or network dependency.
if(EXISTS "${NINFER_SOURCE_DIR}/src/serve/request_error.cpp")
  add_library(ninfer_cpu_protocol STATIC
    "${NINFER_SOURCE_DIR}/src/serve/request_validation.cpp"
    "${NINFER_SOURCE_DIR}/src/serve/request_error.cpp"
    "${NINFER_SOURCE_DIR}/src/serve/openai_common.cpp"
    "${NINFER_SOURCE_DIR}/src/serve/openai_chat_request.cpp"
    "${NINFER_SOURCE_DIR}/src/serve/openai_chat_response.cpp"
    "${NINFER_SOURCE_DIR}/src/serve/openai_responses_request.cpp"
    "${NINFER_SOURCE_DIR}/src/serve/openai_responses_state.cpp"
    "${NINFER_SOURCE_DIR}/src/serve/openai_responses_store.cpp"
    "${NINFER_SOURCE_DIR}/src/serve/openai_responses_response.cpp"
    "${NINFER_SOURCE_DIR}/src/serve/anthropic_messages_request.cpp"
    "${NINFER_SOURCE_DIR}/src/serve/anthropic_messages_response.cpp"
    "${NINFER_SOURCE_DIR}/src/serve/translate.cpp")
  cpu_includes(ninfer_cpu_protocol)
  target_include_directories(ninfer_cpu_protocol PRIVATE
    "${NINFER_SOURCE_DIR}/third_party/cpp-httplib")
  target_link_libraries(ninfer_cpu_protocol PUBLIC ninfer_cpu_text)
  foreach(protocol IN ITEMS openai_schema openai_responses openai_responses_store anthropic_schema)
    cpu_test(ninfer_${protocol}_test tests/test_${protocol}.cpp ninfer_cpu_protocol)
  endforeach()
endif()

cpu_test(ninfer_scheduler_test tests/runtime/test_scheduler.cpp)
cpu_test(ninfer_resource_manager_test tests/runtime/test_resource_manager.cpp)
cpu_test(ninfer_context_cost_test tests/runtime/test_context_cost.cpp)
add_library(ninfer_cpu_context_cost STATIC
  "${NINFER_SOURCE_DIR}/src/runtime/engine/context_cache/context_cost.cpp"
  "${NINFER_SOURCE_DIR}/src/runtime/engine/context_cache/context_cost_defaults.cpp")
cpu_includes(ninfer_cpu_context_cost)
target_link_libraries(ninfer_cpu_context_cost PUBLIC ninfer_cpu_text)
target_link_libraries(ninfer_context_cost_test PRIVATE ninfer_cpu_context_cost)
target_link_libraries(ninfer_resource_manager_test PRIVATE ninfer_cpu_context_cost)
cpu_test(ninfer_metrics_test tests/test_metrics.cpp)
target_sources(ninfer_metrics_test PRIVATE "${NINFER_SOURCE_DIR}/src/serve/metrics.cpp")
include(frontend.cmake)

# Request logs receive an explicit environment; CUDA probing stays in device_environment.cpp.
add_library(ninfer_cpu_product STATIC
  "${NINFER_SOURCE_DIR}/src/runtime/engine/generation_handle.cpp"
  "${NINFER_SOURCE_DIR}/src/product/logging/logging.cpp"
  "${NINFER_SOURCE_DIR}/src/product/logging/pretty_format.cpp"
  "${NINFER_SOURCE_DIR}/src/serve/serve_options.cpp"
  "${NINFER_SOURCE_DIR}/src/serve/request_events.cpp"
  "${NINFER_SOURCE_DIR}/src/serve/operational_log.cpp"
  "${NINFER_SOURCE_DIR}/src/serve/request_log.cpp")
cpu_includes(ninfer_cpu_product)
target_include_directories(ninfer_cpu_product PUBLIC
  "${NINFER_SOURCE_DIR}/third_party/spdlog/include")
target_link_libraries(ninfer_cpu_product PUBLIC ninfer_cpu_protocol Threads::Threads)
cpu_test(ninfer_request_log_test tests/test_request_log.cpp ninfer_cpu_product)
cpu_test(ninfer_serve_options_test tests/test_serve_options.cpp ninfer_cpu_product)
cpu_test(ninfer_cli_options_test tests/test_cli_options.cpp ninfer_cpu_product)
target_sources(ninfer_cli_options_test PRIVATE "${NINFER_SOURCE_DIR}/apps/cli/options.cpp")
target_include_directories(ninfer_cli_options_test PRIVATE "${NINFER_SOURCE_DIR}/apps/cli")
cpu_test(ninfer_memory_options_test tests/test_memory_options.cpp ninfer_cpu_product)
target_sources(ninfer_memory_options_test PRIVATE "${NINFER_SOURCE_DIR}/apps/cli/options.cpp")
target_include_directories(ninfer_memory_options_test PRIVATE "${NINFER_SOURCE_DIR}/apps/cli")
cpu_test(ninfer_engine_failure_monitor_test tests/test_engine_failure_monitor.cpp Threads::Threads)
target_include_directories(ninfer_engine_failure_monitor_test PRIVATE
  "${NINFER_SOURCE_DIR}/third_party/cpp-httplib")
set_tests_properties(ninfer_engine_failure_monitor_test PROPERTIES TIMEOUT 30)
if(CUDA_RUNTIME_INCLUDE_DIR AND CUDA_CRT_INCLUDE_DIR)
  cpu_test(ninfer_kv_capacity_test tests/runtime/test_kv_capacity.cpp)
  target_include_directories(ninfer_kv_capacity_test PRIVATE
    "${CUDA_RUNTIME_INCLUDE_DIR}" "${CUDA_CRT_INCLUDE_DIR}")
  target_sources(ninfer_kv_capacity_test PRIVATE
    "${NINFER_SOURCE_DIR}/src/runtime/engine/kv_capacity.cpp"
    "${NINFER_SOURCE_DIR}/src/runtime/engine/startup_memory.cpp"
    "${NINFER_SOURCE_DIR}/src/core/device_memory.cpp")
else()
  message(STATUS "EXCLUDED: startup memory/KV capacity test requires official CUDA headers")
endif()

# Keep every fixture under the worktree's ignored scratch directory.
get_property(ninfer_cpu_tests DIRECTORY PROPERTY TESTS)
file(MAKE_DIRECTORY "${NINFER_SOURCE_DIR}/.tmp/cpu-tests")
set_tests_properties(${ninfer_cpu_tests} PROPERTIES
  WORKING_DIRECTORY "${NINFER_SOURCE_DIR}/.tmp/cpu-tests"
  ENVIRONMENT "TMPDIR=${NINFER_SOURCE_DIR}/.tmp/cpu-tests;PYTHONDONTWRITEBYTECODE=1")
