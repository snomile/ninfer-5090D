cmake_minimum_required(VERSION 3.28)
project(ninfer_cpu_regression LANGUAGES C CXX)
get_filename_component(NINFER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
enable_testing()

add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE "${NINFER_SOURCE_DIR}/third_party")
add_subdirectory("${NINFER_SOURCE_DIR}/third_party/xgrammar" xgrammar)
add_subdirectory("${NINFER_SOURCE_DIR}/third_party/llama-jinja" llama-jinja EXCLUDE_FROM_ALL)
add_subdirectory("${NINFER_SOURCE_DIR}/third_party/spdlog" spdlog EXCLUDE_FROM_ALL)

function(cpu_includes target)
  target_include_directories(${target} PUBLIC "${NINFER_SOURCE_DIR}/include"
    "${NINFER_SOURCE_DIR}/src" "${NINFER_SOURCE_DIR}/third_party")
endfunction()

add_library(ninfer_cpu_text STATIC)
foreach(source IN ITEMS unicode tokenizer grammar json_schema schema_composition schema_types json_input)
  target_sources(ninfer_cpu_text PRIVATE "${NINFER_SOURCE_DIR}/src/text/${source}.cpp")
endforeach()
target_sources(ninfer_cpu_text PRIVATE
  "${NINFER_SOURCE_DIR}/third_party/utf8proc/utf8proc.c"
  "${NINFER_SOURCE_DIR}/src/runtime/engine/public_types.cpp")
cpu_includes(ninfer_cpu_text)
target_include_directories(ninfer_cpu_text PRIVATE
  "${NINFER_SOURCE_DIR}/third_party/xgrammar/cpp"
  "${NINFER_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson")
target_link_libraries(ninfer_cpu_text PUBLIC ninfer_xgrammar ninfer::json)

add_library(ninfer_cpu_product STATIC
  "${NINFER_SOURCE_DIR}/src/distributed/deployment.cpp"
  "${NINFER_SOURCE_DIR}/src/product/logging/logging.cpp"
  "${NINFER_SOURCE_DIR}/src/product/logging/configuration.cpp"
  "${NINFER_SOURCE_DIR}/src/product/logging/pretty_format.cpp")
foreach(source IN ITEMS options schema file configuration)
  target_sources(ninfer_cpu_product PRIVATE
    "${NINFER_SOURCE_DIR}/src/product/config/${source}.cpp")
endforeach()
cpu_includes(ninfer_cpu_product)
target_include_directories(ninfer_cpu_product PRIVATE "${NINFER_SOURCE_DIR}/third_party/tomlplusplus")
target_link_libraries(ninfer_cpu_product PUBLIC ninfer_cpu_text spdlog::spdlog Threads::Threads)

add_library(ninfer_cpu_protocol STATIC)
foreach(source IN ITEMS request_validation request_error openai_common openai_chat_request
    openai_chat_response openai_responses_request openai_responses_state openai_responses_store
    openai_responses_response anthropic_messages_request anthropic_messages_response translate)
  target_sources(ninfer_cpu_protocol PRIVATE "${NINFER_SOURCE_DIR}/src/serve/${source}.cpp")
endforeach()
cpu_includes(ninfer_cpu_protocol)
target_include_directories(ninfer_cpu_protocol PUBLIC "${NINFER_SOURCE_DIR}/third_party/cpp-httplib")
target_link_libraries(ninfer_cpu_protocol PUBLIC ninfer_cpu_product)

function(cpu_test name source)
  add_executable(${name} "${NINFER_SOURCE_DIR}/${source}")
  cpu_includes(${name})
  target_compile_definitions(${name} PRIVATE NINFER_SOURCE_DIR="${NINFER_SOURCE_DIR}")
  target_link_libraries(${name} PRIVATE ninfer_cpu_text ${ARGN})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

cpu_test(ninfer_grammar_test tests/test_grammar.cpp)
cpu_test(ninfer_json_schema_test tests/text/test_json_schema.cpp)
target_include_directories(ninfer_json_schema_test PRIVATE
  "${NINFER_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson")
cpu_test(ninfer_regex_choice_test tests/text/test_regex_choice.cpp)
cpu_test(ninfer_public_api_test tests/test_public_api.cpp)
foreach(protocol IN ITEMS openai_schema openai_responses openai_responses_store anthropic_schema)
  cpu_test(ninfer_${protocol}_test tests/test_${protocol}.cpp ninfer_cpu_protocol)
endforeach()
cpu_test(ninfer_cli_options_test tests/test_cli_options.cpp ninfer_cpu_product)
cpu_test(ninfer_serve_options_test tests/test_serve_options.cpp ninfer_cpu_protocol)
cpu_test(ninfer_engine_failure_monitor_test tests/test_engine_failure_monitor.cpp Threads::Threads)
target_include_directories(ninfer_engine_failure_monitor_test PRIVATE
  "${NINFER_SOURCE_DIR}/third_party/cpp-httplib")
set_tests_properties(ninfer_engine_failure_monitor_test PROPERTIES TIMEOUT 30)
foreach(probe IN ITEMS json_schema regex_choice)
  if(probe STREQUAL "json_schema")
    set(variable NINFER_SCHEMA_PROBE)
  else()
    set(variable NINFER_REGEX_PROBE)
  endif()
  add_test(NAME ninfer_${probe}_oracle_test COMMAND "${CMAKE_COMMAND}" -E env
    "${variable}=$<TARGET_FILE:ninfer_${probe}_test>"
    "${Python3_EXECUTABLE}" -B "${NINFER_SOURCE_DIR}/tests/text/test_${probe}.py")
endforeach()

include(frontend.cmake)

add_library(ninfer_cpu_runtime STATIC
  "${NINFER_SOURCE_DIR}/src/core/context_pool.cpp"
  "${NINFER_SOURCE_DIR}/src/runtime/engine/context_capacity.cpp"
  "${NINFER_SOURCE_DIR}/src/runtime/engine/startup_memory.cpp"
  "${NINFER_SOURCE_DIR}/src/runtime/engine/context_cache/context_cost.cpp"
  "${NINFER_SOURCE_DIR}/src/runtime/engine/context_cache/context_cost_defaults.cpp"
  "${NINFER_SOURCE_DIR}/src/runtime/engine/context_cache/capture_policy.cpp"
  "${NINFER_SOURCE_DIR}/src/runtime/engine/context_cache/placement_policy.cpp"
  "${NINFER_SOURCE_DIR}/src/runtime/observability/execution.cpp"
  "${NINFER_SOURCE_DIR}/src/product/statistics.cpp"
  "${NINFER_SOURCE_DIR}/src/serve/metrics.cpp")
cpu_includes(ninfer_cpu_runtime)
target_link_libraries(ninfer_cpu_runtime PUBLIC ninfer_cpu_product ninfer_cpu_frontend_support)
foreach(test IN ITEMS scheduler context_cost placement_policy resource_manager context_capacity)
  cpu_test(ninfer_${test}_test tests/runtime/test_${test}.cpp ninfer_cpu_runtime)
endforeach()
cpu_test(ninfer_metrics_test tests/test_metrics.cpp ninfer_cpu_runtime)

get_property(cpu_tests DIRECTORY PROPERTY TESTS)
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/fixtures")
set_tests_properties(${cpu_tests} PROPERTIES
  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/fixtures"
  ENVIRONMENT "TMPDIR=${PROJECT_BINARY_DIR}/fixtures;PYTHONDONTWRITEBYTECODE=1")
