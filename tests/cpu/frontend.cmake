include_guard(GLOBAL)

# Included by tests/cpu/CMakeLists.txt after cpu_test and ninfer_cpu_text exist.
if(NOT TARGET ninfer_cpu_text OR NOT TARGET ninfer_jinja)
  message(FATAL_ERROR "frontend.cmake requires the CPU text and llama-jinja targets")
endif()

find_path(NINFER_CPU_HWLOC_INCLUDE_DIR hwloc.h
  HINTS /opt/homebrew/opt/hwloc/include REQUIRED)
find_library(NINFER_CPU_HWLOC_LIBRARY NAMES hwloc
  HINTS /opt/homebrew/opt/hwloc/lib REQUIRED)
find_path(NINFER_CPU_FFMPEG_INCLUDE_DIR libavformat/avformat.h
  HINTS /opt/homebrew/opt/ffmpeg/include REQUIRED)
set(ninfer_cpu_ffmpeg_libraries)
foreach(component IN ITEMS avformat avcodec avutil swscale)
  find_library(NINFER_CPU_${component}_LIBRARY NAMES ${component}
    HINTS /opt/homebrew/opt/ffmpeg/lib REQUIRED)
  list(APPEND ninfer_cpu_ffmpeg_libraries "${NINFER_CPU_${component}_LIBRARY}")
endforeach()

if(APPLE)
  set(ninfer_cpu_topology_source "${CMAKE_CURRENT_LIST_DIR}/support/darwin_topology.cpp")
  message(STATUS "CPU frontend: TEST-ONLY Darwin hwloc discovery, real TaskPool/start_workers")
  message(STATUS "EXCLUDED on Darwin: CPU system/executor/TaskPool affinity test binaries")
  message(STATUS "Frontend must explicitly EXCLUDE test_media_auto_threads_respect_affinity "
    "under NINFER_CPU_TEST_NO_AFFINITY; the adapter throws, never emulates binding")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
       CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(ninfer_cpu_topology_source "${NINFER_SOURCE_DIR}/src/core/cpu/system.cpp")
else()
  message(FATAL_ERROR "CPU frontend topology support requires Darwin or Linux x86-64")
endif()

add_library(ninfer_cpu_frontend_workers STATIC
  ${ninfer_cpu_topology_source}
  "${NINFER_SOURCE_DIR}/src/core/cpu/workers.cpp"
  "${NINFER_SOURCE_DIR}/src/core/cpu/task_pool.cpp")
cpu_includes(ninfer_cpu_frontend_workers)
target_include_directories(ninfer_cpu_frontend_workers PRIVATE "${NINFER_CPU_HWLOC_INCLUDE_DIR}")
target_link_libraries(ninfer_cpu_frontend_workers PUBLIC Threads::Threads
  "${NINFER_CPU_HWLOC_LIBRARY}")

# These metadata implementations are pure C++; schema retains its real weight parser.
add_library(ninfer_cpu_frontend_resources STATIC
  "${NINFER_SOURCE_DIR}/src/artifact/schema.cpp"
  "${NINFER_SOURCE_DIR}/src/core/weight_view.cpp"
  "${NINFER_SOURCE_DIR}/src/core/tensor.cpp"
  "${NINFER_SOURCE_DIR}/src/core/dtype.cpp"
  "${NINFER_SOURCE_DIR}/src/core/memory.cpp")
cpu_includes(ninfer_cpu_frontend_resources)
target_link_libraries(ninfer_cpu_frontend_resources PUBLIC ninfer::json)

add_library(ninfer_cpu_media_decode STATIC "${NINFER_SOURCE_DIR}/src/media/decode/decode.cpp")
cpu_includes(ninfer_cpu_media_decode)
target_include_directories(ninfer_cpu_media_decode PRIVATE "${NINFER_CPU_FFMPEG_INCLUDE_DIR}")
target_link_libraries(ninfer_cpu_media_decode PUBLIC ${ninfer_cpu_ffmpeg_libraries})

target_sources(ninfer_jinja PRIVATE "${NINFER_SOURCE_DIR}/src/text/jinja.cpp")
cpu_includes(ninfer_jinja)
target_link_libraries(ninfer_jinja PUBLIC ninfer_cpu_text ninfer::json)

add_library(ninfer_cpu_frontend STATIC
  "${NINFER_SOURCE_DIR}/src/models/common/qwen/prefix_identity.cpp")
foreach(source IN ITEMS frontend output_session chat_template prompt_layout digest processor
    media_cache tool_call_parser tool_contract tool_grammar resources tokenizer)
  target_sources(ninfer_cpu_frontend PRIVATE
    "${NINFER_SOURCE_DIR}/src/models/common/qwen/frontend/${source}.cpp")
endforeach()
cpu_includes(ninfer_cpu_frontend)
target_include_directories(ninfer_cpu_frontend PRIVATE
  "${NINFER_SOURCE_DIR}/third_party/xgrammar/cpp"
  "${NINFER_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson")
target_link_libraries(ninfer_cpu_frontend PUBLIC ninfer_cpu_text ninfer_jinja
  ninfer_cpu_frontend_resources ninfer_cpu_media_decode ninfer_cpu_frontend_workers)

cpu_test(ninfer_qwen_frontend_test tests/models/common/qwen/test_frontend.cpp ninfer_cpu_frontend)
target_compile_definitions(ninfer_qwen_frontend_test PRIVATE NINFER_SOURCE_DIR="${NINFER_SOURCE_DIR}")
if(APPLE)
  target_compile_definitions(ninfer_qwen_frontend_test PRIVATE NINFER_CPU_TEST_NO_AFFINITY=1)
endif()
cpu_test(ninfer_qwen_tool_constraints_test tests/models/common/qwen/test_tool_constraints.cpp
  ninfer_cpu_frontend)
cpu_test(ninfer_tool_call_parser_test tests/models/common/qwen/test_tool_call_parser.cpp
  ninfer_cpu_frontend)
add_test(NAME ninfer_qwen_tool_schema_oracle_test
  COMMAND "${CMAKE_COMMAND}" -E env
    "NINFER_TOOL_PROBE=$<TARGET_FILE:ninfer_qwen_tool_constraints_test>"
    "${Python3_EXECUTABLE}" -B
    "${NINFER_SOURCE_DIR}/tests/models/common/qwen/test_tool_schema.py")

# Existing frontend fixtures use std::filesystem::temp_directory_path().
set(ninfer_cpu_frontend_tmp "${NINFER_SOURCE_DIR}/.tmp/cpu-frontend")
file(MAKE_DIRECTORY "${ninfer_cpu_frontend_tmp}")
set_tests_properties(ninfer_qwen_frontend_test ninfer_qwen_tool_constraints_test
  ninfer_tool_call_parser_test ninfer_qwen_tool_schema_oracle_test PROPERTIES
  WORKING_DIRECTORY "${ninfer_cpu_frontend_tmp}"
  ENVIRONMENT "TMPDIR=${ninfer_cpu_frontend_tmp};PYTHONDONTWRITEBYTECODE=1"
  LABELS "cpu;real-frontend")
