find_path(CPU_HWLOC_INCLUDE hwloc.h HINTS /opt/homebrew/opt/hwloc/include REQUIRED)
find_library(CPU_HWLOC_LIBRARY hwloc HINTS /opt/homebrew/opt/hwloc/lib REQUIRED)
find_path(CPU_FFMPEG_INCLUDE libavformat/avformat.h HINTS /opt/homebrew/opt/ffmpeg/include REQUIRED)
find_path(CPU_JPEG_INCLUDE jpeglib.h HINTS /opt/homebrew/opt/jpeg-turbo/include REQUIRED)
find_library(CPU_JPEG_LIBRARY jpeg HINTS /opt/homebrew/opt/jpeg-turbo/lib REQUIRED)
foreach(component IN ITEMS avformat avcodec avutil swscale)
  find_library(CPU_${component}_LIBRARY ${component} HINTS /opt/homebrew/opt/ffmpeg/lib REQUIRED)
  list(APPEND cpu_media_libraries "${CPU_${component}_LIBRARY}")
endforeach()

# Build the actual portable worker implementation. On Darwin only topology discovery
# is substituted; thread binding throws and its Linux-specific test is excluded.
set(cpu_topology "${NINFER_SOURCE_DIR}/src/core/cpu/system.cpp")
if(APPLE)
  set(cpu_topology "${CMAKE_CURRENT_LIST_DIR}/darwin_topology.cpp")
endif()
add_library(ninfer_cpu_frontend_support STATIC ${cpu_topology}
  "${NINFER_SOURCE_DIR}/src/core/cpu/workers.cpp"
  "${NINFER_SOURCE_DIR}/src/core/cpu/task_pool.cpp"
  "${NINFER_SOURCE_DIR}/src/artifact/schema.cpp"
  "${NINFER_SOURCE_DIR}/src/core/weight_view.cpp"
  "${NINFER_SOURCE_DIR}/src/core/tensor.cpp"
  "${NINFER_SOURCE_DIR}/src/core/dtype.cpp"
  "${NINFER_SOURCE_DIR}/src/core/memory.cpp"
  "${NINFER_SOURCE_DIR}/src/core/layout.cpp"
  "${NINFER_SOURCE_DIR}/src/media/decode/decode.cpp"
  "${NINFER_SOURCE_DIR}/src/media/decode/jpeg.cpp")
cpu_includes(ninfer_cpu_frontend_support)
target_include_directories(ninfer_cpu_frontend_support PRIVATE
  "${CPU_HWLOC_INCLUDE}" "${CPU_FFMPEG_INCLUDE}" "${CPU_JPEG_INCLUDE}")
target_link_libraries(ninfer_cpu_frontend_support PUBLIC ninfer::json Threads::Threads
  "${CPU_HWLOC_LIBRARY}" "${CPU_JPEG_LIBRARY}" ${cpu_media_libraries})

target_sources(ninfer_jinja PRIVATE "${NINFER_SOURCE_DIR}/src/text/jinja.cpp")
cpu_includes(ninfer_jinja)
target_link_libraries(ninfer_jinja PUBLIC ninfer_cpu_text)
add_library(ninfer_cpu_frontend STATIC
  "${NINFER_SOURCE_DIR}/src/models/common/qwen/prefix_identity.cpp"
  "${NINFER_SOURCE_DIR}/src/models/common/vision/window.cpp")
foreach(source IN ITEMS resources chat_template prompt output_resources output_session tools
    image_transform media_cache digest)
  target_sources(ninfer_cpu_frontend PRIVATE
    "${NINFER_SOURCE_DIR}/src/models/common/frontend/${source}.cpp")
endforeach()
foreach(source IN ITEMS frontend output chat_template prompt_layout processor tool_call_parser
    tool_contract tool_grammar tool_name_constraint resources)
  target_sources(ninfer_cpu_frontend PRIVATE
    "${NINFER_SOURCE_DIR}/src/models/common/qwen/frontend/${source}.cpp")
endforeach()
cpu_includes(ninfer_cpu_frontend)
target_include_directories(ninfer_cpu_frontend PRIVATE
  "${NINFER_SOURCE_DIR}/third_party/xgrammar/cpp"
  "${NINFER_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson")
target_link_libraries(ninfer_cpu_frontend PUBLIC ninfer_cpu_frontend_support ninfer_cpu_text
  ninfer_jinja)
cpu_test(ninfer_qwen_frontend_test tests/models/common/qwen/test_frontend.cpp ninfer_cpu_frontend)
if(APPLE)
  target_compile_definitions(ninfer_qwen_frontend_test PRIVATE NINFER_CPU_TEST_NO_AFFINITY=1)
endif()
cpu_test(ninfer_qwen_tool_constraints_test tests/models/common/qwen/test_tool_constraints.cpp
  ninfer_cpu_frontend)
cpu_test(ninfer_tool_call_parser_test tests/models/common/qwen/test_tool_call_parser.cpp
  ninfer_cpu_frontend)
add_test(NAME ninfer_qwen_tool_schema_oracle_test COMMAND "${CMAKE_COMMAND}" -E env
  "NINFER_TOOL_PROBE=$<TARGET_FILE:ninfer_qwen_tool_constraints_test>"
  "${Python3_EXECUTABLE}" -B "${NINFER_SOURCE_DIR}/tests/models/common/qwen/test_tool_schema.py")
