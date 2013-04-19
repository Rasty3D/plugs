# Project configuration
project(nux)

# Include directories
include_directories(${ROOT_DIR}/lib/aux/include)
include_directories(${ROOT_DIR}/lib/cudaaux/include)
include_directories(${ROOT_DIR}/lib/mem/include)
include_directories(include)

# Add library
add_library(nux
  src/nuxmemorymanager.cpp
  src/nuxpluginmanager.cpp
  src/nuxplugin.cpp
  src/nuxcontext.cpp)

# Add test
add_executable(test_plugin
  test/test_plugin.cpp)
target_link_libraries(test_plugin
  nux dl)
add_test(test_plugin
  ${EXECUTABLE_OUTPUT_PATH}/test_plugin)