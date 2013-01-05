# Project configuration
project(aux)

# Include directories
include_directories(include)

# Add library
add_library(aux
  src/trie.cpp
  src/processor.cpp
  src/bytebuffer.cpp)

# Add test
add_executable(test_bytebuffer
  test/test_bytebuffer.cpp)
target_link_libraries(test_bytebuffer
  aux)
