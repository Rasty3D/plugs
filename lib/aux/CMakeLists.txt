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
add_test(test_bytebuffer
  ${EXECUTABLE_OUTPUT_PATH}/test_bytebuffer)
  
# Add test
add_executable(test_trie
  test/test_trie.cpp)
target_link_libraries(test_trie
  aux)
add_test(test_trie
  ${EXECUTABLE_OUTPUT_PATH}/test_trie)
