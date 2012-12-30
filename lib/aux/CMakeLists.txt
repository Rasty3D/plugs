# Project configuration
project(aux)

# Include directories
include_directories(include)

# Add library
if (CUDA_FOUND)
  add_library(aux
    src/trie.cpp
    src/processor.cpp)
endif (CUDA_FOUND)
