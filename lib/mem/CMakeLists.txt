# Project configuration
project(mem)

# Include directories
include_directories(${ROOT_DIR}/lib/aux/include)
include_directories(${ROOT_DIR}/lib/cudaaux/include)
include_directories(include)

# Add library
add_library(mem
  src/mem.cpp
  src/memblocks.cpp)

# Add CUDA library
if (CUDA_FOUND)
  cuda_add_library(cudamem
    src/cudamem.cu)
endif (CUDA_FOUND)
