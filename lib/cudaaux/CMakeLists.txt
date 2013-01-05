# Project configuration
project(cudaaux)

# Include directories
include_directories(include)

# Add library
if (CUDA_FOUND)
  cuda_add_library(cudaaux
    src/cudaaux.cu)
endif (CUDA_FOUND)
