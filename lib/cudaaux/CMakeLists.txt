# Project configuration
project(cudaaux)

# Include directories
include_directories(include)

# Add library
cuda_add_library(cudaaux
  src/cudaaux.cu)
