# Project configuration
cmake_minimum_required(VERSION 2.8)
project(plugs)
set(CMAKE_MODULE_PATH ${ROOT_DIR}/cmake)
set(EXECUTABLE_OUTPUT_PATH ../../bin)
set(LIBRARY_OUTPUT_PATH ../../bin)


# Compiler options
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -pedantic -O3")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Wall -pedantic -O3")


# Set Doxygen
set(DOXYFILE_OUTPUT_DIR "doc")
set(DOXYFILE_LATEX "OFF")
include(UseDoxygen OPTIONAL)

if (DOXYGEN_FOUND)
  message("DOXYGEN found")
else (DOXYGEN_FOUND)
  message("DOXYGEN not found")
endif (DOXYGEN_FOUND)


# Set CUDA
set(CUDA_NVCC_FLAGS "-O3;")
find_package(CUDA)

if (CUDA_FOUND)
  message("CUDA found")
  add_definitions(-DCUDA)
  set(DOXYFILE_PREDEFINED "CUDA")
else (CUDA_FOUND)
  message("CUDA not found")
endif (CUDA_FOUND)


# Set subdirectories
add_subdirectory(lib)
add_subdirectory(exe)

