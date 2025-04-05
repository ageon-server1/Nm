cmake_minimum_required(VERSION 3.10)
project(CodespaceEncryption LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package (if not automatically found)
find_package(CUDA REQUIRED)

# Include directories (adjust if needed)
include_directories(${CUDA_INCLUDE_DIRS})

# Add executable target
add_executable(codespace_app main.cpp)

# Link CUDA runtime
target_link_libraries(codespace_app ${CUDA_LIBRARIES})
