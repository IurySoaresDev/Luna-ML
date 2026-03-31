cmake_minimum_required(VERSION 3.16)
project(LunaML VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Collect all library sources
file(GLOB_RECURSE LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Static library
add_library(lunaml STATIC ${LIB_SOURCES})
target_include_directories(lunaml PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Examples
option(NF_BUILD_EXAMPLES "Build example programs" ON)
if(NF_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
option(NF_BUILD_TESTS "Build tests" ON)
if(NF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
