cmake_minimum_required(VERSION 3.10)

project(LAMBDA VERSION 1.0 LANGUAGES CXX)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Force Release mode by default for max performance (-O3 and -DNDEBUG)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(Eigen3 REQUIRED)

set(SOURCES
    src/main.cpp
    src/ldlt.cpp
    src/reduce.cpp
    src/permute.cpp
    src/search.cpp
)

add_executable(lambda ${SOURCES})

target_link_libraries(lambda Eigen3::Eigen)

target_include_directories(lambda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)