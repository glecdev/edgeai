cmake_minimum_required(VERSION 3.22.1)

project("glec_dtg")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SNPE SDK path (must be set locally)
# set(SNPE_ROOT "/path/to/snpe-sdk")

# Add UART reader library
add_library(uart_reader SHARED
        uart_reader.cpp
        can_parser.cpp
)

# Add SNPE engine library
add_library(snpe_engine SHARED
        snpe_engine.cpp
)

# Find required libraries
find_library(log-lib log)
find_library(android-lib android)

# Link libraries for UART reader
target_link_libraries(uart_reader
        ${log-lib}
        ${android-lib}
)

# Link libraries for SNPE engine
target_link_libraries(snpe_engine
        ${log-lib}
        ${android-lib}
        # TODO: Add SNPE libraries when SDK is available
        # ${SNPE_ROOT}/lib/aarch64-android/libSNPE.so
)
