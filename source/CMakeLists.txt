cmake_minimum_required(VERSION 3.16)
project(Domainwall)
include(FetchContent)

find_package(Kokkos QUIET)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_library(FFTW_LIB fftw3 REQUIRED)

if(NOT Kokkos_FOUND)
  message(STATUS "Kokkos not found, fetching...")

    FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        5.1.0   # or your preferred version
    )

    # Set backend BEFORE making available
    set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(kokkos)
endif()

add_executable(dw.x
    main.cpp
    parser.cpp
    parameter_builder.cpp
    integrator.cpp
    state.cpp
    hdf5writer.cpp
    eta_matrix.cpp
    )

target_link_libraries(dw.x
    Kokkos::kokkos
    HDF5::HDF5
    hdf5_cpp
    ${FFTW_LIB})