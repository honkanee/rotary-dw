cmake_minimum_required(VERSION 3.16)
project(Domainwall)

find_package(Kokkos REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

find_library(FFTW_LIB fftw3 REQUIRED)

add_executable(domainwall.x
    main.cpp
    parser.cpp
    parameter_builder.cpp
    integrator.cpp
    state.cpp
    hdf5writer.cpp
    eta_matrix.cpp
    )

target_link_libraries(domainwall.x
    Kokkos::kokkos
    HDF5::HDF5
    hdf5_cpp
    ${FFTW_LIB})

add_executable(eta_matrix_test
    eta_matrix.cpp
    eta_matrix_test.cpp)

target_link_libraries(eta_matrix_test
    Kokkos::kokkos
    HDF5::HDF5
    hdf5_cpp
    ${FFTW_LIB}
)