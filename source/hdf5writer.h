#ifndef HDF5WRITER_H
#define HDF5WRITER_H

#include <H5Cpp.h>
#include <iostream>
#include <Kokkos_Core.hpp>

#include "simulation_parameters.h"
#include "state.h"


// ----------------------
// HDF5 writer
// ----------------------
class HDF5Writer {
public:
    HDF5Writer(SimulationState& state, SimulationParameters& p, std::string& h5_directory);

    void make_output_file();
    
    void write(int step_count);

    void close();


private:
    SimulationParameters p;
    SimulationState& s;
    int step_count = 0;

    H5::H5File output_file;
    H5::CompType rowType;
    H5::DataSet h_dataset;
    H5::DataSet phi_dataset;
    H5::DataSet Bext_dataset;
    H5::DataSet v_mean_dataset;
    int end_t;
    hsize_t rows = 0;

    std::chrono::_V2::system_clock::time_point clock_start;
    std::chrono::_V2::system_clock::time_point run_clock_start;
    int run_time_limit;

    int file_count = 0;
    std::string& h5_directory;
};

#endif