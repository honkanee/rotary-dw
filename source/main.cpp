#include <cstdio>
#include <iostream>
#include <chrono>
#include <filesystem>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include "state.h"
#include "parser.h"
#include "simulation_parameters.h"
#include "input_parameters.h"
#include "parameter_builder.h"
#include "hdf5writer.h"
#include "integrator.h"
#include "eta_matrix.h"

using high_res_clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[])
{
    InputParameters in = parse_input("input.conf");
    SimulationParameters p = build_simulation_parameters(in);
    p.print_SI();

    Kokkos::initialize(argc, argv);
    {

    SimulationState state = SimulationState(p.N, p.Bext, int(p.print_frec/p.calc_v_frec), p.start_with_noise,p.include_first_order);
    std::string out_dir = "./out/";
    if (!std::filesystem::exists(out_dir)) {
    std::filesystem::create_directories(out_dir);
    }
    HDF5Writer writer(state, p, out_dir);
    writer.make_output_file();

    ViewDoubleMatrixType eta_matrix = make_eta_matrix(p);

    Real compute_time = 0.0;
    Real io_time = 0.0;

    for (int step_idx = 0; step_idx < p.nsteps; ++step_idx) {
        auto t0 = high_res_clock::now();
        step(state, p, eta_matrix);

        if (step_idx % p.calc_v_frec == 0) {
            calc_v(state, p);
        }
        auto t1 = high_res_clock::now();
        compute_time += std::chrono::duration<Real>(t1 - t0).count();

        if (step_idx % p.print_frec == 0) {
            auto t_io0 = high_res_clock::now();
            writer.write(step_idx);
            Kokkos::fence();
            state.v_idx = 0;

            auto t_io1 = high_res_clock::now();
            io_time += std::chrono::duration<Real>(t_io1 - t_io0).count();
        }

        if (step_idx*10 % p.nsteps == 0) {
            std::cout << round(100*step_idx / p.nsteps) << " %\n";
        }
    }
    writer.close();
    
    std::cout << "Compute time: " << compute_time << " s\n";
    std::cout << "I/O time:     " << io_time << " s\n";
    std::cout << "Total time:   " << compute_time + io_time << " s\n";

    std::cout << "I/O fraction: "
        << io_time / (compute_time + io_time) * 100.0
        << " %\n";
    
    }

    Kokkos::finalize();



    return 0;
}