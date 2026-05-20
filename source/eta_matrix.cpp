#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <fftw3.h>
#include <iostream>
#include <cmath>
#include <H5Cpp.h>

#include "simulation_parameters.h"
#include "state.h"

// ---------------- TYPES ----------------
using RealView2D = Kokkos::View<double**, Kokkos::LayoutRight>;
using HostRealView2D = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>;

// ---------------- STATISTICS ----------------
Real compute_variance(RealView2D field)
{
    int Nx = field.extent(0);
    int Ny = field.extent(1);

    Real sum = 0.0;
    Real sum2 = 0.0;

    Kokkos::parallel_reduce("variance",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
        KOKKOS_LAMBDA(int i, int j, Real& lsum) {
            lsum += field(i,j);
        }, sum);

    Kokkos::parallel_reduce("variance2",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
        KOKKOS_LAMBDA(int i, int j, Real& lsum2) {
            lsum2 += field(i,j) * field(i,j);
        }, sum2);

    Real N = Nx * Ny;
    Real mean = sum / N;
    return sum2 / N - mean * mean;
}

// ---------------- FFT FILTER ----------------
void gaussian_filter_fft(RealView2D input, RealView2D output,
                         Real sigma, Real xi, Real dx)
{
    int Nx = input.extent(0);
    int Ny = input.extent(1);


    Real Lx = Nx * dx;
    Real Ly = Ny * dx;


    HostRealView2D h_input = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), input);
    HostRealView2D h_output("h_output", Nx, Ny);

    int Nyc = Ny/2 + 1;
    fftw_complex* fft_data = fftw_alloc_complex(Nx * Nyc);

    fftw_plan forward = fftw_plan_dft_r2c_2d(Nx, Ny,
                                            h_input.data(),
                                            fft_data,
                                            FFTW_ESTIMATE);

    fftw_plan backward = fftw_plan_dft_c2r_2d(Nx, Ny,
                                             fft_data,
                                             h_output.data(),
                                             FFTW_ESTIMATE);

    fftw_execute(forward);

    for (int i = 0; i < Nx; i++) {
        int ki = (i <= Nx/2) ? i : i - Nx;
        Real kx = 2.0 * M_PI * ki/Lx;

        for (int j = 0; j < Nyc; j++) {
            Real ky = 2.0 * M_PI * j/Ly;

            Real k2 = kx*kx + ky*ky;
            Real factor = std::exp(-xi*xi*k2/8);

            int idx = i * Nyc + j;
            fft_data[idx][0] *= factor;
            fft_data[idx][1] *= factor;
        }
    }

    fftw_execute(backward);

    Kokkos::deep_copy(output, h_output);

    Real var = compute_variance(output);
    Real rescale_factor = sigma/sqrt(var);

    Kokkos::parallel_for(
        "rescale",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
        KOKKOS_LAMBDA(int i, int j) {
            output(i,j) *= rescale_factor;
        }
    );


    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(fft_data);
}

// ---------------- RANDOM FIELD ----------------
void fill_random(RealView2D field, int seed)
{
    int Nx = field.extent(0);
    int Ny = field.extent(1);

    Kokkos::Random_XorShift64_Pool<> pool(seed);

    Kokkos::parallel_for("init_random",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
        KOKKOS_LAMBDA(int i, int j) {
            auto gen = pool.get_state();
            field(i,j) = gen.normal();
            pool.free_state(gen);
        }
    );
}

void write_eta_field_to_file(const RealView2D field,
                  const std::string& filename,
                  const std::string& dataset_name = "field")
{
    int Nx = field.extent(0);
    int Ny = field.extent(1);

    // Mirror to host
    HostRealView2D h_field = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), field);

    try {
        // ---------------- FILE ----------------
        H5::H5File file(filename, H5F_ACC_TRUNC);

        // ---------------- DIMENSIONS ----------------
        hsize_t dims[2] = { (hsize_t)Nx, (hsize_t)Ny };
        H5::DataSpace dataspace(2, dims);

        // ---------------- DATASET ----------------
        H5::DataSet dataset = file.createDataSet(
            dataset_name,
            H5::PredType::NATIVE_DOUBLE,
            dataspace
        );

        // ---------------- WRITE ----------------
        dataset.write(
            h_field.data(),
            H5::PredType::NATIVE_DOUBLE
        );

        std::cout << "Wrote HDF5: " << filename << std::endl;
    }
    catch (H5::Exception& err) {
        err.printErrorStack();
        throw;
    }
}

ViewDoubleMatrixType make_eta_matrix(const SimulationParameters& p) {

    int Nx = p.N;
    int Ny = p.n_y_eta; // To be changed!
    Real sigma = p.sigma;
    Real xi = p.xi;
    Real dx = p.dx;
    bool write_to_file = p.write_eta_to_file;
    int seed = p.random_seed;

    RealView2D random_field("random_matrix", Nx, Ny);
    RealView2D filtered("eta_matrix", Nx, Ny);

    fill_random(random_field, seed);
    gaussian_filter_fft(random_field, filtered, sigma, xi, dx);

    if (write_to_file) {
        write_eta_field_to_file(filtered, "eta_matrix.h5");
    }

    ViewDoubleMatrixType eta_matrix("eta_matrix", Nx, Ny);
    Kokkos::deep_copy(eta_matrix, filtered);

    return eta_matrix;
}

void update_B_vector(const ViewDoubleMatrixType eta_matrix,
                                     SimulationState& state) {
    Kokkos::parallel_for(
        "make_B_vector",
        state.N,
        KOKKOS_LAMBDA(int i) {
            Real y = state.dw(i).real();
            Real a = y - int(y);
            int yi1 = int(y)%eta_matrix.extent(1);
            int yi2 = (int(y)+1)%eta_matrix.extent(1);
            Real interpolated_from_eta = a*eta_matrix(i,yi2)+(1.0-a)*eta_matrix(i,yi1);
            state.B_vector(i) = state.Bext + interpolated_from_eta;
        });
}
