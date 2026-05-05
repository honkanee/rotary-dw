#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <fftw3.h>
#include <iostream>
#include <cmath>
#include <H5Cpp.h>

#include "eta_matrix.h"

// ---------------- TYPES ----------------
using RealView2D = Kokkos::View<double**, Kokkos::LayoutRight>;
using HostRealView2D = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>;


// ---------------- MAIN TEST ----------------
int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int Nx = 100000;
        int Ny = 3000;

        double sigma = 3;
        double xi = 20;
        double dx = 6;

        RealView2D field("field", Nx, Ny);
        RealView2D filtered("filtered", Nx, Ny);

        fill_random(field);

        double var_before = compute_variance(field);

        gaussian_filter_fft(field, filtered, sigma, xi, dx);

        double var_after = compute_variance(filtered);

        std::cout << "Variance before: " << var_before << std::endl;
        std::cout << "Variance after : " << var_after << std::endl;

        // Optional: check for NaNs
        int nan_count = 0;
        Kokkos::parallel_reduce("nan_check",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{Nx,Ny}),
            KOKKOS_LAMBDA(int i, int j, int& lcount) {
                if (!isfinite(filtered(i,j))) {
                    lcount++;
                }
            }, nan_count);

        std::cout << "NaN count: " << nan_count << std::endl;

        write_eta_field_to_file(filtered, "eta_matrix.h5");
    }
    Kokkos::finalize();

    return 0;
}