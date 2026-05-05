#ifndef STATE_H
#define STATE_H

#include "simulation_parameters.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

using Complex = Kokkos::complex<double>;
using ViewDoubleVectorType = Kokkos::View<double*>;
using ViewDoubleMatrixType = Kokkos::View<double**>;
using ViewComplexVectorType = Kokkos::View<Complex*>;

// ----------------------
// Simulation state
// ----------------------
struct SimulationState {
    ViewComplexVectorType dw;
    ViewDoubleVectorType B_vector;
    ViewDoubleVectorType v_mean_vector;
    int N;
    double Bext;
    double old_position;
    int v_idx;

    SimulationState(int N_, double Bext, int dim_v);
};


template<typename ViewType>
Kokkos::View<typename ViewType::value_type*, Kokkos::SharedSpace>
spatial_average(const ViewType& input, int resolution) {

    using Complex = typename ViewType::value_type;

    int N = input.extent(0);
    int Nbins = resolution;
    int bin_size = N/resolution;

    Kokkos::View<Complex*, Kokkos::SharedSpace> output("avg", Nbins);

    Kokkos::parallel_for("spatial_average", Nbins, KOKKOS_LAMBDA(int i) {
        Complex sum = Complex(0.0, 0.0);

        for (int j = 0; j < bin_size; ++j) {
            sum += input(i * bin_size + j);
        }

        output(i) = sum / static_cast<double>(bin_size);
    });

    return output;
}

#endif