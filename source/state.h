#ifndef STATE_H
#define STATE_H

#include "simulation_parameters.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

using Complex = Kokkos::complex<double>;
using Device = Kokkos::DefaultExecutionSpace;
using MemorySpace = Device::memory_space;

using ViewDoubleVectorType = Kokkos::View<double*, MemorySpace>;
using ViewDoubleMatrixType = Kokkos::View<double**, MemorySpace>;
using ViewComplexVectorType = Kokkos::View<Complex*, MemorySpace>;

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
ViewComplexVectorType
spatial_average(const ViewType& input, int resolution) {

    int N = input.extent(0);
    int Nbins = resolution;
    int bin_size = N/resolution;

    ViewComplexVectorType output("avg", Nbins);

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