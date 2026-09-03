#ifndef STATE_H
#define STATE_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#ifdef USE_FP32
using Real = float;
#else
using Real = double;
#endif

using Complex = Kokkos::complex<Real>;
using Device = Kokkos::DefaultExecutionSpace;
using MemorySpace = Device::memory_space;

using ViewDoubleVectorType = Kokkos::View<Real*, MemorySpace>;
using ViewDoubleMatrixType = Kokkos::View<Real**, MemorySpace>;
using ViewComplexVectorType = Kokkos::View<Complex*, MemorySpace>;

// ----------------------
// Simulation state
// ----------------------
struct SimulationState {
    ViewComplexVectorType dw;
    ViewDoubleVectorType B_vector;
    ViewDoubleVectorType v_mean_vector;
    int N;
    Real Bext;
    Real old_position;
    int v_idx;

    SimulationState(int N_, Real Bext, int dim_v, Real start_with_noise, bool include_first_order);
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

        output(i) = sum / static_cast<Real>(bin_size);
    });

    return output;
}

#endif