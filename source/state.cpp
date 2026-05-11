#include "state.h"
#include "simulation_parameters.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>


SimulationState::SimulationState(int N_, Real Bext_, int dim_v)
    : N(N_), dw("dw", N_), B_vector("B_vector", N_), Bext(Bext_), old_position(0.0), v_mean_vector("v_mean", dim_v), v_idx(0)
    {
        Kokkos::deep_copy(dw, Complex(0.0, 0.0));
        Kokkos::deep_copy(v_mean_vector, 0.0);
    }
