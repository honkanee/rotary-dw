#include "state.h"
#include "simulation_parameters.h"

#include <random>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>


SimulationState::SimulationState(int N_, Real Bext_, int dim_v, Real start_with_noise, bool include_first_order)
    : N(N_), dw("dw", N_), B_vector("B_vector", N_), Bext(Bext_), old_position(0.0), v_mean_vector("v_mean", dim_v), v_idx(0)
    {
        if (start_with_noise > 0) {
            auto h_dw = Kokkos::create_mirror_view(dw);

            std::mt19937 gen(12345);
            std::uniform_real_distribution<double> dist(0, start_with_noise*2*M_PI);

            for (size_t i = 0; i < h_dw.size(); ++i)
                h_dw.data()[i] = Complex(0, -dist(gen));

            Kokkos::deep_copy(dw, h_dw);
        } 
        else {
            Kokkos::deep_copy(dw, Complex(0.0, 0.0));
        }
        
        Kokkos::deep_copy(v_mean_vector, 0.0);
    }
