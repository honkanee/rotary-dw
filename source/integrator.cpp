#include <cstdio>
#include <iostream>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include "integrator.h"
#include "state.h"
#include "parser.h"
#include "simulation_parameters.h"
#include "input_parameters.h"
#include "parameter_builder.h"
#include "hdf5writer.h"
#include "eta_matrix.h"


// ----------------------
// Integrator step
// RK4
// ----------------------
void step(SimulationState& state, SimulationParameters& p, ViewDoubleMatrixType eta_matrix) {
    auto old_dw = state.dw;

    update_B_vector(eta_matrix, state);
    Kokkos::fence();

    ViewComplexVectorType k1 = ViewComplexVectorType("k1", p.N);
    ViewComplexVectorType k2 = ViewComplexVectorType("k2", p.N);
    ViewComplexVectorType k3 = ViewComplexVectorType("k3", p.N);
    ViewComplexVectorType k4 = ViewComplexVectorType("k4", p.N);
    ViewComplexVectorType temp_z_1 = ViewComplexVectorType("temp_z_1", p.N);
    ViewComplexVectorType temp_z_2 = ViewComplexVectorType("temp_z_2", p.N);
    
    Kokkos::parallel_for("calc_k1", state.N, KOKKOS_LAMBDA(int i) {

        Complex dzdt_i = dzdt(old_dw, state.B_vector, p, i);
        k1(i) = dzdt_i;
        temp_z_1(i) = old_dw(i)+0.5*p.dt*dzdt_i;
    });
    Kokkos::fence();

    Kokkos::parallel_for("calc_k2", state.N, KOKKOS_LAMBDA(int i) {

        Complex dzdt_i = dzdt(temp_z_1, state.B_vector, p, i);
        k2(i) = dzdt_i;
        temp_z_2(i) = old_dw(i)+0.5*p.dt*dzdt_i;

    });
    Kokkos::fence();

    Kokkos::parallel_for("calc_k3", state.N, KOKKOS_LAMBDA(int i) {

        Complex dzdt_i = dzdt(temp_z_2, state.B_vector, p, i);
        k3(i) = dzdt_i;
        temp_z_1(i) = old_dw(i)+p.dt*dzdt_i;

    });
    Kokkos::parallel_for("calc_k4", state.N, KOKKOS_LAMBDA(int i) {

        k4(i) = dzdt(temp_z_1, state.B_vector, p, i);

    });
    Kokkos::fence();

    Kokkos::parallel_for("update_dw", state.N, KOKKOS_LAMBDA(int i) {

        state.dw(i) = old_dw(i) + p.dt/6*(k1(i)+2*k2(i)+2*k3(i)+k4(i));
    });
    Kokkos::fence();
}

void calc_v(SimulationState& state, SimulationParameters& p) {
    
    double total_sum = 0;
    // Parallel reduce to calculate sum
    Kokkos::parallel_reduce("sum", state.N, KOKKOS_LAMBDA(int i, double& lsum) {
        lsum += state.dw(i).real();
    }, total_sum);

    double mean = total_sum / state.N;
    Kokkos::parallel_for("update", 1, KOKKOS_LAMBDA(int i) {
        state.v_mean_vector(state.v_idx) =
        (mean - state.old_position) / (p.dt * p.calc_v_frec);
    });
    Kokkos::fence();
    state.old_position = mean;
    ++state.v_idx;
}