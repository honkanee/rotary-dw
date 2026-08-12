#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <cstdio>
#include <iostream>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include "state.h"
#include "parser.h"
#include "simulation_parameters.h"
#include "input_parameters.h"
#include "parameter_builder.h"
#include "hdf5writer.h"


KOKKOS_INLINE_FUNCTION
Complex dzdx_at_i(const ViewComplexVectorType& z, int i, int N, Real dx) {
    int ip = (i + 1) % N;
    int im = (i - 1 + N) % N;

    return (-z(im) + z(ip)) / (2.0 * dx);
}

KOKKOS_INLINE_FUNCTION
Complex dzdx2_at_i(const ViewComplexVectorType& z, int i, int N, Real dx) {
    int ip = (i + 1) % N;
    int im = (i - 1 + N) % N;

    return (z(im)-2.0*z(i)+z(ip)) / (dx*dx);
}

KOKKOS_INLINE_FUNCTION
Complex dzdt(ViewComplexVectorType z, const ViewDoubleVectorType B_vector, const SimulationParameters& p, int i) {

    Complex dzdx = dzdx_at_i(z,i,p.N,p.dx);
    Complex dzdx2 = dzdx2_at_i(z,i,p.N,p.dx);
    Real chi = Kokkos::atan(dzdx.real());
    Real phi_i = -z(i).imag();
    Complex dmi = Complex(0,0);
    if (p.dmi_const != 0) {
        dmi =  p.dmi_const * M_PI * 0.5 * (Complex(0.0,1.0) * (Kokkos::cos(phi_i) + Kokkos::sin(phi_i)*dzdx.real()) + Complex(1.0,0.0) * (dzdx.imag() * Kokkos::sin(phi_i)));
    }

    return p.F*(p.K*dzdx2 - B_vector(i) + 0.5*p.Nn*Complex(0.0,1.0)*Kokkos::sin(2*(phi_i- chi)) + dmi);
}

// ----------------------
// Integrator step
// RK4
// ----------------------
void step(SimulationState& state, SimulationParameters& p, ViewDoubleMatrixType eta_matrix);

void calc_v(SimulationState& state, SimulationParameters& p);

#endif