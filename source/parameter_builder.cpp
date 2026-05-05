#include <Kokkos_Complex.hpp> 

#include "parameter_builder.h"
#include "input_parameters.h"

using Complex = Kokkos::complex<double>;

// Physical constants
const double gamma_e = 1.76e11;
const double mu0 = 1.256637062e-6; // [H/m]


SimulationParameters build_simulation_parameters(const InputParameters& in) {
    SimulationParameters p;

    p.alpha = in.alpha;

    double D = sqrt(in.Aex / (in.Ku - 0.5 * mu0 * in.Ms * in.Ms));

    // Unit conversions to from SI to simulation units
    p.F = Complex(1.0,0) / ( Complex(in.alpha,0.0) + Complex(0.0,1.0));
    p.K = (2 * in.Ku - mu0 * in.Ms * in.Ms) / (mu0 * in.Ms * in.Ms);
    p.Delta = in.Delta / D;
    p.xi = in.xi * 1e-9 / D;
    p.dx = in.dx * 1e-9 / D;
    p.Nn = p.Delta / M_PI * std::log(2.0);
    p.Bext = in.B_ext * 1e-3 / (mu0 * in.Ms);
    p.sigma = in.sigma * 1e-3 / (mu0 * in.Ms);
    p.dmi_const = in.dmi_const / (in.Ms * in.Ms * mu0 * D);
    p.dt = in.dt * 1e-9 * gamma_e * mu0 * in.Ms;

    // Copy simple values
    p.N = in.N;
    p.nsteps = in.nsteps;
    p.print_frec = in.print_frec;
    p.resolution_along_x = in.resolution_along_x;
    p.calc_v_frec = in.calc_v_frec;

    p.ramped_field = in.ramped_field;
    p.print_all = in.print_all;

    p.n_y_eta = in.n_y_eta;
    p.write_eta_to_file = in.write_eta_to_file;

    // Private parts
    p.D_scale = D;
    p.Ms = in.Ms;

    return p;
}