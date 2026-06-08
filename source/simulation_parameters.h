#ifndef SIMULATION_PARAMETERS_H
#define SIMULATION_PARAMETERS_H

#include <string>
#include <iostream>

#include <Kokkos_Complex.hpp>

#include "state.h"
#include "input_parameters.h"

// All parameters in converted units (except D, gamma & mu0)

struct SimulationParameters {
    Real D = 0.0;

    // Material parameters
    Real alpha = 0.0;
    Real dmi_const = 0.0;

    // Derived parameters (computed in builder)
    Real K = 0;
    Complex F = Complex(0.0,0.0);

    // Spatial parameters
    Real dx = 0.0;
    Real xi = 0.0;
    Real Delta = 0.0;
    Real Nn = 0.0;

    // External fields, etc.
    Real Bext = 0.0;
    Real sigma = 0.0;

    // User defined parameters
    int N = 0;
    int nsteps = 0;

    Real dt = 0.0;
    int calc_v_frec = 0;

    Real start_with_noise = 0;

    // Outputs
    int print_frec = 0;
    unsigned long long resolution_along_x = 0;
    int num_rows_per_file = 1000;

    int n_y_eta = 1000;
    bool write_eta_to_file = true;
    int random_seed = 0;

    // RUNOPT
    bool ramped_field = false;
    bool file_partioning = false;

    // ===== Public read-only accessors =====
    Real dx_SI() const { return dx * D_scale; }
    Real dt_SI() const { return dt * time_scale(); }
    Real Bext_SI() const { return Bext * field_scale(); }
    Real sigma_SI() const { return sigma * field_scale(); }
    Real dmi_const_SI() const { return dmi_const * (Ms*Ms*mu0*D_scale); }

    // ===== Scaling helpers =====
    Real length_scale() const { return D_scale; }
    Real time_scale() const { return 1.0 / (gamma_e * mu0 * Ms); }
    Real field_scale() const { return mu0 * Ms; }

    void print_SI() const {
    constexpr Real gamma_e = 1.76e11;
    constexpr Real mu0 = 1.256637062e-6;

    std::cout << "===== Simulation Parameters (SI units) =====\n";

    // Simulation control
    std::cout << "N: " << N << "\n";
    std::cout << "  -> L: " << N*dx*D_scale*1e6 << " µm\n";
    std::cout << "nsteps: " << nsteps << "\n";
    std::cout << "  -> t: " << nsteps*dt*time_scale()*1e6 << " µs\n";

    // Field
    Real field_scale = mu0 * Ms;
    std::cout << "Bext (T): " << Bext * field_scale << "\n";
    std::cout << "sigma (T): " << sigma * field_scale << "\n";

    std::cout << "print_frec: " << print_frec << "\n";
    std::cout << "resolution_along_x: " << resolution_along_x << "\n";

    // Material
    std::cout << "dmi_const (J/m²): "
              << dmi_const * (Ms * Ms * mu0 * D_scale) << "\n\n";

    // Length scale
    std::cout << "D_scale (m): " << D_scale << "\n";

    // Spatial
    std::cout << "dx (m): " << dx * D_scale << "\n";
    std::cout << "Delta (m): " << Delta * D_scale << "\n";
    std::cout << "xi (m): " << xi * D_scale << "\n";

    // Time
    Real time_scale = 1.0 / (gamma_e * mu0 * Ms);
    std::cout << "dt (s): " << dt * time_scale << "\n";

    // Dimensionless (still useful to see)
    std::cout << "alpha: " << alpha << "\n";
    std::cout << "K (dimensionless): " << K << "\n";
    std::cout << "Nn: " << Nn << "\n";

    std::cout << "n_y_eta: " << n_y_eta << "\n";
    std::cout << "       = " << n_y_eta*dx*D_scale*1e6 << " µm\n";
    std::cout << "random_seed: " << random_seed << "\n";

    std::cout << "write_eta_to_file: " << write_eta_to_file << "\n";

    std::cout << "ramped_field: " << ramped_field << "\n";
    std::cout << "start_with_noise: " << start_with_noise << "\n";

    std::cout << "===========================================\n";
};

private:
    // ===== Hidden =====
    Real Ms;
    Real D_scale;

    static constexpr Real gamma_e = 1.76e11;
    static constexpr double mu0 = 1.256637062e-6;

    // Only builder can set these
    friend SimulationParameters build_simulation_parameters(const InputParameters&);
};

#endif