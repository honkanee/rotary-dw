#ifndef INPUT_PARAMETERS_H
#define INPUT_PARAMETERS_H

#include <string>

#include "state.h"

struct InputParameters {
    int N = 0;
    int nsteps = 0;
    Real B_ext = 0.0;
    Real dmi_const = 0.0;

    Real dx = 0.0;
    Real dt = 0.0;

    Real alpha = 0.0;
    Real Ms = 0.0;
    Real Aex = 0.0;
    Real Ku = 0.0;
    Real Delta = 0.5e-9;
    Real xi = 0.0;
    
    Real sigma = 0.0;
    Real start_with_noise = 0;

    int calc_v_frec = 100;

    // Output
    int print_frec = 0;
    int resolution_along_x = 0;

    int n_y_eta = 0;
    bool write_eta_to_file = false;
    int random_seed = 0;

    // RUNOPT
    bool ramped_field = false;
};

#endif