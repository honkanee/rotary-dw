#ifndef INPUT_PARAMETERS_H
#define INPUT_PARAMETERS_H

#include <string>

struct InputParameters {
    int N = 0;
    int nsteps = 0;
    double B_ext = 0.0;
    double dmi_const = 0.0;

    double dx = 0.0;
    double dt = 0.0;

    double alpha = 0.0;
    double Ms = 0.0;
    double Aex = 0.0;
    double Ku = 0.0;
    double Delta = 0.5e-9;
    double xi = 0.0;
    
    double sigma = 0.0;

    int calc_v_frec = 100;

    // Output
    int print_frec = 0;
    int resolution_along_x = 0;

    int n_y_eta = 0;
    bool write_eta_to_file = false;

    // RUNOPT
    bool ramped_field = false;
    bool print_all = false;
};

#endif