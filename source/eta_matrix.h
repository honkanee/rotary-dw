#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <fftw3.h>
#include <iostream>
#include <cmath>
#include <H5Cpp.h>

#include "simulation_parameters.h"
#include "state.h"

// ---------------- TYPES ----------------
using RealView2D = Kokkos::View<double**, Kokkos::LayoutRight, MemorySpace>;
using HostRealView2D = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>;

double compute_variance(RealView2D field);

// ---------------- FFT FILTER ----------------
void gaussian_filter_fft(RealView2D input, RealView2D output,
                         double sigma, double xi, double dx);

// ---------------- RANDOM FIELD ----------------
void fill_random(RealView2D field, int seed);

void write_eta_field_to_file(const RealView2D field,
                  const std::string& filename,
                  const std::string& dataset_name = "field");

ViewDoubleMatrixType make_eta_matrix(const SimulationParameters& p);

void update_B_vector(const ViewDoubleMatrixType eta_matrix,
                                     SimulationState& state);