#include "parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <vector>

// Helper: trim whitespace
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// Convert T/F to bool
bool to_bool(const std::string& val) {
    return (val == "T" || val == "true" || val == "1");
}

std::vector<std::string> required_keys = {
    "N",
    "nsteps",
    "B_ext",
    "dmi_const",
    "dx",
    "dt",
    "print_frec",
    "resolution_along_x",
    "Ms",
    "Ku",
    "Aex",
    "alpha",
    "sigma",
    "xi",
    "Delta",
    "write_eta_to_file",
    "n_y_eta",
    "random_seed",
    "calc_v_frec"
};

std::unordered_set<std::string> found_keys;

InputParameters parse_input(const std::string& filename) {
    InputParameters p;
    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error("Could not open input file");
    }

    std::string line;
    bool in_runopt = false;

    while (std::getline(file, line)) {
        // Remove comments
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        line = trim(line);
        if (line.empty()) continue;

        if (line == "RUNOPT") {
            in_runopt = true;
            continue;
        }

        if (!in_runopt) {
            // Parse key = value
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;

            std::string key = trim(line.substr(0, eq_pos));
            std::string value = trim(line.substr(eq_pos + 1));

            std::stringstream ss(value);

            if (key == "N") ss >> p.N;
            else if (key == "nsteps") ss >> p.nsteps;
            else if (key == "B_ext") ss >> p.B_ext;
            else if (key == "dmi_const") ss >> p.dmi_const;
            else if (key == "dx") ss >> p.dx;
            else if (key == "dt") ss >> p.dt;
            else if (key == "print_frec") ss >> p.print_frec;
            else if (key == "resolution_along_x") ss >> p.resolution_along_x;
            else if (key == "Ms") ss >> p.Ms;
            else if (key == "Ku") ss >> p.Ku;
            else if (key == "Aex") ss >> p.Aex;
            else if (key == "alpha") ss >> p.alpha;
            else if (key == "sigma") ss >> p.sigma;
            else if (key == "xi") ss >> p.xi;
            else if (key == "Delta") ss >> p.Delta;
            else if (key == "write_eta_to_file") p.write_eta_to_file = to_bool(value);
            else if (key == "n_y_eta") ss >> p.n_y_eta;
            else if (key == "random_seed") ss >> p.random_seed;
            else if (key == "calc_v_frec") ss >> p.calc_v_frec;
            else if (key == "start_with_noise") ss >> p.start_with_noise;
            else if (key=="include_first_order") p.include_first_order = to_bool(value);

            // Mark parameter as found
            found_keys.insert(key);
        } else {
            // RUNOPT parsing: <KEY> = VALUE
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;

            std::string key = trim(line.substr(0, eq_pos));
            std::string value = trim(line.substr(eq_pos + 1));

            // Remove <>
            key.erase(std::remove(key.begin(), key.end(), '<'), key.end());
            key.erase(std::remove(key.begin(), key.end(), '>'), key.end());

            if (key == "RAMPED_FIELD") p.ramped_field = to_bool(value);
        }
    }

    bool missing_param = false;

    for (const auto& req : required_keys) {
        if (found_keys.find(req) == found_keys.end()) {
            std::cerr << "Missing required parameter: " << req << std::endl;
            missing_param = true;
        }
    }

    if (missing_param) {
        std::exit(EXIT_FAILURE);
    }

    return p;
}