//===-- dump_from_descriptor.cpp - the new read path, in the old shape ---===//
//
// Reads a `pie.model/1` descriptor and prints the same 134-field dump
// `dump_hf_config` prints for a `config.json`. That is what makes the round
// trip checkable: the two binaries emit the same shape from the two ends of
// the change, so `check_descriptor.sh` can diff them.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "model/descriptor.hpp"

// The emitter is shared with the oracle: including it keeps the two dumps
// field-for-field identical by construction rather than by maintenance.
#define PIE_DUMP_NO_MAIN
#include "dump_hf_config.cpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: dump_from_descriptor <descriptor.json|->\n";
        return 2;
    }
    std::string json;
    if (std::string(argv[1]) == "-") {
        std::ostringstream buffer;
        buffer << std::cin.rdbuf();
        json = buffer.str();
    } else {
        std::ifstream in(argv[1]);
        if (!in) {
            std::cerr << "cannot open " << argv[1] << "\n";
            return 2;
        }
        std::ostringstream buffer;
        buffer << in.rdbuf();
        json = buffer.str();
    }
    try {
        std::cout << dump(pie_cuda_driver::parse_pie_model_descriptor(json)).dump(2) << "\n";
    } catch (const std::exception& err) {
        std::cout << nlohmann::json{{"error", err.what()}}.dump(2) << "\n";
        return 1;
    }
    return 0;
}
