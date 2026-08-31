#include <cstdint>
#include <iostream>
#include <string>

namespace {

struct NormalizedArithmetic {
    std::string provenance;
    std::string operation;
    uint32_t elements_per_block = 0;
    uint32_t stored_bits = 0;
    int32_t value_offset = 0;
    bool signed_values = false;
    bool scaled = false;
    bool biased = false;
};

std::string route_identity(const NormalizedArithmetic& arithmetic) {
    return std::to_string(arithmetic.operation.size()) + ':' +
           arithmetic.operation + ':' +
           std::to_string(arithmetic.elements_per_block) + ':' +
           std::to_string(arithmetic.stored_bits) + ':' +
           std::to_string(arithmetic.value_offset) + ':' +
           std::to_string(arithmetic.signed_values) + ':' +
           std::to_string(arithmetic.scaled) + ':' +
           std::to_string(arithmetic.biased);
}

int failures = 0;
int checks = 0;

void expect(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void show_provenance_independent_policy_example() {
    const NormalizedArithmetic first{"container-A:type-7", "affine", 64, 4,
                                     -8, true, true, true};
    const NormalizedArithmetic second{"container-B:type-99", "affine", 64, 4,
                                      -8, true, true, true};
    expect(route_identity(first) == route_identity(second),
           "equivalent normalized arithmetic must share a route identity");
}

void show_arithmetic_change_policy_example() {
    const NormalizedArithmetic first{"container-A:type-7", "affine", 64, 4,
                                     -8, true, true, true};
    NormalizedArithmetic changed = first;
    changed.biased = false;
    expect(route_identity(first) != route_identity(changed),
           "changed arithmetic must have a different route identity");
}

}  // namespace

int main() {
    show_provenance_independent_policy_example();
    show_arithmetic_change_policy_example();
    if (failures != 0) {
        std::cerr << "test_universal_source_policy_example: " << failures << '/' << checks
                  << " checks FAILED\n";
        return 1;
    }
    std::cout << "test_universal_source_policy_example: OK (" << checks
              << " policy examples)\n";
    return 0;
}
