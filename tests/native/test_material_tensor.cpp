// Run with the Meep environment active:
// $CXX -std=c++17 -I"$CONDA_PREFIX/include" tests/native/test_material_tensor.cpp \
//   -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" -lmeep -lctlgeom \
//   -o /tmp/tama-material-tensor-test && /tmp/tama-material-tensor-test
#include "../../native/meep/material_tensor.hpp"

#include <cassert>
#include <iostream>
#include <limits>

using tama_material_tensor::Matrix;
using tama_material_tensor::averaged_inverse;
using tama_material_tensor::inverse;
using tama_material_tensor::projected;
using tama_material_tensor::projected_average;

double max_error(const Matrix &a, const Matrix &b) {
    double error = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            error = std::max(error, std::abs(a[i][j] - b[i][j]));
    return error;
}

Matrix rotate(const Matrix &a, const Matrix &rotation) {
    Matrix out{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    out[i][j] += rotation[i][k] * a[k][l] * rotation[j][l];
    return out;
}

int main() {
    const Matrix first{{{2, .2, .1}, {.2, 3, .3}, {.1, .3, 4}}};
    const Matrix second{{{8, .5, .2}, {.5, 10, .4}, {.2, .4, 12}}};
    const std::array<double, 3> normal{{.36, .48, .8}};
    assert(max_error(averaged_inverse(first, second, normal, 0), inverse(first)) <
           1e-14);
    assert(max_error(averaged_inverse(first, second, normal, 1), inverse(second)) <
           1e-14);
    assert(max_error(averaged_inverse(first, first, normal, .37), inverse(first)) <
           1e-14);

    const Matrix scalar_first{{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}}};
    const Matrix scalar_second{{{8, 0, 0}, {0, 8, 0}, {0, 0, 8}}};
    const double fill = .37;
    const double arithmetic = (1 - fill) * 2 + fill * 8;
    const double inverse_harmonic = (1 - fill) / 2 + fill / 8;
    Matrix scalar_expected{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            scalar_expected[i][j] =
                normal[i] * normal[j] * (inverse_harmonic - 1 / arithmetic) +
                (i == j ? 1 / arithmetic : 0);
    assert(max_error(averaged_inverse(scalar_first, scalar_second, normal, fill),
                     scalar_expected) < 1e-14);

    const Matrix rotation{{{.6, -.8, 0}, {.8, .6, 0}, {0, 0, 1}}};
    std::array<double, 3> rotated_normal{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            rotated_normal[i] += rotation[i][j] * normal[j];
    assert(max_error(averaged_inverse(rotate(first, rotation), rotate(second, rotation),
                                      rotated_normal, fill),
                     rotate(averaged_inverse(first, second, normal, fill), rotation)) <
           1e-14);

    const double beta = 2, eta = .3;
    const double at_threshold =
        std::tanh(beta * eta) / (std::tanh(beta * eta) + std::tanh(beta * (1 - eta)));
    assert(std::abs(projected(eta, beta, eta) - at_threshold) < 1e-15);
    assert(std::abs(projected(eta - 1e-9, beta, eta) - at_threshold) < 2e-9);
    assert(std::abs(projected(eta + 1e-9, beta, eta) - at_threshold) < 2e-9);
    for (const auto dim : {meep::D1, meep::D2, meep::D3}) {
        assert(std::abs(projected_average({dim, .5, .3, 8, .5}, 1e-10, 100000) - .5) <
               1e-10);
        assert(std::abs(projected_average({dim, .43, .3, 0, .5}, 1e-10, 100000) - .43) <
               1e-14);
        assert(std::abs(projected_average({dim, eta, 0, beta, eta}, 1e-10, 100000) -
                        at_threshold) < 1e-15);
        assert(std::abs(projected_average({dim, eta, 1e-10, beta, eta}, 1e-10, 100000) -
                        at_threshold) < 1e-10);
    }
    const double infinity = std::numeric_limits<double>::infinity();
    assert(projected(eta, infinity, eta) == .5);
    assert(projected(eta - .1, infinity, eta) == 0);
    assert(projected(eta + .1, infinity, eta) == 1);
    assert(std::abs(projected_average({meep::D1, .4, .2, infinity, .3}, 1e-10, 100000) -
                    .75) < 1e-14);
    assert(std::abs(projected_average({meep::D3, .4, .2, infinity, .3}, 1e-10, 100000) -
                    .84375) < 1e-14);
    std::cout
        << "PASS: tensor limits, scalar reduction, rotation covariance, projection normalization and continuity\n";
}
