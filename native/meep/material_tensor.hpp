#pragma once

#include <meep.hpp>
#include <meep/meepgeom.hpp>
#include <ctl-math.h>

#include <array>
#include <cmath>
#include <stdexcept>

namespace tama_material_tensor {

using Matrix = std::array<std::array<double, 3>, 3>;

inline Matrix epsilon(const meep_geom::medium_struct &m) {
    return {{{m.epsilon_diag.x, m.epsilon_offdiag.x.re, m.epsilon_offdiag.y.re},
             {m.epsilon_offdiag.x.re, m.epsilon_diag.y, m.epsilon_offdiag.z.re},
             {m.epsilon_offdiag.y.re, m.epsilon_offdiag.z.re, m.epsilon_diag.z}}};
}

inline bool anisotropic(const meep_geom::medium_struct &m) {
    return m.epsilon_diag.x != m.epsilon_diag.y ||
           m.epsilon_diag.x != m.epsilon_diag.z || m.epsilon_offdiag.x.re != 0 ||
           m.epsilon_offdiag.y.re != 0 || m.epsilon_offdiag.z.re != 0;
}

inline Matrix inverse(const Matrix &a) {
    const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[1][2]) -
                       a[0][1] * (a[0][1] * a[2][2] - a[0][2] * a[1][2]) +
                       a[0][2] * (a[0][1] * a[1][2] - a[0][2] * a[1][1]);
    if (!(a[0][0] > 0) || !(a[0][0] * a[1][1] - a[0][1] * a[0][1] > 0) ||
        !(det > 0) || !std::isfinite(det))
        throw std::runtime_error("averaged permittivity must be positive definite");
    Matrix out{};
    out[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[1][2]) / det;
    out[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[0][2]) / det;
    out[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[0][1]) / det;
    out[0][1] = out[1][0] = (a[0][2] * a[1][2] - a[0][1] * a[2][2]) / det;
    out[0][2] = out[2][0] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
    out[1][2] = out[2][1] = (a[0][1] * a[0][2] - a[0][0] * a[1][2]) / det;
    return out;
}

// Kottke's interface-frame tau average, written without choosing tangent axes.
inline Matrix averaged_inverse(const Matrix &a, const Matrix &b,
                               const std::array<double, 3> &normal, double fill) {
    double reciprocal_normal = 0;
    std::array<double, 3> coupling{};
    Matrix tangential{};
    const Matrix *media[2] = {&a, &b};
    const double weights[2] = {1 - fill, fill};
    for (int k = 0; k < 2; ++k) {
        const Matrix &e = *media[k];
        std::array<double, 3> en{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) en[i] += e[i][j] * normal[j];
        double enn = 0;
        for (int i = 0; i < 3; ++i) enn += normal[i] * en[i];
        if (!(enn > 0))
            throw std::runtime_error("averaging requires positive dielectric endpoints");
        reciprocal_normal += weights[k] / enn;
        for (int i = 0; i < 3; ++i) {
            coupling[i] += weights[k] * en[i] / enn;
            for (int j = 0; j < 3; ++j)
                tangential[i][j] += weights[k] * (e[i][j] - en[i] * en[j] / enn);
        }
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            tangential[i][j] += coupling[i] * coupling[j] / reciprocal_normal;
    return inverse(tangential);
}

struct ProjectionAverage {
    meep::ndim dim;
    double value, variation, beta, eta;
};

inline double projected(double value, double beta, double eta) {
    if (beta == 0) return value;
    if (std::isinf(beta)) return value < eta ? 0.0 : (value > eta ? 1.0 : 0.5);
    return (std::tanh(beta * eta) + std::tanh(beta * (value - eta))) /
           (std::tanh(beta * eta) + std::tanh(beta * (1 - eta)));
}

inline number projection_integrand(integer, number *x, void *data) {
    const auto &p = *static_cast<ProjectionAverage *>(data);
    double position = x[0], weight;
    if (p.dim == meep::D2) {
        position = std::sin(x[0]);
        const double cosine = std::cos(x[0]);
        weight = 2 * cosine * cosine / meep::pi;
    } else if (p.dim == meep::D3) {
        weight = 0.75 * (1 - position * position);
    } else {
        weight = 0.5;
    }
    return weight * projected(p.value + p.variation * position, p.beta, p.eta);
}

inline double projected_average(ProjectionAverage p, double tol, int maxeval) {
    if (p.beta == 0) return p.value;
    if (p.variation == 0) return projected(p.value, p.beta, p.eta);
    if (std::isinf(p.beta)) {
        const double s = (p.eta - p.value) / p.variation;
        if (s <= -1) return 1;
        if (s >= 1) return 0;
        if (p.dim == meep::D2)
            return 0.5 - (s * std::sqrt(1 - s * s) + std::asin(s)) / meep::pi;
        if (p.dim == meep::D3) return 0.5 - 0.75 * s + 0.25 * s * s * s;
        return 0.5 * (1 - s);
    }
    number lo[1] = {p.dim == meep::D2 ? -meep::pi / 2 : -1.0};
    number hi[1] = {-lo[0]}, error;
    integer status;
    return adaptive_integration(projection_integrand, lo, hi, 1, &p, 0, tol,
                                maxeval, &error, &status);
}

inline void generalized_material_row(meep_geom::geom_epsilon *geps, meep::component c,
                                     double row[3], const meep::volume &v,
                                     double tol, int maxeval) {
    using namespace meep_geom;
    if (maxeval == 0 || !meep::is_electric(c) || v.dim == meep::Dcyl) {
        geps->eff_chi1inv_row(c, row, v, tol, maxeval);
        return;
    }
    const vector3 p = vec_to_vector3(v.center());
    int object_index;
    geom_box_tree tree = geom_tree_search(p, geps->restricted_tree, &object_index);
    auto *material = tree ? static_cast<material_data *>(tree->objects[object_index].o->material)
                          : nullptr;
    if (!material || !is_material_grid(material) || !material->do_averaging ||
        !(v.dim == meep::D3 || anisotropic(material->medium_1) ||
          anisotropic(material->medium_2))) {
        geps->eff_chi1inv_row(c, row, v, tol, maxeval);
        return;
    }
    // Scalar 3D also needs the normalized kernel (Meep 1.34 used integer 4/3).
    // Preserve Meep's analytic geometric-interface branch whenever it applies.
    symm_matrix base;
    bool fallback;
    geps->eff_chi1inv_matrix(c, &base, v, tol, maxeval, fallback);
    if (!fallback) {
        const Matrix matrix{{{base.m00, base.m01, base.m02},
                             {base.m01, base.m11, base.m12},
                             {base.m02, base.m12, base.m22}}};
        const int axis = meep::component_direction(c) % 3;
        for (int j = 0; j < 3; ++j) row[j] = matrix[axis][j];
        return;
    }
    const meep::vec gradient = matgrid_grad(p, tree, object_index, material);
    const double magnitude = meep::abs(gradient);
    const double value = matgrid_val(p, tree, object_index, material) + geps->u_p;
    const Matrix first = epsilon(material->medium_1);
    const Matrix second = epsilon(material->medium_2);
    if (magnitude < 1e-8) {
        const double fill = projected(value, material->beta, material->eta);
        Matrix mixed{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                mixed[i][j] = (1 - fill) * first[i][j] + fill * second[i][j];
        const Matrix matrix = inverse(mixed);
        const int axis = meep::component_direction(c) % 3;
        for (int j = 0; j < 3; ++j) row[j] = matrix[axis][j];
        return;
    }
    std::array<double, 3> normal{};
    LOOP_OVER_DIRECTIONS(v.dim, d) { normal[d % 3] = gradient.in_direction(d) / magnitude; }
    const double fill = projected_average(
        {v.dim, value,
         magnitude * v.diameter() / 2, material->beta, material->eta}, tol, maxeval);
    const Matrix matrix = averaged_inverse(first, second, normal, fill);
    const int axis = meep::component_direction(c) % 3;
    for (int j = 0; j < 3; ++j) row[j] = matrix[axis][j];
}

class TensorMaterial final : public meep::material_function {
    meep_geom::geom_epsilon *geps_;

public:
    explicit TensorMaterial(meep_geom::geom_epsilon *geps) : geps_(geps) {}
    void set_volume(const meep::volume &v) override { geps_->set_volume(v); }
    void unset_volume() override { geps_->unset_volume(); }
    double chi1p1(meep::field_type ft, const meep::vec &r) override {
        return geps_->chi1p1(ft, r);
    }
    void eff_chi1inv_row(meep::component c, double row[3], const meep::volume &v,
                        double tol, int maxeval) override {
        generalized_material_row(geps_, c, row, v, tol, maxeval);
    }
};

} // namespace tama_material_tensor
