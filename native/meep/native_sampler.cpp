#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include <meep.hpp>
#include <meep/meepgeom.hpp>
#include <meep/mympi.hpp>
#include <mpi.h>
#include <gsl/gsl_cblas.h>

#include "material_tensor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <map>
#include <memory>
#include <new>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef NPY_CSETREAL
#define NPY_CSETREAL(c, r) ((c)->real = (r))
#endif
#ifndef NPY_CSETIMAG
#define NPY_CSETIMAG(c, i) ((c)->imag = (i))
#endif

static std::complex<double> npy_to_complex(const npy_cdouble &value) {
    return std::complex<double>(npy_creal(value), npy_cimag(value));
}

static void set_npy_complex(npy_cdouble &target, const std::complex<double> &value) {
    NPY_CSETREAL(&target, value.real());
    NPY_CSETIMAG(&target, value.imag());
}

static void add_npy_complex(npy_cdouble &target, const std::complex<double> &value) {
    NPY_CSETREAL(&target, npy_creal(target) + value.real());
    NPY_CSETIMAG(&target, npy_cimag(target) + value.imag());
}

static int popcount_size_t(size_t value) {
    int count = 0;
    while (value != 0) {
        count += static_cast<int>(value & 1u);
        value >>= 1;
    }
    return count;
}

static bool checked_size_product(size_t left, size_t right, size_t &product) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        return false;
    }
    product = left * right;
    return true;
}

static bool checked_native_design_flat_index(size_t nx, size_t ny, size_t nz, size_t x,
                                             size_t y, size_t z, size_t &index) {
    if (x >= nx || y >= ny || z >= nz) {
        return false;
    }
    size_t row = 0;
    size_t plane_offset = 0;
    if (!checked_size_product(x, ny, row) ||
        row > std::numeric_limits<size_t>::max() - y) {
        return false;
    }
    row += y;
    if (!checked_size_product(row, nz, plane_offset) ||
        plane_offset > std::numeric_limits<size_t>::max() - z) {
        return false;
    }
    index = plane_offset + z;
    return true;
}

static constexpr size_t SUPPORT_MASK_WORD_BITS = sizeof(size_t) * CHAR_BIT;

static size_t support_mask_word_count(int nproc) {
    return (static_cast<size_t>(nproc) + SUPPORT_MASK_WORD_BITS - 1) /
           SUPPORT_MASK_WORD_BITS;
}

static int support_mask_popcount(const size_t *mask, size_t word_count) {
    int count = 0;
    for (size_t word_idx = 0; word_idx < word_count; ++word_idx) {
        count += popcount_size_t(mask[word_idx]);
    }
    return count;
}

static bool support_mask_set_rank(size_t *mask, size_t word_count, int rank) {
    if (rank < 0) {
        return false;
    }
    const size_t rank_index = static_cast<size_t>(rank);
    const size_t word_idx = rank_index / SUPPORT_MASK_WORD_BITS;
    if (word_idx >= word_count) {
        return false;
    }
    mask[word_idx] |= static_cast<size_t>(1) << (rank_index % SUPPORT_MASK_WORD_BITS);
    return true;
}

static bool support_mask_contains_rank(const size_t *mask, size_t word_count,
                                       int rank) {
    if (rank < 0) {
        return false;
    }
    const size_t rank_index = static_cast<size_t>(rank);
    const size_t word_idx = rank_index / SUPPORT_MASK_WORD_BITS;
    if (word_idx >= word_count) {
        return false;
    }
    const size_t rank_bit = static_cast<size_t>(1)
                            << (rank_index % SUPPORT_MASK_WORD_BITS);
    return (mask[word_idx] & rank_bit) != 0;
}

static PyObject *support_mask_summary_for_testing(PyObject *, PyObject *args) {
    int nproc = 0;
    PyObject *ranks_obj = nullptr;
    if (!PyArg_ParseTuple(args, "iO:_support_mask_summary_for_testing", &nproc,
                          &ranks_obj)) {
        return nullptr;
    }
    if (nproc < 0) {
        PyErr_SetString(PyExc_ValueError, "nproc must be non-negative");
        return nullptr;
    }

    PyObject *ranks = PySequence_Fast(ranks_obj, "ranks must be a sequence");
    if (!ranks) {
        return nullptr;
    }
    const size_t word_count = support_mask_word_count(nproc);
    std::vector<size_t> mask;
    try {
        mask.assign(word_count, 0);
    } catch (...) {
        Py_DECREF(ranks);
        throw;
    }
    PyObject **items = PySequence_Fast_ITEMS(ranks);
    const Py_ssize_t rank_count = PySequence_Fast_GET_SIZE(ranks);
    for (Py_ssize_t idx = 0; idx < rank_count; ++idx) {
        const long rank = PyLong_AsLong(items[idx]);
        if (PyErr_Occurred()) {
            Py_DECREF(ranks);
            return nullptr;
        }
        if (rank < 0 || rank >= nproc) {
            Py_DECREF(ranks);
            PyErr_SetString(PyExc_ValueError, "rank must satisfy 0 <= rank < nproc");
            return nullptr;
        }
        support_mask_set_rank(mask.data(), word_count, static_cast<int>(rank));
    }
    Py_DECREF(ranks);

    PyObject *membership = PyTuple_New(nproc);
    if (!membership) {
        return nullptr;
    }
    for (int rank = 0; rank < nproc; ++rank) {
        PyObject *contains = support_mask_contains_rank(mask.data(), word_count, rank)
                                 ? Py_True
                                 : Py_False;
        Py_INCREF(contains);
        PyTuple_SET_ITEM(membership, rank, contains);
    }
    return Py_BuildValue("(KKiN)",
                         static_cast<unsigned long long>(SUPPORT_MASK_WORD_BITS),
                         static_cast<unsigned long long>(word_count),
                         support_mask_popcount(mask.data(), word_count), membership);
}

static PyObject *native_design_flat_index_for_testing(PyObject *, PyObject *args) {
    unsigned long long nx = 0;
    unsigned long long ny = 0;
    unsigned long long nz = 0;
    unsigned long long x = 0;
    unsigned long long y = 0;
    unsigned long long z = 0;
    if (!PyArg_ParseTuple(args, "KKKKKK:_native_design_flat_index_for_testing", &nx,
                          &ny, &nz, &x, &y, &z)) {
        return nullptr;
    }
    if (nx > std::numeric_limits<size_t>::max() ||
        ny > std::numeric_limits<size_t>::max() ||
        nz > std::numeric_limits<size_t>::max() ||
        x > std::numeric_limits<size_t>::max() ||
        y > std::numeric_limits<size_t>::max() ||
        z > std::numeric_limits<size_t>::max()) {
        PyErr_SetString(PyExc_OverflowError, "native design index exceeds size_t");
        return nullptr;
    }
    size_t index = 0;
    if (!checked_native_design_flat_index(
            static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz),
            static_cast<size_t>(x), static_cast<size_t>(y), static_cast<size_t>(z),
            index)) {
        PyErr_SetString(PyExc_OverflowError,
                        "native design index is out of range or overflows size_t");
        return nullptr;
    }
    return PyLong_FromSize_t(index);
}

static PyObject *raise_bad_alloc_for_testing(PyObject *, PyObject *) {
    throw std::bad_alloc();
}

static int read_double_sequence(PyObject *obj, std::vector<double> &out,
                                const char *name) {
    PyObject *seq = PySequence_Fast(obj, name);
    if (!seq) {
        return -1;
    }

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    try {
        out.clear();
        out.reserve(static_cast<size_t>(n));

        PyObject **items = PySequence_Fast_ITEMS(seq);
        for (Py_ssize_t i = 0; i < n; ++i) {
            double value = PyFloat_AsDouble(items[i]);
            if (PyErr_Occurred()) {
                Py_DECREF(seq);
                return -1;
            }
            out.push_back(value);
        }
    } catch (...) {
        Py_DECREF(seq);
        throw;
    }

    Py_DECREF(seq);
    return 0;
}

static bool require_cartesian_2d_fields(meep::fields *fields) {
    if (fields->gv.dim == meep::D2) {
        return true;
    }
    PyErr_SetString(PyExc_ValueError,
                    "component-grid sampling requires a Cartesian 2D Meep simulation");
    return false;
}

static bool require_dynamic_field_component(int component_int) {
    if (component_int >= static_cast<int>(meep::Ex) &&
        component_int < meep::NUM_FIELD_COMPONENTS) {
        return true;
    }
    PyErr_SetString(PyExc_ValueError,
                    "component must be an E/H/D/B Meep field component");
    return false;
}

static PyObject *sample_component_grid(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    PyObject *xs_obj = nullptr;
    PyObject *ys_obj = nullptr;
    int component_int = static_cast<int>(meep::Ez);

    if (!PyArg_ParseTuple(args, "KOO|i", &fields_addr, &xs_obj, &ys_obj,
                          &component_int)) {
        return nullptr;
    }

    if (fields_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "fields pointer address must be non-zero");
        return nullptr;
    }

    if (!require_dynamic_field_component(component_int)) {
        return nullptr;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    if (read_double_sequence(xs_obj, xs, "coords_x must be a sequence") < 0 ||
        read_double_sequence(ys_obj, ys, "coords_y must be a sequence") < 0) {
        return nullptr;
    }

    meep::fields *fields =
        reinterpret_cast<meep::fields *>(static_cast<uintptr_t>(fields_addr));
    if (!require_cartesian_2d_fields(fields)) {
        return nullptr;
    }

    npy_intp dims[2] = {
        static_cast<npy_intp>(xs.size()),
        static_cast<npy_intp>(ys.size()),
    };
    PyObject *arr_obj = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }

    meep::component component = static_cast<meep::component>(component_int);
    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj)));

    size_t k = 0;
    try {
        for (double x : xs) {
            for (double y : ys) {
                std::complex<double> value =
                    fields->get_field(component, meep::vec(x, y), true);
                set_npy_complex(data[k], value);
                ++k;
            }
        }
    } catch (const std::exception &e) {
        Py_DECREF(arr_obj);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(arr_obj);
        PyErr_SetString(PyExc_RuntimeError, "unknown error in Meep field sampling");
        return nullptr;
    }

    return arr_obj;
}

static std::complex<double> sample_component_local(meep::fields *fields,
                                                   meep::component component, double x,
                                                   double y) {
    meep::ivec locs[8];
    double weights[8];
    fields->gv.interpolate(component, meep::vec(x, y), locs, weights);

    std::complex<double> value(0.0, 0.0);
    for (int p = 0; p < 8; ++p) {
        if (weights[p] == 0.0) {
            continue;
        }
        meep::component mapped_component = component;
        meep::ivec mapped_loc = locs[p];
        std::complex<double> phase(1.0, 0.0);
        if (!fields->locate_component_point(&mapped_component, &mapped_loc, &phase)) {
            continue;
        }
        for (int chunk_idx = 0; chunk_idx < fields->num_chunks; ++chunk_idx) {
            meep::fields_chunk *chunk = fields->chunks[chunk_idx];
            if (!chunk || !chunk->is_mine() ||
                !chunk->have_component(mapped_component)) {
                continue;
            }
            if (chunk->gv.owns(mapped_loc)) {
                value +=
                    weights[p] * phase * chunk->get_field(mapped_component, mapped_loc);
                break;
            }
        }
    }
    return value;
}

static constexpr const char *COMPONENT_GRID_PLAN_CAPSULE =
    "tama.native_sampler.ComponentGridPlan";
static constexpr const char *COMPONENT_POINT_PLAN_CAPSULE =
    "tama.native_sampler.ComponentPointPlan";
static constexpr const char *EIGENMODE_OVERLAP_PLAN_CAPSULE =
    "tama.native_sampler.EigenmodeOverlapPlan";
static constexpr const char *NATIVE_DESIGN_PLAN_CAPSULE =
    "tama.native_sampler.NativeDesignPlan";
static constexpr long long NATIVE_SIGNAL_CHECK_INTERVAL = 64;

struct SampleEntry {
    int chunk_idx;
    meep::component component;
    meep::ivec loc;
    std::complex<double> weight;
};

struct PointSampleEntry {
    int chunk_idx;
    meep::component component;
    meep::ivec loc;
    std::complex<double> weight;
};

struct HistoryBoundaryGroup {
    std::vector<size_t> result_indices;
    std::vector<size_t> point_indices;
    std::vector<std::complex<double>> local;
    std::vector<std::complex<double>> reduced;
};

struct ComponentGridPlan {
    meep::fields *fields;
    meep::component component;
    size_t nx;
    size_t ny;
    std::vector<int> support_counts;
    size_t support_mask_words;
    std::vector<size_t> support_masks;
    std::vector<int> support_group_ids;
    std::vector<MPI_Comm> support_comms;
    std::vector<std::vector<SampleEntry>> points;
    bool history_sampling_configured = false;
    bool accumulation_configured = false;
    bool adjoint_difference_initialized = false;
    std::vector<size_t> history_local_indices;
    std::vector<size_t> history_indices;
    std::vector<size_t> accumulation_indices;
    std::vector<std::complex<double>> previous_adjoint_values;
    std::vector<HistoryBoundaryGroup> history_boundary_groups;
};

struct ComponentPointPlan {
    meep::fields *fields;
    meep::component component;
    bool cylindrical;
    std::vector<std::vector<PointSampleEntry>> points;
    std::vector<double> monitor_identity;
    std::vector<std::complex<double>> local;
    std::vector<std::complex<double>> reduced;
    bool history_sampling_configured = false;
    std::vector<size_t> history_indices;
};

struct EigenmodeOverlapPlan {
    std::vector<PyObject *> component_plan_capsules;
    std::vector<ComponentPointPlan *> component_plans;
    std::vector<std::vector<std::complex<double>>> weights;
    std::vector<size_t> output_channels;
};

static std::array<std::complex<double>, 2>
sample_eigenmode_overlap_plan_local(const EigenmodeOverlapPlan *plan);

struct NativeDesignEntry {
    int chunk_idx;
    std::uint8_t stencil_size = 0;
    std::ptrdiff_t field_index;
    double integration_weight;
};

struct NativeDesignForwardSample {
    meep::component component;
    std::ptrdiff_t first_index;
    std::ptrdiff_t second_index;
    bool sum_pair;
};

struct NativeDesignBuildRecord {
    NativeDesignEntry entry;
    std::array<size_t, 8> design_indices{};
    std::array<double, 8> design_weights{};
    std::array<std::int32_t, 5> signature{};
};

static_assert(sizeof(NativeDesignEntry) <= 24,
              "native design entry metadata unexpectedly grew");

struct NativeDesignPlan {
    meep::fields *fields;
    meep::component component;
    size_t nx;
    size_t ny;
    size_t nz;
    int dimensions;
    int signature_width;
    bool material_jacobian = false;
    unsigned mirror_axes = 0;
    size_t stencil_capacity;
    std::vector<NativeDesignEntry> entries;
    std::vector<NativeDesignForwardSample> forward_samples;
    std::vector<size_t> design_indices;
    std::vector<double> design_weights;
    std::vector<std::int32_t> signatures;
    std::vector<std::complex<double>> previous_adjoint_values;
    std::vector<double> previous_real_adjoint_values;
    int adjoint_midpoint_kind = 0;
};

static unsigned native_mirror_axes(const meep::fields *fields) {
    if (fields->gv.dim == meep::Dcyl) {
        return 0;
    }
    unsigned axes = 0;
    for (int n = 1; n < fields->S.multiplicity(); ++n) {
        LOOP_OVER_DIRECTIONS(fields->gv.dim, d) {
            if (fields->S.transform(d, n).flipped) {
                axes |= 1u << static_cast<unsigned>(d);
            }
        }
    }
    return axes;
}

// The reduced Yee grid includes a redundant negative half-cell. Integrate
// only the positive half-space, with half measure on each mirror plane.
static double native_mirror_measure(const meep::fields *fields,
                                    const meep::ivec &location) {
    const unsigned axes = native_mirror_axes(fields);
    double weight = 1.0;
    LOOP_OVER_DIRECTIONS(fields->gv.dim, d) {
        if (!(axes & (1u << static_cast<unsigned>(d)))) {
            continue;
        }
        const int offset =
            location.in_direction(d) - fields->S.i_symmetry_point.in_direction(d);
        if (offset < 0) {
            return 0.0;
        }
        if (offset == 0) {
            weight *= 0.5;
        }
    }
    return weight;
}

static bool fold_native_mirror_point(const meep::fields *fields,
                                     meep::component &component, meep::ivec &location,
                                     std::complex<double> &phase) {
    if (!native_mirror_axes(fields)) {
        return true;
    }
    for (int n = 1; n < fields->S.multiplicity(); ++n) {
        if (fields->S.transform(location, n) == location &&
            fields->S.phase_shift(component, n) != std::complex<double>(1.0, 0.0)) {
            return false;
        }
    }
    for (int n = 0; n < fields->S.multiplicity(); ++n) {
        const meep::ivec candidate = fields->S.transform(location, n);
        if (native_mirror_measure(fields, candidate) > 0.0) {
            location = candidate;
            phase *= fields->S.phase_shift(component, n);
            component = fields->S.transform(component, n);
            return true;
        }
    }
    return false;
}

struct MeepGroupCommunicator {
    MPI_Comm comm = MPI_COMM_NULL;
    bool owned = false;
};

static bool mpi_runtime_active() {
    int initialized = 0;
    int finalized = 0;
    return MPI_Initialized(&initialized) == MPI_SUCCESS && initialized &&
           MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized;
}

static void free_owned_communicator(MPI_Comm &comm) {
    if (comm == MPI_COMM_NULL) {
        return;
    }
    if (mpi_runtime_active()) {
        MPI_Comm_free(&comm);
    }
    comm = MPI_COMM_NULL;
}

static void free_owned_group(MPI_Group &group) {
    if (group == MPI_GROUP_NULL) {
        return;
    }
    if (mpi_runtime_active()) {
        MPI_Group_free(&group);
    }
    group = MPI_GROUP_NULL;
}

static void delete_component_grid_plan(ComponentGridPlan *plan) {
    if (!plan) {
        return;
    }
    for (MPI_Comm &comm : plan->support_comms) {
        free_owned_communicator(comm);
    }
    delete plan;
}

static void require_mpi_success(int error, const char *operation) {
    if (error == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    if (MPI_Error_string(error, message, &length) != MPI_SUCCESS) {
        length = 0;
    }
    std::string detail(operation);
    detail += " failed";
    if (length > 0) {
        detail += ": ";
        detail.append(message, static_cast<size_t>(length));
    }
    throw std::runtime_error(detail);
}

static int checked_complex_mpi_double_count(size_t complex_count) {
    if (complex_count > static_cast<size_t>(INT_MAX / 2)) {
        throw std::overflow_error(
            "complex buffer is too large for an MPI_DOUBLE reduction");
    }
    return static_cast<int>(2 * complex_count);
}

static PyObject *complex_mpi_double_count_for_testing(PyObject *, PyObject *args) {
    unsigned long long complex_count = 0;
    if (!PyArg_ParseTuple(args, "K:_complex_mpi_double_count_for_testing",
                          &complex_count)) {
        return nullptr;
    }
    if (complex_count > std::numeric_limits<size_t>::max()) {
        PyErr_SetString(PyExc_OverflowError, "complex count exceeds size_t");
        return nullptr;
    }
    return PyLong_FromLong(
        checked_complex_mpi_double_count(static_cast<size_t>(complex_count)));
}

static MeepGroupCommunicator active_meep_group_communicator() {
    const int nproc = meep::count_processors();
    const int rank = meep::my_rank();
    const int global_rank = meep::my_global_rank();
    int world_size = 0;
    int world_rank = 0;
    require_mpi_success(MPI_Comm_size(MPI_COMM_WORLD, &world_size), "MPI_Comm_size");
    require_mpi_success(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank), "MPI_Comm_rank");
    if (nproc <= 0 || rank < 0 || rank >= nproc || global_rank < 0 ||
        global_rank >= world_size || world_rank != global_rank) {
        throw std::runtime_error("invalid Meep process-group rank metadata");
    }
    if (nproc == world_size) {
        return {MPI_COMM_WORLD, false};
    }

    std::vector<size_t> local_global_ranks(static_cast<size_t>(nproc), 0);
    std::vector<size_t> global_ranks_encoded(static_cast<size_t>(nproc), 0);
    local_global_ranks[static_cast<size_t>(rank)] =
        static_cast<size_t>(global_rank) + 1;
    meep::sum_to_all(local_global_ranks.data(), global_ranks_encoded.data(), nproc);

    std::vector<int> global_ranks(static_cast<size_t>(nproc), -1);
    std::vector<bool> seen_world_ranks(static_cast<size_t>(world_size), false);
    for (int local_rank = 0; local_rank < nproc; ++local_rank) {
        const size_t encoded = global_ranks_encoded[static_cast<size_t>(local_rank)];
        if (encoded == 0 || encoded > static_cast<size_t>(world_size)) {
            throw std::runtime_error(
                "failed to reconstruct the active Meep process group");
        }
        const int mapped_global_rank = static_cast<int>(encoded - 1);
        if (seen_world_ranks[static_cast<size_t>(mapped_global_rank)]) {
            throw std::runtime_error(
                "active Meep process group contains duplicate global ranks");
        }
        seen_world_ranks[static_cast<size_t>(mapped_global_rank)] = true;
        global_ranks[static_cast<size_t>(local_rank)] = mapped_global_rank;
    }
    if (global_ranks[static_cast<size_t>(rank)] != world_rank) {
        throw std::runtime_error(
            "active Meep process-group rank order is inconsistent");
    }

    MPI_Group world_group = MPI_GROUP_NULL;
    MPI_Group active_group = MPI_GROUP_NULL;
    MPI_Comm active_comm = MPI_COMM_NULL;
    bool world_group_valid = false;
    bool active_group_valid = false;
    bool active_comm_valid = false;
    try {
        const int world_group_error = MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        require_mpi_success(world_group_error, "MPI_Comm_group");
        world_group_valid = true;
        const int active_group_error =
            MPI_Group_incl(world_group, nproc, global_ranks.data(), &active_group);
        require_mpi_success(active_group_error, "MPI_Group_incl");
        active_group_valid = true;

        int *tag_ub = nullptr;
        int tag_ub_available = 0;
        require_mpi_success(
            MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &tag_ub_available),
            "MPI_Comm_get_attr");
        if (!tag_ub_available || !tag_ub || *tag_ub < 0) {
            throw std::runtime_error("MPI_TAG_UB is unavailable");
        }
        const uint64_t tag_range =
            static_cast<uint64_t>(*tag_ub) + static_cast<uint64_t>(1);
        const int tag =
            static_cast<int>(static_cast<uint64_t>(global_ranks.front()) % tag_range);
        const int active_comm_error =
            MPI_Comm_create_group(MPI_COMM_WORLD, active_group, tag, &active_comm);
        require_mpi_success(active_comm_error, "MPI_Comm_create_group");
        active_comm_valid = true;
        if (active_comm == MPI_COMM_NULL) {
            throw std::runtime_error("MPI_Comm_create_group returned MPI_COMM_NULL");
        }

        int active_size = 0;
        int active_rank = 0;
        require_mpi_success(MPI_Comm_size(active_comm, &active_size), "MPI_Comm_size");
        require_mpi_success(MPI_Comm_rank(active_comm, &active_rank), "MPI_Comm_rank");
        if (active_size != nproc || active_rank != rank) {
            throw std::runtime_error(
                "reconstructed communicator does not match the Meep process group");
        }
    } catch (...) {
        if (active_comm_valid) {
            free_owned_communicator(active_comm);
        }
        if (active_group_valid) {
            free_owned_group(active_group);
        }
        if (world_group_valid) {
            free_owned_group(world_group);
        }
        throw;
    }
    free_owned_group(active_group);
    free_owned_group(world_group);
    return {active_comm, true};
}

#include "near2far_sources.hpp"

static void allreduce_inplace_chunked(void *buffer, size_t scalar_count,
                                      size_t scalar_size, MPI_Datatype datatype,
                                      size_t max_mpi_count) {
    if (max_mpi_count == 0) {
        throw std::invalid_argument("MPI reduction chunk size must be positive");
    }
    const size_t chunk_limit = std::min(max_mpi_count, static_cast<size_t>(INT_MAX));
    MeepGroupCommunicator active_comm = active_meep_group_communicator();
    try {
        if (scalar_count == 0) {
            require_mpi_success(MPI_Allreduce(MPI_IN_PLACE, buffer, 0, datatype,
                                              MPI_SUM, active_comm.comm),
                                "MPI_Allreduce");
        }
        char *bytes = static_cast<char *>(buffer);
        for (size_t offset = 0; offset < scalar_count;) {
            const size_t chunk_count = std::min(chunk_limit, scalar_count - offset);
            require_mpi_success(MPI_Allreduce(MPI_IN_PLACE,
                                              bytes + offset * scalar_size,
                                              static_cast<int>(chunk_count), datatype,
                                              MPI_SUM, active_comm.comm),
                                "MPI_Allreduce");
            offset += chunk_count;
        }
    } catch (...) {
        if (active_comm.owned) {
            free_owned_communicator(active_comm.comm);
        }
        throw;
    }
    if (active_comm.owned) {
        free_owned_communicator(active_comm.comm);
    }
}

static bool synchronize_native_adjoint_exception(bool local_exception) {
    PyObject *exception_type = nullptr;
    PyObject *exception_value = nullptr;
    PyObject *exception_traceback = nullptr;
    if (local_exception) {
        PyErr_Fetch(&exception_type, &exception_value, &exception_traceback);
    }

    bool group_exception = false;
    try {
        group_exception = meep::count_processors() > 1
                              ? meep::or_to_all(local_exception)
                              : local_exception;
    } catch (...) {
        if (local_exception) {
            PyErr_Restore(exception_type, exception_value, exception_traceback);
        }
        throw;
    }
    if (!group_exception && !local_exception) {
        return false;
    }
    if (local_exception) {
        PyErr_Restore(exception_type, exception_value, exception_traceback);
    } else if (PyErr_CheckSignals() >= 0) {
        PyErr_SetString(PyExc_KeyboardInterrupt,
                        "native adjoint run was interrupted on another MPI rank");
    }
    return true;
}

static bool check_native_adjoint_signals() {
    return synchronize_native_adjoint_exception(PyErr_CheckSignals() < 0);
}

static PyObject *synchronize_native_adjoint_exception_for_testing(PyObject *,
                                                                  PyObject *args) {
    int inject_exception = 0;
    if (!PyArg_ParseTuple(args, "p:_synchronize_native_adjoint_exception_for_testing",
                          &inject_exception)) {
        return nullptr;
    }
    if (inject_exception) {
        PyErr_SetString(PyExc_KeyboardInterrupt, "injected native exception");
    }
    try {
        if (synchronize_native_adjoint_exception(inject_exception != 0)) {
            return nullptr;
        }
    } catch (const std::exception &exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject *check_native_adjoint_signal_for_testing(PyObject *, PyObject *) {
    if (meep::count_processors() != 1) {
        PyErr_SetString(PyExc_RuntimeError,
                        "the native signal test requires one Meep process");
        return nullptr;
    }
    PyErr_SetInterrupt();
    try {
        if (check_native_adjoint_signals()) {
            return nullptr;
        }
    } catch (const std::exception &exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

static int mirror_material_index(int index, int count) {
    return index >= count ? 2 * count - 1 - index : (index < 0 ? -1 - index : index);
}

static void material_axis_stencil(double coordinate, int count, int &index_1,
                                  int &index_2, double &weight_2) {
    coordinate = coordinate < 0.0 ? -coordinate
                                  : (coordinate > 1.0 ? 1.0 - coordinate : coordinate);
    index_1 = mirror_material_index(static_cast<int>(coordinate * count), count);
    weight_2 = coordinate * count - index_1 - 0.5;
    index_2 = mirror_material_index(weight_2 >= 0.0 ? index_1 + 1 : index_1 - 1, count);
    weight_2 = std::fabs(weight_2);
}

static void add_native_design_stencil_entry(NativeDesignBuildRecord &record,
                                            size_t stencil_capacity,
                                            size_t design_index, double weight) {
    if (weight == 0.0) {
        return;
    }
    for (size_t i = 0; i < record.entry.stencil_size; ++i) {
        if (record.design_indices[i] == design_index) {
            record.design_weights[i] += weight;
            return;
        }
    }
    if (record.entry.stencil_size >= stencil_capacity) {
        throw std::runtime_error(
            "MaterialGrid interpolation stencil exceeds its dimensional capacity");
    }
    record.design_indices[record.entry.stencil_size] = design_index;
    record.design_weights[record.entry.stencil_size] = weight;
    record.entry.stencil_size += 1;
}

static std::array<std::int32_t, 5> native_design_signature(const meep::ivec &loc,
                                                           bool cylindrical) {
    return {
        static_cast<std::int32_t>(cylindrical ? loc.r() : loc.x()),
        static_cast<std::int32_t>(cylindrical ? loc.z() : loc.y()),
        static_cast<std::int32_t>(cylindrical ? 0 : loc.z()),
    };
}

static inline double native_design_real_field_value(const NativeDesignPlan *plan,
                                                    const NativeDesignEntry &entry) {
    meep::fields_chunk *chunk = plan->fields->chunks[entry.chunk_idx];
    const meep::component component =
        plan->material_jacobian
            ? meep::direction_component(meep::Dx,
                                        meep::component_direction(plan->component))
            : plan->component;
    return chunk->f[component][0][entry.field_index];
}

static inline std::complex<double>
native_design_field_value(const NativeDesignPlan *plan,
                          const NativeDesignEntry &entry) {
    meep::fields_chunk *chunk = plan->fields->chunks[entry.chunk_idx];
    const meep::component component =
        plan->material_jacobian
            ? meep::direction_component(meep::Dx,
                                        meep::component_direction(plan->component))
            : plan->component;
    const double real_value = chunk->f[component][0][entry.field_index];
    const double imag_value =
        chunk->f[component][1] ? chunk->f[component][1][entry.field_index] : 0.0;
    return {real_value, imag_value};
}

static inline double
native_design_forward_real_field_value(const NativeDesignPlan *plan,
                                       size_t entry_index) {
    const NativeDesignEntry &entry = plan->entries[entry_index];
    if (!plan->material_jacobian) {
        return native_design_real_field_value(plan, entry);
    }
    const NativeDesignForwardSample &sample = plan->forward_samples[entry_index];
    const meep::realnum *field =
        plan->fields->chunks[entry.chunk_idx]->f[sample.component][0];
    return field[sample.first_index] +
           (sample.sum_pair ? field[sample.second_index] : 0.0);
}

static inline std::complex<double>
native_design_forward_field_value(const NativeDesignPlan *plan, size_t entry_index) {
    const NativeDesignEntry &entry = plan->entries[entry_index];
    if (!plan->material_jacobian) {
        return native_design_field_value(plan, entry);
    }
    const NativeDesignForwardSample &sample = plan->forward_samples[entry_index];
    const meep::realnum *imag =
        plan->fields->chunks[entry.chunk_idx]->f[sample.component][1];
    return {
        native_design_forward_real_field_value(plan, entry_index),
        imag ? imag[sample.first_index] +
                   (sample.sum_pair ? imag[sample.second_index] : 0.0)
             : 0.0,
    };
}

static bool zero_vector3(const vector3 &value) {
    return value.x == 0.0 && value.y == 0.0 && value.z == 0.0;
}

static bool zero_cvector3(const cvector3 &value) {
    return value.x.re == 0.0 && value.x.im == 0.0 && value.y.re == 0.0 &&
           value.y.im == 0.0 && value.z.re == 0.0 && value.z.im == 0.0;
}

static bool
isotropic_nondispersive_electric_medium(const meep_geom::medium_struct &medium) {
    return medium.epsilon_diag.x == medium.epsilon_diag.y &&
           medium.epsilon_diag.x == medium.epsilon_diag.z &&
           zero_cvector3(medium.epsilon_offdiag) &&
           medium.mu_diag.x == medium.mu_diag.y &&
           medium.mu_diag.x == medium.mu_diag.z && zero_cvector3(medium.mu_offdiag) &&
           medium.E_susceptibilities.empty() && medium.H_susceptibilities.empty() &&
           zero_vector3(medium.E_chi2_diag) && zero_vector3(medium.E_chi3_diag) &&
           zero_vector3(medium.H_chi2_diag) && zero_vector3(medium.H_chi3_diag) &&
           zero_vector3(medium.D_conductivity_diag) &&
           zero_vector3(medium.B_conductivity_diag);
}

static bool
real_spd_nondispersive_electric_medium(const meep_geom::medium_struct &medium) {
    const double a = medium.epsilon_diag.x;
    const double b = medium.epsilon_diag.y;
    const double c = medium.epsilon_diag.z;
    const double xy = medium.epsilon_offdiag.x.re;
    const double xz = medium.epsilon_offdiag.y.re;
    const double yz = medium.epsilon_offdiag.z.re;
    const double determinant =
        a * b * c + 2.0 * xy * xz * yz - a * yz * yz - b * xz * xz - c * xy * xy;
    return std::isfinite(a) && std::isfinite(b) && std::isfinite(c) &&
           std::isfinite(xy) && std::isfinite(xz) && std::isfinite(yz) &&
           medium.epsilon_offdiag.x.im == 0.0 && medium.epsilon_offdiag.y.im == 0.0 &&
           medium.epsilon_offdiag.z.im == 0.0 && a > 0.0 && a * b - xy * xy > 0.0 &&
           determinant > 0.0 && std::isfinite(medium.mu_diag.x) &&
           medium.mu_diag.x > 0.0 && medium.mu_diag.x == medium.mu_diag.y &&
           medium.mu_diag.x == medium.mu_diag.z && zero_cvector3(medium.mu_offdiag) &&
           medium.E_susceptibilities.empty() && medium.H_susceptibilities.empty() &&
           zero_vector3(medium.E_chi2_diag) && zero_vector3(medium.E_chi3_diag) &&
           zero_vector3(medium.H_chi2_diag) && zero_vector3(medium.H_chi3_diag) &&
           zero_vector3(medium.D_conductivity_diag) &&
           zero_vector3(medium.B_conductivity_diag);
}

static void append_native_design_record(NativeDesignPlan *plan,
                                        const NativeDesignBuildRecord &record,
                                        const NativeDesignForwardSample &sample) {
    plan->entries.push_back(record.entry);
    plan->forward_samples.push_back(sample);
    for (size_t j = 0; j < plan->stencil_capacity; ++j) {
        plan->design_indices.push_back(record.design_indices[j]);
        plan->design_weights.push_back(record.design_weights[j]);
    }
    for (int j = 0; j < plan->signature_width; ++j) {
        plan->signatures.push_back(record.signature[j]);
    }
}

// The normal of an averaged MaterialGrid depends on every interpolation node,
// including nodes whose value-interpolation weight vanishes at the voxel center.
static size_t native_material_support(meep_geom::geom_epsilon *geps,
                                      meep_geom::material_type material,
                                      const meep::vec &location,
                                      const std::array<double, 3> &center,
                                      const std::array<double, 3> &size,
                                      const std::array<int, 3> &shape,
                                      std::array<size_t, 8> &indices) {
    meep_geom::material_type selected = nullptr;
    geps->get_material_pt(selected, location);
    if (selected != material) {
        return 0;
    }
    const vector3 point = meep_geom::vec_to_vector3(location);
    int object_index = 0;
    geom_box_tree tree = geom_tree_search(point, geps->restricted_tree, &object_index);
    if (!tree || tree->objects[object_index].o->material != material) {
        throw std::runtime_error(
            "native material Jacobian could not resolve its MaterialGrid object");
    }
    const vector3 local = vector3_minus(point, tree->objects[object_index].shiftby);
    const double xyz[3] = {local.x, local.y, local.z};
    int nodes[3][2];
    for (int d = 0; d < 3; ++d) {
        const double coordinate =
            size[d] > 0.0 ? 0.5 + (xyz[d] - center[d]) / size[d] : 0.5;
        if (coordinate < -1e-10 || coordinate > 1.0 + 1e-10) {
            throw std::runtime_error(
                "DesignGrid center/size does not match its MaterialGrid block");
        }
        double unused_weight = 0.0;
        material_axis_stencil(std::max(0.0, std::min(1.0, coordinate)), shape[d],
                              nodes[d][0], nodes[d][1], unused_weight);
    }
    size_t count = 0;
    for (int x : nodes[0]) {
        for (int y : nodes[1]) {
            for (int z : nodes[2]) {
                const size_t index =
                    (static_cast<size_t>(x) * shape[1] + y) * shape[2] + z;
                if (std::find(indices.begin(), indices.begin() + count, index) ==
                    indices.begin() + count) {
                    indices[count++] = index;
                }
            }
        }
    }
    return count;
}

struct NativeMaterialVolumeRestore {
    meep_geom::geom_epsilon *geps;
    ~NativeMaterialVolumeRestore() { geps->unset_volume(); }
};

static size_t reflected_native_design_index(const NativeDesignPlan *, size_t, unsigned);

struct NativeMaterialOrbitRestore {
    double *weights;
    std::vector<size_t> indices;
    std::vector<double> originals;
    NativeMaterialOrbitRestore(double *values, const NativeDesignPlan *plan,
                               size_t index)
        : weights(values) {
        for (unsigned reflection = plan->mirror_axes;;
             reflection = (reflection - 1) & plan->mirror_axes) {
            const size_t reflected =
                reflected_native_design_index(plan, index, reflection);
            if (std::find(indices.begin(), indices.end(), reflected) == indices.end()) {
                indices.push_back(reflected);
                originals.push_back(weights[reflected]);
            }
            if (!reflection)
                break;
        }
    }
    void perturb(double step) {
        for (size_t i = 0; i < indices.size(); ++i)
            weights[indices[i]] = originals[i] + step;
    }
    ~NativeMaterialOrbitRestore() {
        for (size_t i = 0; i < indices.size(); ++i)
            weights[indices[i]] = originals[i];
    }
};

static void native_material_row_derivatives(
    const NativeDesignPlan *plan, meep_geom::geom_epsilon *geps,
    meep_geom::material_type material, meep::component component,
    const meep::volume &voxel, const std::array<size_t, 8> &indices, size_t count,
    double tol, int maxeval, double step,
    std::array<std::array<double, 3>, 8> &derivatives) {
    const vector3 point = meep_geom::vec_to_vector3(voxel.center());
    int object_index = 0;
    geom_box_tree tree = geom_tree_search(point, geps->restricted_tree, &object_index);
    const double density =
        meep_geom::matgrid_val(point, tree, object_index, material) + geps->u_p;
    const bool contrasting = tama_material_tensor::epsilon(material->medium_1) !=
                             tama_material_tensor::epsilon(material->medium_2);
    bool averaged_branch = false;
    double normal_magnitude = 0.0;
    const double projected_density =
        tama_material_tensor::projected(density, material->beta, material->eta);
    if (maxeval > 0 && material->do_averaging && contrasting) {
        meep_geom::symm_matrix unused;
        geps->eff_chi1inv_matrix(component, &unused, voxel, tol, maxeval,
                                 averaged_branch);
        normal_magnitude =
            meep::abs(meep_geom::matgrid_grad(point, tree, object_index, material));
    }
    const auto check_normal_branch = [&]() {
        if (averaged_branch && projected_density > 0.0 && projected_density < 1.0 &&
            ((normal_magnitude < 1e-8) !=
             (meep::abs(meep_geom::matgrid_grad(point, tree, object_index, material)) <
              1e-8))) {
            throw std::runtime_error(
                "averaged MaterialGrid derivative stencil crosses a zero-normal branch with an undefined interface normal "
                "at mixed density; no reliable linear gradient exists for this stencil");
        }
    };
    const bool custom_tensor_average =
        averaged_branch && (voxel.dim == meep::D3 ||
                            tama_material_tensor::anisotropic(material->medium_1) ||
                            tama_material_tensor::anisotropic(material->medium_2));
    if (contrasting && std::isfinite(material->beta) && material->beta > 0.0 &&
        material->eta != 0.5 && density == material->eta && !custom_tensor_average) {
        throw std::runtime_error(
            "MaterialGrid gradient is undefined at Meep's asymmetric projection threshold");
    }
    if (contrasting && std::isinf(material->beta) &&
        !(averaged_branch && normal_magnitude > 1e-8)) {
        if (density == material->eta) {
            throw std::runtime_error(
                "MaterialGrid gradient is undefined at the beta=inf projection threshold");
        }
        // An unaveraged hard projection is locally constant away from its jump.
        return;
    }
    double base[3];
    tama_material_tensor::generalized_material_row(geps, component, base, voxel, tol,
                                                   maxeval);
    for (size_t j = 0; j < count; ++j) {
        NativeMaterialOrbitRestore restore(material->weights, plan, indices[j]);
        // Differentiate the constrained density orbit once, even when both
        // partners occur in this interpolation stencil.
        bool duplicate = false;
        for (size_t previous = 0; previous < j; ++previous) {
            if (std::find(restore.indices.begin(), restore.indices.end(),
                          indices[previous]) != restore.indices.end()) {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
            continue;
        double first[3], second[3];
        const bool forward = restore.originals.front() < step;
        const bool backward = restore.originals.front() > 1.0 - step;
        const double h = backward ? -step : step;
        restore.perturb(h);
        check_normal_branch();
        tama_material_tensor::generalized_material_row(geps, component, first, voxel,
                                                       tol, maxeval);
        restore.perturb((forward || backward) ? 2.0 * h : -h);
        check_normal_branch();
        tama_material_tensor::generalized_material_row(geps, component, second, voxel,
                                                       tol, maxeval);
        for (int d = 0; d < 3; ++d) {
            derivatives[j][d] =
                (forward || backward)
                    ? (-3.0 * base[d] + 4.0 * first[d] - second[d]) / (2.0 * h)
                    : (first[d] - second[d]) / (2.0 * h);
            if (!std::isfinite(derivatives[j][d])) {
                throw std::runtime_error(
                    "native material Jacobian contains a non-finite coefficient");
            }
        }
    }
}

static void build_native_material_jacobian_plan(NativeDesignPlan *plan,
                                                meep_geom::geom_epsilon *geps,
                                                meep_geom::material_type material,
                                                const std::array<double, 3> &center,
                                                const std::array<double, 3> &size,
                                                double tol, int maxeval, double step) {
    NativeMaterialVolumeRestore restore_volume{geps};
    const std::array<int, 3> shape = {
        static_cast<int>(plan->nx),
        static_cast<int>(plan->ny),
        static_cast<int>(plan->nz),
    };
    const meep::direction a = meep::component_direction(plan->component);
    const meep::component adjoint_component = meep::direction_component(meep::Dx, a);
    for (int chunk_idx = 0; chunk_idx < plan->fields->num_chunks; ++chunk_idx) {
        meep::fields_chunk *chunk = plan->fields->chunks[chunk_idx];
        if (!chunk || !chunk->is_mine() || !chunk->have_component(plan->component)) {
            continue;
        }
        if (!chunk->f[adjoint_component][0]) {
            throw std::runtime_error(
                "native material Jacobian requires allocated displacement fields");
        }
        geps->set_volume(chunk->gv.pad().surroundings());
        const meep::ivec shift = meep::unit_ivec(chunk->gv.dim, a);
        const std::ptrdiff_t stride_a = chunk->gv.stride(a);
        LOOP_OVER_VOL_OWNED(chunk->gv, plan->component, idx) {
            IVEC_LOOP_ILOC(chunk->gv, iloc);
            const double mirror_weight = native_mirror_measure(plan->fields, iloc);
            if (mirror_weight == 0.0) {
                continue;
            }
            // Diagonal coefficients live at the Yee point; off-diagonals live
            // at the two shared vertices in Meep's electric OFFDIAG stencil.
            for (int node = -1; node < 2; ++node) {
                const meep::ivec coefficient_location =
                    node == -1 ? iloc : (node == 0 ? iloc - shift : iloc + shift);
                const meep::volume voxel = chunk->gv.dV(coefficient_location, 1.0);
                std::array<size_t, 8> indices{};
                const size_t count = native_material_support(
                    geps, material, voxel.center(), center, size, shape, indices);
                if (!count) {
                    continue;
                }
                std::array<std::array<double, 3>, 8> derivatives{};
                native_material_row_derivatives(plan, geps, material, plan->component,
                                                voxel, indices, count, tol, maxeval,
                                                step, derivatives);
                for (int b = 0; b < 3; ++b) {
                    if ((node == -1) != (b == static_cast<int>(a))) {
                        continue;
                    }
                    NativeDesignBuildRecord record;
                    record.entry.chunk_idx = chunk_idx;
                    record.entry.field_index = idx;
                    record.entry.integration_weight =
                        mirror_weight *
                        chunk->gv.dV(plan->component, idx).full_volume();
                    record.signature = native_design_signature(iloc, false);
                    record.signature[plan->dimensions] = b;
                    record.signature[plan->dimensions + 1] = node;
                    for (size_t j = 0; j < count; ++j) {
                        add_native_design_stencil_entry(
                            record, plan->stencil_capacity, indices[j],
                            -(node == -1 ? 1.0 : 0.25) * derivatives[j][b]);
                    }
                    if (!record.entry.stencil_size) {
                        continue;
                    }
                    NativeDesignForwardSample sample;
                    sample.component = meep::direction_component(
                        meep::Dx, static_cast<meep::direction>(b));
                    sample.first_index = idx + (node == 1 ? stride_a : 0);
                    sample.second_index =
                        sample.first_index -
                        chunk->gv.stride(static_cast<meep::direction>(b));
                    sample.sum_pair = node != -1;
                    if (!chunk->f[sample.component][0]) {
                        throw std::runtime_error(
                            "native tensor material plans require force_all_components=True");
                    }
                    if (sample.first_index < 0 ||
                        static_cast<size_t>(sample.first_index) >= chunk->gv.ntot() ||
                        (sample.sum_pair && (sample.second_index < 0 ||
                                             static_cast<size_t>(sample.second_index) >=
                                                 chunk->gv.ntot()))) {
                        throw std::runtime_error(
                            "native tensor material stencil exceeds its displacement-field halo");
                    }
                    append_native_design_record(plan, record, sample);
                }
            }
        }
    }
}

static const size_t *component_grid_plan_support_mask(const ComponentGridPlan *plan,
                                                      size_t point_idx) {
    return plan->support_masks.data() + point_idx * plan->support_mask_words;
}

static void component_grid_plan_destructor(PyObject *capsule) {
    void *ptr = PyCapsule_GetPointer(capsule, COMPONENT_GRID_PLAN_CAPSULE);
    if (!ptr) {
        PyErr_Clear();
        return;
    }
    ComponentGridPlan *plan = reinterpret_cast<ComponentGridPlan *>(ptr);
    delete_component_grid_plan(plan);
}

static ComponentGridPlan *get_component_grid_plan(PyObject *obj) {
    return reinterpret_cast<ComponentGridPlan *>(
        PyCapsule_GetPointer(obj, COMPONENT_GRID_PLAN_CAPSULE));
}

static void component_point_plan_destructor(PyObject *capsule) {
    void *ptr = PyCapsule_GetPointer(capsule, COMPONENT_POINT_PLAN_CAPSULE);
    if (!ptr) {
        PyErr_Clear();
        return;
    }
    delete reinterpret_cast<ComponentPointPlan *>(ptr);
}

static ComponentPointPlan *get_component_point_plan(PyObject *obj) {
    return reinterpret_cast<ComponentPointPlan *>(
        PyCapsule_GetPointer(obj, COMPONENT_POINT_PLAN_CAPSULE));
}

static void delete_eigenmode_overlap_plan(EigenmodeOverlapPlan *plan) {
    if (!plan) {
        return;
    }
    for (PyObject *capsule : plan->component_plan_capsules) {
        Py_XDECREF(capsule);
    }
    delete plan;
}

static void eigenmode_overlap_plan_destructor(PyObject *capsule) {
    void *ptr = PyCapsule_GetPointer(capsule, EIGENMODE_OVERLAP_PLAN_CAPSULE);
    if (!ptr) {
        PyErr_Clear();
        return;
    }
    delete_eigenmode_overlap_plan(reinterpret_cast<EigenmodeOverlapPlan *>(ptr));
}

static EigenmodeOverlapPlan *get_eigenmode_overlap_plan(PyObject *obj) {
    return reinterpret_cast<EigenmodeOverlapPlan *>(
        PyCapsule_GetPointer(obj, EIGENMODE_OVERLAP_PLAN_CAPSULE));
}

static void native_design_plan_destructor(PyObject *capsule) {
    void *ptr = PyCapsule_GetPointer(capsule, NATIVE_DESIGN_PLAN_CAPSULE);
    if (!ptr) {
        PyErr_Clear();
        return;
    }
    delete reinterpret_cast<NativeDesignPlan *>(ptr);
}

static NativeDesignPlan *get_native_design_plan(PyObject *obj) {
    return reinterpret_cast<NativeDesignPlan *>(
        PyCapsule_GetPointer(obj, NATIVE_DESIGN_PLAN_CAPSULE));
}

static std::complex<double> sample_component_plan_local(const ComponentGridPlan *plan,
                                                        size_t point_idx) {
    std::complex<double> value(0.0, 0.0);
    for (const SampleEntry &entry : plan->points[point_idx]) {
        meep::fields_chunk *chunk = plan->fields->chunks[entry.chunk_idx];
        if (!chunk || !chunk->is_mine() || !chunk->have_component(entry.component)) {
            continue;
        }
        value += entry.weight * chunk->get_field(entry.component, entry.loc);
    }
    return value;
}

static std::complex<double>
sample_component_point_plan_local(const ComponentPointPlan *plan, size_t point_idx) {
    std::complex<double> value(0.0, 0.0);
    for (const PointSampleEntry &entry : plan->points[point_idx]) {
        meep::fields_chunk *chunk = plan->fields->chunks[entry.chunk_idx];
        if (!chunk || !chunk->is_mine() || !chunk->have_component(entry.component)) {
            continue;
        }
        value += entry.weight * chunk->get_field(entry.component, entry.loc);
    }
    return value;
}

static size_t component_point_plan_history_width(const ComponentPointPlan *plan) {
    return plan->history_sampling_configured ? plan->history_indices.size()
                                             : plan->points.size();
}

static size_t component_point_plan_history_index(const ComponentPointPlan *plan,
                                                 size_t history_index) {
    return plan->history_sampling_configured ? plan->history_indices[history_index]
                                             : history_index;
}

static int read_index_sequence(PyObject *obj, std::vector<size_t> &indices,
                               size_t max_size, const char *name) {
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(obj, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    if (!arr) {
        return -1;
    }

    if (PyArray_NDIM(arr) != 1) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError, name);
        return -1;
    }

    npy_intp n = PyArray_DIM(arr, 0);
    npy_intp *data = reinterpret_cast<npy_intp *>(PyArray_DATA(arr));
    try {
        indices.clear();
        indices.reserve(static_cast<size_t>(n));
        for (npy_intp i = 0; i < n; ++i) {
            if (data[i] < 0 || static_cast<size_t>(data[i]) >= max_size) {
                Py_DECREF(arr);
                PyErr_SetString(PyExc_IndexError, "sample point index out of range");
                return -1;
            }
            indices.push_back(static_cast<size_t>(data[i]));
        }
    } catch (...) {
        Py_DECREF(arr);
        throw;
    }

    Py_DECREF(arr);
    return 0;
}

static PyObject *create_component_grid_plan(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    PyObject *xs_obj = nullptr;
    PyObject *ys_obj = nullptr;
    int component_int = static_cast<int>(meep::Ez);

    if (!PyArg_ParseTuple(args, "KOO|i", &fields_addr, &xs_obj, &ys_obj,
                          &component_int)) {
        return nullptr;
    }

    if (fields_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "fields pointer address must be non-zero");
        return nullptr;
    }

    if (!require_dynamic_field_component(component_int)) {
        return nullptr;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    if (read_double_sequence(xs_obj, xs, "coords_x must be a sequence") < 0 ||
        read_double_sequence(ys_obj, ys, "coords_y must be a sequence") < 0) {
        return nullptr;
    }

    size_t total_size = xs.size() * ys.size();
    if (total_size > static_cast<size_t>(INT_MAX)) {
        PyErr_SetString(PyExc_OverflowError,
                        "sample grid is too large for Meep MPI reduction");
        return nullptr;
    }

    meep::fields *fields =
        reinterpret_cast<meep::fields *>(static_cast<uintptr_t>(fields_addr));
    if (!require_cartesian_2d_fields(fields)) {
        return nullptr;
    }
    meep::component component = static_cast<meep::component>(component_int);

    std::unique_ptr<ComponentGridPlan, decltype(&delete_component_grid_plan)>
        plan_owner(new ComponentGridPlan, delete_component_grid_plan);
    ComponentGridPlan *plan = plan_owner.get();
    plan->fields = fields;
    plan->component = component;
    plan->nx = xs.size();
    plan->ny = ys.size();
    plan->support_counts.resize(total_size, 0);
    plan->points.resize(total_size);

    try {
        size_t k = 0;
        for (double x : xs) {
            for (double y : ys) {
                meep::ivec locs[8];
                double weights[8];
                fields->gv.interpolate(component, meep::vec(x, y), locs, weights);

                for (int p = 0; p < 8; ++p) {
                    if (weights[p] == 0.0) {
                        continue;
                    }
                    plan->support_counts[k] += 1;
                    meep::component mapped_component = component;
                    meep::ivec mapped_loc = locs[p];
                    std::complex<double> phase(1.0, 0.0);
                    if (!fields->locate_component_point(&mapped_component, &mapped_loc,
                                                        &phase)) {
                        continue;
                    }
                    for (int chunk_idx = 0; chunk_idx < fields->num_chunks;
                         ++chunk_idx) {
                        meep::fields_chunk *chunk = fields->chunks[chunk_idx];
                        if (!chunk || !chunk->is_mine() ||
                            !chunk->have_component(mapped_component)) {
                            continue;
                        }
                        if (chunk->gv.owns(mapped_loc)) {
                            plan->points[k].push_back({
                                chunk_idx,
                                mapped_component,
                                mapped_loc,
                                weights[p] * phase,
                            });
                            break;
                        }
                    }
                }
                ++k;
            }
        }

        const int nproc = meep::count_processors();
        const int rank = meep::my_rank();
        const size_t mask_words = support_mask_word_count(nproc);
        if (total_size != 0 && mask_words > static_cast<size_t>(INT_MAX) / total_size) {
            throw std::runtime_error(
                "support-rank mask is too large for Meep MPI reduction");
        }
        const size_t mask_size = total_size * mask_words;
        plan->support_mask_words = mask_words;
        plan->support_masks.resize(mask_size, 0);
        plan->support_group_ids.resize(total_size, -1);

        std::vector<size_t> local_masks(mask_size, 0);
        for (size_t point_idx = 0; point_idx < total_size; ++point_idx) {
            if (!plan->points[point_idx].empty()) {
                support_mask_set_rank(local_masks.data() + point_idx * mask_words,
                                      mask_words, rank);
            }
        }
        meep::bw_or_to_all(local_masks.data(), plan->support_masks.data(),
                           static_cast<int>(mask_size));

        std::map<std::vector<size_t>, int> boundary_groups;
        for (size_t point_idx = 0; point_idx < total_size; ++point_idx) {
            const size_t *mask = component_grid_plan_support_mask(plan, point_idx);
            if (support_mask_popcount(mask, mask_words) > 1) {
                boundary_groups.emplace(std::vector<size_t>(mask, mask + mask_words),
                                        -1);
            }
        }

        int group_index = 0;
        plan->support_comms.reserve(boundary_groups.size());
        if (!boundary_groups.empty()) {
            MeepGroupCommunicator active_comm = active_meep_group_communicator();
            try {
                for (auto &group : boundary_groups) {
                    group.second = group_index;
                    MPI_Comm comm = MPI_COMM_NULL;
                    int color =
                        support_mask_contains_rank(group.first.data(), mask_words, rank)
                            ? group_index
                            : MPI_UNDEFINED;
                    const int split_error =
                        MPI_Comm_split(active_comm.comm, color, rank, &comm);
                    require_mpi_success(split_error, "MPI_Comm_split");
                    plan->support_comms.push_back(comm);
                    ++group_index;
                }
            } catch (...) {
                if (active_comm.owned) {
                    free_owned_communicator(active_comm.comm);
                }
                throw;
            }
            if (active_comm.owned) {
                free_owned_communicator(active_comm.comm);
            }
        }

        for (size_t point_idx = 0; point_idx < total_size; ++point_idx) {
            const size_t *mask = component_grid_plan_support_mask(plan, point_idx);
            if (support_mask_popcount(mask, mask_words) <= 1) {
                continue;
            }
            auto group =
                boundary_groups.find(std::vector<size_t>(mask, mask + mask_words));
            plan->support_group_ids[point_idx] = group->second;
        }
    } catch (const std::bad_alloc &) {
        return PyErr_NoMemory();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error while creating Meep sample plan");
        return nullptr;
    }

    PyObject *capsule = PyCapsule_New(plan, COMPONENT_GRID_PLAN_CAPSULE,
                                      component_grid_plan_destructor);
    if (!capsule) {
        return nullptr;
    }
    plan_owner.release();
    return capsule;
}

static PyObject *create_component_point_plan(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    PyObject *xs_obj = nullptr;
    PyObject *ys_obj = nullptr;
    PyObject *zs_obj = nullptr;
    int component_int = static_cast<int>(meep::Ez);
    if (PyTuple_Size(args) == 5) {
        if (!PyArg_ParseTuple(args, "KOOOi", &fields_addr, &xs_obj, &ys_obj, &zs_obj,
                              &component_int)) {
            return nullptr;
        }
    } else {
        if (!PyArg_ParseTuple(args, "KOO|i", &fields_addr, &xs_obj, &ys_obj,
                              &component_int)) {
            return nullptr;
        }
    }
    if (fields_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "fields pointer address must be non-zero");
        return nullptr;
    }
    if (!require_dynamic_field_component(component_int)) {
        return nullptr;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    if (read_double_sequence(xs_obj, xs, "point x coordinates must be a sequence") <
            0 ||
        read_double_sequence(ys_obj, ys, "point y coordinates must be a sequence") <
            0) {
        return nullptr;
    }
    if (zs_obj && read_double_sequence(zs_obj, zs,
                                       "point z coordinates must be a sequence") < 0) {
        return nullptr;
    }
    if (xs.size() != ys.size()) {
        PyErr_SetString(PyExc_ValueError, "point x and y coordinate counts must match");
        return nullptr;
    }
    if (zs_obj && xs.size() != zs.size()) {
        PyErr_SetString(PyExc_ValueError,
                        "point x, y, and z coordinate counts must match");
        return nullptr;
    }
    if (xs.size() > static_cast<size_t>(INT_MAX)) {
        PyErr_SetString(PyExc_OverflowError,
                        "too many monitor points for Meep MPI reduction");
        return nullptr;
    }

    std::unique_ptr<ComponentPointPlan> plan_owner(new ComponentPointPlan);
    ComponentPointPlan *plan = plan_owner.get();
    plan->fields =
        reinterpret_cast<meep::fields *>(static_cast<uintptr_t>(fields_addr));
    plan->component = static_cast<meep::component>(component_int);
    plan->cylindrical = plan->fields->gv.dim == meep::Dcyl;
    plan->monitor_identity.push_back(static_cast<double>(component_int));
    for (size_t i = 0; i < xs.size(); ++i) {
        plan->monitor_identity.push_back(xs[i]);
        plan->monitor_identity.push_back(ys[i]);
        plan->monitor_identity.push_back(zs_obj ? zs[i] : 0.0);
    }
    const bool cartesian_3d = plan->fields->gv.dim == meep::D3;
    if (cartesian_3d && !zs_obj) {
        PyErr_SetString(PyExc_ValueError,
                        "Cartesian 3D point plans require z coordinates");
        return nullptr;
    }
    if (!cartesian_3d && zs_obj) {
        PyErr_SetString(PyExc_ValueError,
                        "z coordinates are accepted only for Cartesian 3D point plans");
        return nullptr;
    }
    plan->points.resize(xs.size());
    plan->local.resize(xs.size(), std::complex<double>(0.0, 0.0));
    plan->reduced.resize(xs.size(), std::complex<double>(0.0, 0.0));

    try {
        for (size_t point_idx = 0; point_idx < xs.size(); ++point_idx) {
            meep::ivec locs[8];
            double weights[8];
            plan->fields->gv.interpolate(
                plan->component,
                plan->cylindrical
                    ? meep::veccyl(xs[point_idx], ys[point_idx])
                    : (cartesian_3d
                           ? meep::vec(xs[point_idx], ys[point_idx], zs[point_idx])
                           : meep::vec(xs[point_idx], ys[point_idx])),
                locs, weights);
            auto append_entry = [plan,
                                 point_idx](meep::component interpolation_component,
                                            const meep::ivec &interpolation_location,
                                            std::complex<double> interpolation_weight) {
                if (plan->cylindrical) {
                    const int symmetry_count = plan->fields->S.multiplicity();
                    for (int symmetry_index = 0; symmetry_index < symmetry_count;
                         ++symmetry_index) {
                        meep::ivec location = plan->fields->S.transform(
                            interpolation_location, symmetry_index);
                        meep::component component = plan->fields->S.transform(
                            interpolation_component, symmetry_index);
                        const std::complex<double> symmetry_phase =
                            plan->fields->S.phase_shift(interpolation_component,
                                                        symmetry_index);
                        std::complex<double> periodic_phase(1.0, 0.0);
                        if (!plan->fields->locate_component_point(&component, &location,
                                                                  &periodic_phase)) {
                            continue;
                        }
                        for (int chunk_idx = 0; chunk_idx < plan->fields->num_chunks;
                             ++chunk_idx) {
                            meep::fields_chunk *chunk = plan->fields->chunks[chunk_idx];
                            if (!chunk || !chunk->gv.owns(location)) {
                                continue;
                            }
                            if (chunk->is_mine() && chunk->have_component(component)) {
                                plan->points[point_idx].push_back({
                                    chunk_idx,
                                    component,
                                    location,
                                    interpolation_weight * symmetry_phase *
                                        periodic_phase,
                                });
                            }
                            return;
                        }
                    }
                    return;
                }

                meep::component component = interpolation_component;
                meep::ivec location = interpolation_location;
                std::complex<double> phase(1.0, 0.0);
                if (!plan->fields->locate_component_point(&component, &location,
                                                          &phase)) {
                    return;
                }
                if (!fold_native_mirror_point(plan->fields, component, location,
                                              phase)) {
                    return;
                }
                for (int chunk_idx = 0; chunk_idx < plan->fields->num_chunks;
                     ++chunk_idx) {
                    meep::fields_chunk *chunk = plan->fields->chunks[chunk_idx];
                    if (!chunk || !chunk->gv.owns(location)) {
                        continue;
                    }
                    if (chunk->is_mine() && chunk->have_component(component)) {
                        plan->points[point_idx].push_back({
                            chunk_idx,
                            component,
                            location,
                            interpolation_weight * phase,
                        });
                    }
                    break;
                }
            };
            for (int p = 0; p < 8; ++p) {
                if (weights[p] == 0.0) {
                    continue;
                }
                // Ep and Hr are dependent at r=0 for |m|=1. Expand their
                // covectors through Ep=i*m*Er and Hp=i*m*Hr.
                const bool dependent_axis_component =
                    plan->cylindrical && locs[p].r() == 0 &&
                    std::abs(std::abs(plan->fields->m) - 1.0) <= 1e-12 &&
                    (plan->component == meep::Ep || plan->component == meep::Hr);
                if (!dependent_axis_component) {
                    append_entry(plan->component, locs[p], weights[p]);
                    continue;
                }

                const meep::component independent_component =
                    plan->component == meep::Ep ? meep::Er : meep::Hp;
                const std::complex<double> axis_relation(
                    0.0,
                    plan->component == meep::Ep ? plan->fields->m : -plan->fields->m);
                meep::ivec independent_locs[8];
                double independent_weights[8];
                plan->fields->gv.interpolate(independent_component,
                                             plan->fields->gv[locs[p]],
                                             independent_locs, independent_weights);
                for (int q = 0; q < 8; ++q) {
                    if (independent_weights[q] != 0.0) {
                        append_entry(independent_component, independent_locs[q],
                                     weights[p] * axis_relation *
                                         independent_weights[q]);
                    }
                }
            }
        }
    } catch (const std::bad_alloc &) {
        return PyErr_NoMemory();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error while creating point sample plan");
        return nullptr;
    }

    PyObject *capsule = PyCapsule_New(plan, COMPONENT_POINT_PLAN_CAPSULE,
                                      component_point_plan_destructor);
    if (!capsule) {
        return nullptr;
    }
    plan_owner.release();
    return capsule;
}

static PyObject *configure_native_material_operator(PyObject *, PyObject *args) {
    unsigned long long structure_addr = 0;
    unsigned long long geps_addr = 0;
    int eps_averaging = 0;
    double tol = 1e-4;
    int maxeval = 100000;
    if (!PyArg_ParseTuple(args, "KKpdi", &structure_addr, &geps_addr, &eps_averaging,
                          &tol, &maxeval)) {
        return nullptr;
    }
    if (!structure_addr || !geps_addr || !std::isfinite(tol) || tol <= 0.0 ||
        maxeval <= 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "material operator pointers and subpixel controls must be valid");
        return nullptr;
    }
    auto *structure =
        reinterpret_cast<meep::structure *>(static_cast<uintptr_t>(structure_addr));
    auto *geps =
        reinterpret_cast<meep_geom::geom_epsilon *>(static_cast<uintptr_t>(geps_addr));
    if (eps_averaging) {
        for (int i = 0; i < geps->geometry.num_items; ++i) {
            auto *grid =
                static_cast<meep_geom::material_type>(geps->geometry.items[i].material);
            if (!grid ||
                grid->which_subclass != meep_geom::material_data::MATERIAL_GRID ||
                !grid->do_averaging ||
                !(structure->gv.dim == meep::D3 ||
                  tama_material_tensor::anisotropic(grid->medium_1) ||
                  tama_material_tensor::anisotropic(grid->medium_2) ||
                  !zero_cvector3(grid->medium_1.epsilon_offdiag) ||
                  !zero_cvector3(grid->medium_2.epsilon_offdiag))) {
                continue;
            }
            if (!real_spd_nondispersive_electric_medium(grid->medium_1) ||
                !real_spd_nondispersive_electric_medium(grid->medium_2) ||
                grid->medium_1.mu_diag.x != grid->medium_2.mu_diag.x ||
                grid->medium_1.mu_diag.y != grid->medium_2.mu_diag.y ||
                grid->medium_1.mu_diag.z != grid->medium_2.mu_diag.z ||
                grid->damping != 0.0 ||
                grid->material_grid_kinds != meep_geom::material_data::U_DEFAULT) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "averaged tensor MaterialGrid requires real SPD nondispersive lossless endpoints, "
                    "fixed positive permeability, damping=0, and a non-overlapping grid");
                return nullptr;
            }
        }
    }
    tama_material_tensor::TensorMaterial material(geps);
    structure->set_epsilon(material, eps_averaging != 0, tol, maxeval);
    Py_RETURN_NONE;
}

static PyObject *create_native_design_plan(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    unsigned long long geps_addr = 0;
    double center_x = 0.0;
    double center_y = 0.0;
    double center_z = 0.0;
    double size_x = 0.0;
    double size_y = 0.0;
    double size_z = 0.0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int component_int = static_cast<int>(meep::Ez);
    int material_jacobian = 0;
    int eps_averaging = 0;
    double subpixel_tol = 1e-4;
    int subpixel_maxeval = 100000;
    double material_derivative_step = 1e-5;
    if (!PyArg_ParseTuple(args, "KKddddddiiii|ppdid", &fields_addr, &geps_addr,
                          &center_x, &center_y, &center_z, &size_x, &size_y, &size_z,
                          &nx, &ny, &nz, &component_int, &material_jacobian,
                          &eps_averaging, &subpixel_tol, &subpixel_maxeval,
                          &material_derivative_step)) {
        return nullptr;
    }
    if (fields_addr == 0 || geps_addr == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "fields and geom_epsilon pointer addresses must be non-zero");
        return nullptr;
    }
    meep::fields *fields =
        reinterpret_cast<meep::fields *>(static_cast<uintptr_t>(fields_addr));
    meep_geom::geom_epsilon *geps =
        reinterpret_cast<meep_geom::geom_epsilon *>(static_cast<uintptr_t>(geps_addr));
    const bool is_2d = fields->gv.dim == meep::D2;
    const bool is_3d = fields->gv.dim == meep::D3;
    const bool is_cylindrical = fields->gv.dim == meep::Dcyl;
    if (material_jacobian &&
        (is_cylindrical || !std::isfinite(subpixel_tol) || subpixel_tol <= 0.0 ||
         subpixel_maxeval <= 0 || !std::isfinite(material_derivative_step) ||
         material_derivative_step <= 0.0 || material_derivative_step > 0.25)) {
        PyErr_SetString(
            PyExc_ValueError,
            "material Jacobian plans require Cartesian geometry and valid subpixel/derivative controls");
        return nullptr;
    }
    if (!is_2d && !is_3d && !is_cylindrical) {
        PyErr_SetString(
            PyExc_ValueError,
            "native design plans require a 2D/3D Cartesian or cylindrical simulation");
        return nullptr;
    }
    const bool valid_size =
        is_cylindrical ? size_x > 0.0 && size_z > 0.0
                       : size_x > 0.0 && size_y > 0.0 && (!is_3d || size_z > 0.0);
    if (!valid_size || nx <= 0 || ny <= 0 || nz <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "native design size and shape must be positive");
        return nullptr;
    }
    if (((is_2d || is_cylindrical) && nz != 1) || (is_3d && size_z <= 0.0)) {
        PyErr_SetString(PyExc_ValueError,
                        "native design shape must match the simulation dimensionality");
        return nullptr;
    }
    size_t design_plane_size = 0;
    size_t design_size = 0;
    if (!checked_size_product(static_cast<size_t>(nx), static_cast<size_t>(ny),
                              design_plane_size) ||
        !checked_size_product(design_plane_size, static_cast<size_t>(nz),
                              design_size) ||
        design_size > static_cast<size_t>(std::numeric_limits<npy_intp>::max())) {
        PyErr_SetString(PyExc_OverflowError,
                        "native design shape is too large for a NumPy array");
        return nullptr;
    }
    meep::component component = static_cast<meep::component>(component_int);
    const bool valid_component =
        is_cylindrical
            ? component == meep::Er || component == meep::Ep || component == meep::Ez
            : component == meep::Ex || component == meep::Ey || component == meep::Ez;
    if (!valid_component) {
        PyErr_SetString(PyExc_ValueError,
                        is_cylindrical
                            ? "cylindrical native design plans require Er, Ep, or Ez"
                            : "native design plans require Ex, Ey, or Ez");
        return nullptr;
    }

    meep_geom::material_type target_material = nullptr;
    int material_grid_count = 0;
    for (int object_idx = 0; object_idx < geps->geometry.num_items; ++object_idx) {
        const geometric_object &object = geps->geometry.items[object_idx];
        meep_geom::material_type material =
            static_cast<meep_geom::material_type>(object.material);
        if (material &&
            material->which_subclass == meep_geom::material_data::MATERIAL_GRID) {
            if (object.which_subclass != geometric_object::BLOCK) {
                continue;
            }
            const vector3 block_size = object.subclass.block_data->size;
            if (std::abs(object.center.x - center_x) > 1e-10 ||
                std::abs(object.center.y - center_y) > 1e-10 ||
                std::abs(object.center.z - center_z) > 1e-10 ||
                std::abs(block_size.x - size_x) > 1e-10 ||
                (is_cylindrical
                     ? std::abs(block_size.z - size_z) > 1e-10
                     : std::abs(block_size.y - size_y) > 1e-10 ||
                           (is_3d && std::abs(block_size.z - size_z) > 1e-10))) {
                continue;
            }
            target_material = material;
            material_grid_count += 1;
        }
    }
    if (material_grid_count != 1 || !target_material) {
        PyErr_SetString(
            PyExc_ValueError,
            "native design plans require exactly one matching MaterialGrid Block");
        return nullptr;
    }
    const bool grid_size_matches =
        is_cylindrical ? static_cast<int>(target_material->grid_size.x) == nx &&
                             static_cast<int>(target_material->grid_size.y) == 1 &&
                             static_cast<int>(target_material->grid_size.z) == ny
                       : static_cast<int>(target_material->grid_size.x) == nx &&
                             static_cast<int>(target_material->grid_size.y) == ny &&
                             static_cast<int>(target_material->grid_size.z) == nz;
    if (!grid_size_matches) {
        PyErr_SetString(PyExc_ValueError,
                        "DesignGrid shape must match the MaterialGrid grid_size");
        return nullptr;
    }
    if ((!material_jacobian &&
         (target_material->do_averaging || target_material->beta != 0.0)) ||
        target_material->damping != 0.0 ||
        target_material->material_grid_kinds != meep_geom::material_data::U_DEFAULT) {
        PyErr_SetString(
            PyExc_ValueError,
            material_jacobian
                ? "native material Jacobian plans require a non-overlapping grid with damping=0"
                : "native design plans require a non-overlapping grid with do_averaging=False, beta=0, and damping=0");
        return nullptr;
    }
    if (material_jacobian &&
        (std::isnan(target_material->beta) || target_material->beta < 0.0 ||
         !std::isfinite(target_material->eta) || target_material->eta < 0.0 ||
         target_material->eta > 1.0 ||
         (std::isinf(target_material->beta) &&
          (target_material->eta == 0.0 || target_material->eta == 1.0)))) {
        PyErr_SetString(
            PyExc_ValueError,
            "native material Jacobian plans require beta>=0 and eta in [0,1] (strictly interior for beta=inf)");
        return nullptr;
    }
    const bool valid_media =
        material_jacobian
            ? real_spd_nondispersive_electric_medium(target_material->medium_1) &&
                  real_spd_nondispersive_electric_medium(target_material->medium_2)
            : isotropic_nondispersive_electric_medium(target_material->medium_1) &&
                  isotropic_nondispersive_electric_medium(target_material->medium_2);
    if (!valid_media) {
        PyErr_SetString(
            PyExc_ValueError,
            material_jacobian
                ? "native material Jacobian plans require real symmetric positive-definite nondispersive electric media with positive isotropic permeability"
                : "native design plans require isotropic nondispersive electric media");
        return nullptr;
    }
    if (target_material->medium_1.mu_diag.x != target_material->medium_2.mu_diag.x ||
        target_material->medium_1.mu_diag.y != target_material->medium_2.mu_diag.y ||
        target_material->medium_1.mu_diag.z != target_material->medium_2.mu_diag.z) {
        PyErr_SetString(PyExc_ValueError,
                        "native design plans require fixed permeability");
        return nullptr;
    }

    std::unique_ptr<NativeDesignPlan> plan_owner;
    NativeDesignPlan *plan = nullptr;
    try {
        plan_owner.reset(new NativeDesignPlan);
        plan = plan_owner.get();
        plan->fields = fields;
        plan->component = component;
        plan->nx = static_cast<size_t>(nx);
        plan->ny = static_cast<size_t>(ny);
        plan->nz = static_cast<size_t>(nz);
        plan->dimensions = is_3d ? 3 : 2;
        plan->material_jacobian = material_jacobian != 0;
        plan->mirror_axes = native_mirror_axes(fields);
        plan->signature_width = plan->dimensions + (material_jacobian ? 2 : 0);
        plan->stencil_capacity = is_3d ? 8 : 4;
        if (material_jacobian) {
            build_native_material_jacobian_plan(
                plan, geps, target_material, {center_x, center_y, center_z},
                {size_x, size_y, size_z}, subpixel_tol,
                eps_averaging ? subpixel_maxeval : 0, material_derivative_step);
        } else {
            const double coordinate_tolerance = 1e-10;
            for (int chunk_idx = 0; chunk_idx < fields->num_chunks; ++chunk_idx) {
                meep::fields_chunk *chunk = fields->chunks[chunk_idx];
                if (!chunk || !chunk->is_mine() || !chunk->have_component(component)) {
                    continue;
                }
                LOOP_OVER_VOL_OWNED(chunk->gv, component, idx) {
                    IVEC_LOOP_ILOC(chunk->gv, iloc);
                    IVEC_LOOP_LOC(chunk->gv, location);
                    const double mirror_weight = native_mirror_measure(fields, iloc);
                    if (mirror_weight == 0.0) {
                        continue;
                    }
                    meep_geom::material_type material = nullptr;
                    geps->get_material_pt(material, location);
                    if (material != target_material) {
                        continue;
                    }

                    const vector3 geometry_location =
                        meep_geom::vec_to_vector3(location);
                    int geometry_object_index = 0;
                    geom_box_tree geometry_object_tree =
                        geom_tree_search(geometry_location, geps->restricted_tree,
                                         &geometry_object_index);
                    if (!geometry_object_tree ||
                        static_cast<meep_geom::material_type>(
                            geometry_object_tree->objects[geometry_object_index]
                                .o->material) != target_material) {
                        throw std::runtime_error(
                            "native design plan could not resolve its MaterialGrid object");
                    }
                    // Match Meep's to_geom_box_coords mapping for the selected
                    // periodic geometry copy by removing its lattice shift.
                    const vector3 material_location = vector3_minus(
                        geometry_location,
                        geometry_object_tree->objects[geometry_object_index].shiftby);

                    double rx = 0.5 + (material_location.x - center_x) / size_x;
                    double ry = 0.5 + ((is_cylindrical ? material_location.z
                                                       : material_location.y) -
                                       (is_cylindrical ? center_z : center_y)) /
                                          (is_cylindrical ? size_z : size_y);
                    double rz =
                        is_3d ? 0.5 + (material_location.z - center_z) / size_z : 0.5;
                    if (rx < -coordinate_tolerance || rx > 1.0 + coordinate_tolerance ||
                        ry < -coordinate_tolerance || ry > 1.0 + coordinate_tolerance ||
                        rz < -coordinate_tolerance || rz > 1.0 + coordinate_tolerance) {
                        throw std::runtime_error(
                            "DesignGrid center/size does not match its MaterialGrid block");
                    }
                    rx = std::max(0.0, std::min(1.0, rx));
                    ry = std::max(0.0, std::min(1.0, ry));
                    rz = std::max(0.0, std::min(1.0, rz));

                    int x1 = 0;
                    int x2 = 0;
                    int y1 = 0;
                    int y2 = 0;
                    int z1 = 0;
                    int z2 = 0;
                    double wx2 = 0.0;
                    double wy2 = 0.0;
                    double wz2 = 0.0;
                    material_axis_stencil(rx, nx, x1, x2, wx2);
                    material_axis_stencil(ry, ny, y1, y2, wy2);
                    material_axis_stencil(rz, nz, z1, z2, wz2);

                    NativeDesignBuildRecord record;
                    record.entry.chunk_idx = chunk_idx;
                    record.entry.field_index = idx;
                    record.entry.integration_weight =
                        is_cylindrical && std::abs(location.r()) <= coordinate_tolerance
                            ? 0.0
                            : mirror_weight *
                                  chunk->gv.dV(component, idx).full_volume();
                    record.signature = native_design_signature(iloc, is_cylindrical);
                    const int xs[2] = {x1, x2};
                    const int ys[2] = {y1, y2};
                    const int zs[2] = {z1, z2};
                    const double x_weights[2] = {1.0 - wx2, wx2};
                    const double y_weights[2] = {1.0 - wy2, wy2};
                    const double z_weights[2] = {1.0 - wz2, wz2};
                    for (int xi = 0; xi < 2; ++xi) {
                        for (int yi = 0; yi < 2; ++yi) {
                            for (int zi = 0; zi < 2; ++zi) {
                                size_t design_index = 0;
                                if (!checked_native_design_flat_index(
                                        plan->nx, plan->ny, plan->nz,
                                        static_cast<size_t>(xs[xi]),
                                        static_cast<size_t>(ys[yi]),
                                        static_cast<size_t>(zs[zi]), design_index) ||
                                    design_index >= design_size) {
                                    throw std::overflow_error(
                                        "native design stencil index is out of range");
                                }
                                add_native_design_stencil_entry(
                                    record, plan->stencil_capacity, design_index,
                                    x_weights[xi] * y_weights[yi] * z_weights[zi]);
                            }
                        }
                    }
                    plan->entries.push_back(record.entry);
                    for (size_t stencil_idx = 0; stencil_idx < plan->stencil_capacity;
                         ++stencil_idx) {
                        plan->design_indices.push_back(
                            record.design_indices[stencil_idx]);
                        plan->design_weights.push_back(
                            record.design_weights[stencil_idx]);
                    }
                    for (int axis = 0; axis < plan->dimensions; ++axis) {
                        plan->signatures.push_back(record.signature[axis]);
                    }
                }
            }
        }
        std::vector<size_t> order(plan->entries.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [plan](size_t left, size_t right) {
            for (int axis = 0; axis < plan->signature_width; ++axis) {
                const size_t left_offset = left * plan->signature_width + axis;
                const size_t right_offset = right * plan->signature_width + axis;
                if (plan->signatures[left_offset] != plan->signatures[right_offset]) {
                    return plan->signatures[left_offset] <
                           plan->signatures[right_offset];
                }
            }
            return plan->entries[left].chunk_idx < plan->entries[right].chunk_idx;
        });
        for (size_t i = 1; i < order.size(); ++i) {
            bool duplicate = true;
            for (int axis = 0; axis < plan->signature_width; ++axis) {
                duplicate =
                    duplicate &&
                    plan->signatures[order[i - 1] * plan->signature_width + axis] ==
                        plan->signatures[order[i] * plan->signature_width + axis];
            }
            if (duplicate) {
                throw std::runtime_error(
                    "native design plan contains a duplicate owned Yee point");
            }
        }
        const size_t entry_count = order.size();
        // Invert sorted-position -> original-position in place. Values at or
        // above entry_count mark visited cycles and avoid a second O(n) array.
        for (size_t i = 0; i < entry_count; ++i) {
            if (order[i] >= entry_count) {
                continue;
            }
            size_t current = i;
            size_t next = order[current];
            while (next != i) {
                const size_t following = order[next];
                order[next] = current + entry_count;
                current = next;
                next = following;
            }
            order[i] = current + entry_count;
        }
        for (size_t &destination : order) {
            destination -= entry_count;
        }
        // Apply original-position -> sorted-position to all parallel arrays.
        for (size_t i = 0; i < order.size(); ++i) {
            while (order[i] != i) {
                const size_t other = order[i];
                std::swap(plan->entries[i], plan->entries[other]);
                if (plan->material_jacobian) {
                    std::swap(plan->forward_samples[i], plan->forward_samples[other]);
                }
                for (size_t stencil_idx = 0; stencil_idx < plan->stencil_capacity;
                     ++stencil_idx) {
                    std::swap(
                        plan->design_indices[i * plan->stencil_capacity + stencil_idx],
                        plan->design_indices[other * plan->stencil_capacity +
                                             stencil_idx]);
                    std::swap(
                        plan->design_weights[i * plan->stencil_capacity + stencil_idx],
                        plan->design_weights[other * plan->stencil_capacity +
                                             stencil_idx]);
                }
                for (int axis = 0; axis < plan->signature_width; ++axis) {
                    std::swap(plan->signatures[i * plan->signature_width + axis],
                              plan->signatures[other * plan->signature_width + axis]);
                }
                std::swap(order[i], order[other]);
            }
        }
    } catch (const std::bad_alloc &) {
        return PyErr_NoMemory();
    } catch (const std::exception &exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error while creating a native design plan");
        return nullptr;
    }

    PyObject *capsule =
        PyCapsule_New(plan, NATIVE_DESIGN_PLAN_CAPSULE, native_design_plan_destructor);
    if (!capsule) {
        return nullptr;
    }
    plan_owner.release();
    return capsule;
}

static PyObject *native_design_plan_local_size(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &plan_obj)) {
        return nullptr;
    }
    NativeDesignPlan *plan = get_native_design_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    return PyLong_FromSize_t(plan->entries.size());
}

static PyObject *native_design_plan_signature(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &plan_obj)) {
        return nullptr;
    }
    NativeDesignPlan *plan = get_native_design_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    npy_intp dims[2] = {
        static_cast<npy_intp>(plan->entries.size()),
        static_cast<npy_intp>(plan->signature_width),
    };
    PyObject *array_obj = PyArray_SimpleNew(2, dims, NPY_INT64);
    if (!array_obj) {
        return nullptr;
    }
    npy_int64 *data = reinterpret_cast<npy_int64 *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(array_obj)));
    for (size_t i = 0; i < plan->signatures.size(); ++i) {
        data[i] = static_cast<npy_int64>(plan->signatures[i]);
    }
    return array_obj;
}

static PyObject *sample_native_design_plan_into(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *destination_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &plan_obj, &destination_obj)) {
        return nullptr;
    }
    NativeDesignPlan *plan = get_native_design_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    PyArrayObject *destination = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(destination_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY));
    if (!destination) {
        return nullptr;
    }
    if (PyArray_NDIM(destination) != 1 ||
        PyArray_DIM(destination, 0) != static_cast<npy_intp>(plan->entries.size())) {
        PyArray_DiscardWritebackIfCopy(destination);
        Py_DECREF(destination);
        PyErr_SetString(PyExc_ValueError,
                        "native design destination has the wrong shape");
        return nullptr;
    }
    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(destination));
    try {
        for (size_t i = 0; i < plan->entries.size(); ++i) {
            set_npy_complex(data[i], native_design_forward_field_value(plan, i));
        }
    } catch (const std::exception &exc) {
        PyArray_DiscardWritebackIfCopy(destination);
        Py_DECREF(destination);
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
    if (PyArray_ResolveWritebackIfCopy(destination) < 0) {
        Py_DECREF(destination);
        return nullptr;
    }
    Py_DECREF(destination);
    Py_RETURN_NONE;
}

static PyObject *sample_native_design_plan_real_into(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *destination_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &plan_obj, &destination_obj)) {
        return nullptr;
    }
    NativeDesignPlan *plan = get_native_design_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    if (!plan->fields->is_real) {
        PyErr_SetString(PyExc_RuntimeError,
                        "real native design sampling requires real Meep fields");
        return nullptr;
    }
    if (!PyArray_Check(destination_obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "native design destination must be a NumPy array");
        return nullptr;
    }

    PyArrayObject *destination = reinterpret_cast<PyArrayObject *>(destination_obj);
    if (PyArray_NDIM(destination) != 1 ||
        PyArray_DIM(destination, 0) != static_cast<npy_intp>(plan->entries.size())) {
        PyErr_SetString(PyExc_ValueError,
                        "native design destination has the wrong shape");
        return nullptr;
    }
    if (PyArray_TYPE(destination) != NPY_DOUBLE || !PyArray_ISCARRAY(destination) ||
        !PyArray_ISNOTSWAPPED(destination)) {
        PyErr_SetString(PyExc_ValueError,
                        "real native design sampling requires a writable C-contiguous "
                        "native-endian float64 destination");
        return nullptr;
    }

    double *data = reinterpret_cast<double *>(PyArray_DATA(destination));
    try {
        for (size_t i = 0; i < plan->entries.size(); ++i) {
            data[i] = native_design_forward_real_field_value(plan, i);
        }
    } catch (const std::exception &exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in real native design sampling");
        return nullptr;
    }
    Py_RETURN_NONE;
}

static int native_design_accumulation_arrays(NativeDesignPlan *plan,
                                             PyObject *values_obj,
                                             PyObject *accumulator_obj,
                                             PyArrayObject *&values,
                                             PyArrayObject *&accumulator) {
    values = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(values_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY));
    if (!values) {
        return -1;
    }
    accumulator = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(accumulator_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY));
    if (!accumulator) {
        Py_DECREF(values);
        return -1;
    }
    const bool shapes_ok =
        PyArray_NDIM(values) == 1 &&
        PyArray_DIM(values, 0) == static_cast<npy_intp>(plan->entries.size()) &&
        PyArray_NDIM(accumulator) == 1 &&
        PyArray_DIM(accumulator, 0) ==
            static_cast<npy_intp>(plan->nx * plan->ny * plan->nz);
    if (!shapes_ok) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_ValueError,
                        "native design values or accumulator has the wrong shape");
        return -1;
    }
    return 0;
}

static size_t reflected_native_design_index(const NativeDesignPlan *plan, size_t index,
                                            unsigned reflection) {
    size_t z = index % plan->nz;
    size_t y = (index / plan->nz) % plan->ny;
    size_t x = index / (plan->ny * plan->nz);
    if (reflection & 1u)
        x = plan->nx - 1 - x;
    if (reflection & 2u)
        y = plan->ny - 1 - y;
    if (reflection & 4u)
        z = plan->nz - 1 - z;
    return (x * plan->ny + y) * plan->nz + z;
}

static void accumulate_native_design_entry(const NativeDesignPlan *plan,
                                           size_t entry_index,
                                           const NativeDesignEntry &entry,
                                           const std::complex<double> &product,
                                           npy_cdouble *accumulator) {
    const std::complex<double> weighted_product = entry.integration_weight * product;
    const size_t stencil_offset = entry_index * plan->stencil_capacity;
    for (size_t stencil_idx = 0; stencil_idx < entry.stencil_size; ++stencil_idx) {
        const size_t index = plan->design_indices[stencil_offset + stencil_idx];
        const std::complex<double> value =
            plan->design_weights[stencil_offset + stencil_idx] * weighted_product;
        if (!plan->mirror_axes) {
            add_npy_complex(accumulator[index], value);
            continue;
        }
        const double count =
            static_cast<double>(1u << popcount_size_t(plan->mirror_axes));
        for (unsigned reflection = 0; reflection <= plan->mirror_axes; ++reflection) {
            if (!(reflection & ~plan->mirror_axes)) {
                add_npy_complex(
                    accumulator[reflected_native_design_index(plan, index, reflection)],
                    value / count);
            }
        }
    }
}

static void accumulate_native_design_entry_real(const NativeDesignPlan *plan,
                                                size_t entry_index,
                                                const NativeDesignEntry &entry,
                                                double product, double *accumulator) {
    const double weighted_product = entry.integration_weight * product;
    const size_t stencil_offset = entry_index * plan->stencil_capacity;
    for (size_t stencil_idx = 0; stencil_idx < entry.stencil_size; ++stencil_idx) {
        const size_t index = plan->design_indices[stencil_offset + stencil_idx];
        const double value =
            plan->design_weights[stencil_offset + stencil_idx] * weighted_product;
        if (!plan->mirror_axes) {
            accumulator[index] += value;
            continue;
        }
        const double count =
            static_cast<double>(1u << popcount_size_t(plan->mirror_axes));
        for (unsigned reflection = 0; reflection <= plan->mirror_axes; ++reflection) {
            if (!(reflection & ~plan->mirror_axes)) {
                accumulator[reflected_native_design_index(plan, index, reflection)] +=
                    value / count;
            }
        }
    }
}

static int native_design_real_accumulation_arrays(NativeDesignPlan *plan,
                                                  PyObject *values_obj,
                                                  PyObject *accumulator_obj,
                                                  PyArrayObject *&values,
                                                  PyArrayObject *&accumulator) {
    if (!plan->fields->is_real) {
        PyErr_SetString(PyExc_RuntimeError,
                        "real native design accumulation requires real Meep fields");
        return -1;
    }
    if (!PyArray_Check(values_obj) || !PyArray_Check(accumulator_obj)) {
        PyErr_SetString(
            PyExc_TypeError,
            "real native design values and accumulator must be NumPy arrays");
        return -1;
    }
    values = reinterpret_cast<PyArrayObject *>(values_obj);
    accumulator = reinterpret_cast<PyArrayObject *>(accumulator_obj);
    const bool shapes_ok =
        PyArray_NDIM(values) == 1 &&
        PyArray_DIM(values, 0) == static_cast<npy_intp>(plan->entries.size()) &&
        PyArray_NDIM(accumulator) == 1 &&
        PyArray_DIM(accumulator, 0) ==
            static_cast<npy_intp>(plan->nx * plan->ny * plan->nz);
    if (!shapes_ok) {
        PyErr_SetString(PyExc_ValueError,
                        "native design values or accumulator has the wrong shape");
        return -1;
    }
    if (PyArray_TYPE(values) != NPY_DOUBLE || !PyArray_ISCARRAY_RO(values) ||
        !PyArray_ISNOTSWAPPED(values)) {
        PyErr_SetString(
            PyExc_ValueError,
            "real native design values must be a C-contiguous native-endian "
            "float64 array");
        return -1;
    }
    if (PyArray_TYPE(accumulator) != NPY_DOUBLE || !PyArray_ISCARRAY(accumulator) ||
        !PyArray_ISNOTSWAPPED(accumulator)) {
        PyErr_SetString(
            PyExc_ValueError,
            "real native design accumulator must be a writable C-contiguous "
            "native-endian float64 array");
        return -1;
    }
    return 0;
}

static PyObject *accumulate_native_design_product_local_inplace(PyObject *,
                                                                PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *values_obj = nullptr;
    PyObject *accumulator_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &plan_obj, &values_obj, &accumulator_obj)) {
        return nullptr;
    }
    NativeDesignPlan *plan = get_native_design_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    PyArrayObject *values = nullptr;
    PyArrayObject *accumulator = nullptr;
    if (native_design_accumulation_arrays(plan, values_obj, accumulator_obj, values,
                                          accumulator) < 0) {
        return nullptr;
    }
    const npy_cdouble *value_data =
        reinterpret_cast<const npy_cdouble *>(PyArray_DATA(values));
    npy_cdouble *accumulator_data =
        reinterpret_cast<npy_cdouble *>(PyArray_DATA(accumulator));
    try {
        for (size_t i = 0; i < plan->entries.size(); ++i) {
            const NativeDesignEntry &entry = plan->entries[i];
            accumulate_native_design_entry(plan, i, entry,
                                           npy_to_complex(value_data[i]) *
                                               native_design_field_value(plan, entry),
                                           accumulator_data);
        }
    } catch (const std::exception &exc) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
    Py_DECREF(values);
    if (PyArray_ResolveWritebackIfCopy(accumulator) < 0) {
        Py_DECREF(accumulator);
        return nullptr;
    }
    Py_DECREF(accumulator);
    Py_RETURN_NONE;
}

static PyObject *accumulate_native_design_real_product_local_inplace(PyObject *,
                                                                     PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *values_obj = nullptr;
    PyObject *accumulator_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &plan_obj, &values_obj, &accumulator_obj)) {
        return nullptr;
    }
    NativeDesignPlan *plan = get_native_design_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    PyArrayObject *values = nullptr;
    PyArrayObject *accumulator = nullptr;
    if (native_design_real_accumulation_arrays(plan, values_obj, accumulator_obj,
                                               values, accumulator) < 0) {
        return nullptr;
    }
    const double *value_data = reinterpret_cast<const double *>(PyArray_DATA(values));
    double *accumulator_data = reinterpret_cast<double *>(PyArray_DATA(accumulator));
    try {
        for (size_t i = 0; i < plan->entries.size(); ++i) {
            const NativeDesignEntry &entry = plan->entries[i];
            accumulate_native_design_entry_real(
                plan, i, entry,
                value_data[i] * native_design_real_field_value(plan, entry),
                accumulator_data);
        }
    } catch (const std::exception &exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in real native design accumulation");
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject *
accumulate_native_design_midpoint_product_local_inplace(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *values_obj = nullptr;
    PyObject *accumulator_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &plan_obj, &values_obj, &accumulator_obj)) {
        return nullptr;
    }
    NativeDesignPlan *plan = get_native_design_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    PyArrayObject *values = nullptr;
    PyArrayObject *accumulator = nullptr;
    if (native_design_accumulation_arrays(plan, values_obj, accumulator_obj, values,
                                          accumulator) < 0) {
        return nullptr;
    }
    const npy_cdouble *value_data =
        reinterpret_cast<const npy_cdouble *>(PyArray_DATA(values));
    npy_cdouble *accumulator_data =
        reinterpret_cast<npy_cdouble *>(PyArray_DATA(accumulator));
    try {
        if (plan->adjoint_midpoint_kind == 0) {
            plan->previous_adjoint_values.resize(plan->entries.size());
            for (size_t i = 0; i < plan->entries.size(); ++i) {
                const NativeDesignEntry &entry = plan->entries[i];
                plan->previous_adjoint_values[i] =
                    native_design_field_value(plan, entry);
            }
            plan->adjoint_midpoint_kind = 1;
            Py_DECREF(values);
            if (PyArray_ResolveWritebackIfCopy(accumulator) < 0) {
                Py_DECREF(accumulator);
                return nullptr;
            }
            Py_DECREF(accumulator);
            Py_RETURN_FALSE;
        }
        if (plan->adjoint_midpoint_kind != 1) {
            throw std::runtime_error(
                "native design midpoint plan cannot mix real and complex accumulation");
        }
        for (size_t i = 0; i < plan->entries.size(); ++i) {
            const NativeDesignEntry &entry = plan->entries[i];
            const std::complex<double> current = native_design_field_value(plan, entry);
            const std::complex<double> midpoint =
                0.5 * (current + plan->previous_adjoint_values[i]);
            accumulate_native_design_entry(plan, i, entry,
                                           npy_to_complex(value_data[i]) * midpoint,
                                           accumulator_data);
            plan->previous_adjoint_values[i] = current;
        }
    } catch (const std::bad_alloc &) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        return PyErr_NoMemory();
    } catch (const std::exception &exc) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
    Py_DECREF(values);
    if (PyArray_ResolveWritebackIfCopy(accumulator) < 0) {
        Py_DECREF(accumulator);
        return nullptr;
    }
    Py_DECREF(accumulator);
    Py_RETURN_TRUE;
}

static PyObject *
accumulate_native_design_real_midpoint_product_local_inplace(PyObject *,
                                                             PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *values_obj = nullptr;
    PyObject *accumulator_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &plan_obj, &values_obj, &accumulator_obj)) {
        return nullptr;
    }
    NativeDesignPlan *plan = get_native_design_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    PyArrayObject *values = nullptr;
    PyArrayObject *accumulator = nullptr;
    if (native_design_real_accumulation_arrays(plan, values_obj, accumulator_obj,
                                               values, accumulator) < 0) {
        return nullptr;
    }
    const double *value_data = reinterpret_cast<const double *>(PyArray_DATA(values));
    double *accumulator_data = reinterpret_cast<double *>(PyArray_DATA(accumulator));
    try {
        if (plan->adjoint_midpoint_kind == 0) {
            plan->previous_real_adjoint_values.resize(plan->entries.size());
            for (size_t i = 0; i < plan->entries.size(); ++i) {
                const NativeDesignEntry &entry = plan->entries[i];
                plan->previous_real_adjoint_values[i] =
                    native_design_real_field_value(plan, entry);
            }
            plan->adjoint_midpoint_kind = 2;
            Py_RETURN_FALSE;
        }
        if (plan->adjoint_midpoint_kind != 2) {
            throw std::runtime_error(
                "native design midpoint plan cannot mix real and complex accumulation");
        }
        for (size_t i = 0; i < plan->entries.size(); ++i) {
            const NativeDesignEntry &entry = plan->entries[i];
            const double current = native_design_real_field_value(plan, entry);
            const double midpoint =
                0.5 * (current + plan->previous_real_adjoint_values[i]);
            accumulate_native_design_entry_real(
                plan, i, entry, value_data[i] * midpoint, accumulator_data);
            plan->previous_real_adjoint_values[i] = current;
        }
    } catch (const std::bad_alloc &) {
        return PyErr_NoMemory();
    } catch (const std::exception &exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in real native midpoint design accumulation");
        return nullptr;
    }
    Py_RETURN_TRUE;
}

struct NativeDerivativeStencils {
    int sampling_interval;
    std::ptrdiff_t first_offset;
    size_t support;
    std::vector<double> forward;
};

static std::pair<long long, int> native_floor_divmod(long long value, int divisor) {
    long long quotient = value / divisor;
    long long remainder = value % divisor;
    if (remainder < 0) {
        --quotient;
        remainder += divisor;
    }
    return {quotient, static_cast<int>(remainder)};
}

static void add_native_reconstruction_stencil(
    std::vector<double> &destination, const double *reconstruction_weights,
    int sampling_interval, size_t reconstruction_support,
    std::ptrdiff_t reconstruction_first_offset, std::ptrdiff_t derivative_first_offset,
    long long fine_offset, double scale) {
    const auto coarse_phase = native_floor_divmod(fine_offset, sampling_interval);
    const long long coarse_offset = coarse_phase.first;
    const int phase = coarse_phase.second;
    const double *phase_weights =
        reconstruction_weights + static_cast<size_t>(phase) * reconstruction_support;
    for (size_t weight_index = 0; weight_index < reconstruction_support;
         ++weight_index) {
        const long long sparse_offset = coarse_offset + reconstruction_first_offset +
                                        static_cast<long long>(weight_index);
        const long long derivative_index = sparse_offset - derivative_first_offset;
        if (derivative_index < 0 ||
            derivative_index >= static_cast<long long>(destination.size())) {
            throw std::runtime_error(
                "native derivative stencil exceeded its allocated support");
        }
        destination[static_cast<size_t>(derivative_index)] +=
            scale * phase_weights[weight_index];
    }
}

static NativeDerivativeStencils
make_native_derivative_stencils(const double *reconstruction_weights,
                                int sampling_interval, size_t reconstruction_support,
                                std::ptrdiff_t reconstruction_first_offset, double dt) {
    NativeDerivativeStencils result;
    result.sampling_interval = sampling_interval;
    result.first_offset = reconstruction_first_offset;
    result.support = reconstruction_support + 1;
    result.forward.assign(static_cast<size_t>(sampling_interval) * result.support, 0.0);
    const double forward_scale = 1.0 / dt;
    for (int phase = 0; phase < sampling_interval; ++phase) {
        std::vector<double> row(result.support, 0.0);
        add_native_reconstruction_stencil(
            row, reconstruction_weights, sampling_interval, reconstruction_support,
            reconstruction_first_offset, result.first_offset,
            static_cast<long long>(phase) + 1, forward_scale);
        add_native_reconstruction_stencil(
            row, reconstruction_weights, sampling_interval, reconstruction_support,
            reconstruction_first_offset, result.first_offset,
            static_cast<long long>(phase), -forward_scale);
        std::copy(row.begin(), row.end(),
                  result.forward.begin() + static_cast<size_t>(phase) * result.support);
    }

    return result;
}

static bool native_history_dtype_supported(int type_number) {
    return type_number == NPY_FLOAT16 || type_number == NPY_FLOAT32 ||
           type_number == NPY_FLOAT64 || type_number == NPY_LONGDOUBLE ||
           type_number == NPY_COMPLEX64 || type_number == NPY_COMPLEX128 ||
           type_number == NPY_CLONGDOUBLE;
}

static bool native_history_dtype_complex(int type_number) {
    return type_number == NPY_COMPLEX64 || type_number == NPY_COMPLEX128 ||
           type_number == NPY_CLONGDOUBLE;
}

static npy_half native_double_to_half(double value) {
    const uint16_t sign = std::signbit(value) ? 0x8000u : 0u;
    const double magnitude = std::abs(value);
    if (std::isnan(magnitude)) {
        return static_cast<npy_half>(sign | 0x7e00u);
    }
    if (std::isinf(magnitude)) {
        return static_cast<npy_half>(sign | 0x7c00u);
    }
    if (magnitude == 0.0) {
        return static_cast<npy_half>(sign);
    }

    if (magnitude < std::ldexp(1.0, -14)) {
        const double rounded = std::nearbyint(std::ldexp(magnitude, 24));
        if (rounded <= 0.0) {
            return static_cast<npy_half>(sign);
        }
        if (rounded >= 1024.0) {
            return static_cast<npy_half>(sign | 0x0400u);
        }
        return static_cast<npy_half>(sign | static_cast<uint16_t>(rounded));
    }

    int binary_exponent = 0;
    const double fraction = std::frexp(magnitude, &binary_exponent);
    int half_exponent = binary_exponent - 1 + 15;
    double mantissa = std::nearbyint((2.0 * fraction - 1.0) * 1024.0);
    if (mantissa >= 1024.0) {
        mantissa = 0.0;
        ++half_exponent;
    }
    if (half_exponent >= 31) {
        return static_cast<npy_half>(sign | 0x7c00u);
    }
    return static_cast<npy_half>(sign | (static_cast<uint16_t>(half_exponent) << 10u) |
                                 static_cast<uint16_t>(mantissa));
}

static void write_native_history_value(PyArrayObject *history, npy_intp row,
                                       npy_intp column,
                                       const std::complex<double> &value) {
    char *destination = PyArray_BYTES(history) + row * PyArray_STRIDE(history, 0) +
                        column * PyArray_STRIDE(history, 1);
    switch (PyArray_TYPE(history)) {
    case NPY_FLOAT16:
        *reinterpret_cast<npy_half *>(destination) =
            native_double_to_half(value.real());
        return;
    case NPY_FLOAT32:
        *reinterpret_cast<float *>(destination) = static_cast<float>(value.real());
        return;
    case NPY_FLOAT64:
        *reinterpret_cast<double *>(destination) = value.real();
        return;
    case NPY_LONGDOUBLE:
        *reinterpret_cast<npy_longdouble *>(destination) =
            static_cast<npy_longdouble>(value.real());
        return;
    case NPY_COMPLEX64: {
        npy_cfloat *target = reinterpret_cast<npy_cfloat *>(destination);
        *target = npy_cpackf(static_cast<float>(value.real()),
                             static_cast<float>(value.imag()));
        return;
    }
    case NPY_COMPLEX128:
        set_npy_complex(*reinterpret_cast<npy_cdouble *>(destination), value);
        return;
    case NPY_CLONGDOUBLE: {
        npy_clongdouble *target = reinterpret_cast<npy_clongdouble *>(destination);
        *target = npy_cpackl(static_cast<npy_longdouble>(value.real()),
                             static_cast<npy_longdouble>(value.imag()));
        return;
    }
    default:
        throw std::runtime_error("unsupported native history dtype");
    }
}

static bool validate_native_forward_history(PyArrayObject *history, npy_intp rows,
                                            npy_intp columns, bool require_complex,
                                            const char *name) {
    const int type_number = PyArray_TYPE(history);
    if (!native_history_dtype_supported(type_number) || PyArray_NDIM(history) != 2 ||
        PyArray_DIM(history, 0) < rows || PyArray_DIM(history, 1) != columns ||
        !PyArray_ISWRITEABLE(history) || !PyArray_ISALIGNED(history) ||
        !PyArray_ISNOTSWAPPED(history) ||
        (require_complex && !native_history_dtype_complex(type_number)) ||
        (!require_complex && native_history_dtype_complex(type_number))) {
        PyErr_SetString(PyExc_ValueError, name);
        return false;
    }
    return true;
}

static double native_half_to_double(npy_half value) {
    const uint16_t bits = static_cast<uint16_t>(value);
    const double sign = (bits & 0x8000u) ? -1.0 : 1.0;
    const uint16_t exponent = (bits >> 10u) & 0x1fu;
    const uint16_t significand = bits & 0x03ffu;
    if (exponent == 0) {
        return sign * std::ldexp(static_cast<double>(significand), -24);
    }
    if (exponent == 0x1fu) {
        return significand == 0 ? sign * std::numeric_limits<double>::infinity()
                                : std::numeric_limits<double>::quiet_NaN();
    }
    return sign * std::ldexp(static_cast<double>(0x0400u + significand),
                             static_cast<int>(exponent) - 25);
}

static void add_native_history_row(PyArrayObject *history, npy_intp row, double weight,
                                   bool complex_output, double *destination) {
    const npy_intp width = PyArray_DIM(history, 1);
    const int type_number = PyArray_TYPE(history);
    if (type_number == NPY_FLOAT16) {
        const npy_half *source =
            reinterpret_cast<const npy_half *>(PyArray_DATA(history)) + row * width;
        for (npy_intp point = 0; point < width; ++point) {
            const double value = native_half_to_double(source[point]);
            destination[(complex_output ? 2 : 1) * point] += weight * value;
        }
        return;
    }
    if (type_number == NPY_FLOAT64) {
        const double *source =
            reinterpret_cast<const double *>(PyArray_DATA(history)) + row * width;
        if (complex_output) {
            for (npy_intp point = 0; point < width; ++point) {
                destination[2 * point] += weight * source[point];
            }
        } else {
            for (npy_intp point = 0; point < width; ++point) {
                destination[point] += weight * source[point];
            }
        }
        return;
    }
    if (type_number == NPY_FLOAT32) {
        const float *source =
            reinterpret_cast<const float *>(PyArray_DATA(history)) + row * width;
        if (complex_output) {
            for (npy_intp point = 0; point < width; ++point) {
                destination[2 * point] += weight * source[point];
            }
        } else {
            for (npy_intp point = 0; point < width; ++point) {
                destination[point] += weight * source[point];
            }
        }
        return;
    }
    if (type_number == NPY_LONGDOUBLE) {
        const npy_longdouble *source =
            reinterpret_cast<const npy_longdouble *>(PyArray_DATA(history)) +
            row * width;
        for (npy_intp point = 0; point < width; ++point) {
            destination[(complex_output ? 2 : 1) * point] +=
                weight * static_cast<double>(source[point]);
        }
        return;
    }
    if (type_number == NPY_COMPLEX128) {
        const npy_cdouble *source =
            reinterpret_cast<const npy_cdouble *>(PyArray_DATA(history)) + row * width;
        for (npy_intp point = 0; point < width; ++point) {
            destination[2 * point] += weight * npy_creal(source[point]);
            destination[2 * point + 1] += weight * npy_cimag(source[point]);
        }
        return;
    }
    if (type_number == NPY_COMPLEX64) {
        const npy_cfloat *source =
            reinterpret_cast<const npy_cfloat *>(PyArray_DATA(history)) + row * width;
        for (npy_intp point = 0; point < width; ++point) {
            destination[2 * point] += weight * npy_crealf(source[point]);
            destination[2 * point + 1] += weight * npy_cimagf(source[point]);
        }
        return;
    }
    const npy_clongdouble *source =
        reinterpret_cast<const npy_clongdouble *>(PyArray_DATA(history)) + row * width;
    for (npy_intp point = 0; point < width; ++point) {
        destination[2 * point] +=
            weight * static_cast<double>(npy_creall(source[point]));
        destination[2 * point + 1] +=
            weight * static_cast<double>(npy_cimagl(source[point]));
    }
}

static void
apply_native_derivative_weights_manual(PyArrayObject *history, long long coarse_index,
                                       const NativeDerivativeStencils &stencils,
                                       const double *weights, bool complex_output,
                                       double *destination) {
    const npy_intp history_rows = PyArray_DIM(history, 0);
    const npy_intp width = PyArray_DIM(history, 1);
    const size_t physical_width =
        static_cast<size_t>(width) * (complex_output ? 2u : 1u);
    std::fill(destination, destination + physical_width, 0.0);
    for (size_t offset_index = 0; offset_index < stencils.support; ++offset_index) {
        const long long history_row =
            coarse_index + stencils.first_offset + static_cast<long long>(offset_index);
        if (history_row < 0 || history_row >= history_rows) {
            continue;
        }
        const double weight = weights[offset_index];
        if (weight != 0.0) {
            add_native_history_row(history, static_cast<npy_intp>(history_row), weight,
                                   complex_output, destination);
        }
    }
}

static void fill_native_derivative_block(PyArrayObject *history, long long coarse_index,
                                         long long last_coarse_index,
                                         const NativeDerivativeStencils &stencils,
                                         bool complex_output,
                                         std::vector<double> &destination) {
    const npy_intp width = PyArray_DIM(history, 1);
    const size_t physical_width =
        static_cast<size_t>(width) * (complex_output ? 2u : 1u);
    destination.assign(static_cast<size_t>(stencils.sampling_interval) * physical_width,
                       0.0);
    // No forward step exists after the final sample.
    if (physical_width == 0 || coarse_index == last_coarse_index) {
        return;
    }

    const bool double_blas_path =
        (!complex_output && PyArray_TYPE(history) == NPY_FLOAT64) ||
        (complex_output && PyArray_TYPE(history) == NPY_COMPLEX128);
    if (stencils.sampling_interval > 1 && double_blas_path) {
        const long long raw_start = coarse_index + stencils.first_offset;
        const long long raw_stop = raw_start + static_cast<long long>(stencils.support);
        const long long history_start = std::max(0LL, raw_start);
        const long long history_stop =
            std::min(static_cast<long long>(PyArray_DIM(history, 0)), raw_stop);
        if (history_start < history_stop) {
            const int matrix_rows = stencils.sampling_interval;
            const int matrix_columns = static_cast<int>(physical_width);
            const int shared_size = static_cast<int>(history_stop - history_start);
            const int weight_start = static_cast<int>(history_start - raw_start);
            const double *history_data =
                reinterpret_cast<const double *>(PyArray_DATA(history)) +
                static_cast<size_t>(history_start) * physical_width;
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_rows,
                        matrix_columns, shared_size, 1.0,
                        stencils.forward.data() + weight_start,
                        static_cast<int>(stencils.support), history_data,
                        matrix_columns, 0.0, destination.data(), matrix_columns);
        }
    } else {
        for (int phase = 0; phase < stencils.sampling_interval; ++phase) {
            apply_native_derivative_weights_manual(
                history, coarse_index, stencils,
                stencils.forward.data() + static_cast<size_t>(phase) * stencils.support,
                complex_output,
                destination.data() + static_cast<size_t>(phase) * physical_width);
        }
    }
}

static void accumulate_native_adjoint_real_row(NativeDesignPlan *plan,
                                               const double *forward_derivative,
                                               double *accumulator, bool midpoint) {
    if (midpoint && plan->adjoint_midpoint_kind == 0) {
        plan->previous_real_adjoint_values.resize(plan->entries.size());
        for (size_t index = 0; index < plan->entries.size(); ++index) {
            plan->previous_real_adjoint_values[index] =
                native_design_real_field_value(plan, plan->entries[index]);
        }
        plan->adjoint_midpoint_kind = 2;
        return;
    }
    if (midpoint && plan->adjoint_midpoint_kind != 2) {
        throw std::runtime_error(
            "native adjoint loop cannot mix real and complex midpoint accumulation");
    }
    for (size_t index = 0; index < plan->entries.size(); ++index) {
        const NativeDesignEntry &entry = plan->entries[index];
        const double current = native_design_real_field_value(plan, entry);
        const double adjoint_value =
            midpoint ? 0.5 * (current + plan->previous_real_adjoint_values[index])
                     : current;
        accumulate_native_design_entry_real(
            plan, index, entry, forward_derivative[index] * adjoint_value, accumulator);
        if (midpoint) {
            plan->previous_real_adjoint_values[index] = current;
        }
    }
}

static void accumulate_native_adjoint_complex_row(NativeDesignPlan *plan,
                                                  const double *forward_derivative,
                                                  bool derivative_is_complex,
                                                  npy_cdouble *accumulator,
                                                  bool midpoint) {
    if (midpoint && plan->adjoint_midpoint_kind == 0) {
        plan->previous_adjoint_values.resize(plan->entries.size());
        for (size_t index = 0; index < plan->entries.size(); ++index) {
            plan->previous_adjoint_values[index] =
                native_design_field_value(plan, plan->entries[index]);
        }
        plan->adjoint_midpoint_kind = 1;
        return;
    }
    if (midpoint && plan->adjoint_midpoint_kind != 1) {
        throw std::runtime_error(
            "native adjoint loop cannot mix real and complex midpoint accumulation");
    }
    for (size_t index = 0; index < plan->entries.size(); ++index) {
        const NativeDesignEntry &entry = plan->entries[index];
        const std::complex<double> current = native_design_field_value(plan, entry);
        const std::complex<double> adjoint_value =
            midpoint ? 0.5 * (current + plan->previous_adjoint_values[index]) : current;
        const std::complex<double> derivative =
            derivative_is_complex
                ? std::complex<double>(forward_derivative[2 * index],
                                       forward_derivative[2 * index + 1])
                : std::complex<double>(forward_derivative[index], 0.0);
        accumulate_native_design_entry(plan, index, entry, derivative * adjoint_value,
                                       accumulator);
        if (midpoint) {
            plan->previous_adjoint_values[index] = current;
        }
    }
}

static PyObject *native_forward_step_count(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    double run_until = 0.0;
    if (!PyArg_ParseTuple(args, "Kd:native_forward_step_count", &fields_addr,
                          &run_until)) {
        return nullptr;
    }
    if (fields_addr == 0 || !std::isfinite(run_until) || run_until <= 0.0) {
        PyErr_SetString(PyExc_ValueError,
                        "native forward run time must be positive and finite");
        return nullptr;
    }
    meep::fields *fields =
        reinterpret_cast<meep::fields *>(static_cast<uintptr_t>(fields_addr));
    const long long initial_step = fields->t;
    const double target_time = fields->round_time() + run_until;
    const double estimated_steps = std::ceil(run_until / fields->dt);
    if (!std::isfinite(fields->dt) || fields->dt <= 0.0 ||
        !std::isfinite(target_time) || !std::isfinite(estimated_steps) ||
        estimated_steps >=
            static_cast<double>(std::numeric_limits<long long>::max() - initial_step)) {
        PyErr_SetString(PyExc_OverflowError, "native forward step count exceeds int64");
        return nullptr;
    }
    long long step_count = std::max(1LL, static_cast<long long>(estimated_steps));
    const long long remaining_steps =
        std::numeric_limits<long long>::max() - initial_step;
    while (true) {
        if (step_count > remaining_steps) {
            PyErr_SetString(PyExc_OverflowError,
                            "native forward step count exceeds int64");
            return nullptr;
        }
        if (static_cast<float>((initial_step + step_count) * fields->dt) >=
            target_time) {
            break;
        }
        ++step_count;
    }
    while (step_count > 0 && static_cast<float>((initial_step + step_count - 1) *
                                                fields->dt) >= target_time) {
        --step_count;
    }
    return PyLong_FromLongLong(step_count);
}

static PyObject *run_native_forward_segment(PyObject *, PyObject *args) {
    PyObject *monitor_plans_obj = nullptr;
    PyObject *monitor_histories_obj = nullptr;
    PyObject *design_plans_obj = nullptr;
    PyObject *design_histories_obj = nullptr;
    PyObject *monitor_times_obj = nullptr;
    int sampling_interval = 0;
    long long fine_step_count = 0;
    long long start_fine_index = 0;
    long long sample_count = 0;
    unsigned long long fields_addr = 0;
    if (!PyArg_ParseTuple(args, "OOOOiLLLO|K:run_native_forward_segment",
                          &monitor_plans_obj, &monitor_histories_obj, &design_plans_obj,
                          &design_histories_obj, &sampling_interval, &fine_step_count,
                          &start_fine_index, &sample_count, &monitor_times_obj,
                          &fields_addr)) {
        return nullptr;
    }
    if (sampling_interval < 1 || fine_step_count < 1 ||
        fine_step_count >=
            static_cast<long long>(std::numeric_limits<npy_intp>::max()) ||
        fine_step_count % sampling_interval != 0 || start_fine_index < 0 ||
        start_fine_index > fine_step_count || sample_count < 1 ||
        sample_count > fine_step_count - start_fine_index + 1) {
        PyErr_SetString(PyExc_ValueError,
                        "native forward segment received an invalid time grid");
        return nullptr;
    }
    const auto setup_work_start = std::chrono::steady_clock::now();

    PyObject *monitor_plans = PySequence_Fast(
        monitor_plans_obj, "native forward monitor plans must be a sequence");
    if (!monitor_plans) {
        return nullptr;
    }
    PyObject *monitor_histories = PySequence_Fast(
        monitor_histories_obj, "native forward monitor histories must be a sequence");
    if (!monitor_histories) {
        Py_DECREF(monitor_plans);
        return nullptr;
    }
    PyObject *design_plans = PySequence_Fast(
        design_plans_obj, "native forward design plans must be a sequence");
    if (!design_plans) {
        Py_DECREF(monitor_histories);
        Py_DECREF(monitor_plans);
        return nullptr;
    }
    PyObject *design_histories = PySequence_Fast(
        design_histories_obj, "native forward design histories must be a sequence");
    if (!design_histories) {
        Py_DECREF(design_plans);
        Py_DECREF(monitor_histories);
        Py_DECREF(monitor_plans);
        return nullptr;
    }
    auto cleanup = [&]() {
        Py_DECREF(design_histories);
        Py_DECREF(design_plans);
        Py_DECREF(monitor_histories);
        Py_DECREF(monitor_plans);
    };

    const Py_ssize_t monitor_count = PySequence_Fast_GET_SIZE(monitor_plans);
    const Py_ssize_t design_count = PySequence_Fast_GET_SIZE(design_plans);
    if (PySequence_Fast_GET_SIZE(monitor_histories) != monitor_count ||
        PySequence_Fast_GET_SIZE(design_histories) != design_count) {
        cleanup();
        PyErr_SetString(
            PyExc_ValueError,
            "native forward plans and histories must have matching lengths");
        return nullptr;
    }
    if (!PyArray_Check(monitor_times_obj)) {
        cleanup();
        PyErr_SetString(PyExc_TypeError,
                        "native forward monitor times must be a NumPy array");
        return nullptr;
    }
    PyArrayObject *monitor_times = reinterpret_cast<PyArrayObject *>(monitor_times_obj);
    if (PyArray_NDIM(monitor_times) != 1 ||
        PyArray_DIM(monitor_times, 0) < fine_step_count + 1 ||
        PyArray_TYPE(monitor_times) != NPY_FLOAT64 ||
        !PyArray_ISWRITEABLE(monitor_times) || !PyArray_ISALIGNED(monitor_times) ||
        !PyArray_ISNOTSWAPPED(monitor_times)) {
        cleanup();
        PyErr_SetString(
            PyExc_ValueError,
            "native forward monitor times must be a writable float64 vector");
        return nullptr;
    }

    struct MonitorBinding {
        ComponentPointPlan *point = nullptr;
        EigenmodeOverlapPlan *overlap = nullptr;
        PyArrayObject *history = nullptr;
    };
    struct DesignBinding {
        NativeDesignPlan *plan = nullptr;
        PyArrayObject *history = nullptr;
    };
    std::vector<MonitorBinding> monitors;
    std::vector<DesignBinding> designs;
    try {
        monitors.reserve(static_cast<size_t>(monitor_count));
        designs.reserve(static_cast<size_t>(design_count));
    } catch (const std::bad_alloc &) {
        cleanup();
        return PyErr_NoMemory();
    }

    meep::fields *fields =
        reinterpret_cast<meep::fields *>(static_cast<uintptr_t>(fields_addr));
    for (Py_ssize_t index = 0; index < monitor_count; ++index) {
        PyObject *plan_obj = PySequence_Fast_GET_ITEM(monitor_plans, index);
        PyObject *history_obj = PySequence_Fast_GET_ITEM(monitor_histories, index);
        if (!PyArray_Check(history_obj)) {
            cleanup();
            PyErr_SetString(PyExc_TypeError,
                            "native forward monitor history must be a NumPy array");
            return nullptr;
        }
        MonitorBinding binding;
        binding.history = reinterpret_cast<PyArrayObject *>(history_obj);
        bool require_complex = false;
        npy_intp width = 0;
        if (PyCapsule_IsValid(plan_obj, COMPONENT_POINT_PLAN_CAPSULE)) {
            binding.point = get_component_point_plan(plan_obj);
            if (!binding.point) {
                cleanup();
                return nullptr;
            }
            fields = fields ? fields : binding.point->fields;
            if (binding.point->fields != fields) {
                cleanup();
                PyErr_SetString(
                    PyExc_ValueError,
                    "native forward plans must reference the same Meep fields");
                return nullptr;
            }
            require_complex = !fields->is_real;
            width = static_cast<npy_intp>(
                component_point_plan_history_width(binding.point));
        } else if (PyCapsule_IsValid(plan_obj, EIGENMODE_OVERLAP_PLAN_CAPSULE)) {
            binding.overlap = get_eigenmode_overlap_plan(plan_obj);
            if (!binding.overlap || binding.overlap->component_plans.empty()) {
                cleanup();
                return nullptr;
            }
            meep::fields *overlap_fields = binding.overlap->component_plans[0]->fields;
            fields = fields ? fields : overlap_fields;
            if (overlap_fields != fields) {
                cleanup();
                PyErr_SetString(
                    PyExc_ValueError,
                    "native forward plans must reference the same Meep fields");
                return nullptr;
            }
            require_complex = true;
            width = 2;
        } else {
            cleanup();
            PyErr_SetString(
                PyExc_TypeError,
                "native forward monitor plan has an unsupported capsule type");
            return nullptr;
        }
        if (!validate_native_forward_history(
                binding.history, static_cast<npy_intp>(fine_step_count + 1), width,
                require_complex,
                "native forward monitor history has an incompatible dtype, shape, or layout")) {
            cleanup();
            return nullptr;
        }
        monitors.push_back(binding);
    }

    const npy_intp design_rows =
        static_cast<npy_intp>(fine_step_count / sampling_interval + 1);
    for (Py_ssize_t index = 0; index < design_count; ++index) {
        NativeDesignPlan *plan =
            get_native_design_plan(PySequence_Fast_GET_ITEM(design_plans, index));
        if (!plan) {
            cleanup();
            return nullptr;
        }
        PyObject *history_obj = PySequence_Fast_GET_ITEM(design_histories, index);
        if (!PyArray_Check(history_obj)) {
            cleanup();
            PyErr_SetString(PyExc_TypeError,
                            "native forward design history must be a NumPy array");
            return nullptr;
        }
        PyArrayObject *history = reinterpret_cast<PyArrayObject *>(history_obj);
        fields = fields ? fields : plan->fields;
        if (plan->fields != fields) {
            cleanup();
            PyErr_SetString(PyExc_ValueError,
                            "native forward plans must reference the same Meep fields");
            return nullptr;
        }
        if (!validate_native_forward_history(
                history, design_rows, static_cast<npy_intp>(plan->entries.size()),
                !fields->is_real,
                "native forward design history has an incompatible dtype, shape, or layout")) {
            cleanup();
            return nullptr;
        }
        designs.push_back({plan, history});
    }
    if (!fields) {
        cleanup();
        PyErr_SetString(PyExc_ValueError, "native forward segment has no field plans");
        return nullptr;
    }
    if (fields->t != start_fine_index) {
        cleanup();
        PyErr_SetString(PyExc_RuntimeError,
                        "native forward segment does not match the Meep time grid");
        return nullptr;
    }

    double extra_work_seconds = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - setup_work_start)
                                    .count();
    const long long initial_meep_step = fields->t;
    const int processor_count = meep::count_processors();
    try {
        for (long long offset = 0; offset < sample_count; ++offset) {
            const bool interrupted =
                processor_count == 1 ? PyErr_CheckSignals() < 0
                                     : (offset % NATIVE_SIGNAL_CHECK_INTERVAL == 0 &&
                                        check_native_adjoint_signals());
            if (interrupted) {
                cleanup();
                return nullptr;
            }
            const auto work_start = std::chrono::steady_clock::now();
            const long long fine_index = start_fine_index + offset;
            *reinterpret_cast<double *>(PyArray_BYTES(monitor_times) +
                                        fine_index * PyArray_STRIDE(monitor_times, 0)) =
                fields->time();
            for (const MonitorBinding &monitor : monitors) {
                if (monitor.point) {
                    const size_t history_width =
                        component_point_plan_history_width(monitor.point);
                    for (size_t point = 0; point < history_width; ++point) {
                        write_native_history_value(
                            monitor.history, static_cast<npy_intp>(fine_index),
                            static_cast<npy_intp>(point),
                            sample_component_point_plan_local(
                                monitor.point, component_point_plan_history_index(
                                                   monitor.point, point)));
                    }
                } else {
                    const auto overlap =
                        sample_eigenmode_overlap_plan_local(monitor.overlap);
                    write_native_history_value(monitor.history,
                                               static_cast<npy_intp>(fine_index), 0,
                                               overlap[0]);
                    write_native_history_value(monitor.history,
                                               static_cast<npy_intp>(fine_index), 1,
                                               overlap[1]);
                }
            }
            if (fine_index % sampling_interval == 0) {
                const npy_intp design_row =
                    static_cast<npy_intp>(fine_index / sampling_interval);
                for (const DesignBinding &design : designs) {
                    for (size_t point = 0; point < design.plan->entries.size();
                         ++point) {
                        write_native_history_value(
                            design.history, design_row, static_cast<npy_intp>(point),
                            fields->is_real
                                ? std::complex<double>(
                                      native_design_forward_real_field_value(
                                          design.plan, point),
                                      0.0)
                                : native_design_forward_field_value(design.plan,
                                                                    point));
                    }
                }
            }
            extra_work_seconds += std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - work_start)
                                      .count();

            if (offset + 1 < sample_count) {
                fields->step();
                if (PyErr_Occurred()) {
                    cleanup();
                    return nullptr;
                }
            }
        }
        const bool interrupted = processor_count == 1 ? PyErr_CheckSignals() < 0
                                                      : check_native_adjoint_signals();
        if (interrupted) {
            cleanup();
            return nullptr;
        }
        if (fields->t - initial_meep_step != sample_count - 1) {
            throw std::runtime_error(
                "native forward segment advanced an unexpected number of Meep steps");
        }
    } catch (const std::bad_alloc &) {
        cleanup();
        return PyErr_NoMemory();
    } catch (const std::exception &exc) {
        cleanup();
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    } catch (...) {
        cleanup();
        PyErr_SetString(PyExc_RuntimeError, "unknown error in native forward segment");
        return nullptr;
    }

    cleanup();
    return PyFloat_FromDouble(extra_work_seconds);
}

static PyObject *run_native_design_adjoint_segment(PyObject *, PyObject *args) {
    PyObject *plans_obj = nullptr;
    PyObject *histories_obj = nullptr;
    PyObject *weights_obj = nullptr;
    PyObject *accumulator_obj = nullptr;
    int sampling_interval = 0;
    int reconstruction_first_offset = 0;
    long long fine_step_count = 0;
    long long start_fine_index = 0;
    long long sample_count = 0;
    double dt = 0.0;
    int midpoint = 0;
    PyObject *gradient_offsets_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOOOiiLLLdi|O:run_native_design_adjoint_segment",
                          &plans_obj, &histories_obj, &weights_obj, &accumulator_obj,
                          &sampling_interval, &reconstruction_first_offset,
                          &fine_step_count, &start_fine_index, &sample_count, &dt,
                          &midpoint, &gradient_offsets_obj)) {
        return nullptr;
    }
    if (sampling_interval < 1 || fine_step_count < 1 ||
        fine_step_count % sampling_interval != 0 || start_fine_index < 0 ||
        start_fine_index > fine_step_count || sample_count < 1 ||
        sample_count > start_fine_index + 1 || !std::isfinite(dt) || dt <= 0.0) {
        PyErr_SetString(PyExc_ValueError,
                        "native adjoint segment received an invalid time grid");
        return nullptr;
    }
    const auto setup_work_start = std::chrono::steady_clock::now();

    PyObject *plans =
        PySequence_Fast(plans_obj, "native adjoint plans must be a sequence");
    if (!plans) {
        return nullptr;
    }
    PyObject *histories =
        PySequence_Fast(histories_obj, "native adjoint histories must be a sequence");
    if (!histories) {
        Py_DECREF(plans);
        return nullptr;
    }
    PyArrayObject *weights = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(weights_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY));
    if (!weights) {
        Py_DECREF(histories);
        Py_DECREF(plans);
        return nullptr;
    }
    PyObject *gradient_offsets =
        gradient_offsets_obj == Py_None
            ? nullptr
            : PySequence_Fast(gradient_offsets_obj,
                              "gradient offsets must be a sequence");
    if (gradient_offsets_obj != Py_None && !gradient_offsets) {
        Py_DECREF(weights);
        Py_DECREF(histories);
        Py_DECREF(plans);
        return nullptr;
    }
    auto cleanup = [&]() {
        Py_XDECREF(gradient_offsets);
        Py_DECREF(weights);
        Py_DECREF(histories);
        Py_DECREF(plans);
    };

    const Py_ssize_t component_count = PySequence_Fast_GET_SIZE(plans);
    if (component_count < 1 || PySequence_Fast_GET_SIZE(histories) != component_count ||
        (gradient_offsets &&
         PySequence_Fast_GET_SIZE(gradient_offsets) != component_count)) {
        cleanup();
        PyErr_SetString(
            PyExc_ValueError,
            "native adjoint plans and histories must have the same nonzero length");
        return nullptr;
    }
    if (PyArray_NDIM(weights) != 2 || PyArray_DIM(weights, 0) != sampling_interval ||
        PyArray_DIM(weights, 1) < 1) {
        cleanup();
        PyErr_SetString(PyExc_ValueError,
                        "native adjoint reconstruction weights have the wrong shape");
        return nullptr;
    }
    if (!PyArray_Check(accumulator_obj)) {
        cleanup();
        PyErr_SetString(PyExc_TypeError,
                        "native adjoint accumulator must be a NumPy array");
        return nullptr;
    }
    PyArrayObject *accumulator = reinterpret_cast<PyArrayObject *>(accumulator_obj);
    const int accumulator_type = PyArray_TYPE(accumulator);
    const bool use_real = accumulator_type == NPY_FLOAT64;
    if ((!use_real && accumulator_type != NPY_COMPLEX128) ||
        PyArray_NDIM(accumulator) != 1 || !PyArray_ISCARRAY(accumulator) ||
        !PyArray_ISNOTSWAPPED(accumulator)) {
        cleanup();
        PyErr_SetString(PyExc_ValueError,
                        "native adjoint accumulator must be a writable C-contiguous "
                        "float64 or complex128 vector");
        return nullptr;
    }

    std::vector<NativeDesignPlan *> native_plans;
    std::vector<PyArrayObject *> native_histories;
    std::vector<size_t> native_offsets;
    try {
        native_plans.reserve(static_cast<size_t>(component_count));
        native_histories.reserve(static_cast<size_t>(component_count));
        native_offsets.reserve(static_cast<size_t>(component_count));
    } catch (const std::bad_alloc &) {
        cleanup();
        return PyErr_NoMemory();
    }
    meep::fields *fields = nullptr;
    const npy_intp expected_history_rows =
        static_cast<npy_intp>(fine_step_count / sampling_interval + 1);
    for (Py_ssize_t component_index = 0; component_index < component_count;
         ++component_index) {
        NativeDesignPlan *plan =
            get_native_design_plan(PySequence_Fast_GET_ITEM(plans, component_index));
        if (!plan) {
            cleanup();
            return nullptr;
        }
        PyObject *history_obj = PySequence_Fast_GET_ITEM(histories, component_index);
        if (!PyArray_Check(history_obj)) {
            cleanup();
            PyErr_SetString(PyExc_TypeError,
                            "native adjoint history must be a NumPy array");
            return nullptr;
        }
        PyArrayObject *history = reinterpret_cast<PyArrayObject *>(history_obj);
        const int history_type = PyArray_TYPE(history);
        if (!native_history_dtype_supported(history_type) ||
            PyArray_NDIM(history) != 2 ||
            PyArray_DIM(history, 0) != expected_history_rows ||
            PyArray_DIM(history, 1) != static_cast<npy_intp>(plan->entries.size()) ||
            !PyArray_ISCARRAY_RO(history) || !PyArray_ISNOTSWAPPED(history)) {
            cleanup();
            PyErr_SetString(
                PyExc_ValueError,
                "native adjoint history has an incompatible dtype, shape, or layout");
            return nullptr;
        }
        if (use_real &&
            (!plan->fields->is_real || native_history_dtype_complex(history_type))) {
            cleanup();
            PyErr_SetString(
                PyExc_ValueError,
                "real native adjoint accumulation requires real fields and histories");
            return nullptr;
        }
        const Py_ssize_t gradient_offset =
            gradient_offsets ? PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(
                                   gradient_offsets, component_index))
                             : 0;
        if (PyErr_Occurred()) {
            cleanup();
            return nullptr;
        }
        const size_t region_size = plan->nx * plan->ny * plan->nz;
        const size_t total_size = static_cast<size_t>(PyArray_DIM(accumulator, 0));
        if (gradient_offset < 0 || static_cast<size_t>(gradient_offset) > total_size ||
            region_size > total_size - static_cast<size_t>(gradient_offset) ||
            (!gradient_offsets && total_size != region_size)) {
            cleanup();
            PyErr_SetString(
                PyExc_ValueError,
                "native adjoint accumulator has the wrong design-grid size");
            return nullptr;
        }
        const size_t physical_width = plan->entries.size() * (use_real ? 1u : 2u);
        if (physical_width > static_cast<size_t>(std::numeric_limits<int>::max())) {
            cleanup();
            PyErr_SetString(PyExc_OverflowError,
                            "native adjoint history is too wide for CBLAS");
            return nullptr;
        }
        if (fields && fields != plan->fields) {
            cleanup();
            PyErr_SetString(
                PyExc_ValueError,
                "native adjoint plans must reference the same Meep fields object");
            return nullptr;
        }
        fields = plan->fields;
        native_plans.push_back(plan);
        native_histories.push_back(history);
        native_offsets.push_back(static_cast<size_t>(gradient_offset));
    }

    if (PyArray_DIM(weights, 1) > std::numeric_limits<int>::max() ||
        sampling_interval > std::numeric_limits<int>::max()) {
        cleanup();
        PyErr_SetString(PyExc_OverflowError,
                        "native adjoint reconstruction matrix is too large for CBLAS");
        return nullptr;
    }

    NativeDerivativeStencils stencils;
    try {
        stencils = make_native_derivative_stencils(
            reinterpret_cast<const double *>(PyArray_DATA(weights)), sampling_interval,
            static_cast<size_t>(PyArray_DIM(weights, 1)), reconstruction_first_offset,
            dt);
    } catch (const std::bad_alloc &) {
        cleanup();
        return PyErr_NoMemory();
    } catch (const std::exception &exc) {
        cleanup();
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }

    std::vector<std::vector<double>> derivative_blocks;
    std::vector<long long> cached_coarse_indices;
    try {
        derivative_blocks.resize(static_cast<size_t>(component_count));
        cached_coarse_indices.assign(static_cast<size_t>(component_count),
                                     std::numeric_limits<long long>::min());
    } catch (const std::bad_alloc &) {
        cleanup();
        return PyErr_NoMemory();
    }
    double extra_work_seconds = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - setup_work_start)
                                    .count();
    const long long last_coarse_index = fine_step_count / sampling_interval;
    const long long expected_meep_step = fine_step_count - start_fine_index;
    const double dt_tolerance = 32.0 * std::numeric_limits<double>::epsilon() *
                                std::max({1.0, std::abs(dt), std::abs(fields->dt)});
    if (std::abs(fields->dt - dt) > dt_tolerance || fields->t != expected_meep_step) {
        cleanup();
        PyErr_SetString(PyExc_RuntimeError,
                        "native adjoint segment does not match the forward time grid");
        return nullptr;
    }
    const long long initial_meep_step = fields->t;
    const int processor_count = meep::count_processors();
    try {
        for (long long sample_index = 0; sample_index < sample_count; ++sample_index) {
            const bool interrupted =
                processor_count == 1
                    ? PyErr_CheckSignals() < 0
                    : (sample_index % NATIVE_SIGNAL_CHECK_INTERVAL == 0 &&
                       check_native_adjoint_signals());
            if (interrupted) {
                cleanup();
                return nullptr;
            }
            const auto work_start = std::chrono::steady_clock::now();
            const long long fine_index = start_fine_index - sample_index;
            const long long coarse_index = fine_index / sampling_interval;
            const int phase = static_cast<int>(fine_index % sampling_interval);
            for (Py_ssize_t component_index = 0; component_index < component_count;
                 ++component_index) {
                NativeDesignPlan *plan =
                    native_plans[static_cast<size_t>(component_index)];
                PyArrayObject *history =
                    native_histories[static_cast<size_t>(component_index)];
                std::vector<double> &block =
                    derivative_blocks[static_cast<size_t>(component_index)];
                if (cached_coarse_indices[static_cast<size_t>(component_index)] !=
                    coarse_index) {
                    fill_native_derivative_block(history, coarse_index,
                                                 last_coarse_index, stencils, !use_real,
                                                 block);
                    cached_coarse_indices[static_cast<size_t>(component_index)] =
                        coarse_index;
                }
                const size_t physical_width =
                    plan->entries.size() * (use_real ? 1u : 2u);
                if (physical_width == 0) {
                    continue;
                }
                const double *derivative =
                    block.data() + static_cast<size_t>(phase) * physical_width;
                if (use_real) {
                    accumulate_native_adjoint_real_row(
                        plan, derivative,
                        reinterpret_cast<double *>(PyArray_DATA(accumulator)) +
                            native_offsets[static_cast<size_t>(component_index)],
                        midpoint != 0);
                } else {
                    accumulate_native_adjoint_complex_row(
                        plan, derivative, true,
                        reinterpret_cast<npy_cdouble *>(PyArray_DATA(accumulator)) +
                            native_offsets[static_cast<size_t>(component_index)],
                        midpoint != 0);
                }
            }
            extra_work_seconds += std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - work_start)
                                      .count();

            if (sample_index + 1 < sample_count) {
                fields->step();
                if (PyErr_Occurred()) {
                    cleanup();
                    return nullptr;
                }
            }
        }
        const bool interrupted = processor_count == 1 ? PyErr_CheckSignals() < 0
                                                      : check_native_adjoint_signals();
        if (interrupted) {
            cleanup();
            return nullptr;
        }
        if (fields->t - initial_meep_step != sample_count - 1) {
            throw std::runtime_error(
                "native adjoint segment advanced an unexpected number of Meep steps");
        }
    } catch (const std::bad_alloc &) {
        cleanup();
        return PyErr_NoMemory();
    } catch (const std::exception &exc) {
        cleanup();
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    } catch (...) {
        cleanup();
        PyErr_SetString(PyExc_RuntimeError, "unknown error in native adjoint segment");
        return nullptr;
    }

    cleanup();
    return PyFloat_FromDouble(extra_work_seconds);
}

static PyObject *sample_component_point_plan_allreduced(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &plan_obj)) {
        return nullptr;
    }
    ComponentPointPlan *plan = get_component_point_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    try {
        for (size_t point_idx = 0; point_idx < plan->points.size(); ++point_idx) {
            plan->local[point_idx] = sample_component_point_plan_local(plan, point_idx);
        }
        meep::sum_to_all(plan->local.data(), plan->reduced.data(),
                         static_cast<int>(plan->points.size()));
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in planned monitor sampling");
        return nullptr;
    }

    npy_intp dims[1] = {static_cast<npy_intp>(plan->points.size())};
    PyObject *arr_obj = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }
    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj)));
    for (size_t point_idx = 0; point_idx < plan->points.size(); ++point_idx) {
        set_npy_complex(data[point_idx], plan->reduced[point_idx]);
    }
    return arr_obj;
}

static PyObject *component_point_plan_indexed_stencil(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &plan_obj)) {
        return nullptr;
    }
    ComponentPointPlan *plan = get_component_point_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    if (native_mirror_axes(plan->fields) && meep::count_processors() > 1) {
        const double count = static_cast<double>(plan->points.size());
        if (meep::max_to_all(count) != -meep::max_to_all(-count)) {
            throw std::invalid_argument(
                "Mirror indexed monitors require the same point count on every rank");
        }
        // Per-point source routing requires identical channel identities, not
        // just equal counts, in every process of the active Meep subgroup.
        std::vector<double> reference = plan->monitor_identity;
        MeepGroupCommunicator group = active_meep_group_communicator();
        try {
            for (size_t offset = 0; offset < reference.size();) {
                const int block = static_cast<int>(
                    std::min(reference.size() - offset, static_cast<size_t>(INT_MAX)));
                require_mpi_success(MPI_Bcast(reference.data() + offset, block,
                                              MPI_DOUBLE, 0, group.comm),
                                    "Mirror monitor identity");
                offset += block;
            }
        } catch (...) {
            if (group.owned)
                free_owned_communicator(group.comm);
            throw;
        }
        if (group.owned)
            free_owned_communicator(group.comm);
        if (meep::sum_to_all(static_cast<int>(reference != plan->monitor_identity))) {
            throw std::invalid_argument(
                "Mirror indexed monitors require the same monitor component and coordinates on every rank");
        }
    }
    std::vector<npy_intp> offsets(plan->points.size() + 1, 0);
    std::vector<npy_int64> components;
    std::vector<npy_int64> chunk_indices;
    std::vector<npy_intp> local_indices;
    std::vector<std::complex<double>> amplitudes;
    try {
        for (size_t point_idx = 0; point_idx < plan->points.size(); ++point_idx) {
            using SourceNode = std::tuple<int, int, ptrdiff_t>;
            std::map<SourceNode, std::vector<std::complex<double>>> indexed_entries;
            for (const PointSampleEntry &entry : plan->points[point_idx]) {
                meep::fields_chunk *chunk = plan->fields->chunks[entry.chunk_idx];
                const ptrdiff_t index = chunk->gv.index(entry.component, entry.loc);
                const meep::direction direction =
                    meep::component_direction(entry.component);
                const bool constrained_axis_component =
                    plan->cylindrical && entry.loc.r() == 0 &&
                    ((std::abs(plan->fields->m) < 1e-12 &&
                      (direction == meep::R || direction == meep::P)) ||
                     (std::abs(std::abs(plan->fields->m) - 1.0) < 1e-12 &&
                      direction == meep::Z));
                if (constrained_axis_component) {
                    continue;
                }
                const double volume =
                    native_mirror_measure(plan->fields, entry.loc) *
                    chunk->gv.dV(entry.component, index).full_volume();
                if (!std::isfinite(volume) || volume <= 0.0) {
                    throw std::runtime_error("point-monitor indexed stencil has a "
                                             "non-positive Yee-cell volume");
                }
                const std::complex<double> amplitude = entry.weight / volume;
                std::vector<SourceNode> images;
                const int image_count = native_mirror_axes(plan->fields)
                                            ? plan->fields->S.multiplicity()
                                            : 1;
                for (int n = 0; n < image_count; ++n) {
                    const auto location = plan->fields->S.transform(entry.loc, n);
                    if (n && native_mirror_measure(plan->fields, location) > 0.0) {
                        continue;
                    }
                    const auto component =
                        plan->fields->S.transform(entry.component, n);
                    for (int ci = 0; ci < plan->fields->num_chunks; ++ci) {
                        auto *owner = plan->fields->chunks[ci];
                        if (!owner || !owner->gv.owns(location))
                            continue;
                        const SourceNode key(static_cast<int>(component), ci,
                                             owner->gv.index(component, location));
                        if (std::find(images.begin(), images.end(), key) ==
                            images.end()) {
                            images.push_back(key);
                            auto &values = indexed_entries[key];
                            if (values.empty())
                                values.resize(1, 0.0);
                            // Meep evolves the redundant negative half-cell too;
                            // its source must have the same reflected value.
                            values[0] += amplitude * plan->fields->S.phase_shift(
                                                         entry.component, n);
                        }
                        break;
                    }
                }
            }
            if (native_mirror_axes(plan->fields)) {
                route_native_source_nodes(plan->fields, indexed_entries, 1);
            }
            for (const auto &entry : indexed_entries) {
                if (std::abs(entry.second[0]) <= 1e-15) {
                    continue;
                }
                components.push_back(static_cast<npy_int64>(std::get<0>(entry.first)));
                chunk_indices.push_back(
                    static_cast<npy_int64>(std::get<1>(entry.first)));
                local_indices.push_back(
                    static_cast<npy_intp>(std::get<2>(entry.first)));
                amplitudes.push_back(entry.second[0]);
            }
            offsets[point_idx + 1] = static_cast<npy_intp>(amplitudes.size());
        }
    } catch (const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "unknown error while constructing point-monitor indexed stencil");
        return nullptr;
    }

    npy_intp offset_dims[1] = {
        static_cast<npy_intp>(offsets.size()),
    };
    npy_intp entry_dims[1] = {
        static_cast<npy_intp>(amplitudes.size()),
    };
    PyObject *offsets_obj = PyArray_SimpleNew(1, offset_dims, NPY_INTP);
    PyObject *components_obj = PyArray_SimpleNew(1, entry_dims, NPY_INT64);
    PyObject *chunks_obj = PyArray_SimpleNew(1, entry_dims, NPY_INT64);
    PyObject *indices_obj = PyArray_SimpleNew(1, entry_dims, NPY_INTP);
    PyObject *amplitudes_obj = PyArray_SimpleNew(1, entry_dims, NPY_COMPLEX128);
    if (!offsets_obj || !components_obj || !chunks_obj || !indices_obj ||
        !amplitudes_obj) {
        Py_XDECREF(offsets_obj);
        Py_XDECREF(components_obj);
        Py_XDECREF(chunks_obj);
        Py_XDECREF(indices_obj);
        Py_XDECREF(amplitudes_obj);
        return nullptr;
    }

    std::copy(offsets.begin(), offsets.end(),
              reinterpret_cast<npy_intp *>(
                  PyArray_DATA(reinterpret_cast<PyArrayObject *>(offsets_obj))));
    std::copy(components.begin(), components.end(),
              reinterpret_cast<npy_int64 *>(
                  PyArray_DATA(reinterpret_cast<PyArrayObject *>(components_obj))));
    std::copy(chunk_indices.begin(), chunk_indices.end(),
              reinterpret_cast<npy_int64 *>(
                  PyArray_DATA(reinterpret_cast<PyArrayObject *>(chunks_obj))));
    std::copy(local_indices.begin(), local_indices.end(),
              reinterpret_cast<npy_intp *>(
                  PyArray_DATA(reinterpret_cast<PyArrayObject *>(indices_obj))));
    npy_cdouble *amplitude_data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(amplitudes_obj)));
    for (size_t index = 0; index < amplitudes.size(); ++index) {
        set_npy_complex(amplitude_data[index], amplitudes[index]);
    }

    PyObject *result = PyTuple_New(5);
    if (!result) {
        Py_DECREF(offsets_obj);
        Py_DECREF(components_obj);
        Py_DECREF(chunks_obj);
        Py_DECREF(indices_obj);
        Py_DECREF(amplitudes_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, offsets_obj);
    PyTuple_SET_ITEM(result, 1, components_obj);
    PyTuple_SET_ITEM(result, 2, chunks_obj);
    PyTuple_SET_ITEM(result, 3, indices_obj);
    PyTuple_SET_ITEM(result, 4, amplitudes_obj);
    return result;
}

static PyObject *configure_component_point_plan_history(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *indices_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &plan_obj, &indices_obj)) {
        return nullptr;
    }
    ComponentPointPlan *plan = get_component_point_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    std::vector<size_t> indices;
    try {
        if (read_index_sequence(
                indices_obj, indices, plan->points.size(),
                "point-monitor history indices must be a 1D integer array") < 0) {
            return nullptr;
        }
        std::vector<size_t> sorted(indices);
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            PyErr_SetString(
                PyExc_ValueError,
                "point-monitor history indices must not contain duplicates");
            return nullptr;
        }
        plan->history_indices = std::move(indices);
        plan->history_sampling_configured = true;
    } catch (const std::bad_alloc &) {
        return PyErr_NoMemory();
    } catch (const std::exception &exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error while configuring point-monitor history");
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject *populate_sourcedata(PyObject *, PyObject *args) {
    unsigned long long sourcedata_addr = 0;
    int component_int = static_cast<int>(meep::Ez);
    int chunk_idx = -1;
    Py_ssize_t local_index = 0;
    if (!PyArg_ParseTuple(args, "Kiin", &sourcedata_addr, &component_int, &chunk_idx,
                          &local_index)) {
        return nullptr;
    }
    if (sourcedata_addr == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "sourcedata pointer address must be non-zero");
        return nullptr;
    }
    if (!require_dynamic_field_component(component_int)) {
        return nullptr;
    }
    if (chunk_idx < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "sourcedata chunk index must be non-negative");
        return nullptr;
    }
    if (local_index < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "sourcedata local index must be non-negative");
        return nullptr;
    }

    meep::sourcedata *data =
        reinterpret_cast<meep::sourcedata *>(static_cast<uintptr_t>(sourcedata_addr));
    data->near_fd_comp = static_cast<meep::component>(component_int);
    data->fc_idx = chunk_idx;
    data->idx_arr.assign(1, static_cast<ptrdiff_t>(local_index));
    data->amp_arr.clear();
    Py_RETURN_NONE;
}

static PyObject *merge_sourcedata(PyObject *, PyObject *args) {
    unsigned long long destination_addr = 0;
    PyObject *source_addresses_obj = nullptr;
    if (!PyArg_ParseTuple(args, "KO", &destination_addr, &source_addresses_obj)) {
        return nullptr;
    }
    if (destination_addr == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "destination sourcedata pointer address must be non-zero");
        return nullptr;
    }

    PyObject *source_addresses = PySequence_Fast(
        source_addresses_obj, "source sourcedata addresses must be a sequence");
    if (!source_addresses) {
        return nullptr;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(source_addresses);
    if (count <= 0) {
        Py_DECREF(source_addresses);
        PyErr_SetString(PyExc_ValueError,
                        "at least one source sourcedata object is required");
        return nullptr;
    }

    meep::sourcedata *destination =
        reinterpret_cast<meep::sourcedata *>(static_cast<uintptr_t>(destination_addr));
    destination->idx_arr.clear();
    destination->amp_arr.clear();
    try {
        destination->idx_arr.reserve(static_cast<size_t>(count));
    } catch (...) {
        Py_DECREF(source_addresses);
        throw;
    }

    meep::component component = meep::NO_COMPONENT;
    int chunk_index = -1;
    for (Py_ssize_t index = 0; index < count; ++index) {
        const unsigned long long source_addr = PyLong_AsUnsignedLongLong(
            PySequence_Fast_GET_ITEM(source_addresses, index));
        if (PyErr_Occurred()) {
            Py_DECREF(source_addresses);
            return nullptr;
        }
        if (source_addr == 0) {
            Py_DECREF(source_addresses);
            PyErr_SetString(PyExc_ValueError,
                            "source sourcedata pointer address must be non-zero");
            return nullptr;
        }
        const meep::sourcedata *source = reinterpret_cast<const meep::sourcedata *>(
            static_cast<uintptr_t>(source_addr));
        if (source->idx_arr.size() != 1) {
            Py_DECREF(source_addresses);
            PyErr_SetString(PyExc_ValueError,
                            "source sourcedata must contain exactly one field index");
            return nullptr;
        }
        if (index == 0) {
            component = source->near_fd_comp;
            chunk_index = source->fc_idx;
            destination->near_fd_comp = component;
            destination->fc_idx = chunk_index;
        } else if (source->near_fd_comp != component || source->fc_idx != chunk_index) {
            Py_DECREF(source_addresses);
            PyErr_SetString(
                PyExc_ValueError,
                "merged sourcedata entries must share a component and chunk");
            return nullptr;
        }
        destination->idx_arr.push_back(source->idx_arr[0]);
    }
    Py_DECREF(source_addresses);
    Py_RETURN_NONE;
}

static PyObject *sample_component_point_plan_local_into(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *destination_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &plan_obj, &destination_obj)) {
        return nullptr;
    }
    ComponentPointPlan *plan = get_component_point_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    PyArrayObject *destination = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(destination_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY));
    if (!destination) {
        return nullptr;
    }
    const size_t history_width = component_point_plan_history_width(plan);
    const bool shape_ok =
        PyArray_NDIM(destination) == 1 &&
        PyArray_DIM(destination, 0) == static_cast<npy_intp>(history_width);
    if (!shape_ok) {
        PyArray_DiscardWritebackIfCopy(destination);
        Py_DECREF(destination);
        PyErr_SetString(
            PyExc_ValueError,
            "point-monitor destination must be a 1D complex128 array matching the point count");
        return nullptr;
    }
    npy_cdouble *destination_data =
        reinterpret_cast<npy_cdouble *>(PyArray_DATA(destination));

    try {
        for (size_t point_idx = 0; point_idx < history_width; ++point_idx) {
            set_npy_complex(
                destination_data[point_idx],
                sample_component_point_plan_local(
                    plan, component_point_plan_history_index(plan, point_idx)));
        }
    } catch (const std::exception &e) {
        PyArray_DiscardWritebackIfCopy(destination);
        Py_DECREF(destination);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyArray_DiscardWritebackIfCopy(destination);
        Py_DECREF(destination);
        PyErr_SetString(PyExc_RuntimeError, "unknown error in local monitor sampling");
        return nullptr;
    }

    if (PyArray_ResolveWritebackIfCopy(destination) < 0) {
        Py_DECREF(destination);
        return nullptr;
    }
    Py_DECREF(destination);
    Py_RETURN_NONE;
}

static PyObject *sample_component_point_plan_local_real_into(PyObject *,
                                                             PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *destination_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &plan_obj, &destination_obj)) {
        return nullptr;
    }
    ComponentPointPlan *plan = get_component_point_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    if (!plan->fields->is_real) {
        PyErr_SetString(PyExc_RuntimeError,
                        "real point-monitor sampling requires real Meep fields");
        return nullptr;
    }
    if (!PyArray_Check(destination_obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "point-monitor destination must be a NumPy array");
        return nullptr;
    }

    PyArrayObject *destination = reinterpret_cast<PyArrayObject *>(destination_obj);
    const size_t history_width = component_point_plan_history_width(plan);
    const bool shape_ok =
        PyArray_NDIM(destination) == 1 &&
        PyArray_DIM(destination, 0) == static_cast<npy_intp>(history_width);
    if (!shape_ok) {
        PyErr_SetString(PyExc_ValueError,
                        "point-monitor destination must match the point count");
        return nullptr;
    }
    if (PyArray_TYPE(destination) != NPY_DOUBLE || !PyArray_ISCARRAY(destination) ||
        !PyArray_ISNOTSWAPPED(destination)) {
        PyErr_SetString(PyExc_ValueError,
                        "real point-monitor sampling requires a writable C-contiguous "
                        "native-endian float64 destination");
        return nullptr;
    }

    double *destination_data = reinterpret_cast<double *>(PyArray_DATA(destination));
    try {
        for (size_t point_idx = 0; point_idx < history_width; ++point_idx) {
            destination_data[point_idx] =
                sample_component_point_plan_local(
                    plan, component_point_plan_history_index(plan, point_idx))
                    .real();
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in real local monitor sampling");
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject *create_eigenmode_overlap_plan(PyObject *, PyObject *args) {
    PyObject *component_plans_obj = nullptr;
    PyObject *weights_obj = nullptr;
    PyObject *output_channels_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &component_plans_obj, &weights_obj,
                          &output_channels_obj)) {
        return nullptr;
    }

    PyObject *component_plans =
        PySequence_Fast(component_plans_obj, "component_plans must be a sequence");
    if (!component_plans) {
        return nullptr;
    }
    PyObject *weights = PySequence_Fast(weights_obj, "weights must be a sequence");
    if (!weights) {
        Py_DECREF(component_plans);
        return nullptr;
    }
    PyArrayObject *output_channels = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(output_channels_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    if (!output_channels) {
        Py_DECREF(weights);
        Py_DECREF(component_plans);
        return nullptr;
    }

    const Py_ssize_t count = PySequence_Fast_GET_SIZE(component_plans);
    if (count <= 0 || PySequence_Fast_GET_SIZE(weights) != count ||
        PyArray_NDIM(output_channels) != 1 ||
        PyArray_DIM(output_channels, 0) != count) {
        Py_DECREF(output_channels);
        Py_DECREF(weights);
        Py_DECREF(component_plans);
        PyErr_SetString(
            PyExc_ValueError,
            "component plans, weights, and output channels must have the same nonzero length");
        return nullptr;
    }

    EigenmodeOverlapPlan *plan = new (std::nothrow) EigenmodeOverlapPlan();
    if (!plan) {
        Py_DECREF(output_channels);
        Py_DECREF(weights);
        Py_DECREF(component_plans);
        return PyErr_NoMemory();
    }
    try {
        plan->component_plan_capsules.reserve(static_cast<size_t>(count));
        plan->component_plans.reserve(static_cast<size_t>(count));
        plan->weights.reserve(static_cast<size_t>(count));
        plan->output_channels.reserve(static_cast<size_t>(count));
    } catch (const std::bad_alloc &) {
        delete_eigenmode_overlap_plan(plan);
        Py_DECREF(output_channels);
        Py_DECREF(weights);
        Py_DECREF(component_plans);
        return PyErr_NoMemory();
    }

    meep::fields *fields = nullptr;
    const npy_intp *channel_data =
        reinterpret_cast<const npy_intp *>(PyArray_DATA(output_channels));
    for (Py_ssize_t index = 0; index < count; ++index) {
        PyObject *component_plan_obj = PySequence_Fast_GET_ITEM(component_plans, index);
        ComponentPointPlan *component_plan =
            get_component_point_plan(component_plan_obj);
        if (!component_plan) {
            delete_eigenmode_overlap_plan(plan);
            Py_DECREF(output_channels);
            Py_DECREF(weights);
            Py_DECREF(component_plans);
            return nullptr;
        }
        if (fields && component_plan->fields != fields) {
            delete_eigenmode_overlap_plan(plan);
            Py_DECREF(output_channels);
            Py_DECREF(weights);
            Py_DECREF(component_plans);
            PyErr_SetString(
                PyExc_ValueError,
                "all eigenmode overlap component plans must use the same Meep fields");
            return nullptr;
        }
        fields = component_plan->fields;
        if (channel_data[index] < 0 || channel_data[index] > 1) {
            delete_eigenmode_overlap_plan(plan);
            Py_DECREF(output_channels);
            Py_DECREF(weights);
            Py_DECREF(component_plans);
            PyErr_SetString(PyExc_ValueError,
                            "eigenmode overlap output channels must be zero or one");
            return nullptr;
        }

        PyArrayObject *component_weights = reinterpret_cast<PyArrayObject *>(
            PyArray_FROM_OTF(PySequence_Fast_GET_ITEM(weights, index), NPY_COMPLEX128,
                             NPY_ARRAY_IN_ARRAY));
        if (!component_weights) {
            delete_eigenmode_overlap_plan(plan);
            Py_DECREF(output_channels);
            Py_DECREF(weights);
            Py_DECREF(component_plans);
            return nullptr;
        }
        const bool weight_shape_ok =
            PyArray_NDIM(component_weights) == 1 &&
            PyArray_DIM(component_weights, 0) ==
                static_cast<npy_intp>(component_plan->points.size());
        if (!weight_shape_ok) {
            Py_DECREF(component_weights);
            delete_eigenmode_overlap_plan(plan);
            Py_DECREF(output_channels);
            Py_DECREF(weights);
            Py_DECREF(component_plans);
            PyErr_SetString(
                PyExc_ValueError,
                "each eigenmode overlap weight array must match its point-plan size");
            return nullptr;
        }
        const npy_cdouble *component_weight_data =
            reinterpret_cast<const npy_cdouble *>(PyArray_DATA(component_weights));
        std::vector<std::complex<double>> copied_weights;
        try {
            copied_weights.resize(component_plan->points.size());
            for (size_t point_index = 0; point_index < copied_weights.size();
                 ++point_index) {
                copied_weights[point_index] =
                    npy_to_complex(component_weight_data[point_index]);
            }
        } catch (const std::bad_alloc &) {
            Py_DECREF(component_weights);
            delete_eigenmode_overlap_plan(plan);
            Py_DECREF(output_channels);
            Py_DECREF(weights);
            Py_DECREF(component_plans);
            return PyErr_NoMemory();
        }
        Py_DECREF(component_weights);

        Py_INCREF(component_plan_obj);
        bool capsule_stored = false;
        try {
            plan->component_plan_capsules.push_back(component_plan_obj);
            capsule_stored = true;
            plan->component_plans.push_back(component_plan);
            plan->weights.push_back(std::move(copied_weights));
            plan->output_channels.push_back(static_cast<size_t>(channel_data[index]));
        } catch (const std::bad_alloc &) {
            if (!capsule_stored) {
                Py_DECREF(component_plan_obj);
            }
            delete_eigenmode_overlap_plan(plan);
            Py_DECREF(output_channels);
            Py_DECREF(weights);
            Py_DECREF(component_plans);
            return PyErr_NoMemory();
        }
    }

    Py_DECREF(output_channels);
    Py_DECREF(weights);
    Py_DECREF(component_plans);
    PyObject *capsule = PyCapsule_New(plan, EIGENMODE_OVERLAP_PLAN_CAPSULE,
                                      eigenmode_overlap_plan_destructor);
    if (!capsule) {
        delete_eigenmode_overlap_plan(plan);
        return nullptr;
    }
    return capsule;
}

static std::array<std::complex<double>, 2>
sample_eigenmode_overlap_plan_local(const EigenmodeOverlapPlan *plan) {
    std::array<std::complex<double>, 2> overlap = {
        std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0),
    };
    for (size_t component_index = 0; component_index < plan->component_plans.size();
         ++component_index) {
        ComponentPointPlan *component_plan = plan->component_plans[component_index];
        const std::vector<std::complex<double>> &component_weights =
            plan->weights[component_index];
        std::complex<double> component_overlap(0.0, 0.0);
        for (size_t point_index = 0; point_index < component_weights.size();
             ++point_index) {
            component_overlap +=
                component_weights[point_index] *
                sample_component_point_plan_local(component_plan, point_index);
        }
        overlap[plan->output_channels[component_index]] += component_overlap;
    }
    return overlap;
}

static PyObject *sample_eigenmode_overlap_plan_local_into(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *destination_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &plan_obj, &destination_obj)) {
        return nullptr;
    }
    EigenmodeOverlapPlan *plan = get_eigenmode_overlap_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    if (!PyArray_Check(destination_obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "eigenmode overlap destination must be a NumPy array");
        return nullptr;
    }
    PyArrayObject *destination = reinterpret_cast<PyArrayObject *>(destination_obj);
    if (PyArray_NDIM(destination) != 1 || PyArray_DIM(destination, 0) != 2 ||
        PyArray_TYPE(destination) != NPY_COMPLEX128 || !PyArray_ISCARRAY(destination) ||
        !PyArray_ISNOTSWAPPED(destination)) {
        PyErr_SetString(PyExc_ValueError,
                        "eigenmode overlap destination must be a writable C-contiguous "
                        "native-endian complex128 array of length two");
        return nullptr;
    }

    std::array<std::complex<double>, 2> overlap;
    try {
        overlap = sample_eigenmode_overlap_plan_local(plan);
    } catch (const std::exception &exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in local eigenmode overlap sampling");
        return nullptr;
    }

    npy_cdouble *destination_data =
        reinterpret_cast<npy_cdouble *>(PyArray_DATA(destination));
    set_npy_complex(destination_data[0], overlap[0]);
    set_npy_complex(destination_data[1], overlap[1]);
    Py_RETURN_NONE;
}

static PyObject *component_grid_plan_local_complete_mask(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;

    if (!PyArg_ParseTuple(args, "O", &plan_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    npy_intp dims[2] = {
        static_cast<npy_intp>(plan->nx),
        static_cast<npy_intp>(plan->ny),
    };
    PyObject *arr_obj = PyArray_SimpleNew(2, dims, NPY_BOOL);
    if (!arr_obj) {
        return nullptr;
    }

    npy_bool *data = reinterpret_cast<npy_bool *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj)));
    size_t total_size = plan->nx * plan->ny;
    for (size_t k = 0; k < total_size; ++k) {
        const size_t *mask = component_grid_plan_support_mask(plan, k);
        data[k] = (support_mask_popcount(mask, plan->support_mask_words) == 1 &&
                   support_mask_contains_rank(mask, plan->support_mask_words,
                                              meep::my_rank()))
                      ? NPY_TRUE
                      : NPY_FALSE;
    }

    return arr_obj;
}

static PyObject *component_grid_plan_local_boundary_mask(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;

    if (!PyArg_ParseTuple(args, "O", &plan_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    npy_intp dims[2] = {
        static_cast<npy_intp>(plan->nx),
        static_cast<npy_intp>(plan->ny),
    };
    PyObject *arr_obj = PyArray_SimpleNew(2, dims, NPY_BOOL);
    if (!arr_obj) {
        return nullptr;
    }

    npy_bool *data = reinterpret_cast<npy_bool *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj)));
    size_t total_size = plan->nx * plan->ny;
    for (size_t k = 0; k < total_size; ++k) {
        const size_t *mask = component_grid_plan_support_mask(plan, k);
        data[k] = (support_mask_contains_rank(mask, plan->support_mask_words,
                                              meep::my_rank()) &&
                   support_mask_popcount(mask, plan->support_mask_words) > 1)
                      ? NPY_TRUE
                      : NPY_FALSE;
    }

    return arr_obj;
}

static PyObject *configure_component_grid_plan_history(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *local_indices_obj = nullptr;
    PyObject *boundary_indices_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &plan_obj, &local_indices_obj,
                          &boundary_indices_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    const size_t total_size = plan->nx * plan->ny;
    std::vector<size_t> local_indices;
    std::vector<size_t> boundary_indices;
    if (read_index_sequence(local_indices_obj, local_indices, total_size,
                            "local_indices must be a 1D integer array") < 0 ||
        read_index_sequence(boundary_indices_obj, boundary_indices, total_size,
                            "boundary_indices must be a 1D integer array") < 0) {
        return nullptr;
    }

    std::vector<HistoryBoundaryGroup> groups(plan->support_comms.size());
    const int rank = meep::my_rank();
    for (size_t result_idx = 0; result_idx < boundary_indices.size(); ++result_idx) {
        const size_t point_idx = boundary_indices[result_idx];
        const size_t *mask = component_grid_plan_support_mask(plan, point_idx);
        const int group_id = plan->support_group_ids[point_idx];
        if (!support_mask_contains_rank(mask, plan->support_mask_words, rank) ||
            group_id < 0 || group_id >= static_cast<int>(plan->support_comms.size()) ||
            plan->support_comms[group_id] == MPI_COMM_NULL) {
            PyErr_SetString(
                PyExc_ValueError,
                "history boundary indices must belong to this rank's support groups");
            return nullptr;
        }
        HistoryBoundaryGroup &group = groups[static_cast<size_t>(group_id)];
        group.result_indices.push_back(local_indices.size() + result_idx);
        group.point_indices.push_back(point_idx);
    }
    for (HistoryBoundaryGroup &group : groups) {
        group.local.resize(group.point_indices.size(), std::complex<double>(0.0, 0.0));
        group.reduced.resize(group.point_indices.size(),
                             std::complex<double>(0.0, 0.0));
    }

    std::vector<size_t> accumulation_indices = local_indices;
    accumulation_indices.insert(accumulation_indices.end(), boundary_indices.begin(),
                                boundary_indices.end());
    plan->history_local_indices = std::move(local_indices);
    plan->history_indices = accumulation_indices;
    plan->accumulation_indices = std::move(accumulation_indices);
    plan->previous_adjoint_values.assign(plan->accumulation_indices.size(),
                                         std::complex<double>(0.0, 0.0));
    plan->adjoint_difference_initialized = false;
    plan->history_boundary_groups = std::move(groups);
    plan->history_sampling_configured = true;
    plan->accumulation_configured = true;
    Py_RETURN_NONE;
}

static PyObject *sample_component_grid_plan_history_into(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *destination_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &plan_obj, &destination_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    if (!plan->history_sampling_configured) {
        PyErr_SetString(PyExc_RuntimeError,
                        "component grid history layout is not configured");
        return nullptr;
    }

    PyArrayObject *destination = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(destination_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY));
    if (!destination) {
        return nullptr;
    }
    const size_t history_width = plan->history_indices.size();
    const bool shape_ok =
        PyArray_NDIM(destination) == 1 &&
        PyArray_DIM(destination, 0) == static_cast<npy_intp>(history_width);
    if (!shape_ok) {
        PyArray_DiscardWritebackIfCopy(destination);
        Py_DECREF(destination);
        PyErr_SetString(
            PyExc_ValueError,
            "history destination must be a 1D complex128 array matching the configured width");
        return nullptr;
    }
    npy_cdouble *destination_data =
        reinterpret_cast<npy_cdouble *>(PyArray_DATA(destination));

    try {
        for (size_t i = 0; i < plan->history_local_indices.size(); ++i) {
            set_npy_complex(
                destination_data[i],
                sample_component_plan_local(plan, plan->history_local_indices[i]));
        }
        for (size_t group_id = 0; group_id < plan->history_boundary_groups.size();
             ++group_id) {
            HistoryBoundaryGroup &group = plan->history_boundary_groups[group_id];
            if (group.point_indices.empty()) {
                continue;
            }
            for (size_t i = 0; i < group.point_indices.size(); ++i) {
                group.local[i] =
                    sample_component_plan_local(plan, group.point_indices[i]);
            }
            require_mpi_success(
                MPI_Allreduce(
                    reinterpret_cast<double *>(group.local.data()),
                    reinterpret_cast<double *>(group.reduced.data()),
                    checked_complex_mpi_double_count(group.point_indices.size()),
                    MPI_DOUBLE, MPI_SUM, plan->support_comms[group_id]),
                "MPI_Allreduce");
            for (size_t i = 0; i < group.result_indices.size(); ++i) {
                set_npy_complex(destination_data[group.result_indices[i]],
                                group.reduced[i]);
            }
        }
    } catch (const std::overflow_error &e) {
        PyArray_DiscardWritebackIfCopy(destination);
        Py_DECREF(destination);
        PyErr_SetString(PyExc_OverflowError, e.what());
        return nullptr;
    } catch (const std::exception &e) {
        PyArray_DiscardWritebackIfCopy(destination);
        Py_DECREF(destination);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyArray_DiscardWritebackIfCopy(destination);
        Py_DECREF(destination);
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in planned history sampling");
        return nullptr;
    }

    if (PyArray_ResolveWritebackIfCopy(destination) < 0) {
        Py_DECREF(destination);
        return nullptr;
    }
    Py_DECREF(destination);
    Py_RETURN_NONE;
}

static PyObject *configure_component_grid_plan_accumulation(PyObject *,
                                                            PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *indices_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &plan_obj, &indices_obj)) {
        return nullptr;
    }
    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    std::vector<size_t> indices;
    if (read_index_sequence(indices_obj, indices, plan->nx * plan->ny,
                            "indices must be a 1D integer array") < 0) {
        return nullptr;
    }
    plan->accumulation_indices = std::move(indices);
    plan->previous_adjoint_values.assign(plan->accumulation_indices.size(),
                                         std::complex<double>(0.0, 0.0));
    plan->adjoint_difference_initialized = false;
    plan->accumulation_configured = true;
    Py_RETURN_NONE;
}

static PyObject *sample_component_grid_plan_points_local(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *indices_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OO", &plan_obj, &indices_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    size_t total_size = plan->nx * plan->ny;
    std::vector<size_t> indices;
    if (read_index_sequence(indices_obj, indices, total_size,
                            "indices must be a 1D integer array") < 0) {
        return nullptr;
    }

    npy_intp dims[1] = {
        static_cast<npy_intp>(indices.size()),
    };
    PyObject *arr_obj = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }

    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj)));
    try {
        for (size_t i = 0; i < indices.size(); ++i) {
            set_npy_complex(data[i], sample_component_plan_local(plan, indices[i]));
        }
    } catch (const std::exception &e) {
        Py_DECREF(arr_obj);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(arr_obj);
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in local planned point sampling");
        return nullptr;
    }

    return arr_obj;
}

static PyObject *sample_component_grid_plan_points_allreduced(PyObject *,
                                                              PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *indices_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OO", &plan_obj, &indices_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    size_t total_size = plan->nx * plan->ny;
    std::vector<size_t> indices;
    if (read_index_sequence(indices_obj, indices, total_size,
                            "indices must be a 1D integer array") < 0) {
        return nullptr;
    }
    if (indices.size() > static_cast<size_t>(INT_MAX)) {
        PyErr_SetString(PyExc_OverflowError,
                        "sample point index list is too large for Meep MPI reduction");
        return nullptr;
    }

    std::vector<std::complex<double>> local(indices.size(),
                                            std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> reduced(indices.size(),
                                              std::complex<double>(0.0, 0.0));

    try {
        for (size_t i = 0; i < indices.size(); ++i) {
            local[i] = sample_component_plan_local(plan, indices[i]);
        }
        if (!indices.empty()) {
            meep::sum_to_all(local.data(), reduced.data(),
                             static_cast<int>(indices.size()));
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in reduced planned point sampling");
        return nullptr;
    }

    npy_intp dims[1] = {
        static_cast<npy_intp>(indices.size()),
    };
    PyObject *arr_obj = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }

    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj)));
    for (size_t i = 0; i < indices.size(); ++i) {
        set_npy_complex(data[i], reduced[i]);
    }

    return arr_obj;
}

static PyObject *sample_component_grid_plan_points_support_reduced(PyObject *,
                                                                   PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *indices_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OO", &plan_obj, &indices_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    size_t total_size = plan->nx * plan->ny;
    std::vector<size_t> indices;
    if (read_index_sequence(indices_obj, indices, total_size,
                            "indices must be a 1D integer array") < 0) {
        return nullptr;
    }

    std::map<int, std::vector<std::pair<size_t, size_t>>> groups;
    for (size_t result_idx = 0; result_idx < indices.size(); ++result_idx) {
        size_t point_idx = indices[result_idx];
        const size_t *mask = component_grid_plan_support_mask(plan, point_idx);
        const int group_id = plan->support_group_ids[point_idx];
        if (!support_mask_contains_rank(mask, plan->support_mask_words,
                                        meep::my_rank()) ||
            group_id < 0) {
            PyErr_SetString(
                PyExc_ValueError,
                "support-reduced sampling requires local boundary point indices");
            return nullptr;
        }
        groups[group_id].push_back({result_idx, point_idx});
    }

    std::vector<std::complex<double>> result(indices.size(),
                                             std::complex<double>(0.0, 0.0));

    try {
        for (const auto &group : groups) {
            const int group_id = group.first;
            const auto &items = group.second;
            if (group_id >= static_cast<int>(plan->support_comms.size()) ||
                plan->support_comms[group_id] == MPI_COMM_NULL) {
                PyErr_SetString(PyExc_RuntimeError,
                                "support-rank communicator is unavailable");
                return nullptr;
            }

            std::vector<std::complex<double>> local(items.size(),
                                                    std::complex<double>(0.0, 0.0));
            std::vector<std::complex<double>> reduced(items.size(),
                                                      std::complex<double>(0.0, 0.0));
            for (size_t i = 0; i < items.size(); ++i) {
                local[i] = sample_component_plan_local(plan, items[i].second);
            }

            require_mpi_success(
                MPI_Allreduce(reinterpret_cast<double *>(local.data()),
                              reinterpret_cast<double *>(reduced.data()),
                              checked_complex_mpi_double_count(items.size()),
                              MPI_DOUBLE, MPI_SUM, plan->support_comms[group_id]),
                "MPI_Allreduce");

            for (size_t i = 0; i < items.size(); ++i) {
                result[items[i].first] = reduced[i];
            }
        }
    } catch (const std::overflow_error &e) {
        PyErr_SetString(PyExc_OverflowError, e.what());
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in support-rank planned point sampling");
        return nullptr;
    }

    npy_intp dims[1] = {
        static_cast<npy_intp>(indices.size()),
    };
    PyObject *arr_obj = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }

    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj)));
    for (size_t i = 0; i < indices.size(); ++i) {
        set_npy_complex(data[i], result[i]);
    }

    return arr_obj;
}

static PyObject *sample_component_grid_plan_allreduced(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;

    if (!PyArg_ParseTuple(args, "O", &plan_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    size_t total_size = plan->nx * plan->ny;
    std::vector<std::complex<double>> local(total_size, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> reduced(total_size,
                                              std::complex<double>(0.0, 0.0));

    try {
        for (size_t k = 0; k < total_size; ++k) {
            local[k] = sample_component_plan_local(plan, k);
        }

        meep::sum_to_all(local.data(), reduced.data(), static_cast<int>(total_size));
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in planned Meep field sampling");
        return nullptr;
    }

    npy_intp dims[2] = {
        static_cast<npy_intp>(plan->nx),
        static_cast<npy_intp>(plan->ny),
    };
    PyObject *arr_obj = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }

    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj)));
    for (size_t k = 0; k < total_size; ++k) {
        set_npy_complex(data[k], reduced[k]);
    }

    return arr_obj;
}

static PyObject *sample_component_grid_allreduced(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    PyObject *xs_obj = nullptr;
    PyObject *ys_obj = nullptr;
    int component_int = static_cast<int>(meep::Ez);

    if (!PyArg_ParseTuple(args, "KOO|i", &fields_addr, &xs_obj, &ys_obj,
                          &component_int)) {
        return nullptr;
    }

    if (fields_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "fields pointer address must be non-zero");
        return nullptr;
    }

    if (!require_dynamic_field_component(component_int)) {
        return nullptr;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    if (read_double_sequence(xs_obj, xs, "coords_x must be a sequence") < 0 ||
        read_double_sequence(ys_obj, ys, "coords_y must be a sequence") < 0) {
        return nullptr;
    }

    size_t total_size = xs.size() * ys.size();
    if (total_size > static_cast<size_t>(INT_MAX)) {
        PyErr_SetString(PyExc_OverflowError,
                        "sample grid is too large for Meep MPI reduction");
        return nullptr;
    }

    std::vector<std::complex<double>> local(total_size, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> reduced(total_size,
                                              std::complex<double>(0.0, 0.0));

    meep::fields *fields =
        reinterpret_cast<meep::fields *>(static_cast<uintptr_t>(fields_addr));
    if (!require_cartesian_2d_fields(fields)) {
        return nullptr;
    }
    meep::component component = static_cast<meep::component>(component_int);

    try {
        size_t k = 0;
        for (double x : xs) {
            for (double y : ys) {
                local[k] = sample_component_local(fields, component, x, y);
                ++k;
            }
        }

        meep::sum_to_all(local.data(), reduced.data(), static_cast<int>(total_size));
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in local Meep field sampling");
        return nullptr;
    }

    npy_intp dims[2] = {
        static_cast<npy_intp>(xs.size()),
        static_cast<npy_intp>(ys.size()),
    };
    PyObject *arr_obj = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }

    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj)));
    for (size_t k = 0; k < total_size; ++k) {
        set_npy_complex(data[k], reduced[k]);
    }

    return arr_obj;
}

static PyObject *accumulate_component_product_allreduced(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    PyObject *xs_obj = nullptr;
    PyObject *ys_obj = nullptr;
    int component_int = static_cast<int>(meep::Ez);
    PyObject *multiplier_obj = nullptr;

    if (!PyArg_ParseTuple(args, "KOOiO", &fields_addr, &xs_obj, &ys_obj, &component_int,
                          &multiplier_obj)) {
        return nullptr;
    }

    if (fields_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "fields pointer address must be non-zero");
        return nullptr;
    }

    if (!require_dynamic_field_component(component_int)) {
        return nullptr;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    if (read_double_sequence(xs_obj, xs, "coords_x must be a sequence") < 0 ||
        read_double_sequence(ys_obj, ys, "coords_y must be a sequence") < 0) {
        return nullptr;
    }

    PyArrayObject *multiplier = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(multiplier_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY));
    if (!multiplier) {
        return nullptr;
    }

    if (PyArray_NDIM(multiplier) != 2 ||
        PyArray_DIM(multiplier, 0) != static_cast<npy_intp>(xs.size()) ||
        PyArray_DIM(multiplier, 1) != static_cast<npy_intp>(ys.size())) {
        Py_DECREF(multiplier);
        PyErr_SetString(PyExc_ValueError,
                        "multiplier must have shape (len(coords_x), len(coords_y))");
        return nullptr;
    }

    size_t total_size = xs.size() * ys.size();
    if (total_size > static_cast<size_t>(INT_MAX)) {
        Py_DECREF(multiplier);
        PyErr_SetString(PyExc_OverflowError,
                        "sample grid is too large for Meep MPI reduction");
        return nullptr;
    }

    std::vector<std::complex<double>> local;
    std::vector<std::complex<double>> reduced;
    try {
        local.assign(total_size, std::complex<double>(0.0, 0.0));
        reduced.assign(total_size, std::complex<double>(0.0, 0.0));
    } catch (...) {
        Py_DECREF(multiplier);
        throw;
    }

    meep::fields *fields =
        reinterpret_cast<meep::fields *>(static_cast<uintptr_t>(fields_addr));
    if (!require_cartesian_2d_fields(fields)) {
        Py_DECREF(multiplier);
        return nullptr;
    }
    meep::component component = static_cast<meep::component>(component_int);
    npy_cdouble *mult_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(multiplier));

    try {
        size_t k = 0;
        for (double x : xs) {
            for (double y : ys) {
                std::complex<double> scale = npy_to_complex(mult_data[k]);
                local[k] = sample_component_local(fields, component, x, y) * scale;
                ++k;
            }
        }

        meep::sum_to_all(local.data(), reduced.data(), static_cast<int>(total_size));
    } catch (const std::exception &e) {
        Py_DECREF(multiplier);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(multiplier);
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in local Meep product accumulation");
        return nullptr;
    }

    Py_DECREF(multiplier);

    npy_intp dims[2] = {
        static_cast<npy_intp>(xs.size()),
        static_cast<npy_intp>(ys.size()),
    };
    PyObject *arr_obj = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }

    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj)));
    for (size_t k = 0; k < total_size; ++k) {
        set_npy_complex(data[k], reduced[k]);
    }

    return arr_obj;
}

static PyObject *accumulate_component_product_local_inplace(PyObject *,
                                                            PyObject *args) {
    unsigned long long fields_addr = 0;
    PyObject *xs_obj = nullptr;
    PyObject *ys_obj = nullptr;
    int component_int = static_cast<int>(meep::Ez);
    PyObject *multiplier_obj = nullptr;
    PyObject *accumulator_obj = nullptr;

    if (!PyArg_ParseTuple(args, "KOOiOO", &fields_addr, &xs_obj, &ys_obj,
                          &component_int, &multiplier_obj, &accumulator_obj)) {
        return nullptr;
    }

    if (fields_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "fields pointer address must be non-zero");
        return nullptr;
    }

    if (!require_dynamic_field_component(component_int)) {
        return nullptr;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    if (read_double_sequence(xs_obj, xs, "coords_x must be a sequence") < 0 ||
        read_double_sequence(ys_obj, ys, "coords_y must be a sequence") < 0) {
        return nullptr;
    }

    PyArrayObject *multiplier = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(multiplier_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY));
    if (!multiplier) {
        return nullptr;
    }

    PyArrayObject *accumulator = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(accumulator_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY));
    if (!accumulator) {
        Py_DECREF(multiplier);
        return nullptr;
    }

    bool shape_ok = PyArray_NDIM(multiplier) == 2 && PyArray_NDIM(accumulator) == 2 &&
                    PyArray_DIM(multiplier, 0) == static_cast<npy_intp>(xs.size()) &&
                    PyArray_DIM(multiplier, 1) == static_cast<npy_intp>(ys.size()) &&
                    PyArray_DIM(accumulator, 0) == static_cast<npy_intp>(xs.size()) &&
                    PyArray_DIM(accumulator, 1) == static_cast<npy_intp>(ys.size());
    if (!shape_ok) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(
            PyExc_ValueError,
            "multiplier and accumulator must have shape (len(coords_x), len(coords_y))");
        return nullptr;
    }

    meep::fields *fields =
        reinterpret_cast<meep::fields *>(static_cast<uintptr_t>(fields_addr));
    if (!require_cartesian_2d_fields(fields)) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        return nullptr;
    }
    meep::component component = static_cast<meep::component>(component_int);
    npy_cdouble *mult_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(multiplier));
    npy_cdouble *accum_data =
        reinterpret_cast<npy_cdouble *>(PyArray_DATA(accumulator));

    try {
        size_t k = 0;
        for (double x : xs) {
            for (double y : ys) {
                std::complex<double> scale = npy_to_complex(mult_data[k]);
                std::complex<double> value =
                    sample_component_local(fields, component, x, y) * scale;
                add_npy_complex(accum_data[k], value);
                ++k;
            }
        }
    } catch (const std::exception &e) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in local Meep product accumulation");
        return nullptr;
    }

    Py_DECREF(multiplier);
    if (PyArray_ResolveWritebackIfCopy(accumulator) < 0) {
        Py_DECREF(accumulator);
        return nullptr;
    }
    Py_DECREF(accumulator);

    Py_RETURN_NONE;
}

static PyObject *accumulate_component_product_plan_local_inplace(PyObject *,
                                                                 PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *multiplier_obj = nullptr;
    PyObject *accumulator_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OOO", &plan_obj, &multiplier_obj, &accumulator_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    PyArrayObject *multiplier = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(multiplier_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY));
    if (!multiplier) {
        return nullptr;
    }

    PyArrayObject *accumulator = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(accumulator_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY));
    if (!accumulator) {
        Py_DECREF(multiplier);
        return nullptr;
    }

    bool shape_ok = PyArray_NDIM(multiplier) == 2 && PyArray_NDIM(accumulator) == 2 &&
                    PyArray_DIM(multiplier, 0) == static_cast<npy_intp>(plan->nx) &&
                    PyArray_DIM(multiplier, 1) == static_cast<npy_intp>(plan->ny) &&
                    PyArray_DIM(accumulator, 0) == static_cast<npy_intp>(plan->nx) &&
                    PyArray_DIM(accumulator, 1) == static_cast<npy_intp>(plan->ny);
    if (!shape_ok) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_ValueError,
                        "multiplier and accumulator must match the sample plan shape");
        return nullptr;
    }

    size_t total_size = plan->nx * plan->ny;
    npy_cdouble *mult_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(multiplier));
    npy_cdouble *accum_data =
        reinterpret_cast<npy_cdouble *>(PyArray_DATA(accumulator));

    try {
        for (size_t k = 0; k < total_size; ++k) {
            std::complex<double> scale = npy_to_complex(mult_data[k]);
            std::complex<double> value = sample_component_plan_local(plan, k) * scale;
            add_npy_complex(accum_data[k], value);
        }
    } catch (const std::exception &e) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in planned local Meep product accumulation");
        return nullptr;
    }

    Py_DECREF(multiplier);
    if (PyArray_ResolveWritebackIfCopy(accumulator) < 0) {
        Py_DECREF(accumulator);
        return nullptr;
    }
    Py_DECREF(accumulator);

    Py_RETURN_NONE;
}

static PyObject *
accumulate_component_product_plan_points_local_inplace(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *indices_obj = nullptr;
    PyObject *values_obj = nullptr;
    PyObject *accumulator_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OOOO", &plan_obj, &indices_obj, &values_obj,
                          &accumulator_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    size_t total_size = plan->nx * plan->ny;
    std::vector<size_t> indices;
    if (read_index_sequence(indices_obj, indices, total_size,
                            "indices must be a 1D integer array") < 0) {
        return nullptr;
    }

    PyArrayObject *values = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(values_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY));
    if (!values) {
        return nullptr;
    }

    PyArrayObject *accumulator = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(accumulator_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY));
    if (!accumulator) {
        Py_DECREF(values);
        return nullptr;
    }

    bool shape_ok = PyArray_NDIM(values) == 1 &&
                    PyArray_DIM(values, 0) == static_cast<npy_intp>(indices.size()) &&
                    PyArray_NDIM(accumulator) == 2 &&
                    PyArray_DIM(accumulator, 0) == static_cast<npy_intp>(plan->nx) &&
                    PyArray_DIM(accumulator, 1) == static_cast<npy_intp>(plan->ny);
    if (!shape_ok) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(
            PyExc_ValueError,
            "values must match indices and accumulator must match the sample plan shape");
        return nullptr;
    }

    npy_cdouble *value_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(values));
    npy_cdouble *accum_data =
        reinterpret_cast<npy_cdouble *>(PyArray_DATA(accumulator));

    try {
        for (size_t i = 0; i < indices.size(); ++i) {
            size_t point_idx = indices[i];
            std::complex<double> scale = npy_to_complex(value_data[i]);
            std::complex<double> value =
                sample_component_plan_local(plan, point_idx) * scale;
            add_npy_complex(accum_data[point_idx], value);
        }
    } catch (const std::exception &e) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(
            PyExc_RuntimeError,
            "unknown error in planned indexed local Meep product accumulation");
        return nullptr;
    }

    Py_DECREF(values);
    if (PyArray_ResolveWritebackIfCopy(accumulator) < 0) {
        Py_DECREF(accumulator);
        return nullptr;
    }
    Py_DECREF(accumulator);

    Py_RETURN_NONE;
}

static PyObject *
accumulate_component_product_plan_configured_local_inplace(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *values_obj = nullptr;
    PyObject *accumulator_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &plan_obj, &values_obj, &accumulator_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    if (!plan->accumulation_configured) {
        PyErr_SetString(PyExc_RuntimeError,
                        "component grid accumulation indices are not configured");
        return nullptr;
    }

    PyArrayObject *values = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(values_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY));
    if (!values) {
        return nullptr;
    }
    PyArrayObject *accumulator = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(accumulator_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY));
    if (!accumulator) {
        Py_DECREF(values);
        return nullptr;
    }

    const bool shape_ok =
        PyArray_NDIM(values) == 1 &&
        PyArray_DIM(values, 0) ==
            static_cast<npy_intp>(plan->accumulation_indices.size()) &&
        PyArray_NDIM(accumulator) == 2 &&
        PyArray_DIM(accumulator, 0) == static_cast<npy_intp>(plan->nx) &&
        PyArray_DIM(accumulator, 1) == static_cast<npy_intp>(plan->ny);
    if (!shape_ok) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(
            PyExc_ValueError,
            "values must match configured indices and accumulator must match the sample plan shape");
        return nullptr;
    }

    npy_cdouble *value_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(values));
    npy_cdouble *accum_data =
        reinterpret_cast<npy_cdouble *>(PyArray_DATA(accumulator));
    try {
        for (size_t i = 0; i < plan->accumulation_indices.size(); ++i) {
            const size_t point_idx = plan->accumulation_indices[i];
            const std::complex<double> scale = npy_to_complex(value_data[i]);
            const std::complex<double> value =
                sample_component_plan_local(plan, point_idx) * scale;
            add_npy_complex(accum_data[point_idx], value);
        }
    } catch (const std::exception &e) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown error in configured local Meep product accumulation");
        return nullptr;
    }

    Py_DECREF(values);
    if (PyArray_ResolveWritebackIfCopy(accumulator) < 0) {
        Py_DECREF(accumulator);
        return nullptr;
    }
    Py_DECREF(accumulator);
    Py_RETURN_NONE;
}

static PyObject *
accumulate_component_difference_product_plan_configured_local_inplace(PyObject *,
                                                                      PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *values_obj = nullptr;
    PyObject *accumulator_obj = nullptr;
    double dt = 0.0;
    if (!PyArg_ParseTuple(args, "OOdO", &plan_obj, &values_obj, &dt,
                          &accumulator_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }
    if (!plan->accumulation_configured) {
        PyErr_SetString(PyExc_RuntimeError,
                        "component grid accumulation indices are not configured");
        return nullptr;
    }
    if (!std::isfinite(dt) || dt <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "dt must be a positive finite number");
        return nullptr;
    }

    PyArrayObject *values = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(values_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY));
    if (!values) {
        return nullptr;
    }
    PyArrayObject *accumulator = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(accumulator_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY));
    if (!accumulator) {
        Py_DECREF(values);
        return nullptr;
    }

    const bool shape_ok =
        PyArray_NDIM(values) == 1 &&
        PyArray_DIM(values, 0) ==
            static_cast<npy_intp>(plan->accumulation_indices.size()) &&
        PyArray_NDIM(accumulator) == 2 &&
        PyArray_DIM(accumulator, 0) == static_cast<npy_intp>(plan->nx) &&
        PyArray_DIM(accumulator, 1) == static_cast<npy_intp>(plan->ny);
    if (!shape_ok) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(
            PyExc_ValueError,
            "values must match configured indices and accumulator must match the sample plan shape");
        return nullptr;
    }

    npy_cdouble *value_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(values));
    npy_cdouble *accum_data =
        reinterpret_cast<npy_cdouble *>(PyArray_DATA(accumulator));
    const double inverse_dt = 1.0 / dt;
    bool accumulated = false;
    try {
        if (plan->previous_adjoint_values.size() != plan->accumulation_indices.size()) {
            plan->previous_adjoint_values.assign(plan->accumulation_indices.size(),
                                                 std::complex<double>(0.0, 0.0));
            plan->adjoint_difference_initialized = false;
        }
        for (size_t i = 0; i < plan->accumulation_indices.size(); ++i) {
            const size_t point_idx = plan->accumulation_indices[i];
            const std::complex<double> current =
                sample_component_plan_local(plan, point_idx);
            if (plan->adjoint_difference_initialized) {
                const std::complex<double> forward = npy_to_complex(value_data[i]);
                const std::complex<double> derivative =
                    (current - plan->previous_adjoint_values[i]) * inverse_dt;
                add_npy_complex(accum_data[point_idx], forward * derivative);
            }
            plan->previous_adjoint_values[i] = current;
        }
        accumulated = plan->adjoint_difference_initialized;
        plan->adjoint_difference_initialized = true;
    } catch (const std::exception &e) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(values);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(
            PyExc_RuntimeError,
            "unknown error in configured local Meep difference accumulation");
        return nullptr;
    }

    Py_DECREF(values);
    if (PyArray_ResolveWritebackIfCopy(accumulator) < 0) {
        Py_DECREF(accumulator);
        return nullptr;
    }
    Py_DECREF(accumulator);
    if (accumulated) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *reduce_complex_grid_sum(PyObject *, PyObject *args) {
    PyObject *local_obj = nullptr;

    if (!PyArg_ParseTuple(args, "O", &local_obj)) {
        return nullptr;
    }

    PyArrayObject *local_arr = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(local_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY));
    if (!local_arr) {
        return nullptr;
    }

    if (PyArray_NDIM(local_arr) != 2) {
        Py_DECREF(local_arr);
        PyErr_SetString(PyExc_ValueError, "local grid must be a 2D complex128 array");
        return nullptr;
    }

    npy_intp dims[2] = {
        PyArray_DIM(local_arr, 0),
        PyArray_DIM(local_arr, 1),
    };
    const size_t total_size = static_cast<size_t>(PyArray_SIZE(local_arr));
    size_t scalar_count = 0;
    if (!checked_size_product(total_size, 2, scalar_count)) {
        Py_DECREF(local_arr);
        PyErr_SetString(PyExc_OverflowError,
                        "complex grid element count overflows size_t");
        return nullptr;
    }

    PyObject *reduced_obj = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    if (!reduced_obj) {
        Py_DECREF(local_arr);
        return nullptr;
    }

    const npy_cdouble *local_data =
        reinterpret_cast<const npy_cdouble *>(PyArray_DATA(local_arr));
    npy_cdouble *reduced_data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(reduced_obj)));
    if (total_size != 0) {
        std::copy(local_data, local_data + total_size, reduced_data);
    }

    try {
        allreduce_inplace_chunked(reduced_data, scalar_count, sizeof(double),
                                  MPI_DOUBLE, static_cast<size_t>(INT_MAX));
    } catch (const std::exception &e) {
        Py_DECREF(local_arr);
        Py_DECREF(reduced_obj);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(local_arr);
        Py_DECREF(reduced_obj);
        PyErr_SetString(PyExc_RuntimeError, "unknown error in Meep grid reduction");
        return nullptr;
    }

    Py_DECREF(local_arr);
    return reduced_obj;
}

struct InplaceReductionSpec {
    size_t scalar_count;
    size_t scalar_size;
    MPI_Datatype datatype;
};

static bool inplace_reduction_spec(PyArrayObject *array, bool complex_values,
                                   InplaceReductionSpec &spec) {
    const int type_number = PyArray_TYPE(array);
    const bool dtype_supported =
        complex_values ? (type_number == NPY_COMPLEX64 || type_number == NPY_COMPLEX128)
                       : (type_number == NPY_FLOAT32 || type_number == NPY_FLOAT64);
    if (!dtype_supported || !PyArray_ISCARRAY(array) || !PyArray_ISNOTSWAPPED(array)) {
        PyErr_SetString(PyExc_ValueError,
                        complex_values
                            ? "in-place reduction requires a writable C-contiguous "
                              "native-endian complex64 or complex128 array"
                            : "in-place reduction requires a writable C-contiguous "
                              "native-endian float32 or float64 array");
        return false;
    }

    const size_t element_count = static_cast<size_t>(PyArray_SIZE(array));
    const size_t scalar_multiplier = complex_values ? 2u : 1u;
    if (!checked_size_product(element_count, scalar_multiplier, spec.scalar_count)) {
        PyErr_SetString(PyExc_OverflowError, "grid element count overflows size_t");
        return false;
    }
    const bool single_precision =
        type_number == NPY_FLOAT32 || type_number == NPY_COMPLEX64;
    spec.scalar_size = single_precision ? sizeof(float) : sizeof(double);
    spec.datatype = single_precision ? MPI_FLOAT : MPI_DOUBLE;
    return true;
}

static PyObject *reduce_grid_sum_inplace_impl(PyObject *local_obj, bool complex_values,
                                              size_t max_mpi_count) {
    if (!PyArray_Check(local_obj)) {
        PyErr_SetString(PyExc_TypeError, "local grid must be a NumPy array");
        return nullptr;
    }

    PyArrayObject *local_arr = reinterpret_cast<PyArrayObject *>(local_obj);
    if (PyArray_NDIM(local_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, complex_values
                                              ? "local grid must be a 2D complex array"
                                              : "local grid must be a 2D real array");
        return nullptr;
    }
    InplaceReductionSpec spec;
    if (!inplace_reduction_spec(local_arr, complex_values, spec)) {
        return nullptr;
    }

    allreduce_inplace_chunked(PyArray_DATA(local_arr), spec.scalar_count,
                              spec.scalar_size, spec.datatype, max_mpi_count);
    Py_INCREF(local_obj);
    return local_obj;
}

static PyObject *reduce_complex_grid_sum_inplace(PyObject *, PyObject *args) {
    PyObject *local_obj = nullptr;

    if (!PyArg_ParseTuple(args, "O", &local_obj)) {
        return nullptr;
    }
    return reduce_grid_sum_inplace_impl(local_obj, true, static_cast<size_t>(INT_MAX));
}

static PyObject *reduce_real_grid_sum_inplace(PyObject *, PyObject *args) {
    PyObject *local_obj = nullptr;

    if (!PyArg_ParseTuple(args, "O", &local_obj)) {
        return nullptr;
    }
    return reduce_grid_sum_inplace_impl(local_obj, false, static_cast<size_t>(INT_MAX));
}

static PyObject *reduce_grid_sum_inplace_for_testing(PyObject *, PyObject *args) {
    PyObject *local_obj = nullptr;
    unsigned long long max_mpi_count = 0;
    if (!PyArg_ParseTuple(args, "OK:_reduce_grid_sum_inplace_for_testing", &local_obj,
                          &max_mpi_count)) {
        return nullptr;
    }
    if (max_mpi_count == 0) {
        PyErr_SetString(PyExc_ValueError, "max_mpi_count must be positive");
        return nullptr;
    }
    if (max_mpi_count > std::numeric_limits<size_t>::max()) {
        PyErr_SetString(PyExc_OverflowError, "max_mpi_count exceeds size_t");
        return nullptr;
    }
    if (!PyArray_Check(local_obj)) {
        PyErr_SetString(PyExc_TypeError, "local grid must be a NumPy array");
        return nullptr;
    }
    const int type_number = PyArray_TYPE(reinterpret_cast<PyArrayObject *>(local_obj));
    if (type_number != NPY_FLOAT32 && type_number != NPY_FLOAT64 &&
        type_number != NPY_COMPLEX64 && type_number != NPY_COMPLEX128) {
        PyErr_SetString(
            PyExc_ValueError,
            "test reduction requires float32, float64, complex64, or complex128");
        return nullptr;
    }
    return reduce_grid_sum_inplace_impl(
        local_obj, type_number == NPY_COMPLEX64 || type_number == NPY_COMPLEX128,
        static_cast<size_t>(max_mpi_count));
}

static bool normalize_tabulated_time(double query_time, double support_start,
                                     double support_end, double time_shift,
                                     double &base_time) {
    const double shifted_start = support_start + time_shift;
    const double shifted_end = support_end + time_shift;
    // Reversing a time grid can lose endpoint precision through cancellation.
    const double tolerance = 4.0 * std::numeric_limits<double>::epsilon() *
                             std::max({std::abs(support_start), std::abs(support_end),
                                       std::abs(time_shift)});
    if (query_time < shifted_start - tolerance ||
        query_time > shifted_end + tolerance) {
        return false;
    }
    base_time = std::min(support_end, std::max(support_start, query_time - time_shift));
    return true;
}

typedef struct {
    PyObject_HEAD PyArrayObject *breaks;
    PyArrayObject *coefficients;
} TabulatedCubicObject;

static void tabulated_cubic_dealloc(PyObject *self_obj) {
    TabulatedCubicObject *self = reinterpret_cast<TabulatedCubicObject *>(self_obj);
    PyTypeObject *type = Py_TYPE(self_obj);
    Py_XDECREF(self->breaks);
    Py_XDECREF(self->coefficients);
    type->tp_free(self_obj);
    Py_DECREF(type);
}

static PyObject *tabulated_cubic_new(PyTypeObject *, PyObject *, PyObject *) {
    PyErr_SetString(
        PyExc_TypeError,
        "TabulatedCubic cannot be instantiated directly; use create_tabulated_cubic");
    return nullptr;
}

static PyObject *tabulated_cubic_call(PyObject *self_obj, PyObject *args,
                                      PyObject *kwargs) {
    if (kwargs && PyDict_Size(kwargs) != 0) {
        PyErr_SetString(PyExc_TypeError,
                        "tabulated cubic source accepts no keyword arguments");
        return nullptr;
    }
    double time = 0.0;
    if (!PyArg_ParseTuple(args, "d", &time)) {
        return nullptr;
    }
    TabulatedCubicObject *self = reinterpret_cast<TabulatedCubicObject *>(self_obj);
    const npy_intp n_breaks = PyArray_DIM(self->breaks, 0);
    const double *breaks = reinterpret_cast<const double *>(PyArray_DATA(self->breaks));
    if (!normalize_tabulated_time(time, breaks[0], breaks[n_breaks - 1], 0.0, time)) {
        return PyComplex_FromDoubles(0.0, 0.0);
    }

    const double *upper = std::upper_bound(breaks, breaks + n_breaks, time);
    npy_intp interval = static_cast<npy_intp>(upper - breaks) - 1;
    if (interval < 0) {
        interval = 0;
    } else if (interval >= n_breaks - 1) {
        interval = n_breaks - 2;
    }
    const double offset = time - breaks[interval];
    const npy_intp n_intervals = n_breaks - 1;
    const npy_cdouble *coefficients =
        reinterpret_cast<const npy_cdouble *>(PyArray_DATA(self->coefficients));
    std::complex<double> value = npy_to_complex(coefficients[interval]);
    value = value * offset + npy_to_complex(coefficients[n_intervals + interval]);
    value = value * offset + npy_to_complex(coefficients[2 * n_intervals + interval]);
    value = value * offset + npy_to_complex(coefficients[3 * n_intervals + interval]);
    return PyComplex_FromDoubles(value.real(), value.imag());
}

static PyType_Slot tabulated_cubic_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(tabulated_cubic_dealloc)},
    {Py_tp_new, reinterpret_cast<void *>(tabulated_cubic_new)},
    {Py_tp_call, reinterpret_cast<void *>(tabulated_cubic_call)},
    {0, nullptr},
};

static PyType_Spec tabulated_cubic_spec = {
    "tama.native_sampler.TabulatedCubic",
    sizeof(TabulatedCubicObject),
    0,
    Py_TPFLAGS_DEFAULT,
    tabulated_cubic_slots,
};

static PyTypeObject *tabulated_cubic_type = nullptr;

static PyObject *create_tabulated_cubic(PyObject *, PyObject *args) {
    PyObject *breaks_obj = nullptr;
    PyObject *coefficients_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &breaks_obj, &coefficients_obj)) {
        return nullptr;
    }
    PyArrayObject *breaks = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(breaks_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!breaks) {
        return nullptr;
    }
    PyArrayObject *coefficients = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(coefficients_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY));
    if (!coefficients) {
        Py_DECREF(breaks);
        return nullptr;
    }

    const bool shape_ok = PyArray_NDIM(breaks) == 1 && PyArray_DIM(breaks, 0) >= 2 &&
                          PyArray_NDIM(coefficients) == 2 &&
                          PyArray_DIM(coefficients, 0) == 4 &&
                          PyArray_DIM(coefficients, 1) == PyArray_DIM(breaks, 0) - 1;
    if (!shape_ok) {
        Py_DECREF(breaks);
        Py_DECREF(coefficients);
        PyErr_SetString(PyExc_ValueError,
                        "breaks must have shape (n+1,) and coefficients shape (4, n)");
        return nullptr;
    }
    const double *break_data = reinterpret_cast<const double *>(PyArray_DATA(breaks));
    for (npy_intp i = 1; i < PyArray_DIM(breaks, 0); ++i) {
        if (!(break_data[i] > break_data[i - 1])) {
            Py_DECREF(breaks);
            Py_DECREF(coefficients);
            PyErr_SetString(PyExc_ValueError, "breaks must be strictly increasing");
            return nullptr;
        }
    }

    TabulatedCubicObject *result = reinterpret_cast<TabulatedCubicObject *>(
        tabulated_cubic_type->tp_alloc(tabulated_cubic_type, 0));
    if (!result) {
        Py_DECREF(breaks);
        Py_DECREF(coefficients);
        return nullptr;
    }
    result->breaks = breaks;
    result->coefficients = coefficients;
    return reinterpret_cast<PyObject *>(result);
}

typedef struct {
    PyObject_HEAD PyArrayObject *breaks;
    PyArrayObject *coefficients;
    npy_intp channel;
} TabulatedRealCubicObject;

static void tabulated_real_cubic_dealloc(PyObject *self_obj) {
    TabulatedRealCubicObject *self =
        reinterpret_cast<TabulatedRealCubicObject *>(self_obj);
    PyTypeObject *type = Py_TYPE(self_obj);
    Py_XDECREF(self->breaks);
    Py_XDECREF(self->coefficients);
    type->tp_free(self_obj);
    Py_DECREF(type);
}

static PyObject *tabulated_real_cubic_new(PyTypeObject *, PyObject *, PyObject *) {
    PyErr_SetString(PyExc_TypeError,
                    "TabulatedRealCubic cannot be instantiated directly; use "
                    "create_tabulated_real_cubic_bank");
    return nullptr;
}

static PyObject *tabulated_real_cubic_call(PyObject *self_obj, PyObject *args,
                                           PyObject *kwargs) {
    if (kwargs && PyDict_Size(kwargs) != 0) {
        PyErr_SetString(PyExc_TypeError,
                        "tabulated real cubic source accepts no keyword arguments");
        return nullptr;
    }
    double time = 0.0;
    if (!PyArg_ParseTuple(args, "d", &time)) {
        return nullptr;
    }

    TabulatedRealCubicObject *self =
        reinterpret_cast<TabulatedRealCubicObject *>(self_obj);
    const npy_intp n_breaks = PyArray_DIM(self->breaks, 0);
    const double *breaks = reinterpret_cast<const double *>(PyArray_DATA(self->breaks));
    if (!normalize_tabulated_time(time, breaks[0], breaks[n_breaks - 1], 0.0, time)) {
        return PyFloat_FromDouble(0.0);
    }

    const double *upper = std::upper_bound(breaks, breaks + n_breaks, time);
    npy_intp interval = static_cast<npy_intp>(upper - breaks) - 1;
    if (interval < 0) {
        interval = 0;
    } else if (interval >= n_breaks - 1) {
        interval = n_breaks - 2;
    }
    const double offset = time - breaks[interval];
    const npy_intp n_intervals = n_breaks - 1;
    const npy_intp n_channels = PyArray_DIM(self->coefficients, 2);
    const double *coefficients =
        reinterpret_cast<const double *>(PyArray_DATA(self->coefficients));
    const auto coefficient = [&](npy_intp degree) {
        return coefficients[(degree * n_intervals + interval) * n_channels +
                            self->channel];
    };
    double value = coefficient(0);
    value = value * offset + coefficient(1);
    value = value * offset + coefficient(2);
    value = value * offset + coefficient(3);
    return PyFloat_FromDouble(value);
}

static PyType_Slot tabulated_real_cubic_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(tabulated_real_cubic_dealloc)},
    {Py_tp_new, reinterpret_cast<void *>(tabulated_real_cubic_new)},
    {Py_tp_call, reinterpret_cast<void *>(tabulated_real_cubic_call)},
    {0, nullptr},
};

static PyType_Spec tabulated_real_cubic_spec = {
    "tama.native_sampler.TabulatedRealCubic",
    sizeof(TabulatedRealCubicObject),
    0,
    Py_TPFLAGS_DEFAULT,
    tabulated_real_cubic_slots,
};

static PyTypeObject *tabulated_real_cubic_type = nullptr;

static PyObject *create_tabulated_real_cubic_bank(PyObject *, PyObject *args) {
    PyObject *breaks_obj = nullptr;
    PyObject *coefficients_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &breaks_obj, &coefficients_obj)) {
        return nullptr;
    }
    PyArrayObject *breaks = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(breaks_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!breaks) {
        return nullptr;
    }
    PyArrayObject *coefficients = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(coefficients_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!coefficients) {
        Py_DECREF(breaks);
        return nullptr;
    }

    const bool shape_ok = PyArray_NDIM(breaks) == 1 && PyArray_DIM(breaks, 0) >= 2 &&
                          PyArray_NDIM(coefficients) == 3 &&
                          PyArray_DIM(coefficients, 0) == 4 &&
                          PyArray_DIM(coefficients, 1) == PyArray_DIM(breaks, 0) - 1 &&
                          PyArray_DIM(coefficients, 2) >= 1;
    if (!shape_ok) {
        Py_DECREF(breaks);
        Py_DECREF(coefficients);
        PyErr_SetString(PyExc_ValueError,
                        "breaks must have shape (n+1,) and coefficients shape "
                        "(4, n, channels) with at least one channel");
        return nullptr;
    }
    const double *break_data = reinterpret_cast<const double *>(PyArray_DATA(breaks));
    for (npy_intp i = 1; i < PyArray_DIM(breaks, 0); ++i) {
        if (!(break_data[i] > break_data[i - 1])) {
            Py_DECREF(breaks);
            Py_DECREF(coefficients);
            PyErr_SetString(PyExc_ValueError, "breaks must be strictly increasing");
            return nullptr;
        }
    }

    const npy_intp n_channels = PyArray_DIM(coefficients, 2);
    PyObject *result = PyTuple_New(static_cast<Py_ssize_t>(n_channels));
    if (!result) {
        Py_DECREF(breaks);
        Py_DECREF(coefficients);
        return nullptr;
    }
    for (npy_intp channel = 0; channel < n_channels; ++channel) {
        TabulatedRealCubicObject *source = reinterpret_cast<TabulatedRealCubicObject *>(
            tabulated_real_cubic_type->tp_alloc(tabulated_real_cubic_type, 0));
        if (!source) {
            Py_DECREF(result);
            Py_DECREF(breaks);
            Py_DECREF(coefficients);
            return nullptr;
        }
        Py_INCREF(breaks);
        source->breaks = breaks;
        Py_INCREF(coefficients);
        source->coefficients = coefficients;
        source->channel = channel;
        PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(channel),
                         reinterpret_cast<PyObject *>(source));
    }
    Py_DECREF(breaks);
    Py_DECREF(coefficients);
    return result;
}

typedef struct {
    PyObject_HEAD PyArrayObject *knots;
    PyArrayObject *coefficients;
    npy_intp channel;
    double time_shift;
} TabulatedBSplineObject;

static void tabulated_bspline_dealloc(PyObject *self_obj) {
    TabulatedBSplineObject *self = reinterpret_cast<TabulatedBSplineObject *>(self_obj);
    PyTypeObject *type = Py_TYPE(self_obj);
    Py_XDECREF(self->knots);
    Py_XDECREF(self->coefficients);
    type->tp_free(self_obj);
    Py_DECREF(type);
}

static PyObject *tabulated_bspline_new(PyTypeObject *, PyObject *, PyObject *) {
    PyErr_SetString(PyExc_TypeError,
                    "TabulatedBSpline cannot be instantiated directly; use "
                    "create_tabulated_bspline_bank");
    return nullptr;
}

static double bspline_coefficient_value(double value) { return value; }

static std::complex<double> bspline_coefficient_value(npy_cdouble value) {
    return npy_to_complex(value);
}

template <typename Scalar, typename StoredScalar>
static Scalar evaluate_cubic_bspline(double time, const double *knots,
                                     npy_intp coefficient_count,
                                     const StoredScalar *coefficients,
                                     npy_intp channel_count, npy_intp channel) {
    constexpr npy_intp degree = 3;
    npy_intp span;
    if (time == knots[coefficient_count]) {
        span = coefficient_count - 1;
    } else {
        const double *upper =
            std::upper_bound(knots + degree, knots + coefficient_count + 1, time);
        span = static_cast<npy_intp>(upper - knots) - 1;
    }

    Scalar work[degree + 1];
    for (npy_intp index = 0; index <= degree; ++index) {
        work[index] = bspline_coefficient_value(
            coefficients[(span - degree + index) * channel_count + channel]);
    }
    for (npy_intp level = 1; level <= degree; ++level) {
        for (npy_intp index = degree; index >= level; --index) {
            const npy_intp left_index = span - degree + index;
            const npy_intp right_index = span + 1 + index - level;
            const double alpha =
                (time - knots[left_index]) / (knots[right_index] - knots[left_index]);
            work[index] = (1.0 - alpha) * work[index - 1] + alpha * work[index];
        }
    }
    return work[degree];
}

static PyObject *tabulated_bspline_call(PyObject *self_obj, PyObject *args,
                                        PyObject *kwargs) {
    if (kwargs && PyDict_Size(kwargs) != 0) {
        PyErr_SetString(PyExc_TypeError,
                        "tabulated B-spline source accepts no keyword arguments");
        return nullptr;
    }
    double time = 0.0;
    if (!PyArg_ParseTuple(args, "d", &time)) {
        return nullptr;
    }
    TabulatedBSplineObject *self = reinterpret_cast<TabulatedBSplineObject *>(self_obj);
    const double *knots = reinterpret_cast<const double *>(PyArray_DATA(self->knots));
    const npy_intp coefficient_count = PyArray_DIM(self->coefficients, 0);
    if (!normalize_tabulated_time(time, knots[3], knots[coefficient_count],
                                  self->time_shift, time)) {
        return PyComplex_FromDoubles(0.0, 0.0);
    }
    const npy_intp channel_count = PyArray_DIM(self->coefficients, 1);
    const npy_cdouble *coefficient_data =
        reinterpret_cast<const npy_cdouble *>(PyArray_DATA(self->coefficients));
    const std::complex<double> value = evaluate_cubic_bspline<std::complex<double>>(
        time, knots, coefficient_count, coefficient_data, channel_count, self->channel);
    return PyComplex_FromDoubles(value.real(), value.imag());
}

static PyType_Slot tabulated_bspline_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(tabulated_bspline_dealloc)},
    {Py_tp_new, reinterpret_cast<void *>(tabulated_bspline_new)},
    {Py_tp_call, reinterpret_cast<void *>(tabulated_bspline_call)},
    {0, nullptr},
};

static PyType_Spec tabulated_bspline_spec = {
    "tama.native_sampler.TabulatedBSpline",
    sizeof(TabulatedBSplineObject),
    0,
    Py_TPFLAGS_DEFAULT,
    tabulated_bspline_slots,
};

static PyTypeObject *tabulated_bspline_type = nullptr;

typedef struct {
    PyObject_HEAD PyArrayObject *knots;
    PyArrayObject *coefficients;
    npy_intp channel;
    double time_shift;
} TabulatedRealBSplineObject;

static void tabulated_real_bspline_dealloc(PyObject *self_obj) {
    TabulatedRealBSplineObject *self =
        reinterpret_cast<TabulatedRealBSplineObject *>(self_obj);
    PyTypeObject *type = Py_TYPE(self_obj);
    Py_XDECREF(self->knots);
    Py_XDECREF(self->coefficients);
    type->tp_free(self_obj);
    Py_DECREF(type);
}

static PyObject *tabulated_real_bspline_new(PyTypeObject *, PyObject *, PyObject *) {
    PyErr_SetString(PyExc_TypeError,
                    "TabulatedRealBSpline cannot be instantiated directly; use "
                    "create_tabulated_real_bspline_bank");
    return nullptr;
}

static PyObject *tabulated_real_bspline_call(PyObject *self_obj, PyObject *args,
                                             PyObject *kwargs) {
    if (kwargs && PyDict_Size(kwargs) != 0) {
        PyErr_SetString(PyExc_TypeError,
                        "tabulated real B-spline source accepts no keyword arguments");
        return nullptr;
    }
    double time = 0.0;
    if (!PyArg_ParseTuple(args, "d", &time)) {
        return nullptr;
    }
    TabulatedRealBSplineObject *self =
        reinterpret_cast<TabulatedRealBSplineObject *>(self_obj);
    const double *knots = reinterpret_cast<const double *>(PyArray_DATA(self->knots));
    const npy_intp coefficient_count = PyArray_DIM(self->coefficients, 0);
    if (!normalize_tabulated_time(time, knots[3], knots[coefficient_count],
                                  self->time_shift, time)) {
        return PyFloat_FromDouble(0.0);
    }
    const npy_intp channel_count = PyArray_DIM(self->coefficients, 1);
    const double *coefficient_data =
        reinterpret_cast<const double *>(PyArray_DATA(self->coefficients));
    return PyFloat_FromDouble(
        evaluate_cubic_bspline<double>(time, knots, coefficient_count, coefficient_data,
                                       channel_count, self->channel));
}

static PyType_Slot tabulated_real_bspline_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(tabulated_real_bspline_dealloc)},
    {Py_tp_new, reinterpret_cast<void *>(tabulated_real_bspline_new)},
    {Py_tp_call, reinterpret_cast<void *>(tabulated_real_bspline_call)},
    {0, nullptr},
};

static PyType_Spec tabulated_real_bspline_spec = {
    "tama.native_sampler.TabulatedRealBSpline",
    sizeof(TabulatedRealBSplineObject),
    0,
    Py_TPFLAGS_DEFAULT,
    tabulated_real_bspline_slots,
};

static PyTypeObject *tabulated_real_bspline_type = nullptr;

template <typename ObjectType>
static PyObject *create_tabulated_bspline_bank_impl(PyObject *args, int numpy_type,
                                                    PyTypeObject *source_type,
                                                    const char *factory_name) {
    PyObject *knots_obj = nullptr;
    PyObject *coefficients_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &knots_obj, &coefficients_obj)) {
        return nullptr;
    }
    PyArrayObject *knots = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(knots_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!knots) {
        return nullptr;
    }
    PyArrayObject *coefficients = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(coefficients_obj, numpy_type, NPY_ARRAY_IN_ARRAY));
    if (!coefficients) {
        Py_DECREF(knots);
        return nullptr;
    }
    const bool shape_ok = PyArray_NDIM(knots) == 1 && PyArray_NDIM(coefficients) == 2 &&
                          PyArray_DIM(coefficients, 0) >= 4 &&
                          PyArray_DIM(coefficients, 1) >= 1 &&
                          PyArray_DIM(knots, 0) == PyArray_DIM(coefficients, 0) + 4;
    if (!shape_ok) {
        Py_DECREF(knots);
        Py_DECREF(coefficients);
        PyErr_Format(PyExc_ValueError,
                     "%s requires knots shape (n+4,) and coefficients shape "
                     "(n, channels), with n >= 4",
                     factory_name);
        return nullptr;
    }
    const double *knot_data = reinterpret_cast<const double *>(PyArray_DATA(knots));
    for (npy_intp index = 0; index < PyArray_DIM(knots, 0); ++index) {
        if (!std::isfinite(knot_data[index])) {
            Py_DECREF(knots);
            Py_DECREF(coefficients);
            PyErr_SetString(PyExc_ValueError, "knots must be finite");
            return nullptr;
        }
    }
    const npy_intp coefficient_count = PyArray_DIM(coefficients, 0);
    if (knot_data[0] != knot_data[3] || knot_data[1] != knot_data[3] ||
        knot_data[2] != knot_data[3] ||
        knot_data[coefficient_count + 1] != knot_data[coefficient_count] ||
        knot_data[coefficient_count + 2] != knot_data[coefficient_count] ||
        knot_data[coefficient_count + 3] != knot_data[coefficient_count]) {
        Py_DECREF(knots);
        Py_DECREF(coefficients);
        PyErr_SetString(PyExc_ValueError,
                        "cubic B-spline knots require clamped endpoint multiplicity");
        return nullptr;
    }
    for (npy_intp index = 4; index <= coefficient_count; ++index) {
        if (!(knot_data[index] > knot_data[index - 1])) {
            Py_DECREF(knots);
            Py_DECREF(coefficients);
            PyErr_SetString(PyExc_ValueError,
                            "cubic B-spline support knots must be strictly increasing");
            return nullptr;
        }
    }

    const npy_intp channel_count = PyArray_DIM(coefficients, 1);
    PyObject *result = PyTuple_New(static_cast<Py_ssize_t>(channel_count));
    if (!result) {
        Py_DECREF(knots);
        Py_DECREF(coefficients);
        return nullptr;
    }
    for (npy_intp channel = 0; channel < channel_count; ++channel) {
        ObjectType *source =
            reinterpret_cast<ObjectType *>(source_type->tp_alloc(source_type, 0));
        if (!source) {
            Py_DECREF(result);
            Py_DECREF(knots);
            Py_DECREF(coefficients);
            return nullptr;
        }
        Py_INCREF(knots);
        source->knots = knots;
        Py_INCREF(coefficients);
        source->coefficients = coefficients;
        source->channel = channel;
        source->time_shift = 0.0;
        PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(channel),
                         reinterpret_cast<PyObject *>(source));
    }
    Py_DECREF(knots);
    Py_DECREF(coefficients);
    return result;
}

static PyObject *create_tabulated_bspline_bank(PyObject *, PyObject *args) {
    return create_tabulated_bspline_bank_impl<TabulatedBSplineObject>(
        args, NPY_COMPLEX128, tabulated_bspline_type, "create_tabulated_bspline_bank");
}

static PyObject *create_tabulated_real_bspline_bank(PyObject *, PyObject *args) {
    return create_tabulated_bspline_bank_impl<TabulatedRealBSplineObject>(
        args, NPY_DOUBLE, tabulated_real_bspline_type,
        "create_tabulated_real_bspline_bank");
}

static PyObject *shift_tabulated_bspline(PyObject *, PyObject *args) {
    PyObject *source_obj = nullptr;
    double time_shift = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &source_obj, &time_shift)) {
        return nullptr;
    }
    if (!std::isfinite(time_shift)) {
        PyErr_SetString(PyExc_ValueError, "B-spline time shift must be finite");
        return nullptr;
    }

    if (PyObject_TypeCheck(source_obj, tabulated_bspline_type)) {
        TabulatedBSplineObject *source =
            reinterpret_cast<TabulatedBSplineObject *>(source_obj);
        TabulatedBSplineObject *result = reinterpret_cast<TabulatedBSplineObject *>(
            tabulated_bspline_type->tp_alloc(tabulated_bspline_type, 0));
        if (!result) {
            return nullptr;
        }
        Py_INCREF(source->knots);
        result->knots = source->knots;
        Py_INCREF(source->coefficients);
        result->coefficients = source->coefficients;
        result->channel = source->channel;
        result->time_shift = source->time_shift + time_shift;
        return reinterpret_cast<PyObject *>(result);
    }
    if (PyObject_TypeCheck(source_obj, tabulated_real_bspline_type)) {
        TabulatedRealBSplineObject *source =
            reinterpret_cast<TabulatedRealBSplineObject *>(source_obj);
        TabulatedRealBSplineObject *result =
            reinterpret_cast<TabulatedRealBSplineObject *>(
                tabulated_real_bspline_type->tp_alloc(tabulated_real_bspline_type, 0));
        if (!result) {
            return nullptr;
        }
        Py_INCREF(source->knots);
        result->knots = source->knots;
        Py_INCREF(source->coefficients);
        result->coefficients = source->coefficients;
        result->channel = source->channel;
        result->time_shift = source->time_shift + time_shift;
        return reinterpret_cast<PyObject *>(result);
    }
    PyErr_SetString(PyExc_TypeError,
                    "source must be a native tabulated B-spline callable");
    return nullptr;
}

template <PyCFunction Function>
static PyObject *native_method_boundary(PyObject *self, PyObject *args) noexcept {
    try {
        return Function(self, args);
    } catch (const std::bad_alloc &) {
        return PyErr_NoMemory();
    } catch (const std::length_error &exc) {
        PyErr_SetString(PyExc_OverflowError, exc.what());
        return nullptr;
    } catch (const std::overflow_error &exc) {
        PyErr_SetString(PyExc_OverflowError, exc.what());
        return nullptr;
    } catch (const std::exception &exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "unknown C++ exception in the TAMA native sampler");
        return nullptr;
    }
}

static PyMethodDef TamaNativeSamplerMethods[] = {
    {
        "_native_design_flat_index_for_testing",
        native_method_boundary<native_design_flat_index_for_testing>,
        METH_VARARGS,
        "Compute one checked native design-grid flat index for unit tests.",
    },
    {
        "_complex_mpi_double_count_for_testing",
        native_method_boundary<complex_mpi_double_count_for_testing>,
        METH_VARARGS,
        "Validate a complex-to-MPI_DOUBLE element count for unit tests.",
    },
    {
        "_reduce_grid_sum_inplace_for_testing",
        native_method_boundary<reduce_grid_sum_inplace_for_testing>,
        METH_VARARGS,
        "Run the in-place reducer with a bounded MPI count for unit tests.",
    },
    {
        "_raise_bad_alloc_for_testing",
        native_method_boundary<raise_bad_alloc_for_testing>,
        METH_NOARGS,
        "Raise and translate a synthetic C++ allocation failure for unit tests.",
    },
    {
        "_support_mask_summary_for_testing",
        native_method_boundary<support_mask_summary_for_testing>,
        METH_VARARGS,
        "Build and inspect a synthetic support-rank mask for unit tests.",
    },
    {
        "_synchronize_native_adjoint_exception_for_testing",
        native_method_boundary<synchronize_native_adjoint_exception_for_testing>,
        METH_VARARGS,
        "Synchronize an injected exception across the active Meep process group.",
    },
    {
        "_check_native_adjoint_signal_for_testing",
        native_method_boundary<check_native_adjoint_signal_for_testing>,
        METH_NOARGS,
        "Set and process pending SIGINT at a native adjoint signal checkpoint.",
    },
    {
        "fold_near2far_sources",
        native_method_boundary<fold_near2far_sources>,
        METH_VARARGS,
        "Fold the near-to-far transpose into canonical Mirror source nodes.",
    },
    {
        "configure_native_material_operator",
        native_method_boundary<configure_native_material_operator>,
        METH_VARARGS,
        "Install the averaged tensor dielectric operator before fields are constructed.",
    },
    {
        "create_native_design_plan",
        native_method_boundary<create_native_design_plan>,
        METH_VARARGS,
        "Create a rank-local exact-Yee sampling and MaterialGrid-transpose plan.",
    },
    {
        "native_design_plan_local_size",
        native_method_boundary<native_design_plan_local_size>,
        METH_VARARGS,
        "Return the number of rank-local native Yee points in a design plan.",
    },
    {
        "native_design_plan_signature",
        native_method_boundary<native_design_plan_signature>,
        METH_VARARGS,
        "Return deterministic rank-local integer Yee coordinates for a design plan.",
    },
    {
        "sample_native_design_plan_into",
        native_method_boundary<sample_native_design_plan_into>,
        METH_VARARGS,
        "Sample exact rank-local native Yee values into a complex vector.",
    },
    {
        "sample_native_design_plan_real_into",
        native_method_boundary<sample_native_design_plan_real_into>,
        METH_VARARGS,
        "Sample real rank-local native Yee values into a float64 vector.",
    },
    {
        "accumulate_native_design_product_local_inplace",
        native_method_boundary<accumulate_native_design_product_local_inplace>,
        METH_VARARGS,
        "Accumulate an exact-Yee product through the MaterialGrid transpose.",
    },
    {
        "accumulate_native_design_real_product_local_inplace",
        native_method_boundary<accumulate_native_design_real_product_local_inplace>,
        METH_VARARGS,
        "Accumulate a real exact-Yee product into a float64 design gradient.",
    },
    {
        "accumulate_native_design_midpoint_product_local_inplace",
        native_method_boundary<accumulate_native_design_midpoint_product_local_inplace>,
        METH_VARARGS,
        "Accumulate with the midpoint of consecutive native adjoint fields.",
    },
    {
        "accumulate_native_design_real_midpoint_product_local_inplace",
        native_method_boundary<
            accumulate_native_design_real_midpoint_product_local_inplace>,
        METH_VARARGS,
        "Accumulate real fields with the midpoint of consecutive adjoint samples.",
    },
    {
        "native_forward_step_count",
        native_method_boundary<native_forward_step_count>,
        METH_VARARGS,
        "Return Meep's rounded-time step count for a forward run duration.",
    },
    {
        "run_native_forward_segment",
        native_method_boundary<run_native_forward_segment>,
        METH_VARARGS,
        "Sample monitor and design histories while advancing Meep fields "
        "without a per-step Python control loop.",
    },
    {
        "run_native_design_adjoint_segment",
        native_method_boundary<run_native_design_adjoint_segment>,
        METH_VARARGS,
        "Reconstruct forward derivatives, accumulate native design products, "
        "and advance Meep fields without a per-step Python control loop.",
    },
    {
        "create_component_grid_plan",
        native_method_boundary<create_component_grid_plan>,
        METH_VARARGS,
        "Precompute rank-local interpolation support for a Meep field component over xs x ys.",
    },
    {
        "create_component_point_plan",
        native_method_boundary<create_component_point_plan>,
        METH_VARARGS,
        "Precompute rank-local interpolation support for paired monitor coordinates.",
    },
    {
        "sample_component_point_plan_allreduced",
        native_method_boundary<sample_component_point_plan_allreduced>,
        METH_VARARGS,
        "Sample paired monitor coordinates with one packed MPI all-rank sum.",
    },
    {
        "component_point_plan_indexed_stencil",
        native_method_boundary<component_point_plan_indexed_stencil>,
        METH_VARARGS,
        "Return rank-local point-monitor transpose indices and amplitudes.",
    },
    {
        "configure_component_point_plan_history",
        native_method_boundary<configure_component_point_plan_history>,
        METH_VARARGS,
        "Restrict point-monitor history sampling to configured point indices.",
    },
    {
        "populate_sourcedata",
        native_method_boundary<populate_sourcedata>,
        METH_VARARGS,
        "Populate one Meep sourcedata object with an exact local field index.",
    },
    {
        "merge_sourcedata",
        native_method_boundary<merge_sourcedata>,
        METH_VARARGS,
        "Merge singleton sourcedata indices sharing one component and chunk.",
    },
    {
        "sample_component_point_plan_local_into",
        native_method_boundary<sample_component_point_plan_local_into>,
        METH_VARARGS,
        "Sample rank-local paired monitor contributions into a caller-provided array.",
    },
    {
        "sample_component_point_plan_local_real_into",
        native_method_boundary<sample_component_point_plan_local_real_into>,
        METH_VARARGS,
        "Sample real rank-local monitor contributions into a float64 array.",
    },
    {
        "create_eigenmode_overlap_plan",
        native_method_boundary<create_eigenmode_overlap_plan>,
        METH_VARARGS,
        "Bind component point plans and fixed weights for a two-channel eigenmode overlap.",
    },
    {
        "sample_eigenmode_overlap_plan_local_into",
        native_method_boundary<sample_eigenmode_overlap_plan_local_into>,
        METH_VARARGS,
        "Accumulate rank-local electric and magnetic eigenmode overlaps into a complex array.",
    },
    {
        "sample_component_grid_plan_allreduced",
        native_method_boundary<sample_component_grid_plan_allreduced>,
        METH_VARARGS,
        "Sample a Meep field component using a precomputed rank-local plan and one MPI all-rank sum.",
    },
    {
        "component_grid_plan_local_complete_mask",
        native_method_boundary<component_grid_plan_local_complete_mask>,
        METH_VARARGS,
        "Return a boolean mask for sample points whose interpolation support is fully local to this rank.",
    },
    {
        "component_grid_plan_local_boundary_mask",
        native_method_boundary<component_grid_plan_local_boundary_mask>,
        METH_VARARGS,
        "Return a boolean mask for boundary sample points whose interpolation support includes this rank.",
    },
    {
        "configure_component_grid_plan_history",
        native_method_boundary<configure_component_grid_plan_history>,
        METH_VARARGS,
        "Cache fixed local and boundary history indices and communication buffers in a sample plan.",
    },
    {
        "sample_component_grid_plan_history_into",
        native_method_boundary<sample_component_grid_plan_history_into>,
        METH_VARARGS,
        "Sample a configured history row directly into a caller-provided complex array.",
    },
    {
        "configure_component_grid_plan_accumulation",
        native_method_boundary<configure_component_grid_plan_accumulation>,
        METH_VARARGS,
        "Cache fixed flat indices for repeated plan-based gradient accumulation.",
    },
    {
        "sample_component_grid_plan_points_local",
        native_method_boundary<sample_component_grid_plan_points_local>,
        METH_VARARGS,
        "Sample selected flat point indices from the rank-local part of a precomputed plan.",
    },
    {
        "sample_component_grid_plan_points_allreduced",
        native_method_boundary<sample_component_grid_plan_points_allreduced>,
        METH_VARARGS,
        "Sample selected flat point indices from a precomputed plan and MPI-sum them across ranks.",
    },
    {
        "sample_component_grid_plan_points_support_reduced",
        native_method_boundary<sample_component_grid_plan_points_support_reduced>,
        METH_VARARGS,
        "Sample selected flat boundary point indices and reduce only across their support ranks.",
    },
    {
        "sample_component_grid",
        native_method_boundary<sample_component_grid>,
        METH_VARARGS,
        "Sample a Meep field component over xs x ys using meep::fields::get_field.",
    },
    {
        "sample_component_grid_allreduced",
        native_method_boundary<sample_component_grid_allreduced>,
        METH_VARARGS,
        "Sample a Meep field component using rank-local chunks and one MPI all-rank sum.",
    },
    {
        "accumulate_component_product_allreduced",
        native_method_boundary<accumulate_component_product_allreduced>,
        METH_VARARGS,
        "Accumulate field-component times a complex grid using rank-local chunks and one MPI all-rank sum.",
    },
    {
        "accumulate_component_product_local_inplace",
        native_method_boundary<accumulate_component_product_local_inplace>,
        METH_VARARGS,
        "Accumulate field-component times a complex grid into a rank-local accumulator without MPI reduction.",
    },
    {
        "accumulate_component_product_plan_local_inplace",
        native_method_boundary<accumulate_component_product_plan_local_inplace>,
        METH_VARARGS,
        "Accumulate field-component times a complex grid with a precomputed rank-local plan and no MPI reduction.",
    },
    {
        "accumulate_component_product_plan_points_local_inplace",
        native_method_boundary<accumulate_component_product_plan_points_local_inplace>,
        METH_VARARGS,
        "Accumulate selected field-component products with a precomputed rank-local plan and no MPI reduction.",
    },
    {
        "accumulate_component_product_plan_configured_local_inplace",
        native_method_boundary<
            accumulate_component_product_plan_configured_local_inplace>,
        METH_VARARGS,
        "Accumulate configured selected field-component products without reparsing indices.",
    },
    {
        "accumulate_component_difference_product_plan_configured_local_inplace",
        native_method_boundary<
            accumulate_component_difference_product_plan_configured_local_inplace>,
        METH_VARARGS,
        "Accumulate configured forward values times the current-minus-previous field derivative.",
    },
    {
        "reduce_complex_grid_sum",
        native_method_boundary<reduce_complex_grid_sum>,
        METH_VARARGS,
        "Sum a complex grid across the active Meep process group.",
    },
    {
        "reduce_complex_grid_sum_inplace",
        native_method_boundary<reduce_complex_grid_sum_inplace>,
        METH_VARARGS,
        "Sum a writable complex grid in place across the active Meep process group.",
    },
    {
        "reduce_real_grid_sum_inplace",
        native_method_boundary<reduce_real_grid_sum_inplace>,
        METH_VARARGS,
        "Sum a writable real grid in place across the active Meep process group.",
    },
    {
        "create_tabulated_cubic",
        native_method_boundary<create_tabulated_cubic>,
        METH_VARARGS,
        "Create a native callable from piecewise cubic complex coefficients.",
    },
    {
        "create_tabulated_real_cubic_bank",
        native_method_boundary<create_tabulated_real_cubic_bank>,
        METH_VARARGS,
        "Create shared native callables from piecewise cubic real coefficients.",
    },
    {
        "create_tabulated_bspline_bank",
        native_method_boundary<create_tabulated_bspline_bank>,
        METH_VARARGS,
        "Create shared native complex cubic B-spline callables.",
    },
    {
        "create_tabulated_real_bspline_bank",
        native_method_boundary<create_tabulated_real_bspline_bank>,
        METH_VARARGS,
        "Create shared native real cubic B-spline callables.",
    },
    {
        "shift_tabulated_bspline",
        native_method_boundary<shift_tabulated_bspline>,
        METH_VARARGS,
        "Create a time-shifted view of a native tabulated B-spline.",
    },
    {
        "sample_ez_grid",
        native_method_boundary<sample_component_grid>,
        METH_VARARGS,
        "Compatibility alias for sample_component_grid.",
    },
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef native_sampler_module = {
    PyModuleDef_HEAD_INIT,
    "native_sampler",
    "Native batch sampler for Meep field points.",
    -1,
    TamaNativeSamplerMethods,
};

PyMODINIT_FUNC PyInit_native_sampler(void) {
    import_array();
    PyObject *module = PyModule_Create(&native_sampler_module);
    if (!module) {
        return nullptr;
    }
    if (PyModule_AddIntConstant(module, "API_VERSION", 15) < 0) {
        Py_DECREF(module);
        return nullptr;
    }
    PyObject *type_obj = PyType_FromSpec(&tabulated_cubic_spec);
    if (!type_obj) {
        Py_DECREF(module);
        return nullptr;
    }
    tabulated_cubic_type = reinterpret_cast<PyTypeObject *>(type_obj);
    if (PyModule_AddObject(module, "TabulatedCubic", type_obj) < 0) {
        Py_DECREF(type_obj);
        Py_DECREF(module);
        return nullptr;
    }
    PyObject *real_type_obj = PyType_FromSpec(&tabulated_real_cubic_spec);
    if (!real_type_obj) {
        Py_DECREF(module);
        return nullptr;
    }
    tabulated_real_cubic_type = reinterpret_cast<PyTypeObject *>(real_type_obj);
    if (PyModule_AddObject(module, "TabulatedRealCubic", real_type_obj) < 0) {
        Py_DECREF(real_type_obj);
        Py_DECREF(module);
        return nullptr;
    }
    PyObject *bspline_type_obj = PyType_FromSpec(&tabulated_bspline_spec);
    if (!bspline_type_obj) {
        Py_DECREF(module);
        return nullptr;
    }
    tabulated_bspline_type = reinterpret_cast<PyTypeObject *>(bspline_type_obj);
    if (PyModule_AddObject(module, "TabulatedBSpline", bspline_type_obj) < 0) {
        Py_DECREF(bspline_type_obj);
        Py_DECREF(module);
        return nullptr;
    }
    PyObject *real_bspline_type_obj = PyType_FromSpec(&tabulated_real_bspline_spec);
    if (!real_bspline_type_obj) {
        Py_DECREF(module);
        return nullptr;
    }
    tabulated_real_bspline_type =
        reinterpret_cast<PyTypeObject *>(real_bspline_type_obj);
    if (PyModule_AddObject(module, "TabulatedRealBSpline", real_bspline_type_obj) < 0) {
        Py_DECREF(real_bspline_type_obj);
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
