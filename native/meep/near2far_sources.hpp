// Included after native Mirror and active-Meep-communicator helpers.
#ifndef TAMA_NEAR2FAR_SOURCES_HPP
#define TAMA_NEAR2FAR_SOURCES_HPP

#include <tuple>
#include <set>

using NativeSourceNode = std::tuple<int, int, ptrdiff_t>;
using NativeSourceNodes = std::map<NativeSourceNode, std::vector<std::complex<double>>>;

static void route_native_source_nodes(meep::fields *fields, NativeSourceNodes &nodes,
                                      size_t nfreq) {
    const int nproc = meep::count_processors();
    bool remote = false;
    for (const auto &entry : nodes)
        remote = remote || !fields->chunks[std::get<1>(entry.first)]->is_mine();
    if (nproc > 1 && meep::sum_to_all(static_cast<int>(remote))) {
        // Folding a redundant negative half-cell can change chunk ownership.
        // Route only when needed, and stay within the active Meep subgroup.
        MeepGroupCommunicator group = active_meep_group_communicator();
        try {
            if (nodes.size() > static_cast<size_t>(INT_MAX))
                throw std::overflow_error("too many near2far source nodes for MPI");
            const int local_count = static_cast<int>(nodes.size());
            std::vector<int> counts(nproc), meta_counts(nproc), meta_offsets(nproc),
                amp_counts(nproc), amp_offsets(nproc);
            require_mpi_success(MPI_Allgather(&local_count, 1, MPI_INT, counts.data(),
                                              1, MPI_INT, group.comm),
                                "near2far node counts");
            size_t total = 0;
            for (int rank = 0; rank < nproc; ++rank) {
                if ((total + counts[rank]) > static_cast<size_t>(INT_MAX) / 3 ||
                    (total + counts[rank]) > static_cast<size_t>(INT_MAX) / (2 * nfreq))
                    throw std::overflow_error(
                        "near2far source routing exceeds MPI count limits");
                meta_counts[rank] = 3 * counts[rank];
                meta_offsets[rank] = static_cast<int>(3 * total);
                amp_counts[rank] = static_cast<int>(2 * nfreq * counts[rank]);
                amp_offsets[rank] = static_cast<int>(2 * nfreq * total);
                total += counts[rank];
            }
            std::vector<std::int64_t> local_metadata(3 * nodes.size()),
                metadata(3 * total);
            std::vector<double> local_amplitudes(2 * nfreq * nodes.size()),
                amplitudes(2 * nfreq * total);
            size_t row = 0;
            for (const auto &entry : nodes) {
                local_metadata[3 * row] = std::get<0>(entry.first);
                local_metadata[3 * row + 1] = std::get<1>(entry.first);
                local_metadata[3 * row + 2] = std::get<2>(entry.first);
                for (size_t frequency = 0; frequency < nfreq; ++frequency) {
                    local_amplitudes[2 * (row * nfreq + frequency)] =
                        entry.second[frequency].real();
                    local_amplitudes[2 * (row * nfreq + frequency) + 1] =
                        entry.second[frequency].imag();
                }
                ++row;
            }
            require_mpi_success(MPI_Allgatherv(local_metadata.data(), 3 * local_count,
                                               MPI_INT64_T, metadata.data(),
                                               meta_counts.data(), meta_offsets.data(),
                                               MPI_INT64_T, group.comm),
                                "near2far source indices");
            require_mpi_success(
                MPI_Allgatherv(local_amplitudes.data(),
                               static_cast<int>(2 * nfreq * local_count), MPI_DOUBLE,
                               amplitudes.data(), amp_counts.data(), amp_offsets.data(),
                               MPI_DOUBLE, group.comm),
                "near2far source amplitudes");
            nodes.clear();
            for (size_t index = 0; index < total; ++index) {
                const int chunk = static_cast<int>(metadata[3 * index + 1]);
                if (!fields->chunks[chunk]->is_mine())
                    continue;
                const NativeSourceNode key(static_cast<int>(metadata[3 * index]), chunk,
                                           metadata[3 * index + 2]);
                auto &values = nodes[key];
                if (values.empty())
                    values.resize(nfreq, 0.0);
                for (size_t frequency = 0; frequency < nfreq; ++frequency)
                    values[frequency] += std::complex<double>(
                        amplitudes[2 * (index * nfreq + frequency)],
                        amplitudes[2 * (index * nfreq + frequency) + 1]);
            }
        } catch (...) {
            if (group.owned)
                free_owned_communicator(group.comm);
            throw;
        }
        if (group.owned)
            free_owned_communicator(group.comm);
    }
}

static PyObject *fold_near2far_sources(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0, monitor_addr = 0;
    PyObject *addresses_obj = nullptr;
    if (!PyArg_ParseTuple(args, "KKO", &fields_addr, &monitor_addr, &addresses_obj))
        return nullptr;
    if (!fields_addr || !monitor_addr) {
        PyErr_SetString(PyExc_ValueError,
                        "fields and near2far pointers must be nonzero");
        return nullptr;
    }
    PyObject *addresses =
        PySequence_Fast(addresses_obj, "source addresses must be a sequence");
    if (!addresses)
        return nullptr;
    auto *fields =
        reinterpret_cast<meep::fields *>(static_cast<uintptr_t>(fields_addr));
    auto *monitor =
        reinterpret_cast<meep::dft_near2far *>(static_cast<uintptr_t>(monitor_addr));
    const bool cylindrical = fields->gv.dim == meep::Dcyl;
    const size_t nfreq = monitor->freq.size();
    NativeSourceNodes nodes;
    std::string local_error;
    try {
        size_t source_index = 0;
        for (auto *dft = monitor->F; dft; dft = dft->next_in_dft, ++source_index) {
            if (source_index >=
                static_cast<size_t>(PySequence_Fast_GET_SIZE(addresses)))
                throw std::runtime_error(
                    "near2far source count differs from DFT chunk count");
            const auto address = PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(
                addresses, static_cast<Py_ssize_t>(source_index)));
            if (PyErr_Occurred()) {
                PyErr_Clear();
                throw std::runtime_error("near2far source pointers must be integers");
            }
            if (!address)
                throw std::runtime_error("near2far source pointer must be nonzero");
            const auto *data = reinterpret_cast<const meep::sourcedata *>(
                static_cast<uintptr_t>(address));
            if (!nfreq || data->amp_arr.size() != data->idx_arr.size() * nfreq ||
                data->near_fd_comp != dft->c || data->fc_idx != dft->fc->chunk_idx)
                throw std::runtime_error(
                    "near2far source metadata differs from its DFT chunk");
            size_t node_index = 0;
            LOOP_OVER_IVECS(dft->fc->gv, dft->is, dft->ie, idx) {
                IVEC_LOOP_ILOC(dft->fc->gv, original_location);
                if (node_index >= data->idx_arr.size() ||
                    data->idx_arr[node_index] != idx)
                    throw std::runtime_error(
                        "near2far source indices differ from DFT iteration order");
                const size_t amplitude_offset = node_index++ * nfreq;
                meep::component component = data->near_fd_comp;
                meep::ivec location = original_location;
                // Undo exactly the multiplicity used by Meep near_sourcedata,
                // then transpose the DFT symmetry and canonical Yee projection.
                std::complex<double> phase(1.0, 0.0);
                if (!fields->locate_component_point(&component, &location, &phase) ||
                    !fold_native_mirror_point(fields, component, location, phase))
                    continue;
                phase *= dft->S.phase_shift(dft->c, dft->sn) *
                         static_cast<double>(dft->S.multiplicity(original_location));
                if (cylindrical) {
                    IVEC_LOOP_LOC(dft->fc->gv, source_location);
                    source_location = dft->S.transform(source_location, dft->sn) +
                                      meep::vec(dft->shift * (0.5 * dft->fc->gv.inva));
                    // Undo near_sourcedata's extra 1/r. The NFF quadrature already
                    // contains 2*pi*r, and its axis row has zero weight.
                    if (source_location.r() != 0.0)
                        phase *= source_location.r();
                }
                const double measure = native_mirror_measure(fields, location);
                if (!(measure > 0.0))
                    throw std::runtime_error(
                        "folded near2far source has zero Mirror measure");
                bool found = false;
                std::set<NativeSourceNode> images;
                for (int symmetry = 0; symmetry < fields->S.multiplicity();
                     ++symmetry) {
                    const meep::ivec image = fields->S.transform(location, symmetry);
                    // Normal Yee components may own both sides of a Mirror plane.
                    // Their redundant negative half-cell still needs source injection.
                    if (symmetry && native_mirror_measure(fields, image) > 0.0)
                        continue;
                    const auto image_component =
                        fields->S.transform(component, symmetry);
                    const auto image_phase = fields->S.phase_shift(component, symmetry);
                    for (int chunk_index = 0; chunk_index < fields->num_chunks;
                         ++chunk_index) {
                        auto *chunk = fields->chunks[chunk_index];
                        if (!chunk || !chunk->gv.owns(image))
                            continue;
                        const auto index = chunk->gv.index(image_component, image);
                        const double volume =
                            cylindrical
                                ? chunk->gv.dV(image_component, index).full_volume()
                                : 1.0;
                        if (!(volume > 0.0))
                            throw std::runtime_error(
                                "near2far source has a non-positive Yee-cell volume");
                        const NativeSourceNode key(static_cast<int>(image_component),
                                                   chunk_index, index);
                        if (images.insert(key).second) {
                            auto &amplitudes = nodes[key];
                            if (amplitudes.empty())
                                amplitudes.resize(nfreq, 0.0);
                            for (size_t frequency = 0; frequency < nfreq; ++frequency)
                                amplitudes[frequency] +=
                                    image_phase * phase *
                                    data->amp_arr[amplitude_offset + frequency] /
                                    (measure * volume);
                        }
                        if (!symmetry)
                            found = true;
                        break;
                    }
                }
                if (!found)
                    throw std::runtime_error(
                        "folded near2far source has no owning chunk");
            }
            if (node_index != data->idx_arr.size())
                throw std::runtime_error(
                    "near2far source index count differs from DFT iteration order");
        }
        if (source_index != static_cast<size_t>(PySequence_Fast_GET_SIZE(addresses)))
            throw std::runtime_error(
                "near2far source count differs from DFT chunk count");
    } catch (const std::exception &error) {
        local_error = error.what();
    }
    Py_DECREF(addresses);
    const int nproc = meep::count_processors();
    if (nproc > 1) {
        const int failures = meep::sum_to_all(static_cast<int>(!local_error.empty()));
        if (failures && local_error.empty())
            local_error = "near2far source folding failed on another rank";
    }
    if (!local_error.empty())
        throw std::runtime_error(local_error);

    route_native_source_nodes(fields, nodes, nfreq);

    PyObject *result = PyList_New(0);
    if (!result)
        return nullptr;
    for (auto begin = nodes.begin(); begin != nodes.end();) {
        auto end = begin;
        const int component = std::get<0>(begin->first),
                  chunk = std::get<1>(begin->first);
        while (end != nodes.end() && std::get<0>(end->first) == component &&
               std::get<1>(end->first) == chunk)
            ++end;
        const npy_intp count = static_cast<npy_intp>(std::distance(begin, end));
        npy_intp shape[2] = {count, static_cast<npy_intp>(nfreq)};
        PyObject *indices = PyArray_SimpleNew(1, shape, NPY_INTP);
        PyObject *amplitudes = PyArray_SimpleNew(2, shape, NPY_COMPLEX128);
        if (!indices || !amplitudes) {
            Py_XDECREF(indices);
            Py_XDECREF(amplitudes);
            Py_DECREF(result);
            return nullptr;
        }
        auto *index_values = static_cast<npy_intp *>(
            PyArray_DATA(reinterpret_cast<PyArrayObject *>(indices)));
        auto *amplitude_values = static_cast<npy_cdouble *>(
            PyArray_DATA(reinterpret_cast<PyArrayObject *>(amplitudes)));
        size_t row = 0;
        for (auto entry = begin; entry != end; ++entry, ++row) {
            index_values[row] = std::get<2>(entry->first);
            for (size_t frequency = 0; frequency < nfreq; ++frequency)
                set_npy_complex(amplitude_values[row * nfreq + frequency],
                                entry->second[frequency]);
        }
        PyObject *item = Py_BuildValue("iiNN", component, chunk, indices, amplitudes);
        if (!item || PyList_Append(result, item) < 0) {
            Py_XDECREF(item);
            Py_DECREF(result);
            return nullptr;
        }
        Py_DECREF(item);
        begin = end;
    }
    return result;
}

#endif
