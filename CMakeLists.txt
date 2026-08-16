cmake_minimum_required(VERSION 3.20)

set(TAMA_DEPENDENCY_PREFIX "$ENV{MEEP_CONDA_PREFIX}")
if(NOT TAMA_DEPENDENCY_PREFIX)
    set(TAMA_DEPENDENCY_PREFIX "$ENV{CONDA_PREFIX}")
endif()
if(NOT TAMA_DEPENDENCY_PREFIX)
    message(
        FATAL_ERROR
        "No Meep conda environment was selected. Activate the environment "
        "containing pymeep or set MEEP_CONDA_PREFIX."
    )
endif()
if(TAMA_DEPENDENCY_PREFIX AND UNIX)
    set(
        ENV{PATH}
        "${TAMA_DEPENDENCY_PREFIX}/bin:$ENV{PATH}"
    )
endif()
if(TAMA_DEPENDENCY_PREFIX)
    find_program(
        TAMA_CXX_COMPILER
        NAMES c++ g++ clang++
        HINTS "${TAMA_DEPENDENCY_PREFIX}/bin"
        NO_DEFAULT_PATH
    )
endif()
if(TAMA_CXX_COMPILER)
    set(
        CMAKE_CXX_COMPILER
        "${TAMA_CXX_COMPILER}"
        CACHE FILEPATH "C++ compiler from the Meep environment" FORCE
    )
endif()

project(tama_native LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(TAMA_DEPENDENCY_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${TAMA_DEPENDENCY_PREFIX}")
endif()

find_package(
    Python 3.11 REQUIRED
    COMPONENTS Interpreter Development.Module NumPy
)
find_package(MPI REQUIRED COMPONENTS CXX)

find_path(
    MEEP_INCLUDE_DIR
    NAMES meep.hpp
    PATHS "${TAMA_DEPENDENCY_PREFIX}/include"
    NO_DEFAULT_PATH
)
find_library(
    MEEP_LIBRARY
    NAMES meep
    PATHS "${TAMA_DEPENDENCY_PREFIX}/lib"
    NO_DEFAULT_PATH
)
find_library(
    CBLAS_LIBRARY
    NAMES cblas blas
    PATHS "${TAMA_DEPENDENCY_PREFIX}/lib"
    NO_DEFAULT_PATH
)
find_path(
    GSL_CBLAS_INCLUDE_DIR
    NAMES gsl/gsl_cblas.h
    PATHS "${TAMA_DEPENDENCY_PREFIX}/include"
    NO_DEFAULT_PATH
)

if(
    NOT MEEP_INCLUDE_DIR
    OR NOT MEEP_LIBRARY
    OR NOT CBLAS_LIBRARY
    OR NOT GSL_CBLAS_INCLUDE_DIR
)
    message(
        FATAL_ERROR
        "Meep headers, Meep library, CBLAS library, or GSL CBLAS headers "
        "were not found. "
        "Activate the conda environment containing pymeep or set "
        "MEEP_CONDA_PREFIX."
    )
endif()

Python_add_library(
    native_sampler MODULE WITH_SOABI
    src/tama/native_sampler.cpp
)
target_compile_features(native_sampler PRIVATE cxx_std_17)
target_include_directories(
    native_sampler PRIVATE
    "${MEEP_INCLUDE_DIR}"
    "${GSL_CBLAS_INCLUDE_DIR}"
)
target_link_libraries(
    native_sampler PRIVATE
    Python::NumPy
    MPI::MPI_CXX
    "${MEEP_LIBRARY}"
    "${CBLAS_LIBRARY}"
)
set_target_properties(
    native_sampler PROPERTIES INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
)

if(APPLE)
    set_target_properties(
        native_sampler PROPERTIES INSTALL_RPATH "@loader_path/../../.."
    )
elseif(UNIX)
    set_target_properties(
        native_sampler PROPERTIES INSTALL_RPATH "\$ORIGIN/../../.."
    )
endif()

install(TARGETS native_sampler LIBRARY DESTINATION tama)
