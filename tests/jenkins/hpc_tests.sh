#!/bin/bash

# Fail fast
set -e

# GULP
module purge
pytest tests/core/recipes/gulp_recipes --noconftest

# Gaussian
module purge
module load gaussian/g16
pytest tests/core/recipes/gaussian_recipes --noconftest

# ORCA
module purge
module load openmpi/gcc/4.1.2
pytest tests/core/recipes/orca_recipes --noconftest

# VASP
module purge
module load intel/2021.1.2 intel-mpi/intel/2021.3.1 hdf5/intel-2021.1/1.10.6
pytest tests/core/recipes/vasp_recipes/jenkins --noconftest