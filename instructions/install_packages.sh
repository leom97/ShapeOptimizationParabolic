#!/bin/bash

conda install -c conda-forge mamba
mamba install -c conda-forge matplotlib
mamba install --yes -c conda-forge fenics
pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@2019.1.0
mamba install -c conda-forge gmsh meshio
pip install pygmsh
pip install scipy
pip install moola
pip install ipython
