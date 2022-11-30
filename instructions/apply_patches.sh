#!/bin/bash

echo "Insert path to environment site-packages (e.g. \$HOME/anaconda3/envs/thesis_env/venv_39/lib/python3.9/site-packages, without final slash):"
read site_packages_path

cp "fixes/fenics/tools.py" $site_packages_path"/ffc/uflacs/tools.py"
cp "fixes/moola/__init__.py" $site_packages_path"/moola/algorithms/__init__.py"
cp "fixes/moola/custom_bfgs.py" $site_packages_path"/moola/algorithms/custom_bfgs.py"
cp "fixes/moola/optimisation_algorithm.py" $site_packages_path"/moola/algorithms/optimisation_algorithm.py"
cp "fixes/moola/regularized_newton.py" $site_packages_path"/moola/algorithms/regularized_newton.py"