"""
Main file from which the shape optimization process can be started.
The shape optimization problem parameters and data can be configured using a file to be named problem_data.py.
See the usage in the section "Solution setup" of this script.
Results are automatically saved on disk.

Press CTRL + C to stop the shape optimization process and still save the results up to the stopping point.

The exact mesh and PDEs (those used in the synthetic data production, and the ones we aim at reconstructing), can be
cached for later used and faster loading. To do so, when calling problem.initialize_from_data below, set
regenerate_exact_data=False, and, inn the configuration file problem_data.py, set
exact_geometry_dict["domain"]["reload_xdmf"] = True
"""

# %% Imports

from dolfin import *
import logging

from utilities.shape_optimization import ShapeOptimizationProblem

# %% Setting log and global parameters

parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters['allow_extrapolation'] = True  # I want a function to be taken from a mesh to a slightly different one

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M')

set_log_level(LogLevel.ERROR)

runs_path = "./runs/"  # try with absolute path should this not work. Folder containing the various problem folders

# %% Solution setup

run_name = "examplary_run"
problem_path = runs_path + run_name + "/"

problem = ShapeOptimizationProblem()
problem.initialize_from_data(problem_path,
                             regenerate_exact_data=True)  # initialize from the problem_data.py config file
problem.save_exact_data(problem_path)  # save exact geometry (the one to be reconstructed) and exact PDEs, for caching
_, _ = problem.create_cost_functional(start_at_optimum=False)
# problem.do_taylor_test()

# %% Solution and save things to file
problem.solve()
# problem.do_taylor_test()

# %% Results

problem.visualize_result()
problem.save_results_to_file(problem_path)
