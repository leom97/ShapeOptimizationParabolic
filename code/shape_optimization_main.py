"""
Main file from which the shape optimization process can be started.
The shape optimization problem parameters and data can be configured using a file to be named problem_data.py.
See the usage in the section "Solution setup" of this script.
Results are automatically saved on disk.
"""

# %% Imports

from dolfin import *
import logging

from utilities.shape_optimization import ShapeOptimizationProblem

# %% Setting log and global parameters

parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters['allow_extrapolation'] = True  # I want a function to be taken from a mesh to a slightly different one

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M')

set_log_level(LogLevel.ERROR)

runs_path = "./runs/"   # try with absolute path should this not work

# %% Solution setup

problem_name = "thesis_hourglass_constant_2"
problem_path = runs_path + problem_name + "/"

problem = ShapeOptimizationProblem()
problem.initialize_from_data(problem_path, regenerate_exact_data=True)  # initialize from the problem_data.py config file
problem.save_exact_data(problem_path)   # save exact geometry (the one to be reconstructed) and exact PDEs, for caching
_, _ = problem.create_cost_functional(start_at_optimum=False)
# problem.do_taylor_test()

 # %% Solution and save things to file
problem.solve()
# problem.do_taylor_test()

# %% Results

problem.visualize_result()
problem.save_results_to_file(problem_path)