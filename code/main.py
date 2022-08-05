from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import logging

from utilities.shape_optimization import ShapeOptimizationProblem

# %% Setting log and global parameters

# The refining algortithm is no longer necessary, but let's keep it
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters['allow_extrapolation'] = True  # I want a function to be taken from a mesh to a slightly different one

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M')

set_log_level(LogLevel.ERROR)

runs_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/code/runs/"

# %% Solution set-up

problem_name = "sq_oakley_oscillations_test"
problem_path = runs_path + problem_name + "/"

problem = ShapeOptimizationProblem()
problem.initialize_from_data(problem_path, regenerate_exact_data=True)
problem.save_exact_data(problem_path)
problem.create_cost_functional()
# tape = get_working_tape()
# tape.visualise()
# problem.debug_generic(problem_path)

# %% Solution and save things to file
problem.solve()
problem.do_taylor_test()
problem.visualize_result()
problem.save_results_to_file(problem_path)

# %% Todo
# 2) try elasticity method
# 5) make PDE code faster
# 6) how important is e^-1/t^2 in the initial condition? Does it give use OOC of 2?
# 9) complicate the boundary conditions to highly spatially varying
# 12) implement trapezoidal rule cost function
# 13) implement Stoermer-Verlet
# 15) verify shape derivatives
# 16) try a complicated domain
# 19) do some more error tests with pdes, including initial smoothing, see if OOC2 in time is achievable
# 21) add noise to the working cases, do they still work?