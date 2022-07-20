from dolfin import *
import logging

from utilities.shape_optimization import ShapeOptimizationProblem

# %% Setting log and global parameters

parameters[
    "refinement_algorithm"] = "plaza_with_parent_facets"  # must do this to be able to refine stuff associated with a mesh that is being refined
parameters[
    'allow_extrapolation'] = True  # needed if I want a function to be taken from a mesh to a slightly different one

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M')

set_log_level(LogLevel.ERROR)

# %% Solution


problem_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/code/results/test0/"

problem = ShapeOptimizationProblem()
problem.initialize_from_data(problem_path, method="python")
problem.create_cost_functional()
problem.do_taylor_test()
problem.solve()
problem.visualize_result()
problem.save_results_to_file(problem_path)