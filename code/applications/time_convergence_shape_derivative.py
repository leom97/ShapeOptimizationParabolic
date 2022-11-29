"""
Here I keep the space discretization fix and try to see whether the shape gradient converges, and at which speed, with successive time refinements
The exact PDE is generated with CN and a lot of time steps
"""

from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import numpy as np
import logging

from utilities.pdes import TimeExpressionFromList, HeatEquation
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

# %% Functional set-up

problem_name = "time_convergence_shape_derivatives"
problem_path = runs_path + problem_name + "/"

problem = ShapeOptimizationProblem()

# %% Computing the shape gradient many times

shape_gradients = []
problem.initialize_from_data(problem_path, regenerate_exact_data=True)
problem.save_exact_data(problem_path)

for i in range(5):
    problem.initialize_from_data(problem_path, regenerate_exact_data=False)
    problem.create_cost_functional(start_at_optimum=True, time_steps_factor=2 ** i)
    dj_exact = problem.j.derivative().vector()
    shape_gradients.append(dj_exact)

# %% Results

shape_gradients_np = [g[:] for g in shape_gradients]
shm = np.array(shape_gradients_np)
d = np.diff(shm, axis = 0)
nd = np.max(np.abs(d), axis=1)

print(nd)

# l^\infty norms of difference of shape gradients (doubling the timesteps, implicit euler)
# [2.00161270e-04 8.81790928e-05 4.15806039e-05 2.02226612e-05
#  9.97695624e-06 4.95583080e-06 2.46987221e-06]
# the shape gradient also here seems to be VERY SLOWLY converging to zero

# same thing, but with crank-nicolson. Note: the discretization of the cost functional is done by the rectangle rule!
# [3.21241669e-05 7.67615750e-06 1.87577038e-06 4.63609571e-07]
# the convergebce seems to be at 0, the correct shape gradient

# crank-nicolson + trapezoidal rule
# [2.94335365e-05 7.34005529e-06 1.83377018e-06 4.58362681e-07] (little difference?)

# crank-nicolson + trapezoidal rule + final smoothing (best possible case)