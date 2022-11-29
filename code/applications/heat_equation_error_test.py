from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import logging

from utilities.meshing import AnnulusMesh, CircleMesh
from utilities.pdes import HeatEquation, PDETestProblems

# %% Setting log and global parameters

parameters[
    "refinement_algorithm"] = "plaza_with_parent_facets"  # must do this to be able to refine stuff associated with a mesh that is being refined
parameters[
    'allow_extrapolation'] = True  # needed if I want a function to be taken from a mesh to a slightly different one

# ffc_logger = logging.getLogger('FFC')
# ffc_logger.setLevel(logging.ERROR)
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M')

set_log_level(LogLevel.ERROR)

# %% Error tests

import numpy as np

# Global variables
resolutions = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]  # mesh resolutions

# Test problem
problem_name = "no_source_dirichlet_neumann_zero_compatibility"  # what problem from TestProblems are we going to solve?
problems = PDETestProblems()
problems.override_pickle = False
u_ex, f, u_D, u_N, u0, marker_neumann, marker_dirichlet, T = problems.get_data(problem_name)

# Setting up the PDE class
heq = HeatEquation()
heq.set_PDE_data(u0, source=f, marker_neumann=marker_neumann, marker_dirichlet=marker_dirichlet, neumann_BC=u_N,
                 dirichlet_BC=u_D,
                 exact_solution=u_ex)  # note, no copy is done, the attributes of heq are EXACTLY these guys
heq.set_ODE_scheme("implicit_euler")
heq.verbose = True

# We check that the last error decreases as expected
errors_in_one_time, hs, dts = [], [], []  # errors are only at a certain index
error_index = 2  # this is the index: it means at T/2

# Solve the PDE many times...
for i in range(4):
    annulus = AnnulusMesh(resolution=resolutions[i])
    heq.set_mesh(annulus.mesh, annulus.facet_function)
    hs.append(heq.mesh.hmax())

    if heq.implemented_time_schemes.index(heq.ode_scheme) in [0, 3]:  # implicit and implicit-explicit euler
        N_steps = int(np.ceil(2 / (heq.mesh.hmax() ** 2)))
    elif heq.implemented_time_schemes.index(heq.ode_scheme) == 1:  # crank-nicolson
        N_steps = int(np.ceil(2 / (heq.mesh.hmax() ** 1)))

    logging.info(f"Error level {i} with {N_steps} timesteps")

    heq.set_time_discretization(T, N_steps=N_steps)  # dt = h^2 and N = T/dt
    dts.append(heq.dts[0])

    solution_list, error_list = heq.solve(err_mode="l2")
    current_error = error_list[len(error_list) // 2]
    errors_in_one_time.append(current_error)
    logging.info(f"Error: {current_error}")

errors_in_one_time = np.array(errors_in_one_time)
hs = np.array(hs)
dts = np.array(dts)

# The order of convergence
ooc_space = np.log(errors_in_one_time[1:] / errors_in_one_time[:-1]) / np.log(hs[1:] / hs[:-1])
ooc_time = np.log(errors_in_one_time[1:] / errors_in_one_time[:-1]) / np.log(dts[1:] / dts[:-1])
logging.info(f"OOC wrt space: {ooc_space}")
logging.info(f"OOC wrt time: {ooc_time}")
