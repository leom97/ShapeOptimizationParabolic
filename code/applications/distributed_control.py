from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import logging

from utilities.meshing import AnnulusMesh, CircleMesh
from utilities.pdes import HeatEquation, PDETestProblems
import numpy as np

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

# %% What does dolfin adjoint see?

# Global variables
resolution = 0.05

# Mesh
heq = HeatEquation()
annulus = AnnulusMesh(resolution=resolution)
heq.set_mesh(annulus.mesh, annulus.facet_function)

# Data
u_N = Function(heq.S1h)
u_D = Function(heq.S1h)
marker_dirichlet = [3]
marker_neumann = [2]
u0 = Function(heq.S1h)
f = Function(heq.S1h)
f.vector()[:] = 2
T = 2.0

u_D.rename("DirichletData", "")
u_N.rename("NeumannData", "")
u0.rename("InitialSolution", "")

# Setting up the PDE class
heq.set_PDE_data(u0, source=f, marker_neumann=marker_neumann, marker_dirichlet=marker_dirichlet, neumann_BC=[u_N],
                 dirichlet_BC=[u_D])  # note, no copy is done, the attributes of heq are EXACTLY these guys
heq.set_ODE_scheme("crank_nicolson")
heq.verbose = True

time_sch_idx = heq.implemented_time_schemes.index(heq.ode_scheme)
if time_sch_idx == 0 or time_sch_idx == 3:  # implicit and implicit-explicit euler
    N_steps = int(np.ceil(2 / (heq.mesh.hmax() ** 2)))
elif time_sch_idx == 1:  # crank-nicolson
    N_steps = int(np.ceil(2 / (heq.mesh.hmax() ** 1)))

heq.set_time_discretization(T, N_steps=N_steps)  # dt = h^2 and N = T/dt

# Solution
solution_list, _ = heq.solve(err_mode="none")

# Energy
u = solution_list[-1]
J = assemble(u * u * dx(heq.mesh))

j = ReducedFunctional(J, Control(f))
s_opt = minimize(j, tol=1e-6, options={"gtol": 1e-6, "maxiter": 100, "disp": True})
taylor_test(j, s_opt, s_opt)

# %% Optimization cycle

# rate = 1
#
# for i in range(20):  # super slow!
#
#     energy = j(f)
#     dJ = j.derivative()
#     heq.f.vector()[:] -= rate * dJ.vector()
#     logging.info(f"{i}) energy value: {energy}, gradient norm: {sqrt(assemble(dJ ** 2 * dx(heq.mesh)))}")

vtkfile = File("/home/leonardo_mutti/PycharmProjects/masters_thesis/code/applications/u.pvd")
vtkfile << s_opt