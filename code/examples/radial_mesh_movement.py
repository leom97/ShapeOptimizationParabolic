from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import logging

from utilities.meshing import AnnulusMesh, CircleMesh
from utilities.pdes import HeatEquation, PDETestProblems
from utilities.overloads import transfer_from_sphere, compute_spherical_transfer_matrix

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

# %% Testing forward and backward

import numpy as np

# Read volumetric mesh
annulus = AnnulusMesh(resolution = 0.05)
mesh_vol = annulus.mesh
L1_vol = FiniteElement("Lagrange", mesh_vol.ufl_cell(), 1)
V_vol = FunctionSpace(mesh_vol, L1_vol)

# Read spherical mesh
circle = CircleMesh(resolution = 0.1)
mesh_sph = circle.mesh
L1_sph = FiniteElement("Lagrange", mesh_sph.ufl_cell(), 1)
V_sph = FunctionSpace(mesh_sph, L1_sph)

# We need a transfer matrix at first
M = compute_spherical_transfer_matrix(V_vol, V_sph)

# Transfer from sphere
g = Function(V_sph)
g.vector()[vertex_to_dof_map(V_sph)[7]] = .5 # who knows now what is 7...
g_hat = transfer_from_sphere(g, M, V_vol)

# Create a UFL vector field
x = SpatialCoordinate(mesh_vol)
n = sqrt(dot(x, x))
W = (Constant(2.0) - n) * (x / n)   # note, the boundary is moved a little bit and there is some compenetration... # todo: look into this

# Vector field displacement
VD = VectorFunctionSpace(mesh_vol, "Lagrange", 1)
W_m = project(W * g_hat, VD)

plot(mesh_vol)
plt.show()
ALE.move(mesh_vol, W_m)
plot(mesh_vol)
plt.show()

J = assemble(Constant(1.0) * dx(mesh_vol))

h = Function(V_sph)
h.vector()[:] = 2 * (np.random.rand(V_sph.dim()) - .5)
taylor_test(ReducedFunctional(J, Control(g)), g, h)
