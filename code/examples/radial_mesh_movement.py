from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import logging

from utilities.meshing import AnnulusMesh, CircleMesh
from utilities.overloads import compute_spherical_transfer_matrix, \
    compute_radial_displacement_matrix, radial_displacement

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

# %% Testing forward and backward: create meshes

import numpy as np

# Read volumetric mesh
annulus = AnnulusMesh(resolution=0.05)
mesh_vol = annulus.mesh
L1_vol = FiniteElement("Lagrange", mesh_vol.ufl_cell(), 1)
V_vol = FunctionSpace(mesh_vol, L1_vol)

# Read spherical mesh
circle = CircleMesh(resolution=0.1)
mesh_sph = circle.mesh
L1_sph = FiniteElement("Lagrange", mesh_sph.ufl_cell(), 1)
V_sph = FunctionSpace(mesh_sph, L1_sph)

# %% All the rest

# We need a transfer matrix at first
M = compute_spherical_transfer_matrix(V_vol, V_sph)

# Transfer from sphere
g = Function(V_sph)
g.vector()[vertex_to_dof_map(V_sph)[0]] = .25  # who knows now what is 7...
g_hat = transfer_from_sphere(g, M, V_vol)

# Create a UFL vector field
x = SpatialCoordinate(mesh_vol)
n = sqrt(dot(x, x))
W = (Constant(2.0) - n) * (
        x / n)

# Vector field displacement
VD = VectorFunctionSpace(mesh_vol, "Lagrange", 1)
W_m = project(W * g_hat, VD)

# %% Did it work?

plot(mesh_vol)
plt.show()
ALE.move(mesh_vol, W_m)
plot(mesh_vol)
plt.show()

J = assemble(Constant(1.0) * dx(mesh_vol))

h = Function(V_sph)
h.vector()[:] = 2 * (np.random.rand(V_sph.dim()) - .5)
taylor_test(ReducedFunctional(J, Control(g)), g, h)

# %% A new hope, which apparently, also works

M2 = compute_radial_displacement_matrix(M, VD)
W = radial_displacement(g, M2, VD)

# This really fixes the boundary, nice
plot(mesh_vol)
plt.show()
ALE.move(mesh_vol, W)
plot(mesh_vol)
plt.show()

J2 = assemble(Constant(1.0) * dx(mesh_vol))

h = Function(V_sph)
h.vector()[:] = 2 * (np.random.rand(V_sph.dim()) - .5)
taylor_test(ReducedFunctional(J2, Control(g)), g, h)

# %% Inspection of the mesh

print(W(2, 0))
vtkfile = File("/home/leonardo_mutti/PycharmProjects/masters_thesis/code/examples/u.pvd")
u = Function(V_vol)
vtkfile << u

# %% Notes

# If g is a spike then we have mesh compenetration...
# Also, project is not being very precise, the external boundary is moving ever so slightly -> let's create a new dolfin-adjont module

# And to understand how to map between dofs and vertices on a VectorFunctionSpace, check out https://fenicsproject.org/qa/13595/interpret-vertex_to_dof_map-dof_to_vertex_map-function/
