from dolfin import *
from dolfin.cpp.mesh import MeshFunctionSizet
import numpy as np
import matplotlib.pyplot as plt

set_log_level(LogLevel.ERROR)

# %% Read that mesh

# The volumetric mesh
mesh = Mesh()
with XDMFFile("/home/leonardo_mutti/PycharmProjects/masters_thesis/tests/meshing/mesh.xdmf") as infile:
    infile.read(mesh)

# The boundary conditions
mvc = MeshValueCollection("size_t", mesh, 1)  # 1 means: we consider lines, 1D things
with XDMFFile("/home/leonardo_mutti/PycharmProjects/masters_thesis/tests/meshing/facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = MeshFunctionSizet(mesh, mvc)

L1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, L1)

# %% Some data
T = 1.0
num_steps = 50
dt = T / num_steps

# Dirichlet BC that evolves in time, for the outer ring
f_D = Expression('x[0]*x[0]*sin(2*pi*t)', element=L1, pi=np.pi, t=0)
f_N = Expression('t', element=L1, pi=np.pi, t=0)
u_last = interpolate(Constant(0.0), V)
f = Constant(0.0)

# %% The problem

U = TrialFunction(V)
v = TestFunction(V)

# Functional formulation
dInnerRing = Measure("ds", subdomain_data=mf, subdomain_id=1)
print("Wrong variational form, not multiplyign everything by dt")
F = U * v * dX + dt * inner(grad(U), grad(v)) * dX - u_last * v * dX - f_N*v*dInnerRing
a, L = lhs(F), rhs(F)

# Boundary conditions (note, one will be updated with time)
bcs = [DirichletBC(V, f_D, mf, 2)]  # outer ring, inner ring has already Neumann BCs

# %% Time stepping

u = Function(V)
t = 0

vtkfile = File("solution.pvd")
for n in range(num_steps):
    print("Step: ", n, " of ", num_steps)
    t += dt
    f_D.t = t

    solve(a == L, u, bcs=bcs)
    vtkfile << (u, t)
    u_last.assign(u)
