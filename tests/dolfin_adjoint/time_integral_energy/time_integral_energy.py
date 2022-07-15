from dolfin import *
from fenics_adjoint import *
from dolfin.cpp.mesh import MeshFunctionSizet
import numpy as np
import matplotlib.pyplot as plt

set_log_level(LogLevel.ERROR)

# %% Read that mesh

# The volumetric mesh
mesh = UnitIntervalMesh(20)
L1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, L1)

# %% Some data
T = 2.0
num_steps = 10
dt = T / num_steps

q = Constant(3.0)

# Dirichlet BC that evolves in time, for the outer ring
f_D = Expression('t*q', element=L1, pi=np.pi, t=0, q=q)
f_N = Constant(0.0)
u_last = interpolate(Constant(0.0), V)
f = q

# %% The problem

U = TrialFunction(V)
v = TestFunction(V)

# Dirichlet boundary
def boundary(x):
    return x[0] > 1.0 - DOLFIN_EPS


# Functional formulation
F = U * v * dX + dt * inner(grad(U), grad(v)) * dX - u_last * v * dX - dt*f_N*v*ds -dt*f*v*dX
a, L = lhs(F), rhs(F)

# Boundary conditions (note, one will be updated with time)
bcs = DirichletBC(V, f_D, boundary)

# %% Time stepping and energy

u = Function(V)
u.assign(u_last)

t = 0
J = 0
j_last = assemble(u*u*dX)

vtkfile = File("solution.pvd")
for n in range(num_steps):
    print("Step: ", n, " of ", num_steps)
    t += dt
    f_D.t = t
    f_N.t = t

    solve(a == L, u, bcs=bcs)
    j_curr=  assemble(u*u*dX)
    J = J + (j_curr+j_last)*dt/2
    j_last = j_curr
    vtkfile << (u, t)
    u_last.assign(u)