from dolfin import *
import matplotlib.pyplot as plt

# %% Create mesh

mesh = UnitSquareMesh(4, 4)
plot(mesh)  # dolfin has its own plot
plt.show()

# %% Examine nodes, create full matrix

mesh_coords = mesh.coordinates()  # stored in left right, up down order
FEM = FiniteElement(family="CG", cell=mesh.ufl_cell(), degree=1)
V = FunctionSpace(mesh, FEM)

u = TrialFunction(V)
v = TestFunction(V)
A = assemble(inner(grad(u), grad(v)) * dx)

A_np = A.array()

# %% Create right hand side

f = Expression("3", element=FEM)
b = assemble(f * v * dx)

# %% Create Dirichlet conditions everywhere

def boundary(x, on_boundary):
    # return x[0]+DOLFIN_EPS < 0
    return on_boundary

diri_bc = DirichletBC(V, Constant(0.0), boundary)

diri_bc.apply(A,b)