from dolfin import *
from dolfin.cpp.mesh import MeshFunctionSizet
import matplotlib.pyplot as plt
set_log_level(LogLevel.ERROR)

#%% Read that mesh

# The volumetric mesh
mesh = Mesh()
with XDMFFile("/home/leonardo_mutti/PycharmProjects/masters_thesis/tests/meshing/mesh.xdmf") as infile:
    infile.read(mesh)

# The boundary conditions
mvc = MeshValueCollection("size_t", mesh, 1)    # 1 means: we consider lines, 1D things
with XDMFFile("/home/leonardo_mutti/PycharmProjects/masters_thesis/tests/meshing/facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = MeshFunctionSizet(mesh, mvc)

#%% Function spaces

CG = FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # lagrange element on a triangle, of degree 1
V = FunctionSpace(mesh, CG)

#%% Boundary conditions

one = Constant(1.0)
zero = Constant(0.0)

bcs = [DirichletBC(V, one, mf, 2), DirichletBC(V, zero, mf, 3)] # outer ring, inner ring

#%% Variational form

U = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(U),grad(v))*dX
L = inner(zero, v)*dX

u = Function(V, name = "solution")

solve(a==L, u, bcs = bcs)

#%% Visualize/save

# Plot solution
p = plot(u)
plt.colorbar(p) # https://fenicsproject.discourse.group/t/setting-colorbar-range-on-fenics-plot/241
plt.show()

# For more interesting options, see https://fenicsproject.org/qa/10038/how-to-plot-just-the-values-in-a-subdomain/, https://fenicsproject.org/qa/11876/extract-solution-at-a-set-of-nodes.

# Export to ParaView. Look here to go on from now: https://link.springer.com/content/pdf/10.1007/978-3-319-52462-7.pdf, p. 28

vtkfile = File("solution.pvd")
vtkfile << u