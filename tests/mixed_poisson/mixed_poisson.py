from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Define function G such that G \cdot n = g

# Define function G such that G \cdot n = g
class BoundarySource(UserExpression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        g = sin(5*x[0])
        values[0] = g*n[0]
        values[1] = g*n[1]
    def value_shape(self):
        return (2,)


# Define essential boundary
def boundary(x):
    return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

if __name__=="__main__":
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    g = Expression("sin(5*x[0])", degree=2)

    mesh = UnitSquareMesh(32, 32)
    G = BoundarySource(mesh, degree=2)

    # Define finite elements spaces and build mixed space
    # Note: cell (triangle) is already "defined" in the mesh, so, we just take it from there
    BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
    DG = FiniteElement("DG", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, BDM * DG)

    # Trial
    (sigma, u) = TrialFunctions(W)
    (tau, v) = TestFunctions(W)

    # Define variational form
    a = (dot(sigma, tau) + div(tau) * u + div(sigma) * v) * dx
    L = - f * v * dx

    # Boundary conditions
    bc = DirichletBC(W.sub(0), G, boundary) # hacky and possibly inexact way of imposing "Neumann" BC: we impose Dirichlet BC to the vector valued thing, where G is such that G.n = g

    # Solution
    sol = Function(W)
    solve(a == L, sol, bc)
    (sigma, u) = sol.split()

    # Plot sigma and u
    plt.figure()
    plot(sigma)

    plt.figure()
    plot(u)

    plt.show()