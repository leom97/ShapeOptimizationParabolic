from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Dirichlet boundary
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

if __name__ == "__main__":

    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define BCs
    bc = DirichletBC(V, Constant(0.0), boundary)

    # Variational formulation
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    g = Expression("sin(5*x[0])", degree=2)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    # Compute solution
    sol = Function(V)
    solve(a == L, sol, bc)

    # Plot solution
    plot(sol)
    plt.show()

    # Now, get stiffness matrix, because why not
    A = PETScMatrix()
    assemble(a, tensor = A)

    # Get numpy stiffness matrix
    A_np = np.array(A.array())

    # Vectorize the solution
    sol_np = np.array(sol.vector()[:])

    M = np.reshape(sol_np, (33, 33), order = 'C')
    plt.imshow(M.T)
    plt.show()

    # Problem, what kind of order is adopted? Not a big problem in any case, as what would I do with more complicated meshes?
