from dolfin import *

if __name__=="__main__":

    mesh = UnitSquareMesh(8,8)
    V = FunctionSpace(mesh, "Lagrange", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    def boundary_dirichlet(x):
        return x[0] < DOLFIN_EPS or x[0] > 1 - DOLFIN_EPS

    bc = DirichletBC(V, Constant(0.0), boundary_dirichlet)

    sol = Function(V)

    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=1)
    g = Expression("sin(5*x[0])", degree=1)
    form = dot(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    # Adaptivity

    M = sol*dx
    tol = 1e-5

    # Note, to solve a(u,v)==L(v), you need u to be a trial function. It is assumed that that problem is linear. If you solve a(u,v)-L(v)==0 then u should be a Function, the variational thing is assumed to be nonlinear and convergence is reached in one step of some Newton's method
    # Solve equation a = L with respect to u and the given boundary
    # conditions, such that the estimated error (measured in M) is less
    # than tol
    problem = LinearVariationalProblem(form, L, sol, bc)
    solver = AdaptiveLinearVariationalSolver(problem, M)
    solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "cg"
    solver.parameters["error_control"]["dual_variational_solver"]["symmetric"] = True
    solver.solve(tol)

    solver.summary()

    # Plot solution(s)
    plt.figure()
    plot(sol.root_node(), title="Solution on initial mesh")
    plt.figure()
    plot(sol.leaf_node(), title="Solution on final mesh")
    plt.show()
