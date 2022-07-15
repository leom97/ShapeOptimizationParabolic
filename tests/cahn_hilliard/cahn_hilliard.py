from dolfin import *
import random

if __name__ == "__main__":

    # Define an initial function
    # Solve using Netwon's method, which return du
    # So, define linearized form and rhs
    # Pass them to abstract Newton's class
    # Solve, indicating who is actually the solution

    # Initial condition
    class init_cond(UserExpression):
        def __init__(self, **kwargs):
            random.seed(2 + MPI.rank(MPI.comm_world))
            super().__init__(**kwargs)

        def eval(self, values, x):
            values[0] = 0.63 + 0.02 * (0.5 - random.random())
            values[1] = 0.0

        def value_shape(self):
            return (2, )

    class newton_interface(NonlinearProblem):
        def __init__(self, a, L):
            NonlinearProblem.__init__(self)
            self.L = L
            self.a = a
            # Here, a and L are bi-LINEAR (in test, trial), and LINEAR (in test)

        def F(self, b, x):
            # Assemble the linear form L, put it into b
            assemble(self.L, tensor = b)

        def J(self, A, x):
            # Assemble bilinear form, put it into A
            assemble(self.a, tensor = A)

        # The Newton's method will solve F = 0, J is JF, it will return some delta_u and I will specify to what this delta has to be summed, later on


    # The real stuff
    # Model parameters
    lmbda = 1.0e-02  # surface parameter
    dt = 5.0e-06  # time step
    theta = 0.5  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

    # Form compiler options
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True

    # Create mesh and build function space
    mesh = UnitSquareMesh.create(96, 96, CellType.Type.quadrilateral)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    ME = FunctionSpace(mesh, P1 * P1)   # the solution is a 2D vector!

    # Trial and test function. Note, trial is a delta because we're solving a nonlinear problem
    du = TrialFunction(ME)  # a 2D vector
    q, v = TestFunctions(ME)    # a 2D vector, unpacked

    # Define solution and previous solution for theta method. They must be functions
    u = Function(ME)
    u0 = Function(ME)

    # Split mixed functions
    dc, dmu = split(du)
    c, mu = split(u)
    c0, mu0 = split(u0)

    # Initialize them both: create a concrete instance of init_cond, put it into the FE space, and then,
    u_init = init_cond(degree=1)
    u.interpolate(u_init)
    u0.interpolate(u_init)
