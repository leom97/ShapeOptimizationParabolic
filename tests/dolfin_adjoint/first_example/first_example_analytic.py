from fenics import *
from fenics_adjoint import *
set_log_level(LogLevel.ERROR)

#%% The problem

n = 30
mesh = IntervalMesh(n, -1, 1)
V = FunctionSpace(mesh, "CG", 1)

U = TrialFunction(V)
v = TestFunction(V)

# q = 3.0
nu = Constant(3.0)  # Note, the control variable must be something that Fenics can recognize, so, surely not a float
a = inner(grad(U), grad(v))*dX
l = nu*v*dX

u = Function(V)
solve(a == l, u, bcs = DirichletBC(V, Constant(0.0), "on_boundary"))

J = assemble(u*dX)    # from analytical computations, it is 2/3 nu

dJdnu = compute_gradient(J, Control(nu))    # should be 2/3, which it is

# Let's verify the claim