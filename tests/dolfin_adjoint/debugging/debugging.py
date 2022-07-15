from dolfin import *
from fenics_adjoint import *

n = 30
mesh = UnitSquareMesh(n, n)
V = VectorFunctionSpace(mesh, "CG", 2)

u = project(Expression(("sin(2*pi*x[0])", "cos(2*pi*x[1])"), degree=2), V)
u.rename("init_sol", "The initial solution")  # the second thing is a description
u_next = Function(V, name="u_next")
v = TestFunction(V)

nu = Constant(0.0001, name="nu")

timestep = Constant(0.01, name="dt")

F = (inner((u_next - u) / timestep, v)
     + inner(grad(u_next) * u_next, v)
     + nu * inner(grad(u_next), grad(v))) * dx

bc = DirichletBC(V, (0.0, 0.0), "on_boundary")

tape = get_working_tape()

t = 0.0
end = 0.05
while (t <= end):
    with tape.name_scope("Time_step"):
        solve(F==0, u_next, bc)
        u.assign(u_next)
        t+=float(timestep)

tape.visualise()

# Now in the same folder where log is, type tensorboard --logdir ./log
# And go to the suggested webpage localhost/6006
