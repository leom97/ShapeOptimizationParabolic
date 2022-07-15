from dolfin import *
from dolfin.cpp.mesh import MeshFunctionSizet
import numpy as np
import matplotlib.pyplot as plt

parameters[
    "refinement_algorithm"] = "plaza_with_parent_facets"  # must do this to be able to refine stuff associated with a mesh that is being refined

set_log_level(LogLevel.ERROR)

# %% Findings
# 1) DirichletBC modifies also interior nodes if you tell it to, so, be careful with that!
# 2) a discontinuous second derivative will cause the order of convergence to behave a little crazy, activate or deactivate the flag "smooth_solution" to see this

# %% Some data (e.g. exact solution and what not)

# Mesh path
mesh_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/meshes/annulus/"

# Resolutions
resolutions = [0.2, .1, .05, .025, .0125, 0.00625, 0.003125, 0.0015625]

# Test type
mesh_type = "annulus"  # annulus, square, disk, annulus2
initial_mesh = 0  # index inside the resolutions
builtin_refinement = False
smooth_solution = True # if activated, the tests will be done on a smooth solution

if mesh_type == "annulus" or mesh_type == "disk" or mesh_type == "annulus2":

    # H2 solution for L^\infty rhs (but not differentible rhs)
    class solution(UserExpression):
        def __init__(self, **kwargs):
            """ Construct the source function """
            super().__init__(self, **kwargs)

        def eval(self, values, x):
            """ Evaluate the source function """
            if smooth_solution:
                values[0] = -x[0] ** 2 / 2
            else:
                if x[0] >= 0:
                    values[0] = -x[0] ** 2 / 2
                else:
                    values[0] = 0


    class source(UserExpression):
        def __init__(self, **kwargs):
            """ Construct the source function """
            super().__init__(self, **kwargs)

        def eval(self, values, x):
            """ Evaluate the source function """
            if smooth_solution:
                values[0] = 1
            else:
                if x[0] >= 0:
                    values[0] = 1
                else:
                    values[0] = 0


    class normal_derivative(UserExpression):
        def __init__(self, **kwargs):
            """ Construct the source function """
            super().__init__(self, **kwargs)

        def eval(self, values, x):
            """ Evaluate the source function """
            if smooth_solution:
                if x[0] ** 2 + x[1] ** 2 < 1.5 ** 2:
                    values[0] = + x[0] ** 2
                else:
                    values[0] = - x[0] ** 2 / 2
            else:
                if x[0] >= 0:
                    if x[0] ** 2 + x[1] ** 2 < 1.5 ** 2:
                        values[0] = + x[0] ** 2
                    else:
                        values[0] = - x[0] ** 2 / 2
                else:
                    values[0] = 0


    # Annulus
    u_ex = solution()
    f = source()
    f_D = solution()
    f_N = normal_derivative()  # activate in case of annulus
elif mesh_type == "square":

    # Non-smooth source in Poisson equation: to be done on a unit square mesh
    # H1 but not H2 solution (note, |x|^{-a} \in H^k \iff k+a<n/p)
    ...


    class solution(UserExpression):
        def __init__(self, a, **kwargs):
            """ Construct the source function """
            super().__init__(self, **kwargs)
            self.a = a

        def eval(self, values, x):
            """ Evaluate the source function """
            r = sqrt(x[0] ** 2 + x[1] ** 2)
            values[0] = pow(r, -self.a)  # singularity at the origin


    class source(UserExpression):
        def __init__(self, a, **kwargs):
            """ Construct the source function """
            super().__init__(self, **kwargs)

            # remember, \Delta |x| = (n-1)/|x| (Friesecke, lecture 3)
            # and u_ex = |x|^{-a}
            # so, \Delta u_ex = -a|x|^{-a-2}
            self.a = a

        def eval(self, values, x):
            """ Evaluate the source function """
            r = sqrt(x[0] ** 2 + x[1] ** 2)
            if r <= DOLFIN_EPS:
                values[0] = 0  # note: this won't be used I think
            else:
                values[0] = + self.a * pow(r, -self.a - 2)  # the singularity is on the inner ring


    ...

    a = -2  # -1 -> ooc = 2? Not really, because I cannot define dirichlet conditions at the singularity: just go Neumann
    u_ex = solution(a)
    f_D = solution(a)
    f = source(a)
elif mesh_type == "square_ann":

    # H2 solution for L^\infty rhs (but not differentible rhs)
    class solution(UserExpression):
        def __init__(self, **kwargs):
            """ Construct the source function """
            super().__init__(self, **kwargs)

        def eval(self, values, x):
            """ Evaluate the source function """
            if x[0] >= 0:
                values[0] = -x[0] ** 2 / 2
            else:
                values[0] = 0


    class source(UserExpression):
        def __init__(self, **kwargs):
            """ Construct the source function """
            super().__init__(self, **kwargs)

        def eval(self, values, x):
            """ Evaluate the source function """
            if x[0] >= 0:
                values[0] = 1
            else:
                values[0] = 0


    class normal_derivative(UserExpression):
        def __init__(self, **kwargs):
            """ Construct the source function """
            super().__init__(self, **kwargs)

        def eval(self, values, x):
            """ Evaluate the source function """
            if x[0] >= 0:
                if x[0] ** 2 + x[1] ** 2 < 1.5 ** 2:
                    values[0] = + x[0] ** 2
                else:
                    values[0] = - x[0] ** 2 / 2
            else:
                values[0] = 0


    # Annulus
    u_ex = solution()
    f = source()
    f_D = solution()


# %% Read that mesh

def read_mesh(resolution, mesh_path=mesh_path):
    if mesh_type == "annulus":
        # The volumetric mesh
        mesh = Mesh()
        with XDMFFile(mesh_path + "mesh_" + str(resolution) + ".xdmf") as infile:
            infile.read(mesh)

        # The boundary conditions
        mvc = MeshValueCollection("size_t", mesh, 1)  # 1 means: we consider lines, 1D things
        with XDMFFile(mesh_path + "facet_mesh_" + str(resolution) + ".xdmf") as infile:
            infile.read(mvc, "name_to_read")
        mf = MeshFunctionSizet(mesh, mvc)  # remember, tag 3 is inner ring, tag 2 outer ring

        return mesh, mf

    elif mesh_type == "square":
        N = round(1 / resolution)
        mesh = UnitSquareMesh(N, N)

        return mesh, None

    elif mesh_type == "square_ann":

        N = round(1 / resolution)
        mesh = RectangleMesh(Point(-1.,-1.), Point(1.,1.), N, N)

        return mesh, None

    elif mesh_type == "disk":
        N = round(1 / resolution)
        mesh = UnitDiscMesh.create(MPI.comm_world, N, 1, 2)

        coord = mesh.coordinates()
        coord *= 2  # a disk of radius 2

        return mesh, None

    elif mesh_type == "annulus2":
        N = round(1 / resolution)
        mesh = UnitDiscMesh.create(MPI.comm_world, N, 1, 2)

        coord = mesh.coordinates()
        coord *= 2  # a disk of radius 2

        return mesh, None


# %% The problem

class problem:

    def __init__(self):
        pass

    def set_initial_mesh(self, mesh_and_mf):
        self.mesh = mesh_and_mf[0]
        self.mf = mesh_and_mf[1]
        self.initial = True

    def solve_single_pb(self, resolution, err_type="l2", builtin_refinement=False):

        if builtin_refinement and self.initial == False:
            new_mesh = refine(self.mesh)
            if self.mf is not None:
                self.mf = adapt(self.mf, new_mesh)
            self.mesh = new_mesh
        elif builtin_refinement == False:
            self.mesh, self.mf = read_mesh(resolution)

        self.initial = False  # tells whether this is the first time we're solving the problem

        L1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        V = FunctionSpace(self.mesh, L1)

        U = TrialFunction(V)
        v = TestFunction(V)

        if mesh_type == "annulus":

            # Functional formulation
            dOuterRing = Measure("ds", subdomain_data=self.mf, subdomain_id=2)

            F = inner(grad(U), grad(v)) * dX - f_N * v * dOuterRing - f * v * dX

            # Boundary conditions (note, one will be updated with time)
            def boundary(x):
                return x[0]**2+x[1]**2<1.5**2
            bcs = [DirichletBC(V, f_D, self.mf, 3)]  # inner ring, outer ring has already Neumann BCs
            # bcs = [DirichletBC(V, f_D, boundary)]  # inner ring, outer ring has already Neumann BCs

        elif mesh_type == "square":
            F = inner(grad(U), grad(v)) * dX - f * v * dX

            def boundary(x):
                return True

            bcs = DirichletBC(V, f_D, boundary)

        elif mesh_type ==  "square_ann":
            F = inner(grad(U), grad(v)) * dX - f * v * dX

            def boundary(x):
                return x[1]<1-DOLFIN_EPS

            bcs = DirichletBC(V, f_D, boundary)

        elif mesh_type == "disk":
            F = inner(grad(U), grad(v)) * dX - f * v * dX

            def boundary(x):
                return True

            bcs = DirichletBC(V, f_D, boundary)

        u = Function(V)
        a, L = lhs(F), rhs(F)
        solve(a == L, u, bcs)

        plot(u)
        plt.show()
        plot(self.mesh)
        plt.show()

        if err_type == "l2":
            err = errornorm(u_ex,
                            u)  # errnorm the correct l2 norm, as you can check with simple functions and a unit square
        elif err_type == "linf":
            u_ex_comp = interpolate(u_ex, V)
            err = np.max(np.abs(np.array(u_ex_comp.vector()[:]) - np.array(u.vector()[:])))
        else:
            raise NotImplementedError("Norm not implemented")

        return err, self.mesh.hmax()


# %% Error checking

err_vec = []
mesh_sizes = []

pb = problem()
if builtin_refinement:
    pb.set_initial_mesh(read_mesh(resolutions[initial_mesh]))

for res in resolutions:
    (e, h) = pb.solve_single_pb(res, err_type="l2", builtin_refinement=builtin_refinement)
    err_vec.append(e)
    mesh_sizes.append(h)

err_vec = np.array(err_vec)
mesh_sizes = np.array(mesh_sizes)

ooc = np.log(err_vec[1:] / err_vec[:-1]) / np.log(mesh_sizes[1:] / mesh_sizes[:-1])

print(ooc)
