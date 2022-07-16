from dolfin import *
from dolfin_adjoint import *
import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# Note, we heavily rely on the usage of first order finite elements here...
########################################################################################################################

# Useful sources
# 1) projection onto simplex: https://fenicsproject.discourse.group/t/nearest-point-on-boundary-to-exterior-point/6687/4
# 2) building FEM base: https://fenicsproject.discourse.group/t/accessing-the-basis-function-of-a-finite-element/2426/2

# %% Read volumetric mesh

mesh_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/meshes/annulus/"
resolutions = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]  # mesh resolutions
resolution = resolutions[2]

mesh_vol = Mesh()
with XDMFFile(mesh_path + "mesh_" + str(resolution) + ".xdmf") as infile:
    infile.read(mesh_vol)

mesh_vol.coordinates()[:] *= 2  # for good measure, just to have the inner sphere not equal to 1

L1_vol = FiniteElement("Lagrange", mesh_vol.ufl_cell(), 1)
V_vol = FunctionSpace(mesh_vol, L1_vol)

# %% Read spherical mesh

mesh_path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/meshes/sphere/"

mesh_sph = Mesh()
with XDMFFile(mesh_path + "circle_mesh.xdmf") as infile:
    infile.read(mesh_sph)

L1_sph = FiniteElement("Lagrange", mesh_sph.ufl_cell(), 1)
V_sph = FunctionSpace(mesh_sph, L1_sph)

# %% Coordinates but on the unit sphere, in dof order

vol_points = mesh_vol.coordinates()[:]
vol_points_sphere = vol_points / (np.linalg.norm(vol_points, axis=1)[:, None])  # note, in vertex order
vol_points_sphere_dof = vol_points_sphere[dof_to_vertex_map(V_vol),
                        :]  # this means that the i-th nodal value of a function on V_vol, is the function value at vol_points_sphere_dof[i]

# %% Basis functions on the sphere

tree_sph = mesh_sph.bounding_box_tree()  # todo: this will be an attribute of a class later on


def basis_sph(j, x):
    # here, i is dof order
    cell_index = tree_sph.compute_closest_entity(Point(*x))[0]
    while cell_index > V_sph.dim(): # seems cannot be avoided, this is a bug: https://fenicsproject.discourse.group/t/out-of-range-while-using-compute-closest-entity/8796/2
        x += (np.random.rand() - .5) * 1e-8
        cell_index = tree_sph.compute_closest_entity(Point(*x))[0]
    cell_global_dofs = V_sph.dofmap().cell_dofs(cell_index)
    for local_dof in range(0, len(cell_global_dofs)):
        if (j == cell_global_dofs[local_dof]):
            cell = Cell(V_sph.mesh(), cell_index)
            return V_sph.element().evaluate_basis(local_dof, x,
                                                  cell.get_vertex_coordinates(),
                                                  cell.orientation())
    # If none of this cell's shape functions map to the i-th basis function,
    # then the i-th basis function is zero at x.
    return np.array([0.0])


# %% Building the matrix

# NB: dofs = #mesh_points only for order one FEM!
sph_dim = V_sph.dim()
vol_dim = V_vol.dim()
sphere_dofs = range(sph_dim)
volume_dofs = range(vol_dim)

M = np.zeros((vol_dim, sph_dim))  # operator from sphere to volume

for j in sphere_dofs:
    for i in volume_dofs:
        M[i, j] = basis_sph(j, vol_points_sphere_dof[i, :])

# %% Checking the correctedness, by implementing a forward function

g = Function(V_sph)
g.vector()[vertex_to_dof_map(V_sph)[7]] = 1 # a displacement of 1 at [1,0]

def transfer_from_shere_quick(g):
    gv = g.vector()[:]
    f = Function(V_vol)
    f.vector()[:] = M@gv
    return f

f = transfer_from_shere_quick(g)    # plotting it seems okay

# %% Encapsulation into a nice forward function

# We will have a dolfin-adjoint module with two inputs, only one of which is a dependency
def backend_transfer_from_sphere(g, M, V_vol):
    gv = g.vector()[:]
    f = Function(V_vol)
    f.vector()[:] = M @ gv
    return f

f2 = backend_transfer_from_sphere(g, M, V_vol)    # plotting it seems okay

# %% dolfin-adjoint version

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

class TranferFromSphereBlock(Block):
    def __init__(self, g, M, V_vol, **kwargs):
        super(TranferFromSphereBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(g)

        self.M = M  # 'interpolation' matrix bringing us from g.function_space() to V_vol
        self.V_vol = V_vol  # the volumetric function space, onto which we extend g radially
        self.V_sph_dim = g.function_space().dim()

    def __str__(self):
        return 'TranferFromSphereBlock'

    def recompute_component(self, inputs, block_variable, idx, prepared):
        # note, inputs is the list containing all the dependencies we specified above
        # therefore it will be a list of length 1, containing exactly func (actually, a copy of it)
        return backend_transfer_from_sphere(inputs[0], self.M, self.V_vol)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        # Note: everything in automatic differentiation is just a vector. So, this method takes vectors (the adj_inputs) and returns vectors

        # Explanation
        # A function is e.g. z = f(x,y). If idx==1, here, we must return (d/dy f (x,y))^T * z
        # In the specific case, z = f(x), so idx == 0 always

        adj_input = adj_inputs[0]   # this is a vector, with which we test the adjoint jacobian
        adj_action = self.M.T @ adj_input
        output = Vector(Vector(MPI.comm_world, self.V_sph_dim))
        output[:] = adj_action

        return output
transfer_from_sphere = overload_function(backend_transfer_from_sphere, TranferFromSphereBlock)

# %% Testing forward and backward

# Transfer from sphere
g_hat = transfer_from_sphere(g, M, V_vol)

# Create a UFL vector field
x = SpatialCoordinate(mesh_vol)
n = sqrt(dot(x,x))
W = (Constant(4.0)-n)/Constant(2.0)*(x/n)   # 2 and 4 because we arbitrarily modified the annulus

# Vector field displacement
VD = VectorFunctionSpace(mesh_vol, "Lagrange", 1)
W_m = project(W*g_hat, VD)

# plot(mesh_vol)
# plt.show()
ALE.move(mesh_vol, W_m)
# plot(mesh_vol)
# plt.show()

J = assemble(Constant(1.0) * dx(mesh_vol))

h = Function(V_sph)
h.vector()[:] = 2 * (np.random.rand(V_sph.dim())-.5)
taylor_test(ReducedFunctional(J, Control(g)), g, g)
