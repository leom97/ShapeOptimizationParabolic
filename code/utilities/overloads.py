from dolfin import *
from dolfin_adjoint import *
import numpy as np
from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

########################################################################################################################
# Note, we heavily rely on the usage of first order finite elements here...
########################################################################################################################

# Useful sources
# 1) projection onto simplex: https://fenicsproject.discourse.group/t/nearest-point-on-boundary-to-exterior-point/6687/4
# 2) building FEM base: https://fenicsproject.discourse.group/t/accessing-the-basis-function-of-a-finite-element/2426/2


def compute_spherical_transfer_matrix(V_vol, V_sph):
    """
    We compute a matrix that from a function defined on V_sph (a spherical function), actually, from the vector of its
    nodal values, returns a vector, the nodal values of a function on a volumetric function space V_vol. The resulting
    function if f(x) = g(x/|x|), for g the input function on the sphere. Note, the volumetric domain must not contain 0!
    :param V_vol: volumetric function space
    :param V_sph: spherical function space
    :return: M, the transfer matrix
    """

    mesh_vol = V_vol.mesh()
    mesh_sph = V_sph.mesh()

    # Coordinates but on the unit sphere, in dof order

    vol_points = mesh_vol.coordinates()[:]
    vol_points_sphere = vol_points / (np.linalg.norm(vol_points, axis=1)[:, None])  # note, in vertex order
    vol_points_sphere_dof = vol_points_sphere[dof_to_vertex_map(V_vol),
                            :]  # this means that the i-th nodal value of a function on V_vol, is the function value at vol_points_sphere_dof[i]

    # Basis functions on the sphere

    tree_sph = mesh_sph.bounding_box_tree()  # todo: this will be an attribute of a class later on

    def basis_sph(j, x):
        # here, i is dof order
        cell_index = tree_sph.compute_closest_entity(Point(*x))[0]
        while cell_index > V_sph.dim():  # seems cannot be avoided, this is a bug: https://fenicsproject.discourse.group/t/out-of-range-while-using-compute-closest-entity/8796/2
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

    # Building the matrix

    # NB: dofs = #mesh_points only for order one FEM!
    sph_dim = V_sph.dim()
    vol_dim = V_vol.dim()
    sphere_dofs = range(sph_dim)
    volume_dofs = range(vol_dim)

    M = np.zeros((vol_dim, sph_dim))  # operator from sphere to volume

    for j in sphere_dofs:
        for i in volume_dofs:
            M[i, j] = basis_sph(j, vol_points_sphere_dof[i, :])

    return M

def backend_transfer_from_sphere(g, M, V_vol):
    """
    Takes g, a function on the sphere, returns f, f(x)=g(x/|x|)
    :param g: function on spherical mesh
    :param M: transfer matrix from that spherical mesh to V_vol
    :param V_vol: volumetric mesh
    :return: f
    """
    gv = g.vector()[:]
    f = Function(V_vol)
    f.vector()[:] = M @ gv
    return f


# dolfin-adjoint version
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

        adj_input = adj_inputs[0]  # this is a vector, with which we test the adjoint jacobian
        adj_action = self.M.T @ adj_input
        output = Vector(Vector(MPI.comm_world, self.V_sph_dim))
        output[:] = adj_action

        return output


transfer_from_sphere = overload_function(backend_transfer_from_sphere, TranferFromSphereBlock)