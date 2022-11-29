"""
Some custom modules for dolfin-adjoint. They implement, most notably, the star-shaped parametrization for the domains.
"""

# %% Imports

from dolfin import *
from dolfin_adjoint import *
import numpy as np
from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function
from tqdm import tqdm
import logging


# %% All the rest

def compute_spherical_transfer_matrix(V_vol, V_sph, p=None):
    """
    We compute a matrix that from a function defined on V_sph (a spherical function), actually, from the vector of its
    nodal values, returns a vector, the nodal values of a function on a volumetric function space V_vol. The resulting
    function if f(x) = g(x/|x|), for g the input function on the sphere. Note, the volumetric domain must not contain p!
    :param V_vol: volumetric function space
    :param V_sph: spherical function space
    :param p: the point with respect to which the domain is star shaped. Must be a (n,1) array
    :return: M, the transfer matrix
    """

    if p is None:
        p = np.zeros(V_vol.mesh().topology().dim())[None, :]

    mesh_vol = V_vol.mesh()
    mesh_sph = V_sph.mesh()

    # Coordinates but on the unit sphere, in dof order

    vol_points = mesh_vol.coordinates()[:] - p
    vol_points_sphere = vol_points / (np.linalg.norm(vol_points, axis=-1)[:, None])  # note, in vertex order
    vol_points_sphere_dof = vol_points_sphere[dof_to_vertex_map(V_vol),
                            :]  # this means that the i-th nodal value of a function on V_vol, is the function value at vol_points_sphere_dof[i]

    # Basis functions on the sphere

    tree_sph = mesh_sph.bounding_box_tree()

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

    logging.info(f"Assembling transfer matrix from spherical functions into scalar functions")
    for j in tqdm(sphere_dofs):
        for i in volume_dofs:
            M[i, j] = basis_sph(j, vol_points_sphere_dof[i, :])

    return M

def radial_function_to_square(x, L=1):
    """
    Computes the radial function describing a square with center 0 and side L
    :param x: Nx2 array of coordinates different from 0
    :param L:
    :return:
    """

    x_hat = x / (np.linalg.norm(x, axis=-1)[:, None])

    mask_ud = (-np.sqrt(2) / 2 <= x_hat[:, 0]) * (x_hat[:, 0] <= np.sqrt(2) / 2)
    mask_lr = (1 - mask_ud).astype(bool)

    output = np.zeros(x_hat.shape[0])

    x_ud = x_hat[mask_ud, :]
    x_lr = x_hat[mask_lr, :]

    output[mask_lr] = 0.5 * L / np.sqrt(1 - x_lr[:, 1] ** 2)
    output[mask_ud] = 0.5 * L / np.sqrt(1 - x_ud[:, 0] ** 2)

    return output


def compute_radial_displacement_matrix(M, VD, f_D=None, p=None, eps=1):
    """
    :param f_D: radial lambda function describing the external boundary of our pseudo-annulus. It is assumed infinite dimensional for now (todo: relax this)
    :param eps, p: out reference pseudo-annulus has a hole of radius eps centered at p
    :param M:
    :param VD:
    :param p: a point of shape (1,n)
    :return: M2, a matrix taking the nodal values of a function defined on the sphere, to its radial displacement vector field (nodal coordinates)
    """

    # dof to vertex map, to index the first component of mesh_coords
    d2v = dof_to_vertex_map(VD)[0::2] // 2

    mesh_coords = VD.mesh().coordinates()[:] - p  # note the translation!!
    mesh_coords_dof = mesh_coords[d2v, :]
    mesh_coords_dof_norms = np.linalg.norm(mesh_coords_dof, axis=-1)[:, None]

    border_coords = f_D(mesh_coords_dof)

    multiplier = (border_coords[:, None] - mesh_coords_dof_norms) / (
            border_coords[:, None] - eps) * mesh_coords_dof / mesh_coords_dof_norms

    multiplier[np.abs(border_coords - np.squeeze(mesh_coords_dof_norms)) <= DOLFIN_EPS, :] = 0

    M2 = np.zeros((VD.dim(), M.shape[1]))

    # Let's do it inefficiently at first
    logging.info(f"Assembling transfer matrix from spherical functions into vector fields")
    for i in tqdm(range(M.shape[0])):
        M_row_i = M[i, :][None, :]
        multiplier_i = multiplier[i, :][:, None]

        two_rows_i = multiplier_i @ M_row_i

        M2[(2 * i):(2 * i + 2), :] = two_rows_i

    return M2


def backend_radial_displacement(g, M2, VD):
    """
    :param g: a spherical function on S^n
    :param M2: transfer matrix S^n -> \Omega, for \Omega star shaped wrt p
    :param VD: CG1 vector function space on \Omega
    :return: W, W(x) = (f_D-|x-p|)*(x-p)/|x-p| * g((x-p)/|x-p|), if the original domain is p-star shaped
    """

    gv = g.vector()[:]
    W = Function(VD)
    W.vector()[:] = M2 @ gv
    return W


class RadialDisplacementBlock(Block):
    def __init__(self, g, M2, VD, **kwargs):
        super(RadialDisplacementBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(g)

        self.M2 = M2  # matrix bringing us from g.function_space() to VD
        self.VD = VD  # the volumetric function space, vector fields version
        self.VD_dim = VD.dim()  # the volumetric function space, vector fields version
        self.V_sph_dim = g.function_space().dim()

    def __str__(self):
        return 'RadialDisplacementBlock'

    def recompute_component(self, inputs, block_variable, idx, prepared):
        # note, inputs is the list containing all the dependencies we specified above
        # therefore it will be a list of length 1, containing exactly func (actually, a copy of it)
        return backend_radial_displacement(inputs[0], self.M2, self.VD)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        # Note: everything in automatic differentiation is just a vector. So, this method takes vectors (the adj_inputs) and returns vectors

        # Explanation
        # A function is e.g. z = f(x,y). If idx==1, here, we must return (d/dy f (x,y))^T * z
        # In the specific case, z = f(x), so idx == 0 always

        adj_input = adj_inputs[0]  # this is a vector, with which we test the adjoint jacobian
        adj_action = self.M2.T @ adj_input
        output = Vector(MPI.comm_world, self.V_sph_dim)
        output[:] = adj_action

        return output

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        # see http://www.dolfin-adjoint.org/en/latest/documentation/pyadjoint_docs.html

        return self.recompute_component(tlm_inputs, block_variable, idx, prepared)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)


# transfer_from_sphere = overload_function(backend_transfer_from_sphere, TranferFromSphereBlock)
radial_displacement = overload_function(backend_radial_displacement, RadialDisplacementBlock)
