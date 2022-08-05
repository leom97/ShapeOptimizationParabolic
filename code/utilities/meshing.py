import pygmsh
import gmsh
import meshio
import abc
import numpy as np

from dolfin import *
from dolfin_adjoint import *

from utilities.overloads import radial_function_to_square


class AbstractMesh:
    """
    Class for objects which are either spheres or have a spherical hole inside, and whose boundary is star shaped
    """

    # Adapted from http://jsdokken.com/src/pygmsh_tutorial.html#first
    __metaclass__ = abc.ABCMeta

    def __init__(self, resolution, path, dimension, center, inner_radius):
        self.resolution = resolution
        self.path = path
        self.dimension = dimension
        self.center = center
        self.inner_radius = inner_radius

        # An empty geometry
        self.geometry = pygmsh.geo.Geometry()

        self.mesh = None
        self.mf = None

        # Execute all the mesh generation automatically, return a fenics mesh directly
        self.generate_mesh_xdmf()
        self.xdmf_to_dolfin()

    @abc.abstractmethod
    def create_mesh(self):
        """
        Here one should implement the creation/modeling of the mesh, tagging included. No file saving is performed
        :return:
        """

    def save_to_msh(self):
        resolution = self.resolution
        path = self.path

        self.geometry.save_geometry(
            path + "geo_" + str(resolution) + ".geo_unrolled")  # not exactly working on gmsh though...
        gmsh.write(path + "mesh_" + str(resolution) + ".msh")
        gmsh.clear()
        self.geometry.__exit__()

    def mesh_to_meshio(self, mesh, cell_type, prune_z=False):
        # This link might help to understand better what is going on: https://github.com/iitrabhi/GSoC2019/blob/master/Notebooks/Understanding%20MeshFunction%20and%20MeshValueCollection.ipynb

        cells = mesh.get_cells_type(cell_type)  # get the cells of some type: it will change!
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:, :2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]})
        return out_mesh

    def load_from_msh(self):
        return meshio.read(self.path + "mesh_" + str(self.resolution) + ".msh")

    @abc.abstractmethod
    def generate_mesh_xdmf(self):
        """
        Takes care of doing everything, including saving the mesh to xdmf
        :return:
        """

    @abc.abstractmethod
    def xdmf_to_dolfin(self):
        """
        Takes xdmf files as input and loads the mesh and in case, the facet indicator, into dolfin
        :return:
        """

    @abc.abstractmethod
    def boundary_radial_function(self, x):
        """
        :param x: a spherical point
        :return: the value of the radial function describing the boundary: f(x/|x|) = |x| => x \in boundary
        """


class AnnulusMesh(AbstractMesh):
    """
    Example usage

    from utilities.meshing import AnnulusMesh
    from dolfin import *
    from dolfin_adjoint import *
    import matplotlib.pyplot as plt

    annulus = AnnulusMesh(resolution=.089)
    plot(annulus.mesh)
    plt.show()

    """

    def __init__(self, resolution=.0125, path="/home/leonardo_mutti/PycharmProjects/masters_thesis/meshes/annulus/",
                 center=np.array([0, 0]), inner_radius=1, outer_radius=2):

        self.outer_radius = outer_radius

        super().__init__(resolution, path, 2, center, inner_radius)

        if not isinstance(center, np.ndarray):
            raise ValueError("The center must be a numpy array")
        if len(center) != 2:
            raise Exception("The center is not a 2-dimensional point")
        if 0 >= inner_radius or inner_radius >= outer_radius:
            raise Exception("The radii values are not valid")

    def create_mesh(self):
        resolution = self.resolution

        # Create a model to add data to
        model = self.geometry.__enter__()

        # A circle centered at the origin and radius 1
        circle = model.add_circle([self.center[0], self.center[1], 0.0], self.outer_radius,
                                  mesh_size=2.5 * resolution)  # meshes are always 3D, I will suppress the third component in case

        # A hole
        hole = model.add_circle([self.center[0], self.center[1], 0.0], self.inner_radius, mesh_size=1 * resolution)

        # My surface
        plane_surface = model.add_plane_surface(circle.curve_loop, [hole.curve_loop])

        # Sinchronize, before adding physical entities
        model.synchronize()

        # Tagging/marking boundaries and volume. To get a feeling for what "physical" means: https://bthierry.pages.math.cnrs.fr/tutorial/gmsh/basics/physical_vs_elementary/
        # Boundaries with the same tag should be added simultaneously
        model.add_physical([plane_surface], "volume")
        # model.add_physical([circle.curve_loop], "dirichlet_boundary")  # I could separate the two rings and assign different tags...
        # ... like this:
        model.add_physical(circle.curve_loop.curves, "outer_ring")
        model.add_physical(hole.curve_loop.curves, "inner_ring")

        # Generate mesh
        self.geometry.generate_mesh(dim=2)

    def generate_mesh_xdmf(self):
        resolution = self.resolution
        path = self.path

        self.create_mesh()
        self.save_to_msh()
        mesh_from_file = self.load_from_msh()

        # Using the above function, create line and "plane" mesh
        line_mesh = self.mesh_to_meshio(mesh_from_file, "line", prune_z=True)
        meshio.write(path + "facet_mesh_" + str(resolution) + ".xdmf", line_mesh)
        triangle_mesh = self.mesh_to_meshio(mesh_from_file, "triangle", prune_z=True)
        meshio.write(path + "mesh_" + str(resolution) + ".xdmf", triangle_mesh)

    def xdmf_to_dolfin(self):
        mesh_path = self.path
        resolution = self.resolution

        # The volumetric mesh
        mesh = Mesh()
        with XDMFFile(mesh_path + "mesh_" + str(resolution) + ".xdmf") as infile:
            infile.read(mesh)

        # The boundary meshes
        mvc = MeshValueCollection("size_t", mesh, 1)  # 1 means: we consider lines, 1D things
        with XDMFFile(mesh_path + "facet_mesh_" + str(resolution) + ".xdmf") as infile:
            infile.read(mvc, "name_to_read")
        mf = MeshFunction("size_t", mesh, mvc)  # remember, tag 3 is inner ring, tag 2 outer ring

        self.mesh = mesh
        self.facet_function = mf

        return mesh, mf

    def boundary_radial_function(self, x):
        if not isinstance(x, np.ndarray):
            raise Exception("The query point must be a numpy array")
        return self.outer_radius * np.ones(x.shape[0])


class SquareAnnulusMesh(AbstractMesh):

    def __init__(self, resolution=.0125,
                 path="/home/leonardo_mutti/PycharmProjects/masters_thesis/meshes/square_annulus/", inner_radius=1,
                 side_length=4):

        self.side_length = side_length
        center = np.array([0, 0])
        super().__init__(resolution, path, 2, center, inner_radius)

        if not isinstance(center, np.ndarray):
            raise ValueError("The center must be a numpy array")
        if len(center) != 2:
            raise Exception("The center is not a 2-dimensional point")
        if 0 >= inner_radius or inner_radius >= side_length / 2:
            raise Exception("The radius/side length values are not valid")

    def create_mesh(self):
        L = self.side_length
        H = self.side_length
        c = [self.center[0], self.center[1], 0]
        r = self.inner_radius

        resolution = self.resolution

        # Create a model to add data to
        model = self.geometry.__enter__()

        # The square
        points = [model.add_point((c[0] - L / 2, c[1] - H / 2, 0), mesh_size=2.5 * resolution),
                  model.add_point((c[0] + L / 2, c[1] - H / 2, 0), mesh_size=2.5 * resolution),
                  model.add_point((c[0] + L / 2, c[1] + H / 2, 0), mesh_size=2.5 * resolution),
                  model.add_point((c[0] - L / 2, c[1] + H / 2, 0), mesh_size=2.5 * resolution)]

        # Add lines between all points creating the rectangle
        channel_lines = [model.add_line(points[i], points[i + 1])
                         for i in range(-1, len(points) - 1)]

        channel_loop = model.add_curve_loop(channel_lines)

        # A hole
        hole = model.add_circle(c, r, mesh_size=1 * resolution)

        # My surface
        plane_surface = model.add_plane_surface(channel_loop, [hole.curve_loop])

        # Sinchronize, before adding physical entities
        model.synchronize()

        model.add_physical([plane_surface], "volume")
        model.add_physical([channel_lines[0], channel_lines[1], channel_lines[3], channel_lines[2]], "outer_boundary")
        model.add_physical(hole.curve_loop.curves, "inner_ring")

        # Generate the mesh
        # Generate mesh
        self.geometry.generate_mesh(dim=2)

    def generate_mesh_xdmf(self):
        resolution = self.resolution
        path = self.path

        self.create_mesh()
        self.save_to_msh()
        mesh_from_file = self.load_from_msh()

        # Using the above function, create line and "plane" mesh
        line_mesh = self.mesh_to_meshio(mesh_from_file, "line", prune_z=True)
        meshio.write(path + "facet_mesh_" + str(resolution) + ".xdmf", line_mesh)
        triangle_mesh = self.mesh_to_meshio(mesh_from_file, "triangle", prune_z=True)
        meshio.write(path + "mesh_" + str(resolution) + ".xdmf", triangle_mesh)

    def xdmf_to_dolfin(self):
        mesh_path = self.path
        resolution = self.resolution

        # The volumetric mesh
        mesh = Mesh()
        with XDMFFile(mesh_path + "mesh_" + str(resolution) + ".xdmf") as infile:
            infile.read(mesh)

        # The boundary meshes
        mvc = MeshValueCollection("size_t", mesh, 1)  # 1 means: we consider lines, 1D things
        with XDMFFile(mesh_path + "facet_mesh_" + str(resolution) + ".xdmf") as infile:
            infile.read(mvc, "name_to_read")
        mf = MeshFunction("size_t", mesh, mvc)  # remember, tag 3 is inner ring, tag 2 outer ring

        self.mesh = mesh
        self.facet_function = mf

        return mesh, mf

    def boundary_radial_function(self, x):
        if not isinstance(x, np.ndarray):
            raise Exception("The query point must be a numpy array")
        return radial_function_to_square(x, self.side_length)


class CircleMesh(AbstractMesh):
    """
    Example usage

    from utilities.meshing import CircleMesh
    from dolfin import *
    from dolfin_adjoint import *
    import matplotlib.pyplot as plt

    circle = CircleMesh(resolution=.2)
    plot(circle.mesh)
    plt.show()

    """

    def __init__(self, resolution=.2, path="/home/leonardo_mutti/PycharmProjects/masters_thesis/meshes/circle/"):
        super().__init__(resolution, path, 2, np.array([0, 0]), 1)

    def create_mesh(self):
        resolution = self.resolution

        # An empty geometry
        geometry = pygmsh.geo.Geometry()
        # Create a model to add data to
        model = geometry.__enter__()

        # A circle centered at the origin and radius 1
        circle = model.add_circle([0.0, 0.0, 0.0], 1.0,
                                  mesh_size=resolution)  # meshes are always 3D, I will suppress the third component in case

        # Sinchronize, before adding physical entities
        model.synchronize()

        # (this could also be done in gmsh with the GUI)

        # Tagging/marking boundaries and volume. To get a feeling for what "physical" means: https://bthierry.pages.math.cnrs.fr/tutorial/gmsh/basics/physical_vs_elementary/
        # Boundaries with the same tag should be added simultaneously

        model.add_physical(circle.curve_loop.curves, "circle")

        # Generate the mesh
        geometry.generate_mesh(dim=1)

    def generate_mesh_xdmf(self):
        resolution = self.resolution
        path = self.path

        self.create_mesh()
        self.save_to_msh()
        mesh_from_file = self.load_from_msh()

        line_mesh = self.mesh_to_meshio(mesh_from_file, "line", prune_z=True)
        meshio.write(path + "mesh_" + str(resolution) + ".xdmf", line_mesh)

    def xdmf_to_dolfin(self):
        mesh_path = self.path
        resolution = self.resolution

        mesh = Mesh()
        with XDMFFile(mesh_path + "mesh_" + str(resolution) + ".xdmf") as infile:
            infile.read(mesh)

        self.mesh = mesh

        return mesh

    def boundary_radial_function(self, x):
        if not isinstance(x, np.ndarray):
            raise Exception("The query point must be a numpy array")
        return np.ones(x.shape[0])
