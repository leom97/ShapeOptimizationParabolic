import pygmsh
import gmsh
import meshio
import abc

from dolfin import *
from dolfin_adjoint import *


class AbstractMesh:
    # Adapted from http://jsdokken.com/src/pygmsh_tutorial.html#first
    __metaclass__ = abc.ABCMeta

    def __init__(self, resolution, path):
        self.resolution = resolution
        self.path = path

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

    def __init__(self, resolution=.0125, path="/home/leonardo_mutti/PycharmProjects/masters_thesis/meshes/annulus/"):
        super().__init__(resolution, path)

    def create_mesh(self):
        resolution = self.resolution

        # Create a model to add data to
        model = self.geometry.__enter__()

        # A circle centered at the origin and radius 1
        circle = model.add_circle([0.0, 0.0, 0.0], 2.0,
                                  mesh_size=2.5 * resolution)  # meshes are always 3D, I will suppress the third component in case

        # A hole
        hole = model.add_circle([0.0, 0.0, 0.0], 1.0, mesh_size=1 * resolution)

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
        super().__init__(resolution, path)

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

        # %% (this could also be done in gmsh with the GUI)

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
