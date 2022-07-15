#%%######################
# Generating an annulus #
#########################

# Adapted from http://jsdokken.com/src/pygmsh_tutorial.html#first

import pygmsh
import gmsh
import meshio
import numpy

path = "/home/leonardo_mutti/PycharmProjects/masters_thesis/meshes/annulus/"

# %%
resolution = .0125

# An empty geometry
geometry = pygmsh.geo.Geometry()
# Create a model to add data to
model = geometry.__enter__()

# A circle centered at the origin and radius 1
circle = model.add_circle([0.0, 0.0, 0.0], 2.0, mesh_size=5*resolution)  # meshes are always 3D, I will suppress the third component in case

# A hole
hole = model.add_circle([0.0, 0.0, 0.0], 1.0, mesh_size=1*resolution)

# My surface
plane_surface = model.add_plane_surface(circle.curve_loop,[hole.curve_loop])

# Sinchronize, before adding physical entities
model.synchronize()

#%% (this could also be done in gmsh with the GUI)

# Tagging/marking boundaries and volume. To get a feeling for what "physical" means: https://bthierry.pages.math.cnrs.fr/tutorial/gmsh/basics/physical_vs_elementary/
# Boundaries with the same tag should be added simultaneously
model.add_physical([plane_surface], "volume")
# model.add_physical([circle.curve_loop], "dirichlet_boundary")  # I could separate the two rings and assign different tags...
# ... like this:
model.add_physical(circle.curve_loop.curves, "outer_ring")
model.add_physical(hole.curve_loop.curves, "inner_ring")

# Generate the mesh
geometry.generate_mesh(dim=2)
geometry.save_geometry(path+"annulus_"+str(resolution)+".geo_unrolled")  # not exactly working on gmsh though...
gmsh.write(path+"mesh_"+str(resolution)+".msh")
gmsh.clear()
geometry.__exit__()

# You can now check out the result: open gmsh, import the mesh, view it

#%%##########################
# Saving to a better format #
#############################

mesh_from_file = meshio.read(path+"mesh_"+str(resolution)+".msh")

# We now extract cells and physical data... because we want the mesh and the lines to apply Dirichlet conditions onto
# We therefore generate two files: one for the cells, one for the facets (lines)
# This is a 2D mesh and we thus need to prune out the z values

# This link might help to understand better what is going on: https://github.com/iitrabhi/GSoC2019/blob/master/Notebooks/Understanding%20MeshFunction%20and%20MeshValueCollection.ipynb
def create_mesh(mesh, cell_type, prune_z = False):
    cells = mesh.get_cells_type(cell_type)  # get the cells of some type: it will change!
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

# Using the above function, create line and "plane" mesh
line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(path+"facet_mesh_"+str(resolution)+".xdmf", line_mesh)
triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(path+"mesh_"+str(resolution)+".xdmf", triangle_mesh)
