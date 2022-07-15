from dolfin import *
# from dolfin_adjoint import *
import matplotlib.pyplot as plt
import numpy as np

# %% Define a very simple mesh

mesh = UnitSquareMesh(4, 4)
plot(mesh)
plt.show()


# %% Define a very simple user expression, in 2D
# We are following https://fenicsproject.org/qa/13607/define-function-from-an-expression-with-variable-parameter/
# The above shows how to create a vector field from a user defined expression

class deformation_expression(UserExpression):
    def __init__(self, xy_displacement=None, **kwargs):
        """ Construct the source function """
        super().__init__(self, **kwargs)
        self.x_displ = xy_displacement[0]
        self.y_displ = xy_displacement[1]

    def eval(self, values, x):
        """ Evaluate the source function """
        if .5 - DOLFIN_EPS < x[0] < .5 + DOLFIN_EPS:
            values[0] = float(self.x_displ)
            values[1] = float(self.y_displ)
        else:
            values[0] = 0.0
            values[1] = 0.0

    def value_shape(self):
        return (2,)

# %% Create a vector valued field

VD = VectorFunctionSpace(mesh, "CG", 1) # mesh, family, degree, https://fenicsproject.org/olddocs/dolfin/1.4.0/python/programmers-reference/functions/functionspace/VectorFunctionSpace.html
deformation = deformation_expression(xy_displacement = [Constant(.125), Constant(-.125)] ,element = VD.ufl_element())
deformation.strength = Constant(3.0)
D = interpolate(deformation, VD)

# %% Move that mesh

ALE.move(mesh,D)    # Note, this will move also the vector field D! So, if I do this again, the mesh will move again!

plot(mesh)
plt.show()
