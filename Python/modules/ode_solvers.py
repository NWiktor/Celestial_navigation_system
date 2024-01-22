# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" Summary of this code file goes here. The purpose of this module can be
expanded in multiple sentences. Below a short free-text summary of the included
classes and functions to give an overview. More detailed summary of the
functions can be provided inside the function's body.

Libs
----
* some_module - This is used for imported, non-standard modules, to help track
    dependencies. Summary is not needed.

Help
----
* https://en.wikipedia.org/wiki/Truncated_icosahedron

Contents
--------
"""

# Standard library imports
# First import should be the logging module if any!
import math

# Third party imports
import numpy as np

# Local application imports
# from logger import MAIN_LOGGER as L


def rk4(func, t0, y0, h, *args):
    """ The fourth-order Runge-Kutta method approximates the solution (function) of a first-order ODE.

    Given a y'(t) = f(t, y(t), *args) ODE, and known an initial value of the solution (initial condition) as y(t0)=y0.
    Short explanation: Putting numbers in the ODE gives us the derivative of the solution function at a given place.
    Using this derivative, we can approximate linearly the next value of the solution function. By doing this
    repeteadly, we numerically integrated the ODE, and created the solution function.

    func - ODE function
    t0 - initial time - arbitrary
    y0 - function (solution) value at t0 - initial condition
    h - timestep
    *args - additional constants, or parameters, which is needed for the ODE function

    https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html
    https://www.youtube.com/watch?v=TzX6bg3Kc0E&list=PLOIRBaljOV8hBJS4m6brpmUrncqkyXBjB&index=5
    """
    k1 = func(t0, y0, *args)
    k2 = func(t0 + 0.5 * h, y0 + 0.5 * k1 * h, *args)
    k3 = func(t0 + 0.5 * h, y0 + 0.5 * k2 * h, *args)
    k4 = func(t0 + h, y0 + k3 * h, *args)

    # Returns y1 value, which is the approximation of the y(t) solution-function at t1:
    return y0 + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), k4[3:6]


def convert_spherical_to_cartesian_coords(radius, theta, phi):
    """
    theta = latitude, rad (-90° to +90°)
    phi = longitude, rad (0° - 360°)
    https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
    https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#To_Cartesian_coordinates_2
    """
    x = radius * math.cos(theta) * math.cos(phi)
    y = radius * math.cos(theta) * math.sin(phi)
    z = radius * math.sin(theta)
    return x, y, z


def rotation_z(angle):
    """ Principal Z axis active rotation matrix by an angle. """
    return np.array([[math.cos(angle), -math.sin(angle), 0],
                     [math.sin(angle), math.cos(angle), 0],
                     [0, 0, 1]])


# Include guard
if __name__ == '__main__':
    pass
