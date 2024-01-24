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
import math as m

# Third party imports
import numpy as np

# Local application imports
# from logger import MAIN_LOGGER as L


def runge_kutta_4(func, t0, y0, h, *args):
    """ The fourth-order Runge-Kutta method approximates the solution (function) of a first-order ODE.

    Given a y'(t) = f(t, y(t), *args) ODE, and known an initial value of the solution (initial condition) as y(t0)=y0.
    Short explanation: Putting numbers in the ODE gives us the derivative of the solution function at a given place.
    Using this derivative, we can approximate linearly the next value of the solution function. By doing this
    repeteadly, we numerically integrated the ODE, and created the solution function.

    func - ODE function
    t0 - initial time - arbitrary
    y0 - function (solution) value at t0 - initial condition
    h - timestep
    *args - additional constants, or parameters, which is needed for the ODE function (optional)

    https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html
    https://www.youtube.com/watch?v=TzX6bg3Kc0E&list=PLOIRBaljOV8hBJS4m6brpmUrncqkyXBjB&index=5
    """
    k1 = func(t0, y0, *args)
    k2 = func(t0 + 0.5 * h, y0 + 0.5 * k1 * h, *args)
    k3 = func(t0 + 0.5 * h, y0 + 0.5 * k2 * h, *args)
    k4 = func(t0 + h, y0 + k3 * h, *args)

    # Returns y1 value, which is the approximation of the y(t) solution-function at t1; and k4 which is
    # the derivative of y1 at t1
    return y0 + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), k4[3:6]


def convert_spherical_to_cartesian_coords(radius: float, theta: float, phi: float):
    """
    Convert spherical coordinates to cartesian coordinates.
    theta - latitude, rad (-90° to +90°)
    phi - longitude, rad (0° - 360°)
    https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
    https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#To_Cartesian_coordinates_2
    """
    x = radius * m.cos(theta) * m.cos(phi)
    y = radius * m.cos(theta) * m.sin(phi)
    z = radius * m.sin(theta)
    return x, y, z


def rotation_z(angle):
    """ Principal Z axis active rotation matrix by an angle. """
    return np.array([[m.cos(angle), -m.sin(angle), 0],
                     [m.sin(angle), m.cos(angle), 0],
                     [0, 0, 1]])


# ReST syntax generates 'invalid-escape-sequence' warning
# pylint: disable = anomalous-backslash-in-string
def rodrigues_rotation(vector_v, vector_k, theta):
    """Implements Rodrigues rotation.

    Calculates the rotated V vector (V\ :sub:`rot`\), based on Rodrigues
    rotation formula. The axis of rotation is K, and the rotation angle is
    theta according to the right hand rule:

    .. math::
      \\vec{V_{rot}} = \\vec{V} \\cos\\theta + (\\vec{K} \\times \\vec{V})
      \\sin\\phi + \\vec{K} (\\vec{K} \cdot \\vec{V}) (1 - \\cos\\theta)

    :param vector_v: V vector (Numpy).
    :param vector_k: Unit vector K (rotational axis) (Numpy).
    :param float theta: Rotational angle around vector K.
    :return: Rotated V vector (Numpy).

    """
    v_rot = ((vector_v * m.cos(theta)) + (np.cross(vector_k, vector_v) * m.sin(theta))
             + (vector_k * np.dot(vector_k, vector_v) * (1 - m.cos(theta))))
    return v_rot


# Include guard
if __name__ == '__main__':
    # TODO: add pytests
    pass
