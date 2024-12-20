# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" This module describes the abstract Atmosphere class, and different atmospheric models (e.g. Earth and Mars).

Libs
----
* Matplotlib - for data visualization and test

Help
----
* https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
* https://www.digitaldutch.com/atmoscalc/US_Standard_Atmosphere_1976.pdf
* https://en.wikipedia.org/wiki/Barometric_formula
* http://www.luizmonteiro.com/StdAtm.aspx
* https://www.grc.nasa.gov/www/k-12/airplane/atmosmrm.html

Contents
--------
"""

# Standard library imports
import logging
import math as m
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np

# Local application imports

# Class initializations and global variables
logger = logging.getLogger(__name__)

# Euler angles -> each represents an angular speed
# base CS (OCI) rotated - LVLH (?)
# create a randomly rotating CS
# create function to apply torques (or angles) to stabilize randomly rotating CS
# Simulate detumbling


class Axes(Enum):
    """ Relate indeces to axes. """
    X = 0
    Y = 1
    Z = 2


class DefCS:
    pass


class CoordinateSystemState:
    """ Coordinate system state defined by Euler rotations from the default CS. """

    def __init__(self, name: str, order: str, euler_angles: np.array, angular_velocity: np.array):
        self.name = name
        self.order = order  # i.e.: XYZ or ZXZ
        self.euler_angles = euler_angles
        self.angular_velocity = angular_velocity


def plot_atmosphere(model):
    """ Plot pressure, temperature and density data of the given model, to check its validity. """

    alt = []
    tmp = []
    pres = []
    rho = []
    for i in range(0, 10000):
        data = model.get_atm_params(i * 10)
        alt.append(i*10)
        tmp.append(data[0])
        pres.append(data[1])
        rho.append(data[2])

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='tight', figsize=(19, 9.5))
    fig.suptitle("Atmospheric parameters")
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Temperature")
    ax1.set_xlabel('altitude (m)')
    ax1.set_ylabel('temperature (K)', color="m")
    ax1.plot(alt, tmp, color="m")
    ax1.tick_params(axis='y', labelcolor="m")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Pressure")
    ax2.set_xlabel('altitude (m)')
    ax2.set_ylabel('pressure (kPa)', color="b")
    ax2.plot(alt, pres, color="b")
    ax2.tick_params(axis='y', labelcolor="b")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Air density")
    ax3.set_xlabel('altitude (m)')
    ax3.set_ylabel('air density (kg/m3)', color="g")
    ax3.plot(alt, rho, color="g")
    ax3.tick_params(axis='y', labelcolor="g")

    plt.show()


def module_test():
    pass


# Include guard
if __name__ == '__main__':
    pass
