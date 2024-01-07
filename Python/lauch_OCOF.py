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
# import sys
# import datetime
# import time
import math as m

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from logger import MAIN_LOGGER as l
# from modules.time_functions import julian_date, j2000_date

# Class initializations and global variables
gravitational_constant = 6.67430 * pow(10, -11)  # m^3 kg-1 s-2
standard_gravitational_parameter = 3.986004418 * pow(10, 14)  # m^3/s^2
standard_gravity = 9.81  # m/s^2


# Class and function definitions

class Engine:
    """ Rocket engine class, defined by name, specific_impulse and burn_duration. """

    def __init__(self, name, thrust, specific_impulse, duration):
        self.name = name
        self.thrust = thrust  # N
        self.specific_impulse = specific_impulse  # s
        self.duration = duration  # s


class Atmosphere:

    @staticmethod
    def get_density(altitude):
        """Calculates air density in function of height on Earth, measured from sea level.
        https://en.wikipedia.org/wiki/Density_of_air
        """
        rho_null = 1.204  # kg/m3
        height_scale = 10.4  # km
        if altitude <= 10000:
            return rho_null * m.exp(-altitude/1000/height_scale)

        return 0


class SpaceCraft:
    """  """

    def __init__(self, name, mass, coefficient_of_drag, area, engine, number_of_engines):
        self.name = name
        self.mass = mass  # kg
        self.coefficient_of_drag = coefficient_of_drag  # -
        self.area = area  # m2

        # Engine specs
        self.engine = engine
        self.number_of_engines = number_of_engines
        self.thrust = number_of_engines * self.engine.thrust

    def clear(self):
        """ Erases all loaded and existing training data to allow refresh. """
        self.__dict__ = {}

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class LaunchSite:

    def __init__(self, name, latitude, longitude, distance_from_object_center, atmosphere):
        pass


class Launch:
    """  """

    def __init__(self, name, spacecraft: SpaceCraft, launchsite: LaunchSite):
        self.name = name
        self.spacecraft = spacecraft
        self.launchsite = launchsite

    def generate_points(self, duration):
        position = np.array([0, 0, 0])
        velocity = np.array([0, 0, 0])
        mass = self.spacecraft.mass
        earth_radius = 6371000  # m
        alt = 0  # m
        vel = 0  # m/s

        for i in range(0, duration):
            # Calculate new mass
            # print(mass)  # Current mass
            if i <= self.spacecraft.engine.duration:
                mass -= self.spacecraft.thrust / (self.spacecraft.engine.specific_impulse * standard_gravity)

                acc = (self.spacecraft.thrust/mass + standard_gravitational_parameter/pow(earth_radius + alt, 2)
                       - self.spacecraft.coefficient_of_drag * self.spacecraft.area * Atmosphere.get_density(alt))

            else:
                acc = (standard_gravitational_parameter/pow(earth_radius + alt, 2)
                       - self.spacecraft.coefficient_of_drag * self.spacecraft.area * Atmosphere.get_density(alt))

            vel += acc
            alt += vel

            # Calculate acceleration

            yield acc



# Main function for module testing
def main():
    """  """
    raptor3 = Engine("Raptor 3", 1.81*pow(10, 6), 327, 200)
    legkor = Atmosphere()
    starship = SpaceCraft("Starship", 5000000, 1.14, m.pi * pow(9, 2)/4, raptor3, 33)
    cape = LaunchSite("cape", 10, 20,30, legkor)

    kiloves = Launch("proba", starship, cape)

    i = 0
    x_data = []
    z_data = []

    for val in kiloves.generate_points(300):
        print(val)
        x_data.append(i)
        z_data.append(val)
        i += 1




    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()  # projection='3d')
    ax.set_xlim(0, 500)
    # ax.set_ylim(0, 6000000)
    # ax.set_zlim3d(-150000000, 150000000)

    # celestial_bodies = []

    # for ceb in celestial_bodies:
        # x_data: list = []
        # y_data: list = []
        # z_data: list = []

        # for i in range(0, 365):
            # vector = ceb.get_position(i)
            # x_data.append(vector[0])
            # y_data.append(vector[1])
            # z_data.append(vector[2])

    ax.scatter(x_data, z_data)

    plt.show()


# Include guard
if __name__ == '__main__':
    main()
