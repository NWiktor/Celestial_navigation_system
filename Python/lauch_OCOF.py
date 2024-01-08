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
        self.thrust = thrust  # N aka kg/m/s
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
        if altitude <= 18000:
            return rho_null * m.exp(-altitude*1000/height_scale)
        else:
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

    def __init__(self, name, latitude, longitude, distance_from_barycenter, atmosphere):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.distance_from_barycenter = distance_from_barycenter
        self.atmosphere = atmosphere


class Launch:
    """  """

    def __init__(self, name, spacecraft: SpaceCraft, launchsite: LaunchSite):
        self.name = name
        self.spacecraft = spacecraft
        self.launchsite = launchsite

    def generate_points(self, duration):
        """  """
        position_0 = np.array([0, 0, 0])
        velocity_0 = np.array([0, 0, 0])
        position = np.array([0, 0, 0])
        velocity = np.array([0, 0, 0])

        mass = self.spacecraft.mass
        earth_radius = 6371000  # m
        # acc = 0  # m/s2
        vel = 0  # m/s
        alt = 0  # m

        for i in range(0, duration):

            if i <= self.spacecraft.engine.duration:
                # Calculate new mass
                mass -= self.spacecraft.thrust / (self.spacecraft.engine.specific_impulse * standard_gravity)

                print("Thrust: ", self.spacecraft.thrust/mass)
                print("Gravity: ", standard_gravitational_parameter/pow(earth_radius + alt, 2))
                print("Drag: ", - self.spacecraft.coefficient_of_drag * self.spacecraft.area * Atmosphere.get_density(alt)*pow(vel, 2)/2)

                # Calculate acceleration
                acc = (self.spacecraft.thrust/mass - standard_gravitational_parameter/pow(earth_radius + alt, 2)
                       - self.spacecraft.coefficient_of_drag * self.spacecraft.area
                       * Atmosphere.get_density(alt)*pow(vel, 2)/2)

            else:
                acc = (- standard_gravitational_parameter/pow(earth_radius + alt, 2)
                       - self.spacecraft.coefficient_of_drag * self.spacecraft.area
                       * Atmosphere.get_density(alt)*pow(vel, 2)/2)

            vel += acc
            alt += vel
            print("Mass:", mass)
            print("Acc:", acc)
            print("Velocity:", vel)
            print("Altitude:", alt)

            yield alt, acc, mass


# Main function for module testing
def main():
    """  """
    raptor3 = Engine("Raptor 3", 1.81*pow(10, 6), 327, 200)
    legkor = Atmosphere()

    starship = SpaceCraft("Starship", 5000000, 1.14, m.pi * pow(9, 2)/4, raptor3, 33)
    cape = LaunchSite("Cape Canaveral, Earth", 28.3127, 80.3903, 6371000, legkor)

    kiloves = Launch("proba", starship, cape)

    i = 0
    x_data = []
    alt_data = []
    acc_data = []
    mass_data = []

    for alt, acc, mass in kiloves.generate_points(300):
        x_data.append(i)
        alt_data.append(alt)
        acc_data.append(acc)
        mass_data.append(mass/1000)  # tonna
        i += 1

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2,2,1)  # projection='3d')
    ax.set_xlim(0, 500)
    ax.scatter(x_data, alt_data)
    # ax.set_ylim(0, 6000000)
    # ax.set_zlim3d(-150000000, 150000000)

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.set_xlim(0, 500)
    ax1.scatter(x_data, acc_data)

    ax2 = fig.add_subplot(2, 2, 3)  # projection='3d')
    ax2.set_xlim(0, 500)
    ax2.scatter(x_data, mass_data)

    plt.subplots_adjust(wspace=0.4)
    plt.show()


# Include guard
if __name__ == '__main__':
    main()
