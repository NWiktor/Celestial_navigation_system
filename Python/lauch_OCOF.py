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
    """ Rocket engine class, defined by name and specific_impulse. """

    def __init__(self, name, thrust: float, specific_impulse: float):
        self.name = name
        self.thrust = thrust  # N aka kg/m/s
        self.specific_impulse = specific_impulse  # s


class Stage:
    """ Rocket stage class, defined by engine, number of engines, and burn duration. """

    def __init__(self, engine: Engine, number_of_engines: int, duration: int):
        self.thrust = engine.thrust * number_of_engines  # N aka kg/m/s
        self.duration = duration  # s
        self.specific_impulse = engine.specific_impulse  # s


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


class LaunchSite:

    def __init__(self, name, latitude: float, longitude: float, distance_from_barycenter: float, atmosphere: Atmosphere):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.distance_from_barycenter = distance_from_barycenter
        self.atmosphere = atmosphere


class SpaceCraft:
    """  """

    def __init__(self, name, mass_0, coefficient_of_drag, area, stage1, stage2):
        self.name = name
        self.mass = mass_0  # Starting mass, kg
        self.coefficient_of_drag = coefficient_of_drag  # -
        self.area = area  # Cross-sectional area, m2

        # Stage specs
        self.stage1 = stage1
        self.stage2 = stage2

        # Dynamic vectors
        self.position = np.array([0, 0, 0])
        self.velocity = np.array([0, 0, 0])
        self.acceleration = np.array([0, 0, 0])

    def thrust(self, time):
        """  """

        if time <= self.stage1.duration:
            return self.stage1.thrust / self.mass

        elif time <= self.stage2.duration:
            return self.stage2.thrust / self.mass

        return 0

    def drag(self, air_density, velocity):
        return self.coefficient_of_drag * self.area * air_density * pow(velocity, 2) / 2

    def gravity(self, distance):
        return standard_gravitational_parameter / pow(int(distance), 2)

    def delta_m(self, time):
        """  """
        if time <= self.stage1.duration:
            return self.stage1.thrust / (self.stage1.specific_impulse * standard_gravity)

        elif time <= self.stage2.duration:
            return self.stage2.thrust / (self.stage2.specific_impulse * standard_gravity)

        return 0

    def launch(self, launch_site: LaunchSite, time):
        """  """

        # Yield initial values
        yield self.position, self.velocity, self.acceleration, self.mass
        print("___INITIAL_CONDITIONS___")
        print(f"{self.position=}")
        print(f"{self.velocity=}")
        print(f"{self.acceleration=}")
        print(f"{self.mass=}")

        for i in range(0, time):
            # 1 second has passed
            air_density = launch_site.atmosphere.get_density(self.position[2])
            altitude = self.position[2] + launch_site.distance_from_barycenter

            # Calculate new acceleration
            print(self.gravity(altitude))
            print(self.thrust(i))
            print(self.drag(air_density, self.velocity[2]))


            self.acceleration[2] = (self.thrust(i) - self.drag(air_density, self.velocity[2])
                                     - self.gravity(altitude))
            self.velocity[2] += self.acceleration[2]  # Calculate new velocity
            self.position[2] += self.velocity[2]  # Calculate new elevation

            # Calculate new spacecraft mass
            self.mass -= self.delta_m(i)

            yield self.position, self.velocity, self.acceleration, self.mass
            print(f"___STEP {i}___")
            print(f"{self.position=}")
            print(f"{self.velocity=}")
            print(f"{self.acceleration=}")
            print(f"{self.mass=}")


# Main function for module testing
def main():
    """  """
    # Place
    atmosphere = Atmosphere()
    cape = LaunchSite("Cape Canaveral, Earth", 28.3127, 80.3903, 6371000, atmosphere)

    # Starship hardware specs:
    raptor3 = Engine("Raptor 3", 2.64*pow(10, 6), 327)
    raptor3_vac = Engine("Raptor 3 vac", 2.64*pow(10, 6), 380)
    booster = Stage(raptor3, 33, 159)
    starship = Stage(raptor3_vac, 3, 400)
    oft3 = SpaceCraft("Starship", 5000000, 1.5, m.pi * pow(9, 2)/4, booster, starship)

    # Launch
    i = 0
    time_limit = 900
    x_data = []
    alt_data = []
    vel_data = []
    acc_data = []
    mass_data = []

    for p, v, a, mass in oft3.launch(cape, 900):
        x_data.append(i)
        alt_data.append(p[2]/1000)
        vel_data.append(v[2])
        acc_data.append(a[2])
        mass_data.append(mass/1000)
        i += 1

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='constrained', figsize=(8, 8))
    fig.suptitle("Spacecraft")
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Altitude")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_xlim(0, time_limit)
    ax1.set_ylim(0, 2500)
    ax1.scatter(x_data, alt_data, s=0.5)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Velocity")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_xlim(0, time_limit)
    ax2.scatter(x_data, vel_data, s=0.5)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Acceleration")
    ax3.set_ylabel("Acceleration (m/s^2)")
    ax3.set_xlim(0, time_limit)
    ax3.scatter(x_data, acc_data, s=0.5)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Spacecraft mass")
    ax4.set_ylabel("Mass (ton)")
    ax4.set_xlim(0, time_limit)
    ax4.set_ylim(0, 5000)
    ax4.scatter(x_data, mass_data, s=0.5)

    plt.show()


# Include guard
if __name__ == '__main__':
    main()
