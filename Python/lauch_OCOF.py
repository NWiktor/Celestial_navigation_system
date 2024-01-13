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

# Class and function definitions


class Engine:
    """ Rocket engine class, defined by name and specific_impulse. """

    def __init__(self, name, thrust: float, specific_impulse: float):
        self.name = name
        self.thrust = thrust  # N aka kg/m/s
        self.specific_impulse = specific_impulse  # s


class Stage:
    """ Rocket stage class, defined by engine, number of engines, and max. burn duration. """

    def __init__(self, engine: Engine, empty_mass: float, propellant_mass: float,
                 number_of_engines: int, duration: int):
        self.thrust = engine.thrust * number_of_engines  # N aka kg/m/s
        self.duration = duration  # s
        self._specific_impulse = engine.specific_impulse  # s
        self._empty_mass = empty_mass  # kg
        self._propellant_mass = propellant_mass  # kg

    def get_mass(self):
        """ Returns the actual total mass of the stage. """
        return self._empty_mass + self._propellant_mass

    def burn_mass(self, standard_gravity, burn_time=1):
        """ Calculates delta m after burning the engine for 1 seconds, and sets new mass. """
        delta_m = self.thrust / (self._specific_impulse * standard_gravity) * burn_time

        # Updates itself with new mass, negative values are omitted
        self._propellant_mass = max(0.0, self._propellant_mass - delta_m)


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

    def __init__(self, name, latitude: float, longitude: float, distance_from_barycenter: float,
                 atmosphere: Atmosphere, standard_gravity, standard_gravitational_parameter):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.distance_from_barycenter = distance_from_barycenter
        self.atmosphere = atmosphere
        self.std_gravity = standard_gravity  # 9.81 # m/s^2
        self.std_gravitational_parameter = standard_gravitational_parameter  # 3.986004418 * pow(10, 14) # m^3/s^2


class SpaceCraft:
    """  """

    def __init__(self, name, payload_mass: float, coefficient_of_drag: float, diameter: float, stages: list):
        self.name = name

        # Stage specifications and Mass properties
        self.stage_status = 1
        self.stages = stages
        self.payload_mass = payload_mass
        self.total_mass = self.payload_mass + self.get_stage_mass(self.stage_status)

        # Physical properties
        self.coefficient_of_drag = coefficient_of_drag  # -
        self.diameter = diameter
        self.area = m.pi * pow(diameter, 2) / 4  # Cross-sectional area, m2

        # Dynamic vectors
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])

    def get_stage_mass(self, stage_status):
        """  """
        if stage_status == 1:
            return self.stages[0].get_mass() + self.stages[1].get_mass()

        elif stage_status == 2:
            return self.stages[1].get_mass()

        return 0

    def thrust(self, stage_status):
        """  """

        if stage_status == 1:
            return self.stages[0].thrust / self.total_mass

        elif stage_status == 2:
            return self.stages[1].thrust / self.total_mass

        return 0

    def drag(self, air_density, velocity):
        return self.coefficient_of_drag * self.area * air_density * pow(velocity, 2) / 2

    def gravity(self, std_gravitational_parameter, distance):
        return std_gravitational_parameter / pow(int(distance), 2)

    def update_mass(self, standard_gravity, stage_status):
        """  """

        # Calculate delta m in stage 1
        if stage_status == 1:
            self.stages[0].burn_mass(standard_gravity)

        # Calculate delta m in stage 2
        elif stage_status == 2:
            self.stages[1].burn_mass(standard_gravity)

        # Calculate new total mass
        self.total_mass = self.payload_mass + self.get_stage_mass(stage_status)

    def launch(self, launch_site: LaunchSite, separation_time_1, separation_time_2):
        """  """

        stage1_separation = min(separation_time_1, self.stages[0].duration)
        stage2_separation = min(separation_time_2, self.stages[0].duration + self.stages[1].duration)
        time = int(stage2_separation * 1.5)

        # Yield initial values
        yield self.position, self.velocity, self.acceleration, self.total_mass
        print("___INITIAL_CONDITIONS___")
        print(f"{self.position=}")
        print(f"{self.velocity=}")
        print(f"{self.acceleration=}")
        print(f"{self.total_mass=}")

        # Calculate status parameters, each step is 1 second
        for i in range(0, time):

            #  Calculate stage according to time
            if i <= stage1_separation:
                self.stage_status = 1
            elif i <= stage2_separation:
                self.stage_status = 2
            else:
                self.stage_status = 3

            air_density = launch_site.atmosphere.get_density(self.position[2])
            altitude = self.position[2] + launch_site.distance_from_barycenter

            self.acceleration[2] = (self.thrust(self.stage_status)
                                    - self.drag(air_density, self.velocity[2])
                                    - self.gravity(launch_site.std_gravitational_parameter, altitude))
            self.velocity[2] += self.acceleration[2]  # Calculate new velocity
            self.position[2] += self.velocity[2]  # Calculate new elevation

            # Calculate new spacecraft mass
            self.update_mass(launch_site.std_gravity, self.stage_status)

            yield self.position, self.velocity, self.acceleration, self.total_mass


# Main function for module testing
def main():
    """  """
    # Place
    atmosphere = Atmosphere()
    cape = LaunchSite("Cape Canaveral, Earth", 28.3127, 80.3903, 6371000,
                      atmosphere, 9.81, 3.986004418 * pow(10, 14))

    # Starship hardware specs:
    # raptor3 = Engine("Raptor 3", 2.64*pow(10, 6), 327)
    # raptor3_vac = Engine("Raptor 3 vac", 2.64*pow(10, 6), 380)
    # booster = Stage(raptor3, 33, 159)
    # starship = Stage(raptor3_vac, 3, 400)
    # oft3 = SpaceCraft("Starship", 5000000, 1.5, 9, booster, starship)

    # Falcon9 hardware specs:
    merlin1d_p = Engine("Merlin 1D+", 934 * pow(10, 3), 283)
    merlin1d_vac = Engine("Merlin 1D vac", 934 * pow(10, 3), 348)
    first_stage = Stage(merlin1d_p, 25600, 395700, 9, 162)
    second_stage = Stage(merlin1d_vac, 3900, 92670, 1, 397)
    falcon9 = SpaceCraft("Falcon 9", 22800, 1.5, 5.2, [first_stage, second_stage])

    # Launch
    i = 0
    time_limit = 900
    x_data = []
    alt_data = []
    vel_data = []
    acc_data = []
    mass_data = []

    for p, v, a, mass in falcon9.launch(cape, 120, 500):
        x_data.append(i)
        alt_data.append(p[2]/1000)
        vel_data.append(v[2])
        acc_data.append(a[2]/9.81)
        mass_data.append(mass/1000)
        i += 1

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='constrained', figsize=(19, 9.5))
    fig.suptitle("Falcon9 launch from Cape")
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Altitude - time")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_xlim(0, time_limit)
    # ax1.set_ylim(0, 5000)
    ax1.scatter(x_data, alt_data, s=0.5)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Velocity - time")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_xlim(0, time_limit)
    ax2.set_ylim(0, 30000)
    ax2.scatter(x_data, vel_data, s=0.5)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Acceleration - time")
    ax3.set_ylabel("Acceleration (g)")
    ax3.set_xlim(0, time_limit)
    ax3.set_ylim(0, 10)
    ax3.scatter(x_data, acc_data, s=0.5)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Spacecraft mass - time")
    ax4.set_ylabel("Mass (ton)")
    ax4.set_xlim(0, time_limit)
    ax4.set_ylim(0, 600)
    ax4.scatter(x_data, mass_data, s=0.5)

    # Force plot to full screen, and show it
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.show()


# Include guard
if __name__ == '__main__':
    main()
