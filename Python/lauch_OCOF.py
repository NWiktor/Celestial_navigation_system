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
from dataclasses import dataclass
import math as m

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from logger import MAIN_LOGGER as l


# TODO: implement specific impulse variation between sea-level and vacuum
@dataclass
class Engine:
    """ Rocket engine class, defined by name and specific_impulse. """

    def __init__(self, name: str, thrust: float, specific_impulse: float):
        self.name = name
        self.thrust = thrust  # N aka kg/m/s
        self.specific_impulse = specific_impulse  # s
        # self.specific_impulse_vac = specific_impulse_vac  # s


@dataclass
class Stage:
    """ Rocket stage class, defined by engine, number of engines, and max. burn duration. """
    engine: Engine
    _empty_mass: float  # kg
    _propellant_mass: float  # kg
    number_of_engines: int
    duration: int  # s

    def __post_init__(self):
        self.thrust = self.engine.thrust * self.number_of_engines  # N aka kg/m/s
        self._specific_impulse = self.engine.specific_impulse  # s

    def is_propellant(self):
        """ Returns if there is any fuel left in the stage and capable to generate thrust. """
        if self._propellant_mass > 0:
            return True
        return False

    def get_mass(self):
        """ Returns the actual total mass of the stage. """
        return self._empty_mass + self._propellant_mass

    def burn_mass(self, standard_gravity):
        """ Calculates delta m after burning the engine for 1 seconds, and sets new mass. """
        delta_m = self.thrust / (self._specific_impulse * standard_gravity)

        # Updates itself with new mass, negative values are omitted
        self._propellant_mass = max(0.0, self._propellant_mass - delta_m)


@dataclass
class PlanetLocation:
    """ Launch site class, given by longitude, latitude, distance from barycenter, atmosphere via the get_density
    function, gravity and gravitational parameter.
    """

    def __init__(self, name, latitude: float, longitude: float, distance_from_barycenter: float,
                 standard_gravity, standard_gravitational_parameter):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.distance_from_barycenter = distance_from_barycenter  # m
        self.std_gravity = standard_gravity  # m/s^2
        self.std_gravitational_parameter = standard_gravitational_parameter  # m^3/s^2

    # pylint: disable = unused-argument
    def get_density(self, altitude):
        """ Placeholder function for override. """
        return


@dataclass
class EarthLocation(PlanetLocation):
    """ Launch site class for locations on Earth's surface. """

    def __init__(self, name, latitude: float, longitude: float):
        super().__init__(f"{name}, Earth", latitude, longitude, 6371000, 9.80665,
                         3.986004418e14)

    def get_density(self, altitude: float):
        """ Calculates air density in function of height on Earth, measured from sea level.
        https://en.wikipedia.org/wiki/Density_of_air
        """
        if 0 <= altitude <= 100000:
            return 1.204 * m.exp(-altitude / 10400)
        return 0


class SpaceCraft:
    """ Spacecraft class, defined by name, payload mass, drag coefficient and diameter; and stages. """

    def __init__(self, name, payload_mass: float, coefficient_of_drag: float, diameter: float, stages: list):
        self.name = name

        # Stage specifications and mass properties
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
        """ Sums the mass of each rocket stage. """

        if stage_status == 1:
            return self.stages[0].get_mass() + self.stages[1].get_mass()

        if stage_status == 2:
            return self.stages[1].get_mass()

        return 0

    def thrust(self, stage_status):
        """ Calculates actual thrust (force) of the rocket, depending on actual staging. """

        if stage_status == 1 and self.stages[0].is_propellant():
            return self.stages[0].thrust

        if stage_status == 2 and self.stages[1].is_propellant():
            return self.stages[1].thrust

        return 0

    def drag(self, air_density, velocity):
        """ Calculates actual drag (force) on the rocket, depending on the atmospheric density. """
        return self.coefficient_of_drag * self.area * air_density * pow(velocity, 2) / 2

    @staticmethod
    def gravity(std_gravitational_parameter, distance):
        """ Calculates gravitational forces between two bodies, depending on the distance travelled. """
        return std_gravitational_parameter / pow(int(distance), 2)

    def update_mass(self, standard_gravity, stage_status):
        """ Updates total rocket mass after burning, depending on gravity and actual staging. """

        # Calculate delta m in stage 1
        if stage_status == 1:
            self.stages[0].burn_mass(standard_gravity)

        # Calculate delta m in stage 2
        elif stage_status == 2:
            self.stages[1].burn_mass(standard_gravity)

        # Calculate new total mass
        self.total_mass = self.payload_mass + self.get_stage_mass(stage_status)

    # TODO: implement MECO - stage separation time
    def launch(self, launch_site: PlanetLocation, separation_time_1, separation_time_2):
        """ Yield rocket's status parameters during launch, every second. """

        stage1_separation = min(separation_time_1, self.stages[0].duration)
        stage2_separation = min(separation_time_2, self.stages[0].duration + self.stages[1].duration)
        time = int(stage2_separation * 3)

        # Yield initial values
        yield self.position, self.velocity, self.acceleration, self.total_mass, 0, 0, 9.81

        # Calculate status parameters, each step is 1 second
        for i in range(0, time):

            #  Calculate stage according to time
            if i <= stage1_separation:
                self.stage_status = 1
            elif stage2_separation >= i:
                self.stage_status = 2
            else:
                self.stage_status = 3

            # Calculate flight characteristics
            air_density = launch_site.get_density(self.position[2])
            altitude = self.position[2] + launch_site.distance_from_barycenter
            thrust = self.thrust(self.stage_status) / self.total_mass
            drag = self.drag(air_density, self.velocity[2]) / self.total_mass
            gravity = self.gravity(launch_site.std_gravitational_parameter, altitude)

            l.debug("Air density is %s at %s", air_density, self.position[2])
            l.debug("Drag force is %s at %s", drag, self.position[2])

            # Calculate position, velocity and acceleration
            self.acceleration[2] = thrust - drag - gravity
            self.velocity[2] += self.acceleration[2]
            self.position[2] += self.velocity[2]

            # Calculate new spacecraft mass
            self.update_mass(launch_site.std_gravity, self.stage_status)

            yield self.position, self.velocity, self.acceleration, self.total_mass, thrust, drag, gravity


# Main function for module testing
# pylint: disable=too-many-statements, too-many-locals
def main():
    """ Defines a Spacecraft class and LaunchSite, then calculates and plots status parameters. """
    # Launch-site
    cape = EarthLocation("Cape Canaveral", 28.3127, 80.3903)

    # Starship hardware specs:
    # raptor3 = Engine("Raptor 3", 2.64*pow(10, 6), 327)
    # raptor3_vac = Engine("Raptor 3 vac", 2.64*pow(10, 6), 380)
    # booster = Stage(raptor3, 33, 159)
    # starship = Stage(raptor3_vac, 3, 400)
    # oft3 = SpaceCraft("Starship", 5000000, 1.5, 9, booster, starship)

    # Falcon9 hardware specs:
    # https://aerospaceweb.org/question/aerodynamics/q0231.shtml
    # https://en.wikipedia.org/wiki/Falcon_9#Design
    merlin1d_p = Engine("Merlin 1D+", 934e3, 283)
    merlin1d_vac = Engine("Merlin 1D vac", 934e3, 348)
    first_stage = Stage(merlin1d_p, 25600, 395700, 9, 162)
    second_stage = Stage(merlin1d_vac, 3900, 92670, 1, 397)
    falcon9 = SpaceCraft("Falcon 9", 22800, 0.25, 5.2, [first_stage, second_stage])

    # Launch
    i = 0
    time_limit = 1500
    x_data = []
    alt_data = []
    vel_data = []
    acc_data = []
    mass_data = []
    t_data = []
    d_data = []
    g_data = []

    for p, v, a, mass, t, d, g in falcon9.launch(cape, 130, 465):
        x_data.append(i)
        alt_data.append(p[2]/1000)
        vel_data.append(v[2]/1000)
        acc_data.append(a[2]/9.81)
        mass_data.append(mass/1000)
        t_data.append(t)
        d_data.append(d)
        g_data.append(g)
        i += 1

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='constrained', figsize=(19, 9.5))
    fig.suptitle("Falcon9 launch from Cape")
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.set_title("Altitude - time")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_xlim(0, time_limit)
    ax1.set_ylim(0, 3000)
    ax1.scatter(x_data, alt_data, s=0.5)

    ax2 = fig.add_subplot(4, 2, 2)
    ax2.set_title("Velocity - time")
    ax2.set_ylabel("Velocity (km/s)")
    ax2.set_xlim(0, time_limit)
    ax2.set_ylim(0, 8)
    ax2.scatter(x_data, vel_data, s=0.5)

    ax3 = fig.add_subplot(4, 2, 3)
    ax3.set_title("Acceleration - time")
    ax3.set_ylabel("Acceleration (g)")
    ax3.set_xlim(0, time_limit)
    ax3.set_ylim(-2.5, 7.5)
    ax3.scatter(x_data, acc_data, s=0.5)

    ax4 = fig.add_subplot(4, 2, 4)
    ax4.set_title("Spacecraft mass - time")
    ax4.set_ylabel("Mass (ton)")
    ax4.set_xlim(0, time_limit)
    ax4.set_ylim(0, 600)
    ax4.scatter(x_data, mass_data, s=0.5)

    ax5 = fig.add_subplot(4, 2, 5)
    ax5.set_title("Thrust - time")
    ax5.set_ylabel("Force (N)")
    ax5.set_xlim(0, time_limit)
    # ax5.set_ylim(0, 30000)
    ax5.scatter(x_data, t_data, s=0.5)

    ax6 = fig.add_subplot(4, 2, 6)
    ax6.set_title("Drag - time")
    ax6.set_ylabel("Force (N)")
    ax6.set_xlim(0, time_limit)
    # ax6.set_ylim(-100, 100)
    ax6.scatter(x_data, d_data, s=0.5)

    ax7 = fig.add_subplot(4, 2, 7)
    ax7.set_title("Gravity - time")
    ax7.set_ylabel("Force (N)")
    ax7.set_xlim(0, time_limit)
    # ax7.set_ylim(-100, 100)
    ax7.scatter(x_data, g_data, s=0.5)

    plt.show()


# Include guard
if __name__ == '__main__':
    main()
