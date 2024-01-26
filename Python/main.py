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

# Third party imports
import matplotlib.pyplot as plt

# Local application imports
import modules as mch
from logger import MAIN_LOGGER as L

# Class initializations and global variables
gravitational_constant = 6.67430 * pow(10, -11)  # m^3 kg-1 s-2

# Class and function definitions

# Main function for module testing
def main():
    """  """
    # TODO: add database for object data
    sun = mch.CelestialBody("Nap", "0000", 1.9885 * pow(10, 30), 695700)
    mercury = mch.CelestialBody("Mercury", "0001", 3.3011 * pow(10, 23), 2439.7, sun)
    venus = mch.CelestialBody("Venus", "0002", 4.8675 * pow(10, 24), 6051.8, sun)
    earth = mch.CelestialBody("Earth", "0003", 5.972168 * pow(10, 24), 6371.1009, sun)
    # moon = CelestialObject("Hold", "0031", 7.342*pow(10,22), 1737.4, earth)

    # Orbits
    mercury_orbit = mch.KeplerOrbit(0.205630, 57.91 * pow(10, 6), 7.005, 48.331, 29.124, 174.796)
    mercury_orbit.calculate_orbital_period(sun._mass, mercury._mass)
    mercury.set_orbit(sun, mercury_orbit)

    venus_orbit = mch.KeplerOrbit(0.006772, 108.21 * pow(10, 6), 3.39458, 76.680, 54.884, 50.115)
    venus_orbit.calculate_orbital_period(sun._mass, venus._mass)
    venus.set_orbit(sun, venus_orbit)

    earth_orbit = mch.KeplerOrbit(0.0167086, 149598023, 0.00005, -11.26064, 114.20783, 358.617)
    earth_orbit.set_orbital_period(365.256363004)  # d
    # earth_orbit.calculate_orbital_period(sun._mass, earth._mass)
    earth.set_orbit(sun, earth_orbit)

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-150000000, 150000000)
    ax.set_ylim3d(-150000000, 150000000)
    ax.set_zlim3d(-150000000, 150000000)

    celestial_bodies = [sun, mercury, venus, earth]

    for ceb in celestial_bodies:
        x_data: list = []
        y_data: list = []
        z_data: list = []

        for i in range(0, 365):
            vector = ceb.get_position(i)
            x_data.append(vector[0])
            y_data.append(vector[1])
            z_data.append(vector[2])

        ax.scatter3D(x_data, y_data, z_data)

    plt.show()

    print(mercury_orbit.orbital_period)
    print(venus_orbit.orbital_period)


# Include guard
if __name__ == '__main__':
    main()
