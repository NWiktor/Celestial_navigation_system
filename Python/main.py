# -*- coding: utf-8 -*-
#!/usr/bin/python3

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
import sys
import datetime
import time
import math as m

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from logger import MAIN_LOGGER as l
from modules.time_functions import julian_date, j2000_date

# Class initializations and global variables
gravitational_constant = 6.67430 * pow(10,-11) # m^3 kg-1 s-2

# Class and function definitions


class Orbit():

    def __init__(self, eccentricity, semimajor_axis, inclination,
        longitude_of_ascending_node, argument_of_periapsis, mean_anomaly_at_epoch):
        # Keplerian elements needed for calculating orbit
        # https://en.wikipedia.org/wiki/Orbital_elements
        # https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion
        # https://en.wikipedia.org/wiki/Ellipse
        """  """

        ## Keplerian elements
        self.eccentricity = eccentricity # Eccentricity (e), -
        self.semimajor_axis = semimajor_axis # Semimajor axis (a), km
        self.inclination = inclination # (i), deg
        self.longitude_of_ascending_node = longitude_of_ascending_node
        # Longitude of the ascending node (Ω), deg
        self.argument_of_periapsis = argument_of_periapsis # Argument of periapsis (ω), deg
        self.mean_anomaly_at_epoch = mean_anomaly_at_epoch # Mean anonaly (M0), deg

        ## Orbital parameters
        self.orbital_period = None # (T) day
        self.mean_angular_motion = None # (n) deg/day
        self.rotational_matrix = None

        # Calculate known attributes
        self.calculate_rotational_matrix()


    def calculate_rotational_matrix(self):
        loan = self.longitude_of_ascending_node * m.pi/180
        aop = self.argument_of_periapsis * m.pi/180
        incl = self.inclination * m.pi/180

        self.rotational_matrix = np.array([
            [m.cos(loan)*m.cos(aop) - m.sin(loan)*m.cos(incl)*m.sin(aop),
            -m.cos(loan)*m.sin(aop) - m.sin(loan)*m.cos(incl)*m.cos(aop),
            m.sin(loan)*m.sin(incl)],
            [m.sin(loan)*m.cos(aop) + m.cos(loan)*m.cos(incl)*m.sin(aop),
            -m.sin(loan)*m.sin(aop) + m.cos(loan)*m.cos(incl)*m.cos(aop),
            -m.cos(loan)*m.sin(incl)],
            [m.sin(incl)*m.sin(aop),
            m.sin(incl)*m.cos(aop),
            m.cos(incl)]])


    def calculate_mean_angular_motion(self):
        """  """
        self.mean_angular_motion = 360/self.orbital_period


    def set_orbital_period(self, orbital_period):
        """  """
        self.orbital_period = orbital_period # 24*60*60 seconds aka 1 day
        self.calculate_mean_angular_motion()


    def calculate_orbital_period(self, mass1, mass2=0):
        """  """
        # Converting km to m in semimajor axis, and convert seconds to days
        self.orbital_period = 2 * m.pi * m.sqrt(pow(self.semimajor_axis * 1000,3)
            / ((mass1+mass2)*gravitational_constant) ) / 86400
        l.debug(f"{self.orbital_period=}")
        self.calculate_mean_angular_motion()


    def get_position(self, j2000):
        """ Calculate position and velocity vectors of an object at a given orbit,
        at a given time since J2000 epoch.
        """

        # Calculate mean anomaly at J2000
        mean_anomaly = (self.mean_anomaly_at_epoch + (self.mean_angular_motion * j2000)) % 360 # deg
        l.debug(f"Mean anomaly is {mean_anomaly}°") # deg

        # Calculate eccentric anomaly
        # https://space.stackexchange.com/questions/55356/how-to-find-eccentric-anomaly-by-mean-anomaly
        # Solving for E = M + e*sin(E) using iteration
        eca0 = mean_anomaly * m.pi/180
        n = 0

        while True:
            eca1 = (mean_anomaly*m.pi/180) + self.eccentricity * m.sin(eca0)
            n +=1
            if abs(eca1-eca0) > 0.0000001:
                eca0 = eca1
            else:
                l.debug(f"Eccentric anomaly is {eca1} rad, found at {n}. itaration.")
                break

        # Calculate position and velocity in orbital plane
        x = self.semimajor_axis * (m.cos(eca1) - self.eccentricity) # km
        y = self.semimajor_axis * m.sqrt(1-pow(self.eccentricity,2)) * m.sin(eca1) # km

        # Convert position coordinates to x,y,z coordinates for animation
        # orbital_vector = np.array([x,y,0])
        return self.rotational_matrix.dot(np.array([x,y,0]))


    def clear(self):
        """ Erases all loaded and existing training data to allow refresh. """
        self.__dict__ = {}


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)



class CelestialObject():

    def __init__(self, name, uuid, mass, radius, parent_object=None):
        self._name = name
        self._uuid = uuid # Unique identifier
        self._mass = mass # kg
        self._radius = radius # km ???
        self._parent_object = parent_object
        self._orbit = None


    def set_orbit(self, orbit):
        self._orbit = orbit


    def get_position(self, j2000):
        """  """

        if self._parent_object is None:
            return np.array([0,0,0])

        return self._orbit.get_position(j2000) + self._parent_object.get_position(j2000)


    def clear(self):
        """ Erases all loaded and existing training data to allow refresh. """
        self.__dict__ = {}


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


# Main function for module testing
def main():
    """  """
    sun = CelestialObject("Nap", "0000", 1.9885*pow(10,30), 695700)
    mercury = CelestialObject("Merkur", "0001", 3.3011*pow(10,23), 2439.7, sun)
    venus = CelestialObject("Venus", "0002", 4.8675*pow(10,24), 6051.8, sun)
    earth = CelestialObject("Fold", "0003", 5.972168*pow(10,24), 6371.1009, sun)
    moon = CelestialObject("Hold", "0031", 7.342*pow(10,22), 1737.4, earth)

    # Orbits
    mercury_orbit = Orbit(0.205630, 57.91*pow(10,6), 7.005, 48.331, 29.124, 174.796)
    mercury_orbit.calculate_orbital_period(sun._mass, mercury._mass)
    mercury.set_orbit(mercury_orbit)

    venus_orbit = Orbit(0.006772, 108.21*pow(10,6), 3.39458, 76.680, 54.884, 50.115)
    venus_orbit.calculate_orbital_period(sun._mass, venus._mass)
    venus.set_orbit(venus_orbit)

    earth_orbit = Orbit(0.0167086, 149598023, 0.00005, -11.26064, 114.20783, 358.617)
    earth_orbit.set_orbital_period(365.256363004) # d
    # earth_orbit.calculate_orbital_period(sun._mass, earth._mass)
    earth.set_orbit(earth_orbit)


    ###  Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')

    celestial_bodies = [sun, mercury, venus, earth] 

    for cb in celestial_bodies:
        xdata = []
        ydata = []
        zdata = []

        for i in range(0,365):
            vector = cb.get_position(i)
            xdata.append(vector[0])
            ydata.append(vector[1])
            zdata.append(vector[2])

        ax1.scatter3D(xdata, ydata, zdata)

    plt.show()


    print(mercury_orbit.orbital_period)
    print(venus_orbit.orbital_period)



# Include guard
if __name__ == '__main__':
    main()
