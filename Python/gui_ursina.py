# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" This module contains all relevant class and function for orbit propagation
around a celestial body. The module calculates the trajectory of a two-stage
rocket launched from surface in Object-Centered Inertial reference frame (OCI).

Libs
----
* Ursina

Help
----
* https://www.solarsystemscope.com/textures/

Contents
--------
"""
import datetime
# Standard library imports
# First import should be the logging module if any!
import logging
import math as m

# Third party imports
from ursina import *
from ursina import Entity, scene, Mesh, Cylinder, Circle, Grid, Text, Vec3
from ursina import time, color, duplicate, camera, held_keys, window

# Local application imports
from cls import CelestialBody, CircularOrbit
from utils import time_functions as tf

logger = logging.getLogger(__name__)

CAMERA_AZIMUTH = 55
CAMERA_POLAR = 120
CAMERA_RADIUS = 200

YEARS_TO_SECS = 31_556_926
TIME_SCALE_FACTOR = 1000000  # the passing of time is multiplied by this number
DIMENSION_SCALE_FACTOR = 1000  # all dimensions (in km) are divided by this number
GRID_SIZE_KM = 100000
SUBGRID_RATIO = 5

START_TIME = tf.j2000_date(datetime.datetime.now())
SIMULATION_TIME = tf.j2000_date(datetime.datetime.now())
RUN = True

# TODO: implement central body and satellites - simulate a system
# Central body - 1 pc
# Satellites (keplerian elements) - list
# Spacecraft (calculated by gravity) - list (at least 2)
CENTRAL_BODY = None
SATELLITES = []
SPACECRAFT = []

moon_orbit = None


# TODO: rework this with the PLanet Entity
class _CelestialBodyVisual:
    """ Abstract class for visual / graphical representation of the
    CelestialBody.
    """

    def __init__(self, celestial_body: CelestialBody, radius: int,
                 color: tuple[int, int, int]):
        self.celestial_body = celestial_body
        self.radius = radius  # For 'visualization' only
        self.color = color


class CelestialBodyVisual(Entity):
    """ Creates a planet entity."""
    def __init__(self, radius_km, position=(0, 0, 0), texture_file=None):
        scale = radius_km / DIMENSION_SCALE_FACTOR * 2
        super().__init__(parent=scene, scale=scale, position=position)

        if texture_file is not None:
            self.texture_entity = PlanetTexture(self, texture_file)


class PlanetTexture(Entity):
    """ Creates a texture entity, which is rotated around the X axis, to align
    the texture and the parent body coordinate system."""
    def __init__(self, parent, texture_file):
        super().__init__(parent=parent, position=(0, 0, 0),
                         model='sphere',
                         rotation_x=-90,
                         texture=texture_file)


class Trajectory(Entity):
    """ Creates an entity, representing a trajectory, defined by finite points.

    Example: points = [Vec3(0,0,0), Vec3(0,.5,0), Vec3(1,1,0)]
    """
    def __init__(self, parent, points: list[Vec3]):
        super.__init__(parent=parent, position=(0, 0, 0),
                       model=Mesh(vertices=points, mode='line'))


def update():
    global CAMERA_AZIMUTH, CAMERA_POLAR, CAMERA_RADIUS, \
        SIMULATION_TIME, moon_orbit

    # Camera
    CAMERA_AZIMUTH += held_keys['d'] * 20 * time.dt
    CAMERA_AZIMUTH -= held_keys['a'] * 20 * time.dt
    CAMERA_POLAR += held_keys['w'] * 15 * time.dt
    CAMERA_POLAR -= held_keys['s'] * 15 * time.dt
    CAMERA_RADIUS -= held_keys['up arrow'] * 50 * time.dt
    CAMERA_RADIUS += held_keys['down arrow'] * 50 * time.dt

    camera.x = (CAMERA_RADIUS * m.cos(m.radians(CAMERA_AZIMUTH))
                * m.sin(m.radians(CAMERA_POLAR)))
    camera.y = (CAMERA_RADIUS * m.sin(m.radians(CAMERA_AZIMUTH))
                * m.sin(m.radians(CAMERA_POLAR)))
    camera.z = CAMERA_RADIUS * m.cos(m.radians(CAMERA_POLAR))
    camera.look_at(earth, up=earth.back)

    # Animation run
    if not RUN:
        return

    SIMULATION_TIME += TIME_SCALE_FACTOR * time.dt / YEARS_TO_SECS

    pos = moon_orbit.get_position(SIMULATION_TIME) / DIMENSION_SCALE_FACTOR / 10
    moon.world_x = pos[0]
    moon.world_y = pos[1]
    moon.world_z = -pos[2]

    # Tidal-lock
    moon.rotation_z -= (moon_orbit.mean_angular_motion * 365.25 *
                        TIME_SCALE_FACTOR * time.dt / YEARS_TO_SECS)

    rotation_info.text = (
            f"Simulation start: \t{tf.gregorian_date(START_TIME)} "
            f"(Time scale: {TIME_SCALE_FACTOR})\n"
            f"Simulation time: \t{tf.gregorian_date(SIMULATION_TIME)}\n"
            "---------\n"
            f"Grid size: \t\t{GRID_SIZE_KM} km\n"
            f"Subgrid size: \t{GRID_SIZE_KM/SUBGRID_RATIO} km\n"
            "---------\n"
            f"Camera azimuth.: \t{CAMERA_AZIMUTH:.1f}\n"
            f"Camera polar.: \t{CAMERA_POLAR:.1f}\n"
            f"Camera radius.: \t{CAMERA_RADIUS:.1f}\n"
    )


def input(key):
    global CAMERA_RADIUS, RUN
    if key == 'escape':
        quit()

    if key == 'scroll up':
        CAMERA_RADIUS += 100 * time.dt

    if key == 'scroll down':
        CAMERA_RADIUS -= 100 * time.dt

    if key == 'space' and RUN:
        RUN = False
    elif key == 'space' and not RUN:
        RUN = True


def main():
    global moon_orbit
    # 384_400 - original orbit of moon
    earth = CelestialBodyVisual(6_378, (0, 0, 0),
                   'resource/2k_earth_daymap.jpg')
    moon = CelestialBodyVisual(1_737, (0, 0, 0),
                  'resource/lroc_color_poles_1k.jpg')

    moon_orbit = CircularOrbit(384_748, 28.58,
                               45, 90,
                               0)
    moon_orbit.calculate_orbital_period(5.972E24, 7.34767309E22)

    # NOTE: create a function for this, and put it in a loop
    long_asc_node = Entity(
        model=Cylinder(6, radius=0.1, direction=(1, 0, 0),
                       height=384_748 / DIMENSION_SCALE_FACTOR / 10,
                       thickness=2),
        rotation_z=-45,
        parent=scene, world_scale=1, color=color.hsv(60, 1, 1, .3))

    orbit = Entity(
        model=Circle(120, radius=384_748 / DIMENSION_SCALE_FACTOR / 10,
                     mode='line', thickness=2),
        color=color.hsv(60, 1, 1, .3),
        rotation_x=-28.58,
        parent=long_asc_node)

    periapsis = Entity(
        model=Cylinder(6, radius=0.1, direction=(1, 0, 0),
                       height=384_748 / DIMENSION_SCALE_FACTOR / 10,
                       thickness=2),
        rotation_z=-90,
        parent=orbit, world_scale=1, color=color.hsv(60, 1, 1, .3))

    mean_anomaly = moon_orbit.get_current_mean_anomaly(SIMULATION_TIME)
    moon.parent = periapsis
    moon.position = Vec3(384_748 / DIMENSION_SCALE_FACTOR / 10, 0, 0)
    moon.rotate(Vec3(0, 0, 180 - mean_anomaly))

    return earth, moon


def create_unit_vectors(parent=scene, scale=1, right_handed=False):
    """ Create x, y, z unit vectors at the parent object local coordinate
    system. Shows x as red, y as green and z as blue vector.
    """
    height = 5
    radius = 0.1
    z_dir = 1
    if right_handed:
        z_dir = -1

    x_vector = Entity(model=Cylinder(6, radius=radius,
                                     direction=(1, 0, 0), height=height),
                      parent=parent, world_scale=scale, color=color.red)
    y_vector = Entity(model=Cylinder(6, radius=radius,
                                     direction=(0, 1, 0), height=height),
                      parent=parent, world_scale=scale, color=color.green)
    z_vector = Entity(model=Cylinder(6, radius=radius,
                                     direction=(0, 0, z_dir), height=height),
                      parent=parent, world_scale=scale, color=color.blue)


def create_grid():
    """ Create grid in the x-y plane with the specified increment size. """
    grid_size = GRID_SIZE_KM / DIMENSION_SCALE_FACTOR  # size of 1 grid in units
    grid_number = 10  # number of grids (n x n)
    grid = Entity(model=Grid(grid_number, grid_number),
                  scale=grid_number * grid_size,
                  rotation_x=0,
                  color=color.blue)
    subgrid = duplicate(grid)
    grid.model.thickness = 2
    subgrid.model = Grid(grid_number * SUBGRID_RATIO,
                         grid_number * SUBGRID_RATIO)
    subgrid.color = color.azure


if __name__ == '__main__':
    app = Ursina(vsync=False, fullscreen=True)
    window.color = color.black

    rotation_info = Text(position=window.top_left)

    earth, moon = main()
    create_unit_vectors(scale=3)
    create_unit_vectors(earth, scale=3, right_handed=True)
    create_unit_vectors(moon, scale=3, right_handed=True)

    create_grid()

    app.run()
