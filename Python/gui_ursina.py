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
from os import PathLike
import datetime
# Standard library imports
# First import should be the logging module if any!
import logging
import math as m

# Third party imports
import ursina
from ursina import (Ursina, Entity, scene, Mesh, Cylinder, Circle, Grid, Text,
                    Vec3, time, color, duplicate, camera, held_keys, window)

# Local application imports
from cls import CelestialBody, CircularOrbit, Planet, Earth
from utils import time_functions as tf

logger = logging.getLogger(__name__)

# Camera params
CAMERA_AZIMUTH = 55
CAMERA_POLAR = 120
CAMERA_RADIUS = 350

# Constants for visualization / graphics
YEARS_TO_SECS = 31_556_926
DIMENSION_SCALE_FACTOR = 1_000  # all dim. (in km) are divided by this number
SECOND_SCALE = 5
GRID_SIZE_KM = 100_000
SUBGRID_RATIO = 5

# Simulation (time) params
TIME_SCALE_FACTOR = 50_000  # the passing of time is multiplied by this number
START_TIME = tf.j2000_date(datetime.datetime.now())
SIMULATION_TIME = tf.j2000_date(datetime.datetime.now())
RUN = True

# TODO: implement central body and satellites - simulate a system
CENTRAL_BODY: Entity | None = None  # Only one
SATELLITES = []  # Satellites (keplerian elements) - list
SPACECRAFT = []  # Spacecraft (calculated by gravity) - list (at least 2)

moon_orbit: Entity | None = None


class CelestialBodyVisual(Entity):
    """ Abstract class for visual / graphical representation of the
    CelestialBody.
    """
    def __init__(self, celestial_body: CelestialBody, position=(0, 0, 0),
                 texture_file: PathLike | str = None,
                 body_color: ursina.color = None):
        self.celestial_body = celestial_body
        super().__init__(parent=scene, position=position)

        # Celestial body may not have radius (e.g. comet, asteroid)
        if hasattr(celestial_body, 'surface_radius_m'):
            self.scale = (celestial_body.surface_radius_m / 1000
                          / DIMENSION_SCALE_FACTOR * 2)
        else:
            self.scale = 1.0

        # If texture is defined, we use it; otherwise set color
        if texture_file is not None:
            self.texture_entity = Entity(
                    parent=self, position=position,
                    model='sphere',
                    rotation_x=-90,
                    texture=texture_file
            )

        elif body_color is not None:
            self.color = body_color


class Trajectory(Entity):
    """ Creates an entity, representing a trajectory, defined by finite points.

    Example: points = [Vec3(0,0,0), Vec3(0,.5,0), Vec3(1,1,0)]
    """
    def __init__(self, parent: Entity, points: list[Vec3]):
        self.points = points
        super().__init__(parent=parent,
                         position=(0, 0, 0),
                         model=Mesh(vertices=self.points, mode='line'),
                         color=color.green, thickness=3
                         )

    def add_point(self, point: Vec3, max_points: int = 40):
        if len(self.points) > max_points:
            self.points.pop(0)
        self.points.append(point)


def update():
    global CAMERA_AZIMUTH, CAMERA_POLAR, CAMERA_RADIUS, \
        SIMULATION_TIME, moon_orbit

    # Camera
    CAMERA_AZIMUTH += held_keys['d'] * 20 * time.dt
    CAMERA_AZIMUTH -= held_keys['a'] * 20 * time.dt
    CAMERA_POLAR += held_keys['w'] * 15 * time.dt
    CAMERA_POLAR -= held_keys['s'] * 15 * time.dt
    CAMERA_RADIUS -= held_keys['up arrow'] * 100 * time.dt
    CAMERA_RADIUS += held_keys['down arrow'] * 100 * time.dt

    camera.x = (CAMERA_RADIUS * m.cos(m.radians(CAMERA_AZIMUTH))
                * m.sin(m.radians(CAMERA_POLAR)))
    camera.y = (CAMERA_RADIUS * m.sin(m.radians(CAMERA_AZIMUTH))
                * m.sin(m.radians(CAMERA_POLAR)))
    camera.z = CAMERA_RADIUS * m.cos(m.radians(CAMERA_POLAR))
    camera.look_at(CENTRAL_BODY, up=CENTRAL_BODY.back)

    # Animation run
    if not RUN:
        return

    SIMULATION_TIME += TIME_SCALE_FACTOR * time.dt / YEARS_TO_SECS

    # Rotate central body
    CENTRAL_BODY.rotation_z -= (
            CENTRAL_BODY.celestial_body.angular_velocity_rad_per_s * 180 / m.pi
            * TIME_SCALE_FACTOR * time.dt
    )

    pos = (moon_orbit.get_position(SIMULATION_TIME)
           / DIMENSION_SCALE_FACTOR / SECOND_SCALE)
    moon.world_x = pos[0]
    moon.world_y = pos[1]
    moon.world_z = -pos[2]

    # Tidal-lock
    moon.rotation_z -= (moon_orbit.mean_angular_motion * 365.25
                        * TIME_SCALE_FACTOR * time.dt / YEARS_TO_SECS)

    rotation_info.text = (
            f"Simulation start: \t{tf.gregorian_date(START_TIME)} "
            f"({TIME_SCALE_FACTOR}x)\n"
            f"Simulation time: \t{tf.gregorian_date(SIMULATION_TIME)}\n"
            "---------\n"
            f"Grid size: \t\t{GRID_SIZE_KM:.0f} km\n"
            f"Subgrid size: \t{GRID_SIZE_KM/SUBGRID_RATIO:.0f} km\n"
            "---------\n"
            f"Camera azimuth.: \t{CAMERA_AZIMUTH:.1f}\n"
            f"Camera polar.: \t{CAMERA_POLAR:.1f}\n"
            f"Camera radius.: \t{CAMERA_RADIUS:.1f}\n"
    )


# TODO: set hotkeys for isometric, top and other views
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


def create_central_body():
    """ Set central body of the system. """
    global CENTRAL_BODY
    CENTRAL_BODY = CelestialBodyVisual(Earth(), (0, 0, 0),
                                       'resource/2k_earth_daymap.jpg')


def create_celestial_objects():
    """ Function for creating celestial objects (entities). """
    global SATELLITES, moon_orbit

    # Set satellites
    hold = Planet("0002", "Hold", 7.34767309E22, None,
                  None, None, 9.81, 1_737_000)
    moon_orbit = CircularOrbit(384_748, 28.58,
                               45, 90,
                               0)
    moon_orbit.calculate_orbital_period(5.972E24, 7.34767309E22)
    moon = CelestialBodyVisual(hold, (0, 0, 0),
                               'resource/lroc_color_poles_1k.jpg')

    # NOTE: create a function for this, and put it in a loop
    long_asc_node = Entity(
        model=Cylinder(6,
                       radius=0.1,
                       direction=(1, 0, 0),
                       height=384_748 / DIMENSION_SCALE_FACTOR / SECOND_SCALE,
                       thickness=2),
        rotation_z=-45,
        parent=scene,
        world_scale=1,
        color=color.hsv(60, 1, 1, .3)
    )

    orbit = Entity(
        model=Circle(120,
                     radius=384_748 / DIMENSION_SCALE_FACTOR / SECOND_SCALE,
                     mode='line',
                     thickness=2),
        color=color.hsv(60, 1, 1, .3),
        rotation_x=-28.58,
        parent=long_asc_node
    )

    periapsis = Entity(
        model=Cylinder(6,
                       radius=0.1,
                       direction=(1, 0, 0),
                       height=384_748 / DIMENSION_SCALE_FACTOR / SECOND_SCALE,
                       thickness=2),
        rotation_z=-90,
        parent=orbit,
        world_scale=1,
        color=color.hsv(60, 1, 1, .3)
    )

    mean_anomaly = moon_orbit.get_current_mean_anomaly(SIMULATION_TIME)
    moon.parent = periapsis
    moon.position = Vec3(384_748 / DIMENSION_SCALE_FACTOR / SECOND_SCALE, 0, 0)
    moon.rotate(Vec3(0, 0, 180 - mean_anomaly))

    return moon


def create_unit_vectors(parent=scene, scale=1, right_handed=False):
    """ Create x, y, z unit vectors at the parent object local coordinate
    system. Shows x as red, y as green and z as blue vector.
    """
    res = 6
    length = 5
    radius = 0.1
    zdir = 1
    if right_handed:
        zdir = -1

    Entity(  # x vector
        model=Cylinder(res, radius=radius, direction=(1, 0, 0), height=length),
        parent=parent, world_scale=scale, color=color.red
    )
    Entity(  # y vector
        model=Cylinder(res, radius=radius, direction=(0, 1, 0), height=length),
        parent=parent, world_scale=scale, color=color.green
    )
    Entity(  # z vector
        model=Cylinder(res, radius=radius, direction=(0, 0, zdir),
                       height=length),
        parent=parent, world_scale=scale, color=color.blue
    )


def create_grid(grid_nbr: int = 10):
    """ Create a 'nxn' grid in the x-y plane with the specified grid size,
    divided by the specified subgrid number. """
    grid_size = GRID_SIZE_KM / DIMENSION_SCALE_FACTOR  # size of 1 grid in units
    grid = Entity(model=Grid(grid_nbr, grid_nbr), scale=grid_nbr * grid_size,
                  rotation_x=0, color=color.blue, thickness=0.1)
    subgrid = duplicate(grid)
    grid.model.thickness = 1
    subgrid.model = Grid(grid_nbr * SUBGRID_RATIO, grid_nbr * SUBGRID_RATIO)
    subgrid.color = color.azure


if __name__ == '__main__':
    app = Ursina(vsync=False, fullscreen=True)
    window.color = color.black

    rotation_info = Text(position=window.top_left)
    create_central_body()
    moon = create_celestial_objects()
    create_unit_vectors(scale=3)
    create_unit_vectors(CENTRAL_BODY, scale=3, right_handed=True)
    create_unit_vectors(moon, scale=3, right_handed=True)

    create_grid()

    app.run()
