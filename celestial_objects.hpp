#ifndef CELESTIAL_OBJECT_H
#define CELESTIAL_OBJECT_H

// Standard headers
#include <iostream>
#include <algorithm>
#include <string.h>
#include <vector>
#include <tuple>
#include <cmath>
#include <list>

// Own headers
using namespace std;

// Global variables

// Classes
class Celestial_object {
private:
  string name;
  string uuid; // Unique identifier

  double mass; // kg or solar mass
  double parent_mass; // kg
  double radius; // km

  double orbital_time; // ks
  double mean_motion; // (n) 1/s
  double orbital_period; // s

  // Orbital parameters in parent object coord. system - spherical
  // double longitude; // l
  // double latitude; // b
  // double distance; // r

  // Orbital parameters in parent object coord. system - rectangular
  // double x;
  // double y;
  // double z;

  // Keplerian elements needed for calculating orbit
  // https://en.wikipedia.org/wiki/Orbital_elements
  // https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion
  // https://en.wikipedia.org/wiki/Ellipse
  double eccentricity; // Eccentricity (e)
  double semimajor_axis; // Semimajor axis (a), km
  double inclination; // (i)
  double longitude_of_ascending_node; // Longitude of the ascending node (Ω)
  double argument_of_periapsis; // Argument of periapsis (ω)
  double mean_anomaly_at_epoch; // Mean anomaly at epoch (M0).

  // Private functions
  // double eccentric_anomaly();
  // double true_anomaly();

  //Function to rotate back ellipsoid points to reference by euler angles

public:
  // Constructors
  Celestial_object(string, double, double); // init
  void set_orbital_params(double, double, double, double, double, double);
  //~Celestial_object(); // Destructor

  // Variables
  static constexpr double gravitational_constant = 6.67430 * pow(10,-11); // m^3 kg-1 s-2

  // Functions
  double mean_anomaly(double);
  double eccentric_anomaly(double);
  double true_anomaly(double);

  double normal_time_to_JDN(int, int, int, int, int);

  // https://space.stackexchange.com/questions/23988/how-to-get-true-anomaly-from-time
  vector<double> Get_position_at_time(double time);

};

#endif
