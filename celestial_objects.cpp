// Celestial objects C++

// Standard headers
#include <iostream>
#include <algorithm>
#include <string.h>
#include <vector>
#include <tuple>
#include <cmath>
#include <list>

// Own headers
#include "celestial_objects.hpp"
using namespace std;


// Class methods
Celestial_object::Celestial_object(string Name, double Mass, double Radius) {
  name = Name;
  mass = Mass;
  parent_mass = 1.9885 * pow(10,30); // Sun mass in kg for testing
  radius = Radius;

  cout << "Object " << Name << " initialized!\n";
  cout << "Object mass (m): " << mass << " [kg]\n";
  cout << "Object parent mass (mp): " << parent_mass << " [kg]\n";
  cout << "Object radius (r): " << radius << " [km]\n";
}


void Celestial_object::set_orbital_params(double Eccentricity,
  double Semimajor_axis, double Inclination, double Longitude_of_ascending_node,
  double Argument_of_periapsis, double Mean_anomaly_at_epoch) {
    eccentricity = Eccentricity;
    semimajor_axis = Semimajor_axis;
    inclination = Inclination;
    longitude_of_ascending_node = Longitude_of_ascending_node;
    argument_of_periapsis = Argument_of_periapsis;
    mean_anomaly_at_epoch = Mean_anomaly_at_epoch * M_PI / 180; // deg to rad

    // Calculated parameters
    mean_motion = sqrt(gravitational_constant * parent_mass / \
      pow(semimajor_axis*1000, 3));
    orbital_period = 2*M_PI*(1/mean_motion)/86400;

    cout << "Eccentricity (e): " << eccentricity << " [-]\n";
    cout << "Mean anomaly (M0): " << mean_anomaly_at_epoch << " [rad]\n";
    cout << "Semimajor axis (a): " << semimajor_axis << " [km]\n";
    cout << "Object mean motion (n): " << mean_motion << " [1/s]\n";
    cout << "Orbital period (T): " << orbital_period << " [day]\n";
}


double Celestial_object::mean_anomaly(){
  double mean_anomaly = mean_anomaly_at_epoch + 0;
  return mean_anomaly;
}


double Celestial_object::eccentric_anomaly() {
  cout << "\nEccentric anomaly for " << name << ":\n";

  double m = mean_anomaly();
  double e0 = m;
  double e1;

  for( ; ; ) { // Iterating to infinity
    e1 = m + eccentricity * sin(e0);
    cout << "E0: " << e0 << " E1: " << e1 << "\n";

    if ( abs(e1-e0) > 0.00001 ) {
      e0 = e1;
      continue;
    }

    else {
      return e1;
    }

  }

}

double Celestial_object::true_anomaly(){
  double true_anomaly = 0;
  return true_anomaly;
}
