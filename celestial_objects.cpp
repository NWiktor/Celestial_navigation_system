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
      pow(semimajor_axis*1000, 3)); // rad/s
    orbital_period = (2*M_PI) / mean_motion / 86400; // day

    cout << "Eccentricity (e): " << eccentricity << " [-]\n";
    cout << "Mean anomaly at epoch (M0): " << mean_anomaly_at_epoch << " [rad]\n";
    cout << "Semimajor axis (a): " << semimajor_axis << " [km]\n";
    cout << "Object mean motion (n): " << mean_motion << " [rad/s]\n";
    cout << "Orbital period (T): " << orbital_period << " [day]\n";
}


double Celestial_object::mean_anomaly(double time){
  // define epoch ?
  double epoch = 0; // seconds
  double mean_anomaly = mean_anomaly_at_epoch + mean_motion * (time - epoch);
  cout << "\nMean anomaly for " << name << " is: " << mean_anomaly << " [rad]\n";
  return mean_anomaly;
}


double Celestial_object::eccentric_anomaly(double mean_anomaly) {

  double e0 = mean_anomaly; // rad
  double e1;

  for( ; ; ) { // Iterating to infinity
    e1 = mean_anomaly + eccentricity * sin(e0);
    //cout << "E0: " << e0 << " / E1: " << e1 << "\n";

    if ( abs(e1-e0) > 0.00001 ) {
      e0 = e1;
      continue;
    }

    else {
      cout << "Eccentric anomaly for " << name << " is: " << e1 << "\n";
      return e1;
    }
  }
}

double Celestial_object::true_anomaly(double eccentric_anomaly){

  // double true_anomaly = 2;

  double true_anomaly = 2 * atan( sqrt( (1+eccentricity)/(1-eccentricity) ) * tan(eccentric_anomaly/2) );

  cout << "True anomaly for " << name << " is: " << true_anomaly << "\n";
  return true_anomaly;
}


double Celestial_object::normal_time_to_JDN(int year,
  int month, int day, int hour, int minute){
    // https://quasar.as.utexas.edu/BillInfo/JulianDatesG.html
    // Julian date number of given day by 0 hour !!!
    // https://www.onlineconversion.com/julian_date.htm

    if (month < 3) {
      year -= 1;
      month += 12;
    }

    int a = year/100;
    int b = a/4;
    int c = 2-a+b;
    int e = 365.25 * (year + 4716);
    int f = 30.6001 * (month + 1);
    double jd = c + e + f -1524.5 + day + (double)hour/24 + (double)minute/1440;

    cout.precision(17);
    cout << "Julian date number of given Gregorian date is: " << jd << "[day]\n";
    return jd;
}


double Celestial_object::normal_time_to_J2000(int year,
  int month, int day, int hour, int minute){
    // https://nsidc.org/data/icesat/glas-date-conversion-tool/date_convert/

    double t = normal_time_to_JDN(year, month, day, hour, minute);
     t -= 2451545.0;
     t *= 86400; // seconds in a day
     cout << "J2000 date number of given Gregorian date is: " << t << " [s]\n";
     return t;
}


vector<double> Celestial_object::Get_position_at_time(double time){

  double ma = mean_anomaly(time);
  double ea = eccentric_anomaly(ma);
  double ta = true_anomaly(ea);

  vector<double> position = {0,0,0};

  cout << "Position of " << name << " at " << time << " is: " << ta << "\n";
  return position;
}
