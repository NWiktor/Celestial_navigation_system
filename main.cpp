// Celestial navigation system C++

// Standard headers
#include <iostream>
#include <string.h>
#include <math.h>
#include <chrono>

// Own headers
#include "celestial_objects.hpp"
// using namespace std;
using namespace std::chrono;

int main() {

  // Starting clock
  auto start = high_resolution_clock::now();

  // Business_code
  Celestial_object Fold("Earth", 5.97237 * pow(10,24), 6371);
  //Celestial_object Nereid("Nereid", 0, 0);

  Fold.set_orbital_params(0.0167086, 149598023, 0, 0, 0, 358.617);
  //Nereid.set_orbital_params(0.7417482, 0, 0, 0, 0, 69.95747);

  double act_time = Fold.normal_time_to_J2000(2021,04,14,11,37);

  Fold.get_orbital_coords_at_time(act_time);
  //Fold.eccentric_anomaly();
  //Nereid.eccentric_anomaly();

  // Fold.normal_time_to_JDN(2021,04,13,17,15);

  // Running time display
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "\nTime taken by function: "
    << duration.count() << " microseconds.\n";

  cout << "Program exited normally!\n";
  return 0;
}
