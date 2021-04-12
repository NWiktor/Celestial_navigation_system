// Celestial navigation system C++

// Standard headers
#include <iostream>
#include <string.h>
#include <math.h>

// Own headers
#include "celestial_objects.hpp"
using namespace std;

int main() {

  // Business_code
  Celestial_object Fold("Earth", 5.97237 * pow(10,24), 6371);
  //Celestial_object Nereid("Nereid", 0, 0);

  Fold.set_orbital_params(0.0167086, 149598023, 0, 0, 0, 358.617);
  //Nereid.set_orbital_params(0.7417482, 0, 0, 0, 0, 69.95747);

  Fold.eccentric_anomaly();
  //Nereid.eccentric_anomaly();

  cout << "Program exited normally!\n";
  return 0;
}
