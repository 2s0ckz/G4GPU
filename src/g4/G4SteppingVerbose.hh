// G4SteppingVerbose.
//
// exampleB1.cc's second line asks for stepping verbosity in best units:
//
//     G4int precision = 4;
//     G4SteppingVerbose::UseBestUnit(precision);
//
// In Geant4 that selects G4SteppingVerboseWithUnits, which prints a table of every step when
// /tracking/verbose is on. There is no per-step host printing here - the steps happen on the
// device, millions per second, and a printf per step would be the whole run time - so the
// call is accepted and recorded, and /tracking/verbose says what it can and cannot show.
#pragma once
#include "g4/G4Types.hh"

class G4SteppingVerbose {
 public:
  static void UseBestUnit(G4int precision = 4) { Precision() = precision; }
  static G4int& Precision() {
    static G4int p = 4;
    return p;
  }
};
