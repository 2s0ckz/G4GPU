// G4VUserActionInitialization lives in G4UserActions.hh; this header exists for Geant4-style
// includes.
//
// It pulls in G4RunManager.hh as well, because the base class's protected SetUserAction
// overloads - the ones an ActionInitialization::Build() calls - forward to the run manager and
// are defined there. Geant4 keeps them out of line in libG4run and needs no such include;
// here they are inline, so the definition has to be in scope wherever Build() is compiled.
#pragma once
#include "g4/G4RunManager.hh"
#include "g4/G4UserActions.hh"
