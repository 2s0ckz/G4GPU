// Example B1 on QBBC, with G4NeutronGeneralProcess's interaction-length bookkeeping as a switch -
// the program docs/RISK.md V219 was measured with.
//
// WHAT IT ASKS. `G4NeutronGeneralProcess::PostStepGetPhysicalInteractionLength` (11.1.1, and the
// same in 11.5.0) calls `CurrentCrossSection(track)` FIRST - which sets
// `currentInteractionLength = 1/fLambda` for the material the track is in NOW - and only then
// charges the step just taken:
//
//     CurrentCrossSection(track);
//     ...
//     theNumberOfInteractionLengthLeft -= previousStepSize/currentInteractionLength;
//
// so a step that ended on a boundary is charged at the NEXT material's mean free path.
// `G4HadronicProcess` charges it at the previous call's `currentInteractionLength` and refreshes
// after (`UpdateCrossSectionAndMFP`, then `currentInteractionLength = theMFP`), and so does
// `G4GammaGeneralProcess` ("new mean free path and step limit for the next step"). A neutron
// has no continuous process, so every step it survives ends on a boundary.
//
//   B1NGP_PATCH=0  the stock class. This is the CONTROL, and it is exact: the constructor list
//                  below is QBBC's, so the run reads what ref/B1build's exampleB1.exe reads, to
//                  every digit (the same random stream - measured on two beams in V219).
//   B1NGP_PATCH=1  the same class with the step charged at the mean free path it was TAKEN
//                  with: the base is called with previousStepSize = 0, which refreshes the cross
//                  section and samples a new length if none is carried and subtracts nothing,
//                  and the decrement is then made at the previous call's length.
//
// HOW THE CLASS GETS IN. `G4PhysListUtil::FindNeutronGeneralProcess` looks the process up on the
// neutron's manager by subtype and creates one only if it finds none, and every QBBC
// constructor that needs it (`G4HadronElasticPhysics`, `G4HadronInelasticQBBC` through
// `G4HadProcesses`, `G4NeutronTrackingCut`) goes through it. So one constructor registered
// FIRST, which adds the class to the neutron, is the whole difference from QBBC - the rest of
// the list is QBBC.cc's, line for line. There is no UI command for any of this, which is why it
// is a program (ref/b1neutron, with the general process off altogether, is the other one).
//
// EVERYTHING ELSE IS GEANT4's OWN B1, from the source tree, as in ref/b1neutron.
//
// Usage:  run.bat <macro> <0|1>
#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4HadronElasticPhysicsXS.hh"
#include "G4HadronInelasticQBBC.hh"
#include "G4IonElasticPhysics.hh"
#include "G4IonPhysicsXS.hh"
#include "G4Neutron.hh"
#include "G4NeutronGeneralProcess.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4ProcessManager.hh"
#include "G4RunManagerFactory.hh"
#include "G4StoppingPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4UImanager.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VPhysicsConstructor.hh"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

using namespace B1;

namespace {
long long g_carried = 0;  // lengths carried into a step, in the patched class
long long g_changed = 0;  // of them, across a change of mean free path

class ChargedAtTakenMfp : public G4NeutronGeneralProcess {
 public:
  G4double PostStepGetPhysicalInteractionLength(const G4Track& track, G4double previousStepSize,
                                                G4ForceCondition* condition) override {
    // The time cut returns 0 before anything else in the base (fTimeLimit, 10 us - QBBC's
    // G4NeutronTrackingCut finds the general process and leaves its default); kept exactly.
    if (track.GetGlobalTime() >= 10.0 * CLHEP::microsecond) {
      return G4NeutronGeneralProcess::PostStepGetPhysicalInteractionLength(
          track, previousStepSize, condition);
    }
    const G4double taken_with = currentInteractionLength;
    const G4bool carrying = (theNumberOfInteractionLengthLeft >= 0.0);
    G4double x =
        G4NeutronGeneralProcess::PostStepGetPhysicalInteractionLength(track, 0.0, condition);
    if (carrying && taken_with > 0.0 && taken_with < DBL_MAX) {
      ++g_carried;
      if (taken_with != currentInteractionLength) { ++g_changed; }
      theNumberOfInteractionLengthLeft -= previousStepSize / taken_with;
      theNumberOfInteractionLengthLeft = std::max(theNumberOfInteractionLengthLeft, 0.0);
      x = theNumberOfInteractionLengthLeft * currentInteractionLength;
    }
    return x;
  }
};

class NeutronGeneralFirst : public G4VPhysicsConstructor {
 public:
  explicit NeutronGeneralFirst(bool patched)
      : G4VPhysicsConstructor("b1ngp"), patched_(patched) {}
  void ConstructParticle() override {}
  void ConstructProcess() override {
    G4NeutronGeneralProcess* p =
        patched_ ? static_cast<G4NeutronGeneralProcess*>(new ChargedAtTakenMfp())
                 : new G4NeutronGeneralProcess();
    G4Neutron::Neutron()->GetProcessManager()->AddDiscreteProcess(p);
  }

 private:
  bool patched_;
};

// QBBC::QBBC (physics_lists/lists/src/QBBC.cc, 11.1.1), behind the one constructor above.
class QbbcNgp : public G4VModularPhysicsList {
 public:
  explicit QbbcNgp(bool patched, G4int ver = 1) {
    defaultCutValue = 0.7 * CLHEP::mm;
    SetVerboseLevel(ver);
    RegisterPhysics(new NeutronGeneralFirst(patched));
    RegisterPhysics(new G4EmStandardPhysics(ver));
    RegisterPhysics(new G4EmExtraPhysics(ver));
    RegisterPhysics(new G4DecayPhysics(ver));
    RegisterPhysics(new G4HadronElasticPhysicsXS(ver));
    RegisterPhysics(new G4StoppingPhysics(ver));
    RegisterPhysics(new G4IonPhysicsXS(ver));
    RegisterPhysics(new G4IonElasticPhysics(ver));
    RegisterPhysics(new G4HadronInelasticQBBC(ver));
    RegisterPhysics(new G4NeutronTrackingCut(ver));
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: b1ngp.exe <macro>   (B1NGP_PATCH=0 stock, =1 charged at the taken MFP)\n");
    return 1;
  }
  const char* env = std::getenv("B1NGP_PATCH");
  const bool patched = (env != nullptr && env[0] == '1');
  std::printf("b1ngp: G4NeutronGeneralProcess %s\n",
              patched ? "with each step charged at the mean free path it was taken with"
                      : "as 11.1.1 ships it");
  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
  runManager->SetUserInitialization(new DetectorConstruction());
  runManager->SetUserInitialization(new QbbcNgp(patched));
  runManager->SetUserInitialization(new ActionInitialization());
  G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[1]);
  std::printf("b1ngp: %lld interaction lengths carried into a step, %lld of them across a "
              "change of mean free path\n", g_carried, g_changed);
  delete runManager;
  return 0;
}
