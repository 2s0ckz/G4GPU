// G4VUserDeviceSteppingAction - the Geant4-shaped base for a stepping action that sees real
// steps.
//
// WHY THERE ARE TWO STEPPING ACTIONS
//
// G4UserSteppingAction (G4UserActions.hh) is the one Geant4 users know, and here it is handed
// an *aggregate*: one pseudo-step per volume per event, with the event's whole deposit in it.
// G4Step.hh explains why and is blunt about the cost - a stepping action that sums gets the
// right answer, and one that does anything else does not. It is kept because a great many real
// stepping actions do nothing but sum, and those port unchanged.
//
// This is the other one. A G4VUserDeviceSteppingAction sees every real step of every track,
// with both step points, and can therefore do the things the aggregate cannot: read the
// kinetic energy a particle had as it crossed a boundary, take the LET of the step that
// deposited the energy, build a spectrum, apply a per-step threshold.
//
// WHY IT IS NOT VIRTUAL
//
// Geant4 dispatches UserSteppingAction through a vtable. That costs an indirect call per step,
// which is nothing against Geant4's own per-step cost and is not nothing here: this transport
// runs about 3x10^7 track-steps a second, and a virtual call on the device would defeat
// inlining, force an ABI register spill at every step, and stop the compiler eliminating the
// action entirely when it does nothing.
//
// So the dispatch is static, through CRTP - the base is templated on the derived class:
//
//     class MySteppingAction : public G4VUserDeviceSteppingAction<MySteppingAction> {
//      public:
//       __device__ void UserSteppingAction(const G4DeviceStep& step) { ... }
//     };
//
// The class, the inheritance, the member data and the named method are all exactly what they
// would be in Geant4. What changes is that the derived type is named in the base's template
// argument, and that the method is __device__.
//
// WHAT THE DERIVED CLASS MAY HOLD
//
// It is copied by value into every kernel launch, so its members must be trivially copyable -
// device pointers, counts, cuts, bin edges. No std::vector, no std::string, no ownership. It
// allocates nothing itself: the host allocates its buffers and hands it the pointers, exactly
// as a G4VPrimitiveScorer is given its hits collection.
//
// It also runs in parallel over tracks, so anything accumulated across tracks needs an atomic.
// StepTally in core/step_hook.cuh is that pattern already written down, and reading the memory
// note at the top of that file matters more than anything on this page: an action that stores
// per step scales with the amount of transport and so fails on exactly the runs worth doing.
//
// HOW A PROJECT INSTALLS ONE
//
// See examples in tests/test_custom_hook.cu. In short: define the class, name it as the hook
// type, instantiate the engine for it in one .cu of the project, and hand an instance to the
// run manager. The g4gpu engine itself is not rebuilt.
#pragma once
#include "core/step_hook.cuh"
#include "g4/G4Types.hh"

/// One real step, as a stepping action sees it. The Geant4-shaped spelling of DeviceStep.
///
/// Unlike G4Step, every value on it is this step's own rather than an event total, and
/// GetStepLength() is the true path length - not the distance between the two points, which
/// is shorter by however much multiple scattering deflected the track inside the step.
using G4DeviceStep = g4gpu::DeviceStep<G4double>;
using G4DeviceStepPoint = g4gpu::DeviceStepPoint<G4double>;
using G4DeviceTrack = g4gpu::DeviceTrack<G4double>;

/// Geant4 spells these as bare enumerators of a plain enum, so a stepping action written there
/// says `fGeomBoundary` with no qualification. These aliases let the same line compile here,
/// where the underlying type is an enum class and would otherwise need naming every time.
using G4StepStatus = g4gpu::StepStatus;
constexpr G4StepStatus fUndefined = G4StepStatus::fUndefined;
constexpr G4StepStatus fGeomBoundary = G4StepStatus::fGeomBoundary;
constexpr G4StepStatus fWorldBoundary = G4StepStatus::fWorldBoundary;
constexpr G4StepStatus fPostStepDoItProc = G4StepStatus::fPostStepDoItProc;
constexpr G4StepStatus fAlongStepDoItProc = G4StepStatus::fAlongStepDoItProc;
constexpr G4StepStatus fStopAndKill = G4StepStatus::fStopAndKill;

/// The identity of the process that defined a step. Geant4 hands back a G4VProcess* and you
/// compare its GetProcessName(); there is no process object on the device, so this is an enum.
using G4ProcessId = g4gpu::ProcessId;


/// Base class for a stepping action that sees real steps. CRTP: name your own class as the
/// template argument. See the note above for why this is not virtual.
template <typename Derived>
struct G4VUserDeviceSteppingAction {
  /// The engine calls this; it forwards to your UserSteppingAction. Nothing to override.
  __device__ void operator()(const g4gpu::DeviceStep<G4double>& step) const {
    static_cast<const Derived*>(this)->UserSteppingAction(step);
  }
};
