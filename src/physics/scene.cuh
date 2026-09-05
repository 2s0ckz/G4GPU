// Everything a stepping kernel needs to look up, gathered into one struct.
//
// Split out of the old physics/transport.cuh when the thread-per-event path was retired.
// The scheduler is the only transport loop now, so this header holds the shared description
// of the world and nothing else.
#pragma once
#include "data/brems_data.cuh"
#include "data/materials.cuh"
#include "data/photoelectric_data.cuh"
#include "data/rayleigh_data.cuh"
#include "geometry/navigator.cuh"
#include "physics/em/electron_processes.cuh"
#include "physics/em/hadron_range.cuh"
#include "physics/em/urban_msc.cuh"

namespace g4gpu {

/// Pointers into device memory for the geometry and every physics table.
/// Which processes are active.
///
/// All on is the validated configuration and the default; turning one off is a *study* tool -
/// "how much of this dose is Compton?" - not a physics choice, and the answer is only
/// meaningful because the all-on case is the one checked against Geant4.
///
/// Ionisation is deliberately absent. It is the continuous energy loss along a step, and
/// without it a lepton has infinite range and never stops; there is no configuration in which
/// switching it off yields a number worth having, so the toggle is not offered.
struct ProcessFlags {
  bool photoelectric = true;
  bool compton = true;
  bool rayleigh = true;
  bool pair_production = true;
  bool bremsstrahlung = true;
  bool annihilation = true;
  bool multiple_scattering = true;
};

/// A null table means the corresponding process is switched off.
template <typename real_t>
struct Scene {
  geom::Geometry<real_t> geometry;
  const data::Material<real_t>* materials;
  const em::RangeTable<real_t>* range_table;
  const data::PhotoElectricTable<real_t>* photoelectric;
  const data::BremsTable<real_t>* brems;
  const data::SBTableSet<real_t>* sb;   ///< differential tables, for sampling the photon
  const data::RayleighTable<real_t>* rayleigh;
  const em::UrbanTable<real_t>* msc;    ///< transport mfp and Urban coefficients
  /// Range table for protons and alphas. Null when no hadron can appear in the run, which is
  /// every gamma- or electron-driven run; step_hadron is then never reached.
  const em::HadronRangeTable<real_t>* hadron_range;
  /// The electron production cut expressed as a *range* in mm, not an energy.
  /// G4WentzelVIModel::ComputeTruePathLengthLimit reads it straight off the couple and uses it
  /// to soften its step limit; it is the one place in the transport that wants the cut in the
  /// units it was set in rather than the energy it converts to.
  real_t range_cut;
  /// Retained so existing drivers still construct a Scene the same way; scoring itself is a
  /// per-volume flag now, so that any number of volumes can be scored.
  int scoring_volume;
  ProcessFlags processes{};
};

/// Fraction of the residual range a lepton may travel in one step. This is Geant4's
/// dRoverRange; the full step function with its finalRange term lives in the stepper.
template <typename real_t> __host__ __device__ constexpr real_t kMaxStepFraction() {
  return real_t(0.2);
}

/// Guards against a pathological track looping forever on a boundary.
constexpr int kMaxStepsPerTrack = 10000;

}  // namespace g4gpu
