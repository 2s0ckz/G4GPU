// How a cross section in this directory refuses a sub-case, and why it is a value and not a
// zero.
//
// docs/HADRONIC_PLAN.md section 6 rule 4: a sub-case that is not done is refused with a
// message that names it, at the point it would have been needed. On the host that can be a
// printf and an exit, which is what build_hadron_range_table does. Device code has neither,
// and - worse - the natural sentinel here is exactly the answer a working cross section can
// legitimately give: G4NeutronCaptureXS is genuinely zero above 20 MeV, the Coulomb-barrier
// factor is genuinely zero below the barrier, and the Barashenkov inelastic column is
// genuinely zero at the bottom of several elements' grids. A refusal that returns zero is
// therefore indistinguishable from physics, in the one direction that matters: it removes an
// interaction instead of reporting that it cannot compute one.
//
// So every function that has an unported branch returns the reason alongside the number.
// `xs_fatal_if_refused` is the host-side end of it, for a table builder or a test; a device
// caller checks the field.
#pragma once
#include <cstdio>
#include <cstdlib>

namespace g4gpu::hadronic::xs {

/// The Geant4 function that would have had to be ported. Each name is a symbol that exists in
/// 11.1.1, so a refusal message can be grepped for in the source tree.
enum class XsRefusal : int {
  kNone = 0,
  /// G4HadronNucleonXsc::HyperonNucleonXscNS - s/c/b hyperon on nucleon.
  kHyperonNucleonXscNS,
  /// G4HadronNucleonXsc::SCBMesonNucleonXscNS - s/c/b meson on nucleon.
  kSCBMesonNucleonXscNS,
  /// G4NucleiPropertiesTheoreticalTable and G4NucleiProperties::NuclearMass - a nuclide with
  /// no measured mass in AME2012. See data/nuclei_mass_ame12.hh.
  kNuclearMassNotTabulated,
  /// G4PhotoNuclearCrossSection - the CHIPS photonuclear parameterisation, which
  /// G4GammaNuclearXS needs above its data files' top energy and for hydrogen.
  kPhotoNuclearCrossSection,
  /// G4UPiNuclearCrossSection::IsElementApplicable is `1 < Z`; its Interpolate reads
  /// theZ[idx-1] with idx = 0 for Z = 1 and walks off the front of the table. The BGG pion
  /// classes never call it for hydrogen, and neither does this port.
  kUPiNuclearHydrogen,
  /// G4ComponentAntiNuclNuclearXS - antiproton and anti-nucleus on nucleus.
  kComponentAntiNuclNuclearXS,
  /// A per-element G4PARTICLEXS data file that is not loaded. Missing data is fatal, never a
  /// silent zero (docs/HADRONIC_PLAN.md section 2, src/host/g4data.cuh).
  kMissingParticleXSData,
};

__host__ __device__ inline const char* xs_refusal_name(XsRefusal r) {
  switch (r) {
    case XsRefusal::kNone: return "(none)";
    case XsRefusal::kHyperonNucleonXscNS: return "G4HadronNucleonXsc::HyperonNucleonXscNS";
    case XsRefusal::kSCBMesonNucleonXscNS: return "G4HadronNucleonXsc::SCBMesonNucleonXscNS";
    case XsRefusal::kNuclearMassNotTabulated:
      return "G4NucleiPropertiesTheoreticalTable / G4NucleiProperties::NuclearMass";
    case XsRefusal::kPhotoNuclearCrossSection: return "G4PhotoNuclearCrossSection";
    case XsRefusal::kUPiNuclearHydrogen: return "G4UPiNuclearCrossSection for Z = 1";
    case XsRefusal::kComponentAntiNuclNuclearXS: return "G4ComponentAntiNuclNuclearXS";
    case XsRefusal::kMissingParticleXSData: return "a G4PARTICLEXS data file that is absent";
  }
  return "(unknown)";
}

/// A cross section and whether it is one. `production` and `diffraction` are only filled by
/// the Glauber-Gribov components, which compute them on the way to the inelastic value.
template <typename real_t>
struct HadXs {
  real_t total = 0;
  real_t elastic = 0;
  real_t inelastic = 0;
  real_t production = 0;
  real_t diffraction = 0;
  XsRefusal refused = XsRefusal::kNone;

  __host__ __device__ bool ok() const { return refused == XsRefusal::kNone; }
};

/// One cross section and whether it is one.
template <typename real_t>
struct XsValue {
  real_t value = 0;
  XsRefusal refused = XsRefusal::kNone;

  __host__ __device__ bool ok() const { return refused == XsRefusal::kNone; }
};

/// Host-side: turn a refusal into the loud failure the working rules ask for. Used by table
/// builders and by tests, where continuing with a zero would be the silent approximation.
__host__ inline void xs_fatal_if_refused(XsRefusal r, const char* where) {
  if (r == XsRefusal::kNone) { return; }
  std::printf("\nFATAL: %s needs %s, which is not ported.\n"
              "  Refused by name rather than approximated - see "
              "src/physics/hadronic/xs/refusal.cuh.\n",
              where, xs_refusal_name(r));
  std::exit(2);
}

}  // namespace g4gpu::hadronic::xs
