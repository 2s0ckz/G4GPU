// G4Nucleon - one nucleon inside a 3D nucleus.
//
// Transcribed from G4Nucleon.hh (hadronic/util) in 11.1.1. There is no G4Nucleon.cc worth
// transcribing: everything except `Boost(const G4LorentzVector&)` is inline in the header.
//
// Two things about the Geant4 class shape the struct below.
//
// **`AreYouHit()` is a pointer test, not a flag.** `theSplitableHadron != 0` is what it reads,
// and `Hit(G4int)` - the overload the binary cascade calls - sets that pointer to the literal
// `reinterpret_cast<G4VSplitableHadron*>(1111)`. So "hit" in BIC's sense is a poisoned pointer
// value, while in FTF's sense it is a real G4VSplitableHadron the string model owns. The two
// uses cannot be told apart by `AreYouHit()`, which is why the flag here is a bool plus an
// index: `hit_by` is -1 when untouched, and otherwise identifies WHAT hit it, so a caller that
// needs FTF's splitable hadron has somewhere to put its handle and BIC's 1111 never has to be
// invented. `AreYouHit()` is `hit`, exactly as Geant4 reads it.
//
// **The particle type is a pointer to a singleton definition.** `GetDefinition() ==
// G4Proton::Proton()` is how every consumer asks "is this a proton", so the type has to be an
// identity and not a mass. A small enum does that and is device-storable; the PDG code and the
// PDG mass are derived from it by the two accessors below, which are the only things any
// consumer of a nucleon actually reads.
//
// The momentum is a full four-vector because that is what `G4Fancy3DNucleus::ChooseFermiMomenta`
// writes into it - a sampled Fermi three-momentum with energy `PDGMass - BindingEnergy/A`, which
// is deliberately OFF the mass shell downwards. docs/RISK.md V50 is what that costs downstream.
#ifndef G4GPU_BIC_NUCLEON_CUH
#define G4GPU_BIC_NUCLEON_CUH

#include "core/units.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"

namespace g4gpu::bic {

namespace u = g4gpu::units;
using deex::LorentzVector;
using deex::Vec3d;

/// The three particle definitions `G4Nucleon::SetParticleType` accepts for a nucleus (the three
/// anti-species overloads exist for anti-nuclei, which are refused by name - see
/// `fancy_3d_nucleus.cuh`).
enum NucleonType : int { kNucleonNone = 0, kProton = 1, kNeutron = 2, kLambda = 3 };

/// `GetDefinition()->GetPDGMass()` for the three types.
///
/// The proton and neutron masses are CLHEP's, which is what `G4Proton::Proton()->GetPDGMass()`
/// returns; the Lambda's is G4Lambda's own 1.115683 GeV. Anything else is zero, which is a
/// value no consumer can use by accident - a nucleon with no type never leaves `ChooseNucleons`.
__host__ __device__ inline double nucleon_pdg_mass(int type) {
  if (type == kProton) { return u::proton_mass_c2<double>(); }
  if (type == kNeutron) { return u::neutron_mass_c2<double>(); }
  if (type == kLambda) { return 1.115683 * u::GeV<double>(); }
  return 0.0;
}

/// `G4int(GetDefinition()->GetPDGCharge()/eplus + 0.1)`, the form every consumer uses.
__host__ __device__ inline int nucleon_charge(int type) { return (type == kProton) ? 1 : 0; }

/// The PDG code, for a consumer that reports species rather than reading the enum.
__host__ __device__ inline int nucleon_pdg(int type) {
  if (type == kProton) { return 2212; }
  if (type == kNeutron) { return 2112; }
  if (type == kLambda) { return 3122; }
  return 0;
}

/// G4Nucleon.
struct Nucleon {
  Vec3d position{0.0, 0.0, 0.0};
  LorentzVector momentum;
  double binding_energy = 0.0;
  int type = kNucleonNone;

  /// `AreYouHit()`. `hit_by` carries WHICH object hit it, for a caller that needs Geant4's
  /// `GetSplitableHadron()`; BIC only ever asks the bool.
  bool hit = false;
  int hit_by = -1;

  __host__ __device__ double pdg_mass() const { return nucleon_pdg_mass(type); }
  __host__ __device__ int charge() const { return nucleon_charge(type); }
  __host__ __device__ int pdg() const { return nucleon_pdg(type); }

  /// `G4Nucleon::Boost(const G4ThreeVector& beta)` - the four-momentum only. The POSITION is
  /// not boosted, which is what `DoLorentzContraction` is separately for.
  __host__ __device__ void boost(const Vec3d& beta) { momentum.boost(beta); }

  /// `G4Nucleon::Boost(const G4LorentzVector&)` (G4Nucleon.cc:46), the CERNLIB U101 form. It is
  /// not `boost(arg.boostVector())` and it is not even the same DIRECTION: this one transforms
  /// into the rest frame of `p4`, where `boost(beta)` transforms out of the rest frame into a
  /// frame moving with `beta`. A nucleon at rest given `p4 = (p, E)` comes out with momentum
  /// `-gamma*m*beta` here and `+gamma*m*beta` from the three-vector overload.
  ///
  /// That matters because `G4Fancy3DNucleus::DoLorentzBoost` is overloaded on exactly these two
  /// and forwards each to the matching G4Nucleon method, so the two spellings of "boost this
  /// nucleus" move it in opposite directions. See the note in `fancy_3d_nucleus.cuh`.
  ///
  /// Written out in Geant4's own operation order rather than as a boost by the negated velocity:
  /// the two differ in the last bits, and a cascade's momentum balance is a cancellation.
  __host__ __device__ void boost(const LorentzVector& p4) {
    const double mass = p4.mag();
    const double factor =
        (g4gpu::dot(momentum.v, p4.v) / (p4.e + mass) - momentum.e) / mass;
    const double dot4 = momentum.e * p4.e - g4gpu::dot(momentum.v, p4.v);
    // `1/mass*dot`, the reciprocal-then-multiply Geant4 writes, and not `dot/mass`: the two
    // differ by an ulp and fragment.cuh's header records what an ulp costs a recoil.
    momentum.e = (1.0 / mass) * dot4;
    momentum.v = factor * p4.v + momentum.v;
  }
};

}  // namespace g4gpu::bic

#endif
