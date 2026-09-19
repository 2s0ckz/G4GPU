// G4LowEGammaNuclearModel - QBBC's photon-nuclear final state below 200 MeV.
//
// Transcribed from Geant4 11.1.1, source/processes/hadronic/models/pre_equilibrium/
// exciton_model/src/G4LowEGammaNuclearModel.cc (the class is NOT under gamma_nuclear/, whose
// only occupant is G4LENDorBERTModel).
//
// It is the smallest model in QBBC's hadronic chain and the whole of it is four statements:
//
//     lab4mom.set(0., 0., 0., G4NucleiProperties::GetNuclearMass(A, Z));
//     lab4mom += aTrack.Get4Momentum();
//     G4Fragment frag(A, Z, lab4mom);
//     G4ReactionProductVector* res = fPreco->DeExcite(frag);
//
// The photon is ABSORBED WHOLE. The compound nucleus is the target with the photon's entire
// four-momentum added and the target's own (A, Z) unchanged - not A+1 - so its excitation is
// `|p_target + p_gamma| - M(A,Z)`, which for a photon of energy E on a nucleus at rest is
// `sqrt(M^2 + 2ME) - M`, i.e. E less the recoil. There is no cascade and no direct knock-out:
// every product comes out of P6's exciton stage and P3's evaporation chain.
//
// THE FRAGMENT CARRIES ZERO EXCITONS, AND THAT DECIDES THE WHOLE MODEL
//
// `G4Fragment(A, Z, p)` leaves `numberOfParticles`, `numberOfCharged` and `numberOfHoles` at
// zero; nothing here calls `SetNumberOfExcitedParticle`. So `G4PreCompoundModel::DeExcite`
// enters its main loop (assuming the entry gate passes), calls `CalculateProbability` once -
// which draws nothing - and then exits on `GetNumberOfExcitons() <= 0` in the six-way test,
// straight to `PerformEquilibriumEmission`. The pre-equilibrium stage therefore emits NOTHING
// for this model, and a photo-nuclear reaction below 200 MeV in QBBC is pure evaporation from
// a compound nucleus. That is not an approximation anyone made; it is what the model is, and
// `PrecoStatus::n_preco_products` is asserted to be zero for every case in this package's
// grid.
//
// The one exception is the ENTRY gate, which can skip even that single CalculateProbability:
// `(Z < minZ && A < minA) || U < fLowLimitExc*A || U > A*fHighLimitExc`. A 10 MeV photon on
// carbon gives U = 9.996 MeV and `fLowLimitExc*A` = 0.1*12 = 1.2, so it enters; a 1 MeV photon
// on lead gives U = 0.9999 and `0.1*207` = 20.7, so it does not. Both arms end in the same
// place and the difference is one uniform deviate's worth of stream position - which is why
// `preco::deexcite` is called and its gate not second-guessed here.
//
// EVERY SECONDARY'S TIME IS 1 NANOSECOND, AND IT IS A SIGN FLAG
//
// `news->SetTime((*res)[i]->GetTOF())`. `G4ReactionProduct::timeOfFlight` is never assigned by
// `G4ExcitationHandler` - which sets `SetFormationTime` instead - so it holds whatever the
// constructor left, and `G4ReactionProduct(const G4ParticleDefinition*)` sets it to
//
//     (aParticleDefinition->GetPDGEncoding()<0) ? timeOfFlight=-1.0 : timeOfFlight=1.0;
//
// a SIGN carrying which side of the interaction the product is on, in the internal time unit,
// which is the nanosecond. Every de-excitation product has a positive PDG code (gammas 22,
// neutrons 2112, protons 2212, ions 10LZZZAAAI), so every secondary of this model is emitted
// with a time of +1 ns and the de-excitation chain's real formation time - which for an isomer
// can be microseconds - is discarded. Transcribed as written; docs/RISK.md has the finding.
#ifndef G4GPU_HADRONIC_EMEXTRA_LOW_E_GAMMA_CUH
#define G4GPU_HADRONIC_EMEXTRA_LOW_E_GAMMA_CUH

#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/emextra/config.cuh"
#include "physics/hadronic/precompound/precompound_model.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::emextra {

/// What one `low_e_gamma_apply` did beyond its final state.
struct LowEGammaResult {
  int n_products = 0;         ///< everything P6 handed back
  int n_preco_products = 0;   ///< the pre-equilibrium ejectiles - ALWAYS 0, see the header
  int residual_z = 0;         ///< the compound nucleus, before de-excitation
  int residual_a = 0;
  double excitation = 0.0;    ///< its E*, MeV
  bool skipped_precompound = false;  ///< the entry gate sent it straight to the handler
  EmExtraRefusal refusal = EmExtraRefusal::kNone;
  preco::PrecoRefusal preco_refusal;
};

/// `G4ReactionProduct::GetTOF()` for a de-excitation product: the sign of the PDG code, in ns.
/// See this file's header - it is a flag and it is used as a time.
__host__ __device__ inline double product_tof_ns(int pdg) { return (pdg < 0) ? -1.0 : 1.0; }

/// `theDef->GetPDGMass()` for one de-excitation product, which is what its kinetic energy is
/// measured against. An excited ion's PDG mass includes the excitation; the same expression
/// P3's `deex_kinetic_energy` subtracts, written out because P5's HadSecondary carries the
/// mass as well as the energy.
__host__ __device__ inline double product_pdg_mass(const deex::DeexProduct& p) {
  if (p.a == 0) {
    return (p.pdg == deex::kPdgElectron) ? units::electron_mass_c2<double>() : 0.0;
  }
  return deex::nuclear_mass(p.a, p.z) + p.excitation;
}

/// The PDG code of one de-excitation product. P3 has already resolved the eight species
/// `G4ExcitationHandler` names directly and left `pdg = 0` for everything `G4IonTable` would
/// have built, which is where the nuclear code comes from.
__host__ __device__ inline int product_pdg(const deex::DeexProduct& p) {
  return (p.pdg != 0) ? p.pdg : pdg_nuclear_code(p.z, p.a);
}

/// G4LowEGammaNuclearModel::ApplyYourself.
///
/// `projectile` must be a photon; the model has no `IsApplicable` of its own and Geant4 would
/// happily hand it anything the process's particle is, which for `photonNuclear` is only ever
/// a gamma. The check is here rather than assumed, because this entry point is also reachable
/// from the two lepton models' gamma chains - where the projectile is a real gamma built from
/// the equivalent photon, and where getting that wrong would be silent.
///
/// The secondary's direction is `(*res)[i]->GetMomentum().unit()` when its kinetic energy is
/// positive and `(0,0,1)` when it is zero - Geant4's own guard, and the reason a zero-energy
/// product does not produce a NaN direction. `unit()` of the zero vector is zero in CLHEP and
/// +z in this port (docs/RISK.md V127), so the guard is what makes the two agree and it is
/// written out rather than delegated.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline LowEGammaResult low_e_gamma_apply(
    const HadProjectile<real_t>& projectile, const HadNucleus& target,
    HadFinalState<real_t, kCap>& fs, const data::LevelTable& lt, const deex::FermiPool& pool,
    const preco::PrecoWorkspace& pws, Rng& rng) {
  LowEGammaResult r;
  fs.clear();

  if (target.l != 0) {
    r.refusal = EmExtraRefusal::kHyperNucleus;
    return r;
  }

  const int A = target.a;
  const int Z = target.z;

  // `lab4mom.set(0,0,0, GetNuclearMass(A,Z)); lab4mom += aTrack.Get4Momentum();`
  // G4HadProjectile's four-momentum is along +z with |p| = sqrt(T(T+2m)) and E = T + m, which
  // for a photon is (0, 0, T, T).
  const double plab = double(projectile.momentum());
  const double elab = double(projectile.total_energy());
  const double target_mass = deex::ground_state_mass(Z, A);
  const deex::LorentzVector lab4mom(0.0, 0.0, plab, elab + target_mass);

  deex::Fragment frag = deex::make_fragment(A, Z, lab4mom);
  r.residual_z = Z;
  r.residual_a = A;
  r.excitation = frag.excitation;

  // `frag.SetCreatorModelID(secID)` - a label, not a physical quantity; P5's HadSecondary
  // carries the creator id and this package's callers set it.
  preco::Excitons ex;   // zero particles, zero charged, zero holes - see the header
  const preco::PrecoStatus st = preco::deexcite(frag, ex, lt, pool, pws, rng);
  r.n_products = st.n_products;
  r.n_preco_products = st.n_preco_products;
  r.skipped_precompound = st.skipped_precompound;
  r.preco_refusal = st.ref;
  if (st.ref.any()) { r.refusal = EmExtraRefusal::kSubModel; }

  // `if(res) { theParticleChange.SetStatusChange(stopAndKill); ... }` - DeExcite always
  // returns a vector, even an empty one, so the photon is always absorbed.
  fs.status = HadFinalStateStatus::kStopAndKill;

  for (int i = 0; i < st.n_products; ++i) {
    const deex::DeexProduct& p = pws.products[i];
    HadSecondary<real_t> s;
    s.z = p.z;
    s.a = p.a;
    s.pdg = product_pdg(p);
    const double mass = product_pdg_mass(p);
    s.mass = static_cast<real_t>(mass);
    const double ekin = p.momentum.e - mass;
    s.kin_energy = static_cast<real_t>((ekin > 0.0) ? ekin : 0.0);
    if (ekin > 0.0) {
      const double m = std::sqrt(p.momentum.v.x * p.momentum.v.x
                                 + p.momentum.v.y * p.momentum.v.y
                                 + p.momentum.v.z * p.momentum.v.z);
      if (m > 0.0) {
        s.direction = Vec3<real_t>{static_cast<real_t>(p.momentum.v.x / m),
                                   static_cast<real_t>(p.momentum.v.y / m),
                                   static_cast<real_t>(p.momentum.v.z / m)};
      } else {
        // CLHEP's Hep3Vector::unit() of the zero vector is the zero vector; this port's is +z.
        // Geant4 reaches it here only when the momentum is exactly zero and the kinetic energy
        // is not, which cannot happen for a real product, and the direction is written out so
        // that the two cannot differ silently.
        s.direction = Vec3<real_t>{real_t(0), real_t(0), real_t(0)};
      }
    } else {
      s.direction = Vec3<real_t>{real_t(0), real_t(0), real_t(1)};   // `G4ThreeVector dir(0,0,1)`
    }
    s.time = static_cast<real_t>(product_tof_ns(s.pdg)) * units::ns<real_t>();
    if (!fs.add_secondary(s)) {
      r.refusal = EmExtraRefusal::kCapacity;
      break;
    }
  }
  return r;
}

}  // namespace g4gpu::physics::hadronic::emextra

#endif
