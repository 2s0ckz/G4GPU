// G4ElectroVDNuclearModel and G4MuonVDNuclearModel - the equivalent-photon models QBBC gives
// `electronNuclear`, `positronNuclear` and `muonNuclear`.
//
// Transcribed from Geant4 11.1.1, source/processes/hadronic/models/lepto_nuclear/src/:
//   G4ElectroVDNuclearModel::ApplyYourself / CalculateEMVertex / CalculateHadronicVertex
//   G4MuonVDNuclearModel::ApplyYourself / CalculateEMVertex / CalculateHadronicVertex /
//     MakeSamplingTable
//
// Both models are the same idea and NOT the same code: a virtual photon is drawn at the
// electromagnetic vertex, the lepton is scattered to balance it, and the photon is then handed
// to a hadronic model as a REAL photon below 10 GeV and as a pi0 above. What differs is
// everything about the drawing.
//
//   G4ElectroVDNuclearModel   the photon energy comes from the CHIPS electro-nuclear cross
//                             section's own sampler (`GetEquivalentPhotonEnergy`), Q2 from
//                             `GetEquivalentPhotonQ2`, and the photon is ACCEPTED OR REJECTED
//                             against the ratio of the real-photon cross section at nu and at
//                             nu - Q2/dM times `GetVirtualFactor`. A rejected photon leaves the
//                             electron untouched and produces NO secondaries at all - the model
//                             returns a final state in which nothing happened.
//   G4MuonVDNuclearModel      the photon energy comes from a 5 x 73 x 800 sampling table this
//                             model builds itself out of Kokoulin's double-differential cross
//                             section, the scattering angle from a rejection loop on the
//                             momentum transfer t, and there is no acceptance test: every call
//                             above the threshold produces a photon.
//
// THE ELECTRON MODEL'S THREE GATES, IN ORDER, AND WHAT EACH COSTS
//
//   photonEnergy < leptonKE            the sampler can return more than the electron has
//   photonEnergy > photonQ2/dM         `dM` here is `m_p + m_n` from the PARTICLE TABLE, not
//                                      the CHIPS class's own 938.27+939.57 - two different
//                                      numbers one function apart
//   sigNu*rand() <= sigK*rndFraction   the virtual-photon acceptance - WHICH CANNOT REJECT
//                                      ANYTHING, because `sigNu` is identically zero. See
//                                      `evd_probe_kinetic_energy` and docs/RISK.md V175: the
//                                      probe photon is built with the (definition, TOTAL
//                                      ENERGY, MOMENTUM VECTOR) overload and a UNIT momentum,
//                                      so it has a dynamical mass of sqrt(E^2 - 1) and a
//                                      kinetic energy of at most 1 MeV, which is under
//                                      G4PhotoNuclearCrossSection's 2 MeV threshold at every
//                                      photon energy. The 2,573-line cross section and
//                                      `GetVirtualFactor` behind this test are dead code.
//
// Failing either of the first two returns the initial state: status isAlive, the lepton's own
// energy and direction, no secondaries. That is a real outcome of a real interaction in Geant4 -
// the process has already decided the electron interacts - so it is NOT a refusal here either,
// and `LeptonVdResult::no_photon` records which gate closed. Measured over the campaign: the
// first gate closes for 0.5% to 7% of events depending on the case, the third for none of
// them, and the port's rate agrees with Geant4's to 0.3 of a standard error over 66 cases.
//
// THE MUON MODEL'S LOW-ENERGY RETURN
//
//   epmax = aTrack.GetTotalEnergy() - 0.5*proton_mass_c2;
//   if (epmax <= CutFixed) { ...return the initial track... }
//
// with CutFixed = 0.2 GeV. For a 200 MeV mu- that is 200 + 105.658 - 469.136 = -163.5, so a
// 200 MeV muon-nuclear interaction produces NOTHING - the muon is returned unchanged. The
// threshold is at a total energy of 669.14 MeV, i.e. a kinetic energy of 563.48 MeV.
//
// WHERE THE PHOTON GOES, AND WHY IT IS NOT THE PHOTON PROCESS'S CHAIN
//
// `CalculateHadronicVertex` is `if (gammaE < 10*GeV) bert->ApplyYourself(...)` else convert to
// a pi0 and use `ftfp->ApplyYourself(...)`. Neither model consults `G4EnergyRangeManager`, so
// the photon does NOT go through `photonNuclear`'s windows: there is no `G4LowEGammaNuclearModel`
// below 200 MeV and no QGS above 3 GeV. A 100 MeV equivalent photon from an electron runs the
// BERTINI CASCADE, where the same 100 MeV real photon would have run the pre-compound model.
// Each model owns its own `new G4CascadeInterface` and its own `G4TheoFSGenerator`, and those
// two FTF generators are FTF - `G4FTFModel` with `G4LundStringFragmentation` - unlike the
// photon process's, which is QGS. Three G4TheoFSGenerators in one physics list, two of them FTF
// and one QGS, and only the photon's is the one P13's brief expected.
//
// Both models' Bertini instances are plain `new G4CascadeInterface`, so they take
// `G4CascadeParameters::usePreCompound()` - false - and de-excite with the CASCADE's own
// evaporators, like the photon process's instance and unlike the nucleon and pion ones.
//
// WHAT IS REFUSED, BY NAME
//
//   * The FTF arm above a 10 GeV equivalent photon. Refused by name, `kSubModel`, at the point
//     the pi0 would have been handed over, with `used_ftf` recording that the arm was chosen.
//
//     HOW IT IS TO BE WIRED, when it is: through `src/physics/hadronic/ftf/ftf_entry.cuh` and
//     nothing else - P11d's caller-side contract, which is not in this branch (it lands on
//     main; this package is branched from `integ/bertini`). Its shape is a host
//     `build`/`free` over a pool of `entry::HadronWorkspace`, **329,816 bytes PER THREAD IN
//     FLIGHT**, and a `Handle` a thread takes one slot of before calling `apply`. The slot
//     count this arm needs is ONE PER THREAD THAT CAN HAVE A LEPTON ABOVE 10 GeV IN FLIGHT -
//     not one per lepton, because the arm is entered only for an equivalent photon above
//     10 GeV and that photon is always below the lepton's own kinetic energy. For a
//     galactic-cosmic-ray problem that is a real fraction of the lepton flux and the pool
//     cannot be sized at one.
//
//     It is unreachable from any lepton below 10 GeV, which is every case in this package's
//     campaign, so nothing here is validated against it and nothing here pretends to be.
//   * A hyper-nuclear target.
//   * Every buffer capacity.
#ifndef G4GPU_HADRONIC_EMEXTRA_LEPTON_VD_CUH
#define G4GPU_HADRONIC_EMEXTRA_LEPTON_VD_CUH

#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/bertini/cascade_interface.cuh"
#include "physics/hadronic/emextra/config.cuh"
#include "physics/hadronic/emextra/gamma_chain.cuh"
#include "physics/hadronic/emextra/light_target.cuh"
#include "physics/hadronic/emextra/mu_vd_table.cuh"
#include "physics/hadronic/process.cuh"
#include "physics/hadronic/xs/chips_electronuclear.cuh"
#include "physics/hadronic/xs/chips_photonuclear.cuh"
#include "physics/hadronic/xs/kokoulin_muon_xs.cuh"

namespace g4gpu::physics::hadronic::emextra {

namespace chips = g4gpu::hadronic::xs::chips;
namespace kokoulin = g4gpu::hadronic::xs::kokoulin;

/// Which gate closed when no photon was produced - all three are ordinary outcomes, not
/// refusals.
enum class NoPhotonReason : int {
  kNone = 0,
  kPhotonEnergyAboveLepton,  ///< `photonEnergy < leptonKE` failed
  kBelowQ2OverDM,            ///< `photonEnergy > photonQ2/dM` failed
  kVirtualRejected,          ///< `sigNu*rand() > sigK*rndFraction`
  kMuonBelowCut              ///< the muon model's `epmax <= CutFixed`
};

/// What one lepton-nuclear call did.
struct LeptonVdResult {
  NoPhotonReason no_photon = NoPhotonReason::kNone;
  double photon_energy = 0.0;     ///< nu, MeV - the equivalent photon's energy
  double photon_q2 = 0.0;         ///< Q2, MeV^2 (electron model only; the muon model has t)
  double lepton_final_kin = 0.0;  ///< the scattered lepton's kinetic energy, MeV
  double lepton_cos_theta = 1.0;  ///< cos of its scattering angle about the incident direction
  bool used_bertini = false;
  bool used_ftf = false;          ///< the >= 10 GeV pi0 arm
  int n_secondaries = 0;
  EmExtraRefusal refusal = EmExtraRefusal::kNone;
  bert::ApplyResult bertini;
  LightTargetResult light;
  chips::ElnSampleStatus eln;     ///< the CHIPS sampler's diagnostics (electron model only)
  int t_rejection_tries = 0;      ///< the muon model's `do { ... } while (rand > rej)` count
  bool t_loop_exhausted = false;  ///< its 10,000-try JustWarning
};

// ---------------------------------------------------------------------------------------------
// CalculateHadronicVertex - shared by both models, written once because it IS the same code
// ---------------------------------------------------------------------------------------------

/// `G4*VDNuclearModel::CalculateHadronicVertex`. `gamma_total_energy` is
/// `incident->GetTotalEnergy()`, which for a photon is its energy.
///
/// `gamma_dir` is the photon's direction in the lab; the Bertini and FTF entry points take a
/// projectile along +z, so the caller rotates the secondaries afterwards exactly as
/// `G4HadronicProcess::FillResult` does for every other inelastic model.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline void lepton_hadronic_vertex(double gamma_total_energy,
                                                       const HadNucleus& target,
                                                       HadFinalState<real_t, kCap>& fs,
                                                       const GammaWorkspace& ws,
                                                       const data::LevelTable& lt,
                                                       const deex::FermiPool& pool,
                                                       LeptonVdResult& r, Rng& rng) {
  if (gamma_total_energy < 10.0 * units::GeV<double>()) {
    r.used_bertini = true;
    HadProjectile<real_t> g;
    g.pdg = 22;
    g.mass = real_t(0);
    g.charge = real_t(0);
    g.baryon_number = 0;
    g.kin_energy = static_cast<real_t>(gamma_total_energy);
    if (target.a < 3) {
      r.light = light_target_collide(g, target, fs, bert::default_cascade_params(), ws.bert_ws,
                                     *ws.epo, *ws.global_out, rng);
      r.refusal = r.light.refusal;
      return;
    }
    if (!ws.complete()) {
      r.refusal = EmExtraRefusal::kCapacity;
      return;
    }
    r.bertini = bert::apply_yourself(
        g, target, fs, qbbc_gamma_deexcite_choice(), bert::default_cascade_params(),
        bert::default_interface_limits(), *ws.model, *ws.global_out, *ws.out, *ws.dex_out,
        *ws.tmp, *ws.epo, *ws.bert_ws, lt, pool, ws.preco, 0, rng);
    if (r.bertini.refusal != bert::InterfaceRefusal::kNone) {
      r.refusal = (r.bertini.refusal == bert::InterfaceRefusal::kLightTargetCollider)
                      ? EmExtraRefusal::kLightTargetCollider
                      : EmExtraRefusal::kSubModel;
    }
    return;
  }
  // "At high energies convert incident gamma to a pion". The pi0's MOMENTUM is
  // `sqrt(gammaE^2 - piMass^2)` in the electron model and `sqrt(piKE*(piKE + 2*piMass))` with
  // `piKE = gammaE - piMass` in the muon model - the same number written two ways, and both
  // give a pi0 whose TOTAL energy is the photon's. Transcribed as the kinetic energy P5's
  // HadProjectile carries, which is `gammaE - piMass` either way.
  r.used_ftf = true;
  // Refused by name at the point the hand-over would have happened. It goes through
  // `ftf/ftf_entry.cuh` - P11d's caller-side contract, one 329,816-byte
  // `entry::HadronWorkspace` slot per thread that can have a lepton above 10 GeV in flight -
  // and through nothing else; that header is not in this branch. See the file header.
  r.refusal = EmExtraRefusal::kSubModel;
}

// ---------------------------------------------------------------------------------------------
// G4ElectroVDNuclearModel
// ---------------------------------------------------------------------------------------------

/// `dM` as G4ElectroVDNuclearModel computes it: `G4Proton::Proton()->GetPDGMass() +
/// G4Neutron::Neutron()->GetPDGMass()`, i.e. CLHEP's two masses. NOT the CHIPS class's own
/// `938.27 + 939.57`, which is 3e-6 smaller and which `GetVirtualFactor` uses one call later.
__host__ __device__ inline constexpr double evd_dM() {
  return units::proton_mass_c2<double>() + units::neutron_mass_c2<double>();
}

/// The kinetic energy of the probe photon `CalculateEMVertex` builds for its acceptance test,
/// which is NOT the photon energy. See the acceptance test below and docs/RISK.md V175.
///
///     G4DynamicParticle photon(G4Gamma::Gamma(), photonEnergy, G4ThreeVector(0.,0.,1.));
///
/// resolves to `G4DynamicParticle(const G4ParticleDefinition*, G4double totalEnergy,
/// const G4ThreeVector& aParticleMomentum)`, and the momentum is the UNIT vector, so:
///
///     pModule2 = 1;  mass2 = E^2 - 1;  PDGmass2 = 0
///     mass2 < EnergyMRA2 (1 keV squared) ?  dynamicalMass = 0, kinetic = E
///     |PDGmass2 - mass2| > EnergyMRA2   ?  dynamicalMass = sqrt(mass2),
///                                          kinetic = E - sqrt(E^2 - 1)
///
/// `E - sqrt(E^2 - 1)` is 1 at E = 1 and falls monotonically - 0.27 at 2 MeV, 0.05 at 10 MeV,
/// 2.7 keV at 188 MeV - so it never reaches `G4PhotoNuclearCrossSection`'s `THmin` of 2 MeV and
/// the cross section is always zero. The `mass2 < EnergyMRA2` arm (E within 5e-7 of 1 MeV) is
/// written out too, and gives E itself, which is also below THmin.
__host__ __device__ inline double evd_probe_kinetic_energy(double photon_energy) {
  // CLHEP's EnergyMomentumRelationAllowance is 1 keV; G4DynamicParticle squares it.
  const double kEnergyMRA = 1.0e-3;                  // 1 keV in MeV
  const double kEnergyMRA2 = kEnergyMRA * kEnergyMRA;
  const double mass2 = photon_energy * photon_energy - 1.0;   // |p| = 1 by construction
  if (mass2 < kEnergyMRA2) { return photon_energy; }
  return photon_energy - std::sqrt(mass2);
}

/// G4ElectroVDNuclearModel::ApplyYourself + CalculateEMVertex.
///
/// The final state is the SCATTERED LEPTON (status isAlive, a new energy and direction) plus
/// whatever the hadronic vertex produced. Geant4 writes the lepton into the particle change
/// rather than as a secondary, which is what `HadFinalState::energy_change` and
/// `momentum_change` carry here.
///
/// `xs_state` is the CHIPS electro-nuclear state; it is filled by the cross-section call this
/// function makes first, exactly as the comment in Geant4 requires.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline LeptonVdResult electro_vd_apply(
    const HadProjectile<real_t>& projectile, const HadNucleus& target,
    HadFinalState<real_t, kCap>& fs, const GammaWorkspace& ws, const data::LevelTable& lt,
    const deex::FermiPool& pool, Rng& rng) {
  LeptonVdResult r;
  fs.clear();
  if (target.l != 0) {
    r.refusal = EmExtraRefusal::kHyperNucleus;
    return r;
  }

  // "Set up default particle change (just returns initial state)"
  const double leptonKE = double(projectile.kin_energy);
  fs.status = HadFinalStateStatus::kIsAlive;
  fs.energy_change = static_cast<real_t>(leptonKE);
  fs.momentum_change = Vec3<real_t>{real_t(0), real_t(0), real_t(1)};
  r.lepton_final_kin = leptonKE;
  r.lepton_cos_theta = 1.0;

  chips::ElnState st;
  const g4gpu::hadronic::xs::XsValue<double> exs =
      chips::eln_element_xs<double>(st, leptonKE, target.z);
  if (!exs.ok()) {
    r.refusal = EmExtraRefusal::kSubModel;
    return r;
  }
  const double photonEnergy = chips::eln_equivalent_photon_energy(st, rng, r.eln);
  r.photon_energy = photonEnergy;
  if (!(photonEnergy < leptonKE)) {
    r.no_photon = NoPhotonReason::kPhotonEnergyAboveLepton;
    return r;
  }
  const double photonQ2 = chips::eln_equivalent_photon_q2(st, photonEnergy, rng, r.eln);
  r.photon_q2 = photonQ2;
  if (!(photonEnergy > photonQ2 / evd_dM())) {
    r.no_photon = NoPhotonReason::kBelowQ2OverDM;
    return r;
  }

  // CalculateEMVertex's acceptance test, which CANNOT REJECT ANYTHING. See this file's header
  // and docs/RISK.md V175: the probe photon is built with
  //
  //     G4DynamicParticle photon(G4Gamma::Gamma(), photonEnergy, G4ThreeVector(0.,0.,1.));
  //
  // which selects the `(definition, TOTAL ENERGY, MOMENTUM VECTOR)` overload with a momentum of
  // magnitude ONE, so the constructor computes `mass2 = E^2 - 1`, gives the "photon" a dynamical
  // mass of sqrt(E^2 - 1) and a kinetic energy of `E - sqrt(E^2 - 1)` - at most 1 MeV for any E
  // and 2.7 keV at 188 MeV. `G4PhotoNuclearCrossSection::GetElementCrossSection` reads
  // `GetKineticEnergy()` and returns 0 below `THmin` = 2 MeV, so **`sigNu` is identically zero
  // for every lepton, every energy and every element** and `sigNu*rand() > sigK*rndFraction` is
  // `0 > something >= 0`, which is false. Every virtual photon is accepted.
  //
  // `gammaXS` is `G4PhotoNuclearCrossSection` and not `G4GammaNuclearXS` - an order dependence,
  // measured through `registry_has_PhotoNuclearXS`, docs/RISK.md V172 - and it does not matter
  // either, for the same reason.
  //
  // Reproduced rather than simplified: the probe's kinetic energy is computed the way the
  // constructor computes it and handed to the same cross section, the second evaluation is at
  // `photonEnergy - photonQ2/dM` as `SetKineticEnergy` leaves it, `GetVirtualFactor` is called,
  // and the deviate is DRAWN. Collapsing this to "always accept" would lose the draw and
  // desynchronise the stream, and would hide the fact that a future Geant4 which fixes the
  // constructor call would start rejecting.
  const double probe_kin = evd_probe_kinetic_energy(photonEnergy);
  const g4gpu::hadronic::xs::XsValue<double> snu =
      chips::photo_element_xs<double>(probe_kin, target.z);
  const double shifted = photonEnergy - photonQ2 / evd_dM();
  const g4gpu::hadronic::xs::XsValue<double> sk =
      chips::photo_element_xs<double>(shifted, target.z);
  if (!snu.ok() || !sk.ok()) {
    r.refusal = EmExtraRefusal::kSubModel;
    return r;
  }
  const double sigNu = snu.value;
  const double sigK = sk.value;
  const double rndFraction = chips::eln_virtual_factor(photonEnergy, photonQ2);

  if (sigNu * rng.uniform() > sigK * rndFraction) {
    r.no_photon = NoPhotonReason::kVirtualRejected;
    return r;   // "No gamma produced, return null ptr"
  }

  // Scatter the lepton.
  const double mProj = double(projectile.mass);
  const double mProj2 = mProj * mProj;
  const double iniE = leptonKE + mProj;
  const double finE = iniE - photonEnergy;
  const double iniP = std::sqrt(iniE * iniE - mProj2);
  const double finP = std::sqrt(finE * finE - mProj2);
  double cost = (iniE * finE - mProj2 - photonQ2 / 2.0) / iniP / finP;
  if (cost > 1.0) { cost = 1.0; }
  if (cost < -1.0) { cost = -1.0; }
  const double sint = std::sqrt(1.0 - cost * cost);
  const double phi = units::twopi<double>() * rng.uniform();
  // `dir` is +z here; `ortx = dir.orthogonal().unit()` is CLHEP's `orthogonal()`, which for
  // (0,0,1) returns (1,0,0) - the branch that picks the smallest component. `orty = dir x ortx`
  // is then (0,1,0). Written out for +z because the projectile always arrives along +z in this
  // port and the process rotates the result.
  const double sinx = sint * std::sin(phi);
  const double siny = sint * std::cos(phi);
  fs.momentum_change = Vec3<real_t>{static_cast<real_t>(sinx), static_cast<real_t>(siny),
                                    static_cast<real_t>(cost)};
  fs.energy_change = static_cast<real_t>(finE - mProj);
  r.lepton_final_kin = finE - mProj;
  r.lepton_cos_theta = cost;

  // "Create a gamma with momentum equal to momentum transfer" - its ENERGY is `photonEnergy`
  // and its momentum is `iniP*dir - finP*findir`, whose magnitude is NOT photonEnergy. The
  // G4DynamicParticle constructor taking (definition, kineticEnergy, momentumVector) sets the
  // energy from the first and the DIRECTION from the second, so the photon is on shell at
  // `photonEnergy` travelling along the momentum transfer. CalculateHadronicVertex then reads
  // `GetTotalEnergy()`, which is photonEnergy.
  lepton_hadronic_vertex(photonEnergy, target, fs, ws, lt, pool, r, rng);
  // The hadronic vertex clears the final state, so the scattered lepton is written back after
  // it. `AddSecondaries` in Geant4 appends to a particle change whose status and energy the EM
  // vertex already set; here the order is reversed and the effect is the same.
  fs.status = HadFinalStateStatus::kIsAlive;
  fs.energy_change = static_cast<real_t>(r.lepton_final_kin);
  fs.momentum_change = Vec3<real_t>{static_cast<real_t>(sinx), static_cast<real_t>(siny),
                                    static_cast<real_t>(cost)};
  r.n_secondaries = fs.n_secondaries;
  return r;
}

// G4MuonVDNuclearModel's sampling table - its constants, the 5 x 73 x 801 table and the host
// builder - is physics/hadronic/emextra/mu_vd_table.cuh since docs/RISK.md V233, so that the
// upload side can carry the table without carrying this header's models. `muon_vd_apply`
// below reads it as before.

/// G4MuonVDNuclearModel::CalculateEMVertex, plus the `epmax <= CutFixed` return above it.
///
/// `lnZ` is `G4Pow::logZ(Z)` and the nearest-Z choice is `min |lnZ - logZ(zdat[iz])|`, so the
/// table row is chosen in LOG Z and not in Z: carbon (6) is nearer to 4 than to 13 in the log,
/// which it is not in Z. The energy row is chosen in log E the same way.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline LeptonVdResult muon_vd_apply(
    const HadProjectile<real_t>& projectile, const HadNucleus& target,
    HadFinalState<real_t, kCap>& fs, const MuVdTable& tab, const GammaWorkspace& ws,
    const data::LevelTable& lt, const deex::FermiPool& pool, Rng& rng) {
  LeptonVdResult r;
  fs.clear();
  if (target.l != 0) {
    r.refusal = EmExtraRefusal::kHyperNucleus;
    return r;
  }

  const double Mass = double(projectile.mass);
  const double KineticEnergy = double(projectile.kin_energy);
  const double TotalEnergy = KineticEnergy + Mass;

  fs.status = HadFinalStateStatus::kIsAlive;
  fs.energy_change = static_cast<real_t>(KineticEnergy);
  fs.momentum_change = Vec3<real_t>{real_t(0), real_t(0), real_t(1)};
  r.lepton_final_kin = KineticEnergy;

  const double epmax0 = TotalEnergy - 0.5 * units::proton_mass_c2<double>();
  if (epmax0 <= mu_vd_cut_fixed()) {
    r.no_photon = NoPhotonReason::kMuonBelowCut;
    return r;
  }

  const double epmin = mu_vd_cut_fixed();
  const double epmax = epmax0;
  const double m0 = mu_vd_cut_fixed();

  // Nearest tabulated Z in log Z, then nearest tabulated T in log T.
  const double lnZ = std::log(double(target.z));
  int izz = 0;
  double delmin = 1.e10;
  for (int iz = 0; iz < kMuVdNZ; ++iz) {
    const double del = std::fabs(lnZ - std::log(double(mu_vd_zdat()[iz])));
    if (del < delmin) { delmin = del; izz = iz; }
  }
  int itt = 0;
  delmin = 1.e10;
  for (int it = 0; it < kMuVdNT; ++it) {
    const double del = std::fabs(std::log(KineticEnergy) - std::log(mu_vd_tdat()[it]));
    if (del < delmin) { delmin = del; itt = it; }
  }

  // Sample the energy transfer from the cumulative table. The loop breaks at the first bin
  // whose cumulative is at or above the draw; a table that is all zeros (a zero total cross
  // section) would run off the end and read `GetX(NBIN+1)`, which is out of range - unreachable
  // because every tabulated T has a positive cross section, and guarded here rather than
  // reproduced.
  const double rr = rng.uniform();
  int iy = 0;
  for (iy = 0; iy < kMuVdNBin; ++iy) {
    if (tab.value[izz][itt][iy] >= rr) { break; }
  }
  if (iy >= kMuVdNBin) {
    r.refusal = EmExtraRefusal::kSubModel;
    return r;
  }
  const double pvx = tab.x[iy];
  const double pvx1 = tab.x[iy + 1];
  const double y = pvx + rng.uniform() * (pvx1 - pvx);
  const double x = std::exp(y);
  const double ep = epmin * std::exp(x * std::log(epmax / epmin));
  r.photon_energy = ep;

  // Sample the momentum transfer t by rejection.
  const double yy2 = ep / TotalEnergy;
  const double tmin = Mass * Mass * yy2 * yy2 / (1.0 - yy2);
  const double tmax = 2.0 * units::proton_mass_c2<double>() * ep;
  double t1, t2;
  if (m0 < ep) { t1 = m0 * m0; t2 = ep * ep; }
  else { t1 = ep * ep; t2 = m0 * m0; }
  const double w1 = tmax * t1;
  const double w2 = tmax + t1;
  const double w3 = tmax * (tmin + t1) / (tmin * w2);
  const double y1 = 1.0 - yy2;
  const double y2 = 0.5 * yy2 * yy2;
  const double y3 = y1 + y2;

  double t = 0.0;
  double rej = 0.0;
  int ntry = 0;
  do {
    ntry += 1;
    if (ntry > 10000) {
      // "While count exceeded" - a JustWarning, and Geant4 then USES the last t it drew.
      r.t_loop_exhausted = true;
      break;
    }
    t = w1 / (w2 * std::exp(rng.uniform() * std::log(w3)) - tmax);
    rej = (1.0 - t / tmax) * (y1 * (1.0 - tmin / t) + y2) / (y3 * (1.0 - t / t2));
  } while (rng.uniform() > rej);
  r.t_rejection_tries = ntry;

  const double sinth2 =
      0.5 * (t - tmin) / (2.0 * (TotalEnergy * (TotalEnergy - ep) - Mass * Mass) - tmin);
  const double theta = std::acos(1.0 - 2.0 * sinth2);
  const double phi = units::twopi<double>() * rng.uniform();
  const double sinth = std::sin(theta);
  const double dirx = sinth * std::cos(phi);
  const double diry = sinth * std::sin(phi);
  const double dirz = std::cos(theta);
  // `finalDirection.rotateUz(ParticleDirection)` with the incident direction +z is the identity.

  const double NewKinEnergy = KineticEnergy - ep;
  r.lepton_final_kin = NewKinEnergy;
  r.lepton_cos_theta = dirz;
  fs.energy_change = static_cast<real_t>(NewKinEnergy);
  fs.momentum_change = Vec3<real_t>{static_cast<real_t>(dirx), static_cast<real_t>(diry),
                                    static_cast<real_t>(dirz)};

  // "Now create the emitted gamma": the four-momentum TRANSFER, whose total energy is
  // `TotalEnergy - Ef` = ep. Note Geant4 builds `initMomentum = sqrt(KineticEnergy*(TotalEnergy
  // + Mass))`, which is sqrt(T(T+2m)) written with one fewer addition, and `finalMomentum =
  // sqrt(NewKinEnergy*(NewKinEnergy+2*Mass))`, which is the ordinary form - two spellings of
  // the same quantity, one line apart.
  lepton_hadronic_vertex(ep, target, fs, ws, lt, pool, r, rng);
  fs.status = HadFinalStateStatus::kIsAlive;
  fs.energy_change = static_cast<real_t>(NewKinEnergy);
  fs.momentum_change = Vec3<real_t>{static_cast<real_t>(dirx), static_cast<real_t>(diry),
                                    static_cast<real_t>(dirz)};
  r.n_secondaries = fs.n_secondaries;
  return r;
}

}  // namespace g4gpu::physics::hadronic::emextra

#endif
