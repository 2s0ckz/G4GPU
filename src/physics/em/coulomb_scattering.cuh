// G4CoulombScattering and G4eCoulombScatteringModel (11.1.1) - the last EM process in QBBC's
// chain, and the discrete half of the pair G4WentzelVIModel is the continuous half of.
//
// A charged particle's Coulomb scattering off a nucleus is split in two. Below some angle it
// is many small deflections and is treated as multiple scattering (G4WentzelVIModel, ported in
// em/wentzel_msc.cuh); above it, single scatters are rare enough to be sampled one at a time as
// a discrete process. Both read the SAME cross section - G4WentzelOKandVIxSection, ported in
// em/wentzel_xs.cuh - so nothing here recomputes it. What this file adds is the process's
// cross section (the same screened Rutherford integral, over the OTHER angular interval), the
// single-scatter sampler with its form factor and rejection, and the nuclear recoil.
//
// ---------------------------------------------------------------------------------------
// WHO GETS IT, WITH WHICH MODEL AND WHICH LIMITS
//
// Read out of G4EmStandardPhysics::ConstructProcess and G4EmBuilder::ConstructCharged in
// 11.1.1, checked against `ref/oracle/species_processes.csv`. `isWVI` is a defaulted argument
// (G4EmBuilder.hh:71, `G4bool isWVI = true`) and G4EmStandardPhysics.cc:225 calls
// ConstructCharged with two arguments, so it is TRUE in option0 - which is why every light
// hadron has this process at all.
//
//   species        process?  msc model              SS model                  limits
//   e-, e+         yes       Urban <100 MeV,        G4eCoulombScatteringModel SetMinKinEnergy,
//                            WentzelVI >100 MeV     (created explicitly)      SetLowEnergyLimit
//                                                                             and SetActivation-
//                                                                             LowEnergyLimit,
//                                                                             all = 100 MeV
//   mu+, mu-       yes       G4WentzelVIModel       created in Initialise-    none: 100 eV to
//   pi+, pi-       yes       G4WentzelVIModel       Process                    100 TeV, with a
//   K+, K-         yes       G4WentzelVIModel                                  MATERIAL-dependent
//   p, anti_proton yes       G4WentzelVIModel                                  table threshold
//   alpha, He3     NO        G4UrbanMscModel        -                         -
//   d, t, GenIon   NO        G4UrbanMscModel        -                         -
//   hyperons, b/c  NO        G4UrbanMscModel        -                         -
//
// Three facts in that table are easy to get wrong and each is a line of source:
//
//  * The 100 MeV for e+- is `G4EmParameters::MscEnergyLimit()`, whose default is set in
//    G4EmParameters::Initialise (`energyLimit = 100.0*CLHEP::MeV`, G4EmParameters.cc:155). It
//    is the msc/SS split AND the Urban/WentzelVI split, fetched once at
//    G4EmStandardPhysics.cc:134.
//  * The light hadrons get NO explicit limits (G4EmBuilder.cc:175-204 and :241-269 call
//    RegisterProcess and nothing else), so their table starts at
//    G4CoulombScattering::MinPrimaryEnergy - which depends on the MATERIAL through
//    <A^-2/3> - rather than at a fixed energy. `coulomb_process_min_primary_energy` below.
//  * Ions get the process only in G4EmBuilder::ConstructIonEmPhysicsSS, which option0 never
//    calls. G4EmBuilder::ConstructIonEmPhysics (G4EmBuilder.cc:119-145) takes no isWVI
//    argument and registers msc + ionisation only. So there is no ion Coulomb scattering in
//    QBBC, and `G4IonCoulombScatteringModel` is unreachable.
//
// And one that is dead code: `G4hCoulombScatteringModel` exists in 11.1.1 and NOTHING
// constructs it - `grep -rn "new G4hCoulombScatteringModel" source/` finds no match, only
// #include lines in G4EmStandardPhysicsWVI.cc and G4EmStandardPhysicsSS.cc. Every species
// above uses G4eCoulombScatteringModel, the electron one, hadrons included.
//
// ---------------------------------------------------------------------------------------
// THE ANGULAR INTERVAL, WHICH IS WHERE THE PROCESS'S IDENTITY LIVES
//
// `G4EmParameters::MscThetaLimit()` defaults to pi (G4EmParameters.cc:154, `thetaLimit =
// CLHEP::pi`) and G4CoulombScattering::InitialiseProcess pushes it into the model
// (`model->SetPolarAngleLimit(theta)`, G4CoulombScattering.cc:138).
// G4eCoulombScatteringModel::Initialise then maps `tet >= pi` to `cosThetaMin = -1.0`
// (G4eCoulombScatteringModel.cc:114) and hands that to its own wokvi. Its `cosThetaMax` is
// -1.0 from the constructor and IS NEVER REASSIGNED ANYWHERE IN THE CLASS.
//
// So in option0 the single-scattering process covers the WHOLE angular range, from the
// nuclear-size cut-off `cosTetMaxNuc` down to 180 degrees, and G4WentzelVIModel's msc covers
// the same range - `G4WentzelVIModel::Initialise` maps `tet == pi` through NEITHER of its two
// branches (G4WentzelVIModel.cc:112-113 test `tet <= 0` and `tet < pi`), so its cosThetaMax
// keeps its in-class -1.0 too. **There is no angular handover between msc and
// G4CoulombScattering in option0.** The two are separated by ENERGY for e+- (100 MeV) and, for
// the light hadrons, by nothing but the material-dependent table threshold below.
//
// That is worth stating flatly because it is the opposite of what the pair is for, and it is
// what G4EmStandardPhysicsWVI.cc:113 changes with `param->SetMscThetaLimit(0.15)`. The
// `ssFactor`/`invssFactor` pair (G4WentzelVIModel.cc:789-795, 1.25 and 1/1.2) is internal to
// the msc model's own step limit and its own multiple/single sub-mode split; it never reaches
// this process.
//
// ---------------------------------------------------------------------------------------
// WHAT IS REFUSED, BY NAME
//
//  * ISOTOPE SELECTION. G4eCoulombScatteringModel::SampleSecondaries calls
//    `SelectIsotopeNumber(currentElement)` and then `wokvi->SetTargetMass(
//    G4NucleiProperties::GetNuclearMass(ia, iz))`, so the sampler's `factD` and the recoil
//    energy use the NUCLEAR mass of a sampled isotope - not the element's mean atomic mass,
//    which is what SetupTarget had put there. Reproducing the draw needs per-element natural
//    abundances; `data/natural_isotopes.hh` carries only the SET of stable nuclides ("the
//    abundances are not needed and are not here"), and `data::Material` carries no isotope
//    list at all. So `coulomb_sample_secondaries` takes (iz, ia, target_mass) from its caller
//    and `coulomb_refuse_isotope_selection` says what is missing. The two masses differ by up
//    to a few per cent for a natural element, which is a few per cent on a recoil energy that
//    is almost always below the production threshold and deposited locally.
//  * THE ELEMENT DRAW is `G4VEmModel::SelectTargetAtom`, i.e. G4EmElementSelector's tabulated
//    cumulative cross section on a 20-bins-per-decade grid. This port sums over elements
//    directly everywhere (docs/PORTED.md 1.6), and `coulomb_select_element` does the same
//    here: the same partial cross sections, evaluated at the energy rather than interpolated
//    from a table built at neighbouring energies.
//  * THE GAUSSIAN AND FLAT NUCLEAR FORM FACTORS. `G4EmParameters::NuclearFormfactorType`
//    defaults to fExponentialNF and nothing in QBBC changes it, so `fGaussianNF` and
//    `fFlatNF` (G4WentzelOKandVIxSection.cc:355-365) are absent, as they already are in
//    em/wentzel_xs.cuh's file header.
#pragma once
#include <cmath>

#include "core/particle.cuh"
#include "core/units.cuh"
#include "data/materials.cuh"
#include "data/mott.hh"
#include "data/nuclei_mass_ame12.hh"
#include "physics/em/wentzel_msc.cuh"
#include "physics/em/wentzel_xs.cuh"

namespace g4gpu::em {

/// `G4CoulombScattering::q2Max`, MeV^2.
///
/// `a = FactorForAngleLimit()*hbarc/fermi; q2Max = 0.5*a*a`
/// (G4CoulombScattering.cc:93-94), with the parameter at its default of 1. Character for
/// character the same two lines as G4WentzelOKandVIxSection.cc:113-114, which is why this
/// returns `wentzel_factor_a2` rather than recomputing it: one number doing two jobs, and if
/// the two ever disagreed the transport and its table threshold would disagree with it.
template <typename real_t> __host__ __device__ inline real_t coulomb_q2_max() {
  return wentzel_factor_a2<real_t>();
}

/// `G4CoulombScattering::MinPrimaryEnergy(part, mat)`, MeV - the lower edge of the process's
/// lambda table, and the only energy limit the light hadrons get.
///
///     G4double theta = G4EmParameters::Instance()->MscThetaLimit();
///     if(0.0 < theta) {
///       G4double p2 = q2Max*mat->GetIonisation()->GetInvA23()/(1.0 - cos(theta));
///       G4double mass = part->GetPDGMass();
///       emin = sqrt(p2 + mass*mass) - mass;
///     }
///
/// It is a threshold and not a cut: `G4CoulombScattering::InitialiseProcess` sets
/// `SetStartFromNullFlag(yes)` when theta is pi (G4CoulombScattering.cc:99-104), and
/// G4EmTableUtil.cc:232-240 then starts the table at this energy with a zero below it - but
/// only `if(e >= emin)`, where emin is the process's minKinEnergy. For e+- that is 100 MeV and
/// this value never exceeds 98.15 MeV (hydrogen), so the e+- table always starts at exactly
/// 100 MeV and the branch never fires. For mu/pi/K/p it is 100 eV, so it always does.
///
/// @param inv_a23 the material's <A^-2/3>, G4IonisParamMat::GetInvA23.
template <typename real_t>
__host__ __device__ inline real_t coulomb_process_min_primary_energy(real_t mass,
                                                                     real_t inv_a23) {
  // 1 - cos(pi) = 2 exactly. Written as the literal Geant4 computes rather than as `2`,
  // because MscThetaLimit is the parameter this whole file's angular structure turns on and a
  // hard 2 here would silently survive someone setting it to 0.15.
  const real_t one_minus_cos = real_t(2);
  const real_t p2 = coulomb_q2_max<real_t>() * inv_a23 / one_minus_cos;
  return sqrt(p2 + mass * mass) - mass;
}

/// `G4eCoulombScatteringModel::MinPrimaryEnergy(material, part, cut)`, MeV.
///
/// A different quantity from the one above, on the MODEL rather than the process: the primary
/// energy below which no recoil above the threshold can be produced, computed for the
/// LIGHTEST element in the material because that is the one that recoils most easily.
///
///     G4double cut = std::max(recoilThreshold, (*pCuts)[couple->GetIndex()]);
///     G4int Z = min over elements of Z;
///     G4int A = G4lrint(fNistManager->GetAtomicMassAmu(Z));
///     G4double targetMass = G4NucleiProperties::GetNuclearMass(A, Z);
///     return std::max(cut, 0.5*(cut + sqrt(2*cut*targetMass)));
///
/// `recoilThreshold` is 0.0 from the constructor ("by default does not work",
/// G4eCoulombScatteringModel.cc:83) and nothing in QBBC sets it, so `cut` is the PROTON
/// production threshold - the process declares `SetSecondaryParticle(G4Proton::Proton())`
/// (G4CoulombScattering.cc:68), so the cut vector the model is initialised with is the
/// proton's, not the electron's.
///
/// @param proton_cut the material's proton production threshold in MeV
/// @param z_lightest the smallest Z in the material
template <typename real_t>
__host__ __device__ inline real_t coulomb_model_min_primary_energy(real_t proton_cut,
                                                                   int z_lightest) {
  const real_t cut = fmax(real_t(0), proton_cut);   // recoilThreshold = 0
  const int a = static_cast<int>(data::atomic_mass<real_t>(z_lightest) + real_t(0.5));
  const real_t target_mass = data::nuclear_mass<real_t>(a, z_lightest);
  if (!(target_mass > real_t(0))) { return real_t(0); }
  return fmax(cut, real_t(0.5) * (cut + sqrt(real_t(2) * cut * target_mass)));
}

/// What `G4eCoulombScatteringModel::ComputeCrossSectionPerAtom` returns, plus the electron
/// fraction its sampler needs.
template <typename real_t>
struct CoulombAtomXs {
  real_t total = 0;        ///< nuclear + electron, mm^2
  real_t nuclear = 0;
  real_t electron = 0;
  real_t cos_t_min = 0;    ///< the interval actually integrated, for the sampler
  real_t cos_t_max = 0;
  /// `ecross/(cross + ecross)`, the probability the scatter is off an atomic electron.
  ///
  /// The model has an `elecRatio` MEMBER which `ComputeCrossSectionPerAtom` sets to 0.0 at
  /// its top and never writes again (G4eCoulombScatteringModel.cc:185); the ratio the sampler
  /// uses is a LOCAL recomputed inside SampleSecondaries. So the member is vestigial and the
  /// value belongs to the sampling call, not to the cross-section call - it is carried here
  /// because this port computes the two cross sections once and would otherwise compute them
  /// twice.
  real_t elec_ratio = 0;
};

/// `G4eCoulombScatteringModel::ComputeCrossSectionPerAtom`, mm^2.
///
/// The screened Rutherford integral of em/wentzel_xs.cuh, over the interval from the
/// nuclear-size cut-off down to `cos_theta_max` - which is the complement of the interval
/// G4WentzelVIModel's transport cross section covers.
///
/// @param cos_theta_min  the model's `cosThetaMin`, -1 in option0 (MscThetaLimit == pi)
/// @param cos_theta_max  the model's `cosThetaMax`, -1 always: it is set in the constructor
///                       and never reassigned anywhere in G4eCoulombScatteringModel
/// @param cut            the electron production threshold, for ComputeMaxElectronScattering
template <typename real_t>
__host__ __device__ inline CoulombAtomXs<real_t> coulomb_xs_per_atom(
    const ParticleDef<real_t>& pd, ParticleType type, real_t tkin, real_t inv_a23, int z,
    real_t cut, real_t cos_theta_min, real_t cos_theta_max) {
  CoulombAtomXs<real_t> r;
  if (tkin <= real_t(0)) { return r; }

  // SetupKinematic first, and its answer is the gate. `if(cosThetaMax < costmin)` with
  // costmin = cosTetMaxNuc: when the two are equal there is no interval left and the cross
  // section is zero, which is how the process switches itself off at low energy in a heavy
  // material without any energy limit being set.
  const WentzelState<real_t> s0 =
      wentzel_setup(pd, type, tkin, inv_a23, z, cut, cos_theta_min);
  if (cos_theta_max >= s0.cos_tet_max_nuc) { return r; }

  // SetupTarget. wentzel_setup does SetupKinematic and SetupTarget together, so the per-Z
  // state is already here; what SetupTarget RETURNS is cosTetMaxNuc possibly forced to zero,
  // and that forcing is inside wentzel_setup.
  const real_t costmin = s0.cos_tet_max_nuc;
  // The same proton-on-hydrogen exception on the far end of the interval. Geant4 spells it
  // out twice - once inside SetupTarget for the near end and once here for the far end - and
  // the two are not the same test: this one reads the MODEL's cosThetaMax, which SetupTarget
  // cannot see.
  const bool p_on_h = (z == 1 && type == ParticleType::kProton && cos_theta_max < real_t(0));
  const real_t costmax = p_on_h ? real_t(0) : cos_theta_max;
  if (costmin <= costmax) { return r; }

  r.cos_t_min = costmin;
  r.cos_t_max = costmax;
  r.nuclear = wentzel_nuclear_xs(s0, z, costmin, costmax);
  r.electron = wentzel_electron_xs(s0, costmin, costmax);
  r.total = r.nuclear + r.electron;
  r.elec_ratio = (r.total > real_t(0)) ? r.electron / r.total : real_t(0);
  return r;
}

/// Cross section per unit volume, 1/mm - `G4VEmProcess::CrossSectionPerVolume` summed over the
/// material's elements, which is what the process's lambda table holds.
template <typename real_t>
__host__ __device__ inline real_t coulomb_xs_per_volume(const data::Material<real_t>& m,
                                                        ParticleType type, real_t tkin,
                                                        real_t cos_theta_min,
                                                        real_t cos_theta_max) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || tkin <= real_t(0)) { return real_t(0); }
  real_t xs = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    const CoulombAtomXs<real_t> a = coulomb_xs_per_atom(pd, type, tkin, m.inv_a23, z,
                                                        m.cut_electron, cos_theta_min,
                                                        cos_theta_max);
    xs += m.n_atoms[i] * a.total;
  }
  return xs;
}

/// The element the scatter happens on, by partial cross section.
///
/// `G4VEmModel::SelectTargetAtom` reads G4EmElementSelector's tabulated cumulative
/// distribution; this evaluates the same partial cross sections at the energy. See the
/// refusal list in the file header. Returns the element INDEX in the material.
template <typename real_t, typename Rng>
__host__ __device__ inline int coulomb_select_element(const data::Material<real_t>& m,
                                                      ParticleType type, real_t tkin,
                                                      real_t cos_theta_min,
                                                      real_t cos_theta_max, Rng& rng) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  real_t part[data::kMaxElements];
  real_t sum = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    const CoulombAtomXs<real_t> a = coulomb_xs_per_atom(pd, type, tkin, m.inv_a23, z,
                                                        m.cut_electron, cos_theta_min,
                                                        cos_theta_max);
    sum += m.n_atoms[i] * a.total;
    part[i] = sum;
  }
  if (!(sum > real_t(0))) { return -1; }
  const real_t q = sum * rng.uniform();
  for (int i = 0; i < m.n_elements; ++i) {
    if (q <= part[i]) { return i; }
  }
  return m.n_elements - 1;
}

/// `G4WentzelOKandVIxSection::SampleSingleScattering`, with the exponential nuclear form factor
/// and with factD.
///
/// This is NOT `wv_sample_single` in em/wentzel_msc.cuh, and the difference is one factor:
///
///     grej = (1. - z1*factB + factB1*targetZ*sqrt(z1*factB)*(2. - z1))*fm*fm/(1.0 + z1*factD)
///                                                                     ^^^^^^^^^^^^^^^^^^^^^
///
/// `factD = sqrt(mom2)/targetMass` (G4WentzelOKandVIxSection.hh, SetTargetMass), and
/// `SetTargetMass` is called by SetupTarget for EVERY target (G4WentzelOKandVIxSection.cc:205),
/// which G4WentzelVIModel::SampleScattering calls at line 615 immediately before
/// SampleSingleScattering at line 618. So factD is not zero, in either caller.
/// `wv_sample_single`'s comment says it is - "set only for particles with a magnetic-moment
/// correction; it is zero for the particles here" - and drops the factor. For a 200 MeV proton
/// on oxygen, sqrt(mom2) = 644.5 MeV and targetMass = 14903.9 MeV, so factD = 0.0433 and
/// 1/(1 + z1*factD) runs from 1 at zero angle to 0.920 at 180 degrees: an 8% error in the
/// rejection function at the large angles that are this process's whole subject. Recorded in
/// docs/RISK.md V47 rather than fixed, because em/wentzel_msc.cuh is not this package's file.
///
/// @param target_mass the mass factD is built from, MeV. SetupTarget uses
///        `GetAtomicMassAmu(Z)*amu_c2` for Z > 1 and the proton mass for Z == 1;
///        G4eCoulombScatteringModel::SampleSecondaries then OVERRIDES it with
///        `G4NucleiProperties::GetNuclearMass(ia, iz)` after the cross sections and before
///        this call, so the sampler and the cross section see different masses. The caller
///        passes whichever of the two applies.
/// @return the scattered direction in the frame where the primary moves along +z. `(0,0,1)`
///         when the rejection fails, which is Geant4's "no scattering" answer and not an
///         error: `temp` is set to (0,0,1) before the loop and returned unchanged.
template <typename real_t, typename Rng>
__host__ __device__ inline Vec3<real_t> coulomb_sample_single(const WentzelState<real_t>& s,
                                                              int z, real_t target_mass,
                                                              real_t cos_t_min,
                                                              real_t cos_t_max,
                                                              real_t elec_ratio, Rng& rng) {
  Vec3<real_t> out{real_t(0), real_t(0), real_t(1)};
  real_t formf = s.form_fact_a;
  real_t cost1 = cos_t_min, cost2 = cos_t_max;
  // The electron/nucleus choice consumes a random number whenever elecRatio is positive, and
  // it is drawn BEFORE the angle. Order matters for reproducing a stream.
  if (elec_ratio > real_t(0) && rng.uniform() <= elec_ratio) {
    formf = real_t(0);  // off an atomic electron: no nuclear form factor
    cost1 = fmax(cost1, s.cos_tet_max_elec);
    cost2 = fmax(cost2, s.cos_tet_max_elec);
  }
  if (cost1 <= cost2) { return out; }

  const real_t w1 = real_t(1) - cost1 + s.screen_z;
  const real_t w2 = real_t(1) - cost2 + s.screen_z;
  const real_t z1 = w1 * w2 / (w1 + rng.uniform() * (w2 - w1)) - s.screen_z;
  real_t fm = real_t(1) + formf * z1;
  fm = real_t(1) / (fm * fm);

  const real_t fact_d = (target_mass > real_t(0)) ? sqrt(s.mom2) / target_mass : real_t(0);
  real_t grej;
  if (s.use_mott) {
    // For e- and e+ the rejection function is G4ScreeningMottCrossSection's Mott/Rutherford
    // ratio and factD does not appear. Its beta is the projectile-nucleus relative-system one,
    // so it depends on the target Z and is computed per element, where Geant4 recomputes it
    // with SetupKinematic(tkin, targetZ) at this same point.
    const real_t beta = data::mott_beta<real_t>(z, s.tkin, s.mass);
    grej = data::mott_ratio<real_t>(z, beta, sqrt(z1)) * fm * fm;
  } else {
    grej = (real_t(1) - z1 * s.fact_b
            + wv_fact_b1<real_t>() * real_t(z) * sqrt(z1 * s.fact_b) * (real_t(2) - z1))
           * fm * fm / (real_t(1) + z1 * fact_d);
  }
  if (s.mott_factor * rng.uniform() <= grej) {
    real_t cost = real_t(1) - z1;
    if (cost > real_t(1)) { cost = real_t(1); }
    if (cost < real_t(-1)) { cost = real_t(-1); }
    const real_t sint = sqrt((real_t(1) - cost) * (real_t(1) + cost));
    const real_t phi = units::twopi<real_t>() * rng.uniform();
    out = Vec3<real_t>{sint * cos(phi), sint * sin(phi), cost};
  }
  return out;
}

/// What one `G4eCoulombScatteringModel::SampleSecondaries` call produces.
template <typename real_t>
struct CoulombFinalState {
  real_t cos_theta = 1;    ///< of the primary, about its incoming direction
  real_t phi = 0;
  real_t final_t = 0;      ///< the primary's kinetic energy after the scatter, MeV
  real_t trec = 0;         ///< the recoil's kinetic energy, MeV
  real_t edep = 0;         ///< deposited locally, MeV (the recoil, when it is below threshold)
  bool emit_ion = false;   ///< true when the recoil is emitted as an ion instead
  int ion_z = 0;
  int ion_a = 0;
};

/// The refusal `coulomb_sample_secondaries` cannot make for its caller.
///
/// Reproducing `SelectIsotopeNumber` needs per-element natural abundances, which this port has
/// no table of - see the file header. A caller that has only the element must say so rather
/// than substitute the mean A: `G4NucleiProperties::GetNuclearMass(lrint(<A>), Z)` is not the
/// abundance-weighted mean of the isotope masses, and using it would be an approximation
/// wearing a transcription's clothes.
__host__ __device__ inline const char* coulomb_refuse_isotope_selection() {
  return "G4eCoulombScatteringModel::SampleSecondaries: SelectIsotopeNumber needs per-element "
         "natural abundances; data/natural_isotopes.hh carries the nuclide set only. Pass the "
         "isotope explicitly.";
}

/// `G4eCoulombScatteringModel::SampleSecondaries`, verbatim from the direction sampling to the
/// energy balance.
///
/// The recoil, which is the part that is not just an angle:
///
///     G4double trec = mom2*(1.0 - cost)/(targetMass + (mass + kinEnergy)*(1.0 - cost));
///     trec = std::min(trec, kinEnergy);
///     G4double finalT = kinEnergy - trec;
///     G4double tcut = recoilThreshold;                       // 0.0 in QBBC
///     if(pCuts) { tcut = std::max(tcut, (*pCuts)[currentMaterialIndex]); }
///     if(trec > tcut) { ... emit G4IonTable::GetIon(iz, ia, 0) ... }
///     else { edep = trec; ProposeNonIonizingEnergyDeposit(edep); }
///     if(finalT < 0.0) { edep += finalT; finalT = 0.0; }
///     edep = std::max(edep, 0.0);
///
/// The comment above it in the source says "recoil sampling assuming a small recoil and first
/// order correction to primary 4-momentum", and that is exactly what the expression is: the
/// primary keeps the sampled direction and loses `trec`, with no exact two-body solve. Note
/// that `edep` is proposed as NON-IONIZING when the recoil is below threshold, so a scorer
/// that separates the two will see it there.
///
/// @param proton_cut `(*pCuts)[currentMaterialIndex]` - the PROTON production threshold, MeV
/// @param target_mass `G4NucleiProperties::GetNuclearMass(ia, iz)`, MeV; the caller supplies it
///        together with (iz, ia), because selecting the isotope is refused above
template <typename real_t, typename Rng>
__host__ __device__ inline CoulombFinalState<real_t> coulomb_sample_secondaries(
    const ParticleDef<real_t>& pd, ParticleType type, real_t tkin, real_t inv_a23, int iz,
    int ia, real_t target_mass, real_t electron_cut, real_t proton_cut, real_t cos_theta_min,
    real_t cos_theta_max, Rng& rng) {
  CoulombFinalState<real_t> r;
  r.final_t = tkin;

  const CoulombAtomXs<real_t> a = coulomb_xs_per_atom(pd, type, tkin, inv_a23, iz,
                                                      electron_cut, cos_theta_min,
                                                      cos_theta_max);
  if (!(a.total > real_t(0))) { return r; }

  const WentzelState<real_t> s =
      wentzel_setup(pd, type, tkin, inv_a23, iz, electron_cut, cos_theta_min);
  const Vec3<real_t> dir = coulomb_sample_single(s, iz, target_mass, a.cos_t_min, a.cos_t_max,
                                                 a.elec_ratio, rng);
  const real_t cost = dir.z;
  r.cos_theta = cost;
  r.phi = atan2(dir.y, dir.x);

  real_t trec = s.mom2 * (real_t(1) - cost)
                / (target_mass + (pd.mass + tkin) * (real_t(1) - cost));
  trec = fmin(trec, tkin);
  real_t final_t = tkin - trec;
  real_t edep = real_t(0);

  const real_t tcut = fmax(real_t(0), proton_cut);   // recoilThreshold = 0 in QBBC
  if (trec > tcut) {
    r.emit_ion = true;
    r.ion_z = iz;
    r.ion_a = ia;
  } else {
    edep = trec;
  }
  // "this threshold may be applied only because for low-energy e+e- msc model is applied"
  if (final_t < real_t(0)) {
    edep += final_t;
    final_t = real_t(0);
  }
  r.trec = trec;
  r.edep = fmax(edep, real_t(0));
  r.final_t = final_t;
  return r;
}

}  // namespace g4gpu::em
