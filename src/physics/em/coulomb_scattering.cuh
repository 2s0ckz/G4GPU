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
// THE FOUR SWITCHES IN InitialiseProcess AND Initialise, AND WHERE EACH ONE LANDS IN OPTION0
//
// All four are live code in 11.1.1 and all four are pinned in option0, so this port implements
// one setting of each. Written down because a reader of the code below cannot tell a constant
// that happens to be one from a constant that is one by construction.
//
//  * `mass > CLHEP::GeV || p->GetParticleType() == "nucleus"` (G4CoulombScattering.cc:121)
//    picks G4IonCoulombScatteringModel over G4eCoulombScatteringModel and calls
//    `SetBuildTableFlag(false)` - no lambda table, the cross section computed per step. NEVER
//    TAKEN in option0: the only species that reach InitialiseProcess are the ten in the table
//    above, and the heaviest is the proton at 938.272 MeV. The ions would take it, and they
//    only get the process through ConstructIonEmPhysicsSS. So `build_table` is 1 in every row
//    of `ref/oracle/coulomb_limits.csv`.
//  * `isCombined`, the model's constructor argument (G4eCoulombScatteringModel.cc:72), defaults
//    to true and does two things: `wokvi = new G4WentzelOKandVIxSection(isCombined)` (:88) and
//    gating the cosThetaMin recompute in Initialise (:111). G4CoulombScattering constructs the
//    model with no arguments, so it is TRUE, and `cosThetaMin` is therefore recomputed from
//    PolarAngleLimit rather than left at its constructor value of 1.0. With combined false the
//    interval would be empty and the process would never fire - which is why this is worth a
//    line rather than a shrug.
//  * `fixedCut` (G4eCoulombScatteringModel.cc:79, `fixedCut = -1.0`) overrides the production
//    cut inside both `ComputeCrossSectionPerAtom` (:197) and `SampleSecondaries` (:239) through
//    `G4double cut = (0.0 < fixedCut) ? fixedCut : cutEnergy`. `SetFixedCut` is DECLARED on
//    G4eCoulombScatteringModel, G4hCoulombScatteringModel and G4WentzelVIModel and CALLED
//    NOWHERE in the release, so `cutEnergy` always wins and `GetFixedCut` reads -1. Note what
//    it is not: despite sitting next to the angular limits it is a cut in ENERGY, not in cos.
//  * `SetSingleScatteringFactor` (G4WentzelVIModel.cc:85, `1.25`, with `invssFactor` 1/1.2)
//    belongs to the MSC model's own step limit and its own multiple/single sub-mode split, and
//    never reaches this process. The only override in the release is
//    G4LowEWentzelVIModel.cc:60's `SetSingleScatteringFactor(0.5)`, which option0 does not use.
//
// So the coupling the msc/single-scattering pair is supposed to have - one angular limit shared
// between them - is carried by `MscThetaLimit` alone, and in option0 that is pi and the limit
// does not separate them at all. See the section above.
//
// ---------------------------------------------------------------------------------------
// WHICH PRODUCTION CUT ARRIVES HERE, AND IT IS THE PROTON'S
//
// `G4eCoulombScatteringModel` uses its `cutEnergy` argument in exactly one place: through
// `G4WentzelOKandVIxSection::SetupTarget(iz, cut)` into `ComputeMaxElectronScattering(cut)`,
// where it bounds the energy transfer to an atomic ELECTRON and so sets `cosTetMaxElec` and the
// electron cross section (G4WentzelOKandVIxSection.cc, ComputeMaxElectronScattering). It is a
// delta-ray production threshold by construction. The value the transport hands it is the
// PROTON production cut:
//
//   * G4CoulombScattering's constructor calls `SetSecondaryParticle(G4Proton::Proton())`
//     (G4CoulombScattering.cc:68) - because the recoil it can emit is an ion.
//   * `G4EmModelManager::Initialise` turns that into a cuts INDEX: gamma 0, e- 1, e+ 2, and
//     `else { idx = 3; }` for anything else (G4EmModelManager.cc:463-468), then
//     `theCuts = theCoupleTable->GetEnergyCutsVector(idx)` (:471).
//   * Every later use of a cut in the process reads that vector: the lambda table through
//     `G4EmModelManager::FillLambdaVector`'s `G4double cut = (*theCuts)[i]` (:634), and
//     `G4VEmProcess::PostStepDoIt`'s `SampleSecondaries(..., (*theCuts)[currentCoupleIndex])`
//     (G4VEmProcess.cc:527).
//   * `G4eCoulombScatteringModel::Initialise` also stores that same vector as `pCuts`
//     (G4eCoulombScatteringModel.cc:117), which is what `SampleSecondaries` reads for the
//     recoil threshold. So the two cuts in this model are one number arriving twice.
//
// The electron production cut NEVER reaches this model in QBBC, not even for an e-
// projectile - in G4_WATER the two differ by a factor of four, 0.07 MeV against 0.2776 MeV.
//
// AND IT MAKES NO DIFFERENCE TO THIS PROCESS, WHICH IS THE PART WORTH KNOWING. The cut gates
// exactly one channel and that channel is shut. `ComputeElectronCrossSection`
// (G4WentzelOKandVIxSection.hh:222-230) opens with
//
//     G4double cost1 = std::max(cosTMin, cosTetMaxElec);
//     G4double cost2 = std::max(cosTMax, cosTetMaxElec);
//     return (cost1 <= cost2) ? 0.0 : ...
//
// and this process integrates from cosTMin = cosTetMaxNuc out to cosTMax = -1, so the channel
// is open only when `cosTetMaxElec < cosTetMaxNuc`. For a heavy projectile those are
// 1 - cut*m_e/mom2 and 1 - 0.5*q2Max*<A^-2/3>/mom2, so the condition is
//
//     cut * m_e  >  0.5 * q2Max * <A^-2/3>
//
// with q2Max = 19469 MeV^2 and <A^-2/3> = 0.1686 in water: 0.0358 MeV^2 against 1641, a factor
// of 46,000, and the mom2 cancels so no energy changes it. Measured over all 20,182 active
// rows of `ref/oracle/coulomb_xs.csv` at BOTH cuts: `xs_electron` is 0 in every one, the
// smallest gap `cosTetMaxElec - cosTetMaxNuc` is +1.9e-7, and `elec_ratio` is 0 in all 240
// sampler cells. It would take a ~3.2 GeV production cut in water to open it.
//
// So two things follow. The port must pass the PROTON cut to be right about the plumbing, and
// it costs nothing to be wrong about it in option0 - which is why this is written down rather
// than left to be rediscovered by whoever changes `MscThetaLimit` and moves cosTMin. And
// `wentzel_electron_xs` and `coulomb_sample_single`'s electron branch, both transcribed, are
// NOT exercised through this process by any oracle cell here. The msc model is a different
// caller with a different cosThetaMin and is where they earn their place.
//
// The cut itself needs no table. `G4RToEConvForProton::Convert` is
//
//     // Simple formula - range = Ekin/(100*keV)*(1*mm);
//     return (rangeCut/(1.0*CLHEP::mm)) * (100.0*CLHEP::keV);
//
// (processes/cuts/src/G4RToEConvForProton.cc) - linear in the range cut and independent of the
// material, which is why the oracle's `pcut_MeV` column is 0.07 in all seven materials while
// its `ecut_MeV` spans 0.00099 to 0.61. `coulomb_secondary_cut` below is that one line, and it
// is why this file needs no proton-cut field on `data::Material`.
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
//
// ---------------------------------------------------------------------------------------
// P8b: THE ISOTOPE REFUSAL IS CLOSED, AND THE CALL SITE IS NOW IN THIS FILE
//
// `coulomb_refuse_isotope_selection` was the first item on the list above: reproducing
// `G4VEmModel::SelectIsotopeNumber` needs per-element natural abundances and no table in the
// port had them. `data/isotope_abundance.hh` is G4NistElementBuilder's table now, compared
// isotope by isotope against a QBBC-initialised G4NistManager (`ref/oracle/isotopes.csv`), and
// `data::nist_sample_isotope_n` is `G4EmUtility::SampleRandomIsotope` as written - including
// that a single-isotope element consumes NO uniform. So the recoil's target mass is the sampled
// isotope's `G4NucleiProperties::GetNuclearMass(ia, iz)`, which is what Geant4 hands the
// sampler, rather than the element's mean atomic mass.
//
// `coulomb_fire` at the bottom is the whole of one PostStepDoIt: element, isotope, nuclear mass,
// angle, recoil. It lives here rather than in `stepper.cuh` because `step_lepton` and
// `step_hadron` both make the same call in the same order, and the order is Geant4's - the cross
// sections belong to the ELEMENT and factD and the recoil to the sampled ISOTOPE, which is a
// sequence that must not be written twice.
#pragma once
#include <cmath>

#include "core/particle.cuh"
#include "core/units.cuh"
#include "data/isotope_abundance.hh"
#include "data/materials.cuh"
#include "data/mott.hh"
#include "data/nuclei_mass_ame12.hh"
#include "physics/em/wentzel_msc.cuh"
#include "physics/em/wentzel_xs.cuh"

namespace g4gpu::em {

/// `G4EmParameters::MscEnergyLimit()`, MeV - 100 MeV, set in G4EmParameters::Initialise
/// (`energyLimit = 100.0*CLHEP::MeV`, G4EmParameters.cc:155).
///
/// It does two jobs for e+- and only for e+-: it is the Urban/WentzelVI msc split AND the lower
/// edge of `G4CoulombScattering`'s table, because G4EmStandardPhysics.cc fetches it once at
/// line 134 and hands it to `SetMinKinEnergy`, `SetLowEnergyLimit` and
/// `SetActivationLowEnergyLimit` alike. The light hadrons get none of those calls, so for them
/// the process starts at 100 eV and the only threshold is the material-dependent
/// `coulomb_process_min_primary_energy`.
template <typename real_t> __host__ __device__ constexpr real_t kMscEnergyLimit() {
  return real_t(100);
}

/// `G4VEmProcess`'s own floor for a light hadron, MeV - 100 eV.
///
/// `G4VEmProcess`'s constructor sets `minKinEnergy(0.1*CLHEP::keV)` and nothing in
/// G4EmBuilder::ConstructLightHadrons overrides it, so this is where a muon's, pion's, kaon's
/// or proton's `CoulombScat` table begins. Below it there is no process at all, which is a
/// different statement from `coulomb_process_min_primary_energy` - that is the energy the table
/// is zero below, in a particular material.
template <typename real_t> __host__ __device__ constexpr real_t kCoulombHadronMinEnergy() {
  return real_t(1e-4);
}

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

/// `G4RToEConvForProton::Convert(rangeCut, material)`, MeV - the cut this process is driven by.
///
///     // Simple formula - range = Ekin/(100*keV)*(1*mm);
///     return (rangeCut/(1.0*CLHEP::mm)) * (100.0*CLHEP::keV);
///
/// Linear in the range cut and independent of the material, so 0.07 MeV for QBBC's 0.7 mm in
/// every one of the oracle's seven materials. See the file header: this is the value that
/// reaches `cutEnergy` and `pCuts` alike, because G4CoulombScattering's secondary is the
/// proton. Passing an electron cut instead moves `cosTetMaxElec` - by up to 0.66 in cos, which
/// `tests/test_coulomb_scattering.cu` compares at both cuts - but not the cross section, which
/// is the header's point: the electron channel it gates is closed at every energy.
template <typename real_t> __host__ __device__ inline real_t coulomb_secondary_cut(
    real_t range_cut_mm) {
  return range_cut_mm * real_t(0.1);   // 100 keV per mm, in MeV/mm
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
/// @param cut            the threshold ComputeMaxElectronScattering bounds the energy transfer
///                       to an atomic electron with. It is a delta-ray threshold by
///                       construction and the PROTON production cut by plumbing - see the file
///                       header - so `coulomb_secondary_cut(range_cut_mm)`, not
///                       `Material::cut_electron`. It is an argument and not read off the
///                       material because the material carries no proton cut and does not need
///                       one: the value is a per-run constant.
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
///
/// @param cut `coulomb_secondary_cut(range_cut_mm)`. `G4EmModelManager::FillLambdaVector`
///        builds this table with `(*theCuts)[i]`, and theCuts is the PROTON vector - see the
///        file header. This took `m.cut_electron` until it was measured: the electron cut is
///        four times the proton cut in water and never reaches the model.
/// `__noinline__` since P8b, and see docs/RISK.md V61: inlined, this function's Wentzel setup
/// and per-element loop go into the body of `step_lepton` and of `run_step_hadron`'s fourteen
/// species, and the translation unit that holds all of them killed ptxas with an access
/// violation once `hadElastic` was inlined beside it. One call per step against a function that
/// loops over the material's elements is not a cost worth measuring.
template <typename real_t>
__host__ __device__ __noinline__ real_t coulomb_xs_per_volume(const data::Material<real_t>& m,
                                                              ParticleType type, real_t tkin,
                                                              real_t cut, real_t cos_theta_min,
                                                              real_t cos_theta_max) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || tkin <= real_t(0)) { return real_t(0); }
  real_t xs = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    const CoulombAtomXs<real_t> a = coulomb_xs_per_atom(pd, type, tkin, m.inv_a23, z, cut,
                                                        cos_theta_min, cos_theta_max);
    xs += m.n_atoms[i] * a.total;
  }
  return xs;
}

/// The element the scatter happens on, by partial cross section.
///
/// `G4VEmModel::SelectTargetAtom` reads G4EmElementSelector's tabulated cumulative
/// distribution; this evaluates the same partial cross sections at the energy. See the
/// refusal list in the file header. Returns the element INDEX in the material.
///
/// @param cut `coulomb_secondary_cut(range_cut_mm)`, as in `coulomb_xs_per_volume`.
template <typename real_t, typename Rng>
__host__ __device__ inline int coulomb_select_element(const data::Material<real_t>& m,
                                                      ParticleType type, real_t tkin,
                                                      real_t cut, real_t cos_theta_min,
                                                      real_t cos_theta_max, Rng& rng) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  real_t part[data::kMaxElements];
  real_t sum = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    const CoulombAtomXs<real_t> a = coulomb_xs_per_atom(pd, type, tkin, m.inv_a23, z, cut,
                                                        cos_theta_min, cos_theta_max);
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
/// This is NOT `wv_sample_single` in em/wentzel_msc.cuh, and what separates them is the
/// TARGET MASS in one shared factor:
///
///     grej = (1. - z1*factB + factB1*targetZ*sqrt(z1*factB)*(2. - z1))*fm*fm/(1.0 + z1*factD)
///                                                                     ^^^^^^^^^^^^^^^^^^^^^
///
/// `factD = sqrt(mom2)/targetMass` (G4WentzelOKandVIxSection.hh, SetTargetMass), and
/// `SetTargetMass` is called by SetupTarget for EVERY target (G4WentzelOKandVIxSection.cc:205),
/// which G4WentzelVIModel::SampleScattering calls at line 615 immediately before
/// SampleSingleScattering at line 618. So factD is not zero in either caller - but the two
/// callers build it from different masses, which is why this function takes the mass as an
/// ARGUMENT and `wv_sample_single` derives it from Z through `em::wv_target_mass`. See @p
/// target_mass below.
///
/// `wv_sample_single` dropped the factor entirely until docs/RISK.md V47 was closed, under a
/// comment claiming factD was zero for these particles. For a 200 MeV proton on oxygen,
/// sqrt(mom2) = 644.5 MeV and targetMass = 14903.9 MeV, so factD = 0.0433 and 1/(1 + z1*factD)
/// runs from 1 at zero angle to 0.920 at 180 degrees.
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
  /// The recoil ion's direction, in the same frame as the direction passed in. Only meaningful
  /// when `emit_ion`; `coulomb_recoil` leaves it (0,0,1) because it works in the scattering
  /// frame and does not know the incoming direction.
  Vec3<real_t> ion_dir{real_t(0), real_t(0), real_t(1)};
};

/// `G4eCoulombScatteringModel::SampleSecondaries`' recoil and energy balance, given the angle.
///
///     G4double mom2 = wokvi->GetMomentumSquare();
///     G4double trec = mom2*(1.0 - cost)/(targetMass + (mass + kinEnergy)*(1.0 - cost));
///     trec = std::min(trec, kinEnergy);                       // "the check likely not needed"
///     G4double finalT = kinEnergy - trec;
///     G4double edep = 0.0;
///     G4double tcut = recoilThreshold;                        // 0.0, and never set in QBBC
///     if(pCuts) { tcut = std::max(tcut,(*pCuts)[currentMaterialIndex]); }
///     if(trec > tcut) { ... emit theIonTable->GetIon(iz, ia, 0) with kinetic energy trec ... }
///     else { edep = trec; fParticleChange->ProposeNonIonizingEnergyDeposit(edep); }
///     if(finalT < 0.0) { edep += finalT; finalT = 0.0; }
///     edep = std::max(edep, 0.0);
///
/// (G4eCoulombScatteringModel.cc:277-314). The comment above it says "recoil sampling assuming
/// a small recoil and first order correction to primary 4-momentum", and that is what the
/// expression is: the primary keeps the sampled DIRECTION and loses `trec`, with no exact
/// two-body solve, so the primary's momentum after the step is not the two-body value.
///
/// Split out from `coulomb_sample_secondaries` so it can be driven with an angle rather than an
/// RNG: `tests/test_coulomb_scattering.cu` feeds it the cos(theta) of each of Geant4's own
/// `SampleSecondaries` calls out of `ref/oracle/coulomb_recoil.csv` and compares trec, finalT,
/// edep and the branch exactly. Sampling the angle here instead would leave the arithmetic
/// checkable only through the statistics of a different random stream.
///
/// Note where the deposit goes: below threshold it is proposed as NON-IONIZING as well as
/// local, so a scorer that separates the two sees the recoil there.
///
/// @param mom2 `wokvi->GetMomentumSquare()`, MeV^2 - `WentzelState::mom2`
/// @param tcut `max(recoilThreshold, (*pCuts)[i])`, i.e. `coulomb_secondary_cut(range_cut_mm)`
///        in QBBC, where recoilThreshold is 0 and pCuts is the proton vector
template <typename real_t>
__host__ __device__ inline CoulombFinalState<real_t> coulomb_recoil(real_t mass, real_t tkin,
                                                                    real_t mom2,
                                                                    real_t target_mass,
                                                                    real_t cos_theta,
                                                                    real_t tcut) {
  CoulombFinalState<real_t> r;
  r.cos_theta = cos_theta;
  const real_t one_minus_cost = real_t(1) - cos_theta;
  real_t trec = mom2 * one_minus_cost / (target_mass + (mass + tkin) * one_minus_cost);
  trec = fmin(trec, tkin);
  real_t final_t = tkin - trec;
  real_t edep = real_t(0);
  if (trec > tcut) {
    r.emit_ion = true;
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

/// The recoil ion's direction, from momentum balance (G4eCoulombScatteringModel.cc:296-297):
///
///     G4ThreeVector dir = (direction*sqrt(mom2) -
///                          newDirection*sqrt(finalT*(2*mass + finalT))).unit();
///
/// The primary's momentum before minus its momentum after, with the after magnitude rebuilt
/// from `finalT` - the energy the recoil took - and not from the sampled angle. `mass` is the
/// PROJECTILE's on both sides; the ion's own mass never enters, which is the same first-order
/// approximation `coulomb_recoil` carries. When the rejection failed and `cos_theta` is 1 the
/// two momenta are parallel and the difference is along the primary, so this returns the
/// incoming direction - but that case never emits an ion, because trec is then zero.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> coulomb_recoil_direction(const Vec3<real_t>& in_dir,
                                                                 const Vec3<real_t>& new_dir,
                                                                 real_t mom2, real_t mass,
                                                                 real_t final_t) {
  const real_t p_out = sqrt(final_t * (real_t(2) * mass + final_t));
  return normalize(in_dir * sqrt(mom2) - new_dir * p_out);
}

/// The refusal `coulomb_sample_secondaries` cannot make for its caller.
///
/// KEPT, AND NO LONGER REACHED FROM THIS PORT'S TRANSPORT. P8b added
/// `data/isotope_abundance.hh`, so `coulomb_fire` below does the draw and passes the isotope.
/// What this still says is true of any caller that does NOT have an isotope composition - a
/// material built from explicit isotopes, or an element outside G4NistElementBuilder's
/// Z = 1..107 - and the statement it makes is the one that matters:
/// `G4NucleiProperties::GetNuclearMass(lrint(<A>), Z)` is not the abundance-weighted mean of the
/// isotope masses, so substituting it would be an approximation wearing a transcription's
/// clothes. `tests/test_coulomb_scattering.cu` asserts the text; `coulomb_fire` returns
/// `fired = false` where it would apply.
__host__ __device__ inline const char* coulomb_refuse_isotope_selection() {
  return "G4eCoulombScatteringModel::SampleSecondaries: SelectIsotopeNumber needs per-element "
         "natural abundances; data/natural_isotopes.hh carries the nuclide set only. Pass the "
         "isotope explicitly.";
}

/// `G4eCoulombScatteringModel::SampleSecondaries`, from the angle to the energy balance.
///
/// The order is Geant4's and it matters: the cross sections are computed with SetupTarget's
/// mean atomic mass, then `SetTargetMass(GetNuclearMass(ia, iz))` overrides it, and only then
/// is the angle sampled. So `factD` in the rejection function and `targetMass` in the recoil
/// belong to the sampled ISOTOPE while `cosTetMaxNuc` and the two cross sections belong to the
/// ELEMENT. Selecting the isotope is refused (see the file header), so both (iz, ia) and the
/// nuclear mass come from the caller.
///
/// @param cut `coulomb_secondary_cut(range_cut_mm)`. One value, used twice: as `cutEnergy` in
///        SetupTarget and as `(*pCuts)[i]` in the recoil threshold. In 11.1.1 those are
///        literally the same vector - see the file header - so a caller that had two numbers
///        here would be modelling something the release cannot do.
/// @param target_mass `G4NucleiProperties::GetNuclearMass(ia, iz)`, MeV
/// @param in_dir the primary's direction before the scatter, for the recoil ion's direction and
///        for `rotateUz`. `cos_theta` and `phi` in the result stay in the scattering frame.
template <typename real_t, typename Rng>
__host__ __device__ inline CoulombFinalState<real_t> coulomb_sample_secondaries(
    const ParticleDef<real_t>& pd, ParticleType type, real_t tkin, real_t inv_a23, int iz,
    int ia, real_t target_mass, real_t cut, real_t cos_theta_min, real_t cos_theta_max,
    const Vec3<real_t>& in_dir, Rng& rng) {
  CoulombFinalState<real_t> r;
  r.final_t = tkin;
  r.ion_dir = in_dir;

  const CoulombAtomXs<real_t> a =
      coulomb_xs_per_atom(pd, type, tkin, inv_a23, iz, cut, cos_theta_min, cos_theta_max);
  if (!(a.total > real_t(0))) { return r; }

  const WentzelState<real_t> s =
      wentzel_setup(pd, type, tkin, inv_a23, iz, cut, cos_theta_min);
  const Vec3<real_t> dir = coulomb_sample_single(s, iz, target_mass, a.cos_t_min, a.cos_t_max,
                                                 a.elec_ratio, rng);

  r = coulomb_recoil(pd.mass, tkin, s.mom2, target_mass, dir.z, fmax(real_t(0), cut));
  r.phi = atan2(dir.y, dir.x);
  if (r.emit_ion) {
    r.ion_z = iz;
    r.ion_a = ia;
  }
  // rotateUz, so the ion's direction is in the caller's frame - which is the frame Geant4
  // computes it in, after `newDirection.rotateUz(direction)`.
  r.ion_dir = coulomb_recoil_direction(in_dir, rotate_uz(dir, in_dir), s.mom2, pd.mass,
                                       r.final_t);
  return r;
}

// =============================================================================================
// The call site: one G4CoulombScattering::PostStepDoIt, written once for both steppers (P8b)
// =============================================================================================

/// Whether this species has `CoulombScat` on its process manager in QBBC at all.
///
/// The table at the top of this file, as a predicate. It is exactly `uses_wentzel_msc` for the
/// hadrons - `G4EmBuilder::ConstructLightHadrons` registers a WentzelVI msc and a
/// G4CoulombScattering in the same two lines for mu+-, pi+-, K+- and p/pbar - plus e+-, which
/// get it from G4EmStandardPhysics directly. It is FALSE for alpha, He3, deuteron, triton and
/// GenericIon: they reach the process only through `G4EmBuilder::ConstructIonEmPhysicsSS`,
/// which option0 never calls, so an ion in QBBC has no single Coulomb scattering.
///
/// Not derived from `uses_wentzel_msc` even though the two agree on the hadrons: that predicate
/// records which species this port SUBSTITUTES WentzelVI for (the ions included, as a measured
/// substitution), and this one records which species Geant4 gives a process to. Tying them
/// together would make generalising urban_msc.cuh silently add a process.
__host__ __device__ inline bool has_coulomb_scattering(ParticleType t) {
  switch (t) {
    case ParticleType::kElectron:
    case ParticleType::kPositron:
    case ParticleType::kMuonMinus:
    case ParticleType::kMuonPlus:
    case ParticleType::kPionPlus:
    case ParticleType::kPionMinus:
    case ParticleType::kKaonPlus:
    case ParticleType::kKaonMinus:
    case ParticleType::kProton:
    case ParticleType::kAntiProton:
      return true;
    default:
      return false;
  }
}

/// The lower edge of this species' `CoulombScat` table, MeV, in this material.
///
/// TWO DIFFERENT NUMBERS FOR TWO GROUPS OF SPECIES, AND NEITHER IS THE OTHER'S DEFAULT.
///
/// For e+- it is `G4EmParameters::MscEnergyLimit()` - 100 MeV - because G4EmStandardPhysics
/// calls `SetMinKinEnergy`, `SetLowEnergyLimit` and `SetActivationLowEnergyLimit` with it. The
/// material-dependent threshold is computed too and never wins: it tops out at 98.15 MeV in
/// hydrogen, so the e+- table always starts at exactly 100 MeV.
///
/// For mu/pi/K/p it is the process's own 100 eV floor raised by
/// `coulomb_process_min_primary_energy`, which is where `SetStartFromNullFlag` puts the table's
/// first non-zero node. In water (<A^-2/3> = 0.1686) that is 1.75 MeV for a proton and 0.283 MeV
/// for a muon; in lead (0.02845) 0.295 MeV and 0.0477 MeV. So the threshold is not a detail:
/// it is the difference between a 200 MeV proton having this process over its whole track and
/// having it only above a couple of MeV.
template <typename real_t>
__host__ __device__ inline real_t coulomb_table_min_energy(ParticleType t, real_t mass,
                                                            real_t inv_a23) {
  if (t == ParticleType::kElectron || t == ParticleType::kPositron) {
    return kMscEnergyLimit<real_t>();
  }
  return fmax(kCoulombHadronMinEnergy<real_t>(),
              coulomb_process_min_primary_energy<real_t>(mass, inv_a23));
}

/// What the transport needs back from one `CoulombScat` PostStepDoIt.
template <typename real_t>
struct CoulombStepResult {
  bool fired = false;              ///< false when the element draw found no cross section
  Vec3<real_t> dir{real_t(0), real_t(0), real_t(1)};  ///< the primary's new direction, lab frame
  real_t final_t = 0;             ///< the primary's kinetic energy after the scatter
  real_t edep = 0;                ///< local AND non-ionizing, MeV - the sub-threshold recoil
  bool emit_ion = false;
  int ion_z = 0, ion_a = 0;
  real_t ion_ekin = 0;
  Vec3<real_t> ion_dir{real_t(0), real_t(0), real_t(1)};
};

/// One whole `G4CoulombScattering::PostStepDoIt`: element, isotope, nuclear mass, angle, recoil.
///
/// THE ORDER OF THE THREE RANDOM NUMBERS IS PART OF THE TRANSCRIPTION. `G4VEmProcess::
/// PostStepDoIt` calls `SelectTargetAtom` (one uniform, and none for a single-element material)
/// and then `SampleSecondaries`, which calls `SelectIsotopeNumber` (one uniform, and NONE for a
/// single-isotope element) before `SampleSingleScattering` (one for the angle, one for phi, one
/// for the rejection, and one more first if the electron channel is open). A port that drew them
/// in another order, or drew one Geant4 skips, would be sampling the same distributions off a
/// different stream - which is fine on its own and wrong the moment a second process shares the
/// stream.
///
/// @param cut `coulomb_secondary_cut(range_cut_mm)` - the PROTON production threshold, used both
///        as `cutEnergy` and as the recoil threshold. See the file header.
/// @return `fired = false` when there is no cross section in this material at this energy, which
///         is not an error: it is how the process switches itself off below the material's
///         threshold. The caller must then leave the track alone.
/// `__noinline__`, for the reason `coulomb_xs_per_volume` gives and one more: this is a branch
/// that almost never runs. `CoulombScat`'s mean free path in water is 158 m for a 200 MeV
/// proton and 483 m for a 1 GeV muon (`tests/test_step_hadron.cu` prints them), against B1's
/// 300 mm envelope - so inlining the sampler, the form factor, the Mott ratio and the recoil
/// into every step of every species buys nothing and cost the build.
template <typename real_t, typename Rng>
__host__ __device__ __noinline__ CoulombStepResult<real_t> coulomb_fire(
    const data::Material<real_t>& m, ParticleType type, real_t tkin, const Vec3<real_t>& in_dir,
    real_t cut, Rng& rng) {
  CoulombStepResult<real_t> out;
  out.dir = in_dir;
  out.final_t = tkin;
  const ParticleDef<real_t> pd = particle_def<real_t>(type);

  // cosThetaMin = cosThetaMax = -1: MscThetaLimit is pi in option0 and the model's cosThetaMax
  // is never reassigned. See the file header - the whole angular range, both here and in the
  // msc model, and no handover between them.
  constexpr real_t kCosMin = real_t(-1);
  constexpr real_t kCosMax = real_t(-1);

  const int ie = coulomb_select_element(m, type, tkin, cut, kCosMin, kCosMax, rng);
  if (ie < 0) { return out; }
  const int iz = static_cast<int>(m.z[ie] + real_t(0.5));

  // SelectIsotopeNumber, then GetNuclearMass(ia, iz) - the mass factD and the recoil use, and
  // NOT the element's mean atomic mass that the cross sections above were computed with.
  const int ia = data::nist_sample_isotope_n(iz, static_cast<double>(rng.uniform()));
  if (ia <= 0) { return out; }   // no NIST element at this Z; coulomb_refuse_isotope_selection
  const real_t target_mass = data::nuclear_mass<real_t>(ia, iz);
  if (!(target_mass > real_t(0))) { return out; }

  const CoulombFinalState<real_t> fs = coulomb_sample_secondaries(
      pd, type, tkin, m.inv_a23, iz, ia, target_mass, cut, kCosMin, kCosMax, in_dir, rng);

  out.fired = true;
  out.final_t = fs.final_t;
  out.edep = fs.edep;
  // The sampled angle is about the incoming direction, so rotateUz puts it in the lab frame -
  // which is what G4ParticleChangeForGamma::ProposeMomentumDirection receives after
  // `newDirection.rotateUz(direction)`.
  {
    const real_t sint = sqrt(fmax(real_t(0), (real_t(1) - fs.cos_theta)
                                                 * (real_t(1) + fs.cos_theta)));
    const Vec3<real_t> local{sint * cos(fs.phi), sint * sin(fs.phi), fs.cos_theta};
    out.dir = rotate_uz(local, in_dir);
  }
  if (fs.emit_ion) {
    out.emit_ion = true;
    out.ion_z = fs.ion_z;
    out.ion_a = fs.ion_a;
    out.ion_ekin = fs.trec;
    out.ion_dir = fs.ion_dir;
  }
  return out;
}

}  // namespace g4gpu::em
