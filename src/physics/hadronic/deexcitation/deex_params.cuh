// The de-excitation module's configuration, as Geant4 11.1.1 defaults it.
//
// Transcribed from G4DeexPrecoParameters::SetDefaults() and the G4ExcitationHandler
// constructor. These are not tuning knobs of this port: every dispatch decision below - which
// model sees a fragment, which channel set evaporation offers, whether the level density is
// read from data or evaluated from a formula - is made from one of them, so they are pinned
// here and checked against `ref/oracle/deex_params.csv`, which is the installed Geant4's own
// answer rather than this file's.
//
// The four that decide the shape of the whole module:
//
//   fDeexChannelType = fCombined       G4Evaporation::SetCombinedChannel(), i.e.
//                                      G4EvaporationDefaultGEMFactory: 68 channels - photon
//                                      evaporation, competitive fission, the six light
//                                      ejectiles by G4EvaporationChannel, and 60 GEM channels
//                                      for He6..Mg28.
//   fMinExPerNucleounForMF = 200 GeV   G4ExcitationHandler::BreakItUp reaches
//                                      G4StatMF only when E* > 200 GeV * A. It never does, so
//                                      **statistical multifragmentation is unreachable with
//                                      the defaults** and G4StatMF is not ported. That is a
//                                      finding about the configuration, not a gap: the
//                                      handler's own condition is reproduced, and if a caller
//                                      ever raises E*/A above the limit the port refuses by
//                                      name instead of skipping the model.
//   fLD = true                         "use simple level density model":
//                                      G4NuclearLevelData::GetLevelDensity returns A * 0.075
//                                      per MeV and never consults the level manager, and
//                                      GetPairingCorrection returns G4PairingCorrection's
//                                      Cameron-Gilbert answer rather than the inline
//                                      12/sqrt(A) form. Both branches are ported; this flag
//                                      picks which one runs.
//   fCorrelatedGamma = false           no nuclear polarization, so G4PolarizationTransition
//                                      and G4NuclearPolarization are never constructed and
//                                      G4GammaTransition::SampleDirection is isotropic.
//
// And two more that remove work: fStoreAllLevels = false, so G4LevelReader reads the ten
// internal-conversion coefficients of a transition and discards them (no shell sampling
// table is built), and fPrecoDummy = false / fDeexChannelType != fDummy, so the handler is
// active rather than converting E* straight into recoil kinetic energy.
#ifndef G4GPU_DEEX_PARAMS_CUH
#define G4GPU_DEEX_PARAMS_CUH

#include "core/units.cuh"

namespace g4gpu::deex {

namespace u = g4gpu::units;

// ---------------------------------------------------------------------------------------------
// CLHEP constants this module needs that src/core/units.cuh does not carry. Derived the way
// CLHEP derives them, from the values units.cuh already pins, so there is one definition of
// each physical constant in the project and not two. docs/RISK.md V8 is what a second
// hand-typed constant cost.
// ---------------------------------------------------------------------------------------------

/// CLHEP fermi = 1e-15 m. In this unit system (mm = 1) that is 1e-12.
__host__ __device__ inline constexpr double fermi() { return 1e-12; }

/// CLHEP millibarn = 1e-3 barn, mm^2.
__host__ __device__ inline constexpr double millibarn() { return 1e-3 * u::barn<double>(); }

/// CLHEP elm_coupling = e^2/(4 pi eps0) = classic_electron_radius * m_e c^2, MeV mm.
/// 1.4399644 MeV fm, the coefficient of every Coulomb barrier below.
__host__ __device__ inline constexpr double elm_coupling() {
  return u::classic_electron_radius<double>() * u::electron_mass_c2<double>();
}

/// CLHEP pi2 = pi*pi.
__host__ __device__ inline constexpr double pi2() {
  return u::pi<double>() * u::pi<double>();
}

// ---------------------------------------------------------------------------------------------
// G4DeexPrecoParameters::SetDefaults(), 11.1.1.
// ---------------------------------------------------------------------------------------------

/// The enum G4DeexChannelType, in Geant4's order, so a dumped integer means the same thing.
enum DeexChannelType { kEvaporation = 0, kGEM, kCombined, kGEMVI, kDummy };

struct DeexParameters {
  // Level density and radii
  double level_density = 0.075 / u::MeV<double>();   ///< fLevelDensity
  double r0 = 1.5 * fermi();                         ///< fR0
  double transitions_r0 = 0.6 * fermi();             ///< fTransitionsR0

  // Model windows
  double fbu_energy_limit = 20.0 * u::MeV<double>(); ///< fFBUEnergyLimit
  double fermi_energy = 35.0 * u::MeV<double>();     ///< fFermiEnergy
  double preco_low_energy = 0.1 * u::MeV<double>();  ///< fPrecoLowEnergy
  double preco_high_energy = 30 * u::MeV<double>();  ///< fPrecoHighEnergy
  double pheno_factor = 1.0;                         ///< fPhenoFactor

  // Handler
  double min_excitation = 10 * u::eV<double>();      ///< fMinExcitation
  double max_life_time = 1 * u::ns<double>();        ///< fMaxLifeTime

  /// fMinExPerNucleounForMF. 200 GeV per nucleon, i.e. multifragmentation off.
  double min_ex_per_nucleon_for_mf = 200 * u::GeV<double>();

  int min_z_for_preco = 3;                           ///< fMinZForPreco
  int min_a_for_preco = 5;                           ///< fMinAForPreco
  int preco_type = 3;                                ///< fPrecoType
  int deex_type = 3;                                 ///< fDeexType - OPTxs: 3 = Kalbach
  int two_j_max = 10;                                ///< fTwoJMAX
  int verbose = 1;                                   ///< fVerbose

  bool never_go_back = false;                        ///< fNeverGoBack
  bool use_soft_cutoff = false;                      ///< fUseSoftCutoff
  bool use_cem = true;                               ///< fUseCEM
  bool use_gnash = false;                            ///< fUseGNASH
  bool use_hetc = false;                             ///< fUseHETC
  bool use_angular_gen = false;                      ///< fUseAngularGen
  bool preco_dummy = false;                          ///< fPrecoDummy

  bool correlated_gamma = false;                     ///< fCorrelatedGamma
  bool store_ic_level_data = false;                  ///< fStoreAllLevels
  bool internal_conversion = true;                   ///< fInternalConversion
  bool level_density_flag = true;                    ///< fLD
  bool discrete_excitation_flag = true;              ///< fFD
  bool isomer_production = true;                     ///< fIsomerFlag

  DeexChannelType deex_channel_type = kCombined;     ///< fDeexChannelType
};

/// The one instance. A struct rather than constants so that a test can perturb a flag and
/// watch a decision change - which is how the dispatch on `level_density_flag` and
/// `discrete_excitation_flag` is checked at all, since neither is reachable by any UI command
/// this port has.
__host__ __device__ inline const DeexParameters& deex_params() {
  static const DeexParameters p;
  return p;
}

// ---------------------------------------------------------------------------------------------
// G4ExcitationHandler's own constants, from its constructor. These are NOT in
// G4DeexPrecoParameters and no UI command reaches them; they are hard-coded members.
// ---------------------------------------------------------------------------------------------

constexpr int kMaxZForFermiBreakUp = 9;    ///< G4ExcitationHandler::maxZForFermiBreakUp
constexpr int kMaxAForFermiBreakUp = 17;   ///< G4ExcitationHandler::maxAForFermiBreakUp

/// G4Evaporation::minExcitation, set from GetMinExcitation() in InitialiseChannels(); the
/// 0.1*keV in the constructor is overwritten before first use.
__host__ __device__ inline double evaporation_min_excitation() {
  return deex_params().min_excitation;
}

/// G4FermiBreakUpVI::tolerance - 1 MeV, and not the parameter block's min_excitation. Used
/// only to decide whether a channel's cached static probabilities may be reused.
__host__ __device__ inline constexpr double fermi_breakup_tolerance() {
  return 1.0 * u::MeV<double>();
}

/// G4PhotonEvaporation::fTolerance is 20 eV in the constructor and is then overwritten by
/// Initialise() with GetMinExcitation() = 10 eV. The constructor value never decides anything
/// because G4ExcitationHandler::Initialise runs first; recorded because the two differ and the
/// wrong one is the one written down in the class.
__host__ __device__ inline double photon_evaporation_tolerance() {
  return deex_params().min_excitation;
}

}  // namespace g4gpu::deex

#endif
