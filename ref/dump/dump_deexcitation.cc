// The de-excitation module's oracle: what Geant4 11.1.1 answers, so the port can be diffed
// against it rather than against a reading of the source.
//
// Seven files, in the order src/physics/hadronic/deexcitation/ depends on them:
//
//   deex_params.csv     G4DeexPrecoParameters as installed, plus the constants
//                       G4ExcitationHandler and G4FermiBreakUpVI hard-code. Everything below
//                       is dispatched by these, so they come first and are read from the
//                       install rather than from memory.
//   deex_masses.csv     G4NucleiProperties nuclear mass, mass excess, binding energy, atomic
//                       mass and IsInStableTable over a (Z, A) sweep wide enough to reach all
//                       four of its branches - AME2012, the theoretical table, the Z == A /
//                       Z == 0 special cases and Weizsaecker.
//   deex_corrections.csv  pairing and shell corrections and the level-density parameter, over
//                       the same sweep, with the raw table results and their in-window flags
//                       so a wrong window shows up as a flag and not as a value.
//   deex_coulomb.csv    RadiusCB and both Coulomb-barrier families for every ejectile the
//                       default channel set emits, over a (Z_res, A_res) grid and three U.
//   deex_levelmax.csv   G4NuclearLevelData::GetMaxLevelEnergy (a table compiled from
//                       PhotonEvaporation *5.2*) against G4LevelManager::MaxLevelEnergy (read
//                       from the installed PhotonEvaporation *5.7*), for every (Z, A) in the
//                       AMIN/AMAX window. These are two answers to the same question and the
//                       Fermi break-up pool is built from the first one.
//   deex_levels.csv     every level and every gamma transition of a sample of nuclides across
//                       the table, exactly as G4LevelReader parsed them.
//   deex_probs.csv      G4EvaporationChannel emission probabilities, inverse cross sections
//                       and G4PhotonEvaporation's probability, on a (Z, A, E*) grid.
//   deex_breakup.csv    G4ExcitationHandler::BreakItUp run N times under a fixed seed on a
//                       fixed set of fragments - multiplicity by species, energy moments, and
//                       the residual (Z, A) distribution.
//
// Fixed seeds everywhere anything is sampled, as the existing dumps do, so the file is
// reproducible run to run and a diff means a physics change.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

#include "G4AlphaCoulombBarrier.hh"
#include "G4AlphaEvaporationChannel.hh"
#include "G4ChatterjeeCrossSection.hh"
#include "G4CoulombBarrier.hh"
#include "G4DeuteronCoulombBarrier.hh"
#include "G4DeuteronEvaporationChannel.hh"
#include "G4DeexPrecoParameters.hh"
#include "G4Evaporation.hh"
#include "G4EvaporationDefaultGEMFactory.hh"
#include "G4EvaporationChannel.hh"
#include "G4ExcitationHandler.hh"
#include "G4FermiBreakUpVI.hh"
#include "G4FermiChannels.hh"
#include "G4FermiCoulombBarrier.hh"
#include "G4FermiFragment.hh"
#include "G4FermiFragmentsPoolVI.hh"
#include "G4FermiPair.hh"
#include "G4CompetitiveFission.hh"
#include "G4FissionBarrier.hh"
#include "G4FissionLevelDensityParameter.hh"
#include "G4FissionParameters.hh"
#include "G4FissionProbability.hh"
#include "G4Fragment.hh"
#include "G4He3CoulombBarrier.hh"
#include "G4He3EvaporationChannel.hh"
#include "G4KalbachCrossSection.hh"
#include "G4LevelManager.hh"
#include "G4NeutronCoulombBarrier.hh"
#include "G4NeutronEvaporationChannel.hh"
#include "G4NucLevel.hh"
#include "G4NuclearLevelData.hh"
#include "G4NuclearRadii.hh"
#include "G4NucleiProperties.hh"
#include "G4PairingCorrection.hh"
#include "G4PhotonEvaporation.hh"
#include "G4ProtonCoulombBarrier.hh"
#include "G4ProtonEvaporationChannel.hh"
#include "G4ReactionProduct.hh"
#include "G4ReactionProductVector.hh"
#include "G4Pow.hh"
#include "G4VFermiBreakUp.hh"
#include "G4VEvaporation.hh"
#include "G4ShellCorrection.hh"
#include "G4SystemOfUnits.hh"
#include "G4TritonCoulombBarrier.hh"
#include "G4TritonEvaporationChannel.hh"
#include "Randomize.hh"

namespace {

/// The (Z, A) sweep. Wide enough that every branch of G4NucleiProperties is reached: AME2012
/// stops at A = 273 and Z = 110, the theoretical table runs Z = 8..136 and A = 16..339, and
/// everything outside both is a formula. A ceiling of 3Z + 40 covers the neutron-rich edge of
/// both tables for every Z without spending rows on (Z, A) pairs no nuclear model has ever
/// been asked about.
int sweep_amax(int Z) {
  int hi = 3 * Z + 40;
  if (hi > 340) { hi = 340; }
  return hi;
}
int sweep_amin(int Z) { return (Z < 1) ? 1 : Z; }
constexpr int kSweepZMax = 118;

/// A fragment of (Z, A) with excitation E* and momentum p along z. E* is set through the
/// 4-momentum, which is the only way G4Fragment accepts one: CalculateMassAndExcitationEnergy
/// takes E* = lv.mag() - groundStateMass, so the invariant mass has to be M + E*.
G4Fragment make_fragment(int Z, int A, double eexc, double pz) {
  const double m = G4NucleiProperties::GetNuclearMass(A, Z) + eexc;
  const G4LorentzVector lv(0.0, 0.0, pz, std::sqrt(m * m + pz * pz));
  return G4Fragment(A, Z, lv);
}

// ---------------------------------------------------------------------------------------------

void dump_params() {
  G4NuclearLevelData* nd = G4NuclearLevelData::GetInstance();
  const G4DeexPrecoParameters* p = nd->GetParameters();
  FILE* f = std::fopen("deex_params.csv", "w");
  std::fprintf(f, "name,value,unit\n");
  std::fprintf(f, "LevelDensity,%.17g,1/MeV\n", p->GetLevelDensity() * MeV);
  std::fprintf(f, "R0,%.17g,mm\n", p->GetR0());
  std::fprintf(f, "TransitionsR0,%.17g,mm\n", p->GetTransitionsR0());
  std::fprintf(f, "FBUEnergyLimit,%.17g,MeV\n", p->GetFBUEnergyLimit() / MeV);
  std::fprintf(f, "FermiEnergy,%.17g,MeV\n", p->GetFermiEnergy() / MeV);
  std::fprintf(f, "PrecoLowEnergy,%.17g,MeV\n", p->GetPrecoLowEnergy() / MeV);
  std::fprintf(f, "PrecoHighEnergy,%.17g,MeV\n", p->GetPrecoHighEnergy() / MeV);
  std::fprintf(f, "PhenoFactor,%.17g,\n", p->GetPhenoFactor());
  std::fprintf(f, "MinExcitation,%.17g,MeV\n", p->GetMinExcitation() / MeV);
  std::fprintf(f, "MaxLifeTime,%.17g,ns\n", p->GetMaxLifeTime() / ns);
  std::fprintf(f, "MinExPerNucleounForMF,%.17g,MeV\n", p->GetMinExPerNucleounForMF() / MeV);
  std::fprintf(f, "MinZForPreco,%d,\n", p->GetMinZForPreco());
  std::fprintf(f, "MinAForPreco,%d,\n", p->GetMinAForPreco());
  std::fprintf(f, "PrecoModelType,%d,\n", p->GetPrecoModelType());
  std::fprintf(f, "DeexModelType,%d,\n", p->GetDeexModelType());
  std::fprintf(f, "TwoJMAX,%d,\n", p->GetTwoJMAX());
  std::fprintf(f, "NeverGoBack,%d,\n", p->NeverGoBack() ? 1 : 0);
  std::fprintf(f, "UseSoftCutoff,%d,\n", p->UseSoftCutoff() ? 1 : 0);
  std::fprintf(f, "UseCEM,%d,\n", p->UseCEM() ? 1 : 0);
  std::fprintf(f, "UseGNASH,%d,\n", p->UseGNASH() ? 1 : 0);
  std::fprintf(f, "UseHETC,%d,\n", p->UseHETC() ? 1 : 0);
  std::fprintf(f, "UseAngularGen,%d,\n", p->UseAngularGen() ? 1 : 0);
  std::fprintf(f, "PrecoDummy,%d,\n", p->PrecoDummy() ? 1 : 0);
  std::fprintf(f, "CorrelatedGamma,%d,\n", p->CorrelatedGamma() ? 1 : 0);
  std::fprintf(f, "StoreICLevelData,%d,\n", p->StoreICLevelData() ? 1 : 0);
  std::fprintf(f, "InternalConversionFlag,%d,\n", p->GetInternalConversionFlag() ? 1 : 0);
  std::fprintf(f, "LevelDensityFlag,%d,\n", p->GetLevelDensityFlag() ? 1 : 0);
  std::fprintf(f, "DiscreteExcitationFlag,%d,\n", p->GetDiscreteExcitationFlag() ? 1 : 0);
  std::fprintf(f, "IsomerProduction,%d,\n", p->IsomerProduction() ? 1 : 0);
  std::fprintf(f, "DeexChannelsType,%d,\n", static_cast<int>(p->GetDeexChannelsType()));

  // The constants that are NOT in the parameter class and that no UI command reaches. They
  // decide as much as the parameters do: which fragments Fermi break-up claims, and how many
  // channels evaporation offers.
  G4ExcitationHandler handler;
  handler.Initialise();
  std::fprintf(f, "HandlerMaxZForFermiBreakUp,%d,\n", 9);
  std::fprintf(f, "HandlerMaxAForFermiBreakUp,%d,\n", 17);
  G4VEvaporation* evap = handler.GetEvaporation();
  std::fprintf(f, "EvaporationNumberOfChannels,%d,\n",
               static_cast<int>(evap->GetNumberOfChannels()));
  G4VFermiBreakUp* fbu = handler.GetFermiModel();
  // The pool's own energy limit, read back through IsApplicable: C12 at 1 MeV is inside every
  // window, at 1 TeV outside the energy one, and Fe56 outside the (Z, A) one.
  std::fprintf(f, "FermiIsApplicable_C12_1MeV,%d,\n", fbu->IsApplicable(6, 12, 1.0 * MeV) ? 1 : 0);
  std::fprintf(f, "FermiIsApplicable_C12_19MeV,%d,\n",
               fbu->IsApplicable(6, 12, 19.0 * MeV) ? 1 : 0);
  std::fprintf(f, "FermiIsApplicable_C12_21MeV,%d,\n",
               fbu->IsApplicable(6, 12, 21.0 * MeV) ? 1 : 0);
  std::fprintf(f, "FermiIsApplicable_Fe56_10MeV,%d,\n",
               fbu->IsApplicable(26, 56, 10.0 * MeV) ? 1 : 0);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

void dump_masses() {
  // No atomic-mass column: G4NucleiProperties::GetAtomicMass is private in 11.1.1, "hidden to
  // enforce using GetNuclearMass". The port has it because the three public functions are all
  // built on it, but it is unobservable from outside and so cannot be a compared column - it
  // is checked only indirectly, through the mass excess and the nuclear mass that use it.
  FILE* f = std::fopen("deex_masses.csv", "w");
  std::fprintf(f, "Z,A,in_stable_table,nuclear_mass_MeV,mass_excess_MeV,binding_MeV\n");
  for (int Z = 0; Z <= kSweepZMax; ++Z) {
    for (int A = sweep_amin(Z); A <= sweep_amax(Z); ++A) {
      if (A < Z) { continue; }
      std::fprintf(f, "%d,%d,%d,%.17g,%.17g,%.17g\n", Z, A,
                   G4NucleiProperties::IsInStableTable(A, Z) ? 1 : 0,
                   G4NucleiProperties::GetNuclearMass(A, Z) / MeV,
                   G4NucleiProperties::GetMassExcess(A, Z) / MeV,
                   G4NucleiProperties::GetBindingEnergy(A, Z) / MeV);
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

void dump_corrections() {
  G4NuclearLevelData* nd = G4NuclearLevelData::GetInstance();
  const G4PairingCorrection* pc = nd->GetPairingCorrection();
  const G4ShellCorrection* sc = nd->GetShellCorrection();

  // G4CameronTruranHilfShellCorrections and G4CameronShellPlusPairingCorrections are NOT
  // dumped, and cannot be: their tables are private statics read by an accessor that is inline
  // in the header, so an external translation unit inlines the accessor and then needs the
  // symbol - which this Windows Geant4 build does not export. The link fails with four
  // unresolved externals. Neither table is read by anything in the default configuration
  // (G4ShellCorrection uses Cook then Cameron-Gilbert; G4PairingCorrection uses
  // Cameron-Gilbert then a formula), so the port carries them transcribed from the source and
  // marked as not oracle-comparable rather than pretending they were checked.
  FILE* f = std::fopen("deex_corrections.csv", "w");
  std::fprintf(f, "Z,A,pairing_MeV,fission_pairing_MeV,shell_MeV,level_density_MeV,"
                  "ld_pairing_MeV\n");
  for (int Z = 0; Z <= kSweepZMax; ++Z) {
    for (int A = sweep_amin(Z); A <= sweep_amax(Z); ++A) {
      if (A < Z) { continue; }
      std::fprintf(f, "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n", Z, A,
                   pc->GetPairingCorrection(A, Z) / MeV,
                   pc->GetFissionPairingCorrection(A, Z) / MeV,
                   sc->GetShellCorrection(A, Z) / MeV,
                   nd->GetLevelDensity(Z, A, 0.0) * MeV,
                   nd->GetPairingCorrection(Z, A) / MeV);
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

void dump_coulomb() {
  // The six ejectiles G4EvaporationDefaultGEMFactory gives a G4EvaporationChannel.
  const int ejZ[6] = {0, 1, 1, 1, 2, 2};
  const int ejA[6] = {1, 1, 2, 3, 3, 4};

  FILE* f = std::fopen("deex_coulomb.csv", "w");
  std::fprintf(f, "ejZ,ejA,Zres,Ares,U_MeV,radius_cb_res_fm,radius_cb_ej_fm,barrier_MeV,"
                  "fermi_barrier_MeV,penetration\n");
  // Residuals from the lightest thing that can hold a nucleon to uranium, and a step that
  // keeps every one of RadiusCB's branches in: Z <= 4 is the explicit-radius table, above it
  // the r0 array, and Z > 92 the clamp.
  const int zs[] = {1, 2, 3, 4, 5, 6, 8, 13, 20, 26, 40, 50, 74, 82, 92, 96, 100};
  const double us[] = {0.0, 10.0, 100.0};
  for (int i = 0; i < 6; ++i) {
    G4CoulombBarrier cb(ejA[i], ejZ[i]);
    G4FermiCoulombBarrier fcb(ejA[i], ejZ[i]);
    for (int Zres : zs) {
      // Two mass numbers per Z: near stability and neutron-rich, so the A dependence is
      // exercised and not just the Z one.
      const int as[2] = {2 * Zres + (Zres > 20 ? Zres / 5 : 0), 3 * Zres + 4};
      for (int Ares : as) {
        if (Ares < Zres + 1) { continue; }
        for (double U : us) {
          std::fprintf(f, "%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", ejZ[i], ejA[i],
                       Zres, Ares, U,
                       G4NuclearRadii::RadiusCB(Zres, Ares) / fermi,
                       G4NuclearRadii::RadiusCB(ejZ[i], ejA[i]) / fermi,
                       cb.GetCoulombBarrier(Ares, Zres, U * MeV) / MeV,
                       fcb.GetCoulombBarrier(Ares, Zres, U * MeV) / MeV,
                       cb.BarrierPenetrationFactor(Zres));
        }
      }
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

void dump_levelmax() {
  G4NuclearLevelData* nd = G4NuclearLevelData::GetInstance();
  FILE* f = std::fopen("deex_levelmax.csv", "w");
  std::fprintf(f, "Z,A,compiled_max_MeV,read_max_MeV,n_levels,lifetime0_ns\n");
  for (int Z = 1; Z < 118; ++Z) {
    const int amin = nd->GetMinA(Z);
    const int amax = nd->GetMaxA(Z);
    if (amax <= 0) { continue; }
    for (int A = amin; A <= amax; ++A) {
      const G4LevelManager* man = nd->GetLevelManager(Z, A);
      std::fprintf(f, "%d,%d,%.17g,%.17g,%d,%.17g\n", Z, A,
                   nd->GetMaxLevelEnergy(Z, A) / MeV,
                   man ? man->MaxLevelEnergy() / MeV : -1.0,
                   man ? static_cast<int>(man->NumberOfTransitions()) + 1 : 0,
                   man ? man->LifeTime(0) / ns : -2.0);
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

void dump_levels() {
  G4NuclearLevelData* nd = G4NuclearLevelData::GetInstance();
  // A spread across the table rather than the five elements a water phantom has: the reader's
  // hard cases are a nuclide with no transitions, one with a floating level, one with a
  // hundred-plus levels, and the light ones whose file is one line.
  const int zas[][2] = {
    {1, 1},   {1, 2},   {1, 3},   {2, 4},   {2, 6},   {3, 6},   {3, 7},   {4, 7},
    {4, 9},   {5, 10},  {5, 11},  {6, 12},  {6, 13},  {6, 14},  {7, 14},  {7, 15},
    {8, 16},  {8, 17},  {8, 18},  {10, 20}, {12, 24}, {13, 27}, {14, 28}, {20, 40},
    {22, 48}, {26, 54}, {26, 56}, {26, 57}, {28, 58}, {29, 63}, {36, 84}, {40, 90},
    {47, 107},{50, 120},{54, 132},{56, 138},{64, 156},{74, 184},{79, 197},{82, 206},
    {82, 207},{82, 208},{83, 209},{90, 232},{92, 235},{92, 238},
    // Two nuclides that are here for a reason and not for coverage.
    // z89.a219 is the ONLY file in PhotonEvaporation5.7 that trips G4LevelReader's
    // broken-transition repair - it has a transition from level 24 to level 24, which the
    // reader redirects to the ground state with a warning. Without this row the port's copy of
    // that repair is never exercised: removing it changed nothing in a 46-nuclide sample.
    // z18.a38 has exactly 632 levels, which is exactly G4LevelReader::fLevelMax, so it is the
    // file that decides whether the level cap is a limit or a fit.
    {89, 219}, {18, 38},
  };
  FILE* f = std::fopen("deex_levels.csv", "w");
  std::fprintf(f, "Z,A,level,energy_MeV,lifetime_ns,spin2,parity,floating,ntrans,"
                  "trans,final_index,type,gamma_prob,cum_prob,mp_ratio\n");
  for (const auto& za : zas) {
    const int Z = za[0], A = za[1];
    const G4LevelManager* man = nd->GetLevelManager(Z, A);
    if (!man) {
      std::fprintf(f, "%d,%d,-1,0,0,0,0,0,0,-1,0,0,0,0,0\n", Z, A);
      continue;
    }
    const std::size_t nlev = man->NumberOfTransitions();
    for (std::size_t i = 0; i <= nlev; ++i) {
      const G4NucLevel* lev = man->GetLevel(i);
      const std::size_t nt = lev ? lev->NumberOfTransitions() : 0;
      if (nt == 0) {
        std::fprintf(f, "%d,%d,%d,%.17g,%.17g,%d,%d,%d,%d,-1,0,0,0,0,0\n", Z, A,
                     static_cast<int>(i), man->LevelEnergy(i) / MeV, man->LifeTime(i) / ns,
                     man->SpinTwo(i), man->Parity(i), man->FloatingLevel(i), 0);
        continue;
      }
      for (std::size_t j = 0; j < nt; ++j) {
        std::fprintf(f, "%d,%d,%d,%.17g,%.17g,%d,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g\n", Z, A,
                     static_cast<int>(i), man->LevelEnergy(i) / MeV, man->LifeTime(i) / ns,
                     man->SpinTwo(i), man->Parity(i), man->FloatingLevel(i),
                     static_cast<int>(nt), static_cast<int>(j),
                     static_cast<int>(lev->FinalExcitationIndex(j)), lev->TransitionType(j),
                     lev->GammaProbability(j), lev->GammaCumProbability(j),
                     lev->MultipolarityRatio(j));
      }
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

void dump_probs() {
  // The channel objects, in the factory's order. Constructed directly rather than pulled out
  // of G4Evaporation because GetEmissionProbability is what a channel exposes and the vector
  // inside G4Evaporation is private.
  std::vector<G4EvaporationChannel*> ch;
  ch.push_back(new G4NeutronEvaporationChannel());
  ch.push_back(new G4ProtonEvaporationChannel());
  ch.push_back(new G4DeuteronEvaporationChannel());
  ch.push_back(new G4TritonEvaporationChannel());
  ch.push_back(new G4He3EvaporationChannel());
  ch.push_back(new G4AlphaEvaporationChannel());
  for (auto* c : ch) { c->Initialise(); }
  G4PhotonEvaporation photon;
  photon.Initialise();

  const int zas[][2] = {
    {3, 7},   {4, 9},   {6, 12},  {8, 16},  {10, 20}, {13, 27}, {20, 40},
    {26, 56}, {26, 60}, {40, 90}, {50, 120},{74, 184},{82, 208},{92, 238},
  };
  const double eexc[] = {1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0};

  // The two derived columns are written only when the emission probability came out positive,
  // and NaN otherwise. G4VEvaporationChannel calls ComputeInverseXSection and ComputeProbability
  // "methods for unit tests", and they are exactly that: they read `resA13`, `a0`, `freeU` and
  // `delta1`, all of which G4EvaporationProbability::TotalProbability sets. When
  // GetEmissionProbability returns early - a closed channel - TotalProbability never runs and
  // those members still hold the previous fragment's values, or the constructor's zeros. A
  // zero `resA13` makes Kalbach's `lambda = p3/resA13 + p4` infinite. So the columns are
  // dumped where they mean something and refused where they do not, rather than dumping an
  // infinity for the port to reproduce.
  //
  // The cross section is NOT divided by millibarn: G4EvaporationProbability::CrossSection
  // returns a bare number already in millibarn, and the unit enters one level up, in
  // `pcoeff = fGamma * m * millibarn / (pi hbarc)^2`.
  FILE* f = std::fopen("deex_probs.csv", "w");
  std::fprintf(f, "Z,A,Eexc_MeV,channel,ejZ,ejA,emission_prob,inv_xs_mb_at_5MeV,"
                  "prob_at_5MeV\n");
  for (const auto& za : zas) {
    for (double e : eexc) {
      G4Fragment frag = make_fragment(za[0], za[1], e * MeV, 0.0);
      for (std::size_t i = 0; i < ch.size(); ++i) {
        G4Fragment copy(frag);
        const double p = ch[i]->GetEmissionProbability(&copy);
        double xs = std::nan(""), pk = std::nan("");
        if (p > 0.0) {
          G4Fragment c2(frag);
          xs = ch[i]->ComputeInverseXSection(&c2, 5.0 * MeV);
          G4Fragment c3(frag);
          pk = ch[i]->ComputeProbability(&c3, 5.0 * MeV);
        }
        std::fprintf(f, "%d,%d,%.17g,%d,%d,%d,%.17g,%.17g,%.17g\n", za[0], za[1], e,
                     static_cast<int>(i), ch[i]->GetZ(), ch[i]->GetA(), p, xs, pk);
      }
      G4Fragment cp(frag);
      std::fprintf(f, "%d,%d,%.17g,%d,%d,%d,%.17g,%.17g,%.17g\n", za[0], za[1], e, 100, 0, 0,
                   photon.GetEmissionProbability(&cp), 0.0, 0.0);
    }
  }
  std::fclose(f);

  // The two inverse-cross-section parameterisations, straight out of their static functions,
  // on the same energy grid a channel would sample. Dumped separately from the channel because
  // the channel folds in a Coulomb barrier and a (1 - elim/K) factor, and a disagreement in
  // the composite is otherwise ambiguous between the two.
  FILE* g = std::fopen("deex_invxs.csv", "w");
  std::fprintf(g, "idx,ejZ,ejA,resA,K_MeV,cb_MeV,mu,kalbach_mb,chatterjee_mb\n");
  const int idxZ[6] = {0, 1, 1, 1, 2, 2};
  const int idxA[6] = {1, 1, 2, 3, 3, 4};
  const int resAs[] = {6, 12, 27, 56, 90, 120, 184, 208, 238};
  const double ks[] = {0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0};
  for (int i = 0; i < 6; ++i) {
    // G4EvaporationProbability::CrossSection's index: 0 for a neutron, theA for Z = 1,
    // theA + 1 above that. Reproduced here so the dump indexes paramK the same way.
    const int index = (idxZ[i] == 0) ? 0 : (idxZ[i] == 1 ? idxA[i] : idxA[i] + 1);
    G4CoulombBarrier cbar(idxA[i], idxZ[i]);
    for (int resA : resAs) {
      const int resZ = resA / 2;
      const double resA13 = G4Pow::GetInstance()->Z13(resA);
      const double mu = (index > 0) ? G4KalbachCrossSection::ComputePowerParameter(resA, index)
                                    : 0.0;
      const double cb = cbar.GetCoulombBarrier(resA, resZ, 0.0);
      for (double K : ks) {
        std::fprintf(g, "%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n", index, idxZ[i], idxA[i],
                     resA, K, cb / MeV, mu,
                     G4KalbachCrossSection::ComputeCrossSection(
                         K * MeV, 0.6 * cb, resA13, mu, index, idxZ[i], idxA[i], resA) ,
                     G4ChatterjeeCrossSection::ComputeCrossSection(
                         K * MeV, cb, resA13, mu, index, idxZ[i], resA));
      }
    }
  }
  std::fclose(g);

  for (auto* c : ch) { delete c; }
}

// ---------------------------------------------------------------------------------------------

/// The 60 GEM channels and the fission channel, through the factory that builds them.
///
/// Built from G4EvaporationDefaultGEMFactory rather than by naming 60 classes, for the reason
/// tools/extract_gem_tables.pl reads the same file: the ORDER is observable - G4Evaporation
/// indexes its probability array by it and stops the loop early on it - so the order has to
/// come from the factory in the dump and in the port, not from two lists typed twice.
///
/// The factory's layout is fixed: 0 photon evaporation, 1 competitive fission, 2..7 the six
/// G4EvaporationChannel light ejectiles, 8..67 the sixty G4GEMChannel nuclei. The index is
/// written into the file so a port that disagrees about it fails on the index and not on a
/// probability.
void dump_gem() {
  G4PhotonEvaporation* photon = new G4PhotonEvaporation();
  G4EvaporationDefaultGEMFactory factory(photon);
  std::vector<G4VEvaporationChannel*>* ch = factory.GetChannel();
  for (auto* c : *ch) { c->Initialise(); }
  if (ch->size() != 68) {
    std::printf("dump_deexcitation: factory gave %d channels, expected 68\n",
                static_cast<int>(ch->size()));
  }

  const int zas[][2] = {
    {6, 12},  {8, 16},  {10, 20}, {13, 27}, {20, 40}, {26, 56},
    {26, 60}, {40, 90}, {50, 120},{74, 184},{82, 208},{92, 238},
  };
  const double eexc[] = {5.0, 10.0, 20.0, 50.0, 100.0, 200.0};

  FILE* f = std::fopen("deex_gem.csv", "w");
  std::fprintf(f, "Z,A,Eexc_MeV,channel,emission_prob\n");
  for (const auto& za : zas) {
    for (double e : eexc) {
      G4Fragment frag = make_fragment(za[0], za[1], e * MeV, 0.0);
      // Channel 1 is fission and 8..67 are GEM. The six evaporation channels are in
      // deex_probs.csv already and photon evaporation cannot be asked twice for the same
      // fragment without recomputing its cumulative array, so neither is repeated here.
      for (std::size_t i = 1; i < ch->size(); ++i) {
        if (i >= 2 && i <= 7) { continue; }
        G4Fragment copy(frag);
        std::fprintf(f, "%d,%d,%.17g,%d,%.17g\n", za[0], za[1], e, static_cast<int>(i),
                     (*ch)[i]->GetEmissionProbability(&copy));
      }
    }
  }
  std::fclose(f);

  for (auto* c : *ch) { delete c; }
  delete ch;
}

// ---------------------------------------------------------------------------------------------

/// The fission channel's three deterministic layers, separately.
///
/// The barrier is the one that matters beyond fission itself:
/// G4FissionBarrier::BarashenkovFissionBarrier is the ONLY consumer of
/// G4CameronShellPlusPairingCorrections in the default configuration, and that table cannot be
/// read directly on this build - its accessor is inline in the header and the symbol is not
/// exported, which is why dump_corrections() has no column for it. The barrier IS public, so a
/// wide (Z, A) sweep of it checks the table through the only door there is.
void dump_fission() {
  G4FissionBarrier barrier;
  G4FissionProbability prob;
  G4FissionLevelDensityParameter fldp;
  G4CompetitiveFission chan;
  chan.Initialise();
  G4PairingCorrection* pcorr = G4NuclearLevelData::GetInstance()->GetPairingCorrection();

  FILE* f = std::fopen("deex_fission_barrier.csv", "w");
  std::fprintf(f, "Z,A,U_MeV,barrier_MeV,fission_ldp_perMeV\n");
  for (int Z = 17; Z <= 100; ++Z) {
    const int amin = (2 * Z - 20 > 65) ? 2 * Z - 20 : 65;
    const int amax = (2.8 * Z < 260) ? static_cast<int>(2.8 * Z) : 260;
    for (int A = amin; A <= amax; ++A) {
      for (double U : {0.0, 20.0}) {
        std::fprintf(f, "%d,%d,%.17g,%.17g,%.17g\n", Z, A, U,
                     barrier.FissionBarrier(A, Z, U * MeV) / MeV,
                     fldp.LevelDensityParameter(A, Z, U * MeV) * MeV);
      }
    }
  }
  std::fclose(f);

  // The probability, the channel's own probability, and the five mass-distribution parameters
  // G4FissionParameters derives - which is where every branch on Z (>= 90, == 89, >= 82,
  // below) and the A < 227 boost live.
  FILE* g = std::fopen("deex_fission_prob.csv", "w");
  std::fprintf(g, "Z,A,Eexc_MeV,chan_prob,fiss_prob_at_maxke,barrier_MeV,maxke_MeV,"
                  "As,Sigma1,Sigma2,SigmaS,w\n");
  const int zas[][2] = {
    {17, 37},  {20, 40},  {26, 56},  {40, 90},  {50, 120}, {62, 152},
    {74, 184}, {80, 200}, {82, 208}, {89, 227}, {90, 232}, {92, 235},
    {92, 238}, {94, 239}, {98, 252},
  };
  for (const auto& za : zas) {
    for (double e : {1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0}) {
      G4Fragment frag = make_fragment(za[0], za[1], e * MeV, 0.0);
      G4Fragment copy(frag);
      const double cp = chan.GetEmissionProbability(&copy);
      const double ex = e * MeV - pcorr->GetFissionPairingCorrection(za[1], za[0]);
      double bf = 0.0, mk = 0.0, fp = 0.0;
      G4FissionParameters par;
      if (ex > 0.0 && za[1] >= 65 && za[0] > 16) {
        bf = barrier.FissionBarrier(za[1], za[0], ex);
        mk = ex - bf;
        fp = prob.EmissionProbability(frag, mk);
        par.DefineParameters(za[1], za[0], ex, bf);
      }
      std::fprintf(g, "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                   za[0], za[1], e, cp, fp, bf / MeV, mk / MeV, par.GetAs(), par.GetSigma1(),
                   par.GetSigma2(), par.GetSigmaS(), par.GetW());
    }
  }
  std::fclose(g);
}

// ---------------------------------------------------------------------------------------------

/// The Fermi break-up fragment pool, as a structure rather than as a sampled outcome.
///
/// G4FermiFragmentsPoolVI is publicly constructible and ClosestChannels / HasChannels /
/// IsPhysical are public, so the pool that G4FermiBreakUpVI::SampleDecay walks can be dumped
/// exactly: for every (Z, A) inside the maxZ = 9 / maxA = 17 window and every excitation on a
/// grid, which channel set ClosestChannels selects, what its own excitation and mass are, and
/// every pair in it with its cumulative probability.
///
/// This is the deterministic half of Fermi break-up. The sampled half is only reachable
/// through BreakItUp and is in deex_breakup.csv; without this file a disagreement there could
/// be the pool, the selection, the probability or the kinematics, and there would be no way to
/// tell which.
void dump_fermi() {
  G4FermiFragmentsPoolVI pool;
  G4FermiBreakUpVI fbu;
  fbu.Initialise();

  FILE* f = std::fopen("deex_fermi_pool.csv", "w");
  std::fprintf(f, "Z,A,Eexc_MeV,applicable,has_channels,is_physical,nch,ch_exc_MeV,"
                  "ch_mass_MeV,pair,Z1,A1,exc1_MeV,Z2,A2,exc2_MeV,cumprob\n");
  std::fprintf(stdout, "Fermi pool: maxZ=%d maxA=%d Elim(MeV)=%.17g tol(MeV)=%.17g\n",
               pool.GetMaxZ(), pool.GetMaxA(), pool.GetEnergyLimit() / MeV,
               pool.GetTolerance() / MeV);

  for (int Z = 0; Z < pool.GetMaxZ(); ++Z) {
    for (int A = 1; A < pool.GetMaxA(); ++A) {
      if (A < Z) { continue; }
      const double gmass = G4NucleiProperties::GetNuclearMass(A, Z);
      for (double e : {0.0, 0.001, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0}) {
        const double exc = e * MeV;
        const G4FermiChannels* c = pool.ClosestChannels(Z, A, gmass + exc);
        const int app = fbu.IsApplicable(Z, A, exc) ? 1 : 0;
        const int hc = pool.HasChannels(Z, A, exc) ? 1 : 0;
        const int ip = pool.IsPhysical(Z, A) ? 1 : 0;
        if (c == nullptr) {
          std::fprintf(f, "%d,%d,%.17g,%d,%d,%d,-1,0,0,-1,0,0,0,0,0,0,0\n", Z, A, e, app, hc,
                       ip);
          continue;
        }
        const std::size_t nch = c->GetNumberOfChannels();
        if (nch == 0) {
          std::fprintf(f, "%d,%d,%.17g,%d,%d,%d,0,%.17g,%.17g,-1,0,0,0,0,0,0,0\n", Z, A, e,
                       app, hc, ip, c->GetExcitation() / MeV, c->GetMass() / MeV);
          continue;
        }
        // GetProbabilities() is non-const, and the pool leaves cum_prob[i] = 1.0 for a
        // single-channel set (AddChannel pushes 1.0 and the normalisation loop skips
        // nch == 1). Dumped as it stands so the port reproduces that too.
        std::vector<G4double>& cp =
            const_cast<G4FermiChannels*>(c)->GetProbabilities();
        for (std::size_t k = 0; k < nch; ++k) {
          const G4FermiPair* p = (c->GetChannels())[k];
          const G4FermiFragment* f1 = p->GetFragment1();
          const G4FermiFragment* f2 = p->GetFragment2();
          std::fprintf(f,
                       "%d,%d,%.17g,%d,%d,%d,%d,%.17g,%.17g,%d,%d,%d,%.17g,%d,%d,%.17g,%.17g\n",
                       Z, A, e, app, hc, ip, static_cast<int>(nch), c->GetExcitation() / MeV,
                       c->GetMass() / MeV, static_cast<int>(k), f1->GetZ(), f1->GetA(),
                       f1->GetExcitationEnergy() / MeV, f2->GetZ(), f2->GetA(),
                       f2->GetExcitationEnergy() / MeV, cp[k]);
        }
      }
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

/// One BreakItUp campaign: N calls on the same fragment under one seed, reduced to the
/// quantities a statistical comparison can actually be made on.
struct Campaign {
  int Z, A;
  double eexc;   ///< MeV
  double pz;     ///< MeV/c along z
};

void dump_breakup() {
  G4ExcitationHandler handler;
  handler.Initialise();

  const Campaign cs[] = {
    // light - inside the Fermi break-up window for its whole excitation range
    {6, 12, 5.0, 0.0},    {6, 12, 20.0, 0.0},   {6, 12, 50.0, 0.0},
    // medium
    {26, 56, 1.0, 0.0},   {26, 56, 5.0, 0.0},   {26, 56, 20.0, 0.0},
    {26, 56, 50.0, 0.0},  {26, 56, 100.0, 0.0}, {26, 56, 200.0, 0.0},
    {26, 56, 50.0, 500.0},
    // heavy - the fission-competition region
    {82, 208, 1.0, 0.0},  {82, 208, 10.0, 0.0}, {82, 208, 50.0, 0.0},
    {82, 208, 200.0, 0.0},
    {13, 27, 30.0, 0.0},  {20, 40, 80.0, 0.0},  {92, 238, 30.0, 0.0},
  };
  const int kN = 20000;

  FILE* f = std::fopen("deex_breakup.csv", "w");
  std::fprintf(f, "Z,A,Eexc_MeV,pz_MeV,N,pdg,count,mean_ekin_MeV,mean_ekin2_MeV2\n");
  FILE* g = std::fopen("deex_breakup_residual.csv", "w");
  std::fprintf(g, "Z,A,Eexc_MeV,pz_MeV,N,resZ,resA,count\n");

  for (const Campaign& c : cs) {
    CLHEP::HepRandom::setTheSeed(20260910 + c.Z * 1000 + c.A * 7 + int(c.eexc));
    std::map<int, long long> count;
    std::map<int, double> sum_e, sum_e2;
    std::map<int, long long> residual;   // key = 1000*Z + A of the heaviest product
    for (int n = 0; n < kN; ++n) {
      G4Fragment frag = make_fragment(c.Z, c.A, c.eexc * MeV, c.pz * MeV);
      G4ReactionProductVector* out = handler.BreakItUp(frag);
      if (!out) { continue; }
      int heaviestA = -1, hz = 0, ha = 0;
      for (G4ReactionProduct* rp : *out) {
        const G4ParticleDefinition* d = rp->GetDefinition();
        const int pdg = d->GetPDGEncoding();
        const double ekin = rp->GetKineticEnergy() / MeV;
        ++count[pdg];
        sum_e[pdg] += ekin;
        sum_e2[pdg] += ekin * ekin;
        // Nuclear PDG codes are 10LZZZAAAI; light ions and nucleons are their own codes.
        int z = 0, a = 0;
        if (pdg > 1000000000) {
          a = (pdg / 10) % 1000;
          z = (pdg / 10000) % 1000;
        } else if (pdg == 2112) { a = 1; z = 0; }
        else if (pdg == 2212) { a = 1; z = 1; }
        if (a > heaviestA) { heaviestA = a; hz = z; ha = a; }
        delete rp;
      }
      delete out;
      ++residual[1000 * hz + ha];
    }
    for (const auto& kv : count) {
      const long long n = kv.second;
      std::fprintf(f, "%d,%d,%.17g,%.17g,%d,%d,%lld,%.17g,%.17g\n", c.Z, c.A, c.eexc, c.pz, kN,
                   kv.first, n, sum_e[kv.first] / double(n), sum_e2[kv.first] / double(n));
    }
    for (const auto& kv : residual) {
      std::fprintf(g, "%d,%d,%.17g,%.17g,%d,%d,%d,%lld\n", c.Z, c.A, c.eexc, c.pz, kN,
                   kv.first / 1000, kv.first % 1000, kv.second);
    }
  }
  std::fclose(f);
  std::fclose(g);
}

// ---------------------------------------------------------------------------------------------

void dump_deexcitation(const DumpContext&) {
  dump_params();
  dump_masses();
  dump_corrections();
  dump_coulomb();
  dump_levelmax();
  dump_levels();
  dump_probs();
  dump_gem();
  dump_fission();
  dump_fermi();
  dump_breakup();
}

}  // namespace

G4GPU_REGISTER_DUMP("deexcitation",
                    "deex_params.csv deex_masses.csv deex_corrections.csv deex_coulomb.csv "
                    "deex_levelmax.csv deex_levels.csv deex_probs.csv deex_invxs.csv "
                    "deex_gem.csv deex_fission_barrier.csv deex_fission_prob.csv "
                    "deex_fermi_pool.csv "
                    "deex_breakup.csv deex_breakup_residual.csv",
                    dump_deexcitation);
