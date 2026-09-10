// Oracle for package P5: the hadronic process framework and the elastic final states.
//
// Registered through dump_registry.hh, so nothing shared is edited. Writes:
//
//   elastic_framework.csv    the framework's defaults: the proton production threshold that
//                            becomes the recoil cut, G4HadronicInteraction's fatal energy-check
//                            levels, and the energy windows QBBC's elastic models are registered
//                            with.
//   elastic_overlap.csv      G4EnergyRangeManager::GetHadronicInteraction counted over N draws
//                            at energies inside a BIC/BERT-shaped and a BERT/FTFP-shaped overlap.
//   elastic_zanda.csv        G4CrossSectionDataStore::ComputeCrossSection and SampleZandA: the
//                            per-element partial cross sections and the counted selection
//                            frequencies in a three-element material.
//   elastic_nuclear_mass.csv G4NucleiProperties::GetNuclearMass(A,Z) for every (Z,A) the rest of
//                            these tables uses. Package P3 owns that class; the elastic
//                            kinematics need the number and take it as a functor, so this is the
//                            table the port's tests feed it.
//   elastic_chips_xs.csv     G4Chips{Proton,Neutron}ElasticXS: GetChipsCrossSection and GetHMaxT
//                            (= lastTM/2) on a (Z, N, p) grid, and GetExchangeT sampled under the
//                            prescribed uniform cycle at eight phases per point. GetSlope, which
//                            would have exposed theB1 directly, is private in both classes.
//   elastic_sample_t.csv     DETERMINISTIC sampling. Every model's ApplyYourself is run under a
//                            prescribed sequence of "uniforms", so the answer is a pure function
//                            of the inputs and can be compared to the last digits. Records the
//                            number of uniforms consumed, which is itself a check on the control
//                            flow.
//   elastic_moments.csv      STATISTICAL sampling. The same models with the real engine at a
//                            fixed seed, N samples per point: the moments of -t and of
//                            cos(theta_cm) and a 20-bin histogram in t/tmax.
//   elastic_he_nist_a.csv    G4lrint(GetAtomicMassAmu(Z)) for Z = 1..92 - the A that
//                            G4ElasticHadrNucleusHE builds each element's table at, which is not
//                            the isotope A.
//
// ---------------------------------------------------------------------------------------------
// Why a prescribed random sequence, and not only a fixed seed.
//
// A fixed seed makes Geant4 reproducible but not comparable: the port cannot consume
// HepJamesRandom's stream, so a seeded comparison can only ever be statistical, and a
// statistical comparison of a sampler cannot distinguish "the distribution is right" from "the
// distribution is right and the coefficients are 1e-6 off". Feeding a KNOWN sequence of uniforms
// through CLHEP's engine interface makes the sampler a deterministic function, and then the
// comparison is exact - the same discipline the deterministic oracles use everywhere else in
// this project. The statistical dump is kept as well, because it exercises thousands of draws
// and every branch weight rather than eight.
//
// The sequence is a fixed eight-value cycle with a per-row phase, so the port reproduces it from
// the row's own fields with no data transfer, and the number of draws consumed is recorded so a
// port that took a different branch is caught rather than silently reading a different uniform.

#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "G4Alpha.hh"
#include "G4ChipsElasticModel.hh"
#include "G4ChipsNeutronElasticXS.hh"
#include "G4ChipsProtonElasticXS.hh"
#include "G4CrossSectionDataSetRegistry.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4DynamicParticle.hh"
#include "G4ElasticHadrNucleusHE.hh"
#include "G4Element.hh"
#include "G4EnergyRangeManager.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadronElastic.hh"
#include "G4HadronicInteraction.hh"
#include "G4HadronicParameters.hh"
#include "G4IonTable.hh"
#include "G4Isotope.hh"
#include "G4Material.hh"
#include "G4Neutron.hh"
#include "G4NistManager.hh"
#include "G4NuclNuclDiffuseElastic.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleus.hh"
#include "G4ParticleTable.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandomEngine.h"

namespace {

// ---------------------------------------------------------------------------------------------
// A CLHEP engine that returns a prescribed cycle and counts the draws.
// ---------------------------------------------------------------------------------------------

/// The eight uniforms the deterministic dump cycles through. Chosen to be spread over (0,1) and
/// to avoid the exact endpoints (a 0 would put log(1-0)=0 into every truncated exponential and a
/// 1 would put log(0)); nothing else about them matters, only that the port uses the same eight
/// in the same order.
const double kSeq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};

class CycleEngine : public CLHEP::HepRandomEngine {
 public:
  void reset(int phase) { phase_ = phase; n_ = 0; }
  int draws() const { return n_; }

  double flat() override {
    const double v = kSeq[(n_ + phase_) % 8];
    ++n_;
    return v;
  }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "CycleEngine"; }

 private:
  int phase_ = 0;
  int n_ = 0;
};

// ---------------------------------------------------------------------------------------------
// The grid every sampling table is written on.
// ---------------------------------------------------------------------------------------------

struct TargetSpec { int z; int a; const char* name; };

/// The targets the plan names: H, C, O, Al, Fe, Pb. A is the most abundant isotope of each, which
/// is what SampleZandA would usually return, plus a second isotope for C and Pb so that the
/// isotope dependence of the kinematics is exercised.
const TargetSpec kTargets[] = {
    {1, 1, "H1"},   {6, 12, "C12"},  {6, 13, "C13"}, {8, 16, "O16"},
    {13, 27, "Al27"}, {26, 56, "Fe56"}, {82, 207, "Pb207"}, {82, 208, "Pb208"},
};
constexpr int kNTargets = int(sizeof(kTargets) / sizeof(kTargets[0]));

/// Kinetic energies, MeV. Spread over each model's range: below G4ElasticHadrNucleusHE's 400 MeV
/// hand-over, around it, through the Chips tabulation, and into the multi-GeV region.
const double kEnergies[] = {1.0, 10.0, 100.0, 200.0, 399.0, 401.0, 800.0,
                            1500.0, 3000.0, 6000.0, 20000.0};
constexpr int kNEnergies = int(sizeof(kEnergies) / sizeof(kEnergies[0]));

/// One model's result, reduced to the numbers the port must reproduce.
struct ApplyOut {
  double efinal = 0, edep = 0, erec = 0;
  double dirx = 0, diry = 0, dirz = 1;
  double recx = 0, recy = 0, recz = 1;
  int nsec = 0, recoil_pdg = 0;
  int draws = 0, t_draws = 0;
  double tmax = 0, t = 0, cost_cms = 1;
  double t_from_erec = 0;
};

/// Run one ApplyYourself and read everything out of the final state, then sample -t again on its
/// own so that the model's OWN momentum transfer is on record.
///
/// `pLocalTmax = 4*pcms^2` is recomputed here with G4HadronElastic's own expression rather than
/// read from the model (it is protected).
///
/// The `t` column is the value `SampleInvariantT` returns, obtained by calling that virtual
/// directly (it is public on G4HadronicInteraction) with the engine wound back to the same phase.
/// It has to be, and the first version of this dump got it wrong: it recorded
/// `-t = 2*M*T_rec`, which holds identically for elastic scattering off a target at rest but
/// which Geant4 evaluates as `T_rec = (lv - nlv1).e() - M` - a difference of two ~2e5 MeV
/// energies that can be as small as 3e-10 MeV. At that point the reconstruction has one
/// significant digit, and the `t` column carried an error of up to 8% that looked exactly like a
/// port bug. `t_from_erec` keeps the reconstructed value beside the real one so that the
/// cancellation is visible in the file rather than only in this comment.
///
/// Calling SampleInvariantT a second time is safe for all four models: `pLocalTmax` was set by
/// the ApplyYourself just above with the same (particle, plab, Z, A), and the samplers hold only
/// memoised tables, so the second call re-reads what the first one filled and consumes the same
/// uniforms.
ApplyOut run_apply(G4HadronElastic* model, const G4ParticleDefinition* part, double ekin,
                   int z, int a, CycleEngine* eng, int phase) {
  ApplyOut o;
  G4DynamicParticle dp(part, G4ThreeVector(0, 0, 1), ekin);
  G4HadProjectile proj(dp);
  G4Nucleus nucleus(a, z);
  model->SetRecoilEnergyThreshold(0.0);

  const double m1 = part->GetPDGMass();
  const double plab = std::sqrt(ekin * (ekin + 2.0 * m1));
  const double mass2 = G4NucleiProperties::GetNuclearMass(a, z);
  const double e1 = m1 + ekin;
  const double pcms = plab * mass2 / std::sqrt(m1 * m1 + mass2 * mass2 + 2.0 * mass2 * e1);
  o.tmax = 4.0 * pcms * pcms;

  if (eng) { eng->reset(phase); }
  G4HadFinalState* r = model->ApplyYourself(proj, nucleus);
  if (eng) { o.draws = eng->draws(); }

  o.efinal = r->GetEnergyChange();
  o.edep = r->GetLocalEnergyDeposit();
  const G4ThreeVector& d = r->GetMomentumChange();
  o.dirx = d.x(); o.diry = d.y(); o.dirz = d.z();
  o.nsec = int(r->GetNumberOfSecondaries());
  if (o.nsec > 0) {
    G4DynamicParticle* p = r->GetSecondary(0)->GetParticle();
    o.erec = p->GetKineticEnergy();
    o.recoil_pdg = p->GetDefinition()->GetPDGEncoding();
    const G4ThreeVector rd = p->GetMomentumDirection();
    o.recx = rd.x(); o.recy = rd.y(); o.recz = rd.z();
  } else {
    o.erec = o.edep;  // the recoil was below threshold and became a local deposit
  }
  o.t_from_erec = 2.0 * mass2 * o.erec;
  r->Clear();

  if (eng) { eng->reset(phase); }
  o.t = model->SampleInvariantT(part, plab, z, a);
  if (eng) { o.t_draws = eng->draws(); }
  // The clamp is ApplyYourself's, applied to the same expression it uses.
  o.cost_cms = (o.tmax > 0.0) ? 1.0 - 2.0 * o.t / o.tmax : 1.0;
  if (o.cost_cms > 1.0) { o.cost_cms = 1.0; }
  else if (o.cost_cms < -1.0) { o.cost_cms = -1.0; }
  return o;
}

// ---------------------------------------------------------------------------------------------

void dump_framework(const DumpContext& ctx) {
  FILE* f = std::fopen("elastic_framework.csv", "w");
  std::fprintf(f, "name,value,unit\n");

  // The recoil threshold G4HadronElasticProcess sets per step: the PROTON entry (index 3) of the
  // production-cuts energy table. G4RToEConvForProton::Convert has no material dependence, so
  // every couple must give the same number - which is itself worth recording, because a port
  // that made it material-dependent would still pass a one-material check.
  const G4ProductionCutsTable* pct = G4ProductionCutsTable::GetProductionCutsTable();
  const std::vector<G4double>* pcuts = pct->GetEnergyCutsVector(3);
  const std::size_t ncouple = pct->GetTableSize();
  for (std::size_t i = 0; i < ncouple && i < pcuts->size(); ++i) {
    const G4Material* m = pct->GetMaterialCutsCouple(int(i))->GetMaterial();
    std::fprintf(f, "protonCutEnergy_%s,%.17g,MeV\n", m->GetName().c_str(),
                 (*pcuts)[i] / MeV);
  }
  // And the converter itself, over a range of range cuts, so the formula and not one value is
  // checked. G4ProductionCutsTable::ConvertRangeToEnergy(particle, material, range).
  const G4ParticleDefinition* proton = G4Proton::Proton();
  const G4Material* water = ctx.materials[1];
  const double rcuts[] = {0.0, 0.1, 0.7, 1.0, 10.0, 100.0};
  for (double rc : rcuts) {
    const double e = const_cast<G4ProductionCutsTable*>(pct)->ConvertRangeToEnergy(
        proton, water, rc * mm);
    std::fprintf(f, "convertRangeToEnergy_proton_%.4g_mm,%.17g,MeV\n", rc, e / MeV);
  }

  // G4HadronicInteraction::GetFatalEnergyCheckLevels - what CheckResult re-samples on.
  G4HadronElastic base;
  const std::pair<G4double, G4double> lv = base.GetFatalEnergyCheckLevels();
  std::fprintf(f, "fatalEnergyCheckRelative,%.17g,\n", lv.first);
  std::fprintf(f, "fatalEnergyCheckAbsolute,%.17g,MeV\n", lv.second / MeV);
  const std::pair<G4double, G4double> ev = base.GetEnergyMomentumCheckLevels();
  std::fprintf(f, "epCheckRelativeDefaultIsDBL_MAX,%d,\n", ev.first == DBL_MAX ? 1 : 0);
  std::fprintf(f, "epCheckAbsoluteDefaultIsDBL_MAX,%d,\n", ev.second == DBL_MAX ? 1 : 0);
  std::fprintf(f, "recoilEnergyThresholdDefault,%.17g,MeV\n",
               base.GetRecoilEnergyThreshold() / MeV);

  // The energy windows QBBC's elastic models carry. G4HadronElasticPhysics sets
  // lhep0/he to emax = max(GetMaxEnergy(), 100.1 MeV) and G4IonElasticPhysics sets the ion
  // model's minimum to 0.
  const double emax = G4HadronicParameters::Instance()->GetMaxEnergy();
  std::fprintf(f, "hadronicParametersMaxEnergy,%.17g,MeV\n", emax / MeV);
  std::fprintf(f, "hadronElastic_defaultMinEnergy,%.17g,MeV\n", base.GetMinEnergy() / MeV);
  std::fprintf(f, "hadronElastic_defaultMaxEnergy,%.17g,MeV\n", base.GetMaxEnergy() / MeV);
  {
    G4ElasticHadrNucleusHE he;
    std::fprintf(f, "elasticHadrNucleusHE_minEnergy,%.17g,MeV\n", he.GetMinEnergy() / MeV);
    std::fprintf(f, "elasticHadrNucleusHE_maxEnergy,%.17g,MeV\n", he.GetMaxEnergy() / MeV);
  }
  {
    G4NuclNuclDiffuseElastic nn;
    std::fprintf(f, "nuclNuclDiffuse_ctorMinEnergy,%.17g,MeV\n", nn.GetMinEnergy() / MeV);
    std::fprintf(f, "nuclNuclDiffuse_ctorMaxEnergy,%.17g,MeV\n", nn.GetMaxEnergy() / MeV);
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

/// G4EnergyRangeManager::GetHadronicInteraction with two overlapping models, counted.
///
/// Two G4HadronElastic instances stand in for the cascades: the ranges are what matters, and the
/// range manager only ever asks a model for IsApplicable, GetMinEnergy and GetMaxEnergy. The two
/// overlaps are the ones QBBC actually has - BIC 0-1.5 GeV against BERT 1-6 GeV, and BERT
/// against FTFP 3 GeV-100 TeV.
///
/// EACH PAIR IS REGISTERED BOTH WAYS ROUND, and that is not symmetry for its own sake.
/// `GetHadronicInteraction` remembers only the last two matches, as `(emi1, ema1)` for the most
/// recent, and then branches on `emi1 < emi2`:
///
///     if(emi1 < emi2) mem = ((ema1-e) < rand*(ema1-emi2)) ? memor2 : memory;
///     else            mem = ((ema2-e) < rand*(ema2-emi1)) ? memory : memor2;
///
/// Registering the lower-threshold model first makes the upper one the most recent match, so
/// `emi1 > emi2` and only the second line ever runs. The first version of this dump did exactly
/// that for both pairs and carried a comment claiming both branches were covered; inverting the
/// ternary in the port's first branch then changed nothing that any test could see. Registering
/// the upper one first covers it. Nothing about a physics list forbids that order - the range
/// manager does not sort, it takes what ConstructProcess registered.
void dump_overlap(const DumpContext& ctx) {
  FILE* f = std::fopen("elastic_overlap.csv", "w");
  std::fprintf(f, "overlap,lower_emin,lower_emax,upper_emin,upper_emax,ekin_MeV,ntrial,"
                  "n_upper,p_upper_measured,p_upper_formula\n");

  const G4Material* mat = ctx.materials[1];
  const G4Element* elm = mat->GetElement(0);
  const G4ParticleDefinition* proton = G4Proton::Proton();
  const int kN = 200000;

  struct Pair { const char* name; double lo_min, lo_max, up_min, up_max; bool upper_first; };
  const Pair pairs[] = {
      {"BIC_BERT", 0.0, 1500.0, 1000.0, 6000.0, false},
      {"BERT_FTFP", 1000.0, 6000.0, 3000.0, 100000000.0, false},
      {"BIC_BERT_rev", 0.0, 1500.0, 1000.0, 6000.0, true},
      {"BERT_FTFP_rev", 1000.0, 6000.0, 3000.0, 100000000.0, true},
  };
  const int npairs = int(sizeof(pairs) / sizeof(pairs[0]));
  const double e_bic[] = {1000.0, 1050.0, 1125.0, 1250.0, 1375.0, 1450.0, 1500.0};
  const double e_ftf[] = {3000.0, 3300.0, 3750.0, 4500.0, 5250.0, 5700.0, 6000.0};

  for (int ip = 0; ip < npairs; ++ip) {
    const Pair& pr = pairs[ip];
    G4HadronElastic* lower = new G4HadronElastic("overlapLower");
    G4HadronElastic* upper = new G4HadronElastic("overlapUpper");
    lower->SetMinEnergy(pr.lo_min * MeV); lower->SetMaxEnergy(pr.lo_max * MeV);
    upper->SetMinEnergy(pr.up_min * MeV); upper->SetMaxEnergy(pr.up_max * MeV);
    G4EnergyRangeManager erm;
    if (pr.upper_first) {
      erm.RegisterMe(upper);
      erm.RegisterMe(lower);
    } else {
      erm.RegisterMe(lower);
      erm.RegisterMe(upper);
    }

    const double* es = (ip % 2 == 0) ? e_bic : e_ftf;
    for (int ie = 0; ie < 7; ++ie) {
      const double ekin = es[ie] * MeV;
      G4DynamicParticle dp(proton, G4ThreeVector(0, 0, 1), ekin);
      G4HadProjectile proj(dp);
      int n_upper = 0;
      for (int i = 0; i < kN; ++i) {
        G4Nucleus nucleus(16, 8);
        G4HadronicInteraction* hi = erm.GetHadronicInteraction(proj, nucleus, mat, elm);
        if (hi == upper) { ++n_upper; }
      }
      const double p_meas = double(n_upper) / double(kN);
      double p_form = (ekin / MeV - pr.up_min) / (pr.lo_max - pr.up_min);
      if (p_form < 0) { p_form = 0; }
      if (p_form > 1) { p_form = 1; }
      std::fprintf(f, "%s,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%.17g,%.17g\n", pr.name,
                   pr.lo_min, pr.lo_max, pr.up_min, pr.up_max, ekin / MeV, kN, n_upper, p_meas,
                   p_form);
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

/// G4CrossSectionDataStore::ComputeCrossSection and SampleZandA.
///
/// The store is given the real elastic data set for a proton, so the partial cross sections are
/// the ones QBBC uses, and the material is the dumper's three-element CustomDerivedI (H, C, Pb by
/// mass fraction 0.1/0.6/0.3) so that the element loop and its cumulative array actually matter.
/// Both the partial cross sections and the counted selection frequencies are written, because
/// agreeing on the frequencies while disagreeing on the cross sections would mean the port had
/// two compensating errors.
void dump_zanda(const DumpContext& ctx) {
  FILE* f = std::fopen("elastic_zanda.csv", "w");
  // iso_counts is `N:abundance:isotope_xs_mm2:count` per isotope, semicolon separated.
  std::fprintf(f, "material,ekin_MeV,nelm,total_xs_per_mm,i,Z,natoms_per_mm3,xs_per_atom_mm2,"
                  "xsecelm_cumulative,ntrial,n_selected,frac_selected,n_iso,iso_counts\n");

  const G4ParticleDefinition* proton = G4Proton::Proton();
  auto* chips = static_cast<G4ChipsProtonElasticXS*>(
      G4CrossSectionDataSetRegistry::Instance()->GetCrossSectionDataSet(
          G4ChipsProtonElasticXS::Default_Name()));

  // Two materials: the three-element custom one (index 6) and water (index 1, two elements).
  const int mat_idx[] = {6, 1};
  const double energies[] = {10.0, 100.0, 1000.0};
  const int kN = 200000;

  for (int mi = 0; mi < 2; ++mi) {
    const G4Material* mat = ctx.materials[mat_idx[mi]];
    const std::size_t nelm = mat->GetNumberOfElements();
    const G4double* natoms = mat->GetVecNbOfAtomsPerVolume();
    for (double ek : energies) {
      G4CrossSectionDataStore store;
      store.AddDataSet(chips);
      G4DynamicParticle dp(proton, G4ThreeVector(0, 0, 1), ek * MeV);
      const double total = store.ComputeCrossSection(&dp, mat);

      std::vector<int> counts(nelm, 0);
      std::vector<std::vector<int>> iso_counts(nelm);
      for (std::size_t i = 0; i < nelm; ++i) {
        iso_counts[i].assign(mat->GetElement(i)->GetNumberOfIsotopes(), 0);
      }
      for (int i = 0; i < kN; ++i) {
        G4Nucleus nucleus;
        const G4Element* e = store.SampleZandA(&dp, mat, nucleus);
        for (std::size_t k = 0; k < nelm; ++k) {
          if (mat->GetElement(k) == e) {
            ++counts[k];
            const G4Isotope* iso = nucleus.GetIsotope();
            for (std::size_t j = 0; j < e->GetNumberOfIsotopes(); ++j) {
              if (e->GetIsotope(j) == iso) { ++iso_counts[k][j]; }
            }
            break;
          }
        }
      }
      for (std::size_t i = 0; i < nelm; ++i) {
        const G4Element* e = mat->GetElement(i);
        const double per_atom = store.GetCrossSection(&dp, e, mat);
        double cum = 0.0;
        for (std::size_t k = 0; k <= i; ++k) {
          const double x = natoms[k] * store.GetCrossSection(&dp, mat->GetElement(k), mat);
          cum += (x > 0.0) ? x : 0.0;
        }
        // Per isotope: N, the relative abundance, the isotope cross section (which is what the
        // isotope-wise branch of SampleZandA weights by, and is NOT derivable from the element
        // cross section - that is only their abundance-weighted sum) and the counted selections.
        std::string isos;
        for (std::size_t j = 0; j < e->GetNumberOfIsotopes(); ++j) {
          const G4Isotope* iso = e->GetIsotope(j);
          const double ab = e->GetRelativeAbundanceVector()[j];
          const double xs = store.GetCrossSection(&dp, e->GetZasInt(), iso->GetN(), iso, e, mat);
          char buf[160];
          std::snprintf(buf, sizeof(buf), "%s%d:%.17g:%.17g:%d", j ? ";" : "", iso->GetN(), ab,
                        xs / (mm * mm), iso_counts[i][j]);
          isos += buf;
        }
        std::fprintf(f, "%s,%.17g,%d,%.17g,%d,%d,%.17g,%.17g,%.17g,%d,%d,%.17g,%d,%s\n",
                     mat->GetName().c_str(), ek, int(nelm), total * mm, int(i),
                     e->GetZasInt(), natoms[i] * mm * mm * mm, per_atom / (mm * mm), cum * mm,
                     kN, counts[i], double(counts[i]) / double(kN),
                     int(e->GetNumberOfIsotopes()), isos.c_str());
      }
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

/// G4NucleiProperties::GetNuclearMass(A, Z), the table the elastic kinematics need from P3.
void dump_nuclear_mass(const DumpContext&) {
  FILE* f = std::fopen("elastic_nuclear_mass.csv", "w");
  std::fprintf(f, "Z,A,nuclear_mass_MeV,ion_pdg_mass_MeV\n");
  // Every (Z,A) these tables use, plus every natural isotope of the targets, plus the light ions
  // that appear as projectiles or recoils.
  struct ZA { int z, a; };
  std::vector<ZA> list;
  for (int i = 0; i < kNTargets; ++i) { list.push_back({kTargets[i].z, kTargets[i].a}); }
  const ZA extra[] = {{1, 2}, {1, 3}, {2, 3}, {2, 4}, {6, 14}, {7, 14}, {8, 17}, {8, 18},
                      {13, 26}, {26, 54}, {26, 57}, {26, 58}, {82, 204}, {82, 206},
                      {92, 235}, {92, 238}, {1, 1}, {0, 1}};
  for (const ZA& e : extra) { list.push_back(e); }
  // And, so the port's table is general rather than fitted to the tests, every A that
  // G4ElasticHadrNucleusHE's own rounded-NIST-A rule can produce for Z = 1..92.
  auto* nist = G4NistManager::Instance();
  for (int z = 1; z <= 92; ++z) {
    list.push_back({z, G4lrint(nist->GetAtomicMassAmu(z))});
  }
  for (const ZA& e : list) {
    const double m = G4NucleiProperties::GetNuclearMass(e.a, e.z);
    double ionm = 0.0;
    if (e.z >= 1 && e.a >= e.z) {
      const G4ParticleDefinition* d =
          G4ParticleTable::GetParticleTable()->GetIonTable()->GetIon(e.z, e.a, 0.0);
      if (d) { ionm = d->GetPDGMass(); }
    }
    std::fprintf(f, "%d,%d,%.17g,%.17g\n", e.z, e.a, m / MeV, ionm / MeV);
  }
  std::fclose(f);
}

/// The A that G4ElasticHadrNucleusHE builds each element's table at.
void dump_he_nist_a(const DumpContext&) {
  FILE* f = std::fopen("elastic_he_nist_a.csv", "w");
  std::fprintf(f, "Z,atomic_mass_amu,lrint_A\n");
  auto* nist = G4NistManager::Instance();
  for (int z = 1; z <= 92; ++z) {
    const double amu = nist->GetAtomicMassAmu(z);
    std::fprintf(f, "%d,%.17g,%d\n", z, amu, G4lrint(amu));
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

/// The CHIPS classes through their public interface, deterministically.
///
/// `GetChipsCrossSection` returns the total elastic cross section and, as a side effect, leaves
/// the class's nine interpolated differential parameters and its (-t)max set - which is why
/// `GetHMaxT` and `GetExchangeT` are read immediately after and never before. `GetHMaxT` is
/// `lastTM*GeV^2/2`, so it pins GetQ2max exactly. `GetSlope`, which would have given theB1, is
/// private in both classes and is not reachable.
///
/// `GetExchangeT` is public in both, so the whole t-sampler is exercised directly under the
/// prescribed uniform cycle: no ApplyYourself, no phi draw, no resample branch. That makes every
/// one of the nine interpolated parameters observable, because each of the four channels of the
/// nuclear branch uses a different subset of them and the eight phases visit different channels.
void dump_chips_xs(const DumpContext&) {
  FILE* f = std::fopen("elastic_chips_xs.csv", "w");
  std::fprintf(f, "pdg,Z,N,p_MeV,cross_section_mb,hmaxt_MeV2,phase,ndraws,exchange_t_MeV2\n");

  auto* px = static_cast<G4ChipsProtonElasticXS*>(
      G4CrossSectionDataSetRegistry::Instance()->GetCrossSectionDataSet(
          G4ChipsProtonElasticXS::Default_Name()));
  auto* nx = static_cast<G4ChipsNeutronElasticXS*>(
      G4CrossSectionDataSetRegistry::Instance()->GetCrossSectionDataSet(
          G4ChipsNeutronElasticXS::Default_Name()));

  auto* eng = new CycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);

  // Momenta from below the S-wave limit (13.5 MeV/c) to the top of the tabulation.
  const double moms[] = {1.0, 5.0, 13.0, 14.0, 50.0, 100.0, 300.0, 1000.0, 3000.0,
                         10000.0, 30000.0, 100000.0, 1000000.0};
  for (int t = 0; t < kNTargets; ++t) {
    const int z = kTargets[t].z, a = kTargets[t].a;
    int n = a - z;
    // G4ChipsElasticModel's own remap, applied here so the numbers line up with the model's.
    if (z == 1 && n == 2) { n = 1; } else if (z == 2 && n == 1) { n = 2; }
    for (double p : moms) {
      const double csp = px->GetChipsCrossSection(p * MeV, z, n, 2212);
      const double hmp = px->GetHMaxT();
      for (int phase = 0; phase < 8; ++phase) {
        // Recompute the cross section each time: GetExchangeT reads the class's `lastLP`,
        // `lastTM` and the nine parameters, and only GetChipsCrossSection sets them.
        px->GetChipsCrossSection(p * MeV, z, n, 2212);
        eng->reset(phase);
        const double t_ex = px->GetExchangeT(z, n, 2212);
        std::fprintf(f, "2212,%d,%d,%.17g,%.17g,%.17g,%d,%d,%.17g\n", z, n, p,
                     csp / millibarn, hmp / (MeV * MeV), phase, eng->draws(),
                     t_ex / (MeV * MeV));
      }
      const double csn = nx->GetChipsCrossSection(p * MeV, z, n, 2112);
      const double hmn = nx->GetHMaxT();
      for (int phase = 0; phase < 8; ++phase) {
        nx->GetChipsCrossSection(p * MeV, z, n, 2112);
        eng->reset(phase);
        const double t_ex = nx->GetExchangeT(z, n, 2112);
        std::fprintf(f, "2112,%d,%d,%.17g,%.17g,%.17g,%d,%d,%.17g\n", z, n, p,
                     csn / millibarn, hmn / (MeV * MeV), phase, eng->draws(),
                     t_ex / (MeV * MeV));
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

struct ModelSpec {
  const char* name;
  /// G4HadronElastic, not G4HadronicInteraction: `run_apply` needs `SampleInvariantT`, and all
  /// four elastic models derive from it.
  G4HadronElastic* model;
  const G4ParticleDefinition* part;
};

/// Deterministic sampling: every model under the prescribed uniform cycle.
void dump_sample_t(const DumpContext&) {
  FILE* f = std::fopen("elastic_sample_t.csv", "w");
  std::fprintf(f, "model,pdg,Z,A,ekin_MeV,phase,ndraws,tmax_MeV2,t_MeV2,cost_cms,"
                  "efinal_MeV,edep_MeV,erec_MeV,nsec,recoil_pdg,"
                  "dirx,diry,dirz,recx,recy,recz,t_draws,t_from_erec_MeV2\n");

  auto* eng = new CycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);

  auto* gheisha = new G4HadronElastic("dumpGheisha");
  auto* chips = new G4ChipsElasticModel();
  auto* he = new G4ElasticHadrNucleusHE();
  auto* ion = new G4NuclNuclDiffuseElastic();
  ion->SetMinEnergy(0.0);

  const G4ParticleDefinition* p = G4Proton::Proton();
  const G4ParticleDefinition* n = G4Neutron::Neutron();
  const G4ParticleDefinition* pip = G4PionPlus::PionPlus();
  const G4ParticleDefinition* pim = G4PionMinus::PionMinus();
  const G4ParticleDefinition* alpha = G4Alpha::Alpha();
  const G4ParticleDefinition* c12 =
      G4ParticleTable::GetParticleTable()->GetIonTable()->GetIon(6, 12, 0.0);

  const ModelSpec specs[] = {
      {"Gheisha", gheisha, p},     {"Gheisha", gheisha, alpha},
      {"Chips", chips, p},         {"Chips", chips, n},
      {"HE", he, pip},             {"HE", he, pim},
      {"NNDiffuse", ion, alpha},   {"NNDiffuse", ion, c12},
  };
  const int nspec = int(sizeof(specs) / sizeof(specs[0]));

  for (int s = 0; s < nspec; ++s) {
    if (specs[s].part == nullptr) { continue; }
    for (int t = 0; t < kNTargets; ++t) {
      for (int e = 0; e < kNEnergies; ++e) {
        for (int phase = 0; phase < 8; ++phase) {
          const ApplyOut o = run_apply(specs[s].model, specs[s].part, kEnergies[e] * MeV,
                                       kTargets[t].z, kTargets[t].a, eng, phase);
          std::fprintf(f,
                       "%s,%d,%d,%d,%.17g,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,"
                       "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g\n",
                       specs[s].name, specs[s].part->GetPDGEncoding(), kTargets[t].z,
                       kTargets[t].a, kEnergies[e], phase, o.draws, o.tmax / (MeV * MeV),
                       o.t / (MeV * MeV), o.cost_cms, o.efinal / MeV, o.edep / MeV,
                       o.erec / MeV, o.nsec, o.recoil_pdg, o.dirx, o.diry, o.dirz, o.recx,
                       o.recy, o.recz, o.t_draws, o.t_from_erec / (MeV * MeV));
        }
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

/// Statistical sampling: the same models with the real engine at a fixed seed.
///
/// Twenty bins uniform in t/tmax, plus the first four moments of t/tmax and of cos(theta_cm).
/// The seed is fixed and stated so the file is reproducible; the port cannot follow the same
/// stream, so what is compared is the distribution, with the tolerance the test states.
void dump_moments(const DumpContext&) {
  FILE* f = std::fopen("elastic_moments.csv", "w");
  std::fprintf(f, "model,pdg,Z,A,ekin_MeV,nsample,mean_x,var_x,mean_cost,var_cost,"
                  "frac_below_thr_70keV,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,"
                  "h15,h16,h17,h18,h19\n");

  CLHEP::HepJamesRandom fixed(20250910);
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&fixed);

  auto* gheisha = new G4HadronElastic("dumpGheishaStat");
  auto* chips = new G4ChipsElasticModel();
  auto* he = new G4ElasticHadrNucleusHE();
  auto* ion = new G4NuclNuclDiffuseElastic();
  ion->SetMinEnergy(0.0);

  const G4ParticleDefinition* p = G4Proton::Proton();
  const G4ParticleDefinition* n = G4Neutron::Neutron();
  const G4ParticleDefinition* pip = G4PionPlus::PionPlus();
  const G4ParticleDefinition* pim = G4PionMinus::PionMinus();
  const G4ParticleDefinition* alpha = G4Alpha::Alpha();
  const G4ParticleDefinition* c12 =
      G4ParticleTable::GetParticleTable()->GetIonTable()->GetIon(6, 12, 0.0);

  const ModelSpec specs[] = {
      {"Gheisha", gheisha, p},   {"Gheisha", gheisha, alpha},
      {"Chips", chips, p},       {"Chips", chips, n},
      {"HE", he, pip},           {"HE", he, pim},
      {"NNDiffuse", ion, alpha}, {"NNDiffuse", ion, c12},
  };
  const int nspec = int(sizeof(specs) / sizeof(specs[0]));
  const int kN = 20000;
  const double thr = 0.07 * MeV;  // the 0.7 mm proton cut, the recoil threshold in QBBC

  for (int s = 0; s < nspec; ++s) {
    if (specs[s].part == nullptr) { continue; }
    for (int t = 0; t < kNTargets; ++t) {
      for (int e = 0; e < kNEnergies; ++e) {
        double sx = 0, sxx = 0, sc = 0, scc = 0;
        int below = 0;
        int hist[20] = {0};
        for (int i = 0; i < kN; ++i) {
          const ApplyOut o = run_apply(specs[s].model, specs[s].part, kEnergies[e] * MeV,
                                       kTargets[t].z, kTargets[t].a, nullptr, 0);
          const double x = (o.tmax > 0.0) ? o.t / o.tmax : 0.0;
          sx += x; sxx += x * x;
          sc += o.cost_cms; scc += o.cost_cms * o.cost_cms;
          if (o.erec <= thr) { ++below; }
          int b = int(x * 20.0);
          if (b < 0) { b = 0; }
          if (b > 19) { b = 19; }
          ++hist[b];
        }
        const double mx = sx / kN, mc = sc / kN;
        std::fprintf(f, "%s,%d,%d,%d,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g",
                     specs[s].name, specs[s].part->GetPDGEncoding(), kTargets[t].z,
                     kTargets[t].a, kEnergies[e], kN, mx, sxx / kN - mx * mx, mc,
                     scc / kN - mc * mc, double(below) / double(kN));
        for (int b = 0; b < 20; ++b) { std::fprintf(f, ",%d", hist[b]); }
        std::fprintf(f, "\n");
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

void dump_elastic(const DumpContext& ctx) {
  dump_framework(ctx);
  dump_overlap(ctx);
  dump_zanda(ctx);
  dump_nuclear_mass(ctx);
  dump_he_nist_a(ctx);
  dump_chips_xs(ctx);
  dump_sample_t(ctx);
  dump_moments(ctx);
}

}  // namespace

G4GPU_REGISTER_DUMP("elastic",
                    "elastic_framework.csv elastic_overlap.csv elastic_zanda.csv "
                    "elastic_nuclear_mass.csv elastic_he_nist_a.csv elastic_chips_xs.csv "
                    "elastic_sample_t.csv elastic_moments.csv",
                    dump_elastic);
