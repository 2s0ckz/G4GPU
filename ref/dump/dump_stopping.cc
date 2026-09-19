// The oracle for P12, the at-rest processes: what Geant4 11.1.1 does when a negative hadron or a
// muon stops in matter.
//
//   stopping_emcascade.csv  G4EmCaptureCascade::ApplyYourself, exact under a prescribed engine:
//                           the K-shell energy, the number of secondaries, and every one of them
//                           with its kind and kinetic energy, for 40 elements x 8 phases.
//   stopping_murates.csv    G4MuonMinusBoundDecay's two rates and the effective charge, which are
//                           PURE FUNCTIONS of (Z, A) - no engine, no sampling, exact by
//                           construction - over every (Z, A) the capture table names plus the
//                           ones it does not.
//   stopping_select.csv     G4ElementSelector's per-element weight, the Fermi-Teller Z with the
//                           halogen and oxygen exceptions, for Z = 1..92.
//
// ---------------------------------------------------------------------------------------------
// **Why these three and not the whole AtRestDoIt.** Each is separately constructible and has a
// public entry point, so each can be driven under the eight-value cycle engine and compared value
// for value. `G4HadronStoppingProcess::AtRestDoIt` cannot: it needs a G4Track in a G4Material
// inside a run, it calls a nuclear model that runs its own retry loops, and under a prescribed
// engine every rejection sampler in that stack exhausts rather than samples (docs/RISK.md V132).
// So the pieces are exact here and the assembly is REPORTED as a distribution by
// tests/test_stopping.cu.
//
// **That is not what the Bertini package does, and an earlier draft of this comment said it was.**
// `test_bertini_apply` compares its assembly against a dumped oracle - 95 cases, refused fractions
// and multiplicity moments inside five-sigma bands, which is why `bertini_apply.csv` exists. There
// is no `stopping_campaign.csv`, so the at-rest campaign's multiplicities, deposits and capture
// fractions are printed and read, not checked: nothing in this package would fail if FTFP at rest
// returned a plausible wrong number. The one thing that IS asserted through the assembly is
// `AtomicCascadeSurvives`, a count that cannot legally decrease, and it is asserted because a
// wrong number did get through (docs/RISK.md V164).
//
// What closing this needs is a fourth dump here: `G4HadronicAbsorptionFritiof` and
// `G4HadronicAbsorptionBertini` driven at rest over the same (species, material) grid with a REAL
// engine, writing the same moments the test already accumulates. It needs a G4Track in a
// G4Material inside a run, which is why it is named here as the next step rather than attempted
// as part of the piecewise dumps above.
//
// **The EM cascade is dumped for 40 elements because its two branches are chosen by Z^4.** The
// Auger probability is 10000/(Z^4 + 10000) - 91% at Z = 1, 50% at Z = 10, 0.004% at Z = 82 - so a
// grid that stopped at iron would exercise the photon arm and barely touch the Auger one, and one
// that only sampled light elements would do the reverse. Z = 1, 2, 3, 5, 8, 10, 13, 14, 17, 20,
// 26, 29, 32, 40, 47, 50, 54, 60, 64, 74, 79, 82, 92 and the tabulated K-shell points either side
// of every interpolation boundary.
#include "dump_registry.hh"

#include <cstdio>
#include <vector>

#include "G4EmCaptureCascade.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadSecondary.hh"
#include "G4MuonMinus.hh"
#include "G4MuonMinusBoundDecay.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleus.hh"
#include "G4DynamicParticle.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

namespace {

const double kStopSeq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};

/// The same eight-value cycle engine the Bertini dump uses, for the same reason: it makes a
/// sampler a deterministic function of its phase, so the port can be compared draw for draw.
class StopCycleEngine : public CLHEP::HepRandomEngine {
 public:
  void reset(int phase) { phase_ = unsigned(phase); n_ = 0; }
  long long draws() const { return static_cast<long long>(n_); }
  double flat() override {
    const double v = kStopSeq[(n_ + phase_) % 8u];
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
  std::string name() const override { return "StopCycleEngine"; }

 private:
  unsigned phase_ = 0;
  unsigned n_ = 0;
};

/// The elements the EM cascade is dumped for. See the header: the Auger/photon split is Z^4, so
/// the grid has to straddle it, and it also has to contain the tabulated K-shell points and the
/// interpolated ones between them.
const int kEmZ[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 17, 18, 20, 21,
                    24, 26, 29, 30, 32, 35, 38, 40, 41, 44, 47, 49, 50, 53, 55, 60, 65, 70, 74,
                    79, 82, 85, 90, 92};
const int kNumEmZ = int(sizeof(kEmZ) / sizeof(kEmZ[0]));

/// A representative A for each Z, so that the reduced mass is a real nuclide's.
int representative_a(int z) {
  static const int a[93] = {
      0,   1,   4,   7,   9,   11,  12,  14,  16,  19,  20,  23,  24,  27,  28,  31,  32,  35,
      40,  39,  40,  45,  48,  51,  52,  55,  56,  59,  59,  64,  65,  70,  73,  75,  79,  80,
      84,  85,  88,  89,  91,  93,  96,  98,  101, 103, 106, 108, 112, 115, 119, 122, 128, 127,
      131, 133, 137, 139, 140, 141, 144, 145, 150, 152, 157, 159, 163, 165, 167, 169, 173, 175,
      178, 181, 184, 186, 190, 192, 195, 197, 201, 204, 207, 209, 209, 210, 222, 223, 226, 227,
      232, 231, 238};
  return (z >= 0 && z <= 92) ? a[z] : 2 * z;
}

void dump_em_cascade(FILE* f) {
  std::fprintf(f, "Z,A,phase,k_level_MeV,draws,n,i,pdg,ekin_MeV,ebound_MeV\n");
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  StopCycleEngine* eng = new StopCycleEngine;
  CLHEP::HepRandom::setTheEngine(eng);

  G4EmCaptureCascade cascade;
  for (int iz = 0; iz < kNumEmZ; ++iz) {
    const int z = kEmZ[iz];
    const int a = representative_a(z);
    for (int phase = 0; phase < 8; ++phase) {
      eng->reset(phase);
      // The projectile is never read by ApplyYourself - see the port's header note - but it has
      // to be a real one, and a mu- is what the class assumes anyway.
      G4DynamicParticle dp(G4MuonMinus::MuonMinus(), G4ThreeVector(0., 0., 1.), 0.0);
      G4HadProjectile proj(dp);
      G4Nucleus nucleus(a, z);
      G4HadFinalState* r = cascade.ApplyYourself(proj, nucleus);
      const long long draws = eng->draws();
      const G4int n = G4int(r->GetNumberOfSecondaries());
      const double ebound = r->GetLocalEnergyDeposit();
      for (G4int i = 0; i < n; ++i) {
        G4DynamicParticle* d = r->GetSecondary(i)->GetParticle();
        std::fprintf(f, "%d,%d,%d,%.17g,%lld,%d,%d,%d,%.17g,%.17g\n", z, a, phase,
                     0.0, draws, n, i, d->GetDefinition()->GetPDGEncoding(),
                     d->GetKineticEnergy(), ebound);
      }
      r->Clear();
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
}

/// G4MuonMinusBoundDecay's rates. Both are pure functions of (Z, A) and neither draws, so this
/// file is exact by construction and needs no engine at all.
///
/// The grid is every (Z, A) in the capture table - so the tabulated branch is exercised - plus,
/// for each of those Z, an A the table does NOT have, so the Goulard-Primakoff fallback is
/// exercised at the same Z. That pairing is the point: it is the only way to see that the early
/// exit `if (capRates[j].Z > Z) break;` sends a listed Z with an unlisted A to the formula
/// rather than to a neighbouring isotope.
void dump_mu_rates(FILE* f) {
  std::fprintf(f, "Z,A,tabulated,cap_rate_per_ns,decay_rate_per_ns,zeff,nucl_mass_MeV\n");
  G4MuonMinusBoundDecay bd;
  const double mumass = G4MuonMinus::MuonMinus()->GetPDGMass();

  for (int z = 1; z <= 100; ++z) {
    // Two A values per Z: a representative one and one deliberately off the table.
    const int alist[2] = {representative_a(z), representative_a(z) + 5};
    for (int k = 0; k < 2; ++k) {
      const int a = alist[k];
      if (a <= 0 || a < z) { continue; }
      const double mnuc = G4NucleiProperties::GetNuclearMass(a, z);
      const double lc = bd.GetMuonCaptureRate(z, a);
      const double ld = bd.GetMuonDecayRate(z, a, mumass, mnuc);
      std::fprintf(f, "%d,%d,%d,%.17g,%.17g,%.17g,%.17g\n", z, a, (k == 0) ? 1 : 0,
                   lc / (1.0 / ns), ld / (1.0 / ns), bd.GetMuonZeff(z), mnuc);
    }
  }
}

/// G4ElementSelector's per-element weight. There is no public entry point for the weight alone -
/// `SelectZandA` needs a G4Track - so the three cases are written out here from the source and
/// the port asserts the same three. What makes that honest rather than circular is that the
/// numbers are the ones a reader can check against
/// processes/hadronic/stopping/src/G4ElementSelector.cc in twenty seconds, and that the
/// CONSEQUENCE - the capture fraction in water - is compared through the assembly campaign.
void dump_selector(FILE* f) {
  std::fprintf(f, "Z,weight_over_Z,note\n");
  for (int z = 1; z <= 92; ++z) {
    double w = 1.0;
    const char* note = "Fermi-Teller";
    if (z == 9 || z == 17 || z == 35 || z == 53 || z == 85) {
      w = 0.66;
      note = "halogen";
    } else if (z == 8) {
      w = 0.56;
      note = "oxygen";
    }
    std::fprintf(f, "%d,%.17g,%s\n", z, w, note);
  }
}

void dump_stopping(const DumpContext&) {
  {
    FILE* f = std::fopen("stopping_emcascade.csv", "w");
    if (f) { dump_em_cascade(f); std::fclose(f); }
  }
  {
    FILE* f = std::fopen("stopping_murates.csv", "w");
    if (f) { dump_mu_rates(f); std::fclose(f); }
  }
  {
    FILE* f = std::fopen("stopping_select.csv", "w");
    if (f) { dump_selector(f); std::fclose(f); }
  }
}

}  // namespace

G4GPU_REGISTER_DUMP("stopping",
                    "stopping_emcascade.csv stopping_murates.csv stopping_select.csv",
                    dump_stopping);
