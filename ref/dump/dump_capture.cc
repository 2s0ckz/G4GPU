// Oracle for package P7: neutron radiative capture.
//
// Registered through dump_registry.hh, so nothing shared is edited. Writes:
//
//   capture_masses.csv   G4NucleiProperties::GetNuclearMass(A, Z) for every target in the grid
//                        and for its compound (A+1, Z), and the capture Q value the two imply.
//                        READ OUT of the public function, not recomputed - the mass balance is
//                        the deterministic half of this model and docs/RISK.md V37 is what
//                        happens when a dump does arithmetic instead of reading.
//   capture_det.csv      DETERMINISTIC. G4NeutronRadCapture::ApplyYourself under a prescribed
//                        eight-value uniform cycle, so the whole final state - gamma energies,
//                        directions, the residual's kinetic energy, the cascade's time - is a
//                        pure function of (target, neutron energy, phase) and can be compared
//                        to the last digits. The number of uniforms consumed is recorded, which
//                        is itself a check on the control flow: a port that took a different
//                        branch of GenerateGamma reads a different uniform and diverges from
//                        there, and the draw count says so before the energies do.
//   capture_stat.csv     STATISTICAL. The real engine at a fixed seed, 20,000 captures per
//                        (target, energy): the gamma multiplicity distribution, the gamma
//                        energy spectrum, the residual's kinetic energy and how often the
//                        cascade stops on an isomer.
//
// WHY TWO SETS OF TARGETS, AND WHY THE ISOTOPES OF IRON
//
// `ApplyYourself` increments A and then tests `A <= 4`, so the split between the one-gamma
// branch and photon evaporation is at a TARGET mass number of 3. The grid therefore has to
// straddle it: H1 and H2 take the first branch and everything from He4 up takes the second,
// and both are reached in water. Iron's four stable isotopes are in the grid because the
// capture cross section and the level scheme are per-isotope and `SampleZandA` picks the
// isotope - so a port that got the isotope right and the level manager wrong, or the other way
// round, has to fail on one of Fe54/56/57/58 rather than on an element average.
//
// ONE MODEL PER CALL IN THE DETERMINISTIC TABLE, ONE FOR THE WHOLE RUN IN THE STATISTICAL ONE
//
// `G4NeutronRadCapture` owns one `G4PhotonEvaporation` for the life of the run, and its
// `fIndex` survives from one capture to the next. `G4LevelManager::NearestLevelIndex(energy,
// index)` short-circuits when the hinted level is within 10 eV of the energy asked for, so a
// stale index can change the answer - rarely, because a completed cascade leaves fIndex at 0,
// which is where a fresh state starts, and only a cascade that stopped on an isomer leaves it
// anywhere else. A device port has nowhere to keep per-track model state, so the port uses a
// fresh state per capture (its `persistent` argument is null in transport).
//
// The deterministic table therefore constructs a fresh model per call, so that what it compares
// is the transcription and not the history; the statistical table uses ONE model for all 20,000
// calls, exactly as a run does, so the size of the difference is measured by the comparison
// rather than argued about. If the two disciplines disagree, the isomer-terminated fraction in
// capture_stat.csv is the number that says why.

#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <vector>

#include "G4DynamicParticle.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadSecondary.hh"
#include "G4HadronicParameters.hh"
#include "G4Ions.hh"
#include "G4NeutronRadCapture.hh"
#include "G4Neutron.hh"
#include "G4NuclearLevelData.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleus.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandomEngine.h"

namespace {

// ---------------------------------------------------------------------------------------------
// A CLHEP engine returning a prescribed cycle, counting the draws.
//
// The same eight values and the same rationale as ref/dump/dump_elastic.cc: spread over (0,1)
// and away from both endpoints. They are duplicated rather than shared because the two dumps
// are separate translation units by design (dump_registry.hh) and a header holding eight
// doubles would be the one shared file this arrangement exists to avoid.
// ---------------------------------------------------------------------------------------------

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
// The grid.
// ---------------------------------------------------------------------------------------------

struct TargetSpec { int z; int a; const char* name; };

/// H, C, O, Al, Fe, Pb as the plan asks, with both hydrogen isotopes (the one-gamma branch),
/// the three stable oxygen isotopes (water is what the dose is measured in) and all four
/// stable iron isotopes (the isotope dependence of the level scheme).
///
/// H3 AND He3 ARE HERE BECAUSE THE BRANCH BOUNDARY IS AT A = 3 AND NOTHING ELSE REACHES IT.
///
/// `ApplyYourself` increments A and tests `A <= 4`, so the ONLY target for which `A <= 4` and
/// `A <= 3` are different tests is one with A = 3 - and the first version of this grid had
/// none, so `tests/test_capture.cu` passed with the boundary moved by one. That is the vacuity
/// the plan's working rule 7 is about, and it was found by moving it. The two A = 3 targets
/// also cover the two ends of the branch: He3 + n is bound (a 20.6 MeV Q, the largest here) and
/// its residual is an alpha, while H3 + n is UNBOUND - H4 is heavier than H3 plus a neutron -
/// so `M - mass` is negative, `lowestEnergyLimit` refuses it, and the capture emits nothing at
/// all. Neither path exists anywhere else in the grid.
const TargetSpec kTargets[] = {
    {1, 1, "H1"},     {1, 2, "H2"},     {1, 3, "H3"},     {2, 3, "He3"},
    {6, 12, "C12"},   {6, 13, "C13"},
    {8, 16, "O16"},   {8, 17, "O17"},   {8, 18, "O18"},
    {13, 27, "Al27"},
    {26, 54, "Fe54"}, {26, 56, "Fe56"}, {26, 57, "Fe57"}, {26, 58, "Fe58"},
    {82, 208, "Pb208"},
};
const int kNTargets = int(sizeof(kTargets) / sizeof(kTargets[0]));

/// Neutron kinetic energies, MeV. Thermal is 0.0253 eV = 2.53e-8 MeV, which is where the
/// capture cross section lives; 20 MeV is G4NeutronCaptureXS's ceiling, at and above which the
/// cross section is exactly zero, so 10 MeV is the highest energy the process can reach and it
/// is in the grid for that reason.
const double kEnergies[] = {2.53e-8, 1.0e-3, 0.1, 1.0, 10.0};
const int kNEnergies = int(sizeof(kEnergies) / sizeof(kEnergies[0]));

/// The eight phases of the prescribed cycle, as dump_elastic uses.
const int kNPhases = 8;

/// One secondary, read out of the G4HadFinalState.
struct SecOut {
  int pdg;
  int z;
  int a;
  /// `def->GetPDGMass()` - the DEFINITION's mass, which FillResult compares against.
  double mass;
  /// `p->GetMass()` - the DYNAMICAL mass, which is what the four-momentum implies and is NOT
  /// the same number. G4DynamicParticle's four-momentum constructor replaces the definition's
  /// mass with `sqrt(t*t - |p|^2)` whenever the two mass-SQUAREDs differ by more than
  /// (1e-2 keV)^2 - and one ulp of a nucleus's mass squared already exceeds that, so every
  /// residual on the one-gamma branch carries a dynamical mass an ulp off its PDG one. Dumped
  /// separately because the port has one field for it and the choice of which to store is a
  /// decision the test has to be able to see.
  double dynmass;
  double ekin;
  double dx, dy, dz;
  double time;
  double excitation;
};

/// One call of ApplyYourself, read out.
struct CallOut {
  int nsec = 0;
  int draws = 0;
  int status = 0;
  double energy_change = 0;
  double local_deposit = 0;
  std::vector<SecOut> sec;
};

/// `G4Ions::GetExcitationEnergy()` where the definition is an ion, zero otherwise. It is the
/// number the isomer refusal is about: for a residual that stopped on a metastable level Geant4
/// has already snapped it onto G4ENSDFSTATE inside GetIon, so this is the SNAPPED value and the
/// port's is the raw one.
double excitation_of(const G4ParticleDefinition* def) {
  const G4Ions* ion = dynamic_cast<const G4Ions*>(def);
  return (ion != nullptr) ? ion->GetExcitationEnergy() : 0.0;
}

CallOut run_capture(G4NeutronRadCapture* model, int z, int a, double ekin_MeV,
                    CycleEngine* eng, int phase) {
  CallOut out;
  G4DynamicParticle dp(G4Neutron::Neutron(), G4ThreeVector(0., 0., 1.), ekin_MeV * MeV);
  G4HadProjectile pro(dp);
  G4Nucleus nucl(a, z);
  if (eng != nullptr) { eng->reset(phase); }
  G4HadFinalState* r = model->ApplyYourself(pro, nucl);
  out.draws = (eng != nullptr) ? eng->draws() : 0;
  out.status = int(r->GetStatusChange());
  out.energy_change = r->GetEnergyChange();
  out.local_deposit = r->GetLocalEnergyDeposit();
  const G4int n = G4int(r->GetNumberOfSecondaries());
  out.nsec = n;
  out.sec.reserve(std::size_t(n));
  for (G4int i = 0; i < n; ++i) {
    const G4HadSecondary* hs = r->GetSecondary(i);
    const G4DynamicParticle* p = hs->GetParticle();
    const G4ParticleDefinition* def = p->GetDefinition();
    SecOut s;
    s.pdg = def->GetPDGEncoding();
    s.z = def->GetAtomicNumber();
    s.a = def->GetAtomicMass();
    s.mass = def->GetPDGMass();
    s.dynmass = p->GetMass();
    s.ekin = p->GetKineticEnergy();
    const G4ThreeVector d = p->GetMomentumDirection();
    s.dx = d.x(); s.dy = d.y(); s.dz = d.z();
    s.time = hs->GetTime();
    s.excitation = excitation_of(def);
    out.sec.push_back(s);
  }
  // The model owns theParticleChange and clears it at the top of the next call, so the
  // secondaries above must be read before another call and are.
  return out;
}

// ---------------------------------------------------------------------------------------------
// capture_masses.csv
// ---------------------------------------------------------------------------------------------

void dump_masses(const DumpContext&) {
  FILE* f = std::fopen("capture_masses.csv", "w");
  std::fprintf(f, "name,z,a,target_mass_MeV,compound_mass_MeV,neutron_mass_MeV,q_MeV,"
                  "min_excitation_MeV\n");
  const double mn = G4Neutron::Neutron()->GetPDGMass();
  const double minexc =
      G4NuclearLevelData::GetInstance()->GetParameters()->GetMinExcitation();
  for (int t = 0; t < kNTargets; ++t) {
    const int z = kTargets[t].z, a = kTargets[t].a;
    const double mt = G4NucleiProperties::GetNuclearMass(a, z);
    const double mc = G4NucleiProperties::GetNuclearMass(a + 1, z);
    std::fprintf(f, "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n", kTargets[t].name, z, a, mt, mc,
                 mn, mt + mn - mc, minexc);
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// capture_det.csv
// ---------------------------------------------------------------------------------------------

void dump_det(const DumpContext&) {
  FILE* f = std::fopen("capture_det.csv", "w");
  std::fprintf(f, "name,z,a,ekin_MeV,phase,draws,nsec,status,energy_change_MeV,"
                  "local_deposit_MeV,i,pdg,sec_z,sec_a,sec_mass_MeV,sec_dynmass_MeV,sec_ekin_MeV,dx,dy,dz,"
                  "time_ns,sec_exc_MeV\n");

  CycleEngine cycle;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&cycle);

  for (int t = 0; t < kNTargets; ++t) {
    for (int e = 0; e < kNEnergies; ++e) {
      for (int ph = 0; ph < kNPhases; ++ph) {
        // A FRESH MODEL PER CALL. See the file header: G4PhotonEvaporation's fIndex survives
        // across captures in a run and cannot in this port, so the deterministic table is
        // taken with no history at all.
        G4NeutronRadCapture model;
        model.InitialiseModel();
        const CallOut o =
            run_capture(&model, kTargets[t].z, kTargets[t].a, kEnergies[e], &cycle, ph);
        if (o.nsec == 0) {
          std::fprintf(f, "%s,%d,%d,%.17g,%d,%d,%d,%d,%.17g,%.17g,-1,0,0,0,0,0,0,0,0,0,0,0\n",
                       kTargets[t].name, kTargets[t].z, kTargets[t].a, kEnergies[e], ph,
                       o.draws, o.nsec, o.status, o.energy_change, o.local_deposit);
          continue;
        }
        for (int i = 0; i < o.nsec; ++i) {
          const SecOut& s = o.sec[std::size_t(i)];
          std::fprintf(f,
                       "%s,%d,%d,%.17g,%d,%d,%d,%d,%.17g,%.17g,%d,%d,%d,%d,%.17g,%.17g,"
                       "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                       kTargets[t].name, kTargets[t].z, kTargets[t].a, kEnergies[e], ph,
                       o.draws, o.nsec, o.status, o.energy_change, o.local_deposit, i, s.pdg,
                       s.z, s.a, s.mass, s.dynmass, s.ekin, s.dx, s.dy, s.dz, s.time,
                       s.excitation);
        }
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// capture_stat.csv
// ---------------------------------------------------------------------------------------------

/// Multiplicity bins 0..15 (16 columns, the last one inclusive of everything above), and 20
/// bins of gamma energy uniform over [0, 10] MeV. Both binnings are FIXED so the port bins the
/// same way; the highest gamma a capture can emit is the compound's excitation, which for
/// thermal capture is the Q value - 2.2 MeV on hydrogen, 7.6 MeV on Fe56 - so 10 MeV covers
/// every row and the top bin catches a 10 MeV neutron's own energy on top of it.
const int kNMult = 16;
const int kNSpec = 20;
const double kSpecMax = 10.0;  // MeV

void dump_stat(const DumpContext&) {
  FILE* f = std::fopen("capture_stat.csv", "w");
  std::fprintf(f, "name,z,a,ekin_MeV,nsample,mean_ngamma,var_ngamma,mean_nelectron,"
                  "mean_egamma_MeV,var_egamma_MeV2,mean_etot_gamma_MeV,mean_res_ekin_MeV,"
                  "frac_isomer,mean_time_ns");
  for (int i = 0; i < kNMult; ++i) { std::fprintf(f, ",m%d", i); }
  for (int i = 0; i < kNSpec; ++i) { std::fprintf(f, ",s%d", i); }
  std::fprintf(f, "\n");

  CLHEP::HepJamesRandom fixed(20260910);
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&fixed);

  // ONE model for every call, as a run has. See the file header.
  G4NeutronRadCapture model;
  model.InitialiseModel();

  const int kN = 20000;
  for (int t = 0; t < kNTargets; ++t) {
    for (int e = 0; e < kNEnergies; ++e) {
      double sg = 0, sgg = 0, se = 0, sel = 0, seg = 0, segg = 0, setot = 0, sres = 0, stime = 0;
      long long ngam_total = 0;
      int isomer = 0;
      int mult[kNMult] = {0};
      int spec[kNSpec] = {0};
      for (int i = 0; i < kN; ++i) {
        const CallOut o = run_capture(&model, kTargets[t].z, kTargets[t].a, kEnergies[e],
                                      nullptr, 0);
        int ng = 0, ne = 0;
        double etot = 0;
        for (int j = 0; j < o.nsec; ++j) {
          const SecOut& s = o.sec[std::size_t(j)];
          if (s.pdg == 22) {
            ++ng;
            etot += s.ekin;
            seg += s.ekin;
            segg += s.ekin * s.ekin;
            int b = int(s.ekin / kSpecMax * double(kNSpec));
            if (b < 0) { b = 0; }
            if (b >= kNSpec) { b = kNSpec - 1; }
            ++spec[b];
          } else if (s.pdg == 11) {
            ++ne;
          } else {
            // The residual, whatever it is. There is exactly one per capture on both branches.
            sres += s.ekin;
            stime += s.time;
            if (s.excitation > 0.0) { ++isomer; }
          }
        }
        sg += double(ng);
        sgg += double(ng) * double(ng);
        se += double(ne);
        setot += etot;
        ngam_total += ng;
        int mb = ng;
        if (mb < 0) { mb = 0; }
        if (mb >= kNMult) { mb = kNMult - 1; }
        ++mult[mb];
        (void)sel;
      }
      const double mg = sg / kN;
      const double meg = (ngam_total > 0) ? seg / double(ngam_total) : 0.0;
      const double veg = (ngam_total > 0) ? segg / double(ngam_total) - meg * meg : 0.0;
      std::fprintf(f, "%s,%d,%d,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
                   kTargets[t].name, kTargets[t].z, kTargets[t].a, kEnergies[e], kN, mg,
                   sgg / kN - mg * mg, se / kN, meg, veg, setot / kN, sres / kN,
                   double(isomer) / double(kN), stime / kN);
      for (int i = 0; i < kNMult; ++i) { std::fprintf(f, ",%d", mult[i]); }
      for (int i = 0; i < kNSpec; ++i) { std::fprintf(f, ",%d", spec[i]); }
      std::fprintf(f, "\n");
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

void dump_capture(const DumpContext& ctx) {
  dump_masses(ctx);
  dump_det(ctx);
  dump_stat(ctx);
}

}  // namespace

G4GPU_REGISTER_DUMP("capture", "capture_masses.csv capture_det.csv capture_stat.csv",
                    dump_capture);
