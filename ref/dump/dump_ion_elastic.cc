// P25: ionElastic - G4IonElasticPhysics' process on GenericIon - read off the running Geant4.
//
//   ion_elastic_scope.csv   every process on the managers of GenericIon, a real C12 ion, d, t,
//                           He3 and alpha, with each hadronic process's models, their windows
//                           and the process's integral-approach type (`CrossSectionType()`):
//                           which species HAS ionElastic, asked of the constructed QBBC.
//   ion_elastic_xs.csv      G4ComponentGGNuclNuclXsc's elastic and inelastic element cross
//                           sections (a fresh instance - G4IonElasticPhysics makes its own with
//                           `new`) for eighteen real nuclides on every element of B1's four
//                           materials at the NIST mean mass G4CrossSectionElastic hands it,
//                           0.01 MeV/u - 100 GeV/u, ten a decade.
//   ion_elastic_xs_volume.csv  the same eighteen per volume (1/mm) through
//                           G4HadronicProcessStore::GetElasticCrossSectionPerVolume - which maps
//                           a nucleus of baryon number > 4 to GenericIon and so reaches
//                           ionElastic's own data store - in every material g4dump builds.
//   ion_elastic_apply.csv   the REAL G4HadronElasticProcess::PostStepDoIt of ionElastic on a
//                           G4Track of a real ion, under the LCG below, one call per row: the
//                           process is started (StartTracking), its interaction length is asked
//                           at E_start (PostStepGetPhysicalInteractionLength, which sets the
//                           cross section the integral approach rejects against), the track's
//                           energy is set to E_end, the LCG is reset to the row's phase and
//                           PostStepDoIt runs - the integral rejection, SampleZandA,
//                           ChooseHadronicInteraction, G4HadronElastic::ApplyYourself through
//                           G4NuclNuclDiffuseElastic::SampleInvariantT, the recoil threshold
//                           from the proton cut and the rotation to the track's direction. Six
//                           nuclides x B1's four materials x seven energies per nucleon x 16
//                           phases, E_start = E_end on even phases and 1.25 E_end on odd ones. And
//                           the rows where a recoil goes ABOVE the 70 keV cut, which the grid
//                           never reaches (about one call in 1e5-1e6): for six points the phases
//                           from 1,000 up are scanned and the first eight that emit are kept.
//   ion_elastic_hist.csv    the campaign: 20,000 PostStepDoIt calls per point, one LCG stream
//                           per point - draws, sums, recoil species counts and two histograms.
//
// THE ENGINE is dump_bertini.cc's LcgEngine - Knuth's MMIX LCG, top 53 bits - so the port replays
// the same uniforms (tests/test_ion_elastic.cu) and every call is compared one for one.
#include "dump_registry.hh"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "CLHEP/Random/RandomEngine.h"
#include "G4ComponentGGNuclNuclXsc.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4GenericIon.hh"
#include "G4HadronicInteraction.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronicProcessStore.hh"
#include "G4IonTable.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4Navigator.hh"
#include "G4NistManager.hh"
#include "G4Nucleus.hh"
#include "G4ParticleChange.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4TouchableHistory.hh"
#include "G4Track.hh"
#include "G4TransportationManager.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VProcess.hh"
#include "Randomize.hh"

namespace {

class Lcg : public CLHEP::HepRandomEngine {
 public:
  void reset(int phase) {
    s_ = 88172645463325252ull + 1442695040888963407ull * static_cast<unsigned long long>(phase + 1);
    n_ = 0;
  }
  long long draws() const { return static_cast<long long>(n_); }
  double flat() override {
    s_ = 6364136223846793005ull * s_ + 1442695040888963407ull;
    ++n_;
    return (static_cast<double>(s_ >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  }
  void flatArray(const int size, double* v) override { for (int i = 0; i < size; ++i) v[i] = flat(); }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "Lcg"; }

 private:
  unsigned long long s_ = 0, n_ = 0;
};

struct Nuclide { int z, a; };
const Nuclide kXsNuclides[] = {{3, 6},  {3, 7},  {4, 7},  {4, 9},  {5, 10}, {5, 11},
                               {6, 12}, {6, 13}, {7, 13}, {7, 14}, {7, 15}, {8, 15},
                               {8, 16}, {10, 20}, {18, 40}, {20, 40}, {26, 56}, {82, 208}};

G4ParticleDefinition* ion(int z, int a) {
  return G4IonTable::GetIonTable()->GetIon(z, a, 0.0);
}

std::vector<int> elements_of(const DumpContext& ctx, size_t n_mats) {
  std::vector<int> zs;
  for (size_t k = 0; k < ctx.materials.size() && k < n_mats; ++k) {
    const G4Material* m = ctx.materials[k];
    for (size_t i = 0; i < m->GetNumberOfElements(); ++i) {
      const int z = m->GetElement(int(i))->GetZasInt();
      bool seen = false;
      for (int q : zs) { seen = seen || (q == z); }
      if (!seen) { zs.push_back(z); }
    }
  }
  return zs;
}

void dump_scope() {
  FILE* f = std::fopen("ion_elastic_scope.csv", "w");
  std::fprintf(f, "particle,process,subtype,xs_type,model,emin_MeV,emax_MeV\n");
  std::vector<G4ParticleDefinition*> parts = {
      G4GenericIon::GenericIon(), ion(6, 12),
      G4ParticleTable::GetParticleTable()->FindParticle("deuteron"),
      G4ParticleTable::GetParticleTable()->FindParticle("triton"),
      G4ParticleTable::GetParticleTable()->FindParticle("He3"),
      G4ParticleTable::GetParticleTable()->FindParticle("alpha")};
  for (G4ParticleDefinition* p : parts) {
    if (p == nullptr) { continue; }
    G4ProcessVector* all = p->GetProcessManager()->GetProcessList();
    for (G4int i = 0; i < (G4int)all->size(); ++i) {
      G4VProcess* pr = (*all)[i];
      G4HadronicProcess* hp = dynamic_cast<G4HadronicProcess*>(pr);
      if (hp == nullptr || hp->GetHadronicInteractionList().empty()) {
        std::fprintf(f, "%s,%s,%d,-1,(none),0,0\n", p->GetParticleName().c_str(),
                     pr->GetProcessName().c_str(), pr->GetProcessSubType());
        continue;
      }
      for (const G4HadronicInteraction* m : hp->GetHadronicInteractionList()) {
        std::fprintf(f, "%s,%s,%d,%d,%s,%.17g,%.17g\n", p->GetParticleName().c_str(),
                     pr->GetProcessName().c_str(), pr->GetProcessSubType(),
                     (int)hp->CrossSectionType(), m->GetModelName().c_str(),
                     m->GetMinEnergy() / MeV, m->GetMaxEnergy() / MeV);
      }
    }
  }
  std::fclose(f);
}

void dump_xs(const DumpContext& ctx) {
  // B1's four materials are the first four g4dump builds.
  const std::vector<int> zs = elements_of(ctx, 4);
  auto* cs = new G4ComponentGGNuclNuclXsc();
  G4NistManager* nist = G4NistManager::Instance();
  FILE* f = std::fopen("ion_elastic_xs.csv", "w");
  std::fprintf(f, "Zp,Ap,Zt,At,ekin_MeV,elastic_mm2,inelastic_mm2\n");
  for (const Nuclide& n : kXsNuclides) {
    G4ParticleDefinition* p = ion(n.z, n.a);
    for (int z : zs) {
      const double a = nist->GetAtomicMassAmu(z);
      for (int i = -20; i <= 50; ++i) {
        const double e = std::pow(10.0, i / 10.0) * n.a * MeV;
        const double el = cs->GetElasticElementCrossSection(p, e, z, a);
        const double in = cs->GetInelasticElementCrossSection(p, e, z, a);
        std::fprintf(f, "%d,%d,%d,%.17g,%.17g,%.17g,%.17g\n", n.z, n.a, z, a, e / MeV, el / mm2,
                     in / mm2);
      }
    }
  }
  std::fclose(f);

  FILE* g = std::fopen("ion_elastic_xs_volume.csv", "w");
  std::fprintf(g, "material,Z,A,ekin_MeV,elastic_per_mm\n");
  G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
  for (auto* m : ctx.materials) {
    for (const Nuclide& n : kXsNuclides) {
      G4ParticleDefinition* p = ion(n.z, n.a);
      for (int i = -20; i <= 50; ++i) {
        const double e = std::pow(10.0, i / 10.0) * n.a;
        const double el = store->GetElasticCrossSectionPerVolume(p, e * MeV, m) * mm;
        std::fprintf(g, "%s,%d,%d,%.17g,%.17g\n", m->GetName().c_str(), n.z, n.a, e, el);
      }
    }
  }
  std::fclose(g);
}

/// The ionElastic process on GenericIon's manager - the one every real nuclide shares.
G4HadronicProcess* ion_elastic_process() {
  G4ProcessVector* all = G4GenericIon::GenericIon()->GetProcessManager()->GetProcessList();
  for (G4int i = 0; i < (G4int)all->size(); ++i) {
    if ((*all)[i]->GetProcessName() == "ionElastic") {
      return dynamic_cast<G4HadronicProcess*>((*all)[i]);
    }
  }
  return nullptr;
}

/// The centre of g4dump's box of material `m` ("pv<i>"); the world is water too, so the box and
/// not the world is looked up.
G4ThreeVector centre_of(const G4Material* m) {
  for (G4VPhysicalVolume* pv : *G4PhysicalVolumeStore::GetInstance()) {
    if (pv->GetName().rfind("pv", 0) == 0 && pv->GetLogicalVolume()->GetMaterial() == m) {
      return pv->GetTranslation();
    }
  }
  return G4ThreeVector(0, 0, 0);
}

/// A G4Track of `p` at kinetic energy `e`, direction `dir`, standing in material `m`'s box,
/// with a G4Step the particle change's Initialize can read.
struct IonTrack {
  G4Track* track = nullptr;
  G4DynamicParticle* dp = nullptr;
  G4Step* step = nullptr;
  IonTrack(G4ParticleDefinition* p, double e, const G4ThreeVector& dir, const G4Material* m) {
    const G4ThreeVector pos = centre_of(m);
    G4Navigator* nav =
        G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
    nav->LocateGlobalPointAndSetup(pos, nullptr, false, true);
    G4TouchableHandle th(nav->CreateTouchableHistory());
    dp = new G4DynamicParticle(p, dir, e);
    track = new G4Track(dp, 0.0, pos);
    track->SetTouchableHandle(th);
    step = new G4Step();
    track->SetStep(step);
    // G4Track::GetMaterial and GetMaterialCutsCouple read the STEP's pre-step point, so the
    // step is initialised from the track as G4SteppingManager does at the start of a track.
    step->InitializeStep(track);
  }
  ~IonTrack() {
    delete track;  // owns dp
    delete step;
  }
};

const G4ThreeVector kDir = G4ThreeVector(0.3, -0.4, std::sqrt(1.0 - 0.25));

struct CallResult {
  long long draws = 0;
  double efinal = 0, edep = 0, niel = 0;
  int status = 0, nsec = 0, tz = 0, ta = 0, sec_pdg = 0;
  G4ThreeVector dir, sec_dir;
  double sec_ekin = 0;
};

CallResult call_once(G4HadronicProcess* hp, IonTrack& it, Lcg* eng, int phase) {
  CallResult r;
  eng->reset(phase);
  G4VParticleChange* pc = hp->PostStepDoIt(*it.track, *it.step);
  r.draws = eng->draws();
  auto* p = static_cast<G4ParticleChange*>(pc);
  r.efinal = p->GetEnergy();
  r.dir = *p->GetMomentumDirection();
  r.edep = p->GetLocalEnergyDeposit();
  r.niel = p->GetNonIonizingEnergyDeposit();
  r.status = (int)p->GetTrackStatus();
  r.nsec = p->GetNumberOfSecondaries();
  r.tz = hp->GetTargetNucleus()->GetZ_asInt();
  r.ta = hp->GetTargetNucleus()->GetA_asInt();
  if (r.nsec > 0) {
    G4Track* s = p->GetSecondary(0);
    r.sec_pdg = s->GetDefinition()->GetPDGEncoding();
    r.sec_ekin = s->GetKineticEnergy();
    r.sec_dir = s->GetMomentumDirection();
  }
  for (int i = 0; i < r.nsec; ++i) { delete p->GetSecondary(i); }
  p->Clear();
  return r;
}

/// StartTracking, then the interaction length at `e_start` - which is what sets the cross section
/// the integral approach rejects against - and the track's energy then set to `e_end`.
double prepare(G4HadronicProcess* hp, IonTrack& it, const G4Material* m, double e_start,
               double e_end, double* xs_start) {
  it.dp->SetKineticEnergy(e_start);
  hp->StartTracking(it.track);
  G4ForceCondition cond;
  const double len = hp->PostStepGetPhysicalInteractionLength(*it.track, 0.0, &cond);
  // The value the GPIL just stored as theLastCrossSection, asked again of the same store at the
  // same energy for the CSV; PostStepDoIt recomputes its own at E_end before it reads anything.
  *xs_start = hp->GetCrossSectionDataStore()->ComputeCrossSection(it.dp, m);
  it.dp->SetKineticEnergy(e_end);
  return len;
}

void write_row(FILE* f, const G4Material* m, int z, int a, double e_start, double e_end,
               int phase, double xs_start, const CallResult& r) {
  std::fprintf(f, "%s,%d,%d,%.17g,%.17g,%d,%.17g,%lld,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                  "%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g\n",
               m->GetName().c_str(), z, a, e_start / MeV, e_end / MeV, phase, xs_start * mm,
               r.draws, r.efinal / MeV, r.dir.x(), r.dir.y(), r.dir.z(), r.edep / MeV,
               r.niel / MeV, r.status, r.nsec, r.tz, r.ta, r.sec_pdg, r.sec_ekin / MeV,
               r.sec_dir.x(), r.sec_dir.y(), r.sec_dir.z());
}

void dump_apply(const DumpContext& ctx) {
  G4HadronicProcess* hp = ion_elastic_process();
  if (hp == nullptr) {
    std::printf("dump_ion_elastic: no ionElastic on GenericIon's manager\n");
    return;
  }
  Lcg* eng = new Lcg();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);

  const Nuclide nucl[] = {{6, 12}, {8, 16}, {3, 7}, {7, 14}, {5, 10}, {4, 9}};
  const double e_per_n[] = {1.0, 5.0, 20.0, 100.0, 400.0, 1000.0, 4000.0};
  FILE* f = std::fopen("ion_elastic_apply.csv", "w");
  std::fprintf(f, "material,Z,A,e_start_MeV,e_end_MeV,phase,xs_start_per_mm,draws,efinal,dx,dy,dz,"
                  "edep,niel,status,nsec,tz,ta,sec_pdg,sec_ekin,sx,sy,sz\n");
  for (size_t mi = 0; mi < 4 && mi < ctx.materials.size(); ++mi) {
    const G4Material* m = ctx.materials[mi];
    for (const Nuclide& n : nucl) {
      G4ParticleDefinition* p = ion(n.z, n.a);
      for (double epn : e_per_n) {
        const double e_end = epn * n.a * MeV;
        for (int phase = 0; phase < 16; ++phase) {
          const double e_start = (phase % 2 == 0) ? e_end : 1.25 * e_end;
          IonTrack it(p, e_start, kDir, m);
          double xs_start = 0.0;
          const double len = prepare(hp, it, m, e_start, e_end, &xs_start);
          // A zero cross section at the step's start is an infinite length: Geant4 never calls
          // PostStepDoIt for it, so neither does this dump.
          if (!(len < DBL_MAX)) { continue; }
          const CallResult r = call_once(hp, it, eng, phase);
          write_row(f, m, n.z, n.a, e_start, e_end, phase, xs_start, r);
        }
      }
    }
  }

  // THE RECOIL ABOVE THE CUT, found rather than hoped for. The Coulomb arm's mu = (1-cos)/2 is
  // screened Rutherford, `muC*r*Am/(Am + muC*(1-r))`, so P(mu > x) is about Am/x - and with Am of
  // order 1e-9 a recoil above the proton cut's 70 keV is a one-in-a-million call: the grid above
  // emits none in 2,680. So for six points the phases are SCANNED, from 1,000 up, and the first
  // eight whose PostStepDoIt emitted a recoil are kept as rows of the same file - the branch
  // where the recoil becomes a track, replayed like every other row. A point that finds none in
  // the cap writes none, and the test counts the rows it gets.
  struct Scan { int mi, z, a; double epn; };
  const Scan scans[] = {{1, 6, 12, 5.0}, {1, 3, 7, 5.0},  {3, 8, 16, 20.0},
                        {0, 6, 12, 5.0}, {2, 7, 14, 5.0}, {1, 6, 12, 20.0}};
  for (const Scan& sc : scans) {
    if (sc.mi >= (int)ctx.materials.size()) { continue; }
    const G4Material* m = ctx.materials[sc.mi];
    G4ParticleDefinition* p = ion(sc.z, sc.a);
    const double e = sc.epn * sc.a * MeV;
    IonTrack it(p, e, kDir, m);
    double xs_start = 0.0;
    const double len = prepare(hp, it, m, e, e, &xs_start);
    if (!(len < DBL_MAX)) { continue; }
    int hits = 0;
    long long tried = 0;
    for (int phase = 1000; phase < 1000 + 20000000 && hits < 8; ++phase) {
      const CallResult r = call_once(hp, it, eng, phase);
      ++tried;
      if (r.nsec == 0) { continue; }
      write_row(f, m, sc.z, sc.a, e, e, phase, xs_start, r);
      ++hits;
    }
    std::printf("dump_ion_elastic: %s Z=%d A=%d %g MeV/u: %d recoils above the cut in %lld "
                "calls\n", m->GetName().c_str(), sc.z, sc.a, sc.epn, hits, tried);
  }
  std::fclose(f);

  // The campaign: 20,000 PostStepDoIt calls per point on one stream, the track unchanged between
  // calls (a particle change is a proposal; nothing applies it here), E_start = E_end.
  struct Point { int mi, z, a; double epn; };
  const Point pts[] = {{1, 6, 12, 1000.}, {1, 6, 12, 100.}, {0, 6, 12, 1000.},
                       {3, 8, 16, 300.},  {2, 3, 7, 50.},   {1, 8, 16, 20.}};
  FILE* h = std::fopen("ion_elastic_hist.csv", "w");
  std::fprintf(h, "material,Z,A,ekin_MeV,n,draws,n_sec,n_sec_proton,n_sec_ion,n_below,"
                  "sum_efinal_MeV,sum_erec_MeV,sum_edep_MeV");
  for (int b = 0; b < 40; ++b) { std::fprintf(h, ",er%d", b); }
  for (int b = 0; b < 40; ++b) { std::fprintf(h, ",th%d", b); }
  std::fprintf(h, "\n");
  for (const Point& q : pts) {
    if (q.mi >= (int)ctx.materials.size()) { continue; }
    const G4Material* m = ctx.materials[q.mi];
    G4ParticleDefinition* p = ion(q.z, q.a);
    const double e = q.epn * q.a * MeV;
    IonTrack it(p, e, kDir, m);
    double xs_unused = 0.0;
    prepare(hp, it, m, e, e, &xs_unused);
    eng->reset(11);
    long long er[40] = {0}, th[40] = {0};
    long long n_sec = 0, n_p = 0, n_ion = 0, n_below = 0;
    double sum_ef = 0, sum_er = 0, sum_ed = 0;
    const int N = 20000;
    for (int i = 0; i < N; ++i) {
      G4VParticleChange* pc = hp->PostStepDoIt(*it.track, *it.step);
      auto* pp = static_cast<G4ParticleChange*>(pc);
      sum_ef += pp->GetEnergy();
      sum_ed += pp->GetLocalEnergyDeposit();
      // The lab polar angle against the incoming direction, as log10(1 - cos) in 40 bins
      // over [-16, 0): the Coulomb arm's angles are milliradians and below.
      const double c = pp->GetMomentumDirection()->dot(kDir);
      const double omc = 1.0 - c;
      int bt = (omc > 0.0) ? int((std::log10(omc) + 16.0) * 2.5) : 0;
      bt = bt < 0 ? 0 : (bt > 39 ? 39 : bt);
      ++th[bt];
      if (pp->GetNumberOfSecondaries() > 0) {
        G4Track* s = pp->GetSecondary(0);
        ++n_sec;
        if (s->GetDefinition()->GetPDGEncoding() == 2212) { ++n_p; } else { ++n_ion; }
        const double ek = s->GetKineticEnergy();
        sum_er += ek;
        // log10(E/MeV) over [-2, 4), 40 bins.
        int b = int((std::log10(ek) + 2.0) * (40.0 / 6.0));
        b = b < 0 ? 0 : (b > 39 ? 39 : b);
        ++er[b];
        for (int k = 0; k < pp->GetNumberOfSecondaries(); ++k) { delete pp->GetSecondary(k); }
      } else if (pp->GetLocalEnergyDeposit() > 0.0) {
        ++n_below;
      }
      pp->Clear();
    }
    std::fprintf(h, "%s,%d,%d,%.17g,%d,%lld,%lld,%lld,%lld,%lld,%.17g,%.17g,%.17g",
                 m->GetName().c_str(), q.z, q.a, e / MeV, N, eng->draws(), n_sec, n_p, n_ion,
                 n_below, sum_ef / MeV, sum_er / MeV, sum_ed / MeV);
    for (int b = 0; b < 40; ++b) { std::fprintf(h, ",%lld", er[b]); }
    for (int b = 0; b < 40; ++b) { std::fprintf(h, ",%lld", th[b]); }
    std::fprintf(h, "\n");
  }
  std::fclose(h);
  CLHEP::HepRandom::setTheEngine(saved);
}

void dump_ion_elastic(const DumpContext& ctx) {
  dump_scope();
  dump_xs(ctx);
  dump_apply(ctx);
}

}  // namespace

G4GPU_REGISTER_DUMP("ion_elastic",
                    "ion_elastic_scope.csv ion_elastic_xs.csv ion_elastic_xs_volume.csv "
                    "ion_elastic_apply.csv ion_elastic_hist.csv",
                    dump_ion_elastic);
