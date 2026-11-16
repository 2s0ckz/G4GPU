// P23c: the anti-nucleon family's cross sections and elastic model, read off the running Geant4.
//
//   anti_xs.csv           G4ComponentAntiNuclNuclearXS ("AntiAGlauber", the registry's own
//                         instance) for the six G4HadParticles::GetLightAntiIons() species x the
//                         elements of B1's four materials x 1 keV-100 TeV (10 a decade), at the
//                         NIST mean mass G4CrossSectionElastic/Inelastic hand it: total,
//                         inelastic, elastic (mm^2) and the two anti-nucleon-nucleon numbers (mb).
//   anti_xs_volume.csv    hadElastic and <name>Inelastic per volume (1/mm) for the same six in
//                         the four materials, through G4HadronicProcessStore - the wiring.
//   anti_elastic.csv      G4AntiNuclElastic::SampleInvariantT's deterministic pieces (read through
//                         `#define private public` after one call) and its first 32 samples under
//                         the LCG engine below, per (species, target, kinetic energy).
//   anti_elastic_hist.csv 20,000 G4HadronElastic::ApplyYourself calls through G4AntiNuclElastic
//                         per point - pbar on C12 and Pb208 at 300 MeV, 1 GeV and 10 GeV, plus
//                         anti-d and anti-alpha on C12 at 1 GeV/n - one LCG stream per point: the
//                         lab cos(theta) histogram (50 bins), the draw count, and sum t.
//   anti_scope.csv        strange_scope.csv's rows for the six: every process on each manager
//                         and every hadronic model with its window (test_strange_tracks).
//
// THE ENGINE is dump_bertini.cc's LcgEngine - Knuth's MMIX LCG, top 53 bits - so the port replays
// the same uniforms (tests/test_anti_nucleon.cu) and every sample is compared one for one.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "CLHEP/Random/RandomEngine.h"
#include "G4CrossSectionDataSetRegistry.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4DecayTable.hh"
#include "G4HadronicInteraction.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronicProcessStore.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4VProcess.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Nucleus.hh"
#include "G4NucleiProperties.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#define private public
#define protected public
#include "G4AntiNuclElastic.hh"
#include "G4ComponentAntiNuclNuclearXS.hh"
#undef private
#undef protected

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

const int kSpecies[] = {-2212, -2112, -1000010020, -1000010030, -1000020030, -1000020040};

G4ParticleDefinition* find_pdg(int pdg) {
  return G4ParticleTable::GetParticleTable()->FindParticle(pdg);
}

G4ComponentAntiNuclNuclearXS* component() {
  auto* c = static_cast<G4ComponentAntiNuclNuclearXS*>(
      G4CrossSectionDataSetRegistry::Instance()->GetComponentCrossSection("AntiAGlauber"));
  if (c == nullptr) { c = new G4ComponentAntiNuclNuclearXS(); }
  return c;
}

void dump_xs(const DumpContext& ctx) {
  std::vector<int> zs;
  for (auto* m : ctx.materials) {
    for (size_t i = 0; i < m->GetNumberOfElements(); ++i) {
      const int z = m->GetElement(int(i))->GetZasInt();
      bool seen = false;
      for (int q : zs) { seen = seen || (q == z); }
      if (!seen) { zs.push_back(z); }
    }
  }
  G4ComponentAntiNuclNuclearXS* cs = component();
  G4NistManager* nist = G4NistManager::Instance();
  FILE* f = std::fopen("anti_xs.csv", "w");
  std::fprintf(f, "pdg,Z,A,ekin_MeV,total_mm2,inelastic_mm2,elastic_mm2,nn_total_mb,nn_elastic_mb\n");
  for (int pdg : kSpecies) {
    G4ParticleDefinition* p = find_pdg(pdg);
    if (p == nullptr) { continue; }
    for (int z : zs) {
      const double a = nist->GetAtomicMassAmu(z);
      for (int i = -30; i <= 80; ++i) {
        const double e = std::pow(10.0, i / 10.0) * MeV;
        const double tot = cs->GetTotalElementCrossSection(p, e, z, a);
        const double in = cs->GetInelasticElementCrossSection(p, e, z, a);
        const double el = cs->GetElasticElementCrossSection(p, e, z, a);
        const double nt = cs->GetAntiHadronNucleonTotCrSc(p, e);
        const double ne = cs->GetAntiHadronNucleonElCrSc(p, e);
        std::fprintf(f, "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", pdg, z, a, e / MeV,
                     tot / mm2, in / mm2, el / mm2, nt, ne);
      }
    }
  }
  std::fclose(f);

  FILE* g = std::fopen("anti_xs_volume.csv", "w");
  std::fprintf(g, "material,pdg,ekin_MeV,elastic_per_mm,inelastic_per_mm\n");
  G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
  for (auto* m : ctx.materials) {
    for (int pdg : kSpecies) {
      G4ParticleDefinition* p = find_pdg(pdg);
      if (p == nullptr) { continue; }
      for (int i = -30; i <= 80; ++i) {
        const double e = std::pow(10.0, i / 10.0);
        const double el = store->GetElasticCrossSectionPerVolume(p, e * MeV, m) * mm;
        const double in = store->GetInelasticCrossSectionPerVolume(p, e * MeV, m) * mm;
        std::fprintf(g, "%s,%d,%.17g,%.17g,%.17g\n", m->GetName().c_str(), pdg, e, el, in);
      }
    }
  }
  std::fclose(g);
}

struct Target { int z, a; };
const Target kTargets[] = {{1, 1}, {1, 2}, {2, 4}, {6, 12}, {8, 16}, {20, 40}, {82, 208}};
const double kEkin[] = {100.05, 150.0, 300.0, 1000.0, 3000.0, 10000.0, 100000.0};

void dump_elastic(const DumpContext&) {
  Lcg* eng = new Lcg();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);
  auto* model = new G4AntiNuclElastic();

  FILE* f = std::fopen("anti_elastic.csv", "w");
  std::fprintf(f, "pdg,Z,A,ekin_MeV,plab_MeV,ctet1,ptot,tmax,beta,n,am,ra,ref,ceff,draws32");
  for (int k = 0; k < 32; ++k) { std::fprintf(f, ",t%d", k); }
  std::fprintf(f, "\n");
  for (int pdg : kSpecies) {
    G4ParticleDefinition* p = find_pdg(pdg);
    const int ab = std::abs(p->GetBaryonNumber());
    for (const Target& t : kTargets) {
      for (double ek : kEkin) {
        const double e = ek * ab * MeV;  // per nucleon, so every species meets the same rows
        const double m = p->GetPDGMass();
        const double plab = std::sqrt(e * (e + 2.0 * m));
        eng->reset(0);
        model->fRa = model->fRef = model->fceff = model->fAm = model->fZommerfeld = 0.0;
        model->fBeta = 0.0;
        double ts[32];
        ts[0] = model->SampleInvariantT(p, plab, t.z, t.a);
        for (int k = 1; k < 32; ++k) { ts[k] = model->SampleInvariantT(p, plab, t.z, t.a); }
        const double ra = model->fRa, ref = model->fRef, ceff = model->fceff;
        const double am = model->fAm, n = model->fZommerfeld, beta = model->fBeta;
        const double ptot = model->fptot, tmax = model->fTmax;
        std::fprintf(f, "%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%lld",
                     pdg, t.z, t.a, e / MeV, plab / MeV, model->GetcosTeta1(plab, t.a), ptot, tmax,
                     beta, n, am, ra, ref, ceff, eng->draws());
        for (int k = 0; k < 32; ++k) { std::fprintf(f, ",%.17g", ts[k] / (MeV * MeV)); }
        std::fprintf(f, "\n");
      }
    }
  }
  std::fclose(f);

  // The campaign: whole ApplyYourself calls, one stream per point.
  struct Point { int pdg, z, a; double ekin_per_n; };
  const Point pts[] = {{-2212, 6, 12, 300.}, {-2212, 6, 12, 1000.}, {-2212, 6, 12, 10000.},
                       {-2212, 82, 208, 300.}, {-2212, 82, 208, 1000.}, {-2212, 82, 208, 10000.},
                       {-1000010020, 6, 12, 1000.}, {-1000020040, 6, 12, 1000.}};
  FILE* h = std::fopen("anti_elastic_hist.csv", "w");
  std::fprintf(h, "pdg,Z,A,ekin_MeV,n,draws,sum_t_MeV2,sum_efinal_MeV");
  for (int b = 0; b < 50; ++b) { std::fprintf(h, ",b%d", b); }
  std::fprintf(h, "\n");
  model->SetRecoilEnergyThreshold(0.0);
  for (const Point& q : pts) {
    G4ParticleDefinition* p = find_pdg(q.pdg);
    const double e = q.ekin_per_n * std::abs(p->GetBaryonNumber()) * MeV;
    G4DynamicParticle dp(p, G4ThreeVector(0, 0, 1), e);
    G4HadProjectile proj(dp);
    G4Nucleus nucleus(q.a, q.z);
    long long hist[50] = {0};
    double sum_t = 0, sum_ef = 0;
    eng->reset(7);
    const int N = 20000;
    for (int i = 0; i < N; ++i) {
      G4HadFinalState* r = model->ApplyYourself(proj, nucleus);
      const double c = r->GetMomentumChange().z();
      int b = int((c + 1.0) * 25.0);
      if (b < 0) b = 0;
      if (b > 49) b = 49;
      ++hist[b];
      sum_ef += r->GetEnergyChange();
      if (r->GetNumberOfSecondaries() > 0) {
        sum_t += 2.0 * G4NucleiProperties::GetNuclearMass(q.a, q.z) *
                 r->GetSecondary(0)->GetParticle()->GetKineticEnergy();
      } else {
        sum_t += 2.0 * G4NucleiProperties::GetNuclearMass(q.a, q.z) * r->GetLocalEnergyDeposit();
      }
      r->Clear();
    }
    std::fprintf(h, "%d,%d,%d,%.17g,%d,%lld,%.17g,%.17g", q.pdg, q.z, q.a, e / MeV, N,
                 eng->draws(), sum_t, sum_ef);
    for (int b = 0; b < 50; ++b) { std::fprintf(h, ",%lld", hist[b]); }
    std::fprintf(h, "\n");
  }
  std::fclose(h);
  CLHEP::HepRandom::setTheEngine(saved);
}

/// strange_scope.csv's layout (dump_strange.cc) for the six anti species, so test_strange_tracks
/// reads them with the same parser: every registered process and, for a hadronic one, every
/// model with its window. `bertini_applicable` is -1 - not asked; no anti row registers Bertini.
void dump_scope() {
  FILE* f = std::fopen("anti_scope.csv", "w");
  std::fprintf(f, "species,pdg,mass_MeV,width_MeV,lifetime_ns,stable,shortlived,charge,baryon,"
                  "n_decay_channels,bertini_applicable,process,process_type,process_subtype,"
                  "atrest_index,model,emin_MeV,emax_MeV\n");
  for (int pdg : kSpecies) {
    G4ParticleDefinition* p = find_pdg(pdg);
    const G4DecayTable* dt = p->GetDecayTable();
    char head[512];
    std::snprintf(head, sizeof head, "%s,%d,%.17g,%.17g,%.17g,%d,%d,%.17g,%d,%d,%d",
                  p->GetParticleName().c_str(), pdg, p->GetPDGMass() / MeV,
                  p->GetPDGWidth() / MeV, p->GetPDGLifeTime() / ns, p->GetPDGStable() ? 1 : 0,
                  p->IsShortLived() ? 1 : 0, p->GetPDGCharge() / eplus, p->GetBaryonNumber(),
                  (dt != nullptr) ? dt->entries() : 0, -1);
    G4ProcessManager* pm = p->GetProcessManager();
    G4ProcessVector* all = pm->GetProcessList();
    G4ProcessVector* rest = pm->GetAtRestProcessVector();
    for (G4int i = 0; i < (G4int)all->size(); ++i) {
      G4VProcess* pr = (*all)[i];
      int atrest = -1;
      if (rest != nullptr) {
        for (G4int k = 0; k < (G4int)rest->size(); ++k) {
          if ((*rest)[k] == pr) { atrest = k; }
        }
      }
      G4HadronicProcess* hp = dynamic_cast<G4HadronicProcess*>(pr);
      if (hp == nullptr || hp->GetHadronicInteractionList().empty()) {
        std::fprintf(f, "%s,%s,%d,%d,%d,(none),0,0\n", head, pr->GetProcessName().c_str(),
                     (int)pr->GetProcessType(), pr->GetProcessSubType(), atrest);
        continue;
      }
      for (const G4HadronicInteraction* m : hp->GetHadronicInteractionList()) {
        std::fprintf(f, "%s,%s,%d,%d,%d,%s,%.17g,%.17g\n", head, pr->GetProcessName().c_str(),
                     (int)pr->GetProcessType(), pr->GetProcessSubType(), atrest,
                     m->GetModelName().c_str(), m->GetMinEnergy() / MeV,
                     m->GetMaxEnergy() / MeV);
      }
    }
  }
  std::fclose(f);
}

void dump_anti_nucleon(const DumpContext& ctx) {
  dump_scope();
  dump_xs(ctx);
  dump_elastic(ctx);
}

}  // namespace

G4GPU_REGISTER_DUMP("anti_nucleon",
                    "anti_xs.csv anti_xs_volume.csv anti_elastic.csv anti_elastic_hist.csv "
                    "anti_scope.csv",
                    dump_anti_nucleon);
