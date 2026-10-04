// Oracle for P23b - the strange projectiles QBBC builds processes for, measured on the running
// physics list rather than read off its builders.
//
// Six files:
//
//   strange_scope.csv     every hyperon, anti-hyperon, kaon and b/c hadron QBBC can meet: its
//                         definition, its decay table's size, Bertini's own IsApplicable, and
//                         one row per (process, model) on its process manager with the energy
//                         window the CONSTRUCTED process asks that model for. This is the scope
//                         table of docs/RISK.md V226 as a measurement.
//   strange_scope_xs.csv  the inelastic and elastic cross section per atom the REGISTERED
//                         processes return, beside G4ComponentGGHadronNucleusXsc's own answer at
//                         the same point. Equal means the process carries the Glauber-Gribov
//                         component and nothing else - which is what G4HadronicBuilder writes
//                         and what this file proves.
//   strange_hnxsc.csv     G4HadronNucleonXsc::HadronNucleonXsc - the dispatcher - for every
//                         species above on a proton and a neutron, 1 keV to 100 TeV.
//   strange_ggxs.csv      G4ComponentGGHadronNucleusXsc::ComputeCrossSections, all five
//                         components, for every species above that has an inelastic process.
//   strange_k0type.csv    the two neutral-kaon coin tosses of the Bertini interface under a
//                         prescribed deviate: G4InuclElementaryParticle::type(K0S/K0L) on the way
//                         in and G4CascadeInterface::makeDynamicParticle(K0/anti-K0) on the way
//                         out, with the number of deviates each consumed.
//   strange_lund.csv      the charm and bottom pair-creation probabilities of the
//                         G4LundStringFragmentation QBBC's FTFP builds, read off the object.
//
// WHY A SEPARATE DUMP AND NOT ROWS IN P2'S. `had_hnxsc.csv` and `had_ggcomp.csv` are read by
// tests/test_hadronic_xs.cu through a name map that fails on a name it does not know; P11c met
// the same wall (dump_ftf.cc's `dump_hyperon_xsc`). A file of its own, read by a test of its own,
// changes nothing another package compares.
#include "dump_registry.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "G4CascadeInterface.hh"
#include "G4ComponentGGHadronNucleusXsc.hh"
#include "G4DecayTable.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4HadronNucleonXsc.hh"
#include "G4HadronicInteraction.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronicProcessStore.hh"
#include "G4HadronicParameters.hh"
#include "G4InuclElementaryParticle.hh"
#include "G4InuclParticleNames.hh"
#include "G4KaonZeroLong.hh"
#include "G4KaonZeroShort.hh"
#include "G4LorentzVector.hh"
#include "G4LundStringFragmentation.hh"
#include "G4Material.hh"
#include "G4Neutron.hh"
#include "G4NistManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"
#include "G4VProcess.hh"
#include "Randomize.hh"

namespace {

const double kMm2 = mm * mm;

/// Every species this package is about, by PDG code, in the order of G4HadParticles.cc: the
/// kaons, the six hyperons G4HadParticles::sHyperons lists and the sigma0 it leaves out, the six
/// anti-hyperons and the anti-sigma0, the thirty b/c hadrons of sBCHadrons, and the six
/// heavy-flavour states that list leaves out "because they decay very quickly" together with the
/// three BuildDecayTableForBCHadrons gives a table anyway (etac, J/psi, Upsilon). The strong K0
/// and anti-K0 are here because G4CascadeInterface turns a K0S or a K0L into one of them.
const int kScopeSpecies[] = {
    321, -321, 310, 130, 311, -311,
    3122, 3222, 3112, 3322, 3312, 3334, 3212,
    -3122, -3222, -3112, -3322, -3312, -3334, -3212,
    411, 421, -411, -421, 431, -431,
    521, 511, -521, -511, 531, -531, 541, -541,
    4122, 4232, 4132, 4332, 5122, 5232, 5132, 5332,
    -4122, -4232, -4132, -4332, -5122, -5232, -5132, -5332,
    4222, 4212, 4112, 5222, 5212, 5112, -4222, -4212, -4112, -5222, -5212, -5112,
    441, 443, 553};

/// The species the cross-section files are written for: everything above that QBBC gives an
/// inelastic process, which strange_scope.csv shows is the kaons, sHyperons, sAntiHyperons and
/// sBCHadrons - and NOT sigma0, anti-sigma0, the sigma_c/sigma_b states or the c-cbar/b-bbar
/// mesons. K+ and K- are P2's (had_ggcomp.csv) and are not repeated.
const int kXsSpecies[] = {
    310, 130,
    3122, 3222, 3112, 3322, 3312, 3334,
    -3122, -3222, -3112, -3322, -3312, -3334,
    411, 421, -411, -421, 431, -431,
    521, 511, -521, -511, 531, -531, 541, -541,
    4122, 4232, 4132, 4332, 5122, 5232, 5132, 5332,
    -4122, -4232, -4132, -4332, -5122, -5232, -5132, -5332};

/// The strange ones get every Z the existing GG test uses (1..92); the thirty b/c hadrons share
/// one formula per family and get a twelve-element subset that still reaches hydrogen, the
/// light elements of tissue and water, the structural metals and the heaviest element.
bool every_z(int pdg) {
  const int a = std::abs(pdg);
  return a == 310 || a == 130 || (a > 3000 && a < 4000);
}
const int kZSubset[] = {1, 2, 6, 7, 8, 13, 20, 26, 29, 47, 82, 92};

G4ParticleDefinition* find_pdg(int pdg) {
  return G4ParticleTable::GetParticleTable()->FindParticle(pdg);
}

std::vector<double> decade_scan(double e0, double e1, int per_decade) {
  std::vector<double> v;
  const double step = std::pow(10.0, 1.0 / per_decade);
  for (double e = e0; e <= e1 * 1.0000000001; e *= step) { v.push_back(e); }
  return v;
}

void add_and_sort(std::vector<double>& v, const std::vector<double>& extra, double lo,
                  double hi) {
  for (double e : extra) {
    if (e >= lo && e <= hi) { v.push_back(e); }
  }
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

/// A HepRandomEngine that answers one fixed value for every draw and counts the draws.
class FixedValueEngine : public CLHEP::HepRandomEngine {
 public:
  void set(double v) { v_ = v; n_ = 0; }
  long long draws() const { return n_; }
  double flat() override { ++n_; return v_; }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "FixedValueEngine"; }

 private:
  double v_ = 0.5;
  long long n_ = 0;
};

/// G4CascadeInterface::makeDynamicParticle is protected; this re-exports the one overload the
/// K0 mixing lives in, the same way dump_decay.cc re-exports G4Decay's lengths.
class CascadeProbe : public G4CascadeInterface {
 public:
  using G4CascadeInterface::makeDynamicParticle;
};

// ---------------------------------------------------------------------------------------------
void dump_scope(G4CascadeInterface* bert) {
  FILE* f = std::fopen("strange_scope.csv", "w");
  std::fprintf(f, "species,pdg,mass_MeV,width_MeV,lifetime_ns,stable,shortlived,charge,baryon,"
                  "n_decay_channels,bertini_applicable,process,process_type,process_subtype,"
                  "atrest_index,model,emin_MeV,emax_MeV\n");
  for (int pdg : kScopeSpecies) {
    G4ParticleDefinition* p = find_pdg(pdg);
    if (p == nullptr) {
      std::fprintf(f, "(undefined),%d,0,0,0,0,0,0,0,0,0,(none),0,0,-1,(none),0,0\n", pdg);
      continue;
    }
    const G4DecayTable* dt = p->GetDecayTable();
    // IsApplicable on the PARTICLE DEFINITION, which is what G4EnergyRangeManager reaches
    // through the G4HadProjectile overload. For K0S and K0L it draws a deviate - both answers it
    // can draw (kaonZero, kaonZeroBar) have a channel table, so the answer does not depend on it.
    const int bert_ok = bert->IsApplicable(p) ? 1 : 0;
    char head[512];
    std::snprintf(head, sizeof head, "%s,%d,%.17g,%.17g,%.17g,%d,%d,%.17g,%d,%d,%d",
                  p->GetParticleName().c_str(), pdg, p->GetPDGMass() / MeV,
                  p->GetPDGWidth() / MeV, p->GetPDGLifeTime() / ns,
                  p->GetPDGStable() ? 1 : 0, p->IsShortLived() ? 1 : 0,
                  p->GetPDGCharge() / eplus, p->GetBaryonNumber(),
                  (dt != nullptr) ? dt->entries() : 0, bert_ok);
    G4ProcessManager* pm = p->GetProcessManager();
    if (pm == nullptr) {
      std::fprintf(f, "%s,(no process manager),0,0,-1,(none),0,0\n", head);
      continue;
    }
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
        std::fprintf(f, "%s,%s,%d,%d,%d,%s,%.17g,%.17g\n", head,
                     pr->GetProcessName().c_str(), (int)pr->GetProcessType(),
                     pr->GetProcessSubType(), atrest, m->GetModelName().c_str(),
                     m->GetMinEnergy() / MeV, m->GetMaxEnergy() / MeV);
      }
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// The registered processes' cross sections against the bare component. G4HadronicProcessStore
// reaches `G4HadronicProcess::GetElementCrossSection`, which is the data store the process
// was built with, so a second data set, a scale factor or a different component would show up
// here as a difference - and a zero for a species with no process.
void dump_scope_xs() {
  FILE* f = std::fopen("strange_scope_xs.csv", "w");
  std::fprintf(f, "species,pdg,Z,A_amu,energy_MeV,process_inelastic_mm2,gg_inelastic_mm2,"
                  "process_elastic_mm2,gg_elastic_mm2\n");
  G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
  G4NistManager* nist = G4NistManager::Instance();
  G4ComponentGGHadronNucleusXsc gg;
  // G4HadronicProcess::GetElementCrossSection needs a material to ask the data store with and,
  // handed none, looks for a single-element material of that Z and gives up with had066 when
  // there is none. The NIST elemental materials are what it would find in a run that had them.
  const int zs[] = {1, 6, 8, 13, 26, 82};
  const char* const mats[] = {"G4_H", "G4_C", "G4_O", "G4_Al", "G4_Fe", "G4_Pb"};
  const double es[] = {0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0};
  for (int pdg : kXsSpecies) {
    G4ParticleDefinition* p = find_pdg(pdg);
    if (p == nullptr) { continue; }
    for (int k = 0; k < 6; ++k) {
      const int z = zs[k];
      const G4Material* mat = nist->FindOrBuildMaterial(mats[k]);
      const G4Element* el = mat->GetElement(0);
      const double a = nist->GetAtomicMassAmu(z);
      for (double e : es) {
        const double pin = store->GetInelasticCrossSectionPerAtom(p, e * MeV, el, mat);
        const double gin = gg.GetInelasticElementCrossSection(p, e * MeV, z, a);
        const double pel = store->GetElasticCrossSectionPerAtom(p, e * MeV, el, mat);
        const double gel = gg.GetElasticElementCrossSection(p, e * MeV, z, a);
        std::fprintf(f, "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     p->GetParticleName().c_str(), pdg, z, a, e, pin / kMm2, gin / kMm2,
                     pel / kMm2, gel / kMm2);
      }
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// G4HadronNucleonXsc's dispatcher on both nucleons. The energies are P2's: twelve per decade from
// 1 keV to 100 TeV plus every pLab branch point of HadronNucleonXscNS and HadronNucleonXscPDG
// converted to a kinetic energy for THIS particle - and, because SCBMesonNucleonXscNS and
// HyperonNucleonXscNS evaluate a pi+ and a proton at the projectile's KINETIC energy, the same
// branch points converted for the pi+ and the proton masses as well, so that the substituted
// particle's branches are crossed too.
void dump_hnxsc() {
  FILE* f = std::fopen("strange_hnxsc.csv", "w");
  std::fprintf(f, "particle,pdg,nucleon,energy_MeV,total_mm2,elastic_mm2,inelastic_mm2\n");
  const double plabs[] = {0.02,  0.1,   0.28,  0.38,  0.395676, 0.4,  0.48, 0.5,
                          0.631, 0.65,  0.68,  0.72,  0.73,     0.77, 0.78, 0.8,
                          0.85,  0.88,  0.94,  0.95,  0.98,     1.01, 1.03, 1.05,
                          1.15,  1.3,   1.4,   1.63,  2.0,      2.1,  3.5,  10.0,
                          100.0, 373.0, 1000.0};
  // Every species in the scope table that the dispatcher can be handed, plus eta and eta' -
  // they are in SCBMesonNucleonXscNS's dispatch list (221, 331), which is the only reason they
  // are here.
  std::vector<int> species(std::begin(kScopeSpecies), std::end(kScopeSpecies));
  species.push_back(221);
  species.push_back(331);
  const G4ParticleDefinition* nucleons[] = {G4Proton::Proton(), G4Neutron::Neutron()};
  const double masses[] = {G4Proton::Proton()->GetPDGMass(),
                           find_pdg(211)->GetPDGMass()};
  G4HadronNucleonXsc hn;
  for (int pdg : species) {
    // K+ and K- are P2's.
    if (pdg == 321 || pdg == -321) { continue; }
    G4ParticleDefinition* p = find_pdg(pdg);
    if (p == nullptr) { continue; }
    std::vector<double> es = decade_scan(1e-3, 1e8, 12);
    std::vector<double> extra = {0.1, 100.0};
    for (double pl : plabs) {
      const double pp = pl * GeV;
      const double m = p->GetPDGMass();
      extra.push_back((std::sqrt(pp * pp + m * m) - m) / MeV);
      for (double ms : masses) { extra.push_back((std::sqrt(pp * pp + ms * ms) - ms) / MeV); }
    }
    add_and_sort(es, extra, 1e-3, 1e8);
    for (const G4ParticleDefinition* n : nucleons) {
      for (double e : es) {
        hn.HadronNucleonXsc(p, n, e * MeV);
        std::fprintf(f, "%s,%d,%s,%.17g,%.17g,%.17g,%.17g\n", p->GetParticleName().c_str(),
                     pdg, n->GetParticleName().c_str(), e,
                     hn.GetTotalHadronNucleonXsc() / kMm2,
                     hn.GetElasticHadronNucleonXsc() / kMm2,
                     hn.GetInelasticHadronNucleonXsc() / kMm2);
      }
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// The Glauber-Gribov hadron-nucleus component, every one of its five outputs, on P2's grid for
// had_ggcomp.csv: twelve per decade from 100 keV to 100 TeV plus the hand-over energies, with A
// the rounded NIST mean mass as G4CrossSectionInelastic passes it (through the G4double-A
// overload's G4lrint).
void dump_ggxs() {
  FILE* f = std::fopen("strange_ggxs.csv", "w");
  std::fprintf(f, "particle,pdg,Z,A,energy_MeV,total_mm2,inelastic_mm2,elastic_mm2,"
                  "production_mm2,diffraction_mm2\n");
  G4NistManager* nist = G4NistManager::Instance();
  G4ComponentGGHadronNucleusXsc gg;
  std::vector<double> es = decade_scan(1e-1, 1e8, 12);
  add_and_sort(es, {14.0, 20.0, 91000.0, 20000.0, 100.0}, 1e-1, 1e8);
  for (int pdg : kXsSpecies) {
    G4ParticleDefinition* p = find_pdg(pdg);
    if (p == nullptr) { continue; }
    std::vector<int> zs;
    if (every_z(pdg)) {
      for (int z = 1; z <= 92; ++z) { zs.push_back(z); }
    } else {
      zs.assign(std::begin(kZSubset), std::end(kZSubset));
    }
    for (int z : zs) {
      const int a = G4lrint(nist->GetAtomicMassAmu(z));
      for (double e : es) {
        gg.ComputeCrossSections(p, e * MeV, z, a);
        std::fprintf(f, "%s,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     p->GetParticleName().c_str(), pdg, z, a, e,
                     gg.GetTotalGlauberGribovXsc() / kMm2,
                     gg.GetInelasticGlauberGribovXsc() / kMm2,
                     gg.GetElasticGlauberGribovXsc() / kMm2,
                     gg.GetProductionGlauberGribovXsc() / kMm2,
                     gg.GetDiffractionGlauberGribovXsc() / kMm2);
      }
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// The two coin tosses, under a deviate fixed to each of a set of values. The values straddle
// 0.5 by one ulp on each side and sit on it exactly, because both Geant4 lines are `> 0.5` - so
// 0.5 itself must go to the SECOND outcome in both places, and a port that wrote `>=` or swapped
// the two outcomes fails on exactly one row.
void dump_k0type() {
  FILE* f = std::fopen("strange_k0type.csv", "w");
  std::fprintf(f, "what,pdg_in,u,result,draws\n");
  const double us[] = {0.05, 0.25, 0.4999999999999999, 0.5, 0.5000000000000001, 0.75, 0.95};
  FixedValueEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  // On the way in: G4InuclElementaryParticle::type(const G4ParticleDefinition*), the function
  // both IsApplicable and createBullet call. Its result is the G4InuclParticleNames code.
  const G4ParticleDefinition* weak[] = {G4KaonZeroShort::Definition(),
                                        G4KaonZeroLong::Definition()};
  for (const G4ParticleDefinition* d : weak) {
    for (double u : us) {
      eng.set(u);
      const int t = G4InuclElementaryParticle::type(d);
      std::fprintf(f, "type,%d,%.17g,%d,%lld\n", d->GetPDGEncoding(), u, t, eng.draws());
    }
  }
  // On the way out: makeDynamicParticle on a K0 and on an anti-K0. The result column is the PDG
  // code of the definition it returns (310 or 130).
  CascadeProbe* probe = new CascadeProbe();   // never deleted: docs/RISK.md V174
  const int strong[] = {G4InuclParticleNames::kaonZero, G4InuclParticleNames::kaonZeroBar};
  for (int s : strong) {
    for (double u : us) {
      // Half a GeV of kinetic energy along z; makeDynamicParticle keeps only the direction and
      // the kinetic energy, so neither number changes which definition comes back.
      G4InuclElementaryParticle iep(0.5, s);
      eng.set(u);
      G4DynamicParticle* dp = probe->makeDynamicParticle(iep);
      std::fprintf(f, "mix,%d,%.17g,%d,%lld\n", s == G4InuclParticleNames::kaonZero ? 311 : -311,
                   u, dp ? dp->GetDefinition()->GetPDGEncoding() : 0, eng.draws());
      delete dp;
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// The heavy-flavour pair probabilities, read off a G4LundStringFragmentation built the way
// G4ExcitedStringDecay builds its default one. The getters are protected; a probe re-exports
// them. EnableBCParticles is what turns them on (G4LundStringFragmentation.cc:69-74), so the
// flag is written beside them.
class LundProbe : public G4LundStringFragmentation {
 public:
  using G4VLongitudinalStringDecay::GetProbCCbar;
  using G4VLongitudinalStringDecay::GetProbBBbar;
};

void dump_lund() {
  FILE* f = std::fopen("strange_lund.csv", "w");
  std::fprintf(f, "enable_bc_particles,prob_ccbar,prob_bbbar\n");
  LundProbe* lp = new LundProbe();   // never deleted: docs/RISK.md V174
  std::fprintf(f, "%d,%.17g,%.17g\n",
               G4HadronicParameters::Instance()->EnableBCParticles() ? 1 : 0,
               lp->GetProbCCbar(), lp->GetProbBBbar());
  std::fclose(f);
}

void dump_strange(const DumpContext&) {
  // One Bertini instance for IsApplicable, built after the run manager has initialised the
  // physics list, like every other in this dumper. Heap and never deleted: docs/RISK.md V174.
  G4CascadeInterface* bert = new G4CascadeInterface();
  dump_scope(bert);
  dump_scope_xs();
  dump_hnxsc();
  dump_ggxs();
  dump_k0type();
  dump_lund();
}

}  // namespace

G4GPU_REGISTER_DUMP("strange",
                    "strange_scope.csv strange_scope_xs.csv strange_hnxsc.csv strange_ggxs.csv "
                    "strange_k0type.csv strange_lund.csv",
                    dump_strange);
