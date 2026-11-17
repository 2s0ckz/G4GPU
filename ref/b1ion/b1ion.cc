// Example B1 on QBBC with ionElastic COUNTED - the instrument docs/RISK.md V267 was measured with.
//
// WHAT IT ASKS. P25 wires `ionElastic` (G4IonElasticPhysics: G4ComponentGGNuclNuclXsc +
// G4NuclNuclDiffuseElastic) for every real nuclide. Before it, the port's GenericIon tracks drew
// no elastic interaction length at all - a structural zero, booked nowhere - so how much the
// process does on a B1 beam is a question only the real Geant4 can answer: how many steps of a
// real ion end in ionElastic, how many recoils it makes and at what energy, and how much track
// the ions run in each material against the cross section's mean free path.
//
// WHAT IT COUNTS. A stepping action that calls B1's own first (so the dose is B1's), then, for a
// track whose process manager is G4GenericIon's - every real nuclide, none of d/t/He3/alpha:
//
//   per material  steps, track length, and sum over steps of length x Sigma_el(E_pre), the
//                 expected number of ionElastic interactions from the component - a private
//                 G4ComponentGGNuclNuclXsc made at the first ion step, AFTER QBBC built its own,
//                 summed over the material's elements at the NIST mean mass as
//                 G4CrossSectionElastic hands it (tests/test_ion_elastic.cu holds that sum to
//                 the process store's);
//   ionElastic    steps whose post-step process is ionElastic, by material and by nuclide, the
//                 projectile's energy per nucleon, and the recoil the step made (species and
//                 kinetic energy) or that it made none (the recoil fell under the 70 keV proton
//                 cut and was deposited);
//   ionInelastic  the same steps for the inelastic process, for scale.
//
// NOTHING HERE DRAWS A RANDOM NUMBER OR CHANGES THE STATE OF A PROCESS: the component is its own
// instance and the counters only read the step. So the dose B1 prints is QBBC's to every digit -
// which each run checks against the B1 sweep's own Geant4 column, the proof that the instrument
// did not move the stream.
//
// Usage:  run.bat <macro>
#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"

#include "G4ComponentGGNuclNuclXsc.hh"
#include "G4Element.hh"
#include "G4GenericIon.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4ProcessManager.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UImanager.hh"
#include "G4VProcess.hh"
#include "QBBC.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace B1;

namespace {

constexpr int kMat = 8;      // by G4Material::GetIndex(); B1 builds four
constexpr int kEbins = 50;   // log10(E/A / MeV) over [-3, 4.5), 0.15 a bin
constexpr int kRbins = 40;   // log10(E_recoil / MeV) over [-2, 4)

struct Tally {
  long long steps[kMat] = {0};
  double length[kMat] = {0};
  double expected[kMat] = {0};
  long long el[kMat] = {0};
  long long inel[kMat] = {0};
  long long el_ebin[kEbins] = {0};
  long long ion_len_ebin_steps[kEbins] = {0};
  long long rec_species[4] = {0};  // proton, d/t/He3/alpha, GenericIon nuclide, none
  long long rec_bin_p[kRbins] = {0};
  long long rec_bin_ion[kRbins] = {0};
  double rec_sum[3] = {0};
  double rec_max[3] = {0};
  // The sub-threshold recoils: the step's non-ionizing deposit, which G4HadronElasticProcess
  // proposes equal to the local one and which no other process of option0 sets for an ion.
  double sub_sum = 0, sub_max = 0, sub_min = 1e300;
  long long sub_bin[16] = {0};  // log10(E/MeV) over [-9, -1), 0.5 a bin
  std::map<int, long long> el_by_nuclide;   // 1000*Z + A
  std::map<int, double> len_by_nuclide;
  std::map<int, double> elen_by_nuclide;    // sum of length * E/A, for the mean
  std::string mat_name[kMat];
};
Tally g;

int ebin_of(double e_per_n) {
  int b = int(std::floor((std::log10(e_per_n) + 3.0) / 0.15));
  return std::min(std::max(b, 0), kEbins - 1);
}
int rbin_of(double e) {
  int b = int(std::floor((std::log10(e) + 2.0) * (kRbins / 6.0)));
  return std::min(std::max(b, 0), kRbins - 1);
}

G4ComponentGGNuclNuclXsc* component() {
  static G4ComponentGGNuclNuclXsc* c = nullptr;
  if (c == nullptr) { c = new G4ComponentGGNuclNuclXsc(); }
  return c;
}

/// Sigma_el per volume, 1/mm: G4CrossSectionDataStore's sum of n_i sigma_i over the elements,
/// sigma_i from the component at the NIST mean mass (G4CrossSectionElastic::GetElementCrossSection).
double sigma_el(const G4ParticleDefinition* p, double ekin, const G4Material* m) {
  G4NistManager* nist = G4NistManager::Instance();
  const G4double* n = m->GetVecNbOfAtomsPerVolume();
  double s = 0.0;
  for (size_t i = 0; i < m->GetNumberOfElements(); ++i) {
    const int z = m->GetElement(int(i))->GetZasInt();
    s += n[i] * component()->GetElasticElementCrossSection(p, ekin, z, nist->GetAtomicMassAmu(z));
  }
  return s * mm;
}

class CountingStepping : public SteppingAction {
 public:
  explicit CountingStepping(EventAction* ea) : SteppingAction(ea) {}
  void UserSteppingAction(const G4Step* step) override {
    SteppingAction::UserSteppingAction(step);
    const G4Track* tr = step->GetTrack();
    const G4ParticleDefinition* p = tr->GetDefinition();
    if (p->GetProcessManager() != G4GenericIon::GenericIon()->GetProcessManager()) { return; }
    const G4StepPoint* pre = step->GetPreStepPoint();
    const G4Material* m = pre->GetMaterial();
    const int mi = int(m->GetIndex());
    if (mi < 0 || mi >= kMat) { return; }
    if (g.mat_name[mi].empty()) { g.mat_name[mi] = m->GetName(); }
    const int z = int(std::lround(p->GetPDGCharge() / eplus));
    const int a = p->GetBaryonNumber();
    const int key = 1000 * z + a;
    const double e = pre->GetKineticEnergy();
    const double len = step->GetStepLength() / mm;
    ++g.steps[mi];
    g.length[mi] += len;
    g.len_by_nuclide[key] += len;
    g.elen_by_nuclide[key] += len * e / a;
    ++g.ion_len_ebin_steps[ebin_of(e / a)];
    if (len > 0.0 && e > 0.0) { g.expected[mi] += len * sigma_el(p, e, m); }

    const G4VProcess* proc = step->GetPostStepPoint()->GetProcessDefinedStep();
    if (proc == nullptr) { return; }
    if (proc->GetProcessName() == "ionInelastic") { ++g.inel[mi]; return; }
    if (proc->GetProcessName() != "ionElastic") { return; }
    ++g.el[mi];
    ++g.el_by_nuclide[key];
    // The energy the interaction saw is the POST-step one (PostStepDoIt runs after the
    // continuous loss); the post-step point holds the primary's state after the scatter, so the
    // pre-step energy is the binning variable - the step is short and this is a histogram.
    ++g.el_ebin[ebin_of(e / a)];
    const auto* secs = step->GetSecondaryInCurrentStep();
    if (secs == nullptr || secs->empty()) {
      ++g.rec_species[3];
      const double en = step->GetNonIonizingEnergyDeposit();
      g.sub_sum += en;
      g.sub_max = std::max(g.sub_max, en);
      g.sub_min = std::min(g.sub_min, en);
      int b = (en > 0.0) ? int(std::floor((std::log10(en) + 9.0) * 2.0)) : 0;
      ++g.sub_bin[std::min(std::max(b, 0), 15)];
      return;
    }
    const G4Track* s = (*secs)[0];
    const int pdg = s->GetDefinition()->GetPDGEncoding();
    const double er = s->GetKineticEnergy();
    int k = 2;
    if (pdg == 2212) { k = 0; }
    else if (pdg == 1000010020 || pdg == 1000010030 || pdg == 1000020030 || pdg == 1000020040) {
      k = 1;
    }
    ++g.rec_species[k];
    g.rec_sum[k] += er;
    g.rec_max[k] = std::max(g.rec_max[k], er);
    if (k == 0) { ++g.rec_bin_p[rbin_of(er)]; } else { ++g.rec_bin_ion[rbin_of(er)]; }
  }
};

class Actions : public G4VUserActionInitialization {
 public:
  void BuildForMaster() const override { SetUserAction(new RunAction); }
  // B1's ActionInitialization::Build, line for line, with the stepping action counted.
  void Build() const override {
    SetUserAction(new PrimaryGeneratorAction);
    auto runAction = new RunAction;
    SetUserAction(runAction);
    auto eventAction = new EventAction(runAction);
    SetUserAction(eventAction);
    SetUserAction(new CountingStepping(eventAction));
  }
};

void report() {
  std::printf("b1ion: per material - steps of a real ion, track length (mm), ionElastic expected "
              "(sum of length x Sigma_el), ionElastic counted, ionInelastic counted\n");
  long long tot_el = 0, tot_inel = 0, tot_steps = 0;
  double tot_exp = 0, tot_len = 0;
  for (int i = 0; i < kMat; ++i) {
    if (g.steps[i] == 0) { continue; }
    std::printf("b1ion:   %-22s %12lld %14.6g %12.6g %8lld %8lld\n", g.mat_name[i].c_str(),
                g.steps[i], g.length[i], g.expected[i], g.el[i], g.inel[i]);
    tot_el += g.el[i]; tot_inel += g.inel[i]; tot_steps += g.steps[i];
    tot_exp += g.expected[i]; tot_len += g.length[i];
  }
  std::printf("b1ion:   %-22s %12lld %14.6g %12.6g %8lld %8lld\n", "all", tot_steps, tot_len,
              tot_exp, tot_el, tot_inel);
  std::printf("b1ion: recoils - proton %lld (sum %.6g MeV, max %.6g), d/t/He3/alpha %lld (sum "
              "%.6g, max %.6g), GenericIon %lld (sum %.6g, max %.6g), none (deposited under the "
              "cut) %lld\n", g.rec_species[0], g.rec_sum[0], g.rec_max[0], g.rec_species[1],
              g.rec_sum[1], g.rec_max[1], g.rec_species[2], g.rec_sum[2], g.rec_max[2],
              g.rec_species[3]);
  std::printf("b1ion: sub-threshold recoils (non-ionizing deposit of the ionElastic step) sum "
              "%.6g MeV, min %.6g, max %.6g; log10(E/MeV) [-9,-1) by 0.5:", g.sub_sum,
              (g.rec_species[3] > 0) ? g.sub_min : 0.0, g.sub_max);
  for (int b = 0; b < 16; ++b) { std::printf(" %lld", g.sub_bin[b]); }
  std::printf("\n");
  std::printf("b1ion: recoil energy bins log10(E/MeV) [-2,4) by 0.15 - proton:");
  for (int b = 0; b < kRbins; ++b) { std::printf(" %lld", g.rec_bin_p[b]); }
  std::printf("\nb1ion:   ion:");
  for (int b = 0; b < kRbins; ++b) { std::printf(" %lld", g.rec_bin_ion[b]); }
  std::printf("\nb1ion: ionElastic by log10(E/A / MeV) [-3,4.5) by 0.15:");
  for (int b = 0; b < kEbins; ++b) { std::printf(" %lld", g.el_ebin[b]); }
  std::printf("\nb1ion: ion steps by the same bins:");
  for (int b = 0; b < kEbins; ++b) { std::printf(" %lld", g.ion_len_ebin_steps[b]); }
  std::printf("\nb1ion: nuclides by track length (Z, A: length mm, mean E/A MeV by length, "
              "ionElastic) - the first 15\n");
  std::vector<std::pair<double, int>> order;
  for (const auto& kv : g.len_by_nuclide) { order.push_back({-kv.second, kv.first}); }
  std::sort(order.begin(), order.end());
  for (size_t i = 0; i < order.size() && i < 15; ++i) {
    const int key = order[i].second;
    const double l = -order[i].first;
    std::printf("b1ion:   Z=%3d A=%3d  %12.6g mm  %10.4g MeV/u  %lld\n", key / 1000, key % 1000,
                l, (l > 0) ? g.elen_by_nuclide[key] / l : 0.0, g.el_by_nuclide[key]);
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: b1ion.exe <macro>\n");
    return 1;
  }
  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
  runManager->SetUserInitialization(new DetectorConstruction());
  G4VModularPhysicsList* physicsList = new QBBC;
  physicsList->SetVerboseLevel(1);
  runManager->SetUserInitialization(physicsList);
  runManager->SetUserInitialization(new Actions());
  G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[1]);
  report();
  delete runManager;
  return 0;
}
