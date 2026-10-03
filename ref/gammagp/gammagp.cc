// Which sub-process of QBBC's G4GammaGeneralProcess a photon's FIRST interaction is, counted in
// the running Geant4 11.1.1 - the measurement behind docs/RISK.md V207, V208 and V209 - and,
// since P21, the general process's own tables, node by node, as `BuildPhysicsTable` filled them.
//
// WHY THIS PROGRAM EXISTS
//
// P19 read `G4GammaGeneralProcess::PostStepDoIt` and `BuildPhysicsTable` and found that above
// 100 MeV (zone 3) the photonNuclear branch tests `q + P14 <= 1` right after
// `q + P13 <= 1` failed, with table 13 = (sigN + sigM)/sum and table 14 = sigN/sum - identical
// when sigM = 0, which it is in QBBC (no G4GammaConversionToMuons). So the branch cannot be
// taken and the photo-nuclear share of the total goes to the final `else`, conversion. A reading
// of the source is an argument; this is the measurement. And the same general process sums no
// Rayleigh term above 2 m_e (zones 2 and 3), which the port's photon did until P21 (V208) -
// counted here on the same photons.
//
// WHAT IT DOES - COUNTING
//
// One box of one material, 10 km on a side, so that a primary never reaches the boundary. A
// photon of energy E starts at the centre along +z; the stepping action records which process
// DEFINED the primary's first post-step interaction - `G4GammaGeneralProcess::SelectedProcess`
// sets the post-step point's process to the SUB-process it chose, so that is `conv`, `compt`,
// `phot`, `Rayl` or `photonNuclear` - and kills the primary; the stacking action kills every
// secondary. Nothing else is transported, so a million photons cost seconds.
//
// Beside each count it prints what the count should be if each sub-process took its share of
// the summed cross section: `G4EmCalculator::ComputeCrossSectionPerVolume` for the four EM
// processes and `G4HadronicProcessStore::GetInelasticCrossSectionPerVolume` - photonNuclear's own
// data store, element by element - for sigN. The general process samples from its own
// interpolated tables, so the expected fractions are good to the tables' interpolation and the
// comparison is a counting one.
//
// WHAT IT DOES - TABLES (P21)
//
// The counts say what the general process DID; the tables say what it was asked to do, and the
// port's table 9 has to be the second before the first can mean anything (docs/RISK.md V209).
// `G4GammaGeneralProcess::StorePhysicsTable` is public, so the process writes its own tables -
// `LambdaGeneral0/2/6/10`, the totals of the four zones, and `ProbGeneral1, 3, 4, 7-9, 11-14`,
// the selection fractions - in the binary format, which keeps every double (the ascii one prints
// twelve digits). Each is read back (see `read_stored_table` for why not by Geant4's reader)
// and printed node by node, and at each energy asked for it prints the table's value as the
// general process reads it - `LogVectorValue(e, G4Log(e))`, which is the call `GetProbability`
// and `ComputeGeneralLambda` make, on a `G4PhysicsLogVector` - so the interpolation is Geant4's
// own code, not a copy of it. Nothing here recomputes a table from the sub-processes: the
// numbers are the ones the running process holds. Beside them, at the same energies, each
// sub-process's own cross section (`G4EmCalculator`, and photonNuclear's data store) - what the
// tables were built from, so that what the tables' interpolation costs can be read off one file.
//
// Usage:  gammagp.exe <G4 material name> <events per energy> <E1 MeV> [<E2 MeV> ...]
//         gammagp.exe <G4 material name> tables [<E1 MeV> ...]
//
//   ref\gammagp\run.bat G4_WATER 1000000 20 99.9 100 150
//   ref\gammagp\run.bat G4_BONE_COMPACT_ICRU tables 20 22 60

#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4Gamma.hh"
#include "G4HadronicProcessStore.hh"
#include "G4Log.hh"
#include "G4LogicalVolume.hh"
#include "G4LossTableManager.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4PhysicsLogVector.hh"
#include "G4ProductionCutsTable.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VEmProcess.hh"
#include "G4VProcess.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "QBBC.hh"
#include "Randomize.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace {

std::string g_material = "G4_WATER";
double g_energy = 20 * MeV;
std::map<std::string, long long> g_counts;

class Det : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    G4Material* m = G4NistManager::Instance()->FindOrBuildMaterial(g_material);
    if (m == nullptr) {
      std::printf("FATAL: no material %s\n", g_material.c_str());
      std::exit(1);
    }
    auto* box = new G4Box("World", 5 * km, 5 * km, 5 * km);
    auto* lv = new G4LogicalVolume(box, m, "World");
    return new G4PVPlacement(nullptr, G4ThreeVector(), lv, "World", nullptr, false, 0, true);
  }
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  Gun() : gun_(1) {
    gun_.SetParticleDefinition(G4Gamma::Gamma());
    gun_.SetParticlePosition(G4ThreeVector());
    gun_.SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
  }
  void GeneratePrimaries(G4Event* ev) override {
    gun_.SetParticleEnergy(g_energy);
    gun_.GeneratePrimaryVertex(ev);
  }

 private:
  G4ParticleGun gun_;
};

class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    G4Track* t = step->GetTrack();
    if (t->GetTrackID() != 1) { return; }
    const G4StepPoint* post = step->GetPostStepPoint();
    const G4VProcess* p = post->GetProcessDefinedStep();
    const std::string name = (p != nullptr) ? p->GetProcessName() : std::string("(none)");
    if (post->GetStepStatus() == fGeomBoundary || post->GetStepStatus() == fWorldBoundary) {
      ++g_counts["(left the box)"];
    } else {
      ++g_counts[name];
    }
    t->SetTrackStatus(fStopAndKill);
  }
};

class Stacking : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    return (t->GetParentID() > 0) ? fKill : fUrgent;
  }
};

class Actions : public G4VUserActionInitialization {
 public:
  void Build() const override {
    SetUserAction(new Gun());
    SetUserAction(new Stepping());
    SetUserAction(new Stacking());
  }
};

/// One vector of a table stored by `G4PhysicsTable::StorePhysicsTable` in binary: `G4int` type,
/// then `G4PhysicsVector::Store`'s `edgeMin`, `edgeMax`, `numberOfNodes`, the size, and the
/// (energy, value) pairs interleaved (G4PhysicsVector.cc).
struct StoredVector {
  int type = -1;
  double edge_min = 0, edge_max = 0;
  std::vector<double> e, v;
};

/// READ BACK HERE AND NOT BY `G4PhysicsTable::RetrievePhysicsTable`, and that is a Geant4 11.1.1
/// defect on this platform rather than a preference: its open is inverted -
/// `if(ascii) fIn.open(name, in | binary); else fIn.open(name, in);` - so a BINARY table is read
/// in TEXT mode, where Windows' runtime folds every CR LF pair of the doubles' bytes into LF and
/// stops at the first 0x1A. The store side opens binary correctly. The read below is binary.
bool read_stored_table(const std::string& file, std::vector<StoredVector>& out) {
  std::FILE* f = std::fopen(file.c_str(), "rb");
  if (f == nullptr) { return false; }
  std::size_t n = 0;
  bool ok = std::fread(&n, sizeof n, 1, f) == 1 && n < 1000;
  for (std::size_t k = 0; ok && k < n; ++k) {
    StoredVector sv;
    std::size_t nodes = 0, size = 0;
    ok = std::fread(&sv.type, sizeof(int), 1, f) == 1
         && std::fread(&sv.edge_min, sizeof(double), 1, f) == 1
         && std::fread(&sv.edge_max, sizeof(double), 1, f) == 1
         && std::fread(&nodes, sizeof nodes, 1, f) == 1 && std::fread(&size, sizeof size, 1, f) == 1
         && nodes == size && size >= 2 && size < 100000;
    if (!ok) { break; }
    std::vector<double> pairs(2 * size);
    ok = std::fread(pairs.data(), sizeof(double), pairs.size(), f) == pairs.size();
    for (std::size_t j = 0; ok && j < size; ++j) {
      sv.e.push_back(pairs[2 * j]);
      sv.v.push_back(pairs[2 * j + 1]);
    }
    out.push_back(sv);
  }
  std::fclose(f);
  return ok && !out.empty();
}

/// The general process's tables, stored by the process and evaluated by Geant4's own vector.
///
/// `G4GammaGeneralProcess::StorePhysicsTable` names each file
/// `GetPhysicsTableFileName(part, directory, nam, ascii)` = `<dir>/<nam>.GammaGeneralProc.gamma.dat`
/// with `nam` = "LambdaGeneral<i>" for i = 0, 2, 6, 10 and "ProbGeneral<i>" otherwise
/// (G4GammaGeneralProcess.cc:679-686), and stores only the tables `theT[i]` built - table 5 is
/// never built, and table 1 only when Rayleigh is a sub-process. A table whose file is absent is
/// skipped. One couple here, so each table is one vector.
///
/// The interpolated values come from a `G4PhysicsLogVector` built by the constructor
/// `InitialiseProcess` built the original with - `(edgeMin, edgeMax, nodes - 1, spline)` - and
/// filled with the stored values: its node energies are then Geant4's own arithmetic, and are
/// checked against the stored ones to the bit before anything is printed.
int dump_tables(const std::vector<double>& energies) {
  G4VEmProcess* gp = G4LossTableManager::Instance()->GetGammaGeneralProcess();
  if (gp == nullptr) {
    std::printf("FATAL: QBBC built no G4GammaGeneralProcess - /process/em/UseGeneralProcess "
                "is off, so there are no general-process tables to dump\n");
    return 1;
  }
  // Under the system temporary directory, not the working one: ref/oracle/run.bat runs this from
  // ref/oracle, where a directory of .dat files would be neither ignored nor wanted.
  const std::string dir =
      (std::filesystem::temp_directory_path() / ("g4gpu_gammagp_tables_" + g_material)).string();
  std::filesystem::create_directories(dir);
  if (!gp->StorePhysicsTable(G4Gamma::Gamma(), dir, false)) {
    std::printf("FATAL: G4GammaGeneralProcess::StorePhysicsTable failed for %s\n", dir.c_str());
    return 1;
  }
  std::printf("# %s: G4GammaGeneralProcess's tables as BuildPhysicsTable filled them, stored by\n"
              "# the process (binary) and read back byte for byte. Lambda tables (0, 2, 6, 10)\n"
              "# are 1/mm; the rest are fractions. kind=node is a node; kind=at is a\n"
              "# G4PhysicsLogVector's LogVectorValue(E, G4Log(E)) at an energy asked for;\n"
              "# kind=model is a sub-process's own cross section there, 1/mm (table 100 compt,\n"
              "# 101 conv, 102 phot, 103 Rayl, 104 photonNuclear's data store); kind=lambda is\n"
              "# its GetLambda off its own lambda tables (200 compt, 201 conv, 202 phot, 203 Rayl).\n",
              g_material.c_str());
  std::printf("material,table,kind,index,E_MeV,value\n");
  int found = 0;
  for (int i = 0; i < 15; ++i) {
    const bool lam = (i == 0 || i == 2 || i == 6 || i == 10);
    const std::string nam = std::string(lam ? "LambdaGeneral" : "ProbGeneral") + std::to_string(i);
    const std::string file = dir + "/" + nam + "." + gp->GetProcessName() + ".gamma.dat";
    if (!std::filesystem::exists(file)) { continue; }
    std::vector<StoredVector> stored;
    if (!read_stored_table(file, stored) || stored.size() != 1 || stored[0].type != 2) {
      std::printf("FATAL: %s is not one stored G4PhysicsLogVector\n", file.c_str());
      return 1;
    }
    const StoredVector& sv = stored[0];
    // `spline` exactly as `BuildPhysicsTable` set it: `FillSecondDerivatives` for tables 0, 1
    // and 10-14 (zones 0 and 3), none for 2-9 (zones 1 and 2) - for zone 2, table 9, the
    // whole point (docs/RISK.md V209).
    const G4bool spline = (i <= 1 || i >= 10);
    G4PhysicsLogVector v(sv.edge_min, sv.edge_max, sv.e.size() - 1, spline);
    for (std::size_t j = 0; j < sv.e.size(); ++j) {
      if (v.Energy(j) != sv.e[j]) {
        std::printf("FATAL: %s node %zu is %.17g, and Geant4's own constructor gives %.17g\n",
                    file.c_str(), j, sv.e[j], v.Energy(j));
        return 1;
      }
      v.PutValue(j, sv.v[j]);
    }
    if (spline) { v.FillSecondDerivatives(); }
    ++found;
    for (std::size_t j = 0; j < sv.e.size(); ++j) {
      std::printf("%s,%d,node,%zu,%.17g,%.17g\n", g_material.c_str(), i, j, sv.e[j] / MeV,
                  sv.v[j] * (lam ? mm : 1.0));
    }
    for (const double e : energies) {
      const double ee = e * MeV;
      std::printf("%s,%d,at,-1,%.17g,%.17g\n", g_material.c_str(), i, e,
                  v.LogVectorValue(ee, G4Log(ee)) * (lam ? mm : 1.0));
    }
  }
  if (found == 0) {
    std::printf("FATAL: no general-process table was found under %s\n", dir.c_str());
    return 1;
  }
  // And what the tables were built FROM, at the same energies: each sub-process's own model
  // through `G4EmCalculator::ComputeCrossSectionPerVolume` and photonNuclear's data store
  // through `G4HadronicProcessStore`, 1/mm, as kind=model rows with table = 100 compt,
  // 101 conv, 102 phot, 103 Rayl, 104 photonNuclear. The general process's totals (tables 0, 2,
  // 6, 10) differ from these sums by its interpolation - linear in zones 1 and 2 - and by its
  // sub-processes' own lambda tables, which `BuildPhysicsTable` reads through `GetLambda`;
  // docs/RISK.md has the measurement.
  const G4Material* mat = G4NistManager::Instance()->FindOrBuildMaterial(g_material);
  G4EmCalculator calc;
  const G4ParticleDefinition* gamma = G4Gamma::Gamma();
  const char* names[4] = {"compt", "conv", "phot", "Rayl"};
  // And what `BuildPhysicsTable` actually read at a node: each sub-process's `GetLambda(e,
  // couple, loge)`, off its own lambda tables (kind=lambda, table = 200 compt, 201 conv,
  // 202 phot, 203 Rayl), on the one couple this geometry has. Where these and the models part,
  // the general process's node sums part from the models.
  const G4MaterialCutsCouple* couple =
      G4ProductionCutsTable::GetProductionCutsTable()->GetMaterialCutsCouple(0);
  for (const double e : energies) {
    const double ee = e * MeV;
    for (int k = 0; k < 4; ++k) {
      std::printf("%s,%d,model,-1,%.17g,%.17g\n", g_material.c_str(), 100 + k, e,
                  calc.ComputeCrossSectionPerVolume(ee, gamma, names[k], mat) * mm);
      G4VEmProcess* sub = gp->GetEmProcess(names[k]);
      if (sub != nullptr && couple != nullptr) {
        std::printf("%s,%d,lambda,-1,%.17g,%.17g\n", g_material.c_str(), 200 + k, e,
                    sub->GetLambda(ee, couple, G4Log(ee)) * mm);
      }
    }
    std::printf("%s,104,model,-1,%.17g,%.17g\n", g_material.c_str(), e,
                G4HadronicProcessStore::Instance()->GetInelasticCrossSectionPerVolume(gamma, ee,
                                                                                     mat) * mm);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::printf("usage: gammagp.exe <material> <events> <E1 MeV> [<E2 MeV> ...]\n"
                "       gammagp.exe <material> tables [<E1 MeV> ...]\n");
    return 1;
  }
  g_material = argv[1];
  const bool tables = (std::string(argv[2]) == "tables");
  if (!tables && argc < 4) {
    std::printf("usage: gammagp.exe <material> <events> <E1 MeV> [<E2 MeV> ...]\n");
    return 1;
  }

  auto* rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  rm->SetUserInitialization(new Det());
  rm->SetUserInitialization(new QBBC(0));
  rm->SetUserInitialization(new Actions());
  rm->Initialize();
  CLHEP::HepRandom::setTheSeed(20260926);

  if (tables) {
    // A run of no events builds the physics tables (`RunInitialization` -> `BuildPhysicsTable`)
    // and transports nothing, so the counting mode's random stream is not touched by this one -
    // the two modes are separate invocations.
    rm->BeamOn(0);
    std::vector<double> es;
    for (int k = 3; k < argc; ++k) { es.push_back(std::atof(argv[k])); }
    const int rc = dump_tables(es);
    std::fflush(nullptr);
    std::_Exit(rc);
  }

  const long long n = std::atoll(argv[2]);
  const G4Material* mat = G4NistManager::Instance()->FindOrBuildMaterial(g_material);
  G4EmCalculator calc;
  const G4ParticleDefinition* gamma = G4Gamma::Gamma();
  std::printf("material,E_MeV,events,process,count,fraction,expected_fraction_if_shares\n");
  for (int k = 3; k < argc; ++k) {
    g_energy = std::atof(argv[k]) * MeV;
    g_counts.clear();
    rm->BeamOn(static_cast<G4int>(n));
    const double e = g_energy;
    const double s_compt = calc.ComputeCrossSectionPerVolume(e, gamma, "compt", mat);
    const double s_conv = calc.ComputeCrossSectionPerVolume(e, gamma, "conv", mat);
    const double s_phot = calc.ComputeCrossSectionPerVolume(e, gamma, "phot", mat);
    const double s_rayl = calc.ComputeCrossSectionPerVolume(e, gamma, "Rayl", mat);
    const double s_gn =
        G4HadronicProcessStore::Instance()->GetInelasticCrossSectionPerVolume(gamma, e, mat);
    // The shares each sub-process would have if all five took theirs of one summed total.
    const double sum = s_compt + s_conv + s_phot + s_rayl + s_gn;
    const std::map<std::string, double> share = {{"compt", s_compt / sum},
                                                 {"conv", s_conv / sum},
                                                 {"phot", s_phot / sum},
                                                 {"Rayl", s_rayl / sum},
                                                 {"photonNuclear", s_gn / sum}};
    for (const auto& kv : share) {
      if (g_counts.find(kv.first) == g_counts.end()) { g_counts[kv.first] = 0; }
    }
    for (const auto& kv : g_counts) {
      const auto it = share.find(kv.first);
      std::printf("%s,%.6g,%lld,%s,%lld,%.6e,%.6e\n", g_material.c_str(), e / MeV, n,
                  kv.first.c_str(), kv.second, double(kv.second) / double(n),
                  (it != share.end()) ? it->second : 0.0);
    }
    std::printf("# %s %.6g MeV: sigma/mm compt %.6e conv %.6e phot %.6e Rayl %.6e "
                "photonNuclear %.6e\n", g_material.c_str(), e / MeV, s_compt * mm,
                s_conv * mm, s_phot * mm, s_rayl * mm, s_gn * mm);
    std::fflush(stdout);
  }
  // NOT `delete rm`. Measured: the run manager's teardown dies with 0xC0000005 after every line
  // above is written - the shape docs/RISK.md's "a destructor that frees a class static" section
  // traced in the dumper, and `G4GammaGeneralProcess::theHandler` is such a static. The numbers
  // are complete and flushed by then; the process leaves without running static destructors.
  (void)rm;
  std::fflush(nullptr);
  std::_Exit(0);
}
