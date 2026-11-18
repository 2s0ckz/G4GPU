// P29: FTFP's ion arm re-measured with the de-excited residual - Geant4's side.
//
// WHAT IT IS FOR. docs/RISK.md V236 (P24) found that the port's FTFP ion arm left 25-39% fewer
// secondaries than QBBC's at the five heavy points of tests/test_ftf_model.cu's ion grid, while
// the string products and the decayed list agreed to a few per cent. P28 (V264-V265) then found
// that the port de-excited NO residual of the FTF hand-over and added `ftf_deexcite_residual`;
// the ion arm goes through the same code, and this dump is the Geant4 side of the measurement
// that says whether that closed it - not only in the final-state count V236 compared, but per
// species group, per event, which is what `ftf_modelstat` gives the hadron arm.
//
// THE INSTRUMENT IS V236's. The GenericIon `ionInelastic` process's own "FTFP" (G4IonPhysicsXS ->
// G4FTFBuilder), its transport swapped for `IonArmInterface`, a `G4GeneratorPrecompoundInterface`
// on the precompound model QBBC built for it, which counts the list it is handed (`strings`),
// runs `G4DecayKineticTracks` and counts again (`decayed`), then hands the decayed list to
// Geant4's own `PropagateNuclNucl` - which captures into the two residuals and de-excites BOTH
// (`theDeExcitation->DeExcite`, G4GeneratorPrecompoundInterface.cc:668 and :731; the interface's
// constructor sets `theDeExcitation` to the model passed in, :120). The five points, their event
// counts, their order and their seeds are `dump_capacity`'s (dump_ftf.cc), and the accounting
// below draws no random number, so this dump's `strings`/`decayed`/`fs` histograms ARE
// ftf_capacity.csv's to the byte - tests/test_ion_arm.cu checks that when both files are present.
//
// THE KINETIC ENERGY of a secondary is `E - m_PDG`, as in dump_ftfp_tape.cc: what
// `G4HadronicProcess::FillResult` hands the transport once the off-shell correction has run; a
// nucleus's PDG mass is its ground state's (`G4NucleiProperties::GetNuclearMass`).
//
// ion_arm.csv   case,pz,pa,ekin_per_a_MeV,tz,ta,N,row,key,count,count2,sum_ke,sum_ke2
//   row = strings | decayed | fs : a histogram, key = the list's size, count = events
//         (fs = -1 is an exception: no final state)
//   row = group : key = the species group (`ion_arm_group`, the same function in the test),
//         count = particles, count2 = sum over events of (particles in the event)^2,
//         sum_ke = their kinetic energy, sum_ke2 = sum over events of (the event's group KE)^2
//   row = event : key 0 = the energy balance E_out - E_in (sum_ke, sum_ke2 over events),
//         key -1 = the residual-class (A > 4) excitation m_dyn - m_ground per event
//
// Registered as "ion_arm"; G4GPU_DUMP_ONLY=ion_arm runs it alone.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>

#include "G4DecayKineticTracks.hh"
#include "G4DynamicParticle.hh"
#include "G4GeneratorPrecompoundInterface.hh"
#include "G4GenericIon.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadronicProcess.hh"
#include "G4IonTable.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleus.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4TheoFSGenerator.hh"
#include "G4VIntraNuclearTransportModel.hh"
#include "G4VPreCompoundModel.hh"
#include "Randomize.hh"

namespace {

class IonArmInterface : public G4GeneratorPrecompoundInterface {
 public:
  explicit IonArmInterface(G4VPreCompoundModel* p) : G4GeneratorPrecompoundInterface(p) {}
  G4ReactionProductVector* Propagate(G4KineticTrackVector* s, G4V3DNucleus* t) override {
    count_and_decay(s);
    return G4GeneratorPrecompoundInterface::Propagate(s, t);
  }
  G4ReactionProductVector* PropagateNuclNucl(G4KineticTrackVector* s, G4V3DNucleus* t,
                                             G4V3DNucleus* p) override {
    count_and_decay(s);
    return G4GeneratorPrecompoundInterface::PropagateNuclNucl(s, t, p);
  }
  int strings = -1;
  int decayed = -1;

 private:
  void count_and_decay(G4KineticTrackVector* s) {
    strings = (s != nullptr) ? static_cast<int>(s->size()) : 0;
    if (s != nullptr) { G4DecayKineticTracks decay(s); }
    decayed = (s != nullptr) ? static_cast<int>(s->size()) : 0;
  }
};

template <typename Tag, typename Tag::type M> struct IonArmBridge {
  friend typename Tag::type bridge(Tag) { return M; }
};
/// `G4VIntraNuclearTransportModel::GetDeExcitation` is protected - the precompound model QBBC
/// built is reached through the explicit-instantiation route dump_ftf.cc uses for the same call.
struct TagIonArmDeEx {
  using type = G4VPreCompoundModel* (G4VIntraNuclearTransportModel::*)() const;
  friend type bridge(TagIonArmDeEx);
};
template struct IonArmBridge<TagIonArmDeEx, &G4VIntraNuclearTransportModel::GetDeExcitation>;

/// dump_ftfp_tape.cc's species key: the PDG code for every non-nucleus, 2212/2112 for a nucleon,
/// the nuclear code for A = 2..4 and 1 for anything heavier.
int ion_arm_key(int pdg, int z, int a) {
  if (a <= 0) { return pdg; }
  if (a == 1) { return (z == 1) ? 2212 : (z == 0 ? 2112 : pdg); }
  if (a <= 4) { return 1000000000 + z * 10000 + a * 10; }
  return 1;
}

/// The species groups of the per-species rows (P29's brief): n, p, d, t, He3, alpha, heavier
/// fragments, pi+, pi-, pi0, kaons, eta/eta', gammas, anti-baryons, hyperons, the rest.
/// tests/test_ion_arm.cu has the same function.
int ion_arm_group(int key) {
  switch (key) {
    case 2112: return 1;
    case 2212: return 2;
    case 1000010020: return 3;
    case 1000010030: return 4;
    case 1000020030: return 5;
    case 1000020040: return 6;
    case 211: return 8;
    case -211: return 9;
    case 111: return 10;
    case 321: case -321: case 311: case -311: case 310: case 130: return 11;
    case 221: case 331: return 12;
    case 22: return 13;
    case 3122: case 3222: case 3212: case 3112: case 3322: case 3312: case 3334: return 15;
    default: break;
  }
  if (key == 1 || key > 1000000000) { return 7; }
  if (key < -1000) { return 14; }
  return 16;
}

struct GroupAcc {
  long long count = 0, count2 = 0;
  double ke = 0.0, ke2 = 0.0;
};

void dump_ion_arm(const DumpContext&) {
  FILE* f = std::fopen("ion_arm.csv", "w");
  std::fprintf(f, "case,pz,pa,ekin_per_a_MeV,tz,ta,N,row,key,count,count2,sum_ke,sum_ke2\n");
  G4ParticleDefinition* gi = G4GenericIon::GenericIon();
  G4TheoFSGenerator* ftfp = nullptr;
  G4ProcessManager* pm = gi->GetProcessManager();
  G4ProcessVector* pv = (pm != nullptr) ? pm->GetProcessList() : nullptr;
  for (G4int i = 0; pv != nullptr && i < (G4int)pv->size(); ++i) {
    G4HadronicProcess* hp = dynamic_cast<G4HadronicProcess*>((*pv)[i]);
    if (hp == nullptr || hp->GetProcessName() != "ionInelastic") { continue; }
    ftfp = dynamic_cast<G4TheoFSGenerator*>(hp->GetHadronicModel("FTFP"));
  }
  if (ftfp == nullptr) {
    std::fprintf(f, "NO_FTFP,0,0,0,0,0,0,none,0,0,0,0,0\n");
    std::fclose(f);
    return;
  }
  const G4VIntraNuclearTransportModel* own = ftfp->GetTransport();
  auto* arm = new IonArmInterface((own->*bridge(TagIonArmDeEx{}))());
  ftfp->SetTransport(arm);

  struct CCase { int pz, pa; double per_a; int tz, ta; const char* name; int n; };
  // dump_capacity's five points, order and event counts.
  const CCase cases[] = {{26, 56, 20000.0, 82, 207, "Fe56_20000_Pb207", 2000},
                         {26, 56, 20000.0, 26, 56, "Fe56_20000_Fe56", 500},
                         {26, 56, 8000.0, 82, 207, "Fe56_8000_Pb207", 500},
                         {8, 16, 20000.0, 82, 207, "O16_20000_Pb207", 500},
                         {6, 12, 20000.0, 82, 207, "C12_20000_Pb207", 500}};
  G4IonTable* ions = G4IonTable::GetIonTable();
  for (const CCase& c : cases) {
    const G4ParticleDefinition* ion = ions->GetIon(c.pz, c.pa, 0.0);
    // dump_capacity's seed, so that the histograms reproduce ftf_capacity.csv.
    CLHEP::HepRandom::setTheSeed(24000L + c.pa * 1000 + c.ta + G4int(c.per_a / 1000.0));
    const double e_in = c.per_a * c.pa + ion->GetPDGMass() +
                        G4NucleiProperties::GetNuclearMass(c.ta, c.tz);
    std::map<int, long long> h_strings, h_decayed, h_fs;
    std::map<int, GroupAcc> groups;
    GroupAcc bal, exc;
    for (int ev = 0; ev < c.n; ++ev) {
      G4DynamicParticle dp(ion, G4ThreeVector(0, 0, 1), c.per_a * c.pa * MeV);
      G4HadProjectile proj(dp);
      G4Nucleus nucleus(c.ta, c.tz);
      arm->strings = -1;
      arm->decayed = -1;
      G4HadFinalState* r = nullptr;
      try {
        r = ftfp->ApplyYourself(proj, nucleus);
      } catch (...) {
        r = nullptr;
      }
      ++h_strings[arm->strings];
      ++h_decayed[arm->decayed];
      ++h_fs[(r != nullptr) ? static_cast<int>(r->GetNumberOfSecondaries()) : -1];
      if (r == nullptr) { continue; }
      std::map<int, GroupAcc> ev_groups;
      double e_out = 0.0, ex = 0.0;
      const int n = static_cast<int>(r->GetNumberOfSecondaries());
      for (int i = 0; i < n; ++i) {
        const G4DynamicParticle* d = r->GetSecondary(i)->GetParticle();
        const G4ParticleDefinition* def = d->GetDefinition();
        const int a = def->GetBaryonNumber();
        const int z = int(std::lround(def->GetPDGCharge() / eplus));
        const bool nucleus_like =
            (def->GetParticleType() == "nucleus") ||
            (a == 1 && (def->GetPDGEncoding() == 2212 || def->GetPDGEncoding() == 2112));
        const int key = ion_arm_key(def->GetPDGEncoding(), nucleus_like ? z : 0,
                                    nucleus_like ? a : 0);
        double m_pdg = def->GetPDGMass();
        if (def->GetParticleType() == "nucleus" && a > 1) {
          m_pdg = G4NucleiProperties::GetNuclearMass(a, z);
        }
        const double e = d->GetTotalEnergy();
        e_out += e;
        if (key == 1) { ex += d->GetMass() - m_pdg; }
        GroupAcc& g = ev_groups[ion_arm_group(key)];
        ++g.count;
        g.ke += e - m_pdg;
      }
      for (const auto& kv : ev_groups) {
        GroupAcc& g = groups[kv.first];
        g.count += kv.second.count;
        g.count2 += kv.second.count * kv.second.count;
        g.ke += kv.second.ke;
        g.ke2 += kv.second.ke * kv.second.ke;
      }
      // Group 0: every secondary, the final state's size and its kinetic energy.
      GroupAcc& all = groups[0];
      double ke_all = 0.0;
      for (const auto& kv : ev_groups) { ke_all += kv.second.ke; }
      all.count += n;
      all.count2 += static_cast<long long>(n) * n;
      all.ke += ke_all;
      all.ke2 += ke_all * ke_all;
      const double de = e_out - e_in;
      bal.ke += de;
      bal.ke2 += de * de;
      exc.ke += ex;
      exc.ke2 += ex * ex;
      r->Clear();
    }
    for (const auto* q : {"strings", "decayed", "fs"}) {
      const std::map<int, long long>& h = (q[0] == 's') ? h_strings
                                          : (q[0] == 'd') ? h_decayed : h_fs;
      for (const auto& kv : h) {
        std::fprintf(f, "%s,%d,%d,%.17g,%d,%d,%d,%s,%d,%lld,0,0,0\n", c.name, c.pz, c.pa,
                     c.per_a, c.tz, c.ta, c.n, q, kv.first, kv.second);
      }
    }
    for (const auto& kv : groups) {
      std::fprintf(f, "%s,%d,%d,%.17g,%d,%d,%d,group,%d,%lld,%lld,%.17g,%.17g\n", c.name, c.pz,
                   c.pa, c.per_a, c.tz, c.ta, c.n, kv.first, kv.second.count, kv.second.count2,
                   kv.second.ke, kv.second.ke2);
    }
    std::fprintf(f, "%s,%d,%d,%.17g,%d,%d,%d,event,0,0,0,%.17g,%.17g\n", c.name, c.pz, c.pa,
                 c.per_a, c.tz, c.ta, c.n, bal.ke, bal.ke2);
    std::fprintf(f, "%s,%d,%d,%.17g,%d,%d,%d,event,-1,0,0,%.17g,%.17g\n", c.name, c.pz, c.pa,
                 c.per_a, c.tz, c.ta, c.n, exc.ke, exc.ke2);
    const GroupAcc& all = groups[0];
    std::printf("  ion_arm %-18s <fs> %.2f, <dE> %.3f MeV\n", c.name,
                c.n ? double(all.count) / c.n : 0.0, c.n ? bal.ke / c.n : 0.0);
    std::fflush(stdout);
    std::fflush(f);
  }
  ftfp->SetTransport(const_cast<G4VIntraNuclearTransportModel*>(own));
  std::fclose(f);
}

}  // namespace

G4GPU_REGISTER_DUMP("ion_arm", "ion_arm.csv", dump_ion_arm);
