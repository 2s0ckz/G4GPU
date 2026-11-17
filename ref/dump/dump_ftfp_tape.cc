// P28: the FTFP excess - Geant4's side of the per-interaction tape.
//
// WHAT IT IS FOR. Three example-B1 beams whose interactions are FTFP's (anti_proton 1 GeV,
// proton and pi+ 10 GeV) deposit 4-20% more dose in the phantom than Geant4 11.1.1 QBBC, while
// every beam on Binary or Bertini is inside its band (docs/RISK.md V244, V255). Every earlier FTF
// comparison stops at `G4FTFModel::Scatter` (ftf_modelstat_*.csv): the strings. Nothing compared
// what the WHOLE `G4TheoFSGenerator::ApplyYourself` hands `G4HadronicProcess` - the strings
// fragmented, the resonances decayed, `G4GeneratorPrecompoundInterface::Propagate`'s capture and
// the residual de-excited - against the port's `ftf::entry::apply`. This dump writes that, per
// species, for the projectiles and targets the beams reach.
//
// THE MODELS ARE QBBC'S OWN INSTANCES. In flight: `<particle>Inelastic`'s "FTFP"
// (G4HadronicBuilder::BuildFTFP_BERT for the anti-nucleons, G4HadronInelasticQBBC's for p and
// pi+). At rest: `hFritiofCaptureAtRest`'s, which G4HadronicAbsorptionFritiof builds with a
// G4TheoFSGenerator of its own; G4HadronStoppingProcess::AtRestDoIt hands it a projectile of zero
// kinetic energy, and so does this dump (`T_MeV` = 0).
//
// THE KINETIC ENERGY of a secondary is `E - m_PDG`: what `G4HadronicProcess::FillResult` makes of
// it once the off-shell correction (`ekin += m_dyn - m_PDG` past 1 keV) has run, which is the
// energy the transport receives. A nucleus's PDG mass is its ground state's
// (`G4NucleiProperties::GetNuclearMass`), the same mass the port's `definition_mass_of` uses.
//
// THE SPECIES KEY: the PDG code for every non-nucleus; 2212/2112 for a nucleon whatever made it;
// the nuclear code 100ZZZAAA0 for A = 2..4; and 1 for any heavier fragment (the residual and its
// pieces). `excit_MeV` sums the residual-class fragments' excitation (m_dyn - m_ground).
//
// ftfp_tape.csv       case,pdg,T_MeV,tz,ta,N,species,count,sum_ke,sum_ke2
//                     plus per case: species 0 = the event totals (count = sum of n_sec, sum_ke =
//                     sum of E_out - E_in, sum_ke2 = its square), species -1 = sum of the
//                     residual-class excitation.
// ftfp_tape_hist.csv  case,group,bin,count - kinetic-energy spectra, 30 log bins 0.01 MeV-10 GeV
//                     for the groups pi+-(1), pi0(2), p(3), n(4), gamma(5), A>=2 nuclei(6),
//                     everything else(7).
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include "G4DynamicParticle.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadronicInteraction.hh"
#include "G4HadronicProcess.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleus.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

namespace {

G4HadronicInteraction* find_model(int pdg, const std::string& process, const char* model) {
  G4ParticleDefinition* p = G4ParticleTable::GetParticleTable()->FindParticle(pdg);
  if (p == nullptr) { return nullptr; }
  G4ProcessVector* pv = p->GetProcessManager()->GetProcessList();
  for (G4int i = 0; i < (G4int)pv->size(); ++i) {
    auto* hp = dynamic_cast<G4HadronicProcess*>((*pv)[i]);
    if (hp == nullptr || hp->GetProcessName() != process) { continue; }
    for (G4HadronicInteraction* m : hp->GetHadronicInteractionList()) {
      if (m->GetModelName() == model) { return m; }
    }
  }
  return nullptr;
}

/// The species key the port's test computes the same way (tests/test_ftfp_tape.cu).
int species_key(int pdg, int z, int a) {
  if (a <= 0) { return pdg; }
  if (a == 1) { return (z == 1) ? 2212 : (z == 0 ? 2112 : pdg); }
  if (a <= 4) { return 1000000000 + z * 10000 + a * 10; }
  return 1;
}

int hist_group(int key) {
  if (key == 211 || key == -211) { return 1; }
  if (key == 111) { return 2; }
  if (key == 2212) { return 3; }
  if (key == 2112) { return 4; }
  if (key == 22) { return 5; }
  if (key == 1 || key > 1000000000) { return 6; }
  return 7;
}

int hist_bin(double ke) {
  if (!(ke > 0.01)) { return 0; }
  int b = 1 + int(std::floor(std::log10(ke / 0.01) * 5.0));  // 5 bins a decade from 0.01 MeV
  return (b > 30) ? 30 : b;
}

struct Acc {
  long long count = 0;
  double sum = 0.0, sum2 = 0.0;
};

void dump_ftfp_tape(const DumpContext&) {
  FILE* f = std::fopen("ftfp_tape.csv", "w");
  FILE* fh = std::fopen("ftfp_tape_hist.csv", "w");
  std::fprintf(f, "case,pdg,T_MeV,tz,ta,N,species,count,sum_ke,sum_ke2\n");
  std::fprintf(fh, "case,group,bin,count\n");

  struct Row { int pdg; double t; };
  const Row rows[] = {{-2212, 0.0},    {-2212, 100.0},  {-2212, 300.0}, {-2212, 500.0},
                      {-2212, 700.0},  {-2212, 1000.0}, {2212, 10000.0}, {211, 10000.0}};
  struct Tgt { int z, a; };
  const Tgt tgts[] = {{1, 1}, {6, 12}, {7, 14}, {8, 16}, {20, 40}};
  const char* env_n = std::getenv("G4GPU_FTFP_TAPE_N");
  const int N = (env_n != nullptr) ? std::atoi(env_n) : 5000;

  int case_id = 0;
  for (const Row& r : rows) {
    G4ParticleDefinition* p = G4ParticleTable::GetParticleTable()->FindParticle(r.pdg);
    const bool at_rest = (r.t == 0.0);
    const std::string proc = at_rest ? std::string("hFritiofCaptureAtRest")
                                     : p->GetParticleName() + "Inelastic";
    G4HadronicInteraction* model = find_model(r.pdg, proc, "FTFP");
    if (model == nullptr) {
      std::printf("  ftfp_tape: no FTFP on %s for %d\n", proc.c_str(), r.pdg);
      continue;
    }
    for (const Tgt& t : tgts) {
      ++case_id;
      CLHEP::HepRandom::setTheSeed(28000L + case_id);
      const double m_in = p->GetPDGMass();
      const double e_in = r.t + m_in + G4NucleiProperties::GetNuclearMass(t.a, t.z);
      std::map<int, Acc> acc;
      Acc tot, exc;
      std::map<int, long long> hist;  // group*100 + bin
      int done = 0;
      for (int ev = 0; ev < N; ++ev) {
        G4DynamicParticle dp(p, G4ThreeVector(0, 0, 1), r.t * MeV);
        G4HadProjectile proj(dp);
        G4Nucleus nucleus(t.a, t.z);
        G4HadFinalState* fs = nullptr;
        try {
          fs = model->ApplyYourself(proj, nucleus);
        } catch (...) {
          fs = nullptr;
        }
        if (fs == nullptr) { continue; }
        ++done;
        double e_out = 0.0, ex = 0.0;
        const int n = (int)fs->GetNumberOfSecondaries();
        // A primary left alive (kPrimaryUnchanged's analogue) carries its energy in the change.
        if (fs->GetStatusChange() == isAlive) { e_out += fs->GetEnergyChange() + m_in; }
        for (int i = 0; i < n; ++i) {
          const G4DynamicParticle* d = fs->GetSecondary(i)->GetParticle();
          const G4ParticleDefinition* def = d->GetDefinition();
          const int a = def->GetBaryonNumber();
          const int z = int(std::lround(def->GetPDGCharge() / eplus));
          const bool nucleus_like = (def->GetParticleType() == "nucleus") || (a == 1 && (def->GetPDGEncoding() == 2212 || def->GetPDGEncoding() == 2112));
          const int key = species_key(def->GetPDGEncoding(), nucleus_like ? z : 0,
                                      nucleus_like ? a : 0);
          double m_pdg = def->GetPDGMass();
          if (def->GetParticleType() == "nucleus" && a > 1) {
            m_pdg = G4NucleiProperties::GetNuclearMass(a, z);
          }
          const double e = d->GetTotalEnergy();
          const double ke = e - m_pdg;
          e_out += e;
          if (key == 1) { ex += d->GetMass() - m_pdg; }
          Acc& s = acc[key];
          ++s.count;
          s.sum += ke;
          s.sum2 += ke * ke;
          ++hist[hist_group(key) * 100 + hist_bin(ke)];
        }
        tot.count += n;
        const double de = e_out - e_in;
        tot.sum += de;
        tot.sum2 += de * de;
        exc.sum += ex;
        exc.sum2 += ex * ex;
        fs->Clear();
      }
      std::fprintf(f, "%d,%d,%.17g,%d,%d,%d,0,%lld,%.17g,%.17g\n", case_id, r.pdg, r.t, t.z, t.a,
                   done, tot.count, tot.sum, tot.sum2);
      std::fprintf(f, "%d,%d,%.17g,%d,%d,%d,-1,0,%.17g,%.17g\n", case_id, r.pdg, r.t, t.z, t.a,
                   done, exc.sum, exc.sum2);
      for (const auto& kv : acc) {
        std::fprintf(f, "%d,%d,%.17g,%d,%d,%d,%d,%lld,%.17g,%.17g\n", case_id, r.pdg, r.t, t.z,
                     t.a, done, kv.first, kv.second.count, kv.second.sum, kv.second.sum2);
      }
      for (const auto& kv : hist) {
        std::fprintf(fh, "%d,%d,%d,%lld\n", case_id, kv.first / 100, kv.first % 100, kv.second);
      }
      std::printf("  ftfp_tape case %2d pdg %6d T %7.1f on (%d,%d): %d events, <n> %.2f, "
                  "<dE> %.3f MeV\n", case_id, r.pdg, r.t, t.z, t.a, done,
                  done ? double(tot.count) / done : 0.0, done ? tot.sum / done : 0.0);
      std::fflush(stdout);
      std::fflush(f);
      std::fflush(fh);
    }
  }
  std::fclose(f);
  std::fclose(fh);
}

}  // namespace

G4GPU_REGISTER_DUMP("ftfp_tape", "ftfp_tape.csv ftfp_tape_hist.csv", dump_ftfp_tape);
