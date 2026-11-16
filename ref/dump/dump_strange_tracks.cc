// Oracle for P23a - the strange species as TRACKS: what a stepping kernel needs of each, asked of
// the running QBBC in B1's materials.
//
// Two files:
//
//   strange_hadron_tables.csv  hadron_tables.csv's columns, row for row, for the eight CHARGED
//                              strange baryons (sigma+, sigma-, xi-, omega- and their
//                              antiparticles): G4EmCalculator's restricted and total dE/dx, the
//                              range, and the delta-ray cross section through the hIoni each one
//                              carries. None of the eight has a table of its own - G4hIonisation
//                              scales the proton's or the antiproton's - so these numbers are the
//                              scaling, measured on Geant4's side. tests/test_hadron_range.cu
//                              tapes them as it tapes the deuteron and the triton.
//   strange_tracks_xs.csv      for all sixteen strange species QBBC transports and every material
//                              the dumper builds: the macroscopic hadElastic and <name>Inelastic
//                              cross sections the REGISTERED processes give through
//                              G4HadronicProcessStore, 1/mm, and G4Decay's in-flight mean free
//                              path, mm - the three discrete lengths a strange track draws.
//                              tests/test_strange_tracks.cu compares the port's per-volume
//                              functions and decay length against them.
//
// A file of its own and not rows in hadron_tables.csv, for the reason dump_strange.cc gives:
// test_hadron_range.cu reads that file through a species map that fails on a name it does not
// know, and adding rows to it would change what another test compares.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>

#include "G4Decay.hh"
#include "G4DynamicParticle.hh"
#include "G4EmCalculator.hh"
#include "G4HadronicProcessStore.hh"
#include "G4LossTableManager.hh"
#include "G4PhysicsTable.hh"
#include "G4PhysicsVector.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"

namespace {

/// G4Decay::GetMeanFreePath is protected; this exposes it unchanged.
struct DecayProbe : public G4Decay {
  using G4Decay::GetMeanFreePath;
};

G4ParticleDefinition* find_pdg(int pdg) {
  return G4ParticleTable::GetParticleTable()->FindParticle(pdg);
}

double electron_cut(const G4Material* m) {
  auto* table = G4ProductionCutsTable::GetProductionCutsTable();
  for (size_t i = 0; i < table->GetTableSize(); ++i) {
    const auto* couple = table->GetMaterialCutsCouple(static_cast<G4int>(i));
    if (couple->GetMaterial() != m) { continue; }
    return (*table->GetEnergyCutsVector(1))[i];
  }
  return 0.99e-3 * MeV;
}

// P23c appends the anti-neutron and the four light anti-nuclei, which the same two group kernels
// step: the anti-ions get the same ConstructBasicEmPhysics as the eight charged hyperons.
const int kCharged[] = {3222, 3112, 3312, 3334, -3222, -3112, -3312, -3334,
                        -1000010020, -1000010030, -1000020030, -1000020040};
const int kAll[] = {310, 130, 3122, 3212, 3322, -3122, -3212, -3322,
                    3222, 3112, 3312, 3334, -3222, -3112, -3312, -3334,
                    -2112, -1000010020, -1000010030, -1000020030, -1000020040};

void dump_tables(const DumpContext& ctx) {
  FILE* f = std::fopen("strange_hadron_tables.csv", "w");
  std::fprintf(f,
               "material,particle,mass_MeV,charge,energy_MeV,dedx_ioni,dedx_total,"
               "range_mm,delta_xs_per_mm,dedx_brem,dedx_pair,brem_xs_per_mm,"
               "pair_xs_per_mm,nuclear_dedx\n");
  G4EmCalculator calc;
  for (auto* m : ctx.materials) {
    const double cut = electron_cut(m);
    for (int pdg : kCharged) {
      const G4ParticleDefinition* p = find_pdg(pdg);
      if (p == nullptr) { continue; }
      // The grid of hadron_tables.csv: 1 keV .. 100 TeV, 20 points a decade.
      for (int i = 0; i <= 220; ++i) {
        const double e = 1e-3 * std::pow(10.0, i / 20.0);
        double di = 0, dt = 0, r = 0, dx = 0;
        try { di = calc.ComputeDEDX(e * MeV, p, "hIoni", m, cut) / (MeV / mm); } catch (...) {}
        try { dt = calc.GetDEDX(e * MeV, p, m) / (MeV / mm); } catch (...) {}
        try { r = calc.GetRange(e * MeV, p, m) / mm; } catch (...) {}
        try { dx = calc.ComputeCrossSectionPerVolume(e * MeV, p, "hIoni", m, cut) * mm; } catch (...) {}
        std::fprintf(f, "%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,0,0,0,0,0\n",
                     m->GetName().c_str(), p->GetParticleName().c_str(), p->GetPDGMass() / MeV,
                     p->GetPDGCharge() / eplus, e, di, dt, r, dx);
      }
    }
  }
  std::fclose(f);
}

void dump_xs(const DumpContext& ctx) {
  FILE* f = std::fopen("strange_tracks_xs.csv", "w");
  std::fprintf(f, "material,particle,pdg,energy_MeV,elastic_per_mm,inelastic_per_mm,"
                  "decay_mfp_mm\n");
  G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
  DecayProbe decay;
  for (auto* m : ctx.materials) {
    for (int pdg : kAll) {
      G4ParticleDefinition* p = find_pdg(pdg);
      if (p == nullptr) { continue; }
      // 1 MeV .. 100 TeV, 10 points a decade, plus 3, 6 and 12 GeV - the model windows' edges,
      // which are not where a cross section turns but are where a reader will look.
      for (int i = 0; i <= 80; ++i) {
        const double e = std::pow(10.0, i / 10.0);
        const double el = store->GetElasticCrossSectionPerVolume(p, e * MeV, m) * mm;
        const double in = store->GetInelasticCrossSectionPerVolume(p, e * MeV, m) * mm;
        // The track owns and deletes its dynamic particle.
        G4Track trk(new G4DynamicParticle(p, G4ThreeVector(0, 0, 1), e * MeV), 0.0,
                    G4ThreeVector());
        G4ForceCondition fc = NotForced;
        double mfp = -1;
        if (decay.IsApplicable(*p)) { mfp = decay.GetMeanFreePath(trk, 0.0, &fc) / mm; }
        std::fprintf(f, "%s,%s,%d,%.17g,%.17g,%.17g,%.17g\n", m->GetName().c_str(),
                     p->GetParticleName().c_str(), pdg, e, el, in, mfp);
      }
    }
  }
  std::fclose(f);
}

// Which tables a charged strange baryon's hIoni actually READS, per couple: is the vector the
// base particle's own object, and is it splined. dump_chargeodd.cc found that the negatives'
// tables are read with linear interpolation (no spline) because hBrems is registered first on
// their managers; a strange baryon has no hBrems, so whether it reads its base's linear vector
// or a splined one of its own is a question for the running Geant4 and not for the source.
void dump_vectors(const DumpContext& ctx) {
  FILE* f = std::fopen("strange_vectors.csv", "w");
  std::fprintf(f, "material,particle,base,dedx_shared,dedx_spline,range_shared,range_spline,"
                  "invrange_shared,invrange_spline,dedx_nodes\n");
  auto* ltm = G4LossTableManager::Instance();
  auto* pct = G4ProductionCutsTable::GetProductionCutsTable();
  for (auto* m : ctx.materials) {
    std::size_t idx = 0;
    for (std::size_t i = 0; i < pct->GetTableSize(); ++i) {
      if (pct->GetMaterialCutsCouple(static_cast<G4int>(i))->GetMaterial() == m) {
        idx = i;
        break;
      }
    }
    auto get = [idx](const G4PhysicsTable* t) -> const G4PhysicsVector* {
      return (nullptr != t && idx < t->size()) ? (*t)[idx] : nullptr;
    };
    for (int pdg : kCharged) {
      const G4ParticleDefinition* p = find_pdg(pdg);
      const G4VEnergyLossProcess* proc = (p != nullptr) ? ltm->GetEnergyLossProcess(p) : nullptr;
      if (proc == nullptr) { continue; }
      const G4ParticleDefinition* base = proc->BaseParticle();
      const G4VEnergyLossProcess* bproc =
          (base != nullptr) ? ltm->GetEnergyLossProcess(base) : nullptr;
      const G4PhysicsVector* dv = get(proc->DEDXTable());
      const G4PhysicsVector* rv = get(proc->RangeTableForLoss());
      const G4PhysicsVector* iv = get(proc->InverseRangeTable());
      const G4PhysicsVector* bdv = bproc ? get(bproc->DEDXTable()) : nullptr;
      const G4PhysicsVector* brv = bproc ? get(bproc->RangeTableForLoss()) : nullptr;
      const G4PhysicsVector* biv = bproc ? get(bproc->InverseRangeTable()) : nullptr;
      std::fprintf(f, "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d\n", m->GetName().c_str(),
                   p->GetParticleName().c_str(),
                   base ? base->GetParticleName().c_str() : "-", (dv && dv == bdv) ? 1 : 0,
                   dv ? (int)dv->GetSpline() : -1, (rv && rv == brv) ? 1 : 0,
                   rv ? (int)rv->GetSpline() : -1, (iv && iv == biv) ? 1 : 0,
                   iv ? (int)iv->GetSpline() : -1, dv ? (int)dv->GetVectorLength() : -1);
    }
  }
  std::fclose(f);
}

void dump_strange_tracks(const DumpContext& ctx) {
  dump_tables(ctx);
  dump_xs(ctx);
  dump_vectors(ctx);
}

}  // namespace

G4GPU_REGISTER_DUMP("strange_tracks",
                    "strange_hadron_tables.csv strange_tracks_xs.csv strange_vectors.csv",
                    dump_strange_tracks);
