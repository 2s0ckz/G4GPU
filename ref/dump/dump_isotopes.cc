// Oracle for P8b's isotope abundances: what a NIST-built G4Element actually hands a target draw.
//
// Registered through dump_registry.hh, so nothing shared is edited. Writes:
//
//   isotopes.csv         EXACT. Every element G4NistElementBuilder carries, built through
//                        G4NistManager::FindOrBuildElement(Z), with its isotope list and its
//                        relative-abundance vector read straight out of the G4Element - plus
//                        GetIsotopeAbundance(Z, A), GetNumberOfNistIsotopes(Z) and
//                        GetNistFirstIsotopeN(Z), which are the pre-BuildElement numbers.
//                        Both vectors, because they are DIFFERENT: the abundances go through a
//                        second normalisation inside G4Element::AddIsotope, over the surviving
//                        isotopes rather than over all the tabulated ones, and a port that
//                        reproduced only the first would be wrong in the last places on every
//                        multi-isotope element.
//   isotope_zanda.csv    STATISTICAL. G4CrossSectionDataStore::SampleZandA at a fixed seed,
//                        200,000 draws per (material, data set, energy), for water, air,
//                        compact bone and lead, with the partial cross sections and the
//                        per-isotope cross sections that produced them. Three data sets,
//                        chosen because they take the three different SelectIsotope branches.
//
// WHY THE ELEMENT TABLE IS NOT DERIVABLE FROM ref/oracle/elastic_zanda.csv
//
// That file carries abundances too, and `tests/test_hadronic_process.cu` already checks the
// SELECTION against them - but it feeds Geant4's own abundances in from the CSV, so it tests the
// draw and not the table. The port had no table: `data/natural_isotopes.hh` is deliberately the
// SET and not the weights and `data/isotope_list.hh` is amin/amax/aeff. So the thing to compare
// is the 311 numbers themselves, over every element and not over the four elements of water.
//
// WHY THREE DATA SETS
//
// G4CrossSectionDataStore::SampleZandA has three outcomes and they are decided by the data set,
// not by the material (see src/physics/hadronic/xs/sample_za.cuh):
//
//   G4NeutronElasticXS     SelectIsotope always samples by ABUNDANCE ALONE - it has no
//                          per-isotope data at all.
//   G4NeutronCaptureXS     samples by `abundance_j * IsoCrossSection_j` inside its
//                          amin/amax window, and by abundance alone outside it. Lead is the
//                          interesting material: Z = 82 is inside every window, and its four
//                          isotopes have capture cross sections that differ by more than their
//                          abundances do, so the two branches give visibly different
//                          frequencies and a port that took the wrong one fails on lead while
//                          passing on water.
//   G4ParticleInelasticXS  the proton set, so the projectile is not always a neutron: the
//                          element loop is over the same abundances but the partial cross
//                          sections that pick the ELEMENT are a different shape, which is what
//                          separates an error in the element draw from one in the isotope draw.
//
// THE SEED IS FIXED AND THE COUNTS ARE THE COMPARISON
//
// A frequency is not reproducible across two random engines, so what is compared is the count
// against a binomial expectation with the port's own draw at its own seed: the oracle's counts
// and the analytic weights are both written, and tests/test_isotopes.cu checks the port against
// both. Agreeing with the weights while disagreeing with Geant4's counts would mean the weights
// had been read out of the same misreading as the port's.

#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "G4CrossSectionDataStore.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4Isotope.hh"
#include "G4Material.hh"
#include "G4NeutronCaptureXS.hh"
#include "G4NeutronElasticXS.hh"
#include "G4Neutron.hh"
#include "G4NistManager.hh"
#include "G4Nucleus.hh"
#include "G4ParticleInelasticXS.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

namespace {

void dump_isotope_table(const DumpContext&) {
  FILE* f = std::fopen("isotopes.csv", "w");
  std::fprintf(f, "Z,symbol,n_nist_isotopes,first_isotope_N,n_element_isotopes,j,A,"
                  "rel_abundance,nist_abundance,aeff_amu,element_A_amu\n");

  G4NistManager* nist = G4NistManager::Instance();
  // Z STOPS AT 104 AND THAT IS NOT AN ARBITRARY CEILING.
  //
  // G4NistElementBuilder carries 107 elements (maxNumElements = 108), and for every element
  // above uranium its W[] array holds a fabricated 100 on one isotope - Db is
  // `{0,0,0,0,0,0,0,100,0,0,0}` on A = 255..265, so Db-262 comes out with abundance 1.0 - so
  // BuildElement DOES create a G4Element for them rather than skipping them for having no
  // natural isotopes. G4Element::AddIsotope then calls
  // `G4AtomicShells::GetNumberOfShells(iz)`, whose tables are `[105]`, i.e. Z = 0..104, and
  // Z = 105 raises a FatalException and aborts the process. Found by running this dump to 107:
  // it killed g4dump.exe with "mat060 ... Atomic number out of range Z= 105" after writing
  // every earlier line. So 104 is the highest Z a G4Element can exist for in 11.1.1, whatever
  // the NIST table holds, and the port's table says the same thing.
  for (G4int Z = 1; Z <= 104; ++Z) {
    const G4Element* e = nist->FindOrBuildElement(Z);
    if (e == nullptr) { continue; }
    const G4int nnist = nist->GetNumberOfNistIsotopes(Z);
    const G4int n0 = nist->GetNistFirstIsotopeN(Z);
    const std::size_t ni = e->GetNumberOfIsotopes();
    const G4double* w = e->GetRelativeAbundanceVector();
    for (std::size_t j = 0; j < ni; ++j) {
      const G4Isotope* iso = e->GetIsotope(j);
      std::fprintf(f, "%d,%s,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g\n", Z,
                   e->GetName().c_str(), nnist, n0, int(ni), int(j), iso->GetN(), w[j],
                   nist->GetIsotopeAbundance(Z, iso->GetN()), nist->GetAtomicMassAmu(Z),
                   e->GetA() / (g / mole));
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

struct SetSpec {
  const char* name;
  G4VCrossSectionDataSet* ds;
  const G4ParticleDefinition* particle;
};

void dump_zanda_frequencies(const DumpContext& ctx) {
  FILE* f = std::fopen("isotope_zanda.csv", "w");
  // iso_counts is `A:abundance:iso_xs_mm2:count` per isotope, semicolon separated - the same
  // encoding ref/oracle/elastic_zanda.csv uses, so one parser reads both.
  std::fprintf(f, "dataset,particle,material,ekin_MeV,nelm,total_xs_per_mm,i,Z,"
                  "natoms_per_mm3,xs_per_atom_mm2,xsecelm_cumulative,ntrial,n_selected,"
                  "n_iso,iso_counts\n");

  const G4ParticleDefinition* neutron = G4Neutron::Neutron();
  const G4ParticleDefinition* proton = G4Proton::Proton();

  auto* nElas = new G4NeutronElasticXS();
  auto* nCap = new G4NeutronCaptureXS();
  auto* pInel = new G4ParticleInelasticXS(proton);
  nElas->BuildPhysicsTable(*neutron);
  nCap->BuildPhysicsTable(*neutron);
  pInel->BuildPhysicsTable(*proton);

  const SetSpec sets[] = {
      {"G4NeutronElasticXS", nElas, neutron},
      {"G4NeutronCaptureXS", nCap, neutron},
      {"G4ParticleInelasticXS", pInel, proton},
  };

  // Water, air and compact bone come from the dumper's own detector, in its fixed order, so
  // they are the same materials every other oracle CSV is keyed by. Lead is built here because
  // B1 has no lead and a single high-Z element is what separates the two isotope branches.
  std::vector<const G4Material*> mats;
  mats.push_back(ctx.materials[1]);  // G4_WATER
  mats.push_back(ctx.materials[0]);  // G4_AIR
  mats.push_back(ctx.materials[3]);  // G4_BONE_COMPACT_ICRU
  mats.push_back(G4NistManager::Instance()->FindOrBuildMaterial("G4_Pb"));

  // Three energies per set. 1 MeV and 14 MeV are below G4NeutronGeneralProcess's 20 MeV middle
  // energy, where capture is in the sum at all; 100 MeV is above it and is also inside every
  // set's element table.
  const double energies[] = {1.0, 14.0, 100.0};
  const int kN = 200000;

  // One fixed seed for the whole dump, set here rather than left to whatever the previous dump
  // left behind: this is the only dump in the directory whose output depends on the engine
  // state at entry, and a registry that globs the directory does not fix the order dumps run in.
  CLHEP::HepRandom::setTheSeed(20260911);

  for (const SetSpec& s : sets) {
    for (const G4Material* mat : mats) {
      const std::size_t nelm = mat->GetNumberOfElements();
      const G4double* natoms = mat->GetVecNbOfAtomsPerVolume();
      for (double ek : energies) {
        G4CrossSectionDataStore store;
        store.AddDataSet(s.ds);
        G4DynamicParticle dp(s.particle, G4ThreeVector(0, 0, 1), ek * MeV);
        const double total = store.ComputeCrossSection(&dp, mat);

        std::vector<int> counts(nelm, 0);
        std::vector<std::vector<int>> iso_counts(nelm);
        for (std::size_t i = 0; i < nelm; ++i) {
          iso_counts[i].assign(mat->GetElement(i)->GetNumberOfIsotopes(), 0);
        }
        for (int i = 0; i < kN; ++i) {
          G4Nucleus nucleus;
          const G4Element* e = store.SampleZandA(&dp, mat, nucleus);
          for (std::size_t k = 0; k < nelm; ++k) {
            if (mat->GetElement(k) == e) {
              ++counts[k];
              const G4Isotope* iso = nucleus.GetIsotope();
              for (std::size_t j = 0; j < e->GetNumberOfIsotopes(); ++j) {
                if (e->GetIsotope(j) == iso) { ++iso_counts[k][j]; }
              }
              break;
            }
          }
        }

        for (std::size_t i = 0; i < nelm; ++i) {
          const G4Element* e = mat->GetElement(i);
          const double per_atom = store.GetCrossSection(&dp, e, mat);
          double cum = 0.0;
          for (std::size_t k = 0; k <= i; ++k) {
            const double x = natoms[k] * store.GetCrossSection(&dp, mat->GetElement(k), mat);
            cum += (x > 0.0) ? x : 0.0;
          }
          std::string isos;
          for (std::size_t j = 0; j < e->GetNumberOfIsotopes(); ++j) {
            const G4Isotope* iso = e->GetIsotope(j);
            const double ab = e->GetRelativeAbundanceVector()[j];
            const double xs =
                store.GetCrossSection(&dp, e->GetZasInt(), iso->GetN(), iso, e, mat);
            char buf[192];
            std::snprintf(buf, sizeof(buf), "%s%d:%.17g:%.17g:%d", j ? ";" : "", iso->GetN(),
                          ab, xs / (mm * mm), iso_counts[i][j]);
            isos += buf;
          }
          std::fprintf(f, "%s,%s,%s,%.17g,%d,%.17g,%d,%d,%.17g,%.17g,%.17g,%d,%d,%d,%s\n",
                       s.name, s.particle->GetParticleName().c_str(),
                       mat->GetName().c_str(), ek, int(nelm), total * mm, int(i),
                       e->GetZasInt(), natoms[i] * mm * mm * mm, per_atom / (mm * mm),
                       cum * mm, kN, counts[i], int(e->GetNumberOfIsotopes()), isos.c_str());
        }
      }
    }
  }
  std::fclose(f);
}

void dump_isotopes(const DumpContext& ctx) {
  dump_isotope_table(ctx);
  dump_zanda_frequencies(ctx);
}

}  // namespace

G4GPU_REGISTER_DUMP("isotopes", "isotopes.csv, isotope_zanda.csv", dump_isotopes);
