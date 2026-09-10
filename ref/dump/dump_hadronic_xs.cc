// Oracle for P2 - every hadronic cross section QBBC asks for, from Geant4 itself.
//
// docs/HADRONIC_PLAN.md section 5, package P2. Eleven CSVs, one per layer of the stack, so
// that a disagreement can be localised to a layer rather than to "the proton inelastic cross
// section":
//
//   had_hnxsc.csv            G4HadronNucleonXsc - the bottom of everything
//   had_radii.csv            G4NuclearRadii and G4NucleiProperties::GetNuclearMass
//   had_masses.csv           every nuclide G4NucleiProperties::IsInStableTable admits, because
//                            had_radii.csv only reaches the 398 with a G4PARTICLEXS file and
//                            the port transcribes all 3353
//   had_coulomb.csv          the three Coulomb factors, which are threshold behaviour
//   had_ggcomp.csv           G4ComponentGGHadronNucleusXsc and G4ComponentGGNuclNuclXsc
//   had_bgg.csv              the four BGG data sets, end to end
//   had_particlexs.csv       the five G4PARTICLEXS data sets, per element
//   had_particlexs_iso.csv   the same, per isotope, for every (Z, A) with a file
//   had_neutron_general.csv  G4NeutronGeneralProcess's own five tables, read out of the
//                            process after initialisation - not recomputed
//   had_matelem.csv          each material's elements and atom densities, because the
//                            neutron general table is macroscopic and depends on them
//   had_xspeaks.csv          what G4HadronicProcess *does* tabulate: the cross-section peak
//                            structure of the integral method
//
// WHY THE NEUTRON GENERAL TABLE IS STORED AND READ BACK RATHER THAN RECOMPUTED
//
// The plan's requirement is "the general process's own table values as the oracle for it, not
// just the raw data sets" (section 5, P2 deliverable 7). Rebuilding the grid here from
// fMinEnergy, fMiddleEnergy, nLowE and nHighE would be a second implementation of the same
// arithmetic as the port's, and the two could be wrong together - which is precisely the
// failure docs/RISK.md V5 is about. So the table comes out of the object: StorePhysicsTable is
// public on G4NeutronGeneralProcess, it writes G4PhysicsVector::Store in BINARY (full double
// precision, unlike the ascii form's 12 digits), and the five files are read back here with a
// twenty-line reader for a format that is three doubles, a size_t and 2n doubles. The grid in
// the CSV is therefore Geant4's own binVector, not a formula.
//
// TWELVE POINTS PER DECADE, AND THE BOUNDARIES ON TOP OF THEM
//
// Every energy scan is twelve per decade, as the plan asks. That is not enough on its own:
// each class hands over to another model at a specific energy - 14 MeV, 20 MeV, 91 GeV, the
// top of each data table, 150 MeV for gamma, 20 MeV for the neutron general process's two
// zones - and a logarithmic scan lands on none of them. So the exact boundary energies are
// appended to every scan, and where a boundary is a *momentum* in the source (10, 100, 373,
// 1000 GeV/c in G4HadronNucleonXsc) the corresponding kinetic energy is computed per particle
// and appended too.
#include "dump_registry.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "G4Alpha.hh"
#include "G4AntiNeutron.hh"
#include "G4AntiProton.hh"
#include "G4BGGNucleonElasticXS.hh"
#include "G4BGGNucleonInelasticXS.hh"
#include "G4BGGPionElasticXS.hh"
#include "G4BGGPionInelasticXS.hh"
#include "G4ComponentGGHadronNucleusXsc.hh"
#include "G4ComponentGGNuclNuclXsc.hh"
#include "G4Deuteron.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4Gamma.hh"
#include "G4GammaNuclearXS.hh"
#include "G4HadXSHelper.hh"
#include "G4HadXSTypes.hh"
#include "G4HadronNucleonXsc.hh"
#include "G4HadronicParameters.hh"
#include "G4HadronicProcess.hh"
#include "G4He3.hh"
#include "G4IonTable.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4KaonZeroLong.hh"
#include "G4KaonZeroShort.hh"
#include "G4Material.hh"
#include "G4Neutron.hh"
#include "G4NeutronCaptureXS.hh"
#include "G4NeutronElasticXS.hh"
#include "G4NeutronGeneralProcess.hh"
#include "G4NeutronInelasticXS.hh"
#include "G4NistManager.hh"
#include "G4NuclearRadii.hh"
#include "G4NucleiProperties.hh"
#include "G4ParticleInelasticXS.hh"
#include "G4PhysListUtil.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"
#include "G4Triton.hh"

namespace {

const double kMm2 = mm * mm;

// G4IsotopeList.hh, which is a static header and so not linkable - copied here for the
// isotope loop only. The port has its own transcription in src/data/isotope_list.hh, and
// had_particlexs_iso.csv is what checks the two against each other: a wrong amin[Z] here and
// a wrong amin[Z] there would agree, but a wrong one in only one of them shifts every isotope
// index for that Z and every value in the row.
const int kAmin[] = {
  0,
  1,   3,   6,   9,  10,  12,  14,  16,  19,  20,
 23,  24,  27,  27,  31,  32,  35,  36,  39,  40,
 45,  46,  50,  50,  55,  54,  59,  58,  63,  64,
 69,  70,  75,  74,  79,  78,  85,  84,  89,  90,
 93,  92,  98,  96, 103, 102, 107, 106, 113, 112,
121, 120, 127, 124, 133, 130, 138, 136, 141, 142,
145, 144, 151, 152, 158, 156, 165, 162, 169, 168,
175, 174, 180, 180, 185, 184, 191, 190, 197, 196,
203, 204, 209, 209, 210, 222, 223, 226, 227, 232,
231, 233, 237, 238};
const int kAmax[] = {
  0,
  3,   4,   7,   9,  11,  14,  15,  18,  19,  22,
 23,  26,  27,  30,  31,  36,  37,  40,  41,  48,
 45,  50,  51,  54,  55,  58,  59,  64,  65,  70,
 71,  76,  75,  82,  81,  86,  87,  90,  89,  96,
 94, 100,  98, 104, 103, 110, 109, 116, 115, 124,
123, 130, 129, 136, 137, 138, 139, 142, 141, 150,
145, 154, 153, 160, 159, 164, 165, 170, 169, 176,
176, 180, 181, 186, 187, 192, 193, 198, 197, 204,
205, 208, 209, 209, 210, 222, 223, 226, 227, 232,
231, 238, 237, 244};

/// Twelve points per decade from `e0` to `e1` inclusive of `e0`, stopping at `e1`.
std::vector<double> decade_scan(double e0, double e1, int per_decade = 12) {
  std::vector<double> v;
  const double step = std::pow(10.0, 1.0 / per_decade);
  for (double e = e0; e <= e1 * 1.0000000001; e *= step) { v.push_back(e); }
  return v;
}

/// Adds the exact boundary energies, then sorts and de-duplicates. A logarithmic scan lands
/// on none of them and they are where every one of these classes changes model.
void add_and_sort(std::vector<double>& v, const std::vector<double>& extra, double lo,
                  double hi) {
  for (double e : extra) {
    if (e >= lo && e <= hi) { v.push_back(e); }
  }
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

/// The kinetic energy at which a particle of mass `m` has laboratory momentum `plab`.
/// G4HadronNucleonXsc branches on pLab in GeV/c; the port and this dump must land on the same
/// side of each branch, so the boundaries are converted rather than guessed at.
double tkin_of_plab(double plab_GeV, double m) {
  const double p = plab_GeV * GeV;
  return std::sqrt(p * p + m * m) - m;
}

struct Part {
  const char* name;
  const G4ParticleDefinition* def;
};

// --------------------------------------------------------------- binary table reader
//
// G4PhysicsTable::StorePhysicsTable(file, ascii=false) writes:
//   size_t   number of vectors
//   then per vector:  G4int type, G4double edgeMin, G4double edgeMax, size_t numberOfNodes,
//                     size_t size, then 2*size G4double as (bin, data) pairs
// G4PhysicsVector::Store is the second half of that. Read back rather than trusted: the CSV
// carries the nodes themselves, so a misread shows up as a garbage energy and not as a subtle
// value shift.
struct StoredVector {
  double edge_min = 0, edge_max = 0;
  std::vector<double> e, v;
};

bool read_stored_table(const std::string& path, std::vector<StoredVector>& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) { return false; }
  std::size_t nvec = 0;
  if (std::fread(&nvec, sizeof nvec, 1, f) != 1 || nvec == 0 || nvec > 100000) {
    std::fclose(f);
    return false;
  }
  out.clear();
  for (std::size_t k = 0; k < nvec; ++k) {
    G4int type = 0;
    std::size_t nnodes = 0, siz = 0;
    StoredVector sv;
    if (std::fread(&type, sizeof type, 1, f) != 1) { std::fclose(f); return false; }
    if (std::fread(&sv.edge_min, sizeof(double), 1, f) != 1) { std::fclose(f); return false; }
    if (std::fread(&sv.edge_max, sizeof(double), 1, f) != 1) { std::fclose(f); return false; }
    if (std::fread(&nnodes, sizeof nnodes, 1, f) != 1) { std::fclose(f); return false; }
    if (std::fread(&siz, sizeof siz, 1, f) != 1 || siz == 0 || siz > 1000000) {
      std::fclose(f);
      return false;
    }
    std::vector<double> pairs(2 * siz);
    if (std::fread(pairs.data(), sizeof(double), 2 * siz, f) != 2 * siz) {
      std::fclose(f);
      return false;
    }
    sv.e.resize(siz);
    sv.v.resize(siz);
    for (std::size_t i = 0; i < siz; ++i) {
      sv.e[i] = pairs[2 * i];
      sv.v[i] = pairs[2 * i + 1];
    }
    out.push_back(std::move(sv));
  }
  std::fclose(f);
  return true;
}

// --------------------------------------------------------------- the dump

void dump_hadronic_xs(const DumpContext& ctx) {
  auto* nist = G4NistManager::Instance();
  const G4ParticleDefinition* pro = G4Proton::Proton();
  const G4ParticleDefinition* neu = G4Neutron::Neutron();

  // ------------------------------------------------------------ 1. G4HadronNucleonXsc
  {
    G4HadronNucleonXsc hn;
    FILE* f = std::fopen("had_hnxsc.csv", "w");
    std::fprintf(f, "method,particle,nucleon,energy_MeV,total_mm2,elastic_mm2,"
                    "inelastic_mm2\n");
    const Part parts[] = {
        {"proton", pro},
        {"neutron", neu},
        {"anti_proton", G4AntiProton::AntiProton()},
        {"anti_neutron", G4AntiNeutron::AntiNeutron()},
        {"pi+", G4PionPlus::PionPlus()},
        {"pi-", G4PionMinus::PionMinus()},
        {"kaon+", G4KaonPlus::KaonPlus()},
        {"kaon-", G4KaonMinus::KaonMinus()},
        {"kaon0S", G4KaonZeroShort::KaonZeroShort()},
        {"kaon0L", G4KaonZeroLong::KaonZeroLong()},
        {"gamma", G4Gamma::Gamma()},
    };
    const Part nucleons[] = {{"proton", pro}, {"neutron", neu}};
    // Every pLab boundary that appears in HadronNucleonXscNS, KaonNucleonXscVG and
    // HadronNucleonXscPDG, in GeV/c, plus the two energy boundaries (ekinmin, ekinmaxQB).
    const double plabs[] = {0.02,  0.1,   0.28,  0.38,  0.395676, 0.4,  0.48, 0.5,
                            0.631, 0.65,  0.68,  0.72,  0.73,     0.77, 0.78, 0.8,
                            0.85,  0.88,  0.94,  0.95,  0.98,     1.01, 1.03, 1.05,
                            1.15,  1.3,   1.4,   1.63,  2.0,      2.1,  3.5,  10.0,
                            100.0, 373.0, 1000.0};
    for (const Part& p : parts) {
      std::vector<double> extra = {0.1, 100.0};  // ekinmin, ekinmaxQB
      for (double pl : plabs) { extra.push_back(tkin_of_plab(pl, p.def->GetPDGMass())); }
      std::vector<double> es = decade_scan(1e-3, 1e8);
      add_and_sort(es, extra, 1e-3, 1e8);

      const bool is_kaon = (p.def == G4KaonPlus::KaonPlus() ||
                            p.def == G4KaonMinus::KaonMinus() ||
                            p.def == G4KaonZeroShort::KaonZeroShort() ||
                            p.def == G4KaonZeroLong::KaonZeroLong());
      for (const Part& n : nucleons) {
        for (double e : es) {
          auto row = [&](const char* method) {
            std::fprintf(f, "%s,%s,%s,%.17g,%.17g,%.17g,%.17g\n", method, p.name, n.name, e,
                         hn.GetTotalHadronNucleonXsc() / kMm2,
                         hn.GetElasticHadronNucleonXsc() / kMm2,
                         hn.GetInelasticHadronNucleonXsc() / kMm2);
          };
          hn.HadronNucleonXscPDG(p.def, n.def, e * MeV);
          row("PDG");
          hn.HadronNucleonXscNS(p.def, n.def, e * MeV);
          row("NS");
          hn.HadronNucleonXsc(p.def, n.def, e * MeV);
          row("Dispatch");
          if (is_kaon) {
            hn.KaonNucleonXscNS(p.def, n.def, e * MeV);
            row("KaonNS");
            hn.KaonNucleonXscGG(p.def, n.def, e * MeV);
            row("KaonGG");
            hn.KaonNucleonXscVG(p.def, n.def, e * MeV);
            row("KaonVG");
          }
        }
      }
    }
    std::fclose(f);
  }

  // ------------------------------------------------------------ 2. radii and masses
  {
    FILE* f = std::fopen("had_radii.csv", "w");
    std::fprintf(f, "Z,A,explicit_mm,radius_mm,radius_rms_mm,radius_nngg_mm,radius_ecs_mm,"
                    "radius_hngg_mm,radius_kngg_mm,radius_nd_mm,radius_cb_mm,"
                    "nuclear_mass_MeV,in_ame_table\n");
    for (int z = 1; z <= 92; ++z) {
      // Every A a per-isotope data file can exist for, plus the rounded mean the BGG classes
      // use as theA[Z] - which is not always inside [amin, amax].
      std::vector<int> as;
      for (int a = kAmin[z]; a <= kAmax[z]; ++a) { as.push_back(a); }
      as.push_back(G4lrint(nist->GetAtomicMassAmu(z)));
      std::sort(as.begin(), as.end());
      as.erase(std::unique(as.begin(), as.end()), as.end());
      for (int a : as) {
        const bool in_ame = G4NucleiProperties::IsInStableTable(a, z);
        std::fprintf(f, "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                        "%d\n",
                     z, a, G4NuclearRadii::ExplicitRadius(z, a) / mm,
                     G4NuclearRadii::Radius(z, a) / mm, G4NuclearRadii::RadiusRMS(z, a) / mm,
                     G4NuclearRadii::RadiusNNGG(z, a) / mm,
                     G4NuclearRadii::RadiusECS(z, a) / mm,
                     G4NuclearRadii::RadiusHNGG(a) / mm, G4NuclearRadii::RadiusKNGG(a) / mm,
                     G4NuclearRadii::RadiusND(a) / mm, G4NuclearRadii::RadiusCB(z, a) / mm,
                     G4NucleiProperties::GetNuclearMass(a, z) / MeV, in_ame ? 1 : 0);
      }
    }
    std::fclose(f);
  }

  // ------------------------------------------------------------ 2b. EVERY nuclide AME2012 has
  //
  // had_radii.csv above covers only the (Z, A) pairs a G4PARTICLEXS file can exist for - 398 of
  // them. src/data/nuclei_mass_ame12.hh is a 3353-entry hand transcription of
  // G4NucleiPropertiesTableAME12's mass-excess table, so 2955 of its entries would be compared
  // against nothing: a single mistyped excess, or a row shifted by one in the packed index,
  // would be invisible where the isotope loop does not go and wrong wherever a de-excitation
  // fragment or an ion projectile lands.
  //
  // So: every (Z, A) Geant4 says it has. IsInStableTable is the table's own membership test, so
  // this dumps exactly the table rather than a guess at its extent, and the port's
  // `nuclear_mass_known` has to agree with it entry for entry.
  {
    FILE* f = std::fopen("had_masses.csv", "w");
    std::fprintf(f, "Z,A,nuclear_mass_MeV\n");
    long n = 0;
    // G4NucleiPropertiesTableAME12 is indexed for A up to 273 and Z up to 110; the loop is
    // wider than that on purpose, so that a nuclide Geant4 has and this range would have
    // missed shows up as a row the port refuses and the oracle does not carry.
    for (int a = 1; a <= 295; ++a) {
      for (int z = 0; z <= a && z <= 120; ++z) {
        if (!G4NucleiProperties::IsInStableTable(a, z)) { continue; }
        std::fprintf(f, "%d,%d,%.17g\n", z, a,
                     G4NucleiProperties::GetNuclearMass(a, z) / MeV);
        ++n;
      }
    }
    std::fclose(f);
    std::printf("dump_hadronic_xs: had_masses.csv has %ld nuclides\n", n);
  }

  // ------------------------------------------------------------ 3. the Coulomb factors
  {
    G4HadronNucleonXsc hn;
    FILE* f = std::fopen("had_coulomb.csv", "w");
    std::fprintf(f, "form,particle,Z,A,energy_MeV,factor\n");
    const Part parts[] = {{"proton", pro},
                          {"neutron", neu},
                          {"pi+", G4PionPlus::PionPlus()},
                          {"pi-", G4PionMinus::PionMinus()},
                          {"kaon+", G4KaonPlus::KaonPlus()},
                          {"alpha", G4Alpha::Alpha()}};
    // The particle radius ladder is 0.895 / 0.663 / 0.340 / 0.5 fm keyed on |PDG|, so an
    // alpha is here to exercise the default 0.5 fm arm that no nucleon or pion reaches.
    std::vector<double> es = decade_scan(1e-3, 1e5);
    add_and_sort(es, {14.0, 20.0, 100.0}, 1e-3, 1e5);
    for (const Part& p : parts) {
      for (int z = 1; z <= 92; ++z) {
        const int a = G4lrint(nist->GetAtomicMassAmu(z));
        for (double e : es) {
          std::fprintf(f, "nucleus,%s,%d,%d,%.17g,%.17g\n", p.name, z, a, e,
                       G4NuclearRadii::CoulombFactor(z, a, p.def, e * MeV));
        }
      }
      for (double e : es) {
        std::fprintf(f, "nucleon,%s,1,1,%.17g,%.17g\n", p.name, e,
                     G4NuclearRadii::CoulombFactor(p.def, pro, e * MeV));
        std::fprintf(f, "nucleon_n,%s,0,1,%.17g,%.17g\n", p.name, e,
                     G4NuclearRadii::CoulombFactor(p.def, neu, e * MeV));
        std::fprintf(f, "barrier,%s,1,1,%.17g,%.17g\n", p.name, e,
                     hn.CoulombBarrier(p.def, pro, e * MeV));
      }
    }
    std::fclose(f);
  }

  // ------------------------------------------------------------ 4. the GG components
  {
    G4ComponentGGHadronNucleusXsc gg;
    G4ComponentGGNuclNuclXsc nn;
    FILE* f = std::fopen("had_ggcomp.csv", "w");
    std::fprintf(f, "component,particle,Z,A,energy_MeV,total_mm2,inelastic_mm2,elastic_mm2,"
                    "production_mm2,diffraction_mm2\n");
    const Part hadrons[] = {{"proton", pro},
                            {"neutron", neu},
                            {"pi+", G4PionPlus::PionPlus()},
                            {"pi-", G4PionMinus::PionMinus()},
                            {"kaon+", G4KaonPlus::KaonPlus()},
                            {"kaon-", G4KaonMinus::KaonMinus()}};
    const Part ions[] = {{"deuteron", G4Deuteron::Deuteron()},
                         {"triton", G4Triton::Triton()},
                         {"He3", G4He3::He3()},
                         {"alpha", G4Alpha::Alpha()},
                         {"C12", G4IonTable::GetIonTable()->GetIon(6, 12, 0.0)},
                         {"Fe56", G4IonTable::GetIonTable()->GetIon(26, 56, 0.0)}};
    std::vector<double> es = decade_scan(1e-1, 1e8);
    add_and_sort(es, {14.0, 20.0, 91000.0, 20000.0, 100.0}, 1e-1, 1e8);
    for (const Part& p : hadrons) {
      for (int z = 1; z <= 92; ++z) {
        const int a = G4lrint(nist->GetAtomicMassAmu(z));
        for (double e : es) {
          gg.ComputeCrossSections(p.def, e * MeV, z, a);
          std::fprintf(f, "GGHadronNucleus,%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                       p.name, z, a, e, gg.GetTotalGlauberGribovXsc() / kMm2,
                       gg.GetInelasticGlauberGribovXsc() / kMm2,
                       gg.GetElasticGlauberGribovXsc() / kMm2,
                       gg.GetProductionGlauberGribovXsc() / kMm2,
                       gg.GetDiffractionGlauberGribovXsc() / kMm2);
        }
      }
    }
    for (const Part& p : ions) {
      if (p.def == nullptr) { continue; }
      // ComputeCrossSections is private on G4ComponentGGNuclNuclXsc (it is public on the
      // hadron-nucleus component). GetElasticGlauberGribov is the public inline that calls it
      // and then returns fElasticXsc, so the five members are all filled after one call and
      // the getters below read the same state the process would.
      G4DynamicParticle dpi(const_cast<G4ParticleDefinition*>(p.def), G4ThreeVector(0, 0, 1),
                            100 * MeV);
      for (int z = 1; z <= 92; ++z) {
        const int a = G4lrint(nist->GetAtomicMassAmu(z));
        for (double e : es) {
          dpi.SetKineticEnergy(e * MeV);
          nn.GetElasticGlauberGribov(&dpi, z, a);
          std::fprintf(f, "GGNuclNucl,%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                       p.name, z, a, e, nn.GetTotalGlauberGribovXsc() / kMm2,
                       nn.GetInelasticGlauberGribovXsc() / kMm2,
                       nn.GetElasticGlauberGribovXsc() / kMm2,
                       nn.GetProductionGlauberGribovXsc() / kMm2,
                       nn.GetDiffractionGlauberGribovXsc() / kMm2);
        }
      }
    }
    std::fclose(f);
  }

  // ------------------------------------------------------------ 5. G4UPiNuclearCrossSection
  //
  // THERE IS NO had_upi.csv, AND THE REASON IS A LINKER ONE
  //
  // G4UPiNuclearCrossSection's only cross-section accessors are the two `inline` methods
  // GetElasticCrossSection / GetInelasticCrossSection in its header, and both read the private
  // static members piPlusElastic / piPlusInelastic / piMinusElastic / piMinusInelastic. Geant4
  // on this machine is built as DLLs and those statics are not exported, so an inline expanded
  // in *this* translation unit cannot link:
  //
  //   dump_hadronic_xs.obj : error LNK2019: unresolved external symbol
  //   "private: static class G4PhysicsTable * G4UPiNuclearCrossSection::piPlusElastic"
  //
  // It is not an access problem that a friend or a subclass could fix and it is not something
  // a cast reaches: the symbol is absent from the import library. The class also overrides
  // neither GetElementCrossSection nor GetIsoCrossSection, so the G4VCrossSectionDataSet base
  // interface does not reach it either.
  //
  // What reaches it is G4BGGPionElasticXS / G4BGGPionInelasticXS, which live inside the DLL.
  // Read G4BGGPionElasticXS::GetElementCrossSection: for Z > 1 and
  // fLowEnergy(20 MeV) < ekin <= fGlauberEnergy(91 GeV) it returns
  // `fPion->GetElasticCrossSection(dp, Z, theA[Z])` with nothing applied to it - the raw UPi
  // number - and the inelastic twin does the same with `<` in place of `<=` at 20 MeV. And
  // (Z, theA[Z]) for Z = 2..92 is the *only* argument either BGG class ever passes UPi:
  // theA[Z] = G4lrint(nist->GetAtomicMassAmu(Z)) is set once in BuildPhysicsTable and used at
  // every call site. So had_bgg.csv in that band is a complete oracle for
  // G4UPiNuclearCrossSection, with no loss of coverage, and the UPi node energies are appended
  // to the BGG scan below so that the tabulated nodes - where the starting-index hint in
  // G4PhysicsVector::Value changes the answer in the last bits, see physics_vector.cuh - are
  // hit exactly rather than straddled.
  //
  // The two Coulomb-region factors theCoulombFac*/theLowE* are UPi values at exactly 20 MeV
  // divided by the factor at 20 MeV, so the below-20-MeV band of had_bgg.csv is a second,
  // independent check of the same UPi number at that one energy.

  // ------------------------------------------------------------ 6. the four BGG data sets
  //
  // Two instances per class: `isProton` / `isPiplus` is a per-instance member set by
  // BuildPhysicsTable from its argument, and the per-Z static tables are shared, so one
  // object can only answer for one projectile.
  {
    G4BGGNucleonElasticXS bggElP(pro), bggElN(neu);
    G4BGGNucleonInelasticXS bggInP(pro), bggInN(neu);
    G4BGGPionElasticXS bggPiElP(G4PionPlus::PionPlus()), bggPiElM(G4PionMinus::PionMinus());
    G4BGGPionInelasticXS bggPiInP(G4PionPlus::PionPlus()),
        bggPiInM(G4PionMinus::PionMinus());
    bggElP.BuildPhysicsTable(*pro);
    bggElN.BuildPhysicsTable(*neu);
    bggInP.BuildPhysicsTable(*pro);
    bggInN.BuildPhysicsTable(*neu);
    bggPiElP.BuildPhysicsTable(*G4PionPlus::PionPlus());
    bggPiElM.BuildPhysicsTable(*G4PionMinus::PionMinus());
    bggPiInP.BuildPhysicsTable(*G4PionPlus::PionPlus());
    bggPiInM.BuildPhysicsTable(*G4PionMinus::PionMinus());

    struct Set {
      const char* name;
      const char* particle;
      G4VCrossSectionDataSet* xs;
      const G4ParticleDefinition* def;
    };
    const Set sets[] = {
        {"BGGNucleonElastic", "proton", &bggElP, pro},
        {"BGGNucleonElastic", "neutron", &bggElN, neu},
        {"BGGNucleonInelastic", "proton", &bggInP, pro},
        {"BGGNucleonInelastic", "neutron", &bggInN, neu},
        {"BGGPionElastic", "pi+", &bggPiElP, G4PionPlus::PionPlus()},
        {"BGGPionElastic", "pi-", &bggPiElM, G4PionMinus::PionMinus()},
        {"BGGPionInelastic", "pi+", &bggPiInP, G4PionPlus::PionPlus()},
        {"BGGPionInelastic", "pi-", &bggPiInM, G4PionMinus::PionMinus()},
    };
    FILE* f = std::fopen("had_bgg.csv", "w");
    std::fprintf(f, "dataset,particle,Z,energy_MeV,xs_mm2\n");
    // 1 MeV is fLowestEnergy for the pion classes, 14 MeV fLowEnergy for the nucleon ones,
    // 20 MeV fLowEnergy for the pion ones, 91 GeV fGlauberEnergy for all four. Each is dumped
    // exactly, and 20 MeV is where the `<=` of the elastic class and the `<` of the inelastic
    // one disagree about which branch to take.
    //
    // The UPi node energies are here too, because this CSV is the only oracle for
    // G4UPiNuclearCrossSection (see section 5's comment). All six of its grids' nodes, in GeV
    // as G4UPiNuclearCrossSection.cc spells them, converted to MeV.
    std::vector<double> extra = {1.0, 14.0, 20.0, 91000.0};
    const double upi_nodes_GeV[] = {
        0.02, 0.04, 0.05,  0.06, 0.07, 0.08, 0.09, 0.1,   0.11, 0.12, 0.13,
        0.14, 0.15, 0.16,  0.17, 0.18, 0.19, 0.2,  0.22,  0.24, 0.25, 0.26,
        0.28, 0.3,  0.35,  0.4,  0.45, 0.5,  0.55, 0.575, 0.6,  0.7,  0.8,
        0.9,  1.0,  2.0,   3.0,  5.0,  10.0, 20.0, 50.0,  100.0, 500.0, 1000.0};
    for (double n : upi_nodes_GeV) { extra.push_back(n * 1000.0); }
    std::vector<double> es = decade_scan(1e-3, 1e8);
    add_and_sort(es, extra, 1e-3, 1e8);
    for (const Set& s : sets) {
      G4DynamicParticle dp(s.def, G4ThreeVector(0, 0, 1), 100 * MeV);
      for (int z = 1; z <= 92; ++z) {
        for (double e : es) {
          dp.SetKineticEnergy(e * MeV);
          std::fprintf(f, "%s,%s,%d,%.17g,%.17g\n", s.name, s.particle, z, e,
                       s.xs->GetElementCrossSection(&dp, z, nullptr) / kMm2);
        }
      }
    }
    std::fclose(f);
  }

  // ------------------------------------------------------------ 7/8. the G4PARTICLEXS sets
  {
    G4ParticleInelasticXS pInel(pro);
    G4ParticleInelasticXS dInel(G4Deuteron::Deuteron());
    G4ParticleInelasticXS tInel(G4Triton::Triton());
    G4ParticleInelasticXS hInel(G4He3::He3());
    G4ParticleInelasticXS aInel(G4Alpha::Alpha());
    G4NeutronInelasticXS nInel;
    G4NeutronElasticXS nElas;
    G4NeutronCaptureXS nCap;
    G4GammaNuclearXS gNuc;
    pInel.BuildPhysicsTable(*pro);
    dInel.BuildPhysicsTable(*G4Deuteron::Deuteron());
    tInel.BuildPhysicsTable(*G4Triton::Triton());
    hInel.BuildPhysicsTable(*G4He3::He3());
    aInel.BuildPhysicsTable(*G4Alpha::Alpha());
    nInel.BuildPhysicsTable(*neu);
    nElas.BuildPhysicsTable(*neu);
    nCap.BuildPhysicsTable(*neu);
    gNuc.BuildPhysicsTable(*G4Gamma::Gamma());

    struct Set {
      const char* name;
      const char* particle;
      G4VCrossSectionDataSet* xs;
      const G4ParticleDefinition* def;
      double e0, e1;
      int zmax;
    };
    // The energy range per set is its own: the four nucleon/ion sets run from far below their
    // tables' first node to 100 TeV, so both the flat extrapolation below and the
    // Glauber-Gribov hand-over above are covered; capture runs down to 1e-17 MeV so that the
    // 1e-10 eV floor and the 1/sqrt(E) form below node 1 are both exercised, and up past its
    // 20 MeV ceiling; gamma-nuclear stops at 1 GeV, well past the 130 MeV top of its tables
    // and the 150 MeV CHIPS transition.
    const Set sets[] = {
        {"ParticleInelastic", "proton", &pInel, pro, 1e-6, 1e8, 92},
        {"ParticleInelastic", "deuteron", &dInel, G4Deuteron::Deuteron(), 1e-6, 1e8, 92},
        {"ParticleInelastic", "triton", &tInel, G4Triton::Triton(), 1e-6, 1e8, 92},
        {"ParticleInelastic", "He3", &hInel, G4He3::He3(), 1e-6, 1e8, 92},
        {"ParticleInelastic", "alpha", &aInel, G4Alpha::Alpha(), 1e-6, 1e8, 92},
        {"NeutronInelastic", "neutron", &nInel, neu, 1e-6, 1e8, 92},
        {"NeutronElastic", "neutron", &nElas, neu, 1e-6, 1e8, 92},
        {"NeutronCapture", "neutron", &nCap, neu, 1e-17, 1e2, 92},
        {"GammaNuclear", "gamma", &gNuc, G4Gamma::Gamma(), 1.0, 1e3, 94},
    };
    FILE* f = std::fopen("had_particlexs.csv", "w");
    std::fprintf(f, "dataset,particle,Z,energy_MeV,xs_mm2\n");
    FILE* fi = std::fopen("had_particlexs_iso.csv", "w");
    std::fprintf(fi, "dataset,particle,Z,A,energy_MeV,xs_mm2\n");
    for (const Set& s : sets) {
      G4DynamicParticle dp(s.def, G4ThreeVector(0, 0, 1), 100 * MeV);
      std::vector<double> es = decade_scan(s.e0, s.e1);
      // 20 MeV is the isotope window's edge for the two inelastic sets and the ceiling for
      // capture; 150 MeV is gamma's CHIPS transition; 20 GeV is the top of most of the
      // nucleon tables and so the hand-over energy.
      add_and_sort(es, {1e-10 * eV / MeV, 20.0, 130.0, 150.0, 20000.0}, s.e0, s.e1);
      for (int z = 1; z <= s.zmax; ++z) {
        for (double e : es) {
          dp.SetKineticEnergy(e * MeV);
          std::fprintf(f, "%s,%s,%d,%.17g,%.17g\n", s.name, s.particle, z, e,
                       s.xs->GetElementCrossSection(&dp, z, nullptr) / kMm2);
        }
        if (z > 92 || kAmin[z] >= kAmax[z]) { continue; }
        // Sixteen points, chosen to straddle every isotope boundary rather than to be many.
        const double eiso[16] = {1e-6, 1e-4,  1e-2,  0.1,   1.0,    5.0,    10.0,  19.0,
                                 20.0, 21.0,  50.0,  130.0, 150.0,  1000.0, 2e4,   1e6};
        for (int a = kAmin[z]; a <= kAmax[z]; ++a) {
          for (double e : eiso) {
            if (e < s.e0 || e > s.e1) { continue; }
            dp.SetKineticEnergy(e * MeV);
            std::fprintf(fi, "%s,%s,%d,%d,%.17g,%.17g\n", s.name, s.particle, z, a, e,
                         s.xs->GetIsoCrossSection(&dp, z, a, nullptr, nullptr, nullptr)
                             / kMm2);
          }
        }
      }
    }
    std::fclose(f);
    std::fclose(fi);
  }

  // ------------------------------------------------------------ 9/10. the neutron general
  //                                                             process's own tables
  {
    FILE* fm = std::fopen("had_matelem.csv", "w");
    std::fprintf(fm, "material,index,Z,natoms_per_mm3\n");
    for (std::size_t i = 0; i < ctx.materials.size(); ++i) {
      const G4Material* m = ctx.materials[i];
      const G4double* na = m->GetVecNbOfAtomsPerVolume();
      for (std::size_t j = 0; j < m->GetNumberOfElements(); ++j) {
        std::fprintf(fm, "%s,%zu,%d,%.17g\n", m->GetName().c_str(), j,
                     m->GetElement(j)->GetZasInt(), na[j] * (mm * mm * mm));
      }
    }
    std::fclose(fm);

    auto* nGen = G4PhysListUtil::FindNeutronGeneralProcess();
    FILE* f = std::fopen("had_neutron_general.csv", "w");
    std::fprintf(f, "material,table,index,energy_MeV,value\n");
    // The five tables are named LambdaNeutronGeneral{0,3} and ProbNeutronGeneral{1,2,4} by
    // StorePhysicsTable, and G4VProcess::GetPhysicsTableFileName spells the file
    // "<dir>/<table>.<process>.<particle>.dat".
    const char* names[5] = {"LambdaNeutronGeneral0", "ProbNeutronGeneral1",
                            "ProbNeutronGeneral2", "LambdaNeutronGeneral3",
                            "ProbNeutronGeneral4"};
    bool stored = nGen->StorePhysicsTable(neu, ".", false);
    if (!stored) {
      std::printf("dump_hadronic_xs: G4NeutronGeneralProcess::StorePhysicsTable failed - "
                  "had_neutron_general.csv will be empty\n");
    }
    // The material index in the stored table is G4Material::GetIndex(), which is the global
    // material table's order and NOT the dump program's `materials` order - a NIST material
    // built as a component of another one gets an index too. So the names come from the
    // global table.
    const G4MaterialTable* allmat = G4Material::GetMaterialTable();
    for (int it = 0; it < 5 && stored; ++it) {
      const std::string path =
          std::string("./") + names[it] + "." + nGen->GetProcessName() + ".neutron.dat";
      std::vector<StoredVector> vecs;
      const bool read_ok = read_stored_table(path, vecs);
      // The .dat is an intermediate of this dump, not an oracle file: ref/oracle's .gitignore
      // covers *.csv and would leave five binary tables as untracked files for whoever runs
      // the dumper next. Removed as soon as it has been read.
      std::remove(path.c_str());
      if (!read_ok) {
        std::printf("dump_hadronic_xs: cannot read back %s\n", path.c_str());
        continue;
      }
      for (std::size_t im = 0; im < vecs.size(); ++im) {
        const char* mname =
            (im < allmat->size()) ? (*allmat)[im]->GetName().c_str() : "(unknown)";
        for (std::size_t j = 0; j < vecs[im].e.size(); ++j) {
          std::fprintf(f, "%s,%d,%zu,%.17g,%.17g\n", mname, it, j, vecs[im].e[j] / MeV,
                       // Tables 0 and 3 are macroscopic cross sections, 1/mm; 1, 2 and 4 are
                       // dimensionless probabilities.
                       (it == 0 || it == 3) ? vecs[im].v[j] * mm : vecs[im].v[j]);
        }
      }
    }
    std::fclose(f);
  }

  // ------------------------------------------------------------ 11. what G4HadronicProcess
  //                                                             actually tabulates
  //
  // Not a cross-section table. G4HadronicProcess::PostStepGetPhysicalInteractionLength calls
  // theCrossSectionDataStore->ComputeCrossSection every step - there is no G4PhysicsVector of
  // the cross section anywhere in it. What BuildPhysicsTable does build, for a charged hadron
  // with the integral method on (EnableIntegralInelasticXS and EnableIntegralElasticXS both
  // default true), is the *shape* of the cross section per material: G4HadXSHelper scans
  // 10 points per decade from the process's minKinEnergy to 100 TeV and records up to three
  // peaks and two dips, so that UpdateCrossSectionAndMFP knows which side of a peak it is on.
  //
  // Dumped here because it is derived entirely from P2's cross sections and is the only table
  // between them and a step length for a proton or a pion. P5 owns using it.
  {
    FILE* f = std::fopen("had_xspeaks.csv", "w");
    std::fprintf(f, "process,particle,material,e1peak_MeV,e1deep_MeV,e2peak_MeV,e2deep_MeV,"
                    "e3peak_MeV\n");
    struct Pr {
      const char* name;
      const G4ParticleDefinition* def;
      bool inelastic;
    };
    const Pr prs[] = {{"protonInelastic", pro, true},
                      {"hadElastic_proton", pro, false},
                      {"pi+Inelastic", G4PionPlus::PionPlus(), true},
                      {"pi-Inelastic", G4PionMinus::PionMinus(), true}};
    const G4MaterialTable* allmat = G4Material::GetMaterialTable();
    const G4double emax = G4HadronicParameters::Instance()->GetMaxEnergy();
    for (const Pr& pr : prs) {
      G4HadronicProcess* proc = pr.inelastic ? G4PhysListUtil::FindInelasticProcess(pr.def)
                                             : G4PhysListUtil::FindElasticProcess(pr.def);
      if (proc == nullptr) { continue; }
      // `minKinEnergy = 1*CLHEP::MeV` in G4HadronicProcess's constructor, a private member
      // with no getter and no setter, so it is written out here rather than read back.
      auto* peaks = G4HadXSHelper::FillPeaksStructure(proc, pr.def, 1 * CLHEP::MeV, emax);
      if (peaks == nullptr) { continue; }
      for (std::size_t i = 0; i < peaks->size(); ++i) {
        const G4TwoPeaksHadXS* x = (*peaks)[i];
        if (x == nullptr) { continue; }
        const char* mname =
            (i < allmat->size()) ? (*allmat)[i]->GetName().c_str() : "(unknown)";
        auto cap = [](G4double v) { return (v > 1e30) ? -1.0 : v / MeV; };
        std::fprintf(f, "%s,%s,%s,%.17g,%.17g,%.17g,%.17g,%.17g\n", pr.name,
                     pr.def->GetParticleName().c_str(), mname, cap(x->e1peak),
                     cap(x->e1deep), cap(x->e2peak), cap(x->e2deep), cap(x->e3peak));
      }
    }
    std::fclose(f);
  }
}

}  // namespace

G4GPU_REGISTER_DUMP("hadronic_xs",
                    "had_hnxsc.csv had_radii.csv had_masses.csv had_coulomb.csv had_ggcomp.csv "
                    "had_bgg.csv had_particlexs.csv had_particlexs_iso.csv "
                    "had_neutron_general.csv had_matelem.csv had_xspeaks.csv",
                    dump_hadronic_xs);
