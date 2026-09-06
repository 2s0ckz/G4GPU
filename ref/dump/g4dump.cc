// Oracle dumper: initializes exactly B1's geometry, materials and physics list (QBBC), then
// writes Geant4's own production cuts, cross sections, stopping powers and ranges to CSV.
//
// Everything the GPU port transcribes gets diffed against these files, so a transcription
// error shows up as a number rather than as a dose shift to be hunted later.
//
// Uses G4EmCalculator, which is the supported way to interrogate the EM tables after a run
// has initialized them.
#include "G4RunManagerFactory.hh"
#include "G4EmCalculator.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Cons.hh"
#include "G4Orb.hh"
#include "G4Sphere.hh"
#include "G4Torus.hh"
#include "G4Trd.hh"
#include "G4Para.hh"
#include "G4Trap.hh"
#include "G4EllipticalTube.hh"
#include "G4Ellipsoid.hh"
#include "G4EllipticalCone.hh"
#include "G4Paraboloid.hh"
#include "G4Hype.hh"
#include "G4Tet.hh"
#include "G4Polycone.hh"
#include "G4Polyhedra.hh"
#include "G4UnionSolid.hh"
#include "G4SubtractionSolid.hh"
#include "G4IntersectionSolid.hh"
#include "G4RotationMatrix.hh"
#include "G4VSolid.hh"
#include "G4PhysicalConstants.hh"
#include <cctype>
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4VUserActionInitialization.hh"
#include "G4ParticleGun.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4PionPlus.hh"
#include "G4PionMinus.hh"
#include "G4KaonPlus.hh"
#include "G4Proton.hh"
#include "G4AntiProton.hh"
#include "G4Alpha.hh"
#include "G4He3.hh"
#include "G4GenericIon.hh"
#include "G4KaonMinus.hh"
#include "G4Deuteron.hh"
#include "G4Triton.hh"
#include "G4ProcessManager.hh"
#include "G4eeToTwoGammaModel.hh"
#include "G4eBremsstrahlungRelModel.hh"
#include "G4BetheBlochModel.hh"
#include "G4IonisParamMat.hh"
#include "G4EmCorrections.hh"
#include "G4MuBetheBlochModel.hh"
#include "G4MuBremsstrahlungModel.hh"
#include "G4MuPairProductionModel.hh"
#include "G4hBremsstrahlungModel.hh"
#include "G4hPairProductionModel.hh"
#include "G4WentzelOKandVIxSection.hh"
#include "G4BraggModel.hh"
#include "G4BraggIonModel.hh"
#include "G4PSTARStopping.hh"
#include "G4ICRU90StoppingData.hh"
#include "G4NucleiProperties.hh"
#include "G4ScreeningMottCrossSection.hh"
#include "G4ionEffectiveCharge.hh"
#include "G4IonTable.hh"
#include "G4ASTARStopping.hh"
#include "G4DensityEffectData.hh"
#include "G4Material.hh"
#include "G4ICRU49NuclearStoppingModel.hh"
#include "G4ICRU73QOModel.hh"
#include "G4UrbanMscModel.hh"
#include "G4RayleighAngularGenerator.hh"
#include "G4DataVector.hh"
#include "G4VProcess.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"
#include <cfloat>
#include "G4UImanager.hh"
#include "G4NucleonNuclearCrossSection.hh"
#include "G4IonFluctuations.hh"
#include "G4Neutron.hh"
#include "G4DynamicParticle.hh"
#include "QBBC.hh"

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

namespace {

/// B1's four materials in a trivial world; the geometry only has to exist for the run to
/// initialize, but the materials must be exactly B1's.
class Det : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    // Same NIST names B1 uses, so densities, compositions and mean excitation energies match.
    mats_.push_back(nist->FindOrBuildMaterial("G4_AIR"));
    mats_.push_back(nist->FindOrBuildMaterial("G4_WATER"));
    mats_.push_back(nist->FindOrBuildMaterial("G4_A-150_TISSUE"));
    mats_.push_back(nist->FindOrBuildMaterial("G4_BONE_COMPACT_ICRU"));
    // A material deliberately outside the 74 NIST entries G4PSTARStopping tabulates and
    // outside the Ziegler 1988 molecule list, so G4BraggModel falls through to its
    // per-element Ziegler parameterisation - the branch this port implements.
    {
      auto* si = nist->FindOrBuildElement(14);
      auto* ge = nist->FindOrBuildElement(32);
      auto* custom = new G4Material("CustomSiGe", 4.2 * g / cm3, 2);
      custom->AddElement(si, 0.6);
      custom->AddElement(ge, 0.4);
      mats_.push_back(custom);
    }

    // A material with a chemical formula but a name that is not one of the 74 NIST entries,
    // so G4BraggModel takes its *molecular* branch: the ICRU 49 / Ziegler 1988
    // parameterisation for a compound, chosen by formula rather than by name.
    //
    // Without it that branch is unreachable from this dumper, because every compound in
    // Geant4's eleven-name list is also a NIST material and the name match wins. It was
    // therefore the one branch of G4BraggModel::DEDX with no measurement behind it.
    {
      auto* h = nist->FindOrBuildElement(1);
      auto* o = nist->FindOrBuildElement(8);
      auto* molecular = new G4Material("CustomMolecularWater", 1.0 * g / cm3, 2);
      molecular->AddElement(h, 2);
      molecular->AddElement(o, 1);
      molecular->SetChemicalFormula("H_2O");
      mats_.push_back(molecular);
    }

    // A material with components and **no** SetMeanExcitationEnergy, so Geant4 derives one.
    //
    // Every other custom material here sets it explicitly, which left the derivation with
    // nothing behind it - and the port was passing zero straight through, so a user-built
    // material put log(0) into the density-effect parameters and scored a NaN dose. This is
    // the material that makes G4IonisParamMat::ComputeMeanParameters checkable.
    //
    // Three elements with very different Z, because the derivation is a Z-weighted average of
    // logarithms and a two-element compound of neighbours would not distinguish it from
    // several wrong weightings.
    {
      auto* h = nist->FindOrBuildElement(1);
      auto* c = nist->FindOrBuildElement(6);
      auto* pb = nist->FindOrBuildElement(82);
      auto* derived = new G4Material("CustomDerivedI", 2.5 * g / cm3, 3);
      derived->AddElement(h, 0.1);
      derived->AddElement(c, 0.6);
      derived->AddElement(pb, 0.3);
      mats_.push_back(derived);
    }

    auto* solid = new G4Box("World", 1 * m, 1 * m, 1 * m);
    auto* lv = new G4LogicalVolume(solid, mats_[1], "World");
    auto* pv = new G4PVPlacement(nullptr, {}, lv, "World", nullptr, false, 0);
    // One logical volume per material, so each gets a material-cuts couple and therefore
    // its own production thresholds.
    //
    // Spaced from the actual count, not a constant. The comment used to say "spaced to fit
    // however many materials are in the list" above `(i - 2.0) * 35 cm`, which fits five and
    // was written when there were five; the seventh landed 150 cm out and Geant4 aborted with
    // "Daughter physical volume pv6 is entirely outside mother logical volume World".
    const G4double half = 10 * cm;
    const G4double n = G4double(mats_.size());
    const G4double pitch = (2 * 100 * cm - 2 * half) / (n > 1 ? n : 1);
    for (size_t i = 0; i < mats_.size(); ++i) {
      auto* s = new G4Box("b" + std::to_string(i), half, half, half);
      auto* l = new G4LogicalVolume(s, mats_[i], "lv" + std::to_string(i));
      const G4double x = (G4double(i) - (n - 1) * 0.5) * pitch;
      new G4PVPlacement(nullptr, {x, 0, 0}, l, "pv" + std::to_string(i), lv, false, 0);
    }
    return pv;
  }
  const std::vector<G4Material*>& materials() const { return mats_; }

 private:
  std::vector<G4Material*> mats_;
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  Gun() : gun_(new G4ParticleGun(1)) {
    gun_->SetParticleDefinition(G4Gamma::Gamma());
    gun_->SetParticleEnergy(6 * MeV);
    gun_->SetParticleMomentumDirection({0, 0, 1});
  }
  ~Gun() override { delete gun_; }
  void GeneratePrimaries(G4Event* e) override { gun_->GeneratePrimaryVertex(e); }

 private:
  G4ParticleGun* gun_;
};

class Actions : public G4VUserActionInitialization {
 public:
  void Build() const override { SetUserAction(new Gun()); }
};

}  // namespace

int main() {
  auto* rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  auto* det = new Det();
  rm->SetUserInitialization(det);
  rm->SetUserInitialization(new QBBC(0));   // exactly B1's physics list
  rm->SetUserInitialization(new Actions());
  rm->Initialize();
  // One event forces the EM physics tables to be built before we interrogate them.
  rm->BeamOn(1);

  const auto& mats = det->materials();
  G4EmCalculator calc;
  calc.SetVerbose(0);

  // Production thresholds per material, needed below to request RESTRICTED quantities.
  std::map<const G4Material*, double> ecut;
  std::map<const G4Material*, double> gcut;

  // ---------------- production cuts
  {
    FILE* f = std::fopen("cuts.csv", "w");
    std::fprintf(f, "material,gamma_cut_MeV,electron_cut_MeV,positron_cut_MeV\n");
    auto* table = G4ProductionCutsTable::GetProductionCutsTable();
    for (auto* m : mats) {
      double cg = 0, ce = 0, cp = 0;
      for (size_t i = 0; i < table->GetTableSize(); ++i) {
        const auto* couple = table->GetMaterialCutsCouple(static_cast<G4int>(i));
        if (couple->GetMaterial() != m) { continue; }
        const auto& v = *table->GetEnergyCutsVector(0);  // 0 = gamma
        const auto& ve = *table->GetEnergyCutsVector(1); // 1 = e-
        const auto& vp = *table->GetEnergyCutsVector(2); // 2 = e+
        cg = v[i] / MeV; ce = ve[i] / MeV; cp = vp[i] / MeV;
        break;
      }
      ecut[m] = ce * MeV;
      gcut[m] = cg * MeV;
      std::fprintf(f, "%s,%.9g,%.9g,%.9g\n", m->GetName().c_str(), cg, ce, cp);
    }
    std::fclose(f);
  }

  // ---------------- gamma cross sections, per process, 1/mm
  {
    FILE* f = std::fopen("gamma_xs.csv", "w");
    std::fprintf(f, "material,energy_MeV,phot_per_mm,compt_per_mm,conv_per_mm,Rayl_per_mm\n");
    const char* procs[4] = {"phot", "compt", "conv", "Rayl"};
    for (auto* m : mats) {
      for (int i = 0; i <= 440; ++i) {
        const double e = 1e-3 * std::pow(10.0, i / 40.0);  // 1 keV .. 10 GeV
        double v[4];
        for (int p = 0; p < 4; ++p) {
          v[p] = calc.ComputeCrossSectionPerVolume(e * MeV, G4Gamma::Gamma(), procs[p], m) * mm;
        }
        std::fprintf(f, "%s,%.9g,%.9g,%.9g,%.9g,%.9g\n", m->GetName().c_str(), e, v[0], v[1],
                     v[2], v[3]);
      }
    }
    std::fclose(f);
  }

  // ---------------- electron / positron dE/dx and range
  {
    FILE* f = std::fopen("electron_tables.csv", "w");
    // Restricted quantities pass the material's own production cut; unrestricted pass a
    // huge cut. The pair is what exposes the delta-ray split.
    std::fprintf(f,
                 "material,particle,energy_MeV,cut_MeV,dedx_ioni_restricted,"
                 "dedx_ioni_unrestricted,dedx_brem_default,dedx_total_MeV_per_mm,"
                 "range_mm,csda_range_mm,delta_xs_per_mm,gcut_MeV,"
                 "dedx_brem_restricted,brem_xs_per_mm\n");
    const G4ParticleDefinition* parts[2] = {G4Electron::Electron(), G4Positron::Positron()};
    const double kHuge = 1e6 * MeV;
    for (auto* m : mats) {
      const double cut = ecut.count(m) ? ecut[m] : 0.99e-3 * MeV;
      for (const G4ParticleDefinition* pn : parts) {
        for (int i = 0; i <= 440; ++i) {
          const double e = 1e-3 * std::pow(10.0, i / 40.0);
          const double dr = calc.ComputeDEDX(e * MeV, pn, "eIoni", m, cut) / (MeV / mm);
          const double du = calc.ComputeDEDX(e * MeV, pn, "eIoni", m, kHuge) / (MeV / mm);
          const double db = calc.ComputeDEDX(e * MeV, pn, "eBrem", m) / (MeV / mm);
          const double dt = calc.GetDEDX(e * MeV, pn, m) / (MeV / mm);
          const double r = calc.GetRange(e * MeV, pn, m) / mm;
          const double rc = calc.GetCSDARange(e * MeV, pn, m) / mm;
          const double dx =
              calc.ComputeCrossSectionPerVolume(e * MeV, pn, "eIoni", m, cut) * mm;
          const double gc = gcut.count(m) ? gcut[m] : 0.99e-3 * MeV;
          const double dbr = calc.ComputeDEDX(e * MeV, pn, "eBrem", m, gc) / (MeV / mm);
          const double bxs =
              calc.ComputeCrossSectionPerVolume(e * MeV, pn, "eBrem", m, gc) * mm;
          std::fprintf(f,
                       "%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                       m->GetName().c_str(), pn->GetParticleName().c_str(), e, cut / MeV, dr,
                       du, db, dt, r, rc, dx, gc / MeV, dbr, bxs);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- what is actually registered, and under what names
  // The process name is the key G4EmCalculator looks up; guessing it silently yields zeros.
  {
    FILE* f = std::fopen("processes.csv", "w");
    std::fprintf(f, "particle,process,type,subtype\n");
    const G4ParticleDefinition* all[] = {
        G4Gamma::Gamma(), G4Electron::Electron(), G4Positron::Positron(),
        G4MuonMinus::MuonMinus(), G4MuonPlus::MuonPlus(), G4PionPlus::PionPlus(),
        G4PionMinus::PionMinus(), G4KaonPlus::KaonPlus(), G4Proton::Proton(),
        G4AntiProton::AntiProton(), G4Alpha::Alpha(), G4He3::He3(),
        G4GenericIon::GenericIon()};
    for (const G4ParticleDefinition* pn : all) {
      auto* pm = pn->GetProcessManager();
      if (pm == nullptr) { continue; }
      auto* pv = pm->GetProcessList();
      for (G4int i = 0; i < (G4int)pv->size(); ++i) {
        auto* pr = (*pv)[i];
        std::fprintf(f, "%s,%s,%d,%d\n", pn->GetParticleName().c_str(),
                     pr->GetProcessName().c_str(), (int)pr->GetProcessType(),
                     pr->GetProcessSubType());
      }
    }
    std::fclose(f);
  }

  // ---------------- positron annihilation cross section, 1/mm
  {
    FILE* f = std::fopen("annihilation.csv", "w");
    G4eeToTwoGammaModel twoGamma;
    std::fprintf(f, "material,energy_MeV,annihil_per_mm\n");
    for (auto* m : mats) {
      for (int i = 0; i <= 440; ++i) {
        const double e = 1e-3 * std::pow(10.0, i / 40.0);
        // G4EmCalculator cannot reach the G4VEmProcess-derived models here (FindEmModel
        // returns null and no lambda table is built), so call the model class directly.
        // That is a stronger oracle anyway: it is the same object the physics list uses.
        const double v = twoGamma.CrossSectionPerVolume(m, G4Positron::Positron(), e * MeV)
                         * mm;
        std::fprintf(f, "%s,%.9g,%.9g\n", m->GetName().c_str(), e, v);
      }
    }
    std::fclose(f);
  }

  // ---------------- relativistic bremsstrahlung, calling the model class directly
  // G4EmCalculator picks a model by energy and silently blends across the 1 GeV
  // SeltzerBerger/relativistic boundary, so it cannot say which model produced a number.
  // Instantiating the model removes that ambiguity.
  {
    FILE* f = std::fopen("brems_rel.csv", "w");
    std::fprintf(f, "material,energy_MeV,gcut_MeV,dedx_MeV_per_mm,xs_per_mm,lpm_active\n");
    G4eBremsstrahlungRelModel rel;
    rel.SetLPMFlag(true);
    G4DataVector cuts;
    cuts.push_back(1.0 * keV);
    rel.Initialise(G4Electron::Electron(), cuts);
    for (auto* m : mats) {
      const double gc = gcut.count(m) ? gcut[m] : 0.99e-3 * MeV;
      for (int i = 120; i <= 440; ++i) {  // 1 GeV .. 100 TeV
        const double e = 1e-3 * std::pow(10.0, i / 40.0);
        double d = 0, x = 0;
        try { d = rel.ComputeDEDXPerVolume(m, G4Electron::Electron(), e * MeV, gc) / (MeV / mm); }
        catch (...) {}
        // CrossSectionPerVolume is the G4VEmModel entry point; the model overrides
        // ComputeCrossSectionPerAtom, and the base sums it over the element vector.
        try {
          x = rel.CrossSectionPerVolume(m, G4Electron::Electron(), e * MeV, gc, e * MeV) * mm;
        } catch (...) {}
        std::fprintf(f, "%s,%.9g,%.9g,%.9g,%.9g,%d\n", m->GetName().c_str(), e, gc / MeV, d, x,
                     0);
      }
    }
    std::fclose(f);
  }

  // ---------------- heavy charged particles: dE/dx, range, and the radiative processes
  // Covers everything G4EmBuilder::ConstructCharged registers besides e-/e+.
  {
    FILE* f = std::fopen("hadron_tables.csv", "w");
    std::fprintf(f,
                 "material,particle,mass_MeV,charge,energy_MeV,dedx_ioni,dedx_total,"
                 "range_mm,delta_xs_per_mm,dedx_brem,dedx_pair,brem_xs_per_mm,"
                 "pair_xs_per_mm,nuclear_dedx\n");
    struct P { const G4ParticleDefinition* def; const char* ioni; const char* brem;
               const char* pair; const char* nuc; };
    const P parts[] = {
        {G4MuonMinus::MuonMinus(), "muIoni", "muBrems", "muPairProd", nullptr},
        {G4MuonPlus::MuonPlus(),   "muIoni", "muBrems", "muPairProd", nullptr},
        {G4PionPlus::PionPlus(),   "hIoni",  "hBrems",  "hPairProd",  nullptr},
        {G4PionMinus::PionMinus(), "hIoni",  "hBrems",  "hPairProd",  nullptr},
        {G4KaonPlus::KaonPlus(),   "hIoni",  "hBrems",  "hPairProd",  nullptr},
        {G4KaonMinus::KaonMinus(), "hIoni",  "hBrems",  "hPairProd",  nullptr},
        {G4Proton::Proton(),       "hIoni",  "hBrems",  "hPairProd",  "nuclearStopping"},
        {G4AntiProton::AntiProton(), "hIoni", "hBrems", "hPairProd",  nullptr},
        {G4Alpha::Alpha(),         "ionIoni", nullptr,  nullptr,      "nuclearStopping"},
        {G4He3::He3(),             "ionIoni", nullptr,  nullptr,      "nuclearStopping"},
        // GenericIon is the base particle every non-alpha, non-He3 ion scales from, so its own
        // table is what that scaling is applied to. Deuteron and triton are here because
        // G4EmBuilder gives them G4hIonisation and not G4ionIonisation - charge 1, mass 2 and
        // 3 - which is the split a rule written on charge or mass alone gets wrong.
        {G4GenericIon::GenericIon(), "ionIoni", nullptr, nullptr,  "nuclearStopping"},
        {G4Deuteron::Deuteron(),   "hIoni",  nullptr,  nullptr,      "nuclearStopping"},
        {G4Triton::Triton(),       "hIoni",  nullptr,  nullptr,      "nuclearStopping"},
    };
    for (auto* m : mats) {
      const double cut = ecut.count(m) ? ecut[m] : 0.99e-3 * MeV;
      const double gc = gcut.count(m) ? gcut[m] : 0.99e-3 * MeV;
      for (const P& pp : parts) {
        // 1 keV .. 100 TeV, 20 points per decade
        for (int i = 0; i <= 220; ++i) {
          const double e = 1e-3 * std::pow(10.0, i / 20.0);
          double di = 0, dt = 0, r = 0, dx = 0, db = 0, dp = 0, bx = 0, px = 0, nd = 0;
          try { di = calc.ComputeDEDX(e * MeV, pp.def, pp.ioni, m, cut) / (MeV / mm); } catch (...) {}
          try { dt = calc.GetDEDX(e * MeV, pp.def, m) / (MeV / mm); } catch (...) {}
          try { r = calc.GetRange(e * MeV, pp.def, m) / mm; } catch (...) {}
          try { dx = calc.ComputeCrossSectionPerVolume(e * MeV, pp.def, pp.ioni, m, cut) * mm; } catch (...) {}
          if (pp.brem != nullptr) {
            try { db = calc.ComputeDEDX(e * MeV, pp.def, pp.brem, m, gc) / (MeV / mm); } catch (...) {}
            try { bx = calc.GetCrossSectionPerVolume(e * MeV, pp.def, pp.brem, m) * mm; } catch (...) {}
            if (bx <= 0) { try { bx = calc.ComputeCrossSectionPerVolume(e * MeV, pp.def, pp.brem, m, gc) * mm; } catch (...) {} }
          }
          if (pp.pair != nullptr) {
            try { dp = calc.ComputeDEDX(e * MeV, pp.def, pp.pair, m, cut) / (MeV / mm); } catch (...) {}
            try { px = calc.GetCrossSectionPerVolume(e * MeV, pp.def, pp.pair, m) * mm; } catch (...) {}
            if (px <= 0) { try { px = calc.ComputeCrossSectionPerVolume(e * MeV, pp.def, pp.pair, m, cut) * mm; } catch (...) {} }
          }
          if (pp.nuc != nullptr) {
            try { nd = calc.ComputeDEDX(e * MeV, pp.def, pp.nuc, m) / (MeV / mm); } catch (...) {}
          }
          std::fprintf(f,
                       "%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                       m->GetName().c_str(), pp.def->GetParticleName().c_str(),
                       pp.def->GetPDGMass() / MeV, pp.def->GetPDGCharge() / eplus, e, di, dt,
                       r, dx, db, dp, bx, px, nd);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- the individual Bethe-Bloch correction terms
  // Diffing the assembled dE/dx says only that something is off; these say which term.
  {
    FILE* f = std::fopen("corrections.csv", "w");
    std::fprintf(f, "material,particle,energy_MeV,shell,barkas,bloch,mott,highorder,"
                    "ionbarkas,effq2ratio\n");
    G4EmCorrections corr(0);
    const G4ParticleDefinition* parts[] = {G4Proton::Proton(), G4AntiProton::AntiProton(),
                                           G4MuonMinus::MuonMinus(), G4Alpha::Alpha(),
                                           G4He3::He3()};
    for (auto* m : mats) {
      const double cut = ecut.count(m) ? ecut[m] : 0.99e-3 * MeV;
      for (const G4ParticleDefinition* pn : parts) {
        for (int i = 0; i <= 220; ++i) {
          const double e = 1e-3 * std::pow(10.0, i / 20.0);
          double sh = 0, ba = 0, bl = 0, mo = 0, ho = 0, ib = 0, eq = 0;
          try { sh = corr.ShellCorrection(pn, m, e * MeV); } catch (...) {}
          try { ba = corr.BarkasCorrection(pn, m, e * MeV); } catch (...) {}
          try { bl = corr.BlochCorrection(pn, m, e * MeV); } catch (...) {}
          try { mo = corr.MottCorrection(pn, m, e * MeV); } catch (...) {}
          try { ho = corr.HighOrderCorrections(pn, m, e * MeV, cut) / (MeV / mm); } catch (...) {}
          try { ib = corr.IonBarkasCorrection(pn, m, e * MeV) / (MeV / mm); } catch (...) {}
          try { eq = corr.EffectiveChargeSquareRatio(pn, m, e * MeV); } catch (...) {}
          std::fprintf(f, "%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                       m->GetName().c_str(), pn->GetParticleName().c_str(), e, sh, ba, bl,
                       mo, ho, ib, eq);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- ionisation parameters, at full precision
  // The Bethe-Bloch bracket is a difference of large logs, so a small error in the mean
  // excitation energy or the Sternheimer coefficients shows up as a constant offset. Dump
  // what G4IonisParamMat actually returns rather than inferring it.
  {
    FILE* f = std::fopen("ionisation_params.csv", "w");
    std::fprintf(f, "material,meanExcitation_MeV,Cbar,x0,x1,a,m,delta0,taul,"
                    "shellCorr0,shellCorr1,shellCorr2\n");
    for (auto* m : mats) {
      auto* ip = m->GetIonisation();
      std::fprintf(f,
                   "%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                   "%.17g,%.17g\n",
                   m->GetName().c_str(), ip->GetMeanExcitationEnergy() / MeV,
                   ip->GetCdensity(), ip->GetX0density(), ip->GetX1density(),
                   ip->GetAdensity(), ip->GetMdensity(), ip->GetD0density(), ip->GetTaul(),
                   ip->GetShellCorrectionVector()[0], ip->GetShellCorrectionVector()[1],
                   ip->GetShellCorrectionVector()[2], ip->GetInvA23(), ip->GetZeffective());
    }
    std::fclose(f);
  }

  // Density correction sampled directly, so the port can diff the function not the inputs.
  {
    FILE* f = std::fopen("density_correction.csv", "w");
    std::fprintf(f, "material,x,delta\n");
    for (auto* m : mats) {
      for (int i = -40; i <= 100; ++i) {
        const double x = i * 0.1;
        std::fprintf(f, "%s,%.9g,%.17g\n", m->GetName().c_str(), x,
                     m->GetIonisation()->DensityCorrection(x));
      }
    }
    std::fclose(f);
  }

  // ---------------- Bethe-Bloch, calling the model class directly
  // G4EmCalculator selects a model by energy; near a model boundary its answer does not say
  // which model produced it, and that ambiguity already caused one false diagnosis in this
  // project (see docs/RISK.md). Instantiating G4BetheBlochModel removes it.
  {
    FILE* f = std::fopen("bethe_bloch.csv", "w");
    std::fprintf(f, "material,particle,energy_MeV,cut_MeV,dedx_MeV_per_mm,xs_per_mm\n");
    G4BetheBlochModel bb;
    G4DataVector cuts;
    cuts.push_back(1.0 * keV);
    const G4ParticleDefinition* parts[] = {
        G4MuonMinus::MuonMinus(), G4MuonPlus::MuonPlus(), G4PionPlus::PionPlus(),
        G4PionMinus::PionMinus(), G4KaonPlus::KaonPlus(), G4Proton::Proton(),
        G4AntiProton::AntiProton(), G4Alpha::Alpha(), G4He3::He3()};
    for (const G4ParticleDefinition* pn : parts) {
      bb.Initialise(pn, cuts);
      for (auto* m : mats) {
        const double cut = ecut.count(m) ? ecut[m] : 0.99e-3 * MeV;
        for (int i = 0; i <= 220; ++i) {
          const double e = 1e-3 * std::pow(10.0, i / 20.0);
          double d = 0, x = 0;
          try { d = bb.ComputeDEDXPerVolume(m, pn, e * MeV, cut) / (MeV / mm); } catch (...) {}
          try { x = bb.CrossSectionPerVolume(m, pn, e * MeV, cut, DBL_MAX) * mm; } catch (...) {}
          std::fprintf(f, "%s,%s,%.9g,%.9g,%.9g,%.9g\n", m->GetName().c_str(),
                       pn->GetParticleName().c_str(), e, cut / MeV, d, x);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- muon models, called directly
  {
    FILE* f = std::fopen("muon_models.csv", "w");
    std::fprintf(f, "material,particle,energy_MeV,cut_MeV,gcut_MeV,muBB_dedx,muBB_xs,"
                    "muBrem_dedx,muBrem_xs,muPair_dedx,muPair_xs\n");
    G4MuBetheBlochModel mubb;
    G4MuBremsstrahlungModel mubr;
    G4MuPairProductionModel mupp;
    G4DataVector cuts;
    cuts.push_back(1.0 * keV);
    const G4ParticleDefinition* parts[2] = {G4MuonMinus::MuonMinus(), G4MuonPlus::MuonPlus()};
    for (const G4ParticleDefinition* pn : parts) {
      mubb.Initialise(pn, cuts);
      mubr.Initialise(pn, cuts);
      mupp.Initialise(pn, cuts);
      for (auto* m : mats) {
        const double cut = ecut.count(m) ? ecut[m] : 0.99e-3 * MeV;
        const double gc = gcut.count(m) ? gcut[m] : 0.99e-3 * MeV;
        for (int i = 0; i <= 220; ++i) {
          const double e = 1e-3 * std::pow(10.0, i / 20.0);
          double d1 = 0, x1 = 0, d2 = 0, x2 = 0, d3 = 0, x3 = 0;
          try { d1 = mubb.ComputeDEDXPerVolume(m, pn, e*MeV, cut) / (MeV/mm); } catch (...) {}
          try { x1 = mubb.CrossSectionPerVolume(m, pn, e*MeV, cut, DBL_MAX) * mm; } catch (...) {}
          try { d2 = mubr.ComputeDEDXPerVolume(m, pn, e*MeV, gc) / (MeV/mm); } catch (...) {}
          try { x2 = mubr.CrossSectionPerVolume(m, pn, e*MeV, gc, DBL_MAX) * mm; } catch (...) {}
          // Pair production is identically zero below 4*m_e = 2.04 MeV, and every B1 cut is
          // below that, so probe it at a cut where the model is actually live.
          const double pcut = 4.0 * CLHEP::electron_mass_c2;
          try { d3 = mupp.ComputeDEDXPerVolume(m, pn, e*MeV, 10.0*pcut) / (MeV/mm); } catch (...) {}
          try { x3 = mupp.CrossSectionPerVolume(m, pn, e*MeV, pcut, DBL_MAX) * mm; } catch (...) {}
          std::fprintf(f, "%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                       m->GetName().c_str(), pn->GetParticleName().c_str(), e, cut/MeV,
                       gc/MeV, d1, x1, d2, x2, d3, x3);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- hadron radiative models, called directly
  // G4hBremsstrahlungModel derives from the muon one and overrides only the differential
  // cross section; G4hPairProductionModel overrides nothing at all.
  {
    FILE* f = std::fopen("hadron_radiative.csv", "w");
    std::fprintf(f, "material,particle,energy_MeV,gcut_MeV,pcut_MeV,brem_dedx,brem_xs,"
                    "pair_dedx,pair_xs\n");
    const G4ParticleDefinition* parts[] = {G4PionPlus::PionPlus(), G4KaonPlus::KaonPlus(),
                                           G4Proton::Proton(), G4AntiProton::AntiProton()};
    G4DataVector cuts;
    cuts.push_back(1.0 * keV);
    const double pcut = 4.0 * CLHEP::electron_mass_c2;
    for (const G4ParticleDefinition* pn : parts) {
      G4hBremsstrahlungModel hbr;
      G4hPairProductionModel hpp(pn);
      hbr.Initialise(pn, cuts);
      hpp.Initialise(pn, cuts);
      for (auto* m : mats) {
        const double gc = gcut.count(m) ? gcut[m] : 0.99e-3 * MeV;
        for (int i = 0; i <= 220; ++i) {
          const double e = 1e-3 * std::pow(10.0, i / 20.0);
          double d1 = 0, x1 = 0, d2 = 0, x2 = 0;
          try { d1 = hbr.ComputeDEDXPerVolume(m, pn, e*MeV, gc) / (MeV/mm); } catch (...) {}
          try { x1 = hbr.CrossSectionPerVolume(m, pn, e*MeV, gc, DBL_MAX) * mm; } catch (...) {}
          try { d2 = hpp.ComputeDEDXPerVolume(m, pn, e*MeV, 10.0*pcut) / (MeV/mm); } catch (...) {}
          try { x2 = hpp.CrossSectionPerVolume(m, pn, e*MeV, pcut, DBL_MAX) * mm; } catch (...) {}
          std::fprintf(f, "%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", m->GetName().c_str(),
                       pn->GetParticleName().c_str(), e, gc/MeV, pcut/MeV, d1, x1, d2, x2);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- Wentzel transport cross section, per element, via the shared engine
  // G4WentzelVIModel and G4eCoulombScatteringModel both drive G4WentzelOKandVIxSection;
  // dumping the engine directly avoids the model-selection ambiguity that has bitten twice.
  {
    FILE* f = std::fopen("wentzel.csv", "w");
    std::fprintf(f, "material,particle,Z,energy_MeV,cut_MeV,cosThetaLim,transport_xs,"
                    "nuclear_xs,electron_xs,oneMinusCosNuc,oneMinusCosElec,screenZ\n");
    const G4ParticleDefinition* parts[] = {G4Electron::Electron(), G4Positron::Positron(),
                                           G4MuonMinus::MuonMinus(), G4Proton::Proton(),
                                           G4PionPlus::PionPlus()};
    const double cosLim = std::cos(CLHEP::pi);  // -1, the uncombined limit
    for (const G4ParticleDefinition* pn : parts) {
      G4WentzelOKandVIxSection wokvi(true);
      wokvi.Initialise(pn, cosLim);
      for (auto* m : mats) {
        const double cut = ecut.count(m) ? ecut[m] : 0.99e-3 * MeV;
        const G4ElementVector* ev = m->GetElementVector();
        for (std::size_t ie = 0; ie < m->GetNumberOfElements(); ++ie) {
          const G4int Z = (*ev)[ie]->GetZasInt();
          for (int i = 0; i <= 160; ++i) {
            const double e = 1e-2 * std::pow(10.0, i / 20.0);  // 10 keV .. 100 TeV
            double tx = 0, nx = 0, ex = 0, ctn = 0, cte = 0, sz = 0;
            try {
              ctn = wokvi.SetupKinematic(e * MeV, m);
              const double cost = wokvi.SetupTarget(Z, cut);
              tx = wokvi.ComputeTransportCrossSectionPerAtom(cost) * mm * mm;
              // Evaluate both over (1, cost): ComputeElectronCrossSection collapses to zero
              // when both arguments fall below cosTetMaxElec, which (cost, cosTetMaxNuc)
              // often does.
              nx = wokvi.ComputeNuclearCrossSection(1.0, cost) * mm * mm;
              ex = wokvi.ComputeElectronCrossSection(1.0, cost) * mm * mm;
              cte = 1.0 - wokvi.GetCosThetaElec();
              ctn = 1.0 - cost;
            } catch (...) {}
            std::fprintf(f, "%s,%s,%d,%.9g,%.9g,%.9g,%.17g,%.17g,%.17g,%.17g,%.17g,%.9g\n",
                         m->GetName().c_str(), pn->GetParticleName().c_str(), Z, e, cut / MeV,
                         cosLim, tx, nx, ex, ctn, cte, sz);
          }
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- standard atomic weights for every element Geant4 knows
  // The port hardcoded ten of these and silently returned 0 for the rest, which turned into
  // an infinite atom density for any user material outside B1's element set.
  {
    FILE* f = std::fopen("atomic_masses.csv", "w");
    std::fprintf(f, "Z,name,atomicMassAmu,Neff,plasmaEnergy_eV\n");
    auto* nist = G4NistManager::Instance();
    for (G4int z = 1; z <= 98; ++z) {
      double a = 0, neff = 0, plasma = 0;
      try { a = nist->GetAtomicMassAmu(z); } catch (...) {}
      // Neff is G4Element::GetN(), the effective nucleon number. It is a different quantity
      // from the atomic mass, and it is the one G4ICRU49NuclearStoppingModel takes as the
      // target mass and G4IonisParamMat takes for <A^(-2/3)>.
      try {
        auto* el = nist->FindOrBuildElement(z);
        if (el != nullptr) { neff = el->GetN(); }
      } catch (...) {}
      // Plasma energy from G4DensityEffectData, which G4ICRU73QOModel needs for the
      // oscillator-energy fallback on elements outside its own 26-element table. The
      // Z-1 retry is what G4ICRU73QOModel::GetOscillatorEnergy itself does.
      try {
        auto* ded = mats[0]->GetIonisation()->GetDensityEffectData();
        G4int idx = ded->GetElementIndex(z, kStateUndefined);
        if (idx == -1) { idx = ded->GetElementIndex(z - 1, kStateUndefined); }
        if (idx >= 0) { plasma = ded->GetPlasmaEnergy(idx) / eV; }
      } catch (...) {}
      std::fprintf(f, "%d,%s,%.17g,%.17g,%.17g\n", z,
                   nist->GetNistElementNames()[z].c_str(), a, neff, plasma);
    }
    std::fclose(f);
  }

  // ---------------- Bragg models, called directly, plus which data path they take
  {
    FILE* f = std::fopen("bragg.csv", "w");
    std::fprintf(f, "material,particle,energy_MeV,cut_MeV,bragg_dedx,braggion_dedx,"
                    "pstar_index,pstar_dedx\n");
    // Heap-allocated and deliberately not deleted: these models share global state with the
    // run manager, and destroying them here double-frees it. This is a short-lived dumper.
    auto* pstar = new G4PSTARStopping();
    pstar->Initialise();
    auto* astar = new G4ASTARStopping();
    astar->Initialise();
    G4DataVector cuts;
    cuts.push_back(1.0 * keV);
    const G4ParticleDefinition* parts[] = {G4Proton::Proton(), G4PionPlus::PionPlus(),
                                           G4KaonPlus::KaonPlus(), G4Alpha::Alpha()};
    for (const G4ParticleDefinition* pn : parts) {
      auto* br = new G4BraggModel();
      auto* bi = new G4BraggIonModel();
      br->Initialise(pn, cuts);
      bi->Initialise(pn, cuts);
      for (auto* m : mats) {
        const double cut = ecut.count(m) ? ecut[m] : 0.99e-3 * MeV;
        const G4int ip = pstar->GetIndex(m);
        const G4int ia = astar->GetIndex(m);
        for (int i = 0; i <= 120; ++i) {
          const double e = 1e-4 * std::pow(10.0, i / 20.0);  // 0.1 keV .. 100 MeV
          double d1 = 0, d2 = 0, dp = 0;
          try { d1 = br->ComputeDEDXPerVolume(m, pn, e * MeV, cut) / (MeV / mm); } catch (...) {}
          try { d2 = bi->ComputeDEDXPerVolume(m, pn, e * MeV, cut) / (MeV / mm); } catch (...) {}
          if (ip >= 0) {
            try {
              dp = pstar->GetElectronicDEDX(ip, e * MeV) * m->GetDensity() / (MeV / mm);
            } catch (...) {}
          }
          std::fprintf(f, "%s,%s,%.9g,%.9g,%.9g,%.9g,%d,%.9g,%d\n", m->GetName().c_str(),
                       pn->GetParticleName().c_str(), e, cut / MeV, d1, d2, ip, dp, ia);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- CLHEP's physical constants, as this Geant4 defines them
  //
  // These are not universal: CLHEP pins them per release. 11.1.1 carries
  // electron_mass_c2 = 0.510998910 MeV and amu_c2 = 931.494028 MeV, both of which are older
  // than the current CODATA values a person would look up - and the port had the *newer*
  // amu_c2, 931.49410242, which put the helium effective charge 1.5e-8 off Geant4's answer.
  // That is harmless in a dose and fatal to a 1e-9 comparison, and it took an afternoon to
  // find because every plausible cause was a formula.
  //
  // So the constants are dumped and compared like any other physics. A constant that differs
  // from the reference implementation's is a transcription error, whatever CODATA says.
  {
    FILE* f = std::fopen("constants.csv", "w");
    std::fprintf(f, "name,value,unit\n");
    std::fprintf(f, "electron_mass_c2,%.17g,MeV\n", CLHEP::electron_mass_c2 / MeV);
    std::fprintf(f, "proton_mass_c2,%.17g,MeV\n", CLHEP::proton_mass_c2 / MeV);
    std::fprintf(f, "neutron_mass_c2,%.17g,MeV\n", CLHEP::neutron_mass_c2 / MeV);
    std::fprintf(f, "amu_c2,%.17g,MeV\n", CLHEP::amu_c2 / MeV);
    std::fprintf(f, "Avogadro,%.17g,per_mole\n", CLHEP::Avogadro * mole);
    std::fprintf(f, "classic_electr_radius,%.17g,mm\n", CLHEP::classic_electr_radius / mm);
    std::fprintf(f, "fine_structure_const,%.17g,1\n", CLHEP::fine_structure_const);
    std::fprintf(f, "twopi_mc2_rcl2,%.17g,MeV_mm2\n", CLHEP::twopi_mc2_rcl2 / (MeV * mm * mm));
    std::fprintf(f, "electron_Compton_length,%.17g,mm\n", CLHEP::electron_Compton_length / mm);
    std::fprintf(f, "Bohr_radius,%.17g,mm\n", CLHEP::Bohr_radius / mm);
    std::fprintf(f, "hbarc,%.17g,MeV_mm\n", CLHEP::hbarc / (MeV * mm));
    std::fprintf(f, "barn,%.17g,mm2\n", CLHEP::barn / (mm * mm));
    std::fprintf(f, "pi,%.17g,1\n", CLHEP::pi);
    std::fclose(f);
    std::printf("wrote constants.csv\n");
  }

  // ---------------- G4IonisParamMat's derived per-material quantities
  //
  // These are the numbers every ion and MSC formula is built on, and until now none of them
  // was checked directly: they were only ever visible through a dE/dx that has a dozen other
  // inputs. Two of them changed today - fermi_energy is new, and inv_a23 was switched from an
  // exact 2/3 power to G4Pow's Taylor approximation - so both are dumped at the source.
  //
  // invA23 is the one worth being careful about: Geant4 divides by
  // `G4Pow::A23(G4Element::GetN())`, GetN() returns fNeff (the abundance-weighted nucleon
  // number, not an integer), and A23 is an approximation to the 2/3 power rather than the
  // power itself. Each of those is worth a few parts in 10^4 or 10^5 and each is invisible in
  // a dose.
  {
    FILE* f = std::fopen("material_ionis.csv", "w");
    std::fprintf(f, "material,z_eff,fermi_energy_MeV,inv_a23,lfactor,mean_excitation_MeV,"
                    "electron_density_permm3\n");
    for (auto* m : mats) {
      const G4IonisParamMat* ip = m->GetIonisation();
      std::fprintf(f, "%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", m->GetName().c_str(),
                   ip->GetZeffective(), ip->GetFermiEnergy() / MeV, ip->GetInvA23(),
                   ip->GetLFactor(), ip->GetMeanExcitationEnergy() / MeV,
                   m->GetElectronDensity() * mm3);
    }
    std::fclose(f);
    std::printf("wrote material_ionis.csv\n");
  }

  // ---------------- G4ScreeningMottCrossSection: the Mott/Rutherford ratio
  //
  // G4WentzelOKandVIxSection creates one of these for electrons and positrons unconditionally
  // ("Mott corrections always added") and uses RatioMottRutherfordCosT as the rejection
  // function when it samples a single scatter. For every other particle it uses an analytic
  // expression instead, and the analytic one is what this port had - so the ratio is the last
  // piece of the single-scattering angle distribution for e+-.
  //
  // Two files, because the ratio needs a quantity that is itself a table.
  // G4ScreeningMottCrossSection::SetupKinematic computes beta not from the lab kinematics but
  // from the relativistic reduced mass of the projectile and the *target nucleus*, so it needs
  // a nuclear mass per Z. Geant4 gets that from G4NucleiProperties, which is an AME12 mass
  // table with a mass-excess formula behind it - far more machinery than the 92 numbers this
  // actually uses. So the 92 numbers are dumped.
  //
  // beta is dumped alongside the ratio for the same reason the ion masses were: if the
  // transcription is wrong, the difference between a wrong beta and a wrong polynomial is the
  // difference between a two-line fix and a day.
  {
    FILE* f = std::fopen("mott_target.csv", "w");
    std::fprintf(f, "Z,A,atomic_mass_amu,nuclear_mass_MeV\n");
    auto* nist = G4NistManager::Instance();
    for (G4int z = 1; z <= 92; ++z) {
      const G4double amu = nist->GetAtomicMassAmu(z);
      const G4int a = G4lrint(amu);
      std::fprintf(f, "%d,%d,%.17g,%.17g\n", z, a, amu,
                   G4NucleiProperties::GetNuclearMass(a, z) / MeV);
    }
    std::fclose(f);
    std::printf("wrote mott_target.csv\n");
  }

  // `beta` is private in G4ScreeningMottCrossSection with no accessor, so it cannot be
  // dumped. What replaces it is the fcost = 0 point, which is in the sweep below: there the
  // ratio collapses to the j = 0 row alone,
  //
  //     R(fcost = 0) = sum_k coef[Z][0][k] * (beta - 0.7181228)^k
  //
  // a function of beta and Z and nothing else. So a disagreement at fcost = 0 is the beta
  // chain or the coefficient row, and a disagreement only at fcost > 0 is the angular
  // polynomial. That is the split worth having; the exact value of beta is not.
  {
    FILE* f = std::fopen("mott_ratio.csv", "w");
    std::fprintf(f, "particle,Z,ekin_MeV,fcost,ratio\n");
    auto* smcs = new G4ScreeningMottCrossSection();
    const G4ParticleDefinition* parts[2] = {G4Electron::Electron(), G4Positron::Positron()};
    // A spread of Z rather than all 92: the coefficient table is checked row by row by the
    // count assertion in the extractor, and what needs checking here is the polynomial and
    // the beta chain, which do not depend on Z beyond the lookup.
    const G4int zs[8] = {1, 6, 8, 13, 26, 47, 79, 92};
    for (const G4ParticleDefinition* p : parts) {
      smcs->SetupParticle(p);
      smcs->Initialise(p, 1.0);
      for (G4int z : zs) {
        // 10 keV to 1 GeV. Below ~10 keV WentzelVI is not the model in any standard physics
        // list, and above 1 GeV beta has saturated.
        for (G4int i = 0; i <= 50; ++i) {
          const G4double e = 1e-2 * std::pow(10.0, i / 10.0);
          smcs->SetupKinematic(e * MeV, z);
          // fcost = sqrt(1 - cos theta), which is what the sampler passes: sqrt(z1).
          for (G4int j = 0; j <= 20; ++j) {
            const G4double fcost = j / 20.0 * std::sqrt(2.0);
            std::fprintf(f, "%s,%d,%.17g,%.17g,%.17g\n", p->GetParticleName().c_str(), z, e,
                         fcost, smcs->RatioMottRutherfordCosT(fcost));
          }
        }
      }
    }
    std::fclose(f);
    std::printf("wrote mott_ratio.csv\n");
  }

  // ---------------- ICRU 90 stopping powers, dumped directly
  //
  // G4EmParameters::SetUseICRU90Data is off by default, and turning it on here would change
  // every other number in this file for air, water and graphite - G4BraggModel resolves
  // iICRU90 before iPSTAR, so the flag replaces the stopping power rather than adding to it.
  // So the data object is asked directly, exactly as G4ionEffectiveCharge was: this is the
  // quantity the port's table and spline have to reproduce, and it is the same number the
  // model would return with the flag on.
  //
  // Graphite is added to the material list for this, because it is one of the three materials
  // ICRU 90 covers and nothing else in this dumper needed it. Without it the table would be
  // checked for two rows out of three.
  {
    FILE* f = std::fopen("icru90.csv", "w");
    std::fprintf(f, "material,particle,energy_MeV,mass_stopping_MeV_cm2_g\n");
    auto* icru = G4NistManager::Instance()->GetICRU90StoppingData();
    if (icru == nullptr) {
      std::printf("WARNING: no ICRU90 stopping data - icru90.csv will be empty\n");
    } else {
      const char* names[3] = {"G4_AIR", "G4_WATER", "G4_GRAPHITE"};
      // Built *before* Initialise(), which scans the material table once and latches when it
      // has found all three. Building graphite afterwards left it with no index and two rows
      // of the table untested - the warning below said so, which is the only reason it was
      // noticed rather than shipped.
      for (const char* nm : names) { G4NistManager::Instance()->FindOrBuildMaterial(nm); }
      icru->Initialise();
      for (const char* nm : names) {
        auto* m = G4NistManager::Instance()->FindOrBuildMaterial(nm);
        if (m == nullptr) { continue; }
        const G4int idx = icru->GetIndex(m);
        if (idx < 0) {
          // Loud: a silent -1 here would leave the row untested while the file still parsed.
          std::printf("WARNING: %s has no ICRU90 index - it was not in the material table\n",
                      nm);
          continue;
        }
        // 1 eV to 20 GeV: below the table's first point (1 keV) to exercise the sqrt
        // extrapolation, and above its last (10 GeV for protons) to exercise the clamp.
        for (G4int i = 0; i <= 200; ++i) {
          const G4double e = 1e-6 * std::pow(10.0, i / 20.0);
          std::fprintf(f, "%s,proton,%.17g,%.17g\n", nm, e,
                       icru->GetElectronicDEDXforProton(idx, e * MeV) / (MeV * cm2 / g));
          std::fprintf(f, "%s,alpha,%.17g,%.17g\n", nm, e,
                       icru->GetElectronicDEDXforAlpha(idx, e * MeV) / (MeV * cm2 / g));
        }
      }
    }
    std::fclose(f);
    std::printf("wrote icru90.csv\n");
  }

  // ---------------- G4ionEffectiveCharge, both branches
  //
  // The helium branch (Zi <= 2) was already checked through the Bethe-Bloch corrections, but
  // the heavy-ion branch was not reachable from anything this dumper produced: nothing here
  // used an ion heavier than an alpha. So it is dumped directly, for nine ions from helium to
  // uranium, which is what makes the Zi > 2 transcription checkable at all.
  //
  // 11.1.1 exposes no ComputeCharge - `chargeCorrection` is a private member with no accessor.
  // What it does expose is the pair that defines the correction:
  //
  //     EffectiveCharge(p,m,E)            == effCharge
  //     EffectiveChargeSquareRatio(p,m,E) == (effCharge * chargeCorrection * inveplus)^2
  //
  // so both of Geant4's outputs are dumped exactly as it publishes them, and the correction is
  // whatever makes the two agree. The ratio is dumped rather than a correction derived from it
  // because the ratio is the quantity G4EmCorrections actually consumes: a transcription with
  // the right charge and the wrong correction fails on the ratio, and cannot fail on a
  // correction that was computed from the ratio in the first place.
  //
  // The order of the two calls matters only for cost - EffectiveChargeSquareRatio calls
  // EffectiveCharge itself, and the (particle, material, energy) cache makes the second free.
  //
  // The mass is dumped alongside because the branch divides by it: the port builds its own
  // particle record and has to use the same PDG mass, not a nucleon count times an amu.
  {
    FILE* f = std::fopen("ion_charge.csv", "w");
    std::fprintf(f, "material,ion,Z,A,mass_MeV,energy_MeV,eff_charge,charge_square_ratio\n");
    auto* iec = new G4ionEffectiveCharge();
    auto* itab = G4IonTable::GetIonTable();
    struct Ion {
      G4int z, a;
    };
    const Ion ions[] = {{2, 4},   {3, 7},    {6, 12},   {8, 16},  {13, 27},
                        {26, 56}, {54, 132}, {79, 197}, {92, 238}};
    G4int missing = 0;
    for (const Ion& io : ions) {
      const G4ParticleDefinition* p = itab->GetIon(io.z, io.a, 0.0);
      if (p == nullptr) {
        ++missing;
        continue;
      }
      for (auto* m : mats) {
        for (G4int i = 0; i <= 100; ++i) {
          const G4double e = 1e-3 * std::pow(10.0, i / 20.0);  // 1 keV .. 100 GeV
          const G4double q = iec->EffectiveCharge(p, m, e * MeV) / CLHEP::eplus;
          const G4double r = iec->EffectiveChargeSquareRatio(p, m, e * MeV);
          std::fprintf(f, "%s,%s,%d,%d,%.17g,%.17g,%.17g,%.17g\n", m->GetName().c_str(),
                       p->GetParticleName().c_str(), io.z, io.a, p->GetPDGMass() / MeV, e, q, r);
        }
      }
    }
    std::fclose(f);
    // Loud, because a silent nullptr would leave the heavy-ion branch untested while every
    // check still passed - the exact failure this file exists to rule out.
    if (missing != 0) {
      std::printf("WARNING: G4IonTable had no definition for %d of the %d ions\n", missing,
                  (G4int)(sizeof ions / sizeof ions[0]));
    }
    std::printf("wrote ion_charge.csv\n");
  }

  // ---------------- ICRU73QO, the negative-hadron low-energy model
  {
    FILE* f = std::fopen("icru73qo.csv", "w");
    std::fprintf(f, "material,particle,energy_MeV,cut_MeV,qo_dedx\n");
    G4DataVector cuts;
    cuts.push_back(1.0 * keV);
    const G4ParticleDefinition* parts[] = {G4PionMinus::PionMinus(),
                                           G4AntiProton::AntiProton(),
                                           G4MuonMinus::MuonMinus()};
    for (const G4ParticleDefinition* pn : parts) {
      auto* qo = new G4ICRU73QOModel();
      qo->Initialise(pn, cuts);
      for (auto* m : mats) {
        const double cut = ecut.count(m) ? ecut[m] : 0.99e-3 * MeV;
        for (int i = 0; i <= 100; ++i) {
          const double e = 1e-3 * std::pow(10.0, i / 25.0);  // 1 keV .. 10 MeV
          double d = 0;
          try { d = qo->ComputeDEDXPerVolume(m, pn, e * MeV, cut) / (MeV / mm); } catch (...) {}
          std::fprintf(f, "%s,%s,%.9g,%.9g,%.9g\n", m->GetName().c_str(),
                       pn->GetParticleName().c_str(), e, cut / MeV, d);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- nuclear stopping, called directly
  {
    FILE* f = std::fopen("nuclear_stopping.csv", "w");
    std::fprintf(f, "material,particle,energy_MeV,nuclear_dedx\n");
    G4DataVector cuts;
    cuts.push_back(1.0 * keV);
    const G4ParticleDefinition* parts[] = {G4Proton::Proton(), G4Alpha::Alpha(),
                                           G4He3::He3()};
    // The model needs no per-particle initialisation: its constructor fills the Z^0.23
    // table and Initialise() is empty. Constructing one per particle in a loop is what
    // crashed the dumper.
    auto* ns = new G4ICRU49NuclearStoppingModel();
    // G4VEmModel::lossFlucFlag defaults to true, so ComputeDEDXPerVolume multiplies the
    // result by a Gaussian deviate and returns a different number every call. Comparing a
    // deterministic transcription against single random samples produced scatter of a few
    // percent and one 1.6x outlier that looked like a real error. Turn it off to get the
    // mean, which is what the port computes.
    ns->SetFluctuationFlag(false);
    for (const G4ParticleDefinition* pn : parts) {
      for (auto* m : mats) {
        for (int i = 0; i <= 140; ++i) {
          const double e = 1e-5 * std::pow(10.0, i / 20.0);  // 10 eV .. 100 MeV
          double d = 0;
          try { d = ns->ComputeDEDXPerVolume(m, pn, e * MeV, 0.0) / (MeV / mm); } catch (...) {}
          std::fprintf(f, "%s,%s,%.9g,%.9g\n", m->GetName().c_str(),
                       pn->GetParticleName().c_str(), e, d);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- single Coulomb scattering, e-/e+ (the >100 MeV companion to WentzelVI)
  {
    FILE* f = std::fopen("coulomb.csv", "w");
    std::fprintf(f, "material,particle,energy_MeV,coulomb_per_mm\n");
    const G4ParticleDefinition* parts[4] = {G4Electron::Electron(), G4Positron::Positron(),
                                            G4MuonMinus::MuonMinus(), G4Proton::Proton()};
    for (auto* m : mats) {
      for (const G4ParticleDefinition* pn : parts) {
        for (int i = 0; i <= 180; ++i) {
          const double e = 1e-3 * std::pow(10.0, i / 20.0);
          double v = 0;
          try { v = calc.GetCrossSectionPerVolume(e * MeV, pn, "CoulombScat", m) * mm; }
          catch (...) {}
          if (v <= 0) {
            try {
              const double mfp = calc.GetMeanFreePath(e * MeV, pn, "CoulombScat", m);
              if (mfp > 0 && mfp < DBL_MAX) { v = mm / mfp; }
            } catch (...) {}
          }
          std::fprintf(f, "%s,%s,%.9g,%.9g\n", m->GetName().c_str(),
                       pn->GetParticleName().c_str(), e, v);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- material parameters we derive ourselves
  {
    FILE* f = std::fopen("materials.csv", "w");
    std::fprintf(f,
                 "material,density_g_cm3,electron_density_per_mm3,Zeff,"
                 "mean_excitation_eV,radlen_mm,nuclear_interaction_length_mm\n");
    // Note: G4Material::GetA() raises a G4Exception for multi-element materials, so it is
    // deliberately not dumped here.
    for (auto* m : mats) {
      std::fprintf(f, "%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", m->GetName().c_str(),
                   m->GetDensity() / (g / cm3), m->GetElectronDensity() * mm3,
                   m->GetIonisation()->GetZeffective(),
                   m->GetIonisation()->GetMeanExcitationEnergy() / eV,
                   m->GetRadlen() / mm, m->GetNuclearInterLength() / mm);
    }
    std::fclose(f);
  }

  // ------------------------------------------------------------------ solids
  //
  // Ray queries against each G4VSolid: containment, entry distance and exit distance for a
  // fixed pseudo-random set of points and directions. This is the reference the port's
  // geometry engine is checked against, and it is generated the same way the physics
  // references are - by asking Geant4 directly rather than by reasoning about the shapes.
  {
    struct Entry { const char* name; G4VSolid* solid; double reach; };
    std::vector<Entry> solids;
    solids.push_back({"box", new G4Box("b", 30 * mm, 40 * mm, 50 * mm), 90});
    solids.push_back({"tubs_full", new G4Tubs("t1", 0, 40 * mm, 50 * mm, 0, twopi), 90});
    solids.push_back({"tubs_hollow", new G4Tubs("t2", 15 * mm, 40 * mm, 50 * mm, 0, twopi), 90});
    solids.push_back({"tubs_wedge",
                      new G4Tubs("t3", 10 * mm, 40 * mm, 50 * mm, 20 * deg, 200 * deg), 90});
    solids.push_back({"cons_full",
                      new G4Cons("c1", 0, 20 * mm, 0, 40 * mm, 30 * mm, 0, twopi), 80});
    solids.push_back({"cons_hollow",
                      new G4Cons("c2", 5 * mm, 20 * mm, 10 * mm, 40 * mm, 30 * mm, 0, twopi), 80});
    solids.push_back({"cons_wedge",
                      new G4Cons("c3", 5 * mm, 20 * mm, 10 * mm, 40 * mm, 30 * mm,
                                 30 * deg, 150 * deg), 80});
    solids.push_back({"orb", new G4Orb("o", 45 * mm), 90});
    solids.push_back({"sphere_full",
                      new G4Sphere("s1", 0, 45 * mm, 0, twopi, 0, pi), 90});
    solids.push_back({"sphere_shell",
                      new G4Sphere("s2", 20 * mm, 45 * mm, 0, twopi, 0, pi), 90});
    solids.push_back({"sphere_wedge",
                      new G4Sphere("s3", 20 * mm, 45 * mm, 20 * deg, 200 * deg,
                                   30 * deg, 90 * deg), 90});
    solids.push_back({"torus", new G4Torus("to", 0, 12 * mm, 40 * mm, 0, twopi), 80});
    solids.push_back({"torus_hollow",
                      new G4Torus("to2", 5 * mm, 12 * mm, 40 * mm, 0, twopi), 80});
    solids.push_back({"torus_wedge",
                      new G4Torus("to3", 5 * mm, 12 * mm, 40 * mm, 30 * deg, 200 * deg), 80});
    solids.push_back({"trd", new G4Trd("td", 30 * mm, 20 * mm, 40 * mm, 25 * mm, 50 * mm), 90});
    solids.push_back({"para",
                      new G4Para("pa", 30 * mm, 40 * mm, 50 * mm, 20 * deg, 15 * deg,
                                 25 * deg), 130});
    solids.push_back({"trap",
                      new G4Trap("tp", 50 * mm, 10 * deg, 20 * deg, 30 * mm, 20 * mm, 26 * mm,
                                 0, 35 * mm, 22 * mm, 29 * mm, 0), 110});
    solids.push_back({"eltube", new G4EllipticalTube("et", 30 * mm, 45 * mm, 50 * mm), 90});
    solids.push_back({"ellipsoid",
                      new G4Ellipsoid("el", 30 * mm, 45 * mm, 60 * mm, -40 * mm, 50 * mm), 90});
    solids.push_back({"elcone",
                      new G4EllipticalCone("ec", 0.5, 0.8, 50 * mm, 30 * mm), 90});
    solids.push_back({"paraboloid",
                      new G4Paraboloid("pb", 40 * mm, 10 * mm, 35 * mm), 80});
    solids.push_back({"hype", new G4Hype("hy", 10 * mm, 25 * mm, 15 * deg, 30 * deg, 45 * mm),
                      90});
    {
      G4ThreeVector a(0, 0, 40 * mm), b(40 * mm, 0, -20 * mm), c(-25 * mm, 30 * mm, -20 * mm),
          d(-25 * mm, -30 * mm, -20 * mm);
      solids.push_back({"tet", new G4Tet("te", a, b, c, d), 80});
    }
    {
      const G4int n = 4;
      G4double z[n] = {-50 * mm, -10 * mm, 20 * mm, 50 * mm};
      G4double ri[n] = {0, 8 * mm, 8 * mm, 0};
      G4double ro[n] = {20 * mm, 40 * mm, 25 * mm, 35 * mm};
      solids.push_back({"polycone", new G4Polycone("pc", 0, twopi, n, z, ri, ro), 90});
      solids.push_back({"polycone_wedge",
                        new G4Polycone("pc2", 30 * deg, 180 * deg, n, z, ri, ro), 90});
      solids.push_back({"polyhedra", new G4Polyhedra("ph", 0, twopi, 6, n, z, ri, ro), 90});
    }
    // Booleans over two boxes and a cylinder, the three operations the GUI offers.
    {
      auto* ba = new G4Box("ba", 30 * mm, 30 * mm, 30 * mm);
      auto* bb = new G4Box("bb", 20 * mm, 20 * mm, 50 * mm);
      auto* cy = new G4Tubs("cy", 0, 18 * mm, 60 * mm, 0, twopi);
      G4ThreeVector off(15 * mm, 10 * mm, 0);
      solids.push_back({"union_box_box",
                        new G4UnionSolid("u1", ba, bb, nullptr, off), 100});
      solids.push_back({"subtract_box_cyl",
                        new G4SubtractionSolid("s1", ba, cy, nullptr,
                                               G4ThreeVector(10 * mm, 0, 0)), 80});
      solids.push_back({"intersect_box_box",
                        new G4IntersectionSolid("i1", ba, bb, nullptr, off), 80});
      auto* rot = new G4RotationMatrix();
      rot->rotateY(35 * deg);
      solids.push_back({"subtract_rotated_cyl",
                        new G4SubtractionSolid("s2", ba, cy, rot,
                                               G4ThreeVector(5 * mm, 0, 0)), 80});
    }

    FILE* f = std::fopen("solids.csv", "w");
    std::fprintf(f, "solid,px,py,pz,dx,dy,dz,inside,dist_in,dist_out\n");
    // A fixed linear congruential sequence so the sample set is reproducible and does not
    // depend on Geant4's RNG state.
    unsigned long long seed = 88172645463325252ULL;
    auto rnd = [&]() {
      seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
      return (seed >> 11) * (1.0 / 9007199254740992.0);
    };
    for (const Entry& e : solids) {
      for (int i = 0; i < 4000; ++i) {
        const G4ThreeVector p((2 * rnd() - 1) * e.reach * mm, (2 * rnd() - 1) * e.reach * mm,
                              (2 * rnd() - 1) * e.reach * mm);
        const double ct = 2 * rnd() - 1, st = std::sqrt(1 - ct * ct), ph = twopi * rnd();
        const G4ThreeVector v(st * std::cos(ph), st * std::sin(ph), ct);
        const EInside in = e.solid->Inside(p);
        const int icode = (in == kOutside) ? 0 : ((in == kSurface) ? 1 : 2);
        // Geant4 only promises DistanceToIn from outside and DistanceToOut from inside.
        const double din = (in == kOutside) ? e.solid->DistanceToIn(p, v) : -1.0;
        const double dout = (in == kInside) ? e.solid->DistanceToOut(p, v, false, nullptr, nullptr)
                                            : -1.0;
        std::fprintf(f, "%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g\n", e.name,
                     p.x() / mm, p.y() / mm, p.z() / mm, v.x(), v.y(), v.z(), icode,
                     (din > 0.5 * kInfinity) ? -2.0 : din / mm,
                     (dout > 0.5 * kInfinity) ? -2.0 : dout / mm);
      }
    }
    std::fclose(f);


    // A second, independent determination of the entry distance, derived from Inside() alone.
    //
    // Inside() *is* the definition of the solid; DistanceToIn(p,v) is an algorithm that is
    // supposed to agree with it. Where the two disagree, only Inside() can settle which answer
    // is right - and they do disagree: G4Hype::DistanceToIn returns kInfinity for rays that
    // start inside its inner hyperboloid and cross into the material, and returns a crossing
    // beyond the z half-length for others. Comparing the port against DistanceToIn alone would
    // record those as port defects.
    //
    // The scan steps along the same ray in 0.5 mm increments and bisects the first
    // outside -> inside transition. It cannot see a feature thinner than the step, which is
    // fine for these test solids and is why it is a cross-check rather than the primary
    // reference.
    {
      FILE* g = std::fopen("solid_scan.csv", "w");
      std::fprintf(g, "solid,ray,first_inside_t\n");
      unsigned long long seed2 = 88172645463325252ULL;
      auto rnd2 = [&]() {
        seed2 ^= seed2 << 13; seed2 ^= seed2 >> 7; seed2 ^= seed2 << 17;
        return (seed2 >> 11) * (1.0 / 9007199254740992.0);
      };
      for (const Entry& e : solids) {
        for (int i = 0; i < 4000; ++i) {
          const G4ThreeVector p((2 * rnd2() - 1) * e.reach * mm, (2 * rnd2() - 1) * e.reach * mm,
                                (2 * rnd2() - 1) * e.reach * mm);
          const double ct = 2 * rnd2() - 1, st = std::sqrt(1 - ct * ct), ph = twopi * rnd2();
          const G4ThreeVector v(st * std::cos(ph), st * std::sin(ph), ct);
          if (e.solid->Inside(p) != kOutside) { continue; }  // only entry is being checked

          const double h = 0.5, tmax = 4.0 * e.reach;
          double found = -2.0;
          double t0 = 0.0;
          for (double t = h; t <= tmax; t += h) {
            if (e.solid->Inside(p + t * v * mm) == kInside) {
              double lo = t0, hi = t;
              for (int k = 0; k < 45; ++k) {
                const double mid = 0.5 * (lo + hi);
                if (e.solid->Inside(p + mid * v * mm) == kInside) { hi = mid; } else { lo = mid; }
              }
              found = 0.5 * (lo + hi);
              break;
            }
            t0 = t;
          }
          std::fprintf(g, "%s,%d,%.17g\n", e.name, i, found);
        }
      }
      std::fclose(g);
      std::printf("wrote solid_scan.csv\n");
    }
    std::printf("wrote solids.csv (%d solids)\n", static_cast<int>(solids.size()));
  }

  // ------------------------------------------------------------------ NIST material table
  //
  // Emits a C++ header rather than a CSV. The NIST compositions are compiled into Geant4 as
  // arrays (G4NistMaterialBuilder), not shipped as data files, so this port carries them the
  // same way - but generated from Geant4 rather than retyped, because a mistyped mass fraction
  // in a 300-material table is invisible until a dose comes out wrong in one material only.
  {
    auto* nist = G4NistManager::Instance();
    const std::vector<G4String>& names = nist->GetNistMaterialNames();
    FILE* f = std::fopen("nist_materials.hh", "w");
    std::fprintf(f,
                 "// NIST material compositions, extracted from Geant4 11.1.1 G4NistManager by\n"
                 "// ref/dump/g4dump.cc. Do not edit by hand; re-run the oracle instead.\n"
                 "//\n"
                 "// Each entry is a density in g/cm3, a mean excitation energy in eV, a state,\n"
                 "// the tabulated Sternheimer density-effect parameters where Geant4 has them,\n"
                 "// and the element mass fractions.\n"
                 "#pragma once\n"
                 "#include \"g4/G4Types.hh\"\n\n"
                 "namespace g4gpu::g4::nist {\n\n"
                 "struct NistComponent { int z; double fraction; };\n"
                 "struct NistMaterial {\n"
                 "  const char* name;\n"
                 "  double density_g_cm3;\n"
                 "  double mean_excitation_eV;\n"
                 "  int state;            ///< 0 undefined, 1 solid, 2 liquid, 3 gas\n"
                 "  bool has_sternheimer;\n"
                 "  double cbar, x0, x1, a, m, delta0;\n"
                 "  int n_components;\n"
                 "  const NistComponent* components;\n"
                 "};\n\n");

    std::vector<G4String> kept;
    for (const G4String& nm : names) {
      if (nm.rfind("G4_", 0) != 0) { continue; }
      const G4Material* mm = nist->FindOrBuildMaterial(nm);
      if (mm == nullptr || mm->GetNumberOfElements() == 0) { continue; }
      G4String sym = nm;
      for (char& ch : sym) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) { ch = '_'; }
      }
      std::fprintf(f, "inline constexpr NistComponent c_%s[] = {", sym.c_str());
      const G4ElementVector* ev = mm->GetElementVector();
      const G4double* fr = mm->GetFractionVector();
      for (std::size_t i = 0; i < mm->GetNumberOfElements(); ++i) {
        std::fprintf(f, "%s{%d, %.17g}", (i == 0) ? "" : ", ", (*ev)[i]->GetZasInt(), fr[i]);
      }
      std::fprintf(f, "};\n");
      kept.push_back(nm);
    }

    std::fprintf(f, "\ninline constexpr NistMaterial kNistMaterials[] = {\n");
    for (const G4String& nm : kept) {
      const G4Material* mm = nist->FindOrBuildMaterial(nm);
      auto* ip = mm->GetIonisation();
      G4String sym = nm;
      for (char& ch : sym) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) { ch = '_'; }
      }
      const G4State st = mm->GetState();
      const int state = (st == kStateSolid) ? 1 : (st == kStateLiquid) ? 2
                        : (st == kStateGas) ? 3 : 0;
      // Cdensity is zero only when Geant4 never filled the parameters in.
      const bool has = (ip->GetCdensity() != 0.0);
      std::fprintf(f,
                   "  {\"%s\", %.17g, %.17g, %d, %s, %.17g, %.17g, %.17g, %.17g, %.17g, %.17g,"
                   " %d, c_%s},\n",
                   nm.c_str(), mm->GetDensity() / (g / cm3),
                   ip->GetMeanExcitationEnergy() / eV, state, has ? "true" : "false",
                   ip->GetCdensity(), ip->GetX0density(), ip->GetX1density(),
                   ip->GetAdensity(), ip->GetMdensity(), ip->GetD0density(),
                   static_cast<int>(mm->GetNumberOfElements()), sym.c_str());
    }
    std::fprintf(f, "};\n\ninline constexpr int kNumNistMaterials =\n"
                    "    static_cast<int>(sizeof(kNistMaterials) / sizeof(kNistMaterials[0]));\n"
                    "\n}  // namespace g4gpu::g4::nist\n");
    std::fclose(f);
    std::printf("wrote nist_materials.hh (%d materials)\n", static_cast<int>(kept.size()));
  }

  // ---------------- Barashenkov nucleon-nucleus cross sections
  //
  // G4NucleonNuclearCrossSection wraps G4ComponentBarNucleonNucleusXsc, which is what both
  // G4BGGNucleonElasticXS and G4BGGNucleonInelasticXS delegate to between 14 MeV and 91 GeV -
  // so for a proton at any energy anything is transported at, this table *is* the hadronic
  // cross section. Dumped for every Z from 2 to 92 rather than for the elements one example
  // happens to use: seventeen of those Z values are tabulated in Geant4 and the other
  // seventy-four are an interpolation in A, and the interpolation is the half more likely to
  // be transcribed wrongly.
  //
  // Elastic is total minus inelastic on both sides; it is dumped anyway so that a sign or an
  // ordering error shows up as its own column rather than as a difference of two right ones.
  {
    FILE* f = std::fopen("nucleon_xs.csv", "w");
    std::fprintf(f, "Z,energy_MeV,total_mm2,inelastic_mm2,elastic_mm2,particle\n");
    G4NucleonNuclearCrossSection xs;
    const G4ParticleDefinition* parts[2] = {G4Proton::Proton(), G4Neutron::Neutron()};
    const char* pname[2] = {"proton", "neutron"};
    for (int pi = 0; pi < 2; ++pi) {
      for (G4int Z = 2; Z <= 92; ++Z) {
        // 14 MeV to 1 TeV, 12 points per decade. The tabulated grids are irregular and much
        // coarser than this, so most of these points land between two of Geant4's own and
        // exercise the energy interpolation rather than the table entries.
        for (int i = 0; i <= 59; ++i) {
          const double e = 14.0 * std::pow(10.0, i / 12.0);
          if (e > 1.0e6) { break; }
          G4DynamicParticle dp(parts[pi], G4ThreeVector(0, 0, 1), e * MeV);
          double el = 0, tot = 0, inel = 0;
          try {
            el = xs.GetElasticCrossSection(&dp, Z) / (mm * mm);
            tot = xs.GetTotalXsc() / (mm * mm);
            inel = xs.GetInelasticXsc() / (mm * mm);
          } catch (...) { continue; }
          // %.17g, not the %.9g the older dumps use. Those compare a transcription of a
          // *model* against Geant4's evaluation of it, where a few parts in 1e9 is far
          // below any physics difference. This compares a transcription of a *table* and
          // two linear interpolations against the same table and the same two
          // interpolations: there is nothing between the two answers but double-precision
          // arithmetic in the same order, so they should agree to rounding, and at 9
          // digits the reference's own write precision is the whole disagreement.
          std::fprintf(f, "%d,%.17g,%.17g,%.17g,%.17g,%s\n", Z, e, tot, inel, el, pname[pi]);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- G4IonFluctuations::Dispersion
  //
  // The variance of an ion's energy loss over one step. This is the one deterministic thing in
  // a fluctuation model, so it is the one that can be compared to Geant4 exactly rather than
  // by accumulating samples, and it carries both empirical corrections the model exists for -
  // Yang's charge-state straggling and Geissel's Fermi-gas factor.
  //
  // Dumped for alpha and for proton. The proton is not idle: G4IonFluctuations::Factor takes a
  // different branch below charge 1.5 (no A13 prefactor, no reduced-energy rescaling, a
  // different row of Yang's b table), and a proton is how that branch gets exercised at all -
  // no stock physics list gives a proton this fluctuation model, but G4MuIonisation gives its
  // low-energy model to muons and G4ionIonisation gives it to every ion, so the branch is live
  // and would otherwise be transcribed untested.
  //
  // effChargeSquare is left where InitialiseMe puts it, i.e. equal to chargeSquare, because
  // that is where the transport leaves it: SetParticleAndCharge is called only from
  // G4VEnergyLossProcess under `if(isIon)`, and G4EmTableUtil::CheckIon excludes alpha by
  // name. Dumping it any other way would be testing a configuration Geant4 never runs.
  {
    FILE* f = std::fopen("ion_fluctuation.csv", "w");
    std::fprintf(f, "material,particle,is_gas,vavilov_MeV,energy_MeV,tcut_MeV,tmax_MeV,length_mm,dispersion_MeV2\n");
    const G4ParticleDefinition* parts[2] = {G4Alpha::Alpha(), G4Proton::Proton()};
    const char* pname[2] = {"alpha", "proton"};
    // Step lengths spanning what a transport actually takes: a tenth of a millimetre in the
    // plateau, half a millimetre (the depth-dose slab), and two millimetres where the model's
    // large-fractional-loss branch starts to matter.
    const double lengths[3] = {0.1, 0.5, 2.0};
    for (int pi = 0; pi < 2; ++pi) {
      G4IonFluctuations fluc;
      fluc.InitialiseMe(parts[pi]);
      const double mass = parts[pi]->GetPDGMass();
      // G4IonFluctuations hands everything above this to G4UniversalFluctuation:
      // parameter * charge * particleMass, with parameter = 10 MeV/m_p. Dumped so the port
      // can be checked against the boundary itself and not only against what is either side.
      const double q = parts[pi]->GetPDGCharge() / CLHEP::eplus;
      const double vavilov = (10.0 * MeV / CLHEP::proton_mass_c2) * q * mass;
      for (auto* m : mats) {
        const double cut = ecut.count(m) ? ecut[m] : 0.99e-3 * MeV;
        // 1 keV to 1 GeV, 8 points per decade. The Vavilov handover is at
        // 10 MeV * charge * mass/m_p - 79.45 MeV for an alpha - so this crosses it.
        for (int i = 0; i <= 48; ++i) {
          const double e = 1e-3 * std::pow(10.0, i / 8.0);
          if (e > 1.0e3) { break; }
          G4DynamicParticle dp(parts[pi], G4ThreeVector(0, 0, 1), e * MeV);
          // tmax exactly as G4VEnergyLossProcess computes it for a heavy particle.
          const double tau = e / mass;
          const double me = CLHEP::electron_mass_c2;
          const double ratio = me / mass;
          const double tmax = 2.0 * me * tau * (tau + 2.) /
                              (1. + 2.0 * (tau + 1.) * ratio + ratio * ratio);
          const double tcut = std::min(cut, tmax);
          for (int li = 0; li < 3; ++li) {
            const double len = lengths[li] * mm;
            const double d = fluc.Dispersion(m, &dp, tcut, tmax, len);
            std::fprintf(f, "%s,%s,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                         m->GetName().c_str(), pname[pi],
                         (m->GetState() == kStateGas) ? 1 : 0, vavilov,
                         e, tcut, tmax, lengths[li], d);
          }
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- G4UrbanMscModel::ComputeCrossSectionPerAtom, for every species
  //
  // Urban is not the electron's model. G4hMultipleScattering defaults to it - see
  // G4hMultipleScattering::InitialiseProcess - and G4EmBuilder::ConstructIonEmPhysics hands
  // alpha, He3, deuteron, triton and GenericIon a *fresh* G4hMultipleScattering with no model
  // set, so every one of them scatters by Urban. Only the light hadrons and muons get
  // WentzelVI, and only because ConstructLightHadrons calls SetEmModel on theirs.
  //
  // This port transcribed Urban from the electron path alone - `mass == m_e`, `chargeSquare
  // == 1` - and used WentzelVI for the alpha, which is the wrong model for a species it
  // actually transports. Dumped here per particle and per Z so the general form can be
  // checked rather than assumed.
  //
  // The heavy-particle path is a change of variable and not a different formula: Urban maps a
  // heavy particle onto the electron of the same velocity,
  //
  //     TAU = T/mass;  c = mass*TAU*(TAU+2)/(m_e*(TAU+1));  w = c-2
  //     tau = (w + sqrt(w*w + 4c))/2;  eKineticEnergy = m_e*tau
  //
  // and then runs the electron formula on eKineticEnergy, scaled by chargeSquare. Getting that
  // map wrong gives a plausible cross section with the wrong energy dependence, which is
  // exactly the kind of thing a single-particle test never sees.
  {
    FILE* f = std::fopen("urban_msc.csv", "w");
    std::fprintf(f, "particle,Z,energy_MeV,xs_per_atom_mm2\n");
    G4DataVector ucuts;
    ucuts.push_back(1.0 * keV);
    const G4ParticleDefinition* uparts[] = {
        G4Electron::Electron(), G4Positron::Positron(), G4Proton::Proton(),
        G4AntiProton::AntiProton(), G4Alpha::Alpha(), G4He3::He3(),
        G4MuonMinus::MuonMinus(), G4PionPlus::PionPlus()};
    for (const G4ParticleDefinition* up : uparts) {
      auto* um = new G4UrbanMscModel();
      um->Initialise(up, ucuts);
      // Z from 1 to 92: the two coefficient tables are tabulated at fifteen Z values and
      // interpolated in Z^2 between them, with separate branches below the first and above
      // the last, so the ends matter as much as the middle.
      for (G4int Z = 1; Z <= 92; ++Z) {
        // 1 keV to 10 GeV, 8 points per decade. Tlim is 10 MeV in *electron-equivalent*
        // energy, so for a heavy particle the handover between the tabulated and the
        // high-energy branch sits at a completely different kinetic energy - which is the
        // whole point of dumping more than one mass.
        for (int i = 0; i <= 56; ++i) {
          const double e = 1e-3 * std::pow(10.0, i / 8.0);
          if (e > 1e4) { break; }
          double xs = 0;
          try {
            xs = um->ComputeCrossSectionPerAtom(up, e * MeV, double(Z), 0, 0, 0) / (mm * mm);
          } catch (...) { continue; }
          std::fprintf(f, "%s,%d,%.17g,%.17g\n", up->GetParticleName().c_str(), Z, e, xs);
        }
      }
    }
    std::fclose(f);
  }

  // ---------------- G4RayleighAngularGenerator::SampleDirection
  //
  // The one baked table in this port that had neither an extractor nor a test. Its parameters
  // are compiled into G4RayleighAngularGenerator - they are NOT the re-ff-Z.dat files in
  // G4EMLOW, which are the cross section - so a Geant4 change to them could not be noticed by
  // any means the project had. tools/extract_rayleigh_angular.sh closes the reproducibility
  // half; this closes the validation half.
  //
  // The sampler is a rejection loop, so there is no closed form to diff. What is dumped is the
  // first two moments of cos(theta) over a large sample, which is enough to catch a wrong
  // parameter row, a swapped weight and slope, a mis-transcribed series expansion, or an
  // acceptance test with the wrong power - each of those moves a mean or a width by far more
  // than the standard error of a million draws.
  //
  // Two moments and not one. <cos> alone is insensitive to a symmetric widening: the
  // distribution is forward-peaked, and an error that broadens it while keeping its centre
  // would pass on the mean and fail on the second moment.
  {
    FILE* f = std::fopen("rayleigh_angular.csv", "w");
    std::fprintf(f, "Z,energy_MeV,samples,mean_cos,mean_cos2\n");
    G4RayleighAngularGenerator gen;
    // Z across the fitted range, including the ends: index 0 is a placeholder zero in Geant4's
    // tables and Z=100 is the last row, so both edges are where an off-by-one would land.
    const G4int zs[] = {1, 2, 6, 8, 13, 26, 47, 74, 82, 92, 100};
    // 1 keV to 10 MeV. The low end is where the series expansions below `numlim` are taken and
    // the high end is where they are not, so both branches are exercised.
    const double es[] = {0.001, 0.005, 0.02, 0.1, 0.5, 2.0, 10.0};
    const int kN = 1000000;
    for (G4int Z : zs) {
      for (double e : es) {
        G4DynamicParticle dp(G4Gamma::Gamma(), G4ThreeVector(0, 0, 1), e * MeV);
        // Fixed seed per point so the dump is reproducible run to run.
        CLHEP::HepRandom::setTheSeed(12345 + Z * 131 + G4int(e * 1000));
        double s1 = 0, s2 = 0;
        for (int i = 0; i < kN; ++i) {
          const G4ThreeVector& d = gen.SampleDirection(&dp, 0.0, Z, nullptr);
          const double c = d.z();  // incident direction is +z, so z is cos(theta)
          s1 += c;
          s2 += c * c;
        }
        std::fprintf(f, "%d,%.17g,%d,%.17g,%.17g\n", Z, e, kN, s1 / kN, s2 / kN);
      }
    }
    std::fclose(f);
  }

  std::printf("wrote rayleigh_angular.csv urban_msc.csv ion_fluctuation.csv nucleon_xs.csv icru73qo.csv nuclear_stopping.csv atomic_masses.csv bragg.csv wentzel.csv hadron_radiative.csv muon_models.csv corrections.csv ionisation_params.csv density_correction.csv bethe_bloch.csv brems_rel.csv cuts.csv gamma_xs.csv electron_tables.csv materials.csv annihilation.csv hadron_tables.csv coulomb.csv\n");
  delete rm;
  return 0;
}
