// G4CoulombScattering and G4eCoulombScatteringModel, for src/physics/em/coulomb_scattering.cuh.
//
// Three files, and they answer three different kinds of question.
//
//   coulomb_limits.csv    WHO gets the process, with which model and which limits, read off
//                         the constructed QBBC rather than off G4EmStandardPhysics.cc. The
//                         table in coulomb_scattering.cuh's header is checked against this.
//   coulomb_xs.csv        the cross section per atom, exact, for each species x Z x energy,
//                         plus the angular interval it was integrated over. Deterministic.
//   coulomb_sample.csv    G4WentzelOKandVIxSection::SampleSingleScattering, N = 20,000 per
//                         (species, Z, energy) under a fixed seed: cos(theta) moments and a
//                         histogram, and the recoil energy the model derives from it.
//
// WHY THE SAMPLER IS DUMPED AND NOT THE WHOLE MODEL. `SampleSecondaries` draws the target
// element through G4EmElementSelector and then the ISOTOPE through SelectIsotopeNumber, whose
// per-element natural abundances this port has no table of - `data/natural_isotopes.hh`
// carries the nuclide set only. The isotope decides `factD` in the rejection function and the
// recoil energy, so a whole-model comparison would be comparing two different targets. Fixing
// the (Z, A) and calling the sampler directly compares the angle sampler and nothing else,
// which is the part the port has. The refusal is named in coulomb_scattering.cuh's header and
// in `coulomb_refuse_isotope_selection`.
//
// SEEDS. Each (species, Z, energy) cell sets CLHEP::HepRandom::setTheSeed to a value derived
// from its own indices, so a cell's numbers do not depend on how many cells ran before it and
// re-running one cell reproduces it.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <vector>

#include "G4AntiProton.hh"
#include "G4DataVector.hh"
#include "G4Electron.hh"
#include "G4EmParameters.hh"
#include "G4IonisParamMat.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4Material.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4NistManager.hh"
#include "G4NucleiProperties.hh"
#include "G4ParticleChangeForGamma.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4Positron.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"
#include "G4VEmProcess.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4VMultipleScattering.hh"
#include "G4WentzelOKandVIxSection.hh"
#include "G4eCoulombScatteringModel.hh"
#include "Randomize.hh"

namespace {

struct Sp {
  const G4ParticleDefinition* def;
  const char* name;
};

/// The eight species QBBC gives the process, plus the five it does not - so the "does not" is a
/// dumped fact and not an assumption. G4EmBuilder::ConstructIonEmPhysics registers no
/// G4CoulombScattering, and a row of zeros here is what says so.
std::vector<Sp> species() {
  return {
      {G4Electron::Electron(), "e-"},
      {G4Positron::Positron(), "e+"},
      {G4MuonPlus::MuonPlus(), "mu+"},
      {G4MuonMinus::MuonMinus(), "mu-"},
      {G4PionPlus::PionPlus(), "pi+"},
      {G4PionMinus::PionMinus(), "pi-"},
      {G4KaonPlus::KaonPlus(), "kaon+"},
      {G4KaonMinus::KaonMinus(), "kaon-"},
      {G4Proton::Proton(), "proton"},
      {G4AntiProton::AntiProton(), "anti_proton"},
  };
}

/// The production threshold vector for one secondary index, as a G4DataVector - which is what
/// G4VEmModel::Initialise takes. Index 3 is the proton, and the proton's is the one
/// G4eCoulombScatteringModel gets, because G4CoulombScattering's constructor declares
/// SetSecondaryParticle(G4Proton::Proton()).
G4DataVector cut_vector(std::size_t which) {
  auto* pct = G4ProductionCutsTable::GetProductionCutsTable();
  const auto& v = *pct->GetEnergyCutsVector(which);
  G4DataVector out;
  for (std::size_t i = 0; i < v.size(); ++i) { out.push_back(v[i]); }
  return out;
}

double cut_for(std::size_t which, const G4Material* m) {
  auto* pct = G4ProductionCutsTable::GetProductionCutsTable();
  for (std::size_t i = 0; i < pct->GetTableSize(); ++i) {
    if (pct->GetMaterialCutsCouple(static_cast<G4int>(i))->GetMaterial() == m) {
      return (*pct->GetEnergyCutsVector(which))[i];
    }
  }
  return 0.0;
}

const G4MaterialCutsCouple* couple_for(const G4Material* m) {
  auto* pct = G4ProductionCutsTable::GetProductionCutsTable();
  for (std::size_t i = 0; i < pct->GetTableSize(); ++i) {
    const auto* c = pct->GetMaterialCutsCouple(static_cast<G4int>(i));
    if (c->GetMaterial() == m) { return c; }
  }
  return nullptr;
}

/// The Z values the brief asks for: hydrogen (the exception inside SetupTarget), carbon,
/// oxygen, aluminium, iron and lead - one per decade of screening strength.
const int kZ[6] = {1, 6, 8, 13, 26, 82};

}  // namespace

static void dump_coulomb(const DumpContext& ctx) {
  auto* nist = G4NistManager::Instance();
  auto* param = G4EmParameters::Instance();
  const std::vector<Sp> sp = species();

  // ---------------------------------------------------------------- 1. who gets it
  //
  // Every process on each species' own process manager, so the answer comes from the
  // constructed physics list. `species_processes.csv` already lists the names; what is needed
  // here is the LIMITS, which are per model and are what coulomb_scattering.cuh's header
  // table claims.
  {
    FILE* f = std::fopen("coulomb_limits.csv", "w");
    std::fprintf(f, "particle,process,subtype,build_table,"
                    "model,model_low_MeV,model_high_MeV,activation_low_MeV,"
                    "msc_theta_limit,factor_for_angle_limit,q2max_MeV2,msc_energy_limit_MeV\n");
    const double a = param->FactorForAngleLimit() * CLHEP::hbarc / CLHEP::fermi;
    const double q2max = 0.5 * a * a;
    for (const Sp& s : sp) {
      auto* pm = s.def->GetProcessManager();
      if (nullptr == pm) { continue; }
      G4ProcessVector* pv = pm->GetProcessList();
      bool found = false;
      for (std::size_t i = 0; i < pv->size(); ++i) {
        G4VProcess* p = (*pv)[i];
        if (nullptr == p) { continue; }
        auto* em = dynamic_cast<G4VEmProcess*>(p);
        auto* ms = dynamic_cast<G4VMultipleScattering*>(p);
        if (nullptr == em && nullptr == ms) { continue; }
        const G4String pn = p->GetProcessName();
        if (pn != "CoulombScat" && pn != "msc" && pn != "muMsc") { continue; }
        found = found || (pn == "CoulombScat");
        // One row per model, because the msc process carries two for e+- and the limits are
        // the whole point.
        const G4int nm = (nullptr != em) ? em->NumberOfModels() : ms->NumberOfModels();
        for (G4int k = 0; k < nm; ++k) {
          G4VEmModel* mod = (nullptr != em) ? em->GetModelByIndex(k) : ms->GetModelByIndex(k);
          if (nullptr == mod) { continue; }
          std::fprintf(f, "%s,%s,%d,%d,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                       s.name, pn.c_str(), p->GetProcessSubType(),
                       (nullptr != em) ? 1 : 0,
                       mod->GetName().c_str(), mod->LowEnergyLimit() / MeV,
                       mod->HighEnergyLimit() / MeV, mod->LowEnergyActivationLimit() / MeV,
                       param->MscThetaLimit(), param->FactorForAngleLimit(), q2max,
                       param->MscEnergyLimit() / MeV);
        }
      }
      if (!found) {
        // The species QBBC gives NO G4CoulombScattering. A row rather than an absence, so the
        // port's table can be checked against a statement instead of against silence.
        std::fprintf(f, "%s,none,-1,0,none,0,0,0,%.17g,%.17g,%.17g,%.17g\n", s.name,
                     param->MscThetaLimit(), param->FactorForAngleLimit(), q2max,
                     param->MscEnergyLimit() / MeV);
      }
    }
    // The five species that get msc and no single scattering, named explicitly.
    const char* noss[5] = {"alpha", "He3", "deuteron", "triton", "GenericIon"};
    for (const char* n : noss) {
      auto* d = G4ParticleTable::GetParticleTable()->FindParticle(n);
      if (nullptr == d || nullptr == d->GetProcessManager()) { continue; }
      G4ProcessVector* pv = d->GetProcessManager()->GetProcessList();
      bool has = false;
      for (std::size_t i = 0; i < pv->size(); ++i) {
        if (nullptr != (*pv)[i] && (*pv)[i]->GetProcessName() == "CoulombScat") { has = true; }
      }
      std::fprintf(f, "%s,%s,-1,0,none,0,0,0,%.17g,%.17g,%.17g,%.17g\n", n,
                   has ? "CoulombScat" : "none", param->MscThetaLimit(),
                   param->FactorForAngleLimit(), 0.5 * a * a, param->MscEnergyLimit() / MeV);
    }
    std::fclose(f);
  }

  // ---------------------------------------------------------------- 2. cross section per atom
  //
  // A freshly constructed model rather than the one QBBC registered, which is the pattern the
  // rest of this dumper uses (g4dump.cc constructs its own G4BetheBlochModel, G4ICRU73QOModel
  // and so on): the registered one has a lambda table over it and its state depends on what
  // the last track did. `SetCurrentCouple` is what ComputeCrossSectionPerAtom's
  // DefineMaterial(CurrentCouple()) reads.
  //
  // The model's cosThetaMin comes from SetPolarAngleLimit, which G4CoulombScattering::
  // InitialiseProcess sets to G4EmParameters::MscThetaLimit(). Set here explicitly for the
  // same reason: it is the parameter the whole angular structure turns on.
  {
    G4DataVector pcuts = cut_vector(3);   // proton
    G4DataVector ecuts = cut_vector(1);   // e-
    FILE* f = std::fopen("coulomb_xs.csv", "w");
    std::fprintf(f, "material,particle,Z,A,energy_MeV,ecut_MeV,pcut_MeV,xs_per_atom_mm2,"
                    "xs_nuclear_mm2,xs_electron_mm2,cos_t_min,cos_t_max,cos_tet_max_nuc,"
                    "cos_tet_max_elec,"
                    "inv_a23,target_mass_MeV,proc_min_primary_MeV,model_min_primary_MeV\n");
    for (auto* m : ctx.materials) {
      const G4MaterialCutsCouple* couple = couple_for(m);
      if (nullptr == couple) { continue; }
      const double ecut = cut_for(1, m);
      const double pcut = cut_for(3, m);
      const double inv_a23 = m->GetIonisation()->GetInvA23();
      // The lightest element, for the model's own MinPrimaryEnergy.
      G4int zlight = 300;
      for (std::size_t j = 0; j < m->GetNumberOfElements(); ++j) {
        zlight = std::min(zlight, (*m->GetElementVector())[j]->GetZasInt());
      }
      for (const Sp& s : sp) {
        auto* model = new G4eCoulombScatteringModel();
        model->SetPolarAngleLimit(param->MscThetaLimit());
        model->Initialise(s.def, pcuts);
        model->SetCurrentCouple(couple);
        auto* wokvi = new G4WentzelOKandVIxSection(true);
        wokvi->Initialise(s.def, -1.0);
        const double procmin =
            std::sqrt(0.5 * std::pow(param->FactorForAngleLimit() * CLHEP::hbarc / CLHEP::fermi,
                                     2.0)
                          * inv_a23 / (1.0 - std::cos(param->MscThetaLimit()))
                      + s.def->GetPDGMass() * s.def->GetPDGMass())
            - s.def->GetPDGMass();
        double modelmin = 0.0;
        try { modelmin = model->MinPrimaryEnergy(m, s.def, pcut) / MeV; } catch (...) {}
        for (int iz = 0; iz < 6; ++iz) {
          const int Z = kZ[iz];
          const int A = G4lrint(nist->GetAtomicMassAmu(Z));
          const double tmass = G4NucleiProperties::GetNuclearMass(A, Z);
          // Twelve points per decade over the range the process is active in: from 100 keV
          // (below every threshold, so the zeros are dumped too) to 100 GeV.
          for (int i = 0; i <= 72; ++i) {
            const double e = 1e-1 * std::pow(10.0, i / 12.0);
            double xs = 0;
            try {
              xs = model->ComputeCrossSectionPerAtom(s.def, e * MeV, double(Z), double(A),
                                                     ecut, e * MeV);
            } catch (...) {}
            // The pieces, from the shared engine, at the same point. wokvi's SetupKinematic
            // needs the material for <A^-2/3>; SetupTarget needs the ELECTRON cut, because
            // ComputeMaxElectronScattering is about delta rays and not about recoils.
            double ctnuc = 0, ctelec = 0, xsn = 0, xse = 0, ctmin = 0, ctmax = 0;
            try {
              wokvi->SetupParticle(s.def);
              const double c0 = wokvi->SetupKinematic(e * MeV, m);
              ctmin = wokvi->SetupTarget(Z, ecut);
              ctnuc = c0;
              ctelec = wokvi->GetCosThetaElec();
              ctmax = (1 == Z && s.def == G4Proton::Proton()) ? 0.0 : -1.0;
              if (ctmin > ctmax) {
                xsn = wokvi->ComputeNuclearCrossSection(ctmin, ctmax);
                xse = wokvi->ComputeElectronCrossSection(ctmin, ctmax);
              }
            } catch (...) {}
            std::fprintf(f,
                         // %.17g throughout the deterministic columns. At %.9g every one of
                         // the eight species failed a 1e-12 comparison at 1.6e-8, which is
                         // the print precision and not a disagreement - the CSV was the
                         // limit. docs/RISK.md V37: an oracle is code.
                         "%s,%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                         "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                         m->GetName().c_str(), s.name, Z, A, e, ecut / MeV, pcut / MeV,
                         xs / mm2, xsn / mm2, xse / mm2, ctmin, ctmax, ctnuc, ctelec,
                         inv_a23, tmass / MeV, procmin / MeV, modelmin);
          }
        }
        delete wokvi;
        delete model;
      }
    }
    std::fclose(f);
  }

  // ---------------------------------------------------------------- 3. the angle sampler
  //
  // SampleSingleScattering N times per cell, with (Z, A) and the target mass FIXED so that the
  // only thing being compared is the angle. The recoil energy is then the model's own
  // expression evaluated on each sampled angle, so the second histogram is a function of the
  // first and its agreement is not independent evidence - it is there because it is the
  // quantity a dose sees.
  {
    constexpr int kN = 20000;
    constexpr int kBins = 20;
    G4DataVector pcuts = cut_vector(3);
    FILE* f = std::fopen("coulomb_sample.csv", "w");
    std::fprintf(f, "material,particle,Z,A,energy_MeV,n,n_scattered,mean_cost,mean_cost2,"
                    "mean_one_minus_cost,mean_trec_MeV,max_trec_MeV,elec_ratio,"
                    "cos_t_min,cos_t_max,target_mass_MeV");
    for (int b = 0; b < kBins; ++b) { std::fprintf(f, ",h%d", b); }
    std::fprintf(f, "\n");
    // Water only: the sampler does not see the material except through <A^-2/3>, and that is
    // already a column in coulomb_xs.csv. One material keeps this file at 480 rows.
    const G4Material* m = nullptr;
    for (auto* mm : ctx.materials) {
      if (mm->GetName() == "G4_WATER") { m = mm; }
    }
    if (nullptr != m) {
      const double ecut = cut_for(1, m);
      int si = 0;
      for (const Sp& s : sp) {
        ++si;
        for (int iz = 0; iz < 6; ++iz) {
          const int Z = kZ[iz];
          const int A = G4lrint(nist->GetAtomicMassAmu(Z));
          const double tmass = G4NucleiProperties::GetNuclearMass(A, Z);
          const double mass = s.def->GetPDGMass();
          for (int ie = 0; ie < 4; ++ie) {
            const double e = 200.0 * std::pow(10.0, ie);   // 200 MeV .. 200 GeV
            auto* wokvi = new G4WentzelOKandVIxSection(true);
            wokvi->Initialise(s.def, -1.0);
            wokvi->SetupParticle(s.def);
            wokvi->SetupKinematic(e * MeV, m);
            const double ctmin = wokvi->SetupTarget(Z, ecut);
            const double ctmax = (1 == Z && s.def == G4Proton::Proton()) ? 0.0 : -1.0;
            double ratio = 0.0;
            if (ctmin > ctmax) {
              const double xn = wokvi->ComputeNuclearCrossSection(ctmin, ctmax);
              const double xe = wokvi->ComputeElectronCrossSection(ctmin, ctmax);
              if (xn + xe > 0.0) { ratio = xe / (xn + xe); }
            }
            // SetTargetMass AFTER the cross sections and before the sampler, which is the
            // order G4eCoulombScatteringModel::SampleSecondaries uses - so factD in the
            // rejection function is built from the isotope's NUCLEAR mass and not from the
            // element's mean atomic mass that SetupTarget put there.
            wokvi->SetTargetMass(tmass);
            const double mom2 = wokvi->GetMomentumSquare();
            CLHEP::HepRandom::setTheSeed(770000UL + 1000UL * (unsigned long)si
                                         + 10UL * (unsigned long)iz
                                         + (unsigned long)ie);
            long h[kBins] = {0};
            double s1 = 0, s2 = 0, s3 = 0, st = 0, tmaxr = 0;
            long nsc = 0;
            for (int k = 0; k < kN; ++k) {
              const G4ThreeVector& v = wokvi->SampleSingleScattering(ctmin, ctmax, ratio);
              const double cost = v.z();
              s1 += cost;
              s2 += cost * cost;
              s3 += 1.0 - cost;
              if (cost < 1.0) { ++nsc; }
              const double trec = mom2 * (1.0 - cost) / (tmass + (mass + e) * (1.0 - cost));
              st += trec;
              if (trec > tmaxr) { tmaxr = trec; }
              // Bins uniform in log10(1 - cos), from 1e-10 to 1: the distribution spans ten
              // decades in 1 - cos and a uniform-in-cos histogram would put every entry in
              // one bin.
              const double x = 1.0 - cost;
              int b = kBins - 1;
              if (x > 0.0) {
                b = int((std::log10(x) + 10.0) * 0.1 * kBins);
                if (b < 0) { b = 0; }
                if (b >= kBins) { b = kBins - 1; }
              } else {
                b = 0;
              }
              ++h[b];
            }
            std::fprintf(f, "%s,%s,%d,%d,%.9g,%d,%ld,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
                            "%.17g,%.17g,%.9g",
                         m->GetName().c_str(), s.name, Z, A, e, kN, nsc, s1 / kN, s2 / kN,
                         s3 / kN, st / kN, tmaxr, ratio, ctmin, ctmax, tmass / MeV);
            for (int b = 0; b < kBins; ++b) { std::fprintf(f, ",%ld", h[b]); }
            std::fprintf(f, "\n");
            delete wokvi;
          }
        }
      }
    }
    std::fclose(f);
  }
}

G4GPU_REGISTER_DUMP("coulomb", "coulomb_limits.csv coulomb_xs.csv coulomb_sample.csv",
                    dump_coulomb);
