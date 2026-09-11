// G4CoulombScattering and G4eCoulombScatteringModel, for src/physics/em/coulomb_scattering.cuh.
//
// Four files, and they answer four different kinds of question.
//
//   coulomb_limits.csv    WHO gets the process, with which model and which limits, read off
//                         the constructed QBBC rather than off G4EmStandardPhysics.cc. The
//                         table in coulomb_scattering.cuh's header is checked against this.
//   coulomb_xs.csv        the cross section per atom, exact, for each species x Z x energy,
//                         plus the angular interval it was integrated over. Deterministic, and
//                         at BOTH the electron cut and the proton cut - see below.
//   coulomb_sample.csv    G4WentzelOKandVIxSection::SampleSingleScattering, N = 20,000 per
//                         (species, Z, energy) under a fixed seed: cos(theta) moments and a
//                         histogram, and the recoil energy the model derives from it.
//   coulomb_recoil.csv    the whole of G4eCoulombScatteringModel::SampleSecondaries, one row
//                         per CALL: the angle it drew, the target it drew, and the recoil,
//                         final energy, deposit and ion direction that follow. Deterministic
//                         given the row, which is the point.
//
// WHY THE ANGLE SAMPLER IS DUMPED SEPARATELY FROM THE WHOLE MODEL. `SampleSecondaries` draws
// the target element through G4EmElementSelector and then the ISOTOPE through
// SelectIsotopeNumber, whose per-element natural abundances this port has no table of -
// `data/natural_isotopes.hh` carries the nuclide set only. The isotope decides `factD` in the
// rejection function and the recoil energy, so comparing two whole models would be comparing
// two different targets. So the two halves are dumped by the two things that can be compared:
// coulomb_sample.csv fixes (Z, A) and calls the angle sampler directly, which compares a
// DISTRIBUTION; coulomb_recoil.csv calls the whole model and records the (Z, A) and the angle
// it chose per call, which lets everything downstream of the choice be compared EXACTLY. The
// refusal itself is named in coulomb_scattering.cuh's header and in
// `coulomb_refuse_isotope_selection`.
//
// WHICH CUT. This model's `cutEnergy` is the PROTON production cut, not the electron one:
// G4CoulombScattering's secondary particle is the proton (G4CoulombScattering.cc:68), so
// G4EmModelManager::Initialise takes cuts index 3 (G4EmModelManager.cc:463-471) and both
// FillLambdaVector (:634) and G4VEmProcess::PostStepDoIt (G4VEmProcess.cc:527) hand the model
// that number - which SetupTarget then uses as an ELECTRON production threshold inside
// ComputeMaxElectronScattering. coulomb_xs.csv carries the cross section at BOTH cuts, and the
// answer it gives is that they are equal to the last bit in all 20,182 active rows: the cut
// reaches only `cosTetMaxElec`, and `ComputeElectronCrossSection` returns zero whenever
// cosTetMaxElec >= cosTMin, which over this process's interval it always is. The two columns
// are what MEASURE that rather than assume it, and they would catch the channel opening. The
// sampler and recoil dumps use the proton cut alone, because that is what the transport does.
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
                    "inv_a23,target_mass_MeV,proc_min_primary_MeV,model_min_primary_MeV,"
                    // The same three quantities at the PROTON cut, which is the cut the
                    // transport actually passes: G4CoulombScattering's secondary particle is
                    // the proton, so G4EmModelManager::Initialise picks cuts index 3 and both
                    // FillLambdaVector and PostStepDoIt hand the model that number. The
                    // columns above use the electron cut. The cross sections turn out EQUAL -
                    // see the file header - and `cos_tet_max_elec` does not, so the pair is
                    // the evidence for which of the two the cut can and cannot move.
                    "xs_per_atom_pcut_mm2,xs_electron_pcut_mm2,cos_tet_max_elec_pcut\n");
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
            // And the same three at the proton cut. Only ComputeMaxElectronScattering reads
            // the cut, so cosTetMaxNuc and the nuclear cross section are cut-independent and
            // are not dumped twice.
            double xsp = 0, xsep = 0, ctelecp = 0;
            try {
              xsp = model->ComputeCrossSectionPerAtom(s.def, e * MeV, double(Z), double(A),
                                                      pcut, e * MeV);
              wokvi->SetupParticle(s.def);
              wokvi->SetupKinematic(e * MeV, m);
              const double cminp = wokvi->SetupTarget(Z, pcut);
              ctelecp = wokvi->GetCosThetaElec();
              const double cmaxp = (1 == Z && s.def == G4Proton::Proton()) ? 0.0 : -1.0;
              if (cminp > cmaxp) { xsep = wokvi->ComputeElectronCrossSection(cminp, cmaxp); }
            } catch (...) {}
            std::fprintf(f,
                         // %.17g throughout the deterministic columns. At %.9g every one of
                         // the eight species failed a 1e-12 comparison at 1.6e-8, which is
                         // the print precision and not a disagreement - the CSV was the
                         // limit. docs/RISK.md V37: an oracle is code.
                         "%s,%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                         "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                         m->GetName().c_str(), s.name, Z, A, e, ecut / MeV, pcut / MeV,
                         xs / mm2, xsn / mm2, xse / mm2, ctmin, ctmax, ctnuc, ctelec,
                         inv_a23, tmass / MeV, procmin / MeV, modelmin,
                         xsp / mm2, xsep / mm2, ctelecp);
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
                    "cos_t_min,cos_t_max,target_mass_MeV,cut_MeV");
    for (int b = 0; b < kBins; ++b) { std::fprintf(f, ",h%d", b); }
    std::fprintf(f, "\n");
    // Water only: the sampler does not see the material except through <A^-2/3>, and that is
    // already a column in coulomb_xs.csv. One material keeps this file at 480 rows.
    const G4Material* m = nullptr;
    for (auto* mm : ctx.materials) {
      if (mm->GetName() == "G4_WATER") { m = mm; }
    }
    if (nullptr != m) {
      // The PROTON cut, because that is the one the process hands the model: theCuts is
      // GetEnergyCutsVector(3) for a process whose secondary is the proton
      // (G4EmModelManager.cc:463-471), and G4VEmProcess::PostStepDoIt passes
      // (*theCuts)[currentCoupleIndex] to SampleSecondaries (G4VEmProcess.cc:527). It enters
      // SetupTarget as an ELECTRON production threshold, which is the quirk, and it changes
      // cosTetMaxElec and so the electron/nucleus split this sampler draws from.
      const double cut = cut_for(3, m);
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
            const double ctmin = wokvi->SetupTarget(Z, cut);
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
                            "%.17g,%.17g,%.9g,%.17g",
                         m->GetName().c_str(), s.name, Z, A, e, kN, nsc, s1 / kN, s2 / kN,
                         s3 / kN, st / kN, tmaxr, ratio, ctmin, ctmax, tmass / MeV, cut / MeV);
            for (int b = 0; b < kBins; ++b) { std::fprintf(f, ",%ld", h[b]); }
            std::fprintf(f, "\n");
            delete wokvi;
          }
        }
      }
    }
    std::fclose(f);
  }

  // ---------------------------------------------------------------- 4. the recoil
  //
  // The whole of G4eCoulombScatteringModel::SampleSecondaries, one row per CALL, so the recoil
  // arithmetic can be compared point by point instead of through the statistics of a different
  // random stream. What makes that possible without porting the isotope draw: every quantity
  // the port has to reproduce is a function of the sampled cos(theta) and the target Geant4
  // itself chose, and both are in the row. The port is handed Geant4's angle and its (Z, A) and
  // must produce Geant4's trec, finalT, edep, branch and recoil direction - which is exactly
  // the boundary the refusal draws: selecting the isotope is refused, everything downstream of
  // the selection is not.
  //
  // HOW THE PRIMARY'S FINAL STATE IS READ. SampleSecondaries writes the primary to the model's
  // fParticleChange and returns nothing; G4VEmModel::GetParticleChangeForGamma is protected, so
  // the object cannot be fetched afterwards. But SetParticleChange is public and Initialise
  // only creates one `if(nullptr == fParticleChange)` (G4eCoulombScatteringModel.cc:123), so
  // handing the model one BEFORE Initialise makes it use ours. Its deposits are reset by hand
  // before each call, because ProposeLocalEnergyDeposit and ProposeNonIonizingEnergyDeposit are
  // plain setters and the ion branch never calls the second - a stale value would look like a
  // deposit that this call did not make. InitializeForPostStep would need a G4Track and a
  // G4Step, which this program has no reason to build.
  // THREE PASSES, BECAUSE ONE PASS CANNOT TEST BOTH ARMS OF ONE BRANCH. Only the ion arm
  // reports the target: below threshold the recoil becomes an energy deposit and the (Z, A)
  // Geant4 drew is not observable from outside. A row without (Z, A) cannot test `trec` -
  // the only route to a target mass is to invert Geant4's own trec, and recomputing trec from
  // that would compare a number with itself. The three passes differ ONLY in the cuts vector
  // handed to `Initialise`, which is what `tcut = max(recoilThreshold, (*pCuts)[i])` reads; the
  // `cutEnergy` ARGUMENT stays the real 0.07 MeV throughout, so the angles and cross sections
  // are the transport's in all three.
  //
  //   pcut      the real proton cuts: QBBC's own behaviour, and the reason the other two exist.
  //             It emits an ion on EVERY deflected draw. That is not a quirk of these cells: the
  //             process's angular interval starts at cosTetMaxNuc, so the smallest momentum
  //             transfer it can sample is set by q2Max and <A^-2/3> rather than by the energy,
  //             and the recoil that follows is 0.285 MeV at its smallest here (calcium in bone)
  //             against a 0.07 MeV cut. So in option0 `edep = trec` is unreachable - a Coulomb
  //             scatter always emits its recoil ion - and a test with only this pass would be
  //             leaving that arm transcribed and unmeasured.
  //   zerocut   pCuts zeroed, so tcut is 0 and every deflected draw emits its ion. These rows
  //             carry (Z, A), so the test takes the nuclear mass from the port's own table and
  //             computes trec from Geant4's angle: independent inputs, exact output.
  //   highcut   pCuts set to 1e6 MeV, above any recoil at these energies, so Geant4 takes the
  //             `else` arm on every draw: edep = trec, proposed as non-ionizing AND local.
  //             That arm's rows are what test it, and they test the assignment and the routing
  //             rather than trec, which zerocut has already nailed.
  {
    constexpr int kCalls = 250;
    G4DataVector pcuts = cut_vector(3);
    G4DataVector zerocuts = cut_vector(3);
    for (std::size_t i = 0; i < zerocuts.size(); ++i) { zerocuts[i] = 0.0; }
    G4DataVector highcuts = cut_vector(3);
    for (std::size_t i = 0; i < highcuts.size(); ++i) { highcuts[i] = 1.0e6 * MeV; }
    FILE* f = std::fopen("coulomb_recoil.csv", "w");
    std::fprintf(f, "material,particle,pass,energy_MeV,call,cut_MeV,tcut_MeV,mom2_MeV2,"
                    "cost,dir_x,dir_y,dir_z,final_t_MeV,trec_MeV,edep_MeV,nonion_MeV,n_sec,"
                    "ion_z,ion_a,ion_dir_x,ion_dir_y,ion_dir_z\n");
    // Two materials rather than four: what varies between them here is the element mix Geant4
    // draws from, not the cut, because the cut this model sees is the proton's 0.07 MeV in every
    // material - G4RToEConvForProton::Convert has no material argument.
    const char* kMats[2] = {"G4_WATER", "G4_BONE_COMPACT_ICRU"};
    for (int pass = 0; pass < 3; ++pass) {
      const char* pname = (0 == pass) ? "pcut" : ((1 == pass) ? "zerocut" : "highcut");
      int si = 0;
      for (const Sp& s : sp) {
        ++si;
        // A fresh model, initialised the way G4CoulombScattering::InitialiseProcess initialises
        // the registered one - SetPolarAngleLimit(MscThetaLimit()) is what makes cosThetaMin
        // -1. Not the registered model itself, whose fParticleChange is already the process's.
        auto* pc = new G4ParticleChangeForGamma();
        auto* model = new G4eCoulombScatteringModel();
        model->SetParticleChange(pc);
        model->SetPolarAngleLimit(param->MscThetaLimit());
        model->SetLowEnergyLimit(std::max(param->MinKinEnergy(), model->LowEnergyLimit()));
        model->SetHighEnergyLimit(std::min(param->MaxKinEnergy(), model->HighEnergyLimit()));
        model->Initialise(s.def,
                          (0 == pass) ? pcuts : ((1 == pass) ? zerocuts : highcuts));
        for (int im = 0; im < 2; ++im) {
          const G4Material* m = nullptr;
          for (auto* mm : ctx.materials) {
            if (mm->GetName() == kMats[im]) { m = mm; }
          }
          if (nullptr == m) { continue; }
          const G4MaterialCutsCouple* couple = couple_for(m);
          if (nullptr == couple) { continue; }
          const double cut = cut_for(3, m);
          const double tcut = (0 == pass) ? cut : ((1 == pass) ? 0.0 : 1.0e6);
          model->SetCurrentCouple(couple);
          const double mass = s.def->GetPDGMass();
          // 2 MeV and 20 MeV are here because of what the threshold branch needs. This process
          // samples only the LARGE angles - its interval runs from cosTetMaxNuc out to 180
          // degrees - and cosTetMaxNuc = 1 - 0.5*a2*<A^-2/3>/mom2 moves towards 1 as the energy
          // falls. At 200 MeV the narrowest deflection the sampler can return already gives a
          // 200 MeV projectile a recoil of several hundred keV, well over the 70 keV proton
          // cut, so EVERY scatter emits its ion and `edep = trec` is never taken. Near the
          // process's own threshold the interval is narrow, the recoil is small, and it is.
          const double kE[5] = {2.0, 20.0, 200.0, 2000.0, 20000.0};
          for (int ie = 0; ie < 5; ++ie) {
            const double e = kE[ie];
            // mom2 = tkin*(tkin + 2*mass) (G4WentzelOKandVIxSection::SetupKinematic), a
            // function of the energy alone, dumped so a disagreement can be localised to the
            // kinematics or to the recoil expression rather than to their product.
            const double mom2 = e * MeV * (e * MeV + 2.0 * mass);
            CLHEP::HepRandom::setTheSeed(880000UL + 100000UL * (unsigned long)pass
                                         + 1000UL * (unsigned long)si
                                         + 100UL * (unsigned long)im + (unsigned long)ie);
            for (int k = 0; k < kCalls; ++k) {
              // Along +z, so rotateUz is the identity and the proposed direction's z IS the
              // sampled cos(theta) exactly rather than to within a rotation.
              G4DynamicParticle dp(s.def, G4ThreeVector(0, 0, 1), e * MeV);
              std::vector<G4DynamicParticle*> fvect;
              pc->SetProposedKineticEnergy(e * MeV);
              pc->ProposeMomentumDirection(G4ThreeVector(0, 0, 1));
              pc->ProposeLocalEnergyDeposit(0.0);
              pc->ProposeNonIonizingEnergyDeposit(0.0);
              try {
                model->SampleSecondaries(&fvect, couple, &dp, cut, e * MeV);
              } catch (...) { continue; }
              const G4ThreeVector& nd = pc->GetProposedMomentumDirection();
              int ionz = 0, iona = 0;
              double trec = 0.0, ix = 0, iy = 0, iz3 = 0;
              if (!fvect.empty() && nullptr != fvect[0]) {
                const G4ParticleDefinition* ion = fvect[0]->GetParticleDefinition();
                ionz = ion->GetAtomicNumber();
                iona = ion->GetAtomicMass();
                trec = fvect[0]->GetKineticEnergy() / MeV;
                ix = fvect[0]->GetMomentumDirection().x();
                iy = fvect[0]->GetMomentumDirection().y();
                iz3 = fvect[0]->GetMomentumDirection().z();
              } else {
                // Below threshold the recoil went to the non-ionizing deposit, which is where
                // ProposeNonIonizingEnergyDeposit(edep) put it, so trec is still in the row.
                trec = pc->GetNonIonizingEnergyDeposit() / MeV;
              }
              std::fprintf(f, "%s,%s,%s,%.9g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                              "%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%.17g,%.17g,%.17g\n",
                           m->GetName().c_str(), s.name, pname, e, k, cut / MeV, tcut / MeV,
                           mom2, nd.z(), nd.x(), nd.y(), nd.z(),
                           pc->GetProposedKineticEnergy() / MeV, trec,
                           pc->GetLocalEnergyDeposit() / MeV,
                           pc->GetNonIonizingEnergyDeposit() / MeV,
                           (int)fvect.size(), ionz, iona, ix, iy, iz3);
              for (auto* d : fvect) { delete d; }
            }
          }
        }
        delete model;
        delete pc;
      }
    }
    std::fclose(f);
  }
}

G4GPU_REGISTER_DUMP("coulomb",
                    "coulomb_limits.csv coulomb_xs.csv coulomb_sample.csv coulomb_recoil.csv",
                    dump_coulomb);
