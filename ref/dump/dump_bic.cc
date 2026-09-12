// The binary cascade's oracle: what Geant4 11.1.1 answers for the nucleus model, the nuclear
// fields and the Runge-Kutta propagation that G4BinaryCascade and G4BinaryLightIonReaction are
// built on.
//
// Seven files:
//
//   bic_limits.csv      what the two models report about themselves once QBBC and
//                       G4IonPhysicsXS have configured them: GetMinEnergy, GetMaxEnergy, the
//                       energy-momentum check levels, the model name and the model catalogue
//                       id. These are the numbers docs/HADRONIC_PLAN.md section 2 quotes, read
//                       from the constructors rather than from the table.
//   bic_density.csv     G4NuclearFermiDensity and G4NuclearShellModelDensity: rho0,
//                       GetRelativeDensity, GetDensity and GetDeriv on a radial grid, and
//                       GetRadius over a grid of relative densities, for twelve nuclides that
//                       straddle the A < 17 dispatch.
//   bic_fermi.csv       G4FermiMomentum::GetFermiMomentum over (A, density).
//   bic_nucleus.csv     G4Fancy3DNucleus::Init's scalars per nuclide: both radii, the outer
//                       radius, GetMass, CoulombBarrier, and the A < 17 verdict.
//   bic_nucleons.csv    the REPLAY SET - five complete sampled configurations, nucleon by
//                       nucleon (type, position, four-momentum, binding energy). The port
//                       cannot reproduce Geant4's random stream, so it reproduces Geant4's
//                       nucleus: every downstream comparison below is evaluated on the
//                       configuration in this file, which makes the fields and the propagation
//                       exactly comparable instead of statistically.
//   bic_nucleus_stats.csv  the same Init, 20,000 times per nuclide: radial and momentum
//                       histograms, the proton fraction, the minimum pair distance, the total
//                       three-momentum, and the outer radius. This is the only check the
//                       SAMPLING can have.
//   bic_field.csv       G4RKPropagation::GetField and GetBarrier for proton, neutron, pi+, pi-
//                       and pi0 on a radial grid, on each replay nucleus. Exact.
//   bic_rk.csv          G4RKPropagation::Transport on fixed initial states, step by step:
//                       position, tracking momentum, cascade state and the accumulated
//                       momentum transfer after each of N steps. Exact, and it is the only
//                       thing that checks the adaptive driver, the Richardson extrapolation
//                       and the four surface corrections at once.
//
// **Why the replay set exists.** `G4Fancy3DNucleus::Init` consumes a number of uniform
// deviates that depends on how many placement trials were rejected, and the port's engine is
// Philox where Geant4's is HepJamesRandom. No seed makes the two agree. But everything the
// cascade does AFTER Init is a deterministic function of the configuration, so dumping the
// configuration turns the fields and the propagator into exact comparisons. The same reasoning
// dump_precompound.cc used for its wounded nucleus.
//
// **What cannot be dumped, and what is done instead.** `theBCminP` (45 MeV), `theCutOnP`
// (90/70/50/45 MeV by nucleus mass) and `theCutOnPAbsorb` (0) are private members of
// G4BinaryCascade with no getters and no setters, and no run can be asked what they are: they
// decide which secondaries are captured, and the capture list is not exposed either. So they
// are checked by `tools/extract_bic_constants.pl`, which reads them out of the Geant4 SOURCE
// and fails if the source ever says something else - the form docs/RISK.md V41 recommends.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "G4Alpha.hh"
#include "G4AngularDistribution.hh"
#include "G4AngularDistributionNP.hh"
#include "G4AngularDistributionPP.hh"
#include "G4BinaryCascade.hh"
#include "G4BinaryLightIonReaction.hh"
#include "G4CollisionManager.hh"
#include "G4CollisionNN.hh"
#include "G4CollisionNNElastic.hh"
#include "G4CollisionnpElastic.hh"
#include "G4Deuteron.hh"
#include "G4DynamicParticle.hh"
#include "G4ExcitationHandler.hh"
#include "G4Fancy3DNucleus.hh"
#include "G4FermiMomentum.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadSecondary.hh"
#include "G4HadronicInteractionRegistry.hh"
#include "G4IonTable.hh"
#include "G4KineticTrack.hh"
#include "G4KineticTrackVector.hh"
#include "G4Neutron.hh"
#include "G4NuclearFermiDensity.hh"
#include "G4NuclearShellModelDensity.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleon.hh"
#include "G4Nucleus.hh"
#include "G4PhysicsModelCatalog.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4PionZero.hh"
#include "G4PreCompoundModel.hh"
#include "G4Proton.hh"
#include "G4RKPropagation.hh"
#include "G4Scatterer.hh"
#include "G4SystemOfUnits.hh"
#include "G4VNuclearDensity.hh"
#include "G4XNNElastic.hh"
#include "G4XNNElasticLowE.hh"
#include "G4XNNTotal.hh"
#include "G4XNNTotalLowE.hh"
#include "G4XPDGElastic.hh"
#include "G4XPDGTotal.hh"
#include "G4XnpElastic.hh"
#include "G4XnpElasticLowE.hh"
#include "G4XnpTotal.hh"
#include "G4XnpTotalLowE.hh"
#include "Randomize.hh"

namespace {

struct Nuclide { int a, z; const char* name; };

/// Twelve nuclides. Li6 and C12 are below the A < 17 dispatch (shell model), O16 is the last
/// one below it, and F19 is the first one above (Fermi) - so the branch is crossed inside the
/// grid rather than assumed. C12 is there twice over, because it is also the only A for which
/// `ChoosePositions` takes the alpha-cluster branch and `nucleondistance` is 0.9 fm.
const Nuclide kNuclides[] = {
  {2, 1, "H2"},     {4, 2, "He4"},   {6, 3, "Li6"},   {12, 6, "C12"},
  {16, 8, "O16"},   {19, 9, "F19"},  {27, 13, "Al27"},{40, 20, "Ca40"},
  {56, 26, "Fe56"}, {107, 47, "Ag107"}, {197, 79, "Au197"}, {208, 82, "Pb208"},
};

/// The five nuclei the replay set, the field dump and the RK dump all use. Water is in QBBC's
/// B1 as a material; H2O's two elements are H and O, and O16 is here for it.
const Nuclide kReplay[] = {
  {12, 6, "C12"}, {16, 8, "O16"}, {27, 13, "Al27"}, {56, 26, "Fe56"}, {208, 82, "Pb208"},
};

/// Radial grid, in fermi. 0 is included because both densities are finite there and the Fermi
/// one's `GetDeriv` divides by rho0; 0.15 is half a nucleon-field table step, which is where
/// the linear interpolation is furthest from the formula.
const double kRadiiFermi[] = {0.0,  0.15, 0.3,  0.45, 0.6,  0.9,  1.2,  1.8,  2.4,  3.0,
                              3.6,  4.2,  4.8,  5.4,  6.0,  6.9,  7.8,  8.7,  9.6,  10.5,
                              11.4, 12.3, 13.2, 14.4, 15.6, 16.8, 18.0, 20.0, 24.0, 30.0};

/// The relative densities `GetRadius` is asked for. 0.5 is the default `GetNuclearRadius()`
/// uses and 0.001 is what `ChoosePositions` uses; 0 and 1 are the two ends of the guard, and
/// 1.5 is outside it, where both classes return DBL_MAX.
const double kRelDens[] = {1.5, 1.0, 0.999, 0.5, 0.1, 0.01, 0.001, 1e-6, 0.0, -0.1};

void write_limits() {
  FILE* f = std::fopen("bic_limits.csv", "w");
  std::fprintf(f, "model,name,emin_MeV,emax_MeV,ep_relative,ep_absolute_MeV,model_id\n");

  // Built exactly as G4HadronInelasticQBBC::ConstructProcess and G4IonPhysics::ConstructProcess
  // build them, including the SetMaxEnergy calls, so the limits are the physics list's and not
  // the constructors' defaults.
  auto* handler = new G4ExcitationHandler();
  auto* preco = new G4PreCompoundModel(handler);

  // `GetEnergyMomentumCheckLevels()` returns what SetEnergyMomentumCheckLevels stored - which
  // for G4BinaryCascade is (1%, 1 MeV), set in its constructor, and for
  // G4BinaryLightIonReaction is G4HadronicInteraction's default. `GetFatalEnergyCheckLevels()`
  // is a different pair and is dumped too, because G4HadronicProcess reads one of each.
  auto* bic = new G4BinaryCascade(preco);
  bic->SetMaxEnergy(1.5 * CLHEP::GeV);            // QBBC's emaxBic
  std::pair<G4double, G4double> ep = bic->GetEnergyMomentumCheckLevels();
  std::pair<G4double, G4double> fat = bic->GetFatalEnergyCheckLevels();
  std::fprintf(f, "G4BinaryCascade,%s,%.17g,%.17g,%.17g,%.17g,%d\n",
               bic->GetModelName().c_str(), bic->GetMinEnergy() / MeV,
               bic->GetMaxEnergy() / MeV, ep.first, ep.second / MeV,
               G4PhysicsModelCatalog::GetModelID("model_G4BinaryCascade"));
  std::fprintf(f, "G4BinaryCascade_fatal,%s,%.17g,%.17g,%.17g,%.17g,0\n",
               bic->GetModelName().c_str(), bic->GetMinEnergy() / MeV,
               bic->GetMaxEnergy() / MeV, fat.first, fat.second / MeV);

  auto* blir = new G4BinaryLightIonReaction(preco);
  blir->SetMinEnergy(0.0);
  blir->SetMaxEnergy(6.0 * CLHEP::GeV);           // GetMaxEnergyTransitionFTF_Cascade
  ep = blir->GetEnergyMomentumCheckLevels();
  fat = blir->GetFatalEnergyCheckLevels();
  std::fprintf(f, "G4BinaryLightIonReaction,%s,%.17g,%.17g,%.17g,%.17g,%d\n",
               blir->GetModelName().c_str(), blir->GetMinEnergy() / MeV,
               blir->GetMaxEnergy() / MeV, ep.first, ep.second / MeV,
               G4PhysicsModelCatalog::GetModelID("model_G4BinaryLightIonReaction"));
  std::fprintf(f, "G4BinaryLightIonReaction_fatal,%s,%.17g,%.17g,%.17g,%.17g,0\n",
               blir->GetModelName().c_str(), blir->GetMinEnergy() / MeV,
               blir->GetMaxEnergy() / MeV, fat.first, fat.second / MeV);

  // The constructor's own limits, before any physics list touches them, so a release that
  // changes them is visible: G4BinaryCascade sets 0 to 10.1 GeV and G4BinaryLightIonReaction
  // inherits G4HadronicInteraction's defaults.
  auto* bic_raw = new G4BinaryCascade(preco);
  std::fprintf(f, "G4BinaryCascade_ctor,%s,%.17g,%.17g,0,0,0\n",
               bic_raw->GetModelName().c_str(), bic_raw->GetMinEnergy() / MeV,
               bic_raw->GetMaxEnergy() / MeV);
  auto* blir_raw = new G4BinaryLightIonReaction(preco);
  std::fprintf(f, "G4BinaryLightIonReaction_ctor,%s,%.17g,%.17g,0,0,0\n",
               blir_raw->GetModelName().c_str(), blir_raw->GetMinEnergy() / MeV,
               blir_raw->GetMaxEnergy() / MeV);
  std::fclose(f);
}

void write_density() {
  FILE* f = std::fopen("bic_density.csv", "w");
  std::fprintf(f, "name,a,z,kind,rho0,r_fermi,rel_density,density,deriv\n");
  FILE* g = std::fopen("bic_density_radius.csv", "w");
  std::fprintf(g, "name,a,z,kind,max_rel_density,radius_mm\n");

  for (const Nuclide& n : kNuclides) {
    // `G4Fancy3DNucleus::Init`'s dispatch, reproduced here so that both branches are dumped
    // for the nuclide they are actually used for.
    const bool shell = (n.a < 17);
    G4VNuclearDensity* d = shell ? static_cast<G4VNuclearDensity*>(
                                       new G4NuclearShellModelDensity(n.a, n.z))
                                 : static_cast<G4VNuclearDensity*>(
                                       new G4NuclearFermiDensity(n.a, n.z));
    // rho0 is protected. It is recovered exactly as the ratio of the two public accessors at a
    // point where the relative density is not zero - the origin for the shell model, and for
    // the Fermi shape any point inside.
    const G4ThreeVector origin(0., 0., 0.);
    const double rho0 = d->GetDensity(origin) / d->GetRelativeDensity(origin);

    for (const double rf : kRadiiFermi) {
      const G4ThreeVector pos(0., 0., rf * fermi);
      std::fprintf(f, "%s,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n", n.name, n.a, n.z,
                   shell ? 0 : 1, rho0, rf, d->GetRelativeDensity(pos), d->GetDensity(pos),
                   d->GetDeriv(pos));
    }
    for (const double mrd : kRelDens) {
      std::fprintf(g, "%s,%d,%d,%d,%.17g,%.17g\n", n.name, n.a, n.z, shell ? 0 : 1, mrd,
                   d->GetRadius(mrd));
    }
    delete d;
  }
  std::fclose(f);
  std::fclose(g);
}

void write_fermi() {
  FILE* f = std::fopen("bic_fermi.csv", "w");
  std::fprintf(f, "a,z,density,fermi_momentum_MeV\n");
  // Densities spanning what a nucleus reaches: the centre of lead is about 1.7e35 mm^-3 for
  // `A*GetDensity`, i.e. 8e32 for the normalised `GetDensity` of a heavy nucleus. The grid is
  // decades, plus zero, because `GetFermiMomentum(0)` is A13(0) = 0 and not a division.
  const double kDens[] = {0.0,   1e28, 1e29, 1e30, 1e31, 1e32, 1e33,
                          1e34,  1e35, 1e36, 2.5e32, 7.5e32, 1.3e33};
  const int kA[] = {2, 4, 12, 16, 27, 56, 107, 197, 208, 238};
  for (const int a : kA) {
    G4FermiMomentum fm;
    fm.Init(a, a / 2);
    for (const double dd : kDens) {
      std::fprintf(f, "%d,%d,%.17g,%.17g\n", a, a / 2, dd, fm.GetFermiMomentum(dd) / MeV);
    }
  }
  std::fclose(f);
}

/// One `Init` per nuclide, under a fixed seed, with the scalars and the whole configuration.
void write_nucleus_and_nucleons() {
  FILE* f = std::fopen("bic_nucleus.csv", "w");
  std::fprintf(f, "name,a,z,shell,radius_half_mm,radius_0001_mm,outer_radius_mm,mass_MeV,"
                  "coulomb_barrier_MeV,binding_energy_MeV\n");
  for (const Nuclide& n : kNuclides) {
    CLHEP::HepRandom::setTheSeed(20260911L + n.a);
    G4Fancy3DNucleus nuc;
    nuc.Init(n.a, n.z);
    std::fprintf(f, "%s,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", n.name, n.a, n.z,
                 (n.a < 17) ? 1 : 0, nuc.GetNuclearRadius() / mm,
                 nuc.GetNuclearRadius(0.001) / mm, nuc.GetOuterRadius() / mm,
                 nuc.GetMass() / MeV, nuc.CoulombBarrier() / MeV,
                 G4NucleiProperties::GetBindingEnergy(n.a, n.z) / MeV);
  }
  std::fclose(f);

  // The replay set. One nucleus per nuclide, seeded so the configuration is reproducible if
  // the oracle is regenerated, and dumped in `GetNextNucleon` order - which is the order every
  // consumer sees and the order `SortNucleonsIncZ` would permute.
  FILE* g = std::fopen("bic_nucleons.csv", "w");
  std::fprintf(g, "name,a,z,index,pdg,pos_x,pos_y,pos_z,p_x,p_y,p_z,p_e,binding_MeV,"
                  "outer_radius_mm,mass_MeV\n");
  for (const Nuclide& n : kReplay) {
    CLHEP::HepRandom::setTheSeed(777000L + n.a);
    G4Fancy3DNucleus nuc;
    nuc.Init(n.a, n.z);
    const double router = nuc.GetOuterRadius();
    const double mass = nuc.GetMass();
    nuc.StartLoop();
    G4Nucleon* nucleon = nullptr;
    int i = 0;
    while ((nucleon = nuc.GetNextNucleon()) != nullptr) {
      const G4ThreeVector p = nucleon->GetPosition();
      const G4LorentzVector m = nucleon->GetMomentum();
      std::fprintf(g, "%s,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                      "%.17g,%.17g\n",
                   n.name, n.a, n.z, i, nucleon->GetDefinition()->GetPDGEncoding(), p.x(),
                   p.y(), p.z(), m.x(), m.y(), m.z(), m.e(),
                   nucleon->GetBindingEnergy() / MeV, router / mm, mass / MeV);
      ++i;
    }
  }
  std::fclose(g);
}

/// The fields and the propagator, on each replay nucleus. The nucleus is rebuilt from the same
/// seed as `bic_nucleons.csv` so the two files describe the same configuration.
void write_field_and_rk() {
  FILE* f = std::fopen("bic_field.csv", "w");
  std::fprintf(f, "name,a,z,pdg,r_fermi,field_MeV,barrier_MeV\n");
  FILE* g = std::fopen("bic_rk.csv", "w");
  std::fprintf(g, "name,a,z,case,pdg,step,state,pos_x,pos_y,pos_z,p_x,p_y,p_z,p_e,"
                  "tr_x,tr_y,tr_z,tr_e,mt_x,mt_y,mt_z\n");

  const int kPdg[] = {2212, 2112, 211, -211, 111};

  for (const Nuclide& n : kReplay) {
    CLHEP::HepRandom::setTheSeed(777000L + n.a);
    auto* nuc = new G4Fancy3DNucleus;
    nuc->Init(n.a, n.z);
    G4RKPropagation prop;
    prop.Init(nuc);

    for (const int pdg : kPdg) {
      for (const double rf : kRadiiFermi) {
        const G4ThreeVector pos(0., 0., rf * fermi);
        std::fprintf(f, "%s,%d,%d,%d,%.17g,%.17g,%.17g\n", n.name, n.a, n.z, pdg, rf,
                     prop.GetField(pdg, pos) / MeV, prop.GetBarrier(pdg) / MeV);
      }
    }

    // The RK cases. Each is (species, kinetic energy, impact parameter, start z), chosen so
    // that the six outcomes of Transport are all reached: a nucleon that traverses, one that
    // grazes, one slow enough to be reflected or captured, and a pion of each charge.
    struct RkCase {
      int pdg; double mass; double kin_MeV; double b_fermi; double z0_fermi; double dt_ns;
      int n_steps;
    };
    const double mp = G4Proton::Proton()->GetPDGMass();
    const double mn = G4Neutron::Neutron()->GetPDGMass();
    const double mpi = G4PionPlus::PionPlus()->GetPDGMass();
    const double mpi0 = G4PionZero::PionZero()->GetPDGMass();
    const RkCase kCases[] = {
      {2212, mp,  200.0,  1.0, -20.0, 1.0e-5, 12},
      {2212, mp,   50.0,  3.0, -20.0, 1.0e-5, 12},
      {2212, mp,   20.0,  0.5, -20.0, 2.0e-5, 12},
      {2112, mn,  200.0,  1.0, -20.0, 1.0e-5, 12},
      {2112, mn,   30.0,  4.0, -20.0, 2.0e-5, 12},
      {2112, mn,    5.0,  0.0, -20.0, 5.0e-5, 12},
      { 211, mpi, 150.0,  2.0, -20.0, 1.0e-5, 12},
      {-211, mpi, 150.0,  2.0, -20.0, 1.0e-5, 12},
      { 111, mpi0,150.0,  2.0, -20.0, 1.0e-5, 12},
      // Started INSIDE, which is what a cascade secondary is: no entry barrier to pay.
      {2212, mp,  120.0,  0.0,   0.0, 1.0e-5, 12},
      {2112, mn,   80.0,  0.0,   0.0, 1.0e-5, 12},
    };

    int icase = 0;
    for (const RkCase& c : kCases) {
      const G4ParticleDefinition* def = nullptr;
      if (c.pdg == 2212) { def = G4Proton::Proton(); }
      else if (c.pdg == 2112) { def = G4Neutron::Neutron(); }
      else if (c.pdg == 211) { def = G4PionPlus::PionPlus(); }
      else if (c.pdg == -211) { def = G4PionMinus::PionMinus(); }
      else { def = G4PionZero::PionZero(); }

      const double pmag = std::sqrt(c.kin_MeV * (c.kin_MeV + 2.0 * c.mass));
      const G4ThreeVector r0(c.b_fermi * fermi, 0., c.z0_fermi * fermi);
      const G4LorentzVector p0(0., 0., pmag, c.kin_MeV + c.mass);

      auto* kt = new G4KineticTrack(def, 0., r0, p0);
      // `outside` for a track that starts beyond the nucleus, `inside` for one that starts in
      // it: BIC sets exactly these two and the Transport control flow branches on them.
      kt->SetState(c.z0_fermi < -1.0 ? G4KineticTrack::outside : G4KineticTrack::inside);

      G4KineticTrackVector active;
      active.push_back(kt);
      const G4KineticTrackVector spectators;

      for (int s = 0; s < c.n_steps; ++s) {
        prop.Transport(active, spectators, c.dt_ns * ns);
        const G4ThreeVector mt = prop.GetMomentumTransfer();
        const G4LorentzVector tot = kt->Get4Momentum();
        const G4LorentzVector tr = kt->GetTrackingMomentum();
        std::fprintf(g, "%s,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     n.name, n.a, n.z, icase, c.pdg, s, static_cast<int>(kt->GetState()),
                     kt->GetPosition().x(), kt->GetPosition().y(), kt->GetPosition().z(),
                     tot.x(), tot.y(), tot.z(), tot.e(), tr.x(), tr.y(), tr.z(), tr.e(),
                     mt.x(), mt.y(), mt.z());
      }
      delete kt;
      ++icase;
    }
    delete nuc;
  }
  std::fclose(f);
  std::fclose(g);
}

/// The sampling, 20,000 configurations per nuclide. Histograms are counts, so the port compares
/// them with the binomial sigma tests/test_deex_breakup.cu defines.
void write_nucleus_stats() {
  FILE* f = std::fopen("bic_nucleus_stats.csv", "w");
  std::fprintf(f, "name,a,z,n_events,quantity,bin,low,high,count\n");
  FILE* g = std::fopen("bic_nucleus_moments.csv", "w");
  std::fprintf(g, "name,a,z,n_events,quantity,mean,variance\n");

  const long kN = 20000;
  const int kNRad = 40;
  const double kRadMax = 20.0;       // fermi
  const int kNMom = 40;
  const double kMomMax = 400.0;      // MeV

  // Light, medium and heavy, as docs/HADRONIC_PLAN.md's P9 entry asks for, plus C12 because
  // its placement branch is different from every other nuclide's.
  const Nuclide kStat[] = {{12, 6, "C12"}, {16, 8, "O16"}, {27, 13, "Al27"},
                           {56, 26, "Fe56"}, {208, 82, "Pb208"}};

  for (const Nuclide& n : kStat) {
    CLHEP::HepRandom::setTheSeed(31337L + n.a);
    std::vector<long long> hr(kNRad, 0), hp(kNMom, 0);
    double sum_router = 0, sum_router2 = 0;
    double sum_psum = 0, sum_psum2 = 0;
    double sum_dmin = 0, sum_dmin2 = 0;
    double sum_r = 0, sum_r2 = 0, sum_p = 0, sum_p2 = 0;
    long long n_nucleons = 0;

    G4Fancy3DNucleus nuc;
    for (long ev = 0; ev < kN; ++ev) {
      nuc.Init(n.a, n.z);
      const std::vector<G4Nucleon>& list = nuc.GetNucleons();
      G4ThreeVector psum(0., 0., 0.);
      double dmin2 = 1e30;
      for (std::size_t i = 0; i < list.size(); ++i) {
        const double r = list[i].GetPosition().mag() / fermi;
        const double p = list[i].GetMomentum().vect().mag() / MeV;
        psum += list[i].GetMomentum().vect();
        sum_r += r; sum_r2 += r * r;
        sum_p += p; sum_p2 += p * p;
        ++n_nucleons;
        int ib = static_cast<int>(r / (kRadMax / kNRad));
        if (ib >= kNRad) { ib = kNRad - 1; }
        ++hr[ib];
        int jb = static_cast<int>(p / (kMomMax / kNMom));
        if (jb >= kNMom) { jb = kNMom - 1; }
        ++hp[jb];
        for (std::size_t j = i + 1; j < list.size(); ++j) {
          const double d2 = (list[i].GetPosition() - list[j].GetPosition()).mag2();
          if (d2 < dmin2) { dmin2 = d2; }
        }
      }
      const double router = nuc.GetOuterRadius() / fermi;
      sum_router += router; sum_router2 += router * router;
      const double pm = psum.mag() / MeV;
      sum_psum += pm; sum_psum2 += pm * pm;
      const double dmin = (list.size() > 1) ? std::sqrt(dmin2) / fermi : 0.0;
      sum_dmin += dmin; sum_dmin2 += dmin * dmin;
    }

    for (int i = 0; i < kNRad; ++i) {
      std::fprintf(f, "%s,%d,%d,%ld,radius,%d,%.17g,%.17g,%lld\n", n.name, n.a, n.z, kN, i,
                   i * (kRadMax / kNRad), (i + 1) * (kRadMax / kNRad), hr[i]);
    }
    for (int i = 0; i < kNMom; ++i) {
      std::fprintf(f, "%s,%d,%d,%ld,momentum,%d,%.17g,%.17g,%lld\n", n.name, n.a, n.z, kN, i,
                   i * (kMomMax / kNMom), (i + 1) * (kMomMax / kNMom), hp[i]);
    }
    auto moments = [&](const char* what, double s1, double s2, double nn) {
      const double m = s1 / nn;
      const double v = s2 / nn - m * m;
      std::fprintf(g, "%s,%d,%d,%ld,%s,%.17g,%.17g\n", n.name, n.a, n.z, kN, what, m,
                   (v > 0.0) ? v : 0.0);
    };
    moments("radius", sum_r, sum_r2, static_cast<double>(n_nucleons));
    moments("momentum", sum_p, sum_p2, static_cast<double>(n_nucleons));
    moments("outer_radius", sum_router, sum_router2, static_cast<double>(kN));
    moments("total_momentum", sum_psum, sum_psum2, static_cast<double>(kN));
    moments("min_pair_distance", sum_dmin, sum_dmin2, static_cast<double>(kN));
  }
  std::fclose(f);
  std::fclose(g);
}

/// bic_blir.csv and bic_blir_status.csv - G4BinaryLightIonReaction::ApplyYourself below its
/// 50 MeV/nucleon fusion threshold, which is the arm this package has ported.
///
/// **Why this file exists.** Everything above is the machinery UNDER the two models; this is the
/// only place a whole model is called the way the framework calls it. `FuseNucleiAndPrompound`
/// is private, `SetLighterAsProjectile` is private, and the compound fragment they build is
/// never handed back - so the fusion gate, the swap, the exciton configuration (pA particles of
/// which pZ charged, zero holes) and the boost back out of the Breit frame have exactly one
/// observable: the secondary list. It is compared statistically, the way
/// `dump_precompound.cc`'s `preco_apply.csv` is.
///
/// **What each column can catch.** `status` is the fusion gate's verdict and is DETERMINISTIC
/// per case - `isAlive` means `m2Compound < mFused^2` and the primary came back untouched - so
/// a port that got `GetIonMass` or the kinematics wrong changes an integer, not a distribution.
/// `sum_a` and `sum_z` are the compound's own (pZ+tZ, pA+tA) and must be exact in every event.
/// The per-species counts and energy moments are the de-excitation; `mean_pz` and `mean_e` are
/// the energy-momentum balance the swap would break.
///
/// N is 5,000 and not the 20,000 of the sampling files above: one event here is a complete
/// pre-equilibrium and evaporation cascade on a compound of up to A = 220, where one event
/// there is a nucleus. 5,000 keeps a 5-sigma band at about 1.5% on the common species and the
/// whole file inside a minute on both sides.
void write_blir() {
  auto* handler = new G4ExcitationHandler();
  auto* preco = new G4PreCompoundModel(handler);
  auto* blir = new G4BinaryLightIonReaction(preco);
  blir->SetMinEnergy(0.0);
  blir->SetMaxEnergy(6.0 * CLHEP::GeV);

  G4IonTable* ions = G4IonTable::GetIonTable();

  /// (projectile Z, A; kinetic energy PER NUCLEON; target Z, A). Every case is below the
  /// 50 MeV/nucleon gate AFTER the swap, which for the H1 targets is not the same number: a
  /// C12 at 45 MeV/nucleon on H1 swaps, and the quantity the gate tests becomes
  /// `(gamma-1)*m_proton`, 45.3 MeV. The two alpha-on-H1 rows are there for the OTHER branch:
  /// Li5 is unbound, so `mFused` can exceed the invariant mass of alpha+p and the model returns
  /// the primary alive. Which of them does is the oracle's to say, not this comment's.
  struct BCase { int pz, pa; double ekin_per_a; int tz, ta; const char* name; };
  const BCase kCases[] = {
    {1, 2,  10.0,  6,  12, "d10_C12"},     {1, 2,  45.0,  6,  12, "d45_C12"},
    {1, 2,  10.0, 13,  27, "d10_Al27"},    {1, 2,  45.0, 13,  27, "d45_Al27"},
    {1, 2,  10.0, 82, 208, "d10_Pb208"},   {1, 2,  45.0, 82, 208, "d45_Pb208"},
    {2, 4,  10.0,  6,  12, "a10_C12"},     {2, 4,  25.0,  6,  12, "a25_C12"},
    {2, 4,  45.0,  6,  12, "a45_C12"},     {2, 4,  10.0,  8,  16, "a10_O16"},
    {2, 4,  25.0, 26,  56, "a25_Fe56"},    {2, 4,  45.0, 26,  56, "a45_Fe56"},
    {2, 4,  10.0, 82, 208, "a10_Pb208"},   {2, 4,  45.0, 82, 208, "a45_Pb208"},
    {6, 12, 10.0,  6,  12, "C12_10_C12"},  {6, 12, 25.0, 82, 208, "C12_25_Pb208"},
    {6, 12, 10.0,  1,   1, "C12_10_H1"},   {6, 12, 45.0,  1,   1, "C12_45_H1"},
    {2, 4,   1.0,  1,   1, "a1_H1"},       {2, 4,  10.0,  1,   1, "a10_H1"},
  };
  const int kN = 5000;

  FILE* f = std::fopen("bic_blir.csv", "w");
  std::fprintf(f, "case,pz,pa,ekin_per_a_MeV,tz,ta,N,pdg,count,mean_ekin_MeV,mean_ekin2_MeV2,"
                  "mean_mult2\n");
  FILE* g = std::fopen("bic_blir_status.csv", "w");
  std::fprintf(g, "case,pz,pa,ekin_per_a_MeV,tz,ta,N,status,n_secondaries,sum_z,sum_a,"
                  "mean_e_MeV,mean_pz_MeV,mean_mult\n");

  for (const BCase& c : kCases) {
    const G4ParticleDefinition* part = ions->GetIon(c.pz, c.pa, 0.0);
    if (part == nullptr) {
      std::fprintf(g, "%s,%d,%d,%.17g,%d,%d,%d,NO_ION,0,0,0,0,0,0\n", c.name, c.pz, c.pa,
                   c.ekin_per_a, c.tz, c.ta, kN);
      continue;
    }
    CLHEP::HepRandom::setTheSeed(555000L + c.pa * 1000 + c.ta + G4int(c.ekin_per_a));

    std::map<int, long long> count;
    std::map<int, double> sum_e, sum_e2, sum_k2;
    std::map<int, int> per_event;
    long long n_alive = 0, n_kill = 0, n_sec = 0;
    // The compound's (Z, A) as the secondaries add up to it, and the totals for the balance.
    long long sum_z = -1, sum_a = -1;
    bool za_varies = false;
    double sum_tot_e = 0.0, sum_tot_pz = 0.0;

    for (int n = 0; n < kN; ++n) {
      G4DynamicParticle dp(part, G4ThreeVector(0, 0, 1),
                           c.ekin_per_a * c.pa * MeV);
      G4HadProjectile proj(dp);
      G4Nucleus nucleus(c.ta, c.tz);
      G4HadFinalState* r = blir->ApplyYourself(proj, nucleus);
      if (r == nullptr) { continue; }
      if (r->GetStatusChange() == isAlive) { ++n_alive; continue; }
      ++n_kill;
      per_event.clear();
      long long ez = 0, ea = 0;
      G4LorentzVector tot(0., 0., 0., 0.);
      const std::size_t ns = r->GetNumberOfSecondaries();
      n_sec += static_cast<long long>(ns);
      for (std::size_t i = 0; i < ns; ++i) {
        const G4HadSecondary* s = r->GetSecondary(i);
        const G4DynamicParticle* p = s->GetParticle();
        const int pdg = p->GetDefinition()->GetPDGEncoding();
        const double ekin = p->GetKineticEnergy() / MeV;
        ++count[pdg];
        sum_e[pdg] += ekin;
        sum_e2[pdg] += ekin * ekin;
        ++per_event[pdg];
        tot += p->Get4Momentum();
        if (pdg > 1000000000) {
          ea += (pdg / 10) % 1000;
          ez += (pdg / 10000) % 1000;
        } else if (pdg == 2112) { ea += 1; }
        else if (pdg == 2212) { ea += 1; ez += 1; }
      }
      for (const auto& kv : per_event) {
        sum_k2[kv.first] += double(kv.second) * kv.second;
      }
      if (sum_z < 0) { sum_z = ez; sum_a = ea; }
      else if (ez != sum_z || ea != sum_a) { za_varies = true; }
      sum_tot_e += tot.e() / MeV;
      sum_tot_pz += tot.z() / MeV;
    }

    const double nk = (n_kill > 0) ? double(n_kill) : 1.0;
    std::fprintf(g, "%s,%d,%d,%.17g,%d,%d,%d,%s,%lld,%lld,%lld,%.17g,%.17g,%.17g\n", c.name,
                 c.pz, c.pa, c.ekin_per_a, c.tz, c.ta, kN,
                 (n_alive == kN) ? "isAlive" : ((n_kill == kN) ? "stopAndKill" : "MIXED"),
                 n_sec, za_varies ? -1 : sum_z, za_varies ? -1 : sum_a, sum_tot_e / nk,
                 sum_tot_pz / nk, double(n_sec) / nk);
    for (const auto& kv : count) {
      std::fprintf(f, "%s,%d,%d,%.17g,%d,%d,%d,%d,%lld,%.17g,%.17g,%.17g\n", c.name, c.pz,
                   c.pa, c.ekin_per_a, c.tz, c.ta, kN, kv.first, kv.second,
                   sum_e[kv.first] / double(kv.second), sum_e2[kv.first] / double(kv.second),
                   sum_k2[kv.first] / nk);
    }
  }
  std::fclose(f);
  std::fclose(g);
}

/// bic_apply.csv and bic_apply_status.csv - `G4BinaryCascade::ApplyYourself` BELOW `theBCminP`,
/// the only part of that model this package has.
///
/// The first statement of `ApplyYourself` is
///
///     if (initial4Momentum.e()-initial4Momentum.m() < theBCminP &&
///         (definition==neutron || definition==proton))
///       return theDeExcitation->ApplyYourself(aTrack, aNucleus);
///
/// so a nucleon below 45 MeV never enters the cascade: the whole reaction is
/// `G4PreCompoundModel::ApplyYourself`, which is P6's `apply_yourself_initial_fragment` plus
/// `preco::deexcite`. That branch is complete in the port and this is what checks it, in the
/// shape the framework actually calls - not P6's helper called directly, which is what
/// `preco_apply.csv` already covers.
///
/// The 44 and 46 MeV rows are the point of the file. They straddle `theBCminP` by 2 MeV, the
/// model answers them from two entirely different code paths, and the port answers one and
/// refuses the other. A port that had the threshold at 40 or 50 would produce a cascade's
/// multiplicity where a compound nucleus belongs, and the 46 MeV rows are here so that the
/// refusal is compared against something rather than assumed.
///
/// Columns match bic_blir.csv exactly so that tests/test_bic_apply.cu reads both with one reader.
/// `pz`/`pa` are the projectile's charge and baryon number, which for a nucleon are (0 or 1, 1).
void write_bic_apply() {
  auto* handler = new G4ExcitationHandler();
  auto* preco = new G4PreCompoundModel(handler);
  auto* bic = new G4BinaryCascade(preco);
  bic->SetMaxEnergy(1.5 * CLHEP::GeV);

  struct NCase { int pdg; double ekin; int tz, ta; const char* name; };
  const NCase kCases[] = {
    {2212,  5.0,  6,  12, "p5_C12"},     {2112,  5.0,  6,  12, "n5_C12"},
    {2212, 20.0,  6,  12, "p20_C12"},    {2112, 20.0,  6,  12, "n20_C12"},
    {2212, 44.0,  6,  12, "p44_C12"},    {2112, 44.0,  6,  12, "n44_C12"},
    {2212, 46.0,  6,  12, "p46_C12"},    {2112, 46.0,  6,  12, "n46_C12"},
    {2212, 20.0,  8,  16, "p20_O16"},    {2112, 20.0,  8,  16, "n20_O16"},
    {2212, 20.0, 13,  27, "p20_Al27"},   {2112, 20.0, 13,  27, "n20_Al27"},
    {2212, 44.0, 26,  56, "p44_Fe56"},   {2112, 44.0, 26,  56, "n44_Fe56"},
    {2212, 20.0, 82, 208, "p20_Pb208"},  {2112, 20.0, 82, 208, "n20_Pb208"},
    {2212, 44.0, 82, 208, "p44_Pb208"},  {2112, 44.0, 82, 208, "n44_Pb208"},
  };
  const int kN = 5000;

  FILE* f = std::fopen("bic_apply.csv", "w");
  std::fprintf(f, "case,pz,pa,ekin_per_a_MeV,tz,ta,N,pdg,count,mean_ekin_MeV,mean_ekin2_MeV2,"
                  "mean_mult2\n");
  FILE* g = std::fopen("bic_apply_status.csv", "w");
  std::fprintf(g, "case,pz,pa,ekin_per_a_MeV,tz,ta,N,status,n_secondaries,sum_z,sum_a,"
                  "mean_e_MeV,mean_pz_MeV,mean_mult\n");

  for (const NCase& c : kCases) {
    const G4ParticleDefinition* part =
        (c.pdg == 2212) ? static_cast<const G4ParticleDefinition*>(G4Proton::Proton())
                        : static_cast<const G4ParticleDefinition*>(G4Neutron::Neutron());
    const int pz = (c.pdg == 2212) ? 1 : 0;
    CLHEP::HepRandom::setTheSeed(666000L + c.ta * 100 + G4int(c.ekin) + pz);

    std::map<int, long long> count;
    std::map<int, double> sum_e, sum_e2, sum_k2;
    std::map<int, int> per_event;
    long long n_alive = 0, n_kill = 0, n_sec = 0;
    long long sum_z = -1, sum_a = -1;
    bool za_varies = false;
    double sum_tot_e = 0.0, sum_tot_pz = 0.0;

    for (int n = 0; n < kN; ++n) {
      G4DynamicParticle dp(part, G4ThreeVector(0, 0, 1), c.ekin * MeV);
      G4HadProjectile proj(dp);
      G4Nucleus nucleus(c.ta, c.tz);
      G4HadFinalState* r = bic->ApplyYourself(proj, nucleus);
      if (r == nullptr) { continue; }
      if (r->GetStatusChange() == isAlive) { ++n_alive; continue; }
      ++n_kill;
      per_event.clear();
      long long ez = 0, ea = 0;
      G4LorentzVector tot(0., 0., 0., 0.);
      const std::size_t ns = r->GetNumberOfSecondaries();
      n_sec += static_cast<long long>(ns);
      for (std::size_t i = 0; i < ns; ++i) {
        const G4HadSecondary* s = r->GetSecondary(i);
        const G4DynamicParticle* p = s->GetParticle();
        const int pdg = p->GetDefinition()->GetPDGEncoding();
        const double ekin = p->GetKineticEnergy() / MeV;
        ++count[pdg];
        sum_e[pdg] += ekin;
        sum_e2[pdg] += ekin * ekin;
        ++per_event[pdg];
        tot += p->Get4Momentum();
        if (pdg > 1000000000) {
          ea += (pdg / 10) % 1000;
          ez += (pdg / 10000) % 1000;
        } else if (pdg == 2112) { ea += 1; }
        else if (pdg == 2212) { ea += 1; ez += 1; }
        else if (pdg == 211) { ez += 1; }
        else if (pdg == -211) { ez -= 1; }
      }
      for (const auto& kv : per_event) {
        sum_k2[kv.first] += double(kv.second) * kv.second;
      }
      if (sum_z < 0) { sum_z = ez; sum_a = ea; }
      else if (ez != sum_z || ea != sum_a) { za_varies = true; }
      sum_tot_e += tot.e() / MeV;
      sum_tot_pz += tot.z() / MeV;
    }

    const double nk = (n_kill > 0) ? double(n_kill) : 1.0;
    std::fprintf(g, "%s,%d,%d,%.17g,%d,%d,%d,%s,%lld,%lld,%lld,%.17g,%.17g,%.17g\n", c.name,
                 pz, 1, c.ekin, c.tz, c.ta, kN,
                 (n_alive == kN) ? "isAlive" : ((n_kill == kN) ? "stopAndKill" : "MIXED"),
                 n_sec, za_varies ? -1 : sum_z, za_varies ? -1 : sum_a, sum_tot_e / nk,
                 sum_tot_pz / nk, double(n_sec) / nk);
    for (const auto& kv : count) {
      std::fprintf(f, "%s,%d,%d,%.17g,%d,%d,%d,%d,%lld,%.17g,%.17g,%.17g\n", c.name, pz, 1,
                   c.ekin, c.tz, c.ta, kN, kv.first, kv.second,
                   sum_e[kv.first] / double(kv.second), sum_e2[kv.first] / double(kv.second),
                   sum_k2[kv.first] / nk);
    }
  }
  std::fclose(f);
  std::fclose(g);
}

// =============================================================================================
// The im_r_matrix collision tree (P9b).
// =============================================================================================

/// The eight uniforms the deterministic angular dump cycles through, and the engine that serves
/// them. Same eight as ref/dump/dump_elastic.cc, deliberately: a sampler driven by a prescribed
/// sequence is a deterministic function of its inputs, so `CosTheta` can be compared EXACTLY
/// rather than through a histogram, and one sequence shared across packages means one place to
/// change it. The phase shifts the window so that a sampler which draws two uniforms per call
/// (G4AngularDistribution does) still sees all eight.
const double kImrSeq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};

class ImrCycleEngine : public CLHEP::HepRandomEngine {
 public:
  void reset(int phase) { phase_ = phase; n_ = 0; }
  int draws() const { return n_; }
  double flat() override {
    const double v = kImrSeq[(n_ + phase_) % 8];
    ++n_;
    return v;
  }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "ImrCycleEngine"; }

 private:
  int phase_ = 0;
  int n_ = 0;
};

/// An engine that serves ONE prescribed uniform, for the table sweep below.
///
/// The eight-value cycle above proves the sampler is right at the eight points of the cumulative
/// it reaches, and it was MEASURED that that is all it proves: moving one entry of the 7,020-value
/// NP table by one part in 65,000 changed none of the 1,600 angles it produced. A table sampler
/// has to be checked over its table, not over eight of its rows - so the sweep drives `sample`
/// across (0, 1) in 199 steps at every tabulated energy, which makes the bisection visit
/// essentially every one of the 180 angle bins in every one of the 39 and 40 energy rows.
class ImrFixedEngine : public CLHEP::HepRandomEngine {
 public:
  void set(double v) { v_ = v; n_ = 0; }
  int draws() const { return n_; }
  double flat() override {
    ++n_;
    return v_;
  }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "ImrFixedEngine"; }

 private:
  double v_ = 0.5;
  int n_ = 0;
};

/// The 39 and 40 lab kinetic energies the two tables are tabulated at, in GeV, copied from
/// G4AngularDistributionNP::elab and G4AngularDistributionPP::elab. They are private statics in
/// classes with no accessor, so the sweep cannot ask for them - and `tools/extract_bic_imr.pl`
/// asserts the port's copy against the same source lines, so a release that moves a tabulated
/// energy fails the extractor and then fails here as a shifted angle.
const double kImrElabNP[39] = {0.010, 0.020, 0.030, 0.050, 0.070, 0.100, 0.140, 0.180, 0.240,
                               0.340, 0.420, 0.500, 0.580, 0.620, 0.680, 0.740, 0.800, 0.900,
                               1.000, 1.100, 1.200, 1.300, 1.400, 1.500, 1.600, 1.700, 1.800,
                               1.900, 2.000, 2.200, 2.400, 2.600, 2.800, 3.000, 3.400, 3.800,
                               4.200, 4.600, 5.000};
const double kImrElabPP[40] = {0.010, 0.020, 0.040, 0.070, 0.100, 0.120, 0.140, 0.180, 0.220,
                               0.260, 0.280, 0.300, 0.340, 0.420, 0.520, 0.620, 0.700, 0.800,
                               0.900, 1.000, 1.100, 1.200, 1.300, 1.400, 1.500, 1.600, 1.700,
                               1.800, 1.900, 2.000, 2.200, 2.400, 2.600, 2.800, 3.000, 3.400,
                               3.800, 4.200, 4.600, 5.000};

/// Builds the two kinetic tracks a cross-section source is asked about: particle 1 along +z with
/// whatever energy puts the pair at `sqrt_s`, particle 2 at rest. That is the configuration
/// `G4CollisionComposite::BufferCrossSection` builds too, and it is the only one in which
/// `sqrt_s` determines the answer - every class here reads `(p1+p2).mag()` and nothing else
/// about the kinematics.
///
/// The REQUESTED sqrt(s) and the one Geant4 then computes from the two four-vectors differ by
/// rounding, so the caller dumps the computed one and the port is fed that. Comparing against
/// the requested value would be comparing two different energies at 1e-16 and calling the
/// difference a transcription error.
void imr_make_pair(const G4ParticleDefinition* d1, const G4ParticleDefinition* d2,
                   double sqrt_s, G4LorentzVector& p1, G4LorentzVector& p2) {
  const double m1 = d1->GetPDGMass();
  const double m2 = d2->GetPDGMass();
  const double e1 = (sqrt_s * sqrt_s - m1 * m1 - m2 * m2) / (2.0 * m2);
  const double p = (e1 > m1) ? std::sqrt(e1 * e1 - m1 * m1) : 0.0;
  p1 = G4LorentzVector(G4ThreeVector(0, 0, p), std::sqrt(p * p + m1 * m1));
  p2 = G4LorentzVector(G4ThreeVector(0, 0, 0), m2);
}

void write_imr_xsec() {
  FILE* f = std::fopen("bic_imr_xsec.csv", "w");
  std::fprintf(f, "pair,source,sqrt_s_MeV,sigma_mb\n");

  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* n = G4Neutron::NeutronDefinition();
  struct Pair { const char* name; const G4ParticleDefinition* a; const G4ParticleDefinition* b; };
  const Pair pairs[] = {{"pp", p, p}, {"nn", n, n}, {"np", n, p}, {"pn", p, n}};

  // The grid. 1876.5 MeV is 2 m_p, the threshold; 1877.05 is the first point of
  // G4XNNTotalLowE::ss; 1.8964808 GeV and one log step either side of it are the three places
  // the four log vectors change branch; 3000 and 5000 MeV are the two patch boundaries and are
  // included from BOTH sides at one part in 1e9, because which arm answers at the boundary is
  // decided by a `<` against a `<=` and that is exactly what a transcription gets wrong.
  std::vector<double> grid;
  for (double e = 1876.6; e < 1900.0; e += 1.0) { grid.push_back(e); }
  for (double e = 1900.0; e < 3000.0; e += 20.0) { grid.push_back(e); }
  for (double e = 3000.0; e <= 6000.0; e += 50.0) { grid.push_back(e); }
  const double kEdges[] = {1877.05, 1896.4808, 1877.5,  1877.6,   1878.0,
                           1896.4808 * 0.99,   2999.999999, 3000.0, 3000.000001,
                           4999.999999,        5000.0,      5000.000001,
                           1858.6,  1860.0,   1870.0,  2925.49, 3002.71};
  for (double e : kEdges) { grid.push_back(e); }

  // The four composites and every arm under them, each dumped separately so that a failing
  // composite says which arm it came from instead of only that it disagrees.
  G4XNNTotal xNNTotal;
  G4XnpTotal xnpTotal;
  G4XNNElastic xNNElastic;
  G4XnpElastic xnpElastic;
  G4XNNTotalLowE xNNTotalLowE;
  G4XnpTotalLowE xnpTotalLowE;
  G4XNNElasticLowE xNNElasticLowE;
  G4XnpElasticLowE xnpElasticLowE;
  G4XPDGTotal xPDGTotal;
  G4XPDGElastic xPDGElastic;

  for (const Pair& pr : pairs) {
    for (double want : grid) {
      G4LorentzVector q1, q2;
      imr_make_pair(pr.a, pr.b, want, q1, q2);
      G4KineticTrack t1(pr.a, 0.0, G4ThreeVector(0, 0, 0), q1);
      G4KineticTrack t2(pr.b, 0.0, G4ThreeVector(0, 0, 0), q2);
      const double s = (t1.Get4Momentum() + t2.Get4Momentum()).mag();
      auto row = [&](const char* name, double sigma) {
        std::fprintf(f, "%s,%s,%.17g,%.17g\n", pr.name, name, s, sigma / millibarn);
      };
      row("XNNTotal", xNNTotal.CrossSection(t1, t2));
      row("XNNElastic", xNNElastic.CrossSection(t1, t2));
      row("XnpElastic", xnpElastic.CrossSection(t1, t2));
      row("XnpTotal", xnpTotal.CrossSection(t1, t2));
      row("XNNTotalLowE", xNNTotalLowE.CrossSection(t1, t2));
      row("XNNElasticLowE", xNNElasticLowE.CrossSection(t1, t2));
      row("XnpElasticLowE", xnpElasticLowE.CrossSection(t1, t2));
      row("XnpTotalLowE", xnpTotalLowE.CrossSection(t1, t2));
      row("XPDGTotal", xPDGTotal.CrossSection(t1, t2));
      row("XPDGElastic", xPDGElastic.CrossSection(t1, t2));
    }
  }
  std::fclose(f);
}

void write_imr_angular() {
  FILE* f = std::fopen("bic_imr_angular.csv", "w");
  std::fprintf(f, "dist,s_MeV2,m1_MeV,m2_MeV,phase,cos_theta,draws\n");
  FILE* g = std::fopen("bic_imr_obe.csv", "w");
  std::fprintf(g, "sym,s_MeV2,m1_MeV,m2_MeV,cos_theta,dsigma\n");

  const double mp = G4Proton::ProtonDefinition()->GetPDGMass();
  const double mn = G4Neutron::NeutronDefinition()->GetPDGMass();

  // The lab kinetic energies the two tables are tabulated at, plus points between and outside
  // them: 5 MeV is below elab[0] = 10 MeV (where the energy bisection extrapolates downwards)
  // and 6 GeV above elab[last] = 5 GeV (where it extrapolates upwards). Both extrapolations are
  // live in the cascade - a 1.5 GeV proton's first collision is above nothing, but a nucleon
  // that has already lost most of its energy is below 10 MeV all the time.
  const double kEkinMeV[] = {5.0,   10.0,   15.0,   20.0,   35.0,   50.0,  70.0,   100.0,
                             140.0, 180.0,  240.0,  300.0,  400.0,  500.0, 620.0,  800.0,
                             1000.0, 1200.0, 1500.0, 1900.0, 2500.0, 3000.0, 4000.0, 5000.0,
                             6000.0};
  // Three mass pairs: both on shell, and two off-shell cases, because a cascade nucleon inside
  // the nuclear potential has an actual mass below its PDG mass (docs/PORTED.md 2.1.10) and
  // `ek`'s formula divides by m1 alone.
  struct Masses { double m1, m2; };
  const Masses kMasses[] = {{mp, mp}, {mn, mp}, {mp - 30.0, mp}, {mp, mp - 45.0}};

  auto* eng = new ImrCycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);

  G4AngularDistributionNP np;
  G4AngularDistributionPP pp;
  G4AngularDistribution obeSym(true);
  G4AngularDistribution obeAsym(false);

  for (const Masses& m : kMasses) {
    for (double ekin : kEkinMeV) {
      // S from the lab kinetic energy the way the tables' own `ek` inverts it, so that the
      // energy landing in the bisection is the tabulated one rather than one rounding away.
      const double s = (ekin + m.m1) * 2.0 * m.m1 + m.m1 * m.m1 + m.m2 * m.m2;
      for (int phase = 0; phase < 8; ++phase) {
        eng->reset(phase);
        const double c1 = np.CosTheta(s, m.m1, m.m2);
        std::fprintf(f, "NP,%.17g,%.17g,%.17g,%d,%.17g,%d\n", s, m.m1, m.m2, phase, c1,
                     eng->draws());
        eng->reset(phase);
        const double c2 = pp.CosTheta(s, m.m1, m.m2);
        std::fprintf(f, "PP,%.17g,%.17g,%.17g,%d,%.17g,%d\n", s, m.m1, m.m2, phase, c2,
                     eng->draws());
        eng->reset(phase);
        const double c3 = obeSym.CosTheta(s, m.m1, m.m2);
        std::fprintf(f, "OBEsym,%.17g,%.17g,%.17g,%d,%.17g,%d\n", s, m.m1, m.m2, phase, c3,
                     eng->draws());
        eng->reset(phase);
        const double c4 = obeAsym.CosTheta(s, m.m1, m.m2);
        std::fprintf(f, "OBEasym,%.17g,%.17g,%.17g,%d,%.17g,%d\n", s, m.m1, m.m2, phase, c4,
                     eng->draws());
        eng->reset(phase);
        const double c5 = np.Phi();
        std::fprintf(f, "Phi,%.17g,%.17g,%.17g,%d,%.17g,%d\n", s, m.m1, m.m2, phase, c5,
                     eng->draws());
      }
      // The normalised cumulative itself, which is what the bisection inverts - dumped on a
      // cos(theta) grid so that a disagreement in the formula is separated from a disagreement
      // in the twelve halvings above it.
      for (int i = 0; i <= 20; ++i) {
        const double ct = -1.0 + 0.1 * i;
        std::fprintf(g, "1,%.17g,%.17g,%.17g,%.17g,%.17g\n", s, m.m1, m.m2, ct,
                     obeSym.DifferentialCrossSection(s, m.m1, m.m2, ct));
        std::fprintf(g, "0,%.17g,%.17g,%.17g,%.17g,%.17g\n", s, m.m1, m.m2, ct,
                     obeAsym.DifferentialCrossSection(s, m.m1, m.m2, ct));
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  delete eng;
  std::fclose(f);
  std::fclose(g);
}

/// The table sweep: every tabulated energy of each table, and 199 values of the sampled uniform
/// at each, so that the bisection walks the whole cumulative rather than eight points of it.
/// See ImrFixedEngine for why this exists and what the eight-value cycle was measured not to
/// catch. The energy is set EXACTLY on a tabulated point and also halfway between two, because
/// on a node `delab` cancels the interpolation and between nodes it does not - and a
/// transcription that read the wrong energy row would pass on the nodes alone.
void write_imr_angular_sweep() {
  FILE* f = std::fopen("bic_imr_angular_sweep.csv", "w");
  std::fprintf(f, "dist,s_MeV2,m1_MeV,m2_MeV,sample,cos_theta\n");

  const double mp = G4Proton::ProtonDefinition()->GetPDGMass();
  auto* eng = new ImrFixedEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);
  G4AngularDistributionNP np;
  G4AngularDistributionPP pp;

  auto sweep = [&](const char* name, const double* elab, int n_e, bool is_np) {
    for (int j = 0; j < n_e; ++j) {
      // The tabulated energy, and the midpoint to the next one.
      double ek[2] = {elab[j] * CLHEP::GeV, 0.0};
      int n_ek = 1;
      if (j + 1 < n_e) {
        ek[1] = 0.5 * (elab[j] + elab[j + 1]) * CLHEP::GeV;
        n_ek = 2;
      }
      for (int q = 0; q < n_ek; ++q) {
        const double s = (ek[q] + mp) * 2.0 * mp + mp * mp + mp * mp;
        for (int i = 1; i < 200; ++i) {
          const double sample = i / 200.0;
          eng->set(sample);
          const double c = is_np ? np.CosTheta(s, mp, mp) : pp.CosTheta(s, mp, mp);
          std::fprintf(f, "%s,%.17g,%.17g,%.17g,%.17g,%.17g\n", name, s, mp, mp, sample, c);
        }
      }
    }
  };
  sweep("NP", kImrElabNP, 39, true);
  sweep("PP", kImrElabPP, 40, false);

  CLHEP::HepRandom::setTheEngine(saved);
  delete eng;
  std::fclose(f);
}

/// The two elastic collision channels end to end: `IsInCharge`, the channel's own cross section,
/// `G4CollisionNN`'s recast total, and the two outgoing four-momenta `G4VElasticCollision::
/// FinalState` produces under the prescribed cycle.
///
/// The tracks are built OFF SHELL on purpose in half the cases. A cascade nucleon is below its
/// PDG mass by the nuclear potential, and three separate things in this path read a mass: the
/// composite's recast reads `GetActualMass()` and `GetPDGMass()`, the angular distribution is
/// sampled with the actual masses, and the outgoing momenta are built with the PDG ones. An
/// on-shell-only dump would make all three agree and check none of them.
void write_imr_collision() {
  FILE* f = std::fopen("bic_imr_collision.csv", "w");
  std::fprintf(f,
               "pair,offshell,tilt,sqrt_s_MeV,in1x,in1y,in1z,in1e,in2x,in2y,in2z,in2e,"
               "in_charge_nn,in_charge_nnel,in_charge_npel,"
               "sigma_total_mb,sigma_nnel_mb,sigma_npel_mb\n");
  FILE* g = std::fopen("bic_imr_elastic_fs.csv", "w");
  std::fprintf(g,
               "pair,offshell,tilt,sqrt_s_MeV,in1x,in1y,in1z,in1e,in2x,in2y,in2z,in2e,"
               "phase,empty,p1x,p1y,p1z,p1e,p2x,p2y,p2z,p2e,draws\n");

  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* n = G4Neutron::NeutronDefinition();
  struct Pair { const char* name; const G4ParticleDefinition* a; const G4ParticleDefinition* b; };
  const Pair pairs[] = {{"pp", p, p}, {"nn", n, n}, {"np", n, p}, {"pn", p, n}};

  // sqrt(s) from just above the two-nucleon threshold to 3.5 GeV, which covers QBBC's whole BIC
  // window for a nucleon (1.5 GeV of kinetic energy on a nucleon at rest is sqrt(s) = 2.4 GeV)
  // and runs past it into the patch's transition region.
  std::vector<double> grid;
  for (double e = 1880.0; e <= 2400.0; e += 20.0) { grid.push_back(e); }
  for (double e = 2450.0; e <= 3500.0; e += 50.0) { grid.push_back(e); }

  G4CollisionNN nnComposite;
  G4CollisionNNElastic nnEl;
  G4CollisionnpElastic npEl;

  auto* eng = new ImrCycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);

  for (const Pair& pr : pairs) {
    for (int off = 0; off < 2; ++off) {
      // `off` shifts particle 1's ENERGY down by 30 MeV at fixed three-momentum, which is what
      // a nuclear potential does to a cascade nucleon: the track is then off its mass shell
      // downwards and `GetActualMass()` is below `GetPDGMass()`.
      for (int tilt = 0; tilt < 3; ++tilt) {
        // WHY THE TILT EXISTS. `G4VElasticCollision::FinalState` builds a rotation that takes
        // particle 1 onto the +z axis in the CM frame and inverts it afterwards. With both
        // tracks along z - which is how every projectile arrives, because
        // G4HadProjectile::InitialiseLocal stores (0,0,p,E), docs/RISK.md V75 - both rotations
        // are the IDENTITY and the whole block is unobservable. It was measured: composing with
        // `toZ` instead of `toZ.inverse()` changed none of 24,576 momentum components.
        //
        // Inside a cascade nothing is along z. A track has been through G4RKPropagation and
        // possibly an earlier collision, and the target nucleon carries Fermi momentum. So
        // tilt 1 rotates particle 1 into the x-z plane and tilt 2 gives it all three components
        // AND gives particle 2 a momentum of its own, which is the configuration
        // `G4Scatterer::Scatter` actually hands the channel.
        for (double want : grid) {
          G4LorentzVector q1, q2;
          imr_make_pair(pr.a, pr.b, want, q1, q2);
          if (tilt == 1) {
            G4ThreeVector v = q1.vect();
            v.rotateY(0.5236);  // 30 degrees, into the x-z plane
            q1.setVect(v);
          } else if (tilt == 2) {
            G4ThreeVector v = q1.vect();
            v.rotateY(0.9);
            v.rotateZ(2.1);
            q1.setVect(v);
            // Particle 2 given a Fermi-scale momentum of its own, on its own mass shell.
            const G4ThreeVector v2(37.0, -52.0, 21.0);
            q2 = G4LorentzVector(
                v2, std::sqrt(v2.mag2() + pr.b->GetPDGMass() * pr.b->GetPDGMass()));
          }
          if (off != 0) { q1.setE(q1.e() - 30.0); }
          G4KineticTrack t1(pr.a, 0.0, G4ThreeVector(0, 0, 0), q1);
          G4KineticTrack t2(pr.b, 0.0, G4ThreeVector(0, 0, 0), q2);
          const double s = (t1.Get4Momentum() + t2.Get4Momentum()).mag();
          // The two INPUT four-momenta are dumped as well as sqrt(s). They have to be: in the
          // off-shell rows the energy is lowered at fixed three-momentum, so sqrt(s) no longer
          // determines the pair - a reader who inverted `s = m1^2 + m2^2 + 2 m2 E1` would get a
          // different kinematic configuration with the same invariant mass, and the recast in
          // G4CollisionNN::CrossSection reads the energy and the momentum separately.
          std::fprintf(f,
                       "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                       "%d,%d,%d,%.17g,%.17g,%.17g\n",
                       pr.name, off, tilt, s, q1.x(), q1.y(), q1.z(), q1.t(), q2.x(), q2.y(),
                       q2.z(), q2.t(),
                       nnComposite.IsInCharge(t1, t2) ? 1 : 0, nnEl.IsInCharge(t1, t2) ? 1 : 0,
                       npEl.IsInCharge(t1, t2) ? 1 : 0,
                       nnComposite.CrossSection(t1, t2) / millibarn,
                       nnEl.IsInCharge(t1, t2) ? nnEl.CrossSection(t1, t2) / millibarn : 0.0,
                       npEl.IsInCharge(t1, t2) ? npEl.CrossSection(t1, t2) / millibarn : 0.0);

          const bool is_np = npEl.IsInCharge(t1, t2);
          const bool is_nn = nnEl.IsInCharge(t1, t2);
          if (!is_np && !is_nn) { continue; }
          const char* kIn = "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,";
          for (int phase = 0; phase < 8; ++phase) {
            eng->reset(phase);
            G4KineticTrackVector* fs =
                is_np ? npEl.FinalState(t1, t2) : nnEl.FinalState(t1, t2);
            std::fprintf(g, kIn, pr.name, off, tilt, s, q1.x(), q1.y(), q1.z(), q1.t(), q2.x(),
                         q2.y(), q2.z(), q2.t(), phase);
            if (fs == nullptr || fs->size() < 2) {
              std::fprintf(g, "1,0,0,0,0,0,0,0,0,%d\n", eng->draws());
            } else {
              const G4LorentzVector a = (*fs)[0]->Get4Momentum();
              const G4LorentzVector b = (*fs)[1]->Get4Momentum();
              std::fprintf(g, "0,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n", a.x(),
                           a.y(), a.z(), a.t(), b.x(), b.y(), b.z(), b.t(), eng->draws());
            }
            if (fs != nullptr) {
              for (auto* kt : *fs) { delete kt; }
              delete fs;
            }
          }
        }
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  delete eng;
  std::fclose(f);
  std::fclose(g);
}

/// `G4Scatterer::GetTimeToInteraction` and `GetCrossSection` over a grid of positions and
/// momenta, and `G4CollisionManager`'s ordering over a list built from them.
///
/// The positions are what make this a test of the SCHEDULER rather than of the cross section:
/// the impact parameter decides five of the six exits, and four of the five thresholds are
/// distances. The grid therefore sweeps the transverse offset from 0 to 5 fm, which brackets
/// `sqrt(500 mb/pi)` = 1.26 fm and `sqrt(200 mb/pi)` = 0.80 fm, and includes a target BEHIND the
/// projectile so the negative-time exit is exercised.
void write_imr_scatterer() {
  FILE* f = std::fopen("bic_imr_scatterer.csv", "w");
  std::fprintf(f,
               "pair,along_z,ekin_MeV,bx_fm,dz_fm,p1x,p1y,p1z,p1e,p2x,p2y,p2z,p2e,"
               "time_ns,sigma_mb\n");
  FILE* g = std::fopen("bic_imr_manager.csv", "w");
  std::fprintf(g, "step,op,arg,size,next_index,next_time\n");

  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* n = G4Neutron::NeutronDefinition();
  struct Pair { const char* name; const G4ParticleDefinition* a; const G4ParticleDefinition* b; };
  const Pair pairs[] = {{"pp", p, p}, {"nn", n, n}, {"np", n, p}, {"pn", p, n}};

  // Impact parameters in fermi, straddling all three of the distance thresholds from both
  // sides. ONE MILLIBARN IS 0.1 fm^2, which is the arithmetic the first version of this grid got
  // wrong - it put its "threshold" points at 0.79 and 1.26 fm, a factor of pi off, and four of
  // the six perturbations below then passed because no grid point ever crossed a gate:
  //
  //   sqrt(200 mb / pi)       = 2.5231 fm   the charged and the neutron gates
  //   sqrt(500 mb / pi)       = 3.9894 fm   the CM distance gate
  //   sqrt(500 mb / (0.7 pi)) = 4.7683 fm   the fast LAB gate, with its 0.7 margin
  //
  // and the window between the last two is the ONLY place the 0.7 can decide anything.
  const double kB[] = {0.0,   0.1,   0.5,   1.0,   1.5,   2.0,   2.4,    2.5230, 2.5232,
                       2.6,   2.8,   3.0,   3.5,   3.9,   3.9893, 3.9895, 4.2,   4.5,
                       4.7682, 4.7684, 5.0,  5.5,   7.0,   9.0};
  // Longitudinal separations in fermi; the negative one puts the target behind the projectile.
  const double kDz[] = {-3.0, 0.0, 2.0, 6.0};
  // Kinetic energies spanning QBBC's BIC window for a nucleon. 2 and 5 MeV are there because
  // that is where the total cross section is hundreds of millibarn - the only place where a
  // 200 mb gate can reject a pair the final `distance <= sigma/pi` would have accepted. 1010 MeV
  // is where sqrt(s) crosses 1.91 GeV and the neutron rule switches on.
  const double kT[] = {2.0,   5.0,   12.0,  20.0,  60.0,   150.0,  400.0,
                       900.0, 1010.0, 1100.0, 1500.0};

  G4Scatterer scatterer;

  for (const Pair& pr : pairs) {
    for (int along_z = 0; along_z < 2; ++along_z) {
      for (double ekin : kT) {
        for (double b : kB) {
          for (double dz : kDz) {
            const double m1 = pr.a->GetPDGMass();
            const double m2 = pr.b->GetPDGMass();
            const double e1 = ekin + m1;
            const double pmag = std::sqrt(e1 * e1 - m1 * m1);
            G4ThreeVector v1(0, 0, pmag);
            // `along_z == 0` keeps the projectile on the axis, which takes GetTimeToInteraction's
            // FAST branch (|unit().z() - 1| < 1e-6); `along_z == 1` tilts it by 0.4 rad, which
            // takes the general branch. The two compute the same time by different arithmetic
            // and the port has to have both.
            if (along_z != 0) { v1.rotateY(0.4); }
            const G4LorentzVector q1(v1, e1);
            // The target carries a Fermi-scale momentum of its own, which is what makes the
            // "target at rest for the geometry, moving for the energy" split observable.
            const G4ThreeVector v2(23.0, -41.0, 17.0);
            const G4LorentzVector q2(v2, std::sqrt(v2.mag2() + m2 * m2));
            const G4ThreeVector x1(0, 0, 0);
            const G4ThreeVector x2(b * fermi, 0, dz * fermi);
            G4KineticTrack t1(pr.a, 0.0, x1, q1);
            G4KineticTrack t2(pr.b, 0.0, x2, q2);
            const double t = scatterer.GetTimeToInteraction(t1, t2);
            std::fprintf(f,
                         "%s,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                         "%.17g,%.17g,%.17g\n",
                         pr.name, along_z, ekin, b, dz, q1.x(), q1.y(), q1.z(), q1.t(), q2.x(),
                         q2.y(), q2.z(), q2.t(), (t < DBL_MAX) ? t : -1.0,
                         scatterer.GetCrossSection(t1, t2) / millibarn);
          }
        }
      }
    }
  }

  // G4CollisionManager: a prescribed sequence of adds and removals, with the pending count and
  // the earliest collision after each. The ordering rule is what matters - `GetNextCollision`
  // uses a STRICT `>`, so the FIRST of several equal times wins, and a port that used `>=` would
  // reorder every tie in the cascade.
  {
    G4CollisionManager mgr;
    std::vector<G4KineticTrack*> tracks;
    const G4LorentzVector q(G4ThreeVector(0, 0, 500), std::sqrt(500.0 * 500.0 +
                                                               p->GetPDGMass() *
                                                                   p->GetPDGMass()));
    for (int i = 0; i < 8; ++i) {
      tracks.push_back(new G4KineticTrack(p, 0.0, G4ThreeVector(0, 0, 0), q));
    }
    // Times chosen with two exact ties (0.5 twice, 0.9 twice) so the tie-break is exercised.
    const double times[] = {2.0, 0.5, 1.7, 0.9, 0.5, 3.1, 0.9, 1.2};
    int step = 0;
    // `next_index` is the index of the WINNING collision's primary track, not its time. The
    // first version reported the time, and with two collisions at 0.5 ns and two at 0.9 ns that
    // is the same number whichever wins - so flipping GetNextCollision's strict `>` to `>=`,
    // which reverses every tie, changed nothing. The tie-break is real: `>` keeps the FIRST of
    // equal times, and in the cascade the order collisions were added in is the order the
    // participants were found in.
    auto report = [&](const char* op, double arg) {
      G4CollisionInitialState* next = mgr.GetNextCollision();
      int which = -1;
      if (next != nullptr) {
        for (std::size_t k = 0; k < tracks.size(); ++k) {
          if (next->GetPrimary() == tracks[k]) { which = static_cast<int>(k); break; }
        }
      }
      std::fprintf(g, "%d,%s,%.17g,%d,%d,%.17g\n", step++, op, arg,
                   static_cast<int>(mgr.Entries()), which,
                   next ? next->GetCollisionTime() : -1.0);
    };
    for (int i = 0; i < 8; ++i) {
      mgr.AddCollision(times[i], tracks[i], tracks[(i + 1) % 8]);
      report("add", times[i]);
    }
    // Remove the earliest, twice, so the tie-break is observed resolving.
    for (int i = 0; i < 2; ++i) {
      G4CollisionInitialState* next = mgr.GetNextCollision();
      const double tt = next->GetCollisionTime();
      mgr.RemoveCollision(next);
      report("remove_next", tt);
    }
    // Then cane two tracks, which removes every collision touching either.
    G4KineticTrackVector caned;
    caned.push_back(tracks[3]);
    caned.push_back(tracks[5]);
    mgr.RemoveTracksCollisions(&caned);
    report("remove_tracks", 35.0);
    mgr.ClearAndDestroy();
    report("clear", 0.0);
    for (auto* kt : tracks) { delete kt; }
  }
  std::fclose(f);
  std::fclose(g);
}

void dump_bic(const DumpContext&) {
  write_limits();
  write_density();
  write_fermi();
  write_nucleus_and_nucleons();
  write_field_and_rk();
  write_nucleus_stats();
  write_blir();
  write_bic_apply();
  write_imr_xsec();
  write_imr_angular();
  write_imr_angular_sweep();
  write_imr_collision();
  write_imr_scatterer();
}

}  // namespace

G4GPU_REGISTER_DUMP("bic",
                    "bic_limits.csv bic_density.csv bic_density_radius.csv bic_fermi.csv "
                    "bic_nucleus.csv bic_nucleons.csv bic_field.csv bic_rk.csv "
                    "bic_nucleus_stats.csv bic_nucleus_moments.csv bic_blir.csv "
                    "bic_blir_status.csv bic_apply.csv bic_apply_status.csv "
                    "bic_imr_xsec.csv bic_imr_angular.csv bic_imr_angular_sweep.csv bic_imr_obe.csv "
                    "bic_imr_collision.csv bic_imr_elastic_fs.csv "
                    "bic_imr_scatterer.csv bic_imr_manager.csv",
                    dump_bic);
