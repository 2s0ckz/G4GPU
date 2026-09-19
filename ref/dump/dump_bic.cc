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
#include <cstdlib>
#include <map>
#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "G4Alpha.hh"
#include "G4AngularDistribution.hh"
#include "G4AngularDistributionNP.hh"
#include "G4AngularDistributionPP.hh"
#include "G4BinaryCascade.hh"
#include "G4BinaryLightIonReaction.hh"
#include "G4CollisionManager.hh"
#include "G4MesonAbsorption.hh"
#include "G4CollisionMesonBaryon.hh"
#include "G4CollisionMesonBaryonElastic.hh"
#include "G4CollisionMesonBaryonToResonance.hh"
#include "G4CollisionNN.hh"
#include "G4CollisionNNElastic.hh"
#include "G4CollisionNNToDeltaDelta.hh"
#include "G4CollisionNNToDeltaDeltastar.hh"
#include "G4CollisionNNToDeltaNstar.hh"
#include "G4CollisionNNToNDelta.hh"
#include "G4CollisionNNToNDeltastar.hh"
#include "G4CollisionNNToNNstar.hh"
#include "G4CollisionNNElastic.hh"
#include "G4CollisionnpElastic.hh"
#include "G4Clebsch.hh"
#include "G4BaryonPartialWidth.hh"
#include "G4BaryonWidth.hh"
#include "G4ConcreteMesonBaryonToResonance.hh"
#include "G4ConcreteNNToDeltaDelta.hh"
#include "G4ConcreteNNToDeltaDeltastar.hh"
#include "G4ConcreteNNToDeltaNstar.hh"
#include "G4ConcreteNNToNDelta.hh"
#include "G4ConcreteNNToNDeltaStar.hh"
#include "G4ConcreteNNToNNStar.hh"
#include "G4DetailedBalancePhaseSpaceIntegral.hh"
#include "G4Deuteron.hh"
#include "G4DynamicParticle.hh"
#include "G4ExcitationHandler.hh"
#include "G4Fancy3DNucleus.hh"
#include "G4FermiMomentum.hh"
#include "G4FermiPhaseSpaceDecay.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadSecondary.hh"
#include "G4HadronicInteractionRegistry.hh"
#include "G4IonTable.hh"
#include "G4DecayKineticTracks.hh"
#include "G4IonTable.hh"
#include "G4DecayTable.hh"
#include "G4KineticTrack.hh"
#include "G4SampleResonance.hh"
#include "G4VDecayChannel.hh"
#include "G4KineticTrackVector.hh"
#include "G4Neutron.hh"
#include "G4NuclearFermiDensity.hh"
#include "G4NuclearShellModelDensity.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleon.hh"
#include "G4Nucleus.hh"
#include "G4PhysicsModelCatalog.hh"
#include "G4PionMinus.hh"
#include "G4Pow.hh"
#include "G4PionPlus.hh"
#include "G4PionZero.hh"
#include "G4PreCompoundModel.hh"
#include "G4Proton.hh"
#include "G4ResonanceNames.hh"
#include "G4RKPropagation.hh"
#include "G4Scatterer.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4SystemOfUnits.hh"
#include "G4VNuclearDensity.hh"
#include "G4XAqmElastic.hh"
#include "G4XAqmTotal.hh"
#include "G4XDeltaDeltaTable.hh"
#include "G4XDeltaDeltastarTable.hh"
#include "G4XDeltaNstarTable.hh"
#include "G4XNDeltaTable.hh"
#include "G4XNDeltastarTable.hh"
#include "G4XNNElastic.hh"
#include "G4XNNElasticLowE.hh"
#include "G4XNNTotal.hh"
#include "G4XMesonBaryonElastic.hh"
#include "G4XNNstarTable.hh"
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

/// A SIXTY-FOUR-value ladder, for the one sampler in this package that the eight-value cycle
/// above cannot drive: `G4FermiPhaseSpaceDecay::BetaKopylov`.
///
/// BetaKopylov is a rejection sampler with NO iteration guard:
///
///     do { chi = rndmEngine->flat();
///          F = std::sqrt(g4calc->powN(chi,N)*(1-chi));
///        } while ( Fmax*rndmEngine->flat() > F);
///
/// It draws TWO uniforms per attempt, so an eight-value cycle offers it exactly four distinct
/// (chi, test) pairs, and if none of the four is accepted the loop spins forever. MEASURED: with
/// the eight-value engine the dump ran 338 rows and then hung on the five-body case at threshold,
/// inside Geant4, with no way out. docs/RISK.md V157.
///
/// The ladder is `(2i+1)/128` for i = 0..63, which walks the whole of (0, 1) and so puts some
/// attempt near the mode of chi^N(1-chi) whatever N is. It is used ONLY by write_imr_fps; every
/// other sweep in this file keeps the eight-value cycle, because a longer sequence would weaken
/// the phase-by-phase comparison those rely on.
const double* kImr64Seq() {
  static double v[64];
  static bool built = false;
  if (!built) {
    for (int i = 0; i < 64; ++i) { v[i] = (2.0 * i + 1.0) / 128.0; }
    built = true;
  }
  return v;
}

class ImrCycle64Engine : public CLHEP::HepRandomEngine {
 public:
  void reset(int phase) { phase_ = phase; n_ = 0; }
  int draws() const { return n_; }
  double flat() override {
    const double v = kImr64Seq()[(n_ + phase_) % 64];
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
  std::string name() const override { return "ImrCycle64Engine"; }

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
/// Defined below, beside the measurement that explains why it has to be a leak.
G4Scatterer& imr_scatterer();

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
  // The four PION rows are what make G4CollisionMesonBaryon reachable through G4Scatterer at
  // all: QBBC hands BIC every pi+/pi- below its Bertini window, and until the composite existed
  // this dump had no pair `FindCollision` would answer 1 for. `p_pip` is the same pair with the
  // tracks the other way round, because `IsInCharge` is a disjunction over both orders and a
  // port that tested only (meson, baryon) would pass everything else.
  const Pair pairs[] = {{"pp", p, p},   {"nn", n, n},   {"np", n, p},    {"pn", p, n},
                        {"pip_p", G4PionPlus::PionPlusDefinition(), p},
                        {"pim_p", G4PionMinus::PionMinusDefinition(), p},
                        {"pip_n", G4PionPlus::PionPlusDefinition(), n},
                        {"p_pip", p, G4PionPlus::PionPlusDefinition()}};

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

  // The leaked singleton, not a local: destroying a G4Scatterer empties the static channel
  // list for every one that follows it. docs/RISK.md V155.
  G4Scatterer& scatterer = imr_scatterer();

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

/// The six resonance-production cross-section tables through their own public accessor, and
/// G4DetailedBalancePhaseSpaceIntegral for every resonance it knows.
///
/// `CrossSectionTable()` returns a `G4PhysicsVector` the caller owns, so the dump evaluates it
/// the way `G4XResonance::CrossSection` does - `GetValue(sqrtS, dummy)` - on a grid that includes
/// every one of the 121 tabulated energies and the midpoint between each pair. On a node the
/// interpolation is exact and the value is the table entry; between nodes it is the straight
/// line, and a table read one column off would agree on neither.
void write_imr_resonance() {
  FILE* f = std::fopen("bic_imr_restab.csv", "w");
  std::fprintf(f, "table,mass,sqrt_s_MeV,sigma_mb\n");
  FILE* g = std::fopen("bic_imr_dbi.csv", "w");
  std::fprintf(g, "name,sqrt_s_MeV,integral\n");

  // The short-lived particles have to exist before any of them can be found by name.
  G4ShortLivedConstructor shortLived;
  shortLived.ConstructParticle();
  G4ParticleTable* ptable = G4ParticleTable::GetParticleTable();

  G4XNDeltaTable nd;
  G4XDeltaDeltaTable dd;
  G4XNDeltastarTable ndstar;
  G4XDeltaDeltastarTable ddstar;
  G4XNNstarTable nnstar;
  G4XDeltaNstarTable dnstar;

  const char* kDeltastar[] = {"1600", "1620", "1700", "1900", "1905", "1910", "1920", "1930",
                              "1950"};
  const char* kNstar[] = {"1440", "1520", "1535", "1650", "1675", "1680", "1700", "1710",
                          "1720", "1900", "1990", "2090", "2190", "2220", "2250"};

  // The 121 tabulated energies, in GeV, copied from G4XNDeltaTable::energyTable - private, with
  // no accessor, and asserted against the source by tools/extract_bic_imr.pl on the port's side.
  const double kE[121] = {
    0.0,
    2.014, 2.014, 2.016, 2.018, 2.022, 2.026, 2.031, 2.037, 2.044, 2.052,
    2.061, 2.071, 2.082, 2.094, 2.107, 2.121, 2.135, 2.151, 2.168, 2.185,
    2.204, 2.223, 2.244, 2.265, 2.287, 2.311, 2.335, 2.360, 2.386, 2.413,
    2.441, 2.470, 2.500, 2.531, 2.562, 2.595, 2.629, 2.664, 2.699, 2.736,
    2.773, 2.812, 2.851, 2.891, 2.933, 2.975, 3.018, 3.062, 3.107, 3.153,
    3.200, 3.248, 3.297, 3.347, 3.397, 3.449, 3.502, 3.555, 3.610, 3.666,
    3.722, 3.779, 3.838, 3.897, 3.957, 4.018, 4.081, 4.144, 4.208, 4.273,
    4.339, 4.406, 4.473, 4.542, 4.612, 4.683, 4.754, 4.827, 4.900, 4.975,
    5.000, 6.134, 7.269, 8.403, 9.538, 10.672, 11.807, 12.941, 14.076, 15.210,
    16.345, 17.479, 18.613, 19.748, 20.882, 22.017, 23.151, 24.286, 25.420, 26.555,
    27.689, 28.824, 29.958, 31.092, 32.227, 33.361, 34.496, 35.630, 36.765, 37.899,
    39.034, 40.168, 41.303, 42.437, 43.571, 44.706, 45.840, 46.975, 48.109, 49.244};

  auto sweep = [&](const char* tag, int mass, const G4PhysicsVector* v) {
    if (v == nullptr) { return; }
    G4bool dummy = false;
    for (int i = 0; i < 121; ++i) {
      const double e = kE[i] * CLHEP::GeV;
      std::fprintf(f, "%s,%d,%.17g,%.17g\n", tag, mass, e,
                   const_cast<G4PhysicsVector*>(v)->GetValue(e, dummy) / millibarn);
      if (i + 1 < 121 && kE[i + 1] > kE[i]) {
        const double em = 0.5 * (kE[i] + kE[i + 1]) * CLHEP::GeV;
        std::fprintf(f, "%s,%d,%.17g,%.17g\n", tag, mass, em,
                     const_cast<G4PhysicsVector*>(v)->GetValue(em, dummy) / millibarn);
      }
    }
    // Both ends of the guard, where Value() clamps to the first and last entries.
    const double below = 0.5 * CLHEP::GeV;
    const double above = 60.0 * CLHEP::GeV;
    std::fprintf(f, "%s,%d,%.17g,%.17g\n", tag, mass, below,
                 const_cast<G4PhysicsVector*>(v)->GetValue(below, dummy) / millibarn);
    std::fprintf(f, "%s,%d,%.17g,%.17g\n", tag, mass, above,
                 const_cast<G4PhysicsVector*>(v)->GetValue(above, dummy) / millibarn);
  };

  // The two single-column tables are G4VXResonanceTable subclasses with a no-argument accessor;
  // the four multi-column ones are keyed by particle name and are not.
  sweep("nd", 1232, nd.CrossSectionTable());
  sweep("dd", 1232, dd.CrossSectionTable());
  for (const char* m : kDeltastar) {
    sweep("ndstar", std::atoi(m), ndstar.CrossSectionTable(std::string("delta(") + m + ")+"));
    sweep("ddstar", std::atoi(m), ddstar.CrossSectionTable(std::string("delta(") + m + ")+"));
  }
  for (const char* m : kNstar) {
    sweep("nnstar", std::atoi(m), nnstar.CrossSectionTable(std::string("N(") + m + ")+"));
    sweep("dnstar", std::atoi(m), dnstar.CrossSectionTable(std::string("N(") + m + ")+"));
  }

  // The phase-space integral, for every resonance the class dispatches on. The grid is the
  // 120-point table's own energies plus their midpoints, plus one point past the top - where
  // the loop's `ie < 119` makes the function extrapolate rather than clamp.
  const char* kDbiNames[] = {
    // The ground-state Delta is named "delta+", with no mass in the name - only the excited
    // states carry one. The first version asked for "delta(1232)+" and got a null definition,
    // which is why the dump writes MISSING rather than skipping: a resonance that cannot be
    // found is a row in the oracle, not an absence.
    "delta+", "delta(1600)+", "delta(1620)+", "delta(1700)+", "delta(1900)+",
    "delta(1905)+", "delta(1910)+", "delta(1920)+", "delta(1930)+", "delta(1950)+",
    "N(1440)+", "N(1520)+", "N(1535)+", "N(1650)+", "N(1675)+", "N(1680)+", "N(1700)+",
    "N(1710)+", "N(1720)+", "N(1900)+", "N(1990)+", "N(2090)+", "N(2190)+", "N(2220)+",
    "N(2250)+"};
  for (const char* nm : kDbiNames) {
    G4ParticleDefinition* def = ptable->FindParticle(nm);
    if (def == nullptr) {
      std::fprintf(g, "%s,MISSING,0\n", nm);
      continue;
    }
    G4DetailedBalancePhaseSpaceIntegral integral(def);
    // Fine over the region a cascade actually reaches - 1 to 4 GeV - and then coarse to 61 GeV.
    // The coarse arm is not decoration: the class's search loop is `for (ie = 0; ie < 119; ie++)`,
    // so the LAST of its 120 grid points is never taken as a left edge and above 48.109 GeV the
    // function extrapolates along the last interval instead of clamping. A grid that stopped at
    // 4 GeV cannot see that, and the first version of this dump stopped at 4 GeV: changing the
    // bound to 120 then changed nothing.
    for (int i = 0; i < 240; ++i) {
      const double s = (1.0 + 0.0125 * i) * CLHEP::GeV;
      std::fprintf(g, "%s,%.17g,%.17g\n", nm, s, integral.GetPhaseSpaceIntegral(s));
    }
    for (int i = 0; i < 120; ++i) {
      const double s = (4.0 + 0.475 * i) * CLHEP::GeV;
      std::fprintf(g, "%s,%.17g,%.17g\n", nm, s, integral.GetPhaseSpaceIntegral(s));
    }
  }
  std::fclose(f);
  std::fclose(g);
}

/// G4Clebsch over every isospin combination the binary cascade's channels can form, and G4Pow's
/// log-factorial table that they are all built on.
///
/// The ranges are chosen from what the resonance channels actually pass:
/// `G4VXResonance::IsospinCorrection` calls `Weight(isoIn1, iso3In1, isoIn2, iso3In2, isoOut1,
/// isoOut2)` with isospins of 1 (nucleon), 2 (pion) and 3 (Delta), and `ClebschGordan` is reached
/// with total isospins up to their sum. The sweep goes past that on both sides so that a
/// transcription which is right on the used set and wrong just outside it is still caught.
///
/// The sweep raises NO G4Exception at all, which was measured rather than assumed: an earlier
/// version of this dump installed a handler to swallow the JustWarning banners the rejected
/// combinations were expected to print, and the handler counted zero. It also crashed the whole
/// oracle run, because `G4VExceptionHandler`'s constructor installs itself into the state
/// manager - so the "previous handler" read after constructing it IS it, restoring that put the
/// doomed object back, and the `delete` left the state manager holding freed memory for the next
/// module's first warning to dereference. Every bic CSV was written correctly and the program
/// died between the last fclose and the registry's "wrote" line, with exit code 1 and no message.
/// The handler is gone; this note is what is left of it.
void write_imr_clebsch() {
  FILE* f = std::fopen("bic_imr_clebsch.csv", "w");
  std::fprintf(f, "kind,j1,m1,j2,m2,j3,j4,value\n");


  // logfactorial itself, which everything else is differences of.
  for (int z = 0; z < 512; ++z) {
    std::fprintf(f, "logfact,%d,0,0,0,0,0,%.17g\n", z, G4Pow::GetInstance()->logfactorial(z));
  }
  // TriangleCoeff over every triad up to 2J = 8.
  for (int a = 0; a <= 8; ++a) {
    for (int b = 0; b <= 8; ++b) {
      for (int c = 0; c <= 8; ++c) {
        std::fprintf(f, "triangle,%d,0,%d,0,%d,0,%.17g\n", a, b, c,
                     G4Clebsch::TriangleCoeff(a, b, c));
      }
    }
  }
  // The coefficient and its square, over the whole (2J1, 2M1, 2J2, 2M2, 2J) box. The M values
  // run past their J on purpose: the first two `if`s in ClebschGordanCoeff are what reject them
  // and a port that dropped either would return a number where Geant4 returns zero.
  for (int j1 = 0; j1 <= 4; ++j1) {
    for (int m1 = -j1 - 2; m1 <= j1 + 2; ++m1) {
      for (int j2 = 0; j2 <= 4; ++j2) {
        for (int m2 = -j2 - 2; m2 <= j2 + 2; ++m2) {
          for (int j = 0; j <= 8; ++j) {
            std::fprintf(f, "coeff,%d,%d,%d,%d,%d,0,%.17g\n", j1, m1, j2, m2, j,
                         G4Clebsch::ClebschGordanCoeff(j1, m1, j2, m2, j));
            std::fprintf(f, "cg,%d,%d,%d,%d,%d,0,%.17g\n", j1, m1, j2, m2, j,
                         G4Clebsch::ClebschGordan(j1, m1, j2, m2, j));
          }
        }
      }
    }
  }
  // Weight, over the same box with both outgoing isospins.
  for (int j1 = 0; j1 <= 3; ++j1) {
    for (int m1 = -j1; m1 <= j1; m1 += 2) {
      for (int j2 = 0; j2 <= 3; ++j2) {
        for (int m2 = -j2; m2 <= j2; m2 += 2) {
          for (int o1 = 0; o1 <= 3; ++o1) {
            for (int o2 = 0; o2 <= 3; ++o2) {
              std::fprintf(f, "weight,%d,%d,%d,%d,%d,%d,%.17g\n", j1, m1, j2, m2, o1, o2,
                           G4Clebsch::Weight(j1, m1, j2, m2, o1, o2));
            }
          }
        }
      }
    }
  }
  std::fclose(f);
}

/// The meson-baryon ELASTIC channel: G4XAqmTotal, G4XAqmElastic, G4XMesonBaryonElastic, and
/// G4CollisionMesonBaryonElastic's IsInCharge, CrossSection and FinalState.
///
/// The pairs include a pion on a DELTA as well as on a nucleon, because
/// `G4CollisionMesonBaryonElastic::IsInCharge` is by parton count and accepts one - it is the only
/// way a short-lived particle is ever an incoming track in G4Scatterer's tree.
void write_imr_meson() {
  FILE* f = std::fopen("bic_imr_meson.csv", "w");
  std::fprintf(f,
               "pair,pdg1,pdg2,m1,m2,sqrt_s_MeV,in1x,in1y,in1z,in1e,in2x,in2y,in2z,in2e,"
               "in_charge,aqm_total_mb,aqm_elastic,sigma_mb\n");
  FILE* g = std::fopen("bic_imr_meson_fs.csv", "w");
  std::fprintf(g,
               "pair,sqrt_s_MeV,in1x,in1y,in1z,in1e,in2x,in2y,in2z,in2e,phase,empty,"
               "p1x,p1y,p1z,p1e,p2x,p2y,p2z,p2e,draws\n");

  G4ShortLivedConstructor shortLived;
  shortLived.ConstructParticle();
  G4ParticleTable* ptable = G4ParticleTable::GetParticleTable();

  struct MPair { const char* name; const G4ParticleDefinition* a; const G4ParticleDefinition* b; };
  std::vector<MPair> pairs;
  pairs.push_back({"pip_p", G4PionPlus::PionPlusDefinition(), G4Proton::ProtonDefinition()});
  pairs.push_back({"pim_p", G4PionMinus::PionMinusDefinition(), G4Proton::ProtonDefinition()});
  pairs.push_back({"pi0_n", G4PionZero::PionZeroDefinition(), G4Neutron::NeutronDefinition()});
  pairs.push_back({"p_pip", G4Proton::ProtonDefinition(), G4PionPlus::PionPlusDefinition()});
  // A pion on a Delta(1232)+ and on an N(1440)+, which the parton-count test accepts and the
  // to-resonance channel's generic-type test does not.
  if (ptable->FindParticle(2214) != nullptr) {
    pairs.push_back({"pip_delta", G4PionPlus::PionPlusDefinition(), ptable->FindParticle(2214)});
  }
  if (ptable->FindParticle(12112) != nullptr) {
    pairs.push_back({"pim_n1440", G4PionMinus::PionMinusDefinition(),
                     ptable->FindParticle(12112)});
  }
  // A nucleon pair, which the parton-count test must REJECT (3 and 3).
  pairs.push_back({"p_n", G4Proton::ProtonDefinition(), G4Neutron::NeutronDefinition()});

  G4XAqmTotal aqmTotal;
  G4XAqmElastic aqmElastic;
  G4XMesonBaryonElastic mbElastic;
  G4CollisionMesonBaryonElastic mbChannel;

  auto* eng = new ImrCycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);

  for (const MPair& pr : pairs) {
    const double m1 = pr.a->GetPDGMass();
    const double m2 = pr.b->GetPDGMass();
    for (int tilt = 0; tilt < 2; ++tilt) {
      for (double t = 20.0; t <= 1500.0; t += 20.0) {
        // Particle 1 carrying kinetic energy t, particle 2 at rest or with its own momentum.
        const double e1 = t + m1;
        G4ThreeVector v1(0, 0, std::sqrt(e1 * e1 - m1 * m1));
        G4LorentzVector q2(G4ThreeVector(0, 0, 0), m2);
        if (tilt != 0) {
          v1.rotateY(0.7);
          v1.rotateZ(1.3);
          const G4ThreeVector v2(31.0, -44.0, 19.0);
          q2 = G4LorentzVector(v2, std::sqrt(v2.mag2() + m2 * m2));
        }
        const G4LorentzVector q1(v1, e1);
        G4KineticTrack t1(pr.a, 0.0, G4ThreeVector(0, 0, 0), q1);
        G4KineticTrack t2(pr.b, 0.0, G4ThreeVector(0, 0, 0), q2);
        const double s = (t1.Get4Momentum() + t2.Get4Momentum()).mag();
        const bool inCharge = mbChannel.IsInCharge(t1, t2);
        std::fprintf(f,
                     "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                     "%.17g,%d,%.17g,%.17g,%.17g\n",
                     pr.name, pr.a->GetPDGEncoding(), pr.b->GetPDGEncoding(), m1, m2, s, q1.x(),
                     q1.y(), q1.z(), q1.t(), q2.x(), q2.y(), q2.z(), q2.t(), inCharge ? 1 : 0,
                     aqmTotal.CrossSection(t1, t2) / millibarn,
                     aqmElastic.CrossSection(t1, t2),
                     inCharge ? mbElastic.CrossSection(t1, t2) / millibarn : 0.0);
        if (!inCharge) { continue; }
        for (int phase = 0; phase < 8; ++phase) {
          eng->reset(phase);
          G4KineticTrackVector* fs = mbChannel.FinalState(t1, t2);
          std::fprintf(g, "%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,",
                       pr.name, s, q1.x(), q1.y(), q1.z(), q1.t(), q2.x(), q2.y(), q2.z(),
                       q2.t(), phase);
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
  CLHEP::HepRandom::setTheEngine(saved);
  delete eng;
  std::fclose(f);
  std::fclose(g);
}

/// G4XResonance::CrossSection through the concrete channels that construct it, and the isospin
/// quantum numbers every one of them is scaled by.
///
/// The channels are built exactly as the six `G4CollisionNNTo*` constructors build them -
/// `G4ConcreteNNToNDelta(in1, in2, out1, out2)` and its five siblings - so what is dumped is the
/// whole of `G4VCollision::CrossSection` over `G4XResonance`: the table lookup, the isospin
/// correction, and the `IsShortLived` test that decides whether detailed balance is applied.
/// Every entrance pair here is two nucleons, which is the only kind
/// `G4ConcreteNNTwoBodyResonance::IsInCharge` accepts, so that test is always false - and this
/// dump is the measurement that says so.
void write_imr_resonance_xsec() {
  FILE* f = std::fopen("bic_imr_resxsec.csv", "w");
  std::fprintf(f, "family,mass,in1,in2,out1,out2,sqrt_s_MeV,in_charge,sigma_mb\n");
  FILE* g = std::fopen("bic_imr_species.csv", "w");
  std::fprintf(g, "pdg,name,mass,width,iso,iso3,ispin,shortlived,baryon,charge\n");

  G4ShortLivedConstructor shortLived;
  shortLived.ConstructParticle();
  G4ParticleTable* ptable = G4ParticleTable::GetParticleTable();
  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* n = G4Neutron::NeutronDefinition();

  // Every species the six families put in or out, by the codes G4HadParticleCodes.hh declares.
  const int kSpecies[] = {
      2212, 2112,
      1114, 2114, 2214, 2224,
      31114, 32114, 32214, 32224,  1112, 1212, 2122, 2222,
      11114, 12114, 12214, 12224,  11112, 11212, 12122, 12222,
      1116, 1216, 2126, 2226,      21112, 21212, 22122, 22222,
      21114, 22114, 22214, 22224,  11116, 11216, 12126, 12226,
      1118, 2118, 2218, 2228,
      12212, 12112, 2124, 1214, 22212, 22112, 32212, 32112, 2216, 2116,
      12216, 12116, 22124, 21214, 42212, 42112, 32124, 31214, 42124, 41214,
      12218, 12118, 52214, 52114, 2128, 1218, 100002210, 100002110,
      100012210, 100012110};

  for (int code : kSpecies) {
    const G4ParticleDefinition* d = ptable->FindParticle(code);
    if (d == nullptr) {
      std::fprintf(g, "%d,MISSING,0,0,0,0,0,0,0,0\n", code);
      continue;
    }
    std::fprintf(g, "%d,%s,%.17g,%.17g,%d,%d,%d,%d,%d,%.17g\n", code,
                 d->GetParticleName().c_str(), d->GetPDGMass(), d->GetPDGWidth(),
                 d->GetPDGiIsospin(), d->GetPDGiIsospin3(), d->GetPDGiSpin(),
                 d->IsShortLived() ? 1 : 0, d->GetBaryonNumber(),
                 d->GetPDGCharge() / CLHEP::eplus);
  }

  // One representative channel from each of the six families, in the exact form its own
  // constructor builds: the entrance pair, the exit pair, and the table class that goes with it.
  struct Chan {
    const char* family;
    int mass;
    const G4ParticleDefinition* i1;
    const G4ParticleDefinition* i2;
    int o1;
    int o2;
    int kind;  // 0 NDelta, 1 DeltaDelta, 2 NDeltastar, 3 DeltaDeltastar, 4 NNstar, 5 DeltaNstar
  };
  std::vector<Chan> chans;
  // MakeNNToNDelta's six, for Delta(1232).
  chans.push_back({"nd", 1232, n, n, 2112, 2114, 0});
  chans.push_back({"nd", 1232, n, n, 2212, 1114, 0});
  chans.push_back({"nd", 1232, n, p, 2212, 2114, 0});
  chans.push_back({"nd", 1232, n, p, 2112, 2214, 0});
  chans.push_back({"nd", 1232, p, p, 2112, 2224, 0});
  chans.push_back({"nd", 1232, p, p, 2212, 2214, 0});
  // G4CollisionNNToDeltaDelta's explicit six.
  chans.push_back({"dd", 1232, n, n, 2114, 2114, 1});
  chans.push_back({"dd", 1232, n, n, 1114, 2214, 1});
  chans.push_back({"dd", 1232, n, p, 2114, 2214, 1});
  chans.push_back({"dd", 1232, n, p, 1114, 2224, 1});
  chans.push_back({"dd", 1232, p, p, 2214, 2214, 1});
  chans.push_back({"dd", 1232, p, p, 2114, 2224, 1});
  // MakeNNToNDelta again, for two Delta* multiplets.
  chans.push_back({"ndstar", 1600, n, n, 2112, 32114, 2});
  chans.push_back({"ndstar", 1600, p, p, 2112, 32224, 2});
  chans.push_back({"ndstar", 1950, n, p, 2212, 2118, 2});
  chans.push_back({"ndstar", 1950, p, p, 2212, 2218, 2});
  // MakeNNToDeltaDelta, for a Delta* multiplet - Delta(1232) x Delta*.
  chans.push_back({"ddstar", 1600, n, n, 1114, 32214, 3});
  chans.push_back({"ddstar", 1600, n, p, 2214, 32114, 3});
  chans.push_back({"ddstar", 1950, p, p, 2224, 2118, 3});
  // MakeNNToNNStar's four, for two N* multiplets.
  chans.push_back({"nnstar", 1440, n, n, 2112, 12112, 4});
  chans.push_back({"nnstar", 1440, p, p, 2212, 12212, 4});
  chans.push_back({"nnstar", 1440, n, p, 2112, 12212, 4});
  chans.push_back({"nnstar", 1440, n, p, 2212, 12112, 4});
  chans.push_back({"nnstar", 2250, p, p, 2212, 100012210, 4});
  // MakeNNToDeltaNstar's six, for one N* multiplet.
  chans.push_back({"dnstar", 1440, n, n, 2114, 12112, 5});
  chans.push_back({"dnstar", 1440, n, n, 1114, 12212, 5});
  chans.push_back({"dnstar", 1440, p, p, 2214, 12212, 5});
  chans.push_back({"dnstar", 1440, p, p, 2224, 12112, 5});
  chans.push_back({"dnstar", 1440, n, p, 2114, 12212, 5});
  chans.push_back({"dnstar", 1440, n, p, 2214, 12112, 5});

  for (const Chan& c : chans) {
    const G4ParticleDefinition* o1 = ptable->FindParticle(c.o1);
    const G4ParticleDefinition* o2 = ptable->FindParticle(c.o2);
    if (o1 == nullptr || o2 == nullptr) { continue; }
    G4VCollision* ch = nullptr;
    switch (c.kind) {
      case 0: ch = new G4ConcreteNNToNDelta(c.i1, c.i2, o1, o2); break;
      case 1: ch = new G4ConcreteNNToDeltaDelta(c.i1, c.i2, o1, o2); break;
      case 2: ch = new G4ConcreteNNToNDeltaStar(c.i1, c.i2, o1, o2); break;
      case 3: ch = new G4ConcreteNNToDeltaDeltastar(c.i1, c.i2, o1, o2); break;
      case 4: ch = new G4ConcreteNNToNNStar(c.i1, c.i2, o1, o2); break;
      default: ch = new G4ConcreteNNToDeltaNstar(c.i1, c.i2, o1, o2); break;
    }
    for (double want = 1900.0; want <= 6000.0; want += 25.0) {
      G4LorentzVector q1, q2;
      imr_make_pair(c.i1, c.i2, want, q1, q2);
      G4KineticTrack t1(c.i1, 0.0, G4ThreeVector(0, 0, 0), q1);
      G4KineticTrack t2(c.i2, 0.0, G4ThreeVector(0, 0, 0), q2);
      const double s = (t1.Get4Momentum() + t2.Get4Momentum()).mag();
      const bool inCharge = ch->IsInCharge(t1, t2);
      std::fprintf(f, "%s,%d,%d,%d,%d,%d,%.17g,%d,%.17g\n", c.family, c.mass,
                   c.i1->GetPDGEncoding(), c.i2->GetPDGEncoding(), c.o1, c.o2, s,
                   inCharge ? 1 : 0, inCharge ? ch->CrossSection(t1, t2) / millibarn : 0.0);
    }
    delete ch;
  }
  std::fclose(f);
  std::fclose(g);
}

/// The eight partial cross sections `G4CollisionComposite::FinalState` selects on, straight out
/// of `G4CollisionNN`'s own components, and the selection itself.
///
/// `G4CollisionNN::GetComponents()` returns a null pointer (see collision_nn.cuh), so the
/// components cannot be reached through the composite - they are rebuilt here in the same order
/// its `GROUP8` registers them, which is also the order `FinalState` accumulates in. The two
/// elastic ones answer directly and the six resonance ones from their 32-point buffers, so what
/// is compared is the whole nesting: 306 concrete channels, nine middle-layer buffers twice over,
/// and the 0.01 mb floor applied once per buffer.
void write_imr_nnchannels() {
  FILE* f = std::fopen("bic_imr_nnpartial.csv", "w");
  // The input four-momenta, not just sqrt(s). Particle 1 is along +z and particle 2 at rest, so
  // three numbers are enough - and they are needed, because rebuilding the pair by inverting
  // `s = m1^2 + m2^2 + 2 m2 E1` lands an ulp or two away, and the buffer's interpolation slope
  // for the N-Delta channel just above threshold is 0.18 mb per MeV, which turns that ulp into
  // 4e-14 of the answer. The same mistake in the same shape as the first bic_imr_collision dump.
  std::fprintf(f, "pair,sqrt_s_MeV,in1z,in1e,in2e,component,sigma_mb,total_mb\n");
  FILE* g = std::fopen("bic_imr_nnselect.csv", "w");
  std::fprintf(g, "pair,sqrt_s_MeV,in1z,in1e,in2e,phase,selected,draws\n");

  G4ShortLivedConstructor shortLived;
  shortLived.ConstructParticle();
  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* n = G4Neutron::NeutronDefinition();
  struct Pair { const char* name; const G4ParticleDefinition* a; const G4ParticleDefinition* b; };
  const Pair pairs[] = {{"pp", p, p}, {"nn", n, n}, {"np", n, p}, {"pn", p, n}};

  // The eight components, rebuilt in G4CollisionNN's GROUP8 order.
  std::vector<G4VCollision*> comps;
  comps.push_back(new G4CollisionnpElastic());
  comps.push_back(new G4CollisionNNElastic());
  comps.push_back(new G4CollisionNNToNDelta());
  comps.push_back(new G4CollisionNNToDeltaDelta());
  comps.push_back(new G4CollisionNNToNDeltastar());
  comps.push_back(new G4CollisionNNToDeltaDeltastar());
  comps.push_back(new G4CollisionNNToNNstar());
  comps.push_back(new G4CollisionNNToDeltaNstar());

  // The buffer's own 32-point grid, rebuilt exactly as G4CollisionComposite::BufferCrossSection
  // builds it - the kinetic energy on the lighter of the two definitions - together with each
  // component's cross section AT those tracks. `theBuffer` is private and there is no accessor,
  // so this is the only way to see the nodes; and the nodes are what the interpolation between
  // them amplifies, so a disagreement in the interpolated value has to be separable from a
  // disagreement in the grid.
  FILE* h = std::fopen("bic_imr_nnbuffer.csv", "w");
  std::fprintf(h, "pair,point,T_GeV,sqrt_s_MeV,component,sigma_mb\n");
  {
    const G4double kT32[32] = {.01, .03, .05, .1,  .15, .2,  .3,  .4,  .5,  .6, .7,
                               .8,  .9,  1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0,
                               3.5, 4.0, 5.0, 6.0, 8.0, 10., 15,  20,  50,  100};
    for (const Pair& pr : pairs) {
      for (int tt = 0; tt < 32; ++tt) {
        const G4double aT = kT32[tt] * CLHEP::GeV;
        const G4double aM = pr.a->GetPDGMass();
        const G4double bM = pr.b->GetPDGMass();
        G4double aE = aM;
        G4double bE = bM;
        G4ThreeVector aMom(0, 0, 0);
        G4ThreeVector bMom(0, 0, 0);
        if (aM <= bM) {
          aE += aT;
          aMom = G4ThreeVector(0, 0, std::sqrt(aE * aE - aM * aM));
        } else {
          bE += aT;
          bMom = G4ThreeVector(0, 0, std::sqrt(bE * bE - bM * bM));
        }
        const G4LorentzVector a4(aE, aMom);
        const G4LorentzVector b4(bE, bMom);
        G4KineticTrack a(pr.a, 0.0, G4ThreeVector(0, 0, 0), const_cast<G4LorentzVector&>(a4));
        G4KineticTrack b(pr.b, 0.0, G4ThreeVector(0, 0, 0), const_cast<G4LorentzVector&>(b4));
        const G4double sqrts = (a4 + b4).mag();
        for (int i = 0; i < 8; ++i) {
          const double x = comps[i]->IsInCharge(a, b) ? comps[i]->CrossSection(a, b) : 0.0;
          std::fprintf(h, "%s,%d,%.17g,%.17g,%d,%.17g\n", pr.name, tt, kT32[tt], sqrts, i,
                       x / millibarn);
        }
      }
    }
  }
  std::fclose(h);

  auto* eng = new ImrCycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();

  for (const Pair& pr : pairs) {
    for (double want = 1880.0; want <= 6000.0; want += 20.0) {
      G4LorentzVector q1, q2;
      imr_make_pair(pr.a, pr.b, want, q1, q2);
      G4KineticTrack t1(pr.a, 0.0, G4ThreeVector(0, 0, 0), q1);
      G4KineticTrack t2(pr.b, 0.0, G4ThreeVector(0, 0, 0), q2);
      const double s = (t1.Get4Momentum() + t2.Get4Momentum()).mag();
      double partial[8];
      for (int i = 0; i < 8; ++i) {
        partial[i] = comps[i]->IsInCharge(t1, t2) ? comps[i]->CrossSection(t1, t2) : 0.0;
        std::fprintf(f, "%s,%.17g,%.17g,%.17g,%.17g,%d,%.17g\n", pr.name, s, q1.z(), q1.t(),
                     q2.t(), i, partial[i] / millibarn);
      }
      // The selection, under the prescribed cycle. One uniform, the running sum, and the index
      // of the component it lands in - which is what decides whether a collision is elastic or
      // makes a resonance.
      CLHEP::HepRandom::setTheEngine(eng);
      for (int phase = 0; phase < 8; ++phase) {
        eng->reset(phase);
        double sum = 0.0;
        for (int i = 0; i < 8; ++i) { sum += partial[i]; }
        const double random = G4UniformRand() * sum;
        double running = 0.0;
        int selected = -1;
        for (int i = 0; i < 8; ++i) {
          running += partial[i];
          if (running > random) { selected = i; break; }
        }
        std::fprintf(g, "%s,%.17g,%.17g,%.17g,%.17g,%d,%d,%d\n", pr.name, s, q1.z(), q1.t(),
                     q2.t(), phase, selected, eng->draws());
      }
      CLHEP::HepRandom::setTheEngine(saved);
    }
  }
  for (auto* c : comps) { delete c; }
  delete eng;
  std::fclose(f);
  std::fclose(g);
}

/// G4VScatteringCollision::FinalState through the concrete channels that inherit it - the two
/// outgoing four-momenta with the resonance masses sampled from a Breit-Wigner - and
/// G4VAnnihilationCollision::FinalState, which G4ConcreteMesonBaryonToResonance inherits.
///
/// The number of uniforms each call consumes is dumped with the momenta, because it is what the
/// cascade's random stream depends on and because it varies: a stable product draws none for its
/// mass, a short-lived one draws exactly one, and the zero-width shortcut draws none either.
void write_imr_resonance_fs() {
  FILE* f = std::fopen("bic_imr_resfs.csv", "w");
  std::fprintf(f,
               "family,in1,in2,out1,out2,sqrt_s_MeV,in1z,in1e,in2e,phase,n,"
               "p1x,p1y,p1z,p1e,p2x,p2y,p2z,p2e,draws\n");
  FILE* g = std::fopen("bic_imr_annihfs.csv", "w");
  std::fprintf(g, "name,sqrt_s_MeV,in1z,in1e,in2e,p1x,p1y,p1z,p1e,draws\n");

  G4ShortLivedConstructor shortLived;
  shortLived.ConstructParticle();
  G4ParticleTable* ptable = G4ParticleTable::GetParticleTable();
  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* n = G4Neutron::NeutronDefinition();

  struct Chan {
    const char* family;
    const G4ParticleDefinition* i1;
    const G4ParticleDefinition* i2;
    int o1;
    int o2;
    int kind;
  };
  std::vector<Chan> chans;
  // One nucleon-plus-resonance channel and one resonance-pair channel from each family, so that
  // both the one-sample and the two-sample paths are exercised - and the delta- specifically,
  // whose width is 117 MeV where its three partners are 120.
  chans.push_back({"nd", p, p, 2112, 2224, 0});
  chans.push_back({"nd", n, n, 2212, 1114, 0});     // a delta-
  chans.push_back({"dd", n, n, 1114, 2214, 1});     // two short-lived
  chans.push_back({"dd", p, p, 2214, 2214, 1});
  chans.push_back({"ndstar", n, p, 2212, 32114, 2});
  chans.push_back({"ndstar", p, p, 2212, 2218, 2});
  chans.push_back({"ddstar", n, n, 1114, 32214, 3});
  chans.push_back({"ddstar", p, p, 2224, 2118, 3});
  chans.push_back({"nnstar", p, p, 2212, 12212, 4});
  chans.push_back({"nnstar", n, n, 2112, 100012110, 4});
  chans.push_back({"dnstar", n, n, 1114, 12212, 5});
  chans.push_back({"dnstar", p, p, 2224, 12112, 5});

  auto* eng = new ImrCycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);

  for (const Chan& c : chans) {
    const G4ParticleDefinition* o1 = ptable->FindParticle(c.o1);
    const G4ParticleDefinition* o2 = ptable->FindParticle(c.o2);
    if (o1 == nullptr || o2 == nullptr) { continue; }
    G4VCollision* ch = nullptr;
    switch (c.kind) {
      case 0: ch = new G4ConcreteNNToNDelta(c.i1, c.i2, o1, o2); break;
      case 1: ch = new G4ConcreteNNToDeltaDelta(c.i1, c.i2, o1, o2); break;
      case 2: ch = new G4ConcreteNNToNDeltaStar(c.i1, c.i2, o1, o2); break;
      case 3: ch = new G4ConcreteNNToDeltaDeltastar(c.i1, c.i2, o1, o2); break;
      case 4: ch = new G4ConcreteNNToNNStar(c.i1, c.i2, o1, o2); break;
      default: ch = new G4ConcreteNNToDeltaNstar(c.i1, c.i2, o1, o2); break;
    }
    for (double want = 2200.0; want <= 6000.0; want += 100.0) {
      G4LorentzVector q1, q2;
      imr_make_pair(c.i1, c.i2, want, q1, q2);
      G4KineticTrack t1(c.i1, 0.0, G4ThreeVector(0, 0, 0), q1);
      G4KineticTrack t2(c.i2, 0.0, G4ThreeVector(0, 0, 0), q2);
      const double s = (t1.Get4Momentum() + t2.Get4Momentum()).mag();
      for (int phase = 0; phase < 8; ++phase) {
        eng->reset(phase);
        G4KineticTrackVector* fs = ch->FinalState(t1, t2);
        const int nprod = (fs == nullptr) ? 0 : static_cast<int>(fs->size());
        std::fprintf(f, "%s,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%d,", c.family,
                     c.i1->GetPDGEncoding(), c.i2->GetPDGEncoding(), c.o1, c.o2, s, q1.z(),
                     q1.t(), q2.t(), phase, nprod);
        if (nprod < 2) {
          std::fprintf(f, "0,0,0,0,0,0,0,0,%d\n", eng->draws());
        } else {
          const G4LorentzVector a = (*fs)[0]->Get4Momentum();
          const G4LorentzVector b = (*fs)[1]->Get4Momentum();
          std::fprintf(f, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n", a.x(), a.y(),
                       a.z(), a.t(), b.x(), b.y(), b.z(), b.t(), eng->draws());
        }
        if (fs != nullptr) {
          for (auto* kt : *fs) { delete kt; }
          delete fs;
        }
      }
    }
    delete ch;
  }

  // G4VAnnihilationCollision::FinalState, through G4ConcreteMesonBaryonToResonance - the only
  // subclass the binary cascade reaches. It draws NO uniform.
  {
    const G4ParticleDefinition* pip = G4PionPlus::PionPlusDefinition();
    // The partial-width LABEL is not decoration: G4XAnnihilationChannel looks it up in
    // G4ResonancePartialWidth and stores whatever comes back, so a label that is not in the
    // table leaves a null pointer for the cross section to dereference. The first version of this
    // block passed "dump" and killed the whole oracle run after writing 69 of its 112 rows.
    // The PION CHARGE has to match the resonance's isospin. `GetOutgoingParticle` adds the two
    // incoming iso3 values and asks G4ParticleTypeConverter for the state of the outgoing
    // multiplet with that iso3 - and if there is none it prints one line and THROWS a
    // G4HadronicException. A pi+ on a proton is iso3 = +3, which a Delta (isospin 3/2) has and an
    // N* (isospin 1/2) does not, so pi+ p -> N* kills the program. The first version of this
    // block asked for exactly that and took the whole oracle run down with it.
    struct Res { int code; const char* label; int pion; };
    const Res kRes[] = {{2214, "D1232_Npi", 211}, {32214, "D1600_Npi", 211},
                        {12212, "N1440_Npi", -211}};
    for (const Res& r : kRes) {
      const int code = r.code;
      const G4ParticleDefinition* res = ptable->FindParticle(code);
      if (res == nullptr) { continue; }
      const G4ParticleDefinition* pion =
          (r.pion > 0) ? pip : static_cast<const G4ParticleDefinition*>(
                                   G4PionMinus::PionMinusDefinition());
      G4ConcreteMesonBaryonToResonance mb(p, pion, res, r.label);
      for (double want = 1200.0; want <= 3000.0; want += 50.0) {
        G4LorentzVector q1, q2;
        imr_make_pair(pion, p, want, q1, q2);
        G4KineticTrack t1(pion, 0.0, G4ThreeVector(0, 0, 0), q1);
        G4KineticTrack t2(p, 0.0, G4ThreeVector(0, 0, 0), q2);
        const double s = (t1.Get4Momentum() + t2.Get4Momentum()).mag();
        eng->reset(0);
        G4KineticTrackVector* fs = mb.FinalState(t1, t2);
        if (fs != nullptr && !fs->empty()) {
          const G4LorentzVector a = (*fs)[0]->Get4Momentum();
          std::fprintf(g, "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n", code, s,
                       q1.z(), q1.t(), q2.t(), a.x(), a.y(), a.z(), a.t(), eng->draws());
        }
        if (fs != nullptr) {
          for (auto* kt : *fs) { delete kt; }
          delete fs;
        }
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  delete eng;
  std::fclose(f);
  std::fclose(g);
}

/// G4XAnnihilationChannel::CrossSection through every one of the 25 channels
/// G4CollisionMesonBaryonToResonance builds, and the two mass-dependent width tables underneath
/// it - the only cross section a pion in the binary cascade has.
///
/// The pion charge is chosen per resonance so that the outgoing iso3 exists: a Delta (isospin
/// 3/2) can take a pi+ on a proton and an N* (isospin 1/2) cannot, and asking for the impossible
/// one throws (docs/RISK.md V110).
void write_imr_annih() {
  FILE* f = std::fopen("bic_imr_annih.csv", "w");
  std::fprintf(f,
               "label,res_pdg,pion_pdg,sqrt_s_MeV,in1z,in1e,in2e,in_charge,sigma_mb,"
               "width_MeV,partial_MeV\n");

  G4ShortLivedConstructor shortLived;
  shortLived.ConstructParticle();
  G4ParticleTable* ptable = G4ParticleTable::GetParticleTable();
  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* pip = G4PionPlus::PionPlusDefinition();
  const G4ParticleDefinition* pim = G4PionMinus::PionMinusDefinition();

  struct R { int code; const char* label; bool delta; };
  const R kAll[] = {
      {2214, "D1232_Npi", true},  {32214, "D1600_Npi", true}, {2122, "D1620_Npi", true},
      {12214, "D1700_Npi", true}, {12122, "D1900_Npi", true}, {2126, "D1905_Npi", true},
      {22122, "D1910_Npi", true}, {22214, "D1920_Npi", true}, {12126, "D1930_Npi", true},
      {2218, "D1950_Npi", true},
      {12212, "N1440_Npi", false}, {2124, "N1520_Npi", false}, {22212, "N1535_Npi", false},
      {32212, "N1650_Npi", false}, {2216, "N1675_Npi", false}, {12216, "N1680_Npi", false},
      {22124, "N1700_Npi", false}, {42212, "N1710_Npi", false}, {32124, "N1720_Npi", false},
      {42124, "N1900_Npi", false}, {12218, "N1990_Npi", false}, {52214, "N2090_Npi", false},
      {2128, "N2190_Npi", false},  {100002210, "N2220_Npi", false},
      {100012210, "N2250_Npi", false}};

  G4BaryonWidth theWidth;
  G4BaryonPartialWidth thePartWidth;
  G4ResonanceNames theNames;

  for (const R& r : kAll) {
    const G4ParticleDefinition* res = ptable->FindParticle(r.code);
    if (res == nullptr) { continue; }
    // A Delta takes iso3 = +3 from a pi+ on a proton; an N* needs iso3 = +1, so a pi- on a
    // proton (-2 + 1 = -1) reaches its neutron-like partner instead, and the pi+ would throw.
    // TWO pion charges per Delta channel, not one. With a pi+ on a proton the total iso3 is +3,
    // which only one Clebsch-Gordan path reaches, so the NORMALISATION in
    // NormalizedClebschGordan divides by 1 and the `isoRes < iso3` guard compares 3 with 3 and
    // does not fire - both were MEASURED unobservable on a pi+-only grid. A pi0 on a proton is
    // iso3 = +1, where two paths contribute and the normalisation bites. `IsInCharge` compares
    // G4ParticleTypeConverter GENERIC types, and all three pions are PION, so a pi0 is a
    // configuration the cascade really produces.
    for (int pc = 0; pc < 2; ++pc) {
    const G4ParticleDefinition* pion =
        (pc == 1) ? static_cast<const G4ParticleDefinition*>(G4PionZero::PionZeroDefinition())
                  : (r.delta ? pip : pim);
    if (pc == 1 && !r.delta) { continue; }  // pi0 p is iso3 = +1, which an N* does reach
    const G4ParticleDefinition* out = res;
    G4ConcreteMesonBaryonToResonance ch(p, pion, out, r.label);
    // The two width vectors, straight from the tables, so a disagreement in the cross section
    // can be traced to one of the three factors rather than to their product.
    const G4String shortName = theNames.ShortName(res->GetParticleName());
    G4PhysicsVector* wv = theWidth.MassDependentWidth(shortName);
    G4PhysicsVector* pv = thePartWidth.MassDependentWidth(r.label);
    for (double want = 1100.0; want <= 3000.0; want += 25.0) {
      G4LorentzVector q1, q2;
      imr_make_pair(pion, p, want, q1, q2);
      G4KineticTrack t1(pion, 0.0, G4ThreeVector(0, 0, 0), q1);
      G4KineticTrack t2(p, 0.0, G4ThreeVector(0, 0, 0), q2);
      const double s = (t1.Get4Momentum() + t2.Get4Momentum()).mag();
      const bool inCharge = ch.IsInCharge(t1, t2);
      G4bool dummy = false;
      const double w = (wv != nullptr) ? wv->GetValue(s, dummy) : res->GetPDGWidth();
      const double pw = (pv != nullptr) ? pv->GetValue(s, dummy) : res->GetPDGWidth();
      std::fprintf(f, "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g\n", r.label,
                   r.code, pion->GetPDGEncoding(), s, q1.z(), q1.t(), q2.t(), inCharge ? 1 : 0,
                   inCharge ? ch.CrossSection(t1, t2) / millibarn : 0.0, w, pw);
    }
    delete wv;
    delete pv;
    }
  }
  std::fclose(f);
}

/// G4CollisionMesonBaryon's two partial cross sections and the selection between them - the whole
/// of what a pion in the binary cascade does, once G4Scatterer has found the channel.
// G4KineticTrack's resonance machinery, layer one: the DECAY TABLES themselves and the two
// quantities built directly on them.
//
// Geant4 does not hold these in an array a transcription could copy - `G4ShortLivedConstructor`
// builds them out of `G4ExcitedBaryonConstructor` at run time, channel by channel - so they are
// dumped from Geant4's own `G4DecayTable` and the port's copy is checked against this file. That
// is the same standing the isospin list in bic_imr_species.csv has.
//
// The species set is a TRANSITIVE CLOSURE, not a list: it starts from the 25 resonance multiplets
// G4CollisionMesonBaryonToResonance and the 306 G4Concrete* channels can produce, plus the pions
// and nucleons, and follows every decay daughter until nothing new appears. A port that carried
// only the produced set would find a daughter it has no table for the first time a resonance
// decays to another resonance, which N(1520) -> Delta pi does immediately.
void write_imr_decay() {
  FILE* f = std::fopen("bic_imr_decaytable.csv", "w");
  std::fprintf(f, "pdg,name,mass,width,shortlived,charge,baryon,min_mass,n_channels,channel,br,"
                  "n_daughters,d0,d1,d2,d3\n");
  FILE* g = std::fopen("bic_imr_actualwidth.csv", "w");
  std::fprintf(g, "pdg,actual_mass,n_channels,channel,actual_width,total_width\n");

  G4ShortLivedConstructor shortLived;
  shortLived.ConstructParticle();
  G4ParticleTable* ptable = G4ParticleTable::GetParticleTable();
  G4SampleResonance sampler;

  // The seeds: both nucleons, the three pions, and every charge state of the 25 multiplets.
  const int kSeeds[] = {
      2212, 2112, 211, -211, 111,
      1114, 2114, 2214, 2224,
      31114, 32114, 32214, 32224,  1112, 1212, 2122, 2222,
      11114, 12114, 12214, 12224,  11112, 11212, 12122, 12222,
      1116, 1216, 2126, 2226,      21112, 21212, 22122, 22222,
      21114, 22114, 22214, 22224,  11116, 11216, 12126, 12226,
      1118, 2118, 2218, 2228,
      12212, 12112, 2124, 1214, 22212, 22112, 32212, 32112, 2216, 2116,
      12216, 12116, 22124, 21214, 42212, 42112, 32124, 31214, 42124, 41214,
      12218, 12118, 52214, 52114, 2128, 1218, 100002210, 100002110,
      100012210, 100012110,
      // FTFP's 27 short-lived species, from docs/HADRONIC_PLAN.md section 9.3, measured by P11c
      // over 14,000 ftf::apply_yourself events: 14.3% of all tracks and 72.6% of all events.
      // The engine these tables feed is shared - G4DecayKineticTracks is G4KineticTrack::Decay
      // in a loop, and both G4DecayStrongResonances::Propagate and
      // G4GeneratorPrecompoundInterface::Propagate open with it - so the species set has to be
      // the UNION. Sixteen of the 27 are already above; the eleven that are not bring the
      // K*, the Sigma(1385), the Xi(1530), the phi, the f2(1270), the a2(1320) and the
      // anti-Deltas, and with them the first four-body channels in this file.
      113, 213, -213, 223, 313, 323, -313, -323, 333, 225, 115,
      3114, 3214, 3224, -3114, -3214, -3224, 3314, 3324,
      -1114, -2114, -2214, -2224,
      // The two `G4HadronBuilder` can return that P11d found missing from the closure when it
      // audited this table against docs/HADRONIC_PLAN.md section 9: anti-Xi(1530)- and
      // anti-Xi(1530)0. They are the anti-particles of 3314 and 3324 on the line above, which
      // were already here, and P11d measured them at 0 of 161,568 string products - so nothing
      // has depended on them yet, and the point of adding them is that a refusal for a species
      // this table simply forgot is indistinguishable from one for a species nobody produces.
      -3314, -3324};

  std::vector<int> order;
  std::set<int> seen;
  std::vector<int> queue(std::begin(kSeeds), std::end(kSeeds));

  // AND every hadron Geant4 defines whose `IsShortLived()` is FALSE.
  //
  // This is not about decaying them - the engine never does, because `G4DecayKineticTracks`
  // decays only short-lived tracks. It is about the difference between "not short-lived" and
  // "I have never heard of this code". P11d measured 50 codes FTFP produces that were in
  // neither camp: eta' (331), Omega- (3334), the anti-Xis (-3312, -3322) and the charm and
  // bottom hadrons. Geant4 walks past all of them in silence; the port had to set
  // `unknown_species` for them, because a table that does not carry a code cannot say which
  // kind of unknown it is - and a caller obeying that flag would have refused eta' at 0.011%
  // of pion-beam products. With the rows here, the flag means what it says.
  //
  // The rule is `GetParticleType()` in {meson, baryon}, `!IsShortLived()`, and an encoding
  // below 10000 in magnitude, which is what excludes the nuclei and the 100002210-style
  // excited nucleons (the latter are short-lived and already seeds). It is a RULE and not a
  // list of codes, so that nothing here can be mistyped; the set it picks out is QBBC's own,
  // which is the same authority every other table in this file has. Their daughters join the
  // closure like any others.
  {
    G4ParticleTableIterator<G4String, G4ParticleDefinition*>* it = ptable->GetIterator();
    it->reset();
    std::vector<int> longlived;
    while ((*it)()) {
      const G4ParticleDefinition* d = it->value();
      if (d == nullptr || d->IsShortLived()) { continue; }
      const G4String& type = d->GetParticleType();
      if (type != "meson" && type != "baryon") { continue; }
      const int code = d->GetPDGEncoding();
      if (code == 0 || code >= 10000 || code <= -10000) { continue; }
      longlived.push_back(code);
    }
    std::sort(longlived.begin(), longlived.end());
    queue.insert(queue.end(), longlived.begin(), longlived.end());
  }
  // WIDENING THIS SEED LIST, for a caller outside the binary cascade. The engine these tables
  // feed is shared - `G4DecayKineticTracks`, which the Fritiof string model runs over its own
  // products, is `G4KineticTrack::Decay` in a loop - so the species set has to cover whatever
  // that caller produces as well. Add the codes to `kSeeds` above; the closure does the rest.
  //
  // The obvious shortcut, seeding from `G4ParticleTable` with every definition whose
  // `IsShortLived()` is true, does NOT work and was tried: the closure grows from 94 species to
  // at least 117, reaches the ANTI-baryon resonances, and the dumper dies part-way through
  // `anti_N(2250)0`, mid-`fprintf`, with nothing on stderr. Whatever is wrong there is upstream
  // of this file and is not worth chasing to widen a table; an explicit list of the codes a
  // caller actually produces is both safer and a better record of who needs what.

  while (!queue.empty()) {
    const int code = queue.front();
    queue.erase(queue.begin());
    if (seen.count(code) != 0) { continue; }
    seen.insert(code);
    order.push_back(code);
    const G4ParticleDefinition* d = ptable->FindParticle(code);
    if (d == nullptr) { continue; }
    const G4DecayTable* table = d->GetDecayTable();
    if (table == nullptr) { continue; }
    for (int c = 0; c < table->entries(); ++c) {
      G4VDecayChannel* ch = const_cast<G4DecayTable*>(table)->GetDecayChannel(c);
      for (int j = 0; j < ch->GetNumberOfDaughters(); ++j) {
        const G4ParticleDefinition* dd = ch->GetDaughter(j);
        if (dd != nullptr && seen.count(dd->GetPDGEncoding()) == 0) {
          queue.push_back(dd->GetPDGEncoding());
        }
      }
    }
  }

  for (int code : order) {
    const G4ParticleDefinition* d = ptable->FindParticle(code);
    if (d == nullptr) {
      std::fprintf(f, "%d,MISSING,0,0,0,0,0,0,0,-1,0,0,0,0,0,0\n", code);
      continue;
    }
    // GetMinimumMass recurses over the daughters' own tables, keeping only channels whose
    // branching ratio is above 0.10 - and falling back to the single most probable channel when
    // none is. It is dumped for every species, stable ones included, because for those it is
    // simply the PDG mass and a port that recursed there anyway would be caught here.
    const double min_mass = sampler.GetMinimumMass(d);
    const G4DecayTable* table = d->GetDecayTable();
    const int n = (table != nullptr) ? table->entries() : 0;
    if (n == 0) {
      std::fprintf(f, "%d,%s,%.17g,%.17g,%d,%d,%d,%.17g,0,-1,0,0,0,0,0,0\n", code,
                   d->GetParticleName().c_str(), d->GetPDGMass(), d->GetPDGWidth(),
                   d->IsShortLived() ? 1 : 0, G4lrint(d->GetPDGCharge() / CLHEP::eplus),
                   d->GetBaryonNumber(), min_mass);
      continue;
    }
    for (int c = 0; c < n; ++c) {
      G4VDecayChannel* ch = const_cast<G4DecayTable*>(table)->GetDecayChannel(c);
      const int nd = ch->GetNumberOfDaughters();
      int dcode[4] = {0, 0, 0, 0};
      for (int j = 0; j < nd && j < 4; ++j) {
        const G4ParticleDefinition* dd = ch->GetDaughter(j);
        dcode[j] = (dd != nullptr) ? dd->GetPDGEncoding() : 0;
      }
      std::fprintf(f, "%d,%s,%.17g,%.17g,%d,%d,%d,%.17g,%d,%d,%.17g,%d,%d,%d,%d,%d\n", code,
                   d->GetParticleName().c_str(), d->GetPDGMass(), d->GetPDGWidth(),
                   d->IsShortLived() ? 1 : 0, G4lrint(d->GetPDGCharge() / CLHEP::eplus),
                   d->GetBaryonNumber(), min_mass, n, c, ch->GetBR(), nd, dcode[0],
                   dcode[1], dcode[2], dcode[3]);
    }
  }

  // Layer two: theActualWidth[], built in the G4KineticTrack constructor by integrating a
  // Breit-Wigner over each channel's short-lived daughters with G4Integrator::Simpson at 100
  // iterations. GetActualWidth() and GetnChannels() are public, so this reads the array Geant4
  // built rather than a quantity derived from it.
  for (int code : order) {
    const G4ParticleDefinition* d = ptable->FindParticle(code);
    if (d == nullptr || d->GetDecayTable() == nullptr) { continue; }
    // kaon0 is NOT a function of its arguments here. The G4KineticTrack constructor replaces a
    // K0 or an anti-K0 with K0S or K0L on a coin flip before it reads any decay table, so a
    // track built from 311 reports whichever of the two the flip landed on - six channels or
    // two - and consumes one uniform doing it. It is dumped as the substitution it is, below.
    if (code == 311 || code == -311) { continue; }
    const double pole = d->GetPDGMass();
    const double w = d->GetPDGWidth();
    // Actual masses from well below the pole to well above it. The Breit-Wigner tail is what the
    // integrals see, and the width ratio is sensitive to the phase space closing: at an actual
    // mass below the sum of a channel's daughter masses the channel's momentum ratio is zero.
    const double kFrac[] = {0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.25, 1.50};
    for (double fr : kFrac) {
      double m = pole * fr;
      if (w > 0.0 && fr != 1.00) { m = pole + (fr - 1.0) * 6.0 * w; }
      if (m < 1.0) { continue; }
      const G4LorentzVector q(G4ThreeVector(0, 0, 0), m);
      G4KineticTrack kt(d, 0.0, G4ThreeVector(0, 0, 0), q);
      const int n = kt.GetnChannels();
      const G4double* aw = kt.GetActualWidth();
      double total = 0.0;
      for (int c = n - 1; c >= 0; --c) { total += aw[c]; }
      for (int c = 0; c < n; ++c) {
        std::fprintf(g, "%d,%.17g,%d,%d,%.17g,%.17g\n", code, kt.GetActualMass(), n, c, aw[c],
                     total);
      }
    }
  }

  // The K0 substitution itself, under the prescribed engine, so that the one uniform it draws
  // and the identity it lands on are both exact. G4KineticTrack.cc:
  //
  //     if (G4KaonZero::KaonZero() == theDefinition || G4AntiKaonZero::AntiKaonZero() == ...)
  //     { if (G4UniformRand()<0.5) theDefinition = KaonZeroShort; else KaonZeroLong; }
  //
  // The cascade reaches it: N(1650)0, N(1710)0, N(1720)0 and N(1990)0 all have a Lambda K0
  // channel, so a K0 becomes a cascade track the moment one of those decays.
  FILE* k = std::fopen("bic_imr_k0flip.csv", "w");
  std::fprintf(k, "pdg,phase,result_pdg,n_channels,draws\n");
  {
    auto* eng = new ImrCycleEngine();
    CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
    CLHEP::HepRandom::setTheEngine(eng);
    for (int code : {311, -311}) {
      const G4ParticleDefinition* d = ptable->FindParticle(code);
      for (int phase = 0; phase < 8; ++phase) {
        eng->reset(phase);
        const G4LorentzVector q(G4ThreeVector(0, 0, 0), d->GetPDGMass());
        G4KineticTrack kt(d, 0.0, G4ThreeVector(0, 0, 0), q);
        std::fprintf(k, "%d,%d,%d,%d,%d\n", code, phase,
                     kt.GetDefinition()->GetPDGEncoding(), kt.GetnChannels(), eng->draws());
      }
    }
    CLHEP::HepRandom::setTheEngine(saved);
    delete eng;
  }
  std::fclose(k);

  // SampleResidualLifetime, which is what G4BinaryCascade schedules a G4BCDecay at. It is
  // hbar_Planck * (-1/totalWidth) * log(uniform) * the4Momentum.gamma(), so it needs a MOVING
  // track: with the resonance at rest the Lorentz factor is 1 and a port that dropped it
  // entirely would agree everywhere.
  FILE* t = std::fopen("bic_imr_lifetime.csv", "w");
  std::fprintf(t, "pdg,actual_mass,pz,e,gamma,phase,lifetime_ns,total_width,draws\n");
  {
    auto* eng = new ImrCycleEngine();
    CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
    const int kRes[] = {2214, 2224, 1114, 2114, 12212, 2124, 22212, 32212, 12216, 22124,
                        42212, 32124, 2218, 2128, 100012210};
    for (int code : kRes) {
      const G4ParticleDefinition* d = ptable->FindParticle(code);
      if (d == nullptr) { continue; }
      const double pole = d->GetPDGMass();
      for (double fr : {0.90, 1.00, 1.15}) {
        const double m = pole * fr;
        for (double pz : {0.0, 300.0, 1500.0}) {
          const G4LorentzVector q(G4ThreeVector(0, 0, pz), std::sqrt(pz * pz + m * m));
          G4KineticTrack kt(d, 0.0, G4ThreeVector(0, 0, 0), q);
          double total = 0.0;
          const G4double* aw = kt.GetActualWidth();
          for (int c = kt.GetnChannels() - 1; c >= 0; --c) { total += aw[c]; }
          CLHEP::HepRandom::setTheEngine(eng);
          for (int phase = 0; phase < 8; ++phase) {
            eng->reset(phase);
            const double life = kt.SampleResidualLifetime();
            std::fprintf(t, "%d,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%d\n", code,
                         kt.GetActualMass(), q.z(), q.t(), q.gamma(), phase, life, total,
                         eng->draws());
          }
          CLHEP::HepRandom::setTheEngine(saved);
        }
      }
    }
    delete eng;
  }
  std::fclose(t);
  std::fclose(f);
  std::fclose(g);
}

// G4KineticTrack::Decay under the prescribed engine: the channel it chooses, the daughter masses
// it samples and the phase-space momenta it builds, all as functions of their arguments.
//
// The products are dumped IN THE ORDER Decay returns them, which is the order the cascade pushes
// them onto its track list - G4DecayProducts::PopProducts pops from the back, so it is the phase
// space's push order reversed. A port that returned them the other way round would agree on
// every four-momentum and disagree on every subsequent random number in the event.
void write_imr_decayfs() {
  FILE* f = std::fopen("bic_imr_decayfs.csv", "w");
  std::fprintf(f, "pdg,actual_mass,pz,phase,n_products,i,prod_pdg,px,py,pz_out,e,draws\n");

  G4ShortLivedConstructor shortLived;
  shortLived.ConstructParticle();
  G4ParticleTable* ptable = G4ParticleTable::GetParticleTable();

  // Resonances with two-body channels only, resonances with three-body channels, and both charge
  // states of several multiplets so that the isospin split of the branching ratios is exercised.
  const int kRes[] = {2214, 2224, 1114, 2114,
                      12212, 12112, 2124, 1214, 22212, 22112, 32212, 32112,
                      2216, 12216, 22124, 42212, 32124, 42124, 12218, 52214,
                      2128, 100002210, 100012210, 2218, 2226, 12126, 1112, 32224,
                      // FTFP's species, docs/HADRONIC_PLAN.md section 9.3. They are not decoration:
                      // f2(1270) is the ONLY parent with four-body channels in the whole closure,
                      // so it is the only thing that reaches ManyBodyDecayIt; a2(1320)0 owns both
                      // of the two three-body channels with a short-lived daughter, which is the
                      // branch that was dead until FTFP's species joined the table; and phi has
                      // the rho0 pi0 channel that needs SampleMass on a resonance daughter.
                      113, 213, -213, 223, 313, 323, -313, -323, 333, 225, 115,
                      3114, 3214, 3224, -3114, -3214, -3224, 3314, 3324,
                      -1114, -2114, -2214, -2224};

  auto* eng = new ImrCycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();

  for (int code : kRes) {
    const G4ParticleDefinition* d = ptable->FindParticle(code);
    if (d == nullptr) { continue; }
    const double pole = d->GetPDGMass();
    for (double fr : {0.95, 1.00, 1.10}) {
      const double m = pole * fr;
      for (double pz : {0.0, 700.0}) {
        for (int phase = 0; phase < 8; ++phase) {
          const G4LorentzVector q(G4ThreeVector(0, 0, pz), std::sqrt(pz * pz + m * m));
          CLHEP::HepRandom::setTheEngine(eng);
          eng->reset(phase);
          // The track is CONSTRUCTED inside the engine's scope on purpose: for these species the
          // constructor draws nothing (only a K0 does), and the dump asserts that by comparing
          // the draw count against the port, which builds its widths without an engine at all.
          G4KineticTrack kt(d, 0.0, G4ThreeVector(0, 0, 0), q);
          G4KineticTrackVector* products = kt.Decay();
          const int n = (products != nullptr) ? static_cast<int>(products->size()) : 0;
          for (int i = 0; i < n; ++i) {
            const G4KineticTrack* p = (*products)[i];
            const G4LorentzVector& p4 = p->Get4Momentum();
            std::fprintf(f, "%d,%.17g,%.17g,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%d\n", code,
                         kt.GetActualMass(), pz, phase, n, i,
                         p->GetDefinition()->GetPDGEncoding(), p4.x(), p4.y(), p4.z(), p4.t(),
                         eng->draws());
          }
          if (n == 0) {
            std::fprintf(f, "%d,%.17g,%.17g,%d,0,-1,0,0,0,0,0,%d\n", code, kt.GetActualMass(),
                         pz, phase, eng->draws());
          }
          if (products != nullptr) {
            for (auto* p : *products) { delete p; }
            delete products;
          }
          CLHEP::HepRandom::setTheEngine(saved);
        }
      }
    }
  }
  delete eng;
  std::fclose(f);
}

// G4MesonAbsorption: the scheduling, the cluster partner and the two outgoing nucleons.
//
// GetTimeToAbsorption and AbsorptionCrossSection are both PRIVATE, so the cross section cannot be
// read out directly - but it is not needed directly. The verdict `distance <= sigma/pi` flips at
// exactly one impact parameter, so scanning the transverse offset and dumping whether a collision
// was scheduled measures the cross section through the geometry that uses it, which is the only
// thing the cascade cares about. The scan runs to 1.5 fm in 0.025 fm steps, which is 0.3 mb per
// step against cross sections of 1 to 6 mb.
//
// The pair (proton, pi-) is in the list for one reason: it is the ordering that reaches the
// `else if (bT == pi+ || bT != pi-)` branch with a pi- in slot b, where the condition is false,
// the kinetic energy stays at its initialiser of zero, and `it[-2]` is read. See
// src/physics/hadronic/bic/im_r/absorption.cuh and docs/RISK.md V153. If Geant4's answer there is
// anything but the 2 mb the cancellation predicts, this dump is where it shows.
void write_imr_absorb() {
  FILE* f = std::fopen("bic_imr_absorb.csv", "w");
  std::fprintf(f, "pair,ekin_MeV,b_fm,scheduled,time_ns,n_targets,t0_pdg,t1_pdg\n");
  FILE* g = std::fopen("bic_imr_absorbfs.csv", "w");
  std::fprintf(g, "pair,ekin_MeV,phase,d1,d2,p1x,p1y,p1z,p1e,p2x,p2y,p2z,p2e,draws\n");
  FILE* h = std::fopen("bic_imr_absorbcluster.csv", "w");
  std::fprintf(h, "set,n_targets,first,partner\n");

  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* n = G4Neutron::NeutronDefinition();
  const G4ParticleDefinition* pip = G4PionPlus::PionPlusDefinition();
  const G4ParticleDefinition* pim = G4PionMinus::PionMinusDefinition();
  const G4ParticleDefinition* pi0 = G4PionZero::PionZeroDefinition();

  struct Pair { const char* name; const G4ParticleDefinition* pro; const G4ParticleDefinition* c0; };
  const Pair pairs[] = {{"pip_p", pip, p}, {"pim_p", pim, p}, {"pim_n", pim, n},
                        {"p_pim", p, pim}, {"n_pim", n, pim}, {"pi0_p", pi0, p},
                        {"pip_n", pip, n}, {"p_pip", p, pip}};

  G4MesonAbsorption absorber;

  for (const Pair& pr : pairs) {
    for (double ekin : {5.0, 40.0, 90.0, 140.0, 200.0, 280.0, 350.0}) {
      for (int ib = 0; ib <= 60; ++ib) {
        const double b = 0.025 * ib;
        const double m1 = pr.pro->GetPDGMass();
        const double e1 = ekin + m1;
        const G4LorentzVector q1(G4ThreeVector(0, 0, std::sqrt(e1 * e1 - m1 * m1)), e1);
        G4KineticTrack t0(pr.pro, 0.0, G4ThreeVector(0, 0, 0), q1);
        // Three candidates: the partner of the pair at (b, 0, 4 fm), and two spectators, one
        // neutron and one proton, far enough away that the geometry of the first pair is
        // untouched. BOTH charges are needed. FindAndFillCluster rejects a candidate that would
        // take charge(projectile) + charge(first target) + charge(candidate) outside [0, 2], and
        // with only a proton spectator a pi+ on a proton has nothing left to cluster with - the
        // sum is 3 - so no absorption is scheduled at any impact parameter. MEASURED: with the
        // single proton spectator the first version of this dump had, pip_p and p_pip produced
        // not one scheduled collision in 427 rows each, and the scan said nothing about the
        // cross section it was written to measure.
        // The candidate is NOT at rest. If it were, a pair whose kinetic energy is taken from
        // the candidate would have t = 0 and would fall into the same `it[-2]` branch as the
        // `!=` bug - and the two would be indistinguishable. MEASURED: with the candidate at
        // rest, p_pip and p_pim both scheduled exactly 11 impact parameters at every one of the
        // seven energies, the flat 2 mb, and the bug was invisible.
        const double m2 = pr.c0->GetPDGMass();
        const G4ThreeVector k2(150.0, -80.0, 60.0);
        const G4LorentzVector q2(k2, std::sqrt(k2.mag2() + m2 * m2));
        G4KineticTrack* c0 =
            new G4KineticTrack(pr.c0, 0.0, G4ThreeVector(b * fermi, 0, 4.0 * fermi), q2);
        G4KineticTrack* c1 = new G4KineticTrack(
            n, 0.0, G4ThreeVector(-1.0 * fermi, 2.0 * fermi, 3.0 * fermi),
            G4LorentzVector(G4ThreeVector(0, 0, 0), n->GetPDGMass()));
        G4KineticTrack* c2 = new G4KineticTrack(
            p, 0.0, G4ThreeVector(2.0 * fermi, -1.0 * fermi, -3.0 * fermi),
            G4LorentzVector(G4ThreeVector(0, 0, 0), p->GetPDGMass()));
        std::vector<G4KineticTrack*> cands;
        cands.push_back(c0);
        cands.push_back(c1);
        cands.push_back(c2);
        const std::vector<G4CollisionInitialState*>& coll =
            absorber.GetCollisions(&t0, cands, 0.0);
        int found = 0;
        double time = -1.0;
        int nt = 0, t0p = 0, t1p = 0;
        for (auto* ci : coll) {
          if (ci->GetPrimary() != &t0) { continue; }
          G4KineticTrackVector& tv = ci->GetTargetCollection();
          if (tv.empty() || tv[0] != c0) { continue; }
          found = 1;
          time = ci->GetCollisionTime();
          nt = static_cast<int>(tv.size());
          t0p = tv[0]->GetDefinition()->GetPDGEncoding();
          t1p = (nt > 1) ? tv[1]->GetDefinition()->GetPDGEncoding() : 0;
          break;
        }
        std::fprintf(f, "%s,%.17g,%.17g,%d,%.17g,%d,%d,%d\n", pr.name, ekin, b, found, time, nt,
                     t0p, t1p);
        for (auto* ci : coll) { delete ci; }
        delete c0;
        delete c1;
        delete c2;
      }
    }
  }

  // FindAndFillCluster on its own, through the partner the scheduled collision carries. Five
  // candidates at five positions with three charge patterns: the charge filter rejects any
  // candidate that would take the sum of (projectile + first target + candidate) outside [0, 2],
  // and among the survivors the winner is the one with the smallest |r_first + r_candidate| -
  // a SUM, so the winner is the one most nearly opposite the first target through the origin.
  // docs/RISK.md V154. The positions below are chosen so that the nearest and the most-opposite
  // candidates are different tracks, which is the only way to tell the two rules apart.
  {
    struct Cand { double x, y, z; int charge; };
    const Cand kSets[3][5] = {
        {{0.2, 0.0, 4.1, 1}, {-0.2, 0.0, -4.1, 1}, {3.0, 1.0, 1.0, 0}, {-3.0, -1.0, -1.0, 0},
         {0.1, 0.1, 4.0, 0}},
        {{1.0, 1.0, 1.0, 1}, {-1.0, -1.0, -1.0, 1}, {0.5, 0.5, 0.5, 1}, {-0.5, -0.5, -0.5, 1},
         {2.0, 0.0, 0.0, 0}},
        {{0.0, 0.0, 5.0, 0}, {0.0, 0.0, -5.0, 0}, {1.0, 0.0, 4.5, 1}, {-1.0, 0.0, -4.5, 1},
         {0.0, 3.0, 0.0, 0}}};
    for (int set = 0; set < 3; ++set) {
      const double m1 = pip->GetPDGMass();
      const double e1 = 140.0 + m1;
      const G4LorentzVector q1(G4ThreeVector(0, 0, std::sqrt(e1 * e1 - m1 * m1)), e1);
      G4KineticTrack pro(pip, 0.0, G4ThreeVector(0, 0, 0), q1);
      std::vector<G4KineticTrack*> cands;
      for (int k = 0; k < 5; ++k) {
        const Cand& cd = kSets[set][k];
        const G4ParticleDefinition* d = (cd.charge == 1) ? p : n;
        cands.push_back(new G4KineticTrack(
            d, 0.0, G4ThreeVector(cd.x * fermi, cd.y * fermi, cd.z * fermi),
            G4LorentzVector(G4ThreeVector(0, 0, 0), d->GetPDGMass())));
      }
      const std::vector<G4CollisionInitialState*>& coll = absorber.GetCollisions(&pro, cands, 0.0);
      for (auto* ci : coll) {
        G4KineticTrackVector& tv = ci->GetTargetCollection();
        int first = -1, partner = -1;
        for (int k = 0; k < 5; ++k) {
          if (!tv.empty() && tv[0] == cands[k]) { first = k; }
          if (tv.size() > 1 && tv[1] == cands[k]) { partner = k; }
        }
        std::fprintf(h, "%d,%d,%d,%d\n", set, static_cast<int>(tv.size()), first, partner);
      }
      for (auto* ci : coll) { delete ci; }
      for (auto* q : cands) { delete q; }
    }
  }

  // The final state, under the prescribed engine. The projectile is a charged pion and the two
  // targets are the four combinations of proton and neutron, so both the identity swap and the
  // charge fixing - which draws a SECOND uniform only when both nucleons sit on the same side of
  // the pion's charge - are exercised.
  {
    auto* eng = new ImrCycleEngine();
    CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
    struct FsCase { const char* name; const G4ParticleDefinition* pro;
                    const G4ParticleDefinition* a; const G4ParticleDefinition* b; };
    const FsCase cases[] = {{"pip_pp", pip, p, p}, {"pip_pn", pip, p, n}, {"pip_np", pip, n, p},
                            {"pip_nn", pip, n, n}, {"pim_pp", pim, p, p}, {"pim_pn", pim, p, n},
                            {"pim_np", pim, n, p}, {"pim_nn", pim, n, n},
                            {"pi0_pn", pi0, p, n}};
    for (const FsCase& c : cases) {
      for (double ekin : {40.0, 150.0, 300.0}) {
        const double m1 = c.pro->GetPDGMass();
        const double e1 = ekin + m1;
        // A tilted projectile, so that the rotation to z is not the identity - the same lesson
        // the elastic final state taught (docs/RISK.md V75).
        G4ThreeVector v1(0, 0, std::sqrt(e1 * e1 - m1 * m1));
        v1.rotateY(0.4);
        v1.rotateZ(0.9);
        const G4LorentzVector q1(v1, e1);
        G4KineticTrack pro(c.pro, 0.0, G4ThreeVector(0, 0, 0), q1);
        const G4ThreeVector k1(31.0, -17.0, 44.0);
        const G4ThreeVector k2(-23.0, 51.0, -12.0);
        G4KineticTrack* ta =
            new G4KineticTrack(c.a, 0.0, G4ThreeVector(0.5 * fermi, 0, 2.0 * fermi),
                               G4LorentzVector(k1, std::sqrt(k1.mag2() + c.a->GetPDGMass() *
                                                                            c.a->GetPDGMass())));
        G4KineticTrack* tb =
            new G4KineticTrack(c.b, 0.0, G4ThreeVector(-0.5 * fermi, 1.0 * fermi, 2.5 * fermi),
                               G4LorentzVector(k2, std::sqrt(k2.mag2() + c.b->GetPDGMass() *
                                                                            c.b->GetPDGMass())));
        std::vector<G4KineticTrack*> targets;
        targets.push_back(ta);
        targets.push_back(tb);
        CLHEP::HepRandom::setTheEngine(eng);
        for (int phase = 0; phase < 8; ++phase) {
          eng->reset(phase);
          G4KineticTrackVector* out = absorber.GetFinalState(&pro, targets);
          if (out != nullptr && out->size() == 2) {
            const G4LorentzVector& a4 = (*out)[0]->Get4Momentum();
            const G4LorentzVector& b4 = (*out)[1]->Get4Momentum();
            std::fprintf(g,
                         "%s,%.17g,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n",
                         c.name, ekin, phase, (*out)[0]->GetDefinition()->GetPDGEncoding(),
                         (*out)[1]->GetDefinition()->GetPDGEncoding(), a4.x(), a4.y(), a4.z(),
                         a4.t(), b4.x(), b4.y(), b4.z(), b4.t(), eng->draws());
          }
          if (out != nullptr) {
            for (auto* q : *out) { delete q; }
            delete out;
          }
        }
        CLHEP::HepRandom::setTheEngine(saved);
        delete ta;
        delete tb;
      }
    }
    delete eng;
  }
  std::fclose(f);
  std::fclose(g);
  std::fclose(h);
}

// G4DecayKineticTracks at the LIST level - the transitive pass the Fritiof model and the binary
// cascade both open with.
//
// The four-momenta are already compared decay by decay in bic_imr_decayfs.csv; what this file is
// for is everything the list-level loop adds on top of them, none of which any per-decay
// comparison can see:
//
//   * the ORDER the survivors come back in. The loop walks a list that GROWS, appends daughters
//     at the end, nulls the parent's slot and compacts afterwards - so the output is the
//     non-decaying originals in their original order, then the daughters generation by
//     generation. A port that rebuilt the list any other way would agree on every momentum and
//     put every later random number in the event out of step.
//   * the RECURSION. A daughter that is itself short-lived is decayed when the walk reaches it,
//     with no recursive call - N(1520)+ -> delta++ pi- is in the sets below for that reason.
//   * that a short-lived track whose Decay() returns nothing SURVIVES, because the null-slot
//     bookkeeping only runs when daughters came back.
//   * the parent bookkeeping: CreatorModelID inherited, ParentResonanceDef set to the parent's
//     definition and ParentResonanceID to round(parent 4-momentum .mag() / keV).
void write_imr_kdecay() {
  FILE* f = std::fopen("bic_imr_kdecay.csv", "w");
  std::fprintf(f, "set,phase,n_in,n_out,i,pdg,px,py,pz,e,parent_pdg,parent_id,creator,draws\n");

  G4ShortLivedConstructor shortLived;
  shortLived.ConstructParticle();
  G4ParticleTable* ptable = G4ParticleTable::GetParticleTable();

  // Seven lists. Each entry is (pdg, pz in MeV). The stable tracks are there to fix the order
  // the survivors have to come back in, and they are deliberately NOT all at the front.
  struct Entry { int pdg; double pz; };
  const Entry kSets[7][6] = {
      {{2212, 500}, {2214, 300}, {2124, 100}, {0, 0}, {0, 0}, {0, 0}},
      {{2214, 0}, {2212, 900}, {1114, 250}, {2112, 50}, {0, 0}, {0, 0}},
      {{211, 400}, {113, 600}, {223, 200}, {-211, 100}, {0, 0}, {0, 0}},
      {{225, 800}, {115, 300}, {2212, 0}, {0, 0}, {0, 0}, {0, 0}},
      {{333, 150}, {313, 450}, {3214, 200}, {321, 300}, {0, 0}, {0, 0}},
      {{2224, 700}, {12212, 350}, {32124, 150}, {2112, 250}, {22124, 80}, {2212, 600}},
      {{2212, 100}, {2112, 200}, {211, 300}, {0, 0}, {0, 0}, {0, 0}}};

  auto* eng = new ImrCycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();

  for (int set = 0; set < 7; ++set) {
    for (int phase = 0; phase < 8; ++phase) {
      CLHEP::HepRandom::setTheEngine(eng);
      eng->reset(phase);
      G4KineticTrackVector* list = new G4KineticTrackVector;
      int n_in = 0;
      for (int k = 0; k < 6; ++k) {
        if (kSets[set][k].pdg == 0) { continue; }
        const G4ParticleDefinition* d = ptable->FindParticle(kSets[set][k].pdg);
        if (d == nullptr) { continue; }
        const double m = d->GetPDGMass();
        const double pz = kSets[set][k].pz;
        const G4LorentzVector q(G4ThreeVector(0, 0, pz), std::sqrt(pz * pz + m * m));
        G4KineticTrack* t = new G4KineticTrack(d, 0.0, G4ThreeVector(0, 0, 0), q);
        t->SetCreatorModelID(70 + k);
        list->push_back(t);
        ++n_in;
      }
      G4DecayKineticTracks decay(list);
      const int n_out = static_cast<int>(list->size());
      for (int i = 0; i < n_out; ++i) {
        const G4KineticTrack* t = (*list)[i];
        const G4LorentzVector& p = t->Get4Momentum();
        const G4ParticleDefinition* par = t->GetParentResonanceDef();
        std::fprintf(f, "%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%d\n", set, phase,
                     n_in, n_out, i, t->GetDefinition()->GetPDGEncoding(), p.x(), p.y(), p.z(),
                     p.t(), (par != nullptr) ? par->GetPDGEncoding() : 0,
                     t->GetParentResonanceID(), t->GetCreatorModelID(), eng->draws());
      }
      for (auto* t : *list) { delete t; }
      delete list;
      CLHEP::HepRandom::setTheEngine(saved);
    }
  }
  delete eng;
  std::fclose(f);
}

// G4Scatterer::Scatter - the whole collision tree end to end, from two tracks to their products.
//
// The leaves are already compared one at a time elsewhere in this file; what only this dump can
// see is the DISPATCH: which of G4CollisionNN's eight components a uniform lands in, which of the
// nine middle composites under two of them, which of the 306 concrete channels under that, and
// how many uniforms the whole path consumed. The product identities name the concrete channel and
// the draw count names the depth, so the two together pin the path without needing access to any
// of the private caches.
/// The one G4Scatterer this file uses, and it is DELIBERATELY NEVER DESTROYED.
///
/// `G4Scatterer::~G4Scatterer` deletes every registered collision and clears the static
/// `G4Scatterer::collisions`, and does NOT reset the file-scope `setupDone` that guards the
/// registration. So the first G4Scatterer to be destroyed empties the channel list for every
/// G4Scatterer that will ever be constructed afterwards, in the same process. See
/// `write_imr_scatterlife` below, which measures it, and docs/RISK.md V155.
///
/// This dump found it the hard way: `write_imr_scatterer` had a local `G4Scatterer scatterer;`,
/// and every one of the 648 rows `write_imr_scatter` produced after that function returned had a
/// cross section of exactly zero and drew no random at all.
G4Scatterer& imr_scatterer() {
  static G4Scatterer* s = new G4Scatterer();  // never deleted, on purpose
  return *s;
}

/// The lifetime bug itself, in three measurements on the same pair: from a scatterer built while
/// the list is intact, then after ONE scatterer somewhere else has been destroyed, then from a
/// freshly constructed one. The second and third are the same number and it is zero.
// **THIS SWEEP POISONS THE PROCESS AND MUST RUN LAST.** `{ G4Scatterer doomed; }` below is the
// whole point of it - it MEASURES what V155 costs - and the cost is that every `G4Scatterer` in
// the process, including the leaked singleton and any made afterwards, has an empty channel list
// from that line on. MEASURED, the second time: with this sweep in its old place, four dumps in
// a row of `write_imr_propagate` came back with `Propagate` returning NULL for all forty cases,
// zero random draws and `GetCrossSection` exactly 0.0 for a 400 MeV proton on a proton, where
// the same pair gives 25.6957 mb earlier in the same run. `dump_bic()` calls this one last.
void write_imr_scatterlife() {
  FILE* f = std::fopen("bic_imr_scatterlife.csv", "w");
  std::fprintf(f, "step,sigma_mb,time_ns\n");
  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const double m = p->GetPDGMass();
  const double e1 = 200.0 + m;
  const G4LorentzVector q1(G4ThreeVector(0, 0, std::sqrt(e1 * e1 - m * m)), e1);
  const G4LorentzVector q2(G4ThreeVector(0, 0, 0), m);
  G4KineticTrack t1(p, 0.0, G4ThreeVector(0, 0, 0), q1);
  G4KineticTrack t2(p, 0.0, G4ThreeVector(0.5 * fermi, 0, 3.0 * fermi), q2);
  auto row = [&](const char* step, const G4Scatterer& sc) {
    const double t = sc.GetTimeToInteraction(t1, t2);
    std::fprintf(f, "%s,%.17g,%.17g\n", step, sc.GetCrossSection(t1, t2) / millibarn,
                 (t < DBL_MAX) ? t : -1.0);
  };
  row("intact", imr_scatterer());
  { G4Scatterer doomed; row("before_destruction", doomed); }
  row("after_destruction", imr_scatterer());
  { G4Scatterer fresh; row("freshly_constructed", fresh); }
  std::fclose(f);
}

void write_imr_scatter() {

  FILE* f = std::fopen("bic_imr_scatter.csv", "w");
  std::fprintf(f, "pair,ekin_MeV,phase,n,i,pdg,px,py,pz,e,draws\n");

  G4ShortLivedConstructor shortLived;
  shortLived.ConstructParticle();
  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* n = G4Neutron::NeutronDefinition();
  const G4ParticleDefinition* pip = G4PionPlus::PionPlusDefinition();
  const G4ParticleDefinition* pim = G4PionMinus::PionMinusDefinition();
  const G4ParticleDefinition* pi0 = G4PionZero::PionZeroDefinition();
  struct Pair { const char* name; const G4ParticleDefinition* a; const G4ParticleDefinition* b; };
  const Pair pairs[] = {{"pp", p, p},   {"nn", n, n},     {"np", n, p},     {"pn", p, n},
                        {"pip_p", pip, p}, {"pim_p", pim, p}, {"pi0_p", pi0, p},
                        {"pip_n", pip, n}, {"pim_n", pim, n}};

  // Energies chosen so that every one of the eight nucleon components is reachable somewhere:
  // elastic only below the pion threshold, N-Delta from about 800 MeV, and the Delta* and N*
  // families as sqrt(s) climbs past their thresholds.
  const double kT[] = {50.0, 200.0, 500.0, 800.0, 1000.0, 1400.0, 2000.0, 3000.0, 5000.0};

  G4Scatterer& scatterer = imr_scatterer();
  auto* eng = new ImrCycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();

  for (const Pair& pr : pairs) {
    for (double ekin : kT) {
      const double m1 = pr.a->GetPDGMass();
      const double m2 = pr.b->GetPDGMass();
      const double e1 = ekin + m1;
      // A tilted projectile and a target with its own momentum, so the boosts and the rotations
      // in every leaf are live - with both tracks along +z the rotations are the identity and a
      // composed inverse cannot be told from a composed forward (docs/RISK.md V75).
      G4ThreeVector v1(0, 0, std::sqrt(e1 * e1 - m1 * m1));
      v1.rotateY(0.35);
      v1.rotateZ(1.1);
      const G4LorentzVector q1(v1, e1);
      const G4ThreeVector v2(37.0, -29.0, 19.0);
      const G4LorentzVector q2(v2, std::sqrt(v2.mag2() + m2 * m2));
      for (int phase = 0; phase < 8; ++phase) {
        G4KineticTrack t1(pr.a, 0.0, G4ThreeVector(0, 0, 0), q1);
        G4KineticTrack t2(pr.b, 0.0, G4ThreeVector(0, 0, 0), q2);
        CLHEP::HepRandom::setTheEngine(eng);
        eng->reset(phase);
        const double diag_sigma = scatterer.GetCrossSection(t1, t2);
        G4KineticTrackVector* products = scatterer.Scatter(t1, t2);
        const int np = (products != nullptr) ? static_cast<int>(products->size()) : 0;
        if (np == 0) {
          std::fprintf(f, "%s,%.17g,%d,0,-1,0,%.17g,0,0,0,%d\n", pr.name, ekin, phase,
                       diag_sigma / millibarn, eng->draws());
        }
        for (int i = 0; i < np; ++i) {
          const G4KineticTrack* q = (*products)[i];
          const G4LorentzVector& p4 = q->Get4Momentum();
          std::fprintf(f, "%s,%.17g,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%d\n", pr.name, ekin,
                       phase, np, i, q->GetDefinition()->GetPDGEncoding(), p4.x(), p4.y(), p4.z(),
                       p4.t(), eng->draws());
        }
        if (products != nullptr) {
          for (auto* q : *products) { delete q; }
          delete products;
        }
        CLHEP::HepRandom::setTheEngine(saved);
      }
    }
  }
  delete eng;
  std::fclose(f);
}

// G4BinaryCascade::GetIonMass's first arm, which is G4IonTable::GetIonMass(Z,A).
//
// The other three arms - charge above the mass number, an all-neutral remnant, an empty nucleus -
// are branches on (Z, A) and not table lookups, so they are asserted in the test against the
// arithmetic the source writes. This file is the one arm that reads a table, and it reads it for
// every (Z, A) a cascade can leave behind: the residual of a spallation on a QBBC target walks
// down in both, so the grid is the whole triangle up to the heaviest target and not a list.
void write_imr_ionmass() {
  FILE* f = std::fopen("bic_imr_ionmass.csv", "w");
  std::fprintf(f, "z,a,mass\n");
  G4IonTable* it = G4ParticleTable::GetParticleTable()->GetIonTable();
  for (int a = 1; a <= 240; ++a) {
    for (int z = 1; z <= a && z <= 94; ++z) {
      // Only the cells a cascade can reach: within 30 nucleons of the beta-stable line, which
      // covers every residual a 1.5 GeV projectile can leave, plus the light corner in full.
      const int zstable = (a <= 4) ? z : static_cast<int>(a / (1.98 + 0.0155 * std::pow(
                                          static_cast<double>(a), 2.0 / 3.0)) + 0.5);
      if (a > 20 && (z < zstable - 15 || z > zstable + 15)) { continue; }
      std::fprintf(f, "%d,%d,%.17g\n", z, a, it->GetIonMass(z, a));
    }
  }
  std::fclose(f);
}

// G4BinaryCascade::CheckPauliPrinciple and the field its two Fermi corrections read.
//
// CheckPauliPrinciple is private, so this dump re-expresses it from the three public ingredients
// the source names - G4FermiMomentum::GetFermiMomentum, G4V3DNucleus::GetNuclearDensity()->
// GetDensity and G4V3DNucleus::CoulombBarrier - in the order and the arithmetic the source
// writes:
//
//     eFermi = sqrt( sqr(fermiMom.GetFermiMomentum(density->GetDensity(pos))) + p4.mag2() );
//     if (proton) eFermi -= the3DNucleus->CoulombBarrier();
//     blocked if ( mom.e() < eFermi )
//
// That is not a weaker oracle than calling the function: every term in it comes from Geant4 and
// the composition is three lines of the source quoted above. What it cannot check is the loop
// around it, which the test asserts structurally instead.
//
// G4RKPropagation::GetField(encoding, pos) IS public - an inline in the header - so the two
// Fermi corrections' one ingredient is dumped directly, including its ZERO for a resonance,
// which is the whole reason CorrectShortlivedPrimaryForFermi substitutes the neutron's field.
void write_imr_pauli() {
  FILE* f = std::fopen("bic_imr_pauli.csv", "w");
  std::fprintf(f, "a,z,r_fm,pdg,pz,rho,pfermi,coulomb,efermi,blocked\n");
  FILE* g = std::fopen("bic_imr_field.csv", "w");
  std::fprintf(g, "a,z,r_fm,pdg,field\n");

  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* n = G4Neutron::NeutronDefinition();
  struct Target { int a; int z; };
  const Target kT[] = {{12, 6}, {16, 8}, {27, 13}, {56, 26}, {208, 82}};
  const double kR[] = {0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0};
  const double kPz[] = {50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 600.0, 1000.0};
  const int kField[] = {2212, 2112, 211, -211, 111, 2214, 12212, 3122};

  for (const Target& t : kT) {
    // The nucleus is LEAKED and the propagator is a local, which is the pattern
    // `write_field_and_rk` above uses and the only one that survives. `G4RKPropagation::Init`
    // stores the nucleus pointer and keeps reading it, so a nucleus on the stack that goes out
    // of scope each iteration takes the propagator's field map with it - the dump died there
    // with nothing on stderr, twice, before this was the shape.
    CLHEP::HepRandom::setTheSeed(910000L + t.a);
    auto* nucleus = new G4Fancy3DNucleus;
    nucleus->Init(t.a, t.z);
    G4FermiMomentum fermiMom;
    fermiMom.Init(t.a, t.z);
    const G4VNuclearDensity* density = nucleus->GetNuclearDensity();
    const double coulomb = nucleus->CoulombBarrier();
    G4RKPropagation prop;
    prop.Init(nucleus);
    for (double r : kR) {
      const G4ThreeVector pos(r * fermi, 0, 0);
      const double rho = density->GetDensity(pos);
      const double pf = fermiMom.GetFermiMomentum(rho);
      for (int code : kField) {
        std::fprintf(g, "%d,%d,%.17g,%d,%.17g\n", t.a, t.z, r, code, prop.GetField(code, pos));
      }
      for (const G4ParticleDefinition* d : {p, n}) {
        const double m = d->GetPDGMass();
        for (double pz : kPz) {
          const G4LorentzVector mom(G4ThreeVector(0, 0, pz), std::sqrt(pz * pz + m * m));
          double efermi = std::sqrt(pf * pf + mom.mag2());
          if (d == p) { efermi -= coulomb; }
          std::fprintf(f, "%d,%d,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n", t.a, t.z, r,
                       d->GetPDGEncoding(), pz, rho, pf, coulomb, efermi,
                       (mom.e() < efermi) ? 1 : 0);
        }
      }
    }
  }
  std::fclose(f);
  std::fclose(g);
}

// G4BinaryCascade::Capture's decision and G4Absorber::WillBeAbsorbed, both re-expressed from
// their public ingredients - see write_imr_pauli above for why that is a faithful oracle.
//
// Capture is all-or-nothing and is decided by a MEAN: `capturedEnergy/particlesBelowCut <
// 0.2*theCutOnP`, where particlesBelowCut counts EVERY nucleon inside whatever its energy,
// because the `if(energy < theCutOnP)` that would have separated them is commented out upstream
// and so is the `particlesAboveCut==0 &&` that used the result. Each nucleon contributes
// `e() - actualMass + GetField(pdg,pos) - GetBarrier(pdg)`, which can be NEGATIVE.
//
// The synthetic lists below are built to straddle the gate: one of slow nucleons deep in the
// well, one of fast ones, and mixtures where a single deep nucleon pulls the mean under.
void write_imr_capture() {
  FILE* f = std::fopen("bic_imr_capture.csv", "w");
  std::fprintf(f, "a,z,list,n_inside,captured_energy,mean,cut,capture\n");
  FILE* g = std::fopen("bic_imr_absorbcut.csv", "w");
  std::fprintf(g, "pdg,ekin,cut,absorbed\n");
  // The per-entry field the decision is built from, so that the test can drive
  // `capture_decision` with the field Geant4 computed rather than rebuilding a whole nucleus to
  // get it. The field itself is P9's and is already compared in bic_field.csv; what is new here
  // is the sum over the list, the count, and the gate on the mean.
  FILE* h = std::fopen("bic_imr_capturefield.csv", "w");
  std::fprintf(h, "a,z,list,k,pdg,r_fm,pmag,field_minus_barrier\n");

  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* n = G4Neutron::NeutronDefinition();
  struct Target { int a; int z; };
  const Target kT[] = {{12, 6}, {27, 13}, {56, 26}, {208, 82}};
  // (pdg, r in fm, |p| in MeV/c) triples; a zero pdg ends a list.
  const int kLists = 8;
  struct Entry { int pdg; double r; double pmag; };
  const Entry kSets[8][6] = {
      {{2212, 1.0, 20.0}, {2112, 1.5, 25.0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
      // The nuclear well is 35 to 55 MeV deep, so a nucleon needs a kinetic energy above about
      // 45 + 9 MeV before it lifts the mean over the gate at all. MEASURED: the first version of
      // these lists topped out at 350 MeV/c, which is 64 MeV of kinetic energy, and every one of
      // the 24 rows captured - the verdict column was a constant and the oracle proved nothing.
      // 900 and 1200 MeV/c are 370 and 620 MeV and they do not.
      {{2212, 1.0, 900.0}, {2112, 1.5, 1200.0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
      {{2212, 0.5, 10.0}, {2112, 0.5, 1000.0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
      {{2112, 0.2, 5.0}, {2112, 0.4, 8.0}, {2212, 6.0, 1400.0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
      {{2212, 8.0, 60.0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
      {{2212, 1.0, 150.0}, {2112, 1.0, 150.0}, {2212, 1.0, 150.0}, {2112, 1.0, 150.0},
       {2212, 1.0, 150.0}, {0, 0, 0}},
      // Two lists tuned to land BETWEEN 0.2*theCutOnP and theCutOnP, which is the only window
      // where the 0.2 decides anything. MEASURED: with the six lists above, replacing the gate
      // `mean < 0.2*theCutOnP` by `mean < theCutOnP` changed none of the 24 verdicts - every one
      // of them was tens of MeV clear of both. A proton at 421 MeV/c one fermi into carbon has
      // a mean of about +24 MeV, which is over 9 and under 45.
      {{2212, 1.0, 421.0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
      {{2212, 1.0, 380.0}, {2112, 1.2, 390.0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};

  for (const Target& t : kT) {
    CLHEP::HepRandom::setTheSeed(920000L + t.a);
    auto* nucleus = new G4Fancy3DNucleus;
    nucleus->Init(t.a, t.z);
    G4RKPropagation prop;
    prop.Init(nucleus);
    const double cut = 45.0 * MeV;   // theCutOnP is always 45; docs/RISK.md V72
    for (int l = 0; l < kLists; ++l) {
      double captured = 0.0;
      int below = 0;
      for (int k = 0; k < 6; ++k) {
        const Entry& e = kSets[l][k];
        if (e.pdg == 0) { break; }
        const G4ParticleDefinition* d = (e.pdg == 2212) ? p : n;
        const double m = d->GetPDGMass();
        const G4ThreeVector pos(e.r * fermi, 0, 0);
        const double energy = std::sqrt(e.pmag * e.pmag + m * m);
        const double field = prop.GetField(e.pdg, pos) - prop.GetBarrier(e.pdg);
        captured += energy - m + field;
        std::fprintf(h, "%d,%d,%d,%d,%d,%.17g,%.17g,%.17g\n", t.a, t.z, l, below, e.pdg, e.r,
                     e.pmag, field);
        ++below;
      }
      const double mean = (below > 0) ? captured / below : 0.0;
      std::fprintf(f, "%d,%d,%d,%d,%.17g,%.17g,%.17g,%d\n", t.a, t.z, l, below, captured, mean,
                   cut, (below > 0 && mean < 0.2 * cut) ? 1 : 0);
    }
  }

  // G4Absorber::WillBeAbsorbed, at the threshold the constructor sets and at one that is not
  // zero, so that the predicate is shown to be live and the CONSTANT shown to be what kills it.
  for (int pdg : {211, -211, 111, 2212, 2112}) {
    const G4ParticleDefinition* d = G4ParticleTable::GetParticleTable()->FindParticle(pdg);
    const double m = d->GetPDGMass();
    for (double ekin : {0.0, 1.0, 25.0, 100.0, 500.0, 2000.0}) {
      for (double cut : {0.0, 50.0}) {
        const double e = ekin + m;
        const bool is_pion = (pdg == 211 || pdg == 111 || pdg == -211);
        std::fprintf(g, "%d,%.17g,%.17g,%d\n", pdg, ekin, cut,
                     ((e - m < cut) && is_pion) ? 1 : 0);
      }
    }
  }
  std::fclose(f);
  std::fclose(g);
  std::fclose(h);
}

// G4BinaryCascade::CorrectBarionsOnBoundary, re-expressed from G4IonTable::GetIonMass and
// G4RKPropagation::GetBarrier, both public - see write_imr_pauli for why that is faithful.
//
// The formula and the branch are what this compares; the two ingredients are already compared in
// bic_imr_ionmass.csv and bic_field.csv. The sets below cross a proton, a neutron and a Delta++
// in both directions, one at a time and together, so that the 1/secondaries share-out and the
// PROTON-in/NEUTRON-out asymmetry (docs/RISK.md V156) both show.
void write_imr_boundary() {
  FILE* f = std::fopen("bic_imr_boundary.csv", "w");
  std::fprintf(f, "a,z,dir,set,n_cross,mass_secondary,mass_initial,mass_final,correction,"
                  "a_after,z_after,barrier_p,barrier_n\n");

  G4IonTable* it = G4ParticleTable::GetParticleTable()->GetIonTable();
  const double mp = G4Proton::Proton()->GetPDGMass();
  const double mn = G4Neutron::Neutron()->GetPDGMass();
  struct Target { int a; int z; };
  const Target kT[] = {{12, 6}, {27, 13}, {56, 26}, {208, 82}};
  // Each set is a list of (baryon, charge, is_nucleon, pdg_mass) crossings.
  struct Cross { int baryon; int charge; int is_nucleon; double pdg_mass; };
  const Cross kSets[5][3] = {
      {{1, 1, 1, 0.0}, {0, 0, 0, 0.0}, {0, 0, 0, 0.0}},          // one proton
      {{1, 0, 1, 0.0}, {0, 0, 0, 0.0}, {0, 0, 0, 0.0}},          // one neutron
      {{1, 2, 0, 1232.0}, {0, 0, 0, 0.0}, {0, 0, 0, 0.0}},       // one Delta++
      {{1, 1, 1, 0.0}, {1, 0, 1, 0.0}, {0, 0, 0, 0.0}},          // a proton and a neutron
      {{1, 1, 1, 0.0}, {1, 2, 0, 1232.0}, {1, 0, 1, 0.0}}};      // proton, Delta++, neutron

  auto ionmass = [&](int z, int a) {
    if (z > 0 && a >= z) { return it->GetIonMass(z, a); }
    if (a > 0 && z > 0) { return it->GetIonMass(a, a); }
    if (a >= 0 && z <= 0) { return a * mn; }
    return 0.0;
  };

  for (const Target& t : kT) {
    CLHEP::HepRandom::setTheSeed(930000L + t.a);
    auto* nucleus = new G4Fancy3DNucleus;
    nucleus->Init(t.a, t.z);
    G4RKPropagation prop;
    prop.Init(nucleus);
    for (int dir = 0; dir < 2; ++dir) {   // 0 = in, 1 = out
      for (int set = 0; set < 5; ++set) {
        int n = 0, barions = 0, charge = 0;
        double mass_secondary = 0.0;
        for (int k = 0; k < 3; ++k) {
          const Cross& c = kSets[set][k];
          if (c.baryon == 0 && c.charge == 0 && c.is_nucleon == 0) { break; }
          ++n;
          charge += c.charge;
          barions += c.baryon;
          if (c.is_nucleon != 0) {
            mass_secondary += (c.charge == 1) ? mp : mn;
          } else {
            // The whole point: a resonance going IN counts as a proton and going OUT as a
            // neutron. Its own PDG mass is never used.
            mass_secondary += (dir == 0) ? mp : mn;
          }
        }
        const double mass_initial = ionmass(t.z, t.a);
        const int a_after = (dir == 0) ? (t.a + barions) : (t.a - barions);
        const int z_after = (dir == 0) ? (t.z + charge) : (t.z - charge);
        const double mass_final = ionmass(z_after, a_after);
        double correction = (dir == 0) ? (mass_secondary + mass_initial - mass_final)
                                       : (mass_initial - mass_final - mass_secondary);
        if (n > 1) { correction /= n; }
        std::fprintf(f, "%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%d,%.17g,%.17g\n", t.a, t.z,
                     dir, set, n, mass_secondary, mass_initial, mass_final, correction, a_after,
                     z_after, prop.GetBarrier(2212), prop.GetBarrier(2112));
      }
    }
  }
  std::fclose(f);
}

// G4FermiPhaseSpaceDecay::Decay - Kopylov's n-body sampling, which G4BinaryCascade is the ONLY
// class in Geant4 to instantiate (P3's fermi_breakup.cuh records that the Fermi break-up model
// itself does not use it). DecayVoidNucleus is the one caller, for an all-neutron remnant.
//
// The class is public and default-constructible, so this is a direct comparison and not a
// re-expression: the same masses, the same parent mass, the same prescribed uniforms.
void write_imr_fps() {
  FILE* f = std::fopen("bic_imr_fps.csv", "w");
  std::fprintf(f, "case,n,parent_mass,phase,k,px,py,pz,e,draws\n");

  const double mn = G4Neutron::Neutron()->GetPDGMass();
  const double mp = G4Proton::Proton()->GetPDGMass();
  // Two to eight bodies, equal and unequal masses, and parent masses from just above threshold
  // to well above it - the `max(M, mtot + eV)` guard is reached by the first of those.
  struct Case { int n; double masses[8]; };
  const Case kCases[6] = {
      {2, {mn, mn, 0, 0, 0, 0, 0, 0}},
      {3, {mn, mn, mp, 0, 0, 0, 0, 0}},
      {4, {mn, mn, mn, mp, 0, 0, 0, 0}},
      {5, {mn, mp, mn, mp, mn, 0, 0, 0}},
      {6, {mn, mn, mn, mn, mn, mn, 0, 0}},
      {8, {mn, mp, mn, mp, mn, mp, mn, mp}}};

  // The SIXTY-FOUR-value ladder, not the eight-value cycle: BetaKopylov's rejection loop has
  // no iteration guard and the eight-value cycle hangs it. See ImrCycle64Engine and V157.
  auto* eng = new ImrCycle64Engine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  G4FermiPhaseSpaceDecay decay;

  for (int c = 0; c < 6; ++c) {
    const Case& cs = kCases[c];
    std::vector<G4double> masses;
    double mtot = 0.0;
    for (int k = 0; k < cs.n; ++k) {
      masses.push_back(cs.masses[k]);
      mtot += cs.masses[k];
    }
    for (double extra : {0.0, 1.0, 20.0, 200.0}) {
      const double parent = mtot + extra;
      CLHEP::HepRandom::setTheEngine(eng);
      for (int phase = 0; phase < 8; ++phase) {
        eng->reset(phase);
        std::vector<G4LorentzVector*>* p = decay.Decay(parent, masses);
        for (int k = 0; k < static_cast<int>(p->size()); ++k) {
          std::fprintf(f, "%d,%d,%.17g,%d,%d,%.17g,%.17g,%.17g,%.17g,%d\n", c, cs.n, parent,
                       phase, k, (*p)[k]->x(), (*p)[k]->y(), (*p)[k]->z(), (*p)[k]->t(),
                       eng->draws());
          delete (*p)[k];
        }
        delete p;
      }
      CLHEP::HepRandom::setTheEngine(saved);
    }
  }
  delete eng;
  std::fclose(f);
}

void write_imr_mbselect() {


  FILE* f = std::fopen("bic_imr_mbpartial.csv", "w");
  std::fprintf(f, "pair,sqrt_s_MeV,in1z,in1e,in2e,component,sigma_mb\n");
  FILE* g = std::fopen("bic_imr_mbselect.csv", "w");
  std::fprintf(g, "pair,sqrt_s_MeV,in1z,in1e,in2e,phase,selected,draws\n");

  G4ShortLivedConstructor shortLived;
  shortLived.ConstructParticle();
  const G4ParticleDefinition* p = G4Proton::ProtonDefinition();
  const G4ParticleDefinition* n = G4Neutron::NeutronDefinition();
  struct Pair { const char* name; const G4ParticleDefinition* a; const G4ParticleDefinition* b; };
  const Pair pairs[] = {
      {"pip_p", G4PionPlus::PionPlusDefinition(), p},
      {"pim_p", G4PionMinus::PionMinusDefinition(), p},
      {"pi0_p", G4PionZero::PionZeroDefinition(), p},
      {"pip_n", G4PionPlus::PionPlusDefinition(), n},
      {"pim_n", G4PionMinus::PionMinusDefinition(), n},
      // A pion on a RESONANCE, which the cascade makes as soon as the first Delta is produced.
      // The two components disagree about it: G4CollisionMesonBaryonElastic::IsInCharge counts
      // partons and accepts it, G4ConcreteMesonBaryonToResonance::IsInCharge compares
      // G4ParticleTypeConverter generic types and a Delta++ is D1232 and not NUCLEON, so it
      // rejects it. The composite is therefore ELASTIC-ONLY here, and that is the only pair in
      // this sweep where component 0 is identically zero.
      {"pip_dpp", G4PionPlus::PionPlusDefinition(),
       G4ParticleTable::GetParticleTable()->FindParticle(2224)}};

  std::vector<G4VCollision*> comps;
  comps.push_back(new G4CollisionMesonBaryonToResonance());
  comps.push_back(new G4CollisionMesonBaryonElastic());
  // The parent composite, so the buffered TOTAL - the number G4Scatterer turns into a radius -
  // is compared and not merely the two partials that go into its nodes.
  G4CollisionMesonBaryon parent;

  auto* eng = new ImrCycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();

  for (const Pair& pr : pairs) {
    const double m1 = pr.a->GetPDGMass();
    const double m2 = pr.b->GetPDGMass();
    // Up to 3 GeV, not 1.5: G4XMesonBaryonElastic is identically zero below 2 GeV/c lab momentum
    // (docs/RISK.md V107), which for a pion is T = 1865 MeV, so a sweep that stopped at 1.5 GeV
    // compared a partial that was zero in every row and could not tell a buffered elastic from a
    // raw one. MEASURED: with the range at 1500 MeV, buffering the elastic partial - which is
    // wrong, G4CollisionMesonBaryonElastic has a real cross-section source - passed 1500 of 1500.
    for (double t = 10.0; t <= 3000.0; t += 10.0) {
      const double e1 = t + m1;
      const G4LorentzVector q1(G4ThreeVector(0, 0, std::sqrt(e1 * e1 - m1 * m1)), e1);
      const G4LorentzVector q2(G4ThreeVector(0, 0, 0), m2);
      G4KineticTrack t1(pr.a, 0.0, G4ThreeVector(0, 0, 0), q1);
      G4KineticTrack t2(pr.b, 0.0, G4ThreeVector(0, 0, 0), q2);
      const double s = (t1.Get4Momentum() + t2.Get4Momentum()).mag();
      double partial[2];
      const double total = parent.CrossSection(t1, t2);
      for (int i = 0; i < 2; ++i) {
        partial[i] = comps[i]->IsInCharge(t1, t2) ? comps[i]->CrossSection(t1, t2) : 0.0;
        std::fprintf(f, "%s,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g\n", pr.name, s, q1.z(), q1.t(),
                     q2.t(), i, partial[i] / millibarn, total / millibarn);
      }
      CLHEP::HepRandom::setTheEngine(eng);
      for (int phase = 0; phase < 8; ++phase) {
        eng->reset(phase);
        double sum = partial[0] + partial[1];
        const double random = G4UniformRand() * sum;
        double running = 0.0;
        int selected = -1;
        for (int i = 0; i < 2; ++i) {
          running += partial[i];
          if (running > random) { selected = i; break; }
        }
        std::fprintf(g, "%s,%.17g,%.17g,%.17g,%.17g,%d,%d,%d\n", pr.name, s, q1.z(), q1.t(),
                     q2.t(), phase, selected, eng->draws());
      }
      CLHEP::HepRandom::setTheEngine(saved);
    }
  }
  for (auto* c : comps) { delete c; }
  delete eng;
  std::fclose(f);
  std::fclose(g);
}

// ---------------------------------------------------------------------------------------------
// G4BinaryCascade::Propagate, called directly, with the random stream RECORDED
// ---------------------------------------------------------------------------------------------
//
// `Propagate` is public and virtual, so the whole cascade can be run from here on a nucleus and
// a secondary list this file built. That is the only way to get an EXACT per-call oracle for a
// loop where every turn consumes random numbers: there is no intermediate quantity to compare,
// only the products, and the products depend on every draw that came before them.
//
// ## THE TAPE IS WHAT MAKES THE COMPARISON BITWISE
//
// A prescribed cycle cannot drive this. Eight values put every nucleon of the nucleus at
// essentially the same place, and they hang `BetaKopylov` outright (docs/RISK.md V157). So the
// engine here is CLHEP's own `HepJamesRandom` at a fixed seed, wrapped so that every `flat()` it
// serves is appended to a vector, and the vector is written out with the case. The port's test
// replays it: its `Rng::uniform()` hands back `tape[i++]`.
//
// That is a far stronger check than a histogram. If the port draws a different NUMBER of
// uniforms anywhere - one extra rejection, one missing `SampleResidualLifetime`, a
// `FindCollisions` called before the list was updated instead of after - then its very next
// number is somebody else's and every product after that point is wrong. The draw count goes out
// with the row, so a divergence says WHERE.
//
// ## THE NUCLEUS IS DUMPED, NOT REPLAYED
//
// `G4Fancy3DNucleus::Init` consumes thousands of uniforms for lead and P9 has already validated
// it bitwise. So the tape starts AFTER Init and the nucleus goes out as data - one row per
// nucleon, position and momentum - which the port loads directly. Everything else about the
// nucleus is a deterministic function of (A, Z): the density, the Fermi momentum table and
// `nucleondistance`. This block is therefore a test of the CASCADE and of nothing else.
//
// ## THE PRECOMPOUND MODEL IS REPLACED BY ONE THAT RECORDS AND RETURNS NOTHING
//
// `G4BinaryCascade`'s constructor takes a `G4VPreCompoundModel*`, so `DeExcite`'s exit can be a
// model of this file's own: `ImrCapturePreco` writes down the `G4Fragment` it is handed - A, Z,
// the excitation energy, the three exciton counters and the four-momentum - and returns an EMPTY
// product vector. What comes back from `Propagate` is then the CASCADE's final state alone, which
// is exactly the boundary the port draws: `propagate` takes the de-excitation as a parameter so
// that P6 can be called behind `__noinline__`, and P6's own output has its own oracle.
//
// ## THE IMPACT PARAMETER IS SAMPLED FROM THE TAPE, AND SO IS THE RETRY
//
// A single impact parameter is not enough to get a cascade, and that is not a defect of this
// sweep - it is the reason `ApplyYourself` has a 200-try inner loop. MEASURED: a 400 MeV proton
// aimed at the exact centre of C12 finds NO collision at all. The twelve nucleons sit at
// transverse distances of 1.03 to 2.63 fermi from the axis, and `GetTimeToInteraction` needs
// `b*b <= sigma/pi`, which for a 25 mb pp cross section is `b < 0.89` fermi. Nothing is that
// close. The first version of this block ran 120 central-ish cases and every one of them came
// back NULL with zero draws.
//
// So the sweep runs `ApplyYourself`'s INNER loop, exactly as the source writes it: resample the
// position, build the track, call `Propagate`, and stop when it returns something. The tape
// therefore holds the FAILED tries as well - two uniforms per rejection inside `GetSpherePoint`,
// two more per accepted pair, and nothing at all from a `Propagate` that found no collision -
// and the port has to reproduce all of them to stay in step. `ntries` goes out with the row.
//
// `GetSpherePoint` itself is private, so `imr_sphere_point` below rebuilds it from its public
// ingredients - `Hep3Vector::orthogonal()`, the cross product, and the same rejection loop on
// the same engine - which is the pattern `write_imr_pauli` uses and explains.

/// CLHEP's HepJamesRandom, with every value it serves recorded in order.
class ImrTapeEngine : public CLHEP::HepRandomEngine {
 public:
  explicit ImrTapeEngine(long seed) { base_.setSeed(seed, 0); }
  void restart(long seed) {
    base_.setSeed(seed, 0);
    tape_.clear();
  }
  const std::vector<double>& tape() const { return tape_; }
  double flat() override {
    const double v = base_.flat();
    tape_.push_back(v);
    return v;
  }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long s, int i) override { base_.setSeed(s, i); }
  void setSeeds(const long* s, int i) override { base_.setSeeds(s, i); }
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "ImrTapeEngine"; }

 private:
  CLHEP::HepJamesRandom base_;
  std::vector<double> tape_;
};

/// A `G4VPreCompoundModel` that records the fragment and de-excites nothing.
class ImrCapturePreco : public G4VPreCompoundModel {
 public:
  ImrCapturePreco() : G4VPreCompoundModel(nullptr, "ImrCapturePreco") {}
  G4HadFinalState* ApplyYourself(const G4HadProjectile&, G4Nucleus&) override { return nullptr; }
  void DeExciteModelDescription(std::ostream&) const override {}
  G4ReactionProductVector* DeExcite(G4Fragment& frag) override {
    called = true;
    a = frag.GetA_asInt();
    z = frag.GetZ_asInt();
    u = frag.GetExcitationEnergy();
    holes = frag.GetNumberOfHoles();
    particles = frag.GetNumberOfParticles();
    charged = frag.GetNumberOfCharged();
    mom = frag.GetMomentum();
    return new G4ReactionProductVector();
  }
  void reset() {
    called = false;
    a = z = holes = particles = charged = 0;
    u = 0.0;
    mom = G4LorentzVector(0, 0, 0, 0);
  }
  bool called = false;
  int a = 0, z = 0, holes = 0, particles = 0, charged = 0;
  double u = 0.0;
  G4LorentzVector mom;
};

/// `G4BinaryCascade::GetSpherePoint`, re-expressed from its public ingredients, drawing from
/// whichever engine is installed - see the block comment above.
///
/// The 1.5 is the source's. Its own comment beside it says "plus -1*r*mom->vect()->unit()",
/// i.e. one radius back, and the line says one and a half. docs/RISK.md V162.
G4ThreeVector imr_sphere_point(double r, const G4LorentzVector& mom4) {
  const G4ThreeVector mom = mom4.vect();
  const G4ThreeVector o1 = mom.orthogonal();
  const G4ThreeVector o2 = mom.cross(o1);
  double x1 = 0.0;
  double x2 = 0.0;
  do {
    x1 = (G4UniformRand() - .5) * 2;
    x2 = (G4UniformRand() - .5) * 2;
  } while (x1 * x1 + x2 * x2 > 1.);
  return G4ThreeVector(r * (x1 * o1.unit() + x2 * o2.unit() - 1.5 * mom.unit()));
}

void write_imr_propagate() {
  FILE* f = std::fopen("bic_imr_prop.csv", "w");
  std::fprintf(f, "case,a,z,pdg,ekin,ntries,posx,posy,posz,px,py,pz,e,ndraws,nprod,"
                  "fragcalled,fraga,fragz,fragu,fragholes,fragparticles,fragcharged,"
                  "fragpx,fragpy,fragpz,frage,bicid,ncand,tmin\n");
  FILE* g = std::fopen("bic_imr_propnuc.csv", "w");
  std::fprintf(g, "case,i,pdg,x,y,z,px,py,pz,e\n");
  FILE* h = std::fopen("bic_imr_proptape.csv", "w");
  std::fprintf(h, "case,i,u\n");
  FILE* dbg = std::fopen("bic_imr_propdiag.csv", "w");
  std::fprintf(dbg, "case,ntry,pdg,dx_fm,dy_fm,dz_fm,sigma_mb,t\n");
  FILE* q = std::fopen("bic_imr_propfs.csv", "w");
  std::fprintf(q, "case,i,pdg,px,py,pz,e,newlyadded,creatorid\n");

  const G4ParticleDefinition* kBeam[] = {
      G4Proton::ProtonDefinition(), G4Neutron::NeutronDefinition(),
      G4PionPlus::PionPlusDefinition(), G4PionMinus::PionMinusDefinition()};
  // Two energies per species rather than the campaign's four: this block is the BITWISE check
  // and one tape is thousands of numbers, so the grid here is small and the campaign's
  // statistics are what covers the range.
  const double kEkin[] = {400.0, 800.0};

  auto* preco = new ImrCapturePreco();
  auto* bic = new G4BinaryCascade(preco);
  auto* eng = new ImrTapeEngine(20260918L);
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  const int bicid = G4PhysicsModelCatalog::GetModelID("model_G4BinaryCascade");

  int icase = 0;
  for (const Nuclide& t : kReplay) {
    for (const G4ParticleDefinition* d : kBeam) {
      for (double ekin : kEkin) {
        // The nucleus is built OUTSIDE the tape and LEAKED, for the reason write_imr_pauli
        // gives at length: G4RKPropagation keeps the pointer and reads it for the whole run.
        CLHEP::HepRandom::setTheEngine(saved);
        CLHEP::HepRandom::setTheSeed(730000L + 17 * icase);
        auto* nucleus = new G4Fancy3DNucleus;
        nucleus->Init(t.a, t.z);

        const double m = d->GetPDGMass();
        const double e = ekin + m;
        const G4LorentzVector p4(G4ThreeVector(0, 0, std::sqrt(e * e - m * m)), e);
        const double radius = nucleus->GetOuterRadius() + 3 * fermi;

        // The nucleus as Propagate will see it, before anything has touched it.
        nucleus->StartLoop();
        G4Nucleon* nuc = nullptr;
        int in = 0;
        while ((nuc = nucleus->GetNextNucleon()) != nullptr) {
          const G4LorentzVector& nm = nuc->GetMomentum();
          std::fprintf(g, "%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", icase, in,
                       nuc->GetDefinition()->GetPDGEncoding(), nuc->GetPosition().x(),
                       nuc->GetPosition().y(), nuc->GetPosition().z(), nm.x(), nm.y(), nm.z(),
                       nm.t());
          ++in;
        }

        preco->reset();
        eng->restart(20260918L + icase);
        CLHEP::HepRandom::setTheEngine(eng);

        // `ApplyYourself`'s INNER loop, verbatim: sample the position, run the cascade, and stop
        // when `Propagate` returns something. The nucleus is NOT rebuilt - that is the outer
        // loop, and its ingredient is P9's oracle.
        G4ReactionProductVector* products = nullptr;
        G4ThreeVector pos(0, 0, 0);
        int ncand = 0;
        double tmin = DBL_MAX;
        int ntries = 0;
        int collision_loop_max_count = 200;
        do {
          pos = imr_sphere_point(1.1 * radius, p4);
          ++ntries;
          // `FindCollisions` for a nucleon or pion projectile reduces to
          // `G4Scatterer::GetCollisions`, and THAT is public: the number of target nucleons with
          // a finite `GetTimeToInteraction` is what `theCollisionMgr->Entries()` will be, and the
          // earliest of their times is what `GetNextCollision` will return. Both go out with the
          // row, so a case that found nothing says whether the port disagrees about the COUNT or
          // about what happened afterwards. `GetTimeToInteraction` draws nothing, so asking it
          // here does not move the tape. The target tracks are put on mass shell the way
          // `BuildTargetList` puts them.
          ncand = 0;
          tmin = DBL_MAX;
          {
            G4KineticTrack probe(d, 0.0, pos, p4);
            probe.SetState(G4KineticTrack::outside);
            nucleus->StartLoop();
            G4Nucleon* tn = nullptr;
            while ((tn = nucleus->GetNextNucleon()) != nullptr) {
              G4LorentzVector tm = tn->GetMomentum();
              const double tmass = tn->GetDefinition()->GetPDGMass();
              tm.setE(std::sqrt(tm.vect().mag2() + tmass * tmass));
              G4KineticTrack tt(tn->GetDefinition(), 0.0, tn->GetPosition(), tm);
              tt.SetState(G4KineticTrack::inside);
              const double tc = imr_scatterer().GetTimeToInteraction(probe, tt);
              if (dbg != nullptr && ntries <= 3) {
                const G4ThreeVector dp = tt.GetPosition() - probe.GetPosition();
                std::fprintf(dbg, "%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n", icase, ntries,
                             tn->GetDefinition()->GetPDGEncoding(), dp.x() / fermi,
                             dp.y() / fermi, dp.z() / fermi,
                             imr_scatterer().GetCrossSection(probe, tt) / millibarn,
                             (tc < DBL_MAX) ? tc : -1.0);
              }
              if (tc < DBL_MAX) {
                ++ncand;
                if (tc < tmin) { tmin = tc; }
              }
            }
          }
          auto* secondaries = new G4KineticTrackVector;
          auto* kt = new G4KineticTrack(d, 0.0, pos, p4);
          kt->SetState(G4KineticTrack::outside);
          secondaries->push_back(kt);
          products = bic->Propagate(secondaries, nucleus);
        } while (!products && --collision_loop_max_count > 0);
        CLHEP::HepRandom::setTheEngine(saved);

        const int np = (products != nullptr) ? static_cast<int>(products->size()) : -1;
        std::fprintf(f, "%d,%d,%d,%d,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                        "%.17g,%d,%d,%d,%d,%d,%.17g,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%d,"
                        "%.17g\n",
                     icase, t.a, t.z, d->GetPDGEncoding(), ekin, ntries, pos.x(), pos.y(),
                     pos.z(), p4.x(), p4.y(), p4.z(), p4.t(),
                     static_cast<int>(eng->tape().size()), np, preco->called ? 1 : 0, preco->a,
                     preco->z, preco->u, preco->holes, preco->particles, preco->charged,
                     preco->mom.x(), preco->mom.y(), preco->mom.z(), preco->mom.t(), bicid,
                     ncand, (tmin < DBL_MAX) ? tmin : -1.0);
        const std::vector<double>& tape = eng->tape();
        for (size_t k = 0; k < tape.size(); ++k) {
          std::fprintf(h, "%d,%d,%.17g\n", icase, static_cast<int>(k), tape[k]);
        }
        for (int k = 0; k < np; ++k) {
          const G4ReactionProduct* rp = (*products)[k];
          std::fprintf(q, "%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%d\n", icase, k,
                       rp->GetDefinition()->GetPDGEncoding(), rp->GetMomentum().x(),
                       rp->GetMomentum().y(), rp->GetMomentum().z(), rp->GetTotalEnergy(),
                       rp->GetNewlyAdded() ? 1 : 0, rp->GetCreatorModelID());
        }
        if (products != nullptr) {
          for (auto* rp : *products) { delete rp; }
          delete products;
        }
        ++icase;
      }
    }
  }
  delete eng;
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
  std::fclose(g);
  std::fclose(h);
  std::fclose(q);
  std::fclose(dbg);
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
  write_imr_resonance();
  write_imr_clebsch();
  write_imr_meson();
  write_imr_resonance_xsec();
  write_imr_nnchannels();
  write_imr_resonance_fs();
  write_imr_annih();
  write_imr_mbselect();
  write_imr_decay();
  write_imr_decayfs();
  write_imr_absorb();
  write_imr_kdecay();
  write_imr_scatter();
  write_imr_ionmass();
  write_imr_pauli();
  write_imr_capture();
  write_imr_boundary();
  write_imr_fps();
  write_imr_propagate();
  // LAST, always: it destroys a G4Scatterer on purpose and empties the static channel list
  // every other sweep in this file depends on. See its own header and docs/RISK.md V155.
  write_imr_scatterlife();
}

}  // namespace

G4GPU_REGISTER_DUMP("bic",
                    "bic_limits.csv bic_density.csv bic_density_radius.csv bic_fermi.csv "
                    "bic_nucleus.csv bic_nucleons.csv bic_field.csv bic_rk.csv "
                    "bic_nucleus_stats.csv bic_nucleus_moments.csv bic_blir.csv "
                    "bic_blir_status.csv bic_apply.csv bic_apply_status.csv "
                    "bic_imr_xsec.csv bic_imr_angular.csv bic_imr_angular_sweep.csv bic_imr_obe.csv "
                    "bic_imr_collision.csv bic_imr_elastic_fs.csv "
                    "bic_imr_scatterer.csv bic_imr_manager.csv "
                    "bic_imr_restab.csv bic_imr_dbi.csv bic_imr_clebsch.csv "
                    "bic_imr_meson.csv bic_imr_meson_fs.csv "
                    "bic_imr_resxsec.csv bic_imr_species.csv "
                    "bic_imr_nnpartial.csv bic_imr_nnselect.csv bic_imr_nnbuffer.csv "
                    "bic_imr_resfs.csv bic_imr_annihfs.csv bic_imr_annih.csv "
                    "bic_imr_mbpartial.csv bic_imr_mbselect.csv "
                    "bic_imr_decaytable.csv bic_imr_actualwidth.csv bic_imr_k0flip.csv "
                    "bic_imr_lifetime.csv bic_imr_decayfs.csv "
                    "bic_imr_absorb.csv bic_imr_absorbfs.csv bic_imr_absorbcluster.csv "
                    "bic_imr_kdecay.csv bic_imr_scatter.csv bic_imr_scatterlife.csv "
                    "bic_imr_ionmass.csv bic_imr_pauli.csv "
                    "bic_imr_capture.csv bic_imr_absorbcut.csv bic_imr_capturefield.csv "
                    "bic_imr_boundary.csv bic_imr_prop.csv bic_imr_propnuc.csv "
                    "bic_imr_proptape.csv bic_imr_propfs.csv bic_imr_propdiag.csv",
                    dump_bic);
