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
#include "G4BinaryCascade.hh"
#include "G4BinaryLightIonReaction.hh"
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
#include "G4SystemOfUnits.hh"
#include "G4VNuclearDensity.hh"
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

void dump_bic(const DumpContext&) {
  write_limits();
  write_density();
  write_fermi();
  write_nucleus_and_nucleons();
  write_field_and_rk();
  write_nucleus_stats();
  write_blir();
}

}  // namespace

G4GPU_REGISTER_DUMP("bic",
                    "bic_limits.csv bic_density.csv bic_density_radius.csv bic_fermi.csv "
                    "bic_nucleus.csv bic_nucleons.csv bic_field.csv bic_rk.csv "
                    "bic_nucleus_stats.csv bic_nucleus_moments.csv bic_blir.csv "
                    "bic_blir_status.csv",
                    dump_bic);
