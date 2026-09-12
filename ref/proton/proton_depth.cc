// Proton depth-dose in water: the transport-level check for the hadron stepper.
//
// **One source file, two builds.** This compiles against the real Geant4 11.1.1 install and
// against this port's Geant4-shaped headers, and both write depth_dose.csv in the same format.
// That is the whole point: every other transport comparison in this repository (docs/RESULT.md)
// runs *different* code on the two sides and trusts that the two descriptions of the geometry
// and the beam agree. Here they cannot disagree, because there is one description.
//
// What it measures is the thing a proton calculation exists to produce: where the Bragg peak
// is and what the dose looks like on the way to it. Every piece of the hadron physics feeds
// into that, and each in a way the depth-dose curve distinguishes:
//
//   the range table          -> where the peak sits
//   the restricted dE/dx     -> how high the plateau is
//   delta-ray production     -> how much of the plateau dose is carried by electrons, and so
//                               how far from the track it lands
//   nuclear stopping         -> the last few microns, worth ~0.1% of the total
//   fluctuations             -> the width of the distal falloff, and *only* that. It is the
//                               one metric here that no other part of the physics touches.
//
// **Physics list: QBBC on both sides, with exactly what the port lacks inactivated on the
// Geant4 side.** It was `G4EmStandardPhysics` and nothing else, and the note here said "this
// port has no hadronic physics at all, so comparing against QBBC would measure that absence
// rather than the stepper". That stopped being true when P8 wired `G4Decay` and P8b wired
// `hadElastic` and `CoulombScat`: the port now has a hadronic process acting on the proton, and
// a reference taken without one is a different experiment. `build_all.bat`'s gate failed on
// exactly that - the plateau 1.32% high, which is what elastic scattering is worth here.
//
// So the rule docs/HADRONIC_PLAN.md section 4 states for every comparison applies to this one
// too: **inactivate, on the Geant4 side, ONLY what the port still lacks.** The list is in
// `inactivate()` below, each line with the package that closes it, and the run PRINTS
// `/particle/process/dump` for the proton so the configuration is recorded by what ran rather
// than by what was intended (docs/RISK.md V43: `/process/inactivate` ignores a name the species
// does not have, silently).
//
// What is left ACTIVE on both sides: `msc`, `hIoni`, `ionIoni`, `hadElastic`, `CoulombScat`,
// `Decay`, and the whole of `G4EmStandardPhysics` for the electrons and gammas.
//
// **Known omissions, stated up front.** The inelastic reactions (P9-P11) and the ion's own
// `ionElastic` (`had::ElasticChannel::kIonDiffuseNotWired`) are off on both sides. And the
// port's Bragg peak sits about 0.23 mm proximal of Geant4's, a 0.3% range difference that
// docs/RISK.md V5 records in detail, including the seven things it has been measured *not*
// to be.
//
// Read the four metrics separately, which is what tools/compare_depth.ps1 does. A stopping
// power 1% high and a range table 1% long cancel in the plateau and add in R80, so a single
// aggregate agreement figure would pass a calculation that is wrong twice.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4PSEnergyDeposit.hh"
#include "G4SDManager.hh"
#include "G4UImanager.hh"
#include "QBBC.hh"
#include "Randomize.hh"

// ---------------------------------------------------------------- the phantom, in one place
//
// Both builds read these, so the two geometries cannot drift apart.
namespace cfg {
constexpr int kMaxBins = 400;          ///< storage bound; the run picks how many it uses
/// mm, half-width of a slab transverse to the beam.
///
/// 150 AND NOT 50, AND THAT IS WHAT KEEPS THE CONSERVATION CHECK EXACT once `hadElastic` is in
/// the reference. `tools/compare_depth.ps1`'s first metric is energy in against energy
/// deposited, with a 1e-6 limit and the comment "it should be exact on both sides - the phantom
/// is deeper than the range". Deeper is not wider: an elastic scatter off oxygen leaves a
/// 100 MeV proton with nearly all of its energy at any angle, so one scattered near 90 degrees
/// runs its whole 77 mm range sideways and left a 50 mm half-width phantom carrying it. At
/// 100,000 events that was 112 MeV of 10,000,000 - 1.1e-5, eleven times the limit - and it is
/// energy Geant4 genuinely transported out of the box rather than anything wrong with either
/// side.
///
/// So the phantom is widened to hold it rather than the limit widened to excuse it (the
/// package's own rule for the plateau, applied here). 150 mm exceeds a 100 MeV proton's range,
/// so a proton scattered at any angle stops inside; it stays inside the 200 mm world. Measured:
/// the deposited total goes from 99.9989% of the beam energy to 100.0000%.
///
/// What it changes about the curve is the same 1.1e-5, and in the direction of a standard
/// integral depth dose: the energy that used to leave is now binned at the depth it was
/// scattered from.
/// RUNTIME SINCE P14c, AND THE TWO REASONS ARE THE SAME NUMBER SEEN TWICE.
///
/// The file was a proton harness: 100 mm of water holds a 100 MeV proton's 77 mm range with
/// room to spare, and 150 mm of half-width holds an elastically scattered one (the paragraph
/// above). An ELECTRON asks both questions again and gets different answers. A 1 GeV electron
/// in water is a shower, not a track: X0 is 360.8 mm, the shower maximum sits near 2 X0 and
/// containment wants of order 20 X0 longitudinally, while the transverse scale is the Moliere
/// radius, 21 MeV / Ec * X0, about 92 mm - so 95% of the energy is inside 2 R_M and a 150 mm
/// half-width is marginal rather than generous.
///
/// So they are arguments now, and `depth_mm` with them, rather than a second copy of this file
/// with three constants changed. The DEFAULTS are the proton's, exactly, so that
/// `build_all.bat`'s existing gate runs the same phantom it has always run - a depth-dose
/// comparison whose geometry moved is not a comparison.
inline double& half_xy() { static double v = 150.0; return v; }
inline double& world_half() { static double v = 200.0; return v; }
inline double& depth() { static double v = 100.0; return v; }

// Slab thickness and count are runtime, not compile-time, and that is worth the small
// awkwardness: the phantom's own slab boundaries chop every step, so the binning is not just
// a readout - it is part of the transport. Being able to re-bin the same phantom depth without
// recompiling is how "does the answer depend on the step size?" gets asked of *both* sides at
// once, which is a question no amount of reading either stepper answers.
inline int& bins() { static int v = 200; return v; }
inline double& slab() { static double v = 0.5; return v; }
inline double depth_lo(int i) { return i * slab(); }
inline double depth_hi(int i) { return (i + 1) * slab(); }
}  // namespace cfg

// The bins. A global because the stepping action, the detector construction and main all need
// it and threading it through three Geant4 base classes would add nothing.
static double g_edep[cfg::kMaxBins];
static std::vector<G4LogicalVolume*> g_slab_lv;

// ---------------------------------------------------------------- geometry
class Phantom : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    G4Material* vac = nist->FindOrBuildMaterial("G4_Galactic");
    G4Material* water = nist->FindOrBuildMaterial("G4_WATER");

    auto* world_box = new G4Box("World", cfg::world_half() * mm, cfg::world_half() * mm,
                                cfg::world_half() * mm);
    auto* world_lv = new G4LogicalVolume(world_box, vac, "World");
    auto* world = new G4PVPlacement(nullptr, G4ThreeVector(), world_lv, "World", nullptr, false,
                                    0, false);

    // One logical volume per slab rather than one replicated: the port keys its per-volume
    // aggregate step on the physical volume's own logical volume, and this way the stepping
    // action below can find the bin by pointer identity in both builds.
    auto* slab_box = new G4Box("Slab", cfg::half_xy() * mm, cfg::half_xy() * mm,
                               0.5 * cfg::slab() * mm);
    g_slab_lv.resize(cfg::bins());
    for (int i = 0; i < cfg::bins(); ++i) {
      char name[32];
      std::snprintf(name, sizeof name, "Slab%03d", i);
      auto* lv = new G4LogicalVolume(slab_box, water, name);
      g_slab_lv[i] = lv;
      const double zc = 0.5 * (cfg::depth_lo(i) + cfg::depth_hi(i));
      new G4PVPlacement(nullptr, G4ThreeVector(0, 0, zc * mm), lv, name, world_lv, false, i,
                        false);
    }
    return world;
  }

  void ConstructSDandField() override {
    // A sensitive detector per slab. On the real Geant4 side this is what makes the slab's
    // hits real; on the port's side it is what makes the volume *scored*, and only a scored
    // volume produces the per-event aggregate step the action below reads.
    for (int i = 0; i < cfg::bins(); ++i) {
      char name[48];
      std::snprintf(name, sizeof name, "slabSD%03d", i);
      auto* det = new G4MultiFunctionalDetector(name);
      G4SDManager::GetSDMpointer()->AddNewDetector(det);
      det->RegisterPrimitive(new G4PSEnergyDeposit("edep"));
      SetSensitiveDetector(g_slab_lv[i], det);
    }
  }
};

// ---------------------------------------------------------------- the beam
class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  Gun() : gun_(new G4ParticleGun(1)) {
    // THE NAME IS LOOKED UP AND THE LOOKUP IS CHECKED. `FindParticle` returns nullptr for a
    // name the table does not carry and `SetParticleDefinition(nullptr)` is a G4Exception on
    // the real Geant4 and a silent null on the port's shim, so a typo would otherwise be a
    // crash on one side and a run with no primaries on the other. The two builds must fail the
    // same way or the harness is not one harness.
    auto* def = G4ParticleTable::GetParticleTable()->FindParticle(particle_);
    if (def == nullptr) {
      std::printf("FATAL: no particle named '%s'\n", particle_.c_str());
      std::exit(2);
    }
    gun_->SetParticleDefinition(def);
    gun_->SetParticleEnergy(energy_ * MeV);
    gun_->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    // Just outside the first slab, in vacuum, so the full energy enters the water.
    gun_->SetParticlePosition(G4ThreeVector(0, 0, -1.0 * mm));
  }
  ~Gun() override { delete gun_; }
  void GeneratePrimaries(G4Event* evt) override { gun_->GeneratePrimaryVertex(evt); }
  static void SetEnergy(double e) { energy_ = e; }
  static void SetParticle(const std::string& p) { particle_ = p; }
  static const std::string& Particle() { return particle_; }

 private:
  G4ParticleGun* gun_;
  static double energy_;
  static std::string particle_;
};
double Gun::energy_ = 100.0;
std::string Gun::particle_ = "proton";

// ---------------------------------------------------------------- the binning
//
// Identical on both sides, and deliberately written to use only what the port's aggregate step
// carries: the deposit and which volume it happened in. It must NOT use the step's position -
// the port hands out one step per volume per event whose position is not a point on any track.
// See the note at the top of g4/G4Step.hh.
class Binner : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const G4double e = step->GetTotalEnergyDeposit();
    if (e <= 0) { return; }
    const auto* touch = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
    if (touch == nullptr) { return; }
    G4LogicalVolume* lv = touch->GetLogicalVolume();
    for (int i = 0; i < cfg::bins(); ++i) {
      if (g_slab_lv[i] == lv) {
        g_edep[i] += e / MeV;
        return;
      }
    }
  }
};

class Actions : public G4VUserActionInitialization {
 public:
  void Build() const override {
    SetUserAction(new Gun());
    SetUserAction(new Binner());
  }
};

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
  const int n_events = (argc > 1) ? std::atoi(argv[1]) : 20000;
  const double energy = (argc > 2) ? std::atof(argv[2]) : 100.0;
  const char* out = (argc > 3) ? argv[3] : "depth_dose.csv";
  // The production range cut, in mm. A parameter because it is the one knob that
  // switches delta-ray production off on *both* sides at once: raise it until the cut
  // energy exceeds the maximum transfer and no proton can make a transportable
  // electron. That is how a discrepancy gets attributed to the delta rays or ruled out.
  const double range_cut = (argc > 4) ? std::atof(argv[4]) : 0.7;
  // Slab thickness in mm; the count is chosen to fill the phantom depth.
  const double slab = (argc > 5) ? std::atof(argv[5]) : 0.5;
  // The species, by name, and the phantom it needs. See cfg::half_xy above for why an electron
  // needs a different phantom from a proton and why that is an argument rather than a fork.
  const std::string particle = (argc > 6) ? argv[6] : "proton";
  const double depth_mm = (argc > 7) ? std::atof(argv[7]) : 100.0;
  const double half_xy = (argc > 8) ? std::atof(argv[8]) : 150.0;
  cfg::slab() = slab;
  cfg::depth() = depth_mm;
  cfg::half_xy() = half_xy;
  // The world holds the phantom with the same 50 mm of clearance the proton geometry had.
  cfg::world_half() = std::max(depth_mm, half_xy) + 50.0;
  cfg::bins() = static_cast<int>(depth_mm / slab + 0.5);
  if (cfg::bins() > cfg::kMaxBins) {
    std::printf("FATAL: %g mm of phantom in %g mm slabs is %d bins and the bound is %d.\n"
                "  A silently truncated phantom is a depth-dose curve that stops early and\n"
                "  says nothing about it, and the total-energy line below would then read as\n"
                "  a physics disagreement. Choose a coarser slab.\n",
                depth_mm, slab, cfg::bins(), cfg::kMaxBins);
    return 2;
  }
  Gun::SetEnergy(energy);
  Gun::SetParticle(particle);

  auto* rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);

  // QBBC on both sides. On the port's side that is its own QBBC shim, whose hadronic component
  // is whatever it has wired - `HadronicStage::kStage1`, decay, hadElastic and CoulombScat all
  // on, which is the engine's default and the configuration every stage-1 number is measured
  // in. On Geant4's side it is the real one, and the block after Initialize() is what makes the
  // two comparable.
  auto* phys = new QBBC();
  phys->SetDefaultCutValue(range_cut * mm);
  rm->SetUserInitialization(phys);
  rm->SetUserInitialization(new Phantom());
  rm->SetUserInitialization(new Actions());

#ifdef G4GPU_PORT
  // The port allocates n_scorers * batch doubles for per-event scores. With 200 scorers the
  // million-event default batch would be 1.6 GB, so the batch is sized to the run.
  rm->SetBatchSize((n_events < 5000) ? n_events : 5000);
#endif

  rm->Initialize();

#ifndef G4GPU_PORT
  // ---------------------------------------------------------------- the like-for-like list
  //
  // ONLY WHAT THE PORT LACKS, and every line names the package that closes it. This is
  // docs/HADRONIC_PLAN.md section 4's rule applied to this comparison: "Nothing is ever
  // compared against a Geant4 that is running physics the port does not have."
  //
  // `/process/inactivate <name>` takes no particle argument here, so it applies to every
  // species that has the process - which is what is wanted, because a proton's elastic recoils
  // are deuterons, tritons, He3s, alphas and heavier nuclei and each of those carries its own
  // inelastic process.
  //
  // IT ALSO SILENTLY IGNORES A NAME NOTHING HAS (docs/RISK.md V43), which is why the dump
  // below is printed rather than trusted: three of these names are here for species this run
  // cannot make, so that the list is the same in a run that can.
  {
    G4UImanager* ui = G4UImanager::GetUIpointer();
    const char* off[] = {
        // The radiative processes of a charged hadron. em/muon_radiative.cuh has both models'
        // dE/dx exactly and neither model's SampleSecondaries (docs/PORTED.md 1.3).
        "hBrems", "hPairProd", "muBrems", "muPairProd",
        // Every inelastic final state that has a name of its own: P9 (binary cascade),
        // P10 (Bertini), P11 (FTFP).
        "protonInelastic", "dInelastic", "tInelastic", "He3Inelastic", "alphaInelastic",
        "ionInelastic",
        // The neutron's three sub-processes are one process and /process/inactivate can only
        // take it whole (docs/RISK.md V53). P8c leaves step_neutral's cross section at zero, so
        // the whole process comes off. `neutronInelastic` is deliberately NOT in this list: the
        // UI answers `illegal process (or type) name` for it, because it is inside the general
        // process and is on no manager. No neutron is made in this configuration anyway -
        // nothing but an inelastic reaction produces one - so this line costs the comparison
        // nothing and keeps its statement true.
        "NeutronGeneralProc",
        // The ion's own elastic process. G4IonElasticPhysics gives GenericIon "ionElastic"
        // (G4ComponentGGNuclNuclXsc + G4NuclNuclDiffuseElastic); both halves are ported and the
        // channel is not wired - had::ElasticChannel::kIonDiffuseNotWired.
        "ionElastic",
        // Electro-, positron- and muon-nuclear: P13. These three the UI accepts.
        //
        // `photonNuclear` IS NOT HERE AND CANNOT BE, which is V53's mechanism a second time and
        // was found by the UI rejecting it. `G4EmStandardPhysics::ConstructProcess` calls
        // `SetGeneralProcessActive(true)`, so `G4EmExtraPhysics::ConstructGammaElectroNuclear`
        // takes its `gproc != nullptr` branch and does `gproc->AddHadProcess(gnuc)` instead of
        // `ph->RegisterProcess(gnuc, gamma)` - the gamma's photo-nuclear is a sub-process of
        // `G4GammaGeneralProcess`, exactly as the neutron's inelastic is of
        // `G4NeutronGeneralProcess`, and the only name that reaches it is `GammaGeneralProc`,
        // which would take Compton, the photoelectric effect, Rayleigh and conversion off with
        // it. Those this port HAS, so switching them off would break the rule this list exists
        // for. It is inert here and the arithmetic is why: nothing in this configuration makes
        // a photon above about 0.5 MeV (protonInelastic and hBrems are off, so the only
        // photons are the bremsstrahlung of a delta ray whose own energy is capped by the
        // proton's maximum transfer), and `G4GammaNuclearXS` is a giant-resonance cross
        // section that starts near 10 MeV.
        // For an ELECTRON beam these three are the whole of the like-for-like list that
        // bites: `electronNuclear` and `positronNuclear` are on the primary itself, and the
        // photo-nuclear reaction of its bremsstrahlung is inside `G4GammaGeneralProcess` and
        // cannot be inactivated alone (the paragraph above). A 1 GeV electron's photons DO
        // reach the giant resonance, unlike the proton configuration this list was written
        // for, so that omission is no longer inert and is stated in the report rather than
        // assumed away.
        "electronNuclear", "positronNuclear", "muonNuclear",
        // The at-rest captures: P12. Unreachable here (nothing negative is made) and listed so
        // that this list is the stage-1 one.
        "hBertiniCaptureAtRest", "hFritiofCaptureAtRest", "muMinusCaptureAtRest",
    };
    for (const char* p : off) {
      ui->ApplyCommand(G4String("/process/inactivate ") + p);
    }
    // The stage, recorded by what RAN. Printed for the proton and for GenericIon, which is the
    // species every elastic recoil heavier than an alpha is and the one P8c added transport
    // for; `hadElastic`, `CoulombScat`, `msc`, `hIoni`, `ionIoni` and `Decay` must read Active
    // in it and every name above InActive.
    ui->ApplyCommand("/particle/select proton");
    ui->ApplyCommand("/particle/process/dump");
    ui->ApplyCommand("/particle/select GenericIon");
    ui->ApplyCommand("/particle/process/dump");
  }
#endif

  G4Random::setTheSeed(12345);
  for (int i = 0; i < cfg::bins(); ++i) { g_edep[i] = 0; }
  rm->BeamOn(n_events);

  double total = 0;
  for (int i = 0; i < cfg::bins(); ++i) { total += g_edep[i]; }

  FILE* f = std::fopen(out, "w");
  if (f == nullptr) {
    std::printf("cannot write %s\n", out);
    return 1;
  }
  std::fprintf(f,
               "# events=%d energy_MeV=%g bins=%d slab_mm=%g cut_mm=%g total_MeV=%.9g "
               "particle=%s depth_mm=%g half_xy_mm=%g\n",
               n_events, energy, cfg::bins(), cfg::slab(), range_cut, total,
               Gun::Particle().c_str(), cfg::depth(), cfg::half_xy());
  std::fprintf(f, "bin,z_lo_mm,z_hi_mm,edep_MeV\n");
  for (int i = 0; i < cfg::bins(); ++i) {
    std::fprintf(f, "%d,%.4f,%.4f,%.9g\n", i, cfg::depth_lo(i), cfg::depth_hi(i), g_edep[i]);
  }
  std::fclose(f);

  std::printf("%d %s of %g MeV in %g mm of water (half-width %g mm) -> %s\n", n_events,
              Gun::Particle().c_str(), energy, cfg::depth(), cfg::half_xy(), out);
  std::printf("  total deposited %.6g MeV of %.6g MeV in  (%.4f%%)\n", total, n_events * energy,
              100.0 * total / (n_events * energy));
  delete rm;
  return 0;
}
