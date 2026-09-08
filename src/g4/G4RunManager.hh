// G4RunManager: owns the user initialisations, flattens the detector, and runs events.
//
// The call sequence a Geant4 user writes is preserved:
//
//     auto* rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
//     rm->SetUserInitialization(new DetectorConstruction);
//     rm->SetUserInitialization(new QBBC);
//     rm->SetUserInitialization(new ActionInitialization);
//     rm->Initialize();
//     rm->BeamOn(10000);
//
// What differs is where the work happens. Initialize() flattens the detector once and uploads
// it; BeamOn() runs the whole request on the device in batches.
//
// The per-event virtuals ARE called. Energy deposit is already accumulated per event on the
// device - the score buffer is one row per scorer and one column per event in the batch - so
// BeginOfEventAction, an aggregated step per scored volume, and EndOfEventAction cost a host
// virtual call per event and no device work at all. Read the note at the top of g4/G4Step.hh
// for what an aggregated step is and what it is not; the summary is that a stepping action
// which adds up what it is given gets exactly the right answer, and one that does anything
// else does not.
//
// The chain is only installed when a user action asks for it. Without an event or stepping
// action the run costs what it always did.
#pragma once
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>
#include "g4/G4Flatten.hh"
#include "g4/G4Step.hh"
#include "g4/G4UserActions.hh"
#include "g4/G4VModularPhysicsList.hh"
#include "g4/Randomize.hh"  // the constructor seeds the host engine from seed_ too
#include "host/transport_run.cuh"

/// The step hook the Geant4-shaped API runs on, and how a project chooses its own.
///
/// A StepHook is a device functor called once per real step of every track. It is the only way
/// to see a real step here: G4UserSteppingAction and G4VPrimitiveScorer::Accept are handed a
/// per-event aggregate instead, and G4Step.hh explains why. Write one by deriving from
/// G4VUserDeviceSteppingAction, which is the Geant4-shaped way in; core/step_hook.cuh is the
/// machinery underneath, and its memory note matters more than anything here, because the
/// decision that determines whether a step-level analysis scales is made in the hook.
///
/// The default is StepTap, which copies steps into a capped buffer and stays disabled by its
/// own null pointer until a caller fills it in. It costs the stock kernels one predicated load
/// per step and nothing else.
///
/// A PROJECT CHOOSES ITS OWN BY DEFINING G4STEP_HOOK, and does not edit this file or rebuild
/// g4gpu. That is the arrangement Geant4 has - the library is built once, a project compiles
/// against it - and it is what this macro is for. In one .cu of the project:
///
///     #include "MySteppingAction.hh"                 // your class
///     #define G4STEP_HOOK MySteppingAction           // before G4RunManager.hh
///     #include "g4/G4RunManager.hh"
///     #include "host/transport_run_impl.cuh"         // the kernels, in YOUR translation unit
///     namespace g4gpu::host {
///       template class TransportEngine<double, MySteppingAction>;
///     }
///
/// or equivalently with -DG4STEP_HOOK=MySteppingAction on the command line. The explicit
/// instantiation is the part that cannot be avoided: a hook is device code, so a new hook type
/// is a new kernel, and something has to compile it. What the macro buys is that the something
/// is the project rather than the engine.
///
/// One .cu, one instantiation - see the note at the top of transport_run_impl.cuh for what
/// happens if you do it in two.
#ifndef G4STEP_HOOK
#define G4STEP_HOOK g4gpu::StepTap<G4double>
#endif
using G4StepHook = G4STEP_HOOK;

class G4RunManager {
 public:
  /// Seeds BOTH random streams from seed_, so the state a process starts in is the state
  /// SetRandomSeed(seed_) would put it in - which is what makes "give me that run again"
  /// expressible at all.
  ///
  /// It was not, until it was asked for. CLHEP's engine began on its own default seed and the
  /// device stream on this class's, two unrelated numbers from unrelated code, so no argument
  /// to SetRandomSeed or /random/setSeeds could reproduce a freshly started process: the
  /// showers would replay and the primaries would not.
  ///
  /// Here rather than in Initialize() because a project that wants its own host seed writes
  /// G4Random::setTheSeed() in main() after constructing the run manager, as Geant4's examples
  /// do, and last writer wins. Doing it in Initialize() would silently outrank that.
  G4RunManager() {
    engine_ = new g4gpu::host::TransportEngine<G4double, G4StepHook>();
    Instance() = this;
    G4Random::setTheSeed(static_cast<long>(seed_));
  }
  ~G4RunManager() {
    engine_->Free();
    delete engine_;
  }

  static G4RunManager*& Instance() {
    static G4RunManager* inst = nullptr;
    return inst;
  }
  /// Geant4 spells the accessor GetRunManager(); examples use it from anywhere.
  static G4RunManager* GetRunManager() { return Instance(); }

  void SetUserInitialization(G4VUserDetectorConstruction* d) { detector_.reset(d); }
  void SetUserInitialization(G4VUserPhysicsList* p) {
    physics_.reset(p);
    if (p != nullptr) {
      processes_ = p->Flags();
      if (p->HasCutValue()) { range_cut_mm_ = p->GetDefaultCutValue() / mm; }
    }
  }
  void SetUserInitialization(G4VUserActionInitialization* a) { action_init_.reset(a); }

  void SetUserAction(G4VUserPrimaryGeneratorAction* a) { primary_.reset(a); }
  void SetUserAction(G4UserRunAction* a) { run_action_.reset(a); }
  void SetUserAction(G4UserEventAction* a) { event_action_.reset(a); }
  void SetUserAction(G4UserSteppingAction* a) { stepping_action_.reset(a); }
  /// Accepted, and never called: a track is created and killed inside a device kernel, where
  /// a host virtual cannot be reached. Saying so here is the alternative to accepting the
  /// registration and silently doing nothing with it.
  void SetUserAction(G4UserTrackingAction* a) {
    tracking_action_.reset(a);
    std::printf(
        "note: a G4UserTrackingAction is registered but will not be called. Tracks are\n"
        "      created and killed on the device. Per-track quantities are available as\n"
        "      primitive scorers (G4PSTrackLength, G4PSNofStep) or through the trajectory\n"
        "      buffer the viewer reads.\n");
  }
  void SetUserAction(G4UserStackingAction* a) {
    stacking_action_.reset(a);
    std::printf(
        "note: a G4UserStackingAction is registered but will not be called. The stack is a\n"
        "      pair of device buffers per species, drained breadth-first; there is no\n"
        "      per-track classification hook.\n");
  }

  const G4VUserDetectorConstruction* GetUserDetectorConstruction() const {
    return detector_.get();
  }
  const G4VUserPrimaryGeneratorAction* GetUserPrimaryGeneratorAction() const {
    return primary_.get();
  }
  const G4UserRunAction* GetUserRunAction() const { return run_action_.get(); }
  const G4VUserPhysicsList* GetUserPhysicsList() const { return physics_.get(); }

  /// Production range cut, as /run/setCut does. Must be set before Initialize().
  void SetCutValue(G4double range) { range_cut_mm_ = range / mm; }
  G4double GetCutValue() const { return range_cut_mm_ * mm; }

  /// How many events go to the device at once. Larger keeps the GPU fuller; the cost is
  /// memory, since track buffers are multiples of it.
  /// Events per batch. 0 - the default - lets the engine size it from the memory the device
  /// actually has free; see TransportEngine::Upload. Set a positive number to choose it
  /// yourself, which is honoured even if it will not fit, with a warning.
  ///
  /// After Initialize(), this reads back whatever was actually chosen rather than the 0 that
  /// asked for automatic - so a caller can check it, and code that needs one batch to hold a
  /// whole run can compare against a real number.
  /// The share of free device memory the track buffers may take. See
  /// TransportEngine::SetMemoryFraction. Set before Initialize().
  void SetMemoryFraction(G4double f) { engine_->SetMemoryFraction(f); }
  G4double GetMemoryFraction() const { return engine_->GetMemoryFraction(); }

  void SetBatchSize(G4int n) { batch_ = n; }
  G4int GetBatchSize() const { return batch_; }

  /// Which physics processes to run. All on is the validated default; see ProcessFlags. A
  /// physics list set through SetUserInitialization sets these, so call this after it.
  void SetProcesses(const g4gpu::ProcessFlags& f) { processes_ = f; }
  const g4gpu::ProcessFlags& GetProcesses() const { return processes_; }

  /// The counter-based key every track's random stream is derived from.
  ///
  /// Two runs with different seeds are independent samples. Two runs with the same seed AND at
  /// the same position in the stream are identical down to the last bit, whatever the batch
  /// size - which means a freshly started process replays, and a second BeamOn within one
  /// process does not. That is Geant4's behaviour and the reason for stream_pos_ below.
  ///
  /// Setting the seed restarts BOTH streams, as /random/setSeeds does and as reseeding
  /// Geant4's one engine does: the host engine that draws the primaries is reseeded, and the
  /// device position goes back to zero. So `SetRandomSeed(s)` twice in one process gives the
  /// same run twice, and `SetRandomSeed(GetRandomSeed())` returns a process to its start.
  ///
  /// Both, because half of it is worse than neither. Reseeding the showers and not the
  /// primaries gives a run that repeats one and not the other, which is neither a repeat nor
  /// an independent sample, and nothing in the output says which.
  void SetRandomSeed(unsigned int s) {
    seed_ = s;
    stream_pos_ = 0;
    G4Random::setTheSeed(static_cast<long>(s));
  }
  unsigned int GetRandomSeed() const { return seed_; }
  /// How far into the seed's stream the next run will start. Advances by one per primary.
  /// Exposed so a test can assert that it moved, which is the observable behind "two runs in
  /// one process are not the same run".
  long long GetRandomStreamPosition() const { return stream_pos_; }

  /// Accepted for source compatibility. There is no per-event seed file to write: a run is
  /// reproducible from its seed and event index alone.
  void SetRandomNumberStore(G4bool) {}
  void SetVerboseLevel(G4int v) { verbose_ = v; }
  G4int GetVerboseLevel() const { return verbose_; }
  void SetPrintProgress(G4int n) { print_progress_ = n; }

  void Initialize();

  /// Runs @p n_events. The macro and n_select arguments of Geant4's BeamOn select events for
  /// verbose printing, which has no equivalent here; they are accepted and ignored.
  void BeamOn(G4int n_events, const char* macro = nullptr, G4int n_select = -1);

  /// Configures the gun and runs n events, splitting between the sources it registered.
  /// Everything that runs events goes through this - see the definition for why.
  g4gpu::host::RunStats RunEvents(G4int n_events, std::vector<double>& sums,
                                  std::vector<double>& sums_sq,
                                  g4gpu::vis::TrajectoryBuffer traj = {},
                                  std::vector<int>* shares = nullptr);

  /// Energy deposit per voxel cell from the last run, accumulated over every sub-run.
  const std::vector<double>& VoxelScores() const { return run_voxel_scores_; }

  const G4Run* GetCurrentRun() const { return &run_; }
  const g4gpu::host::RunStats& GetLastRunStats() const { return last_stats_; }
  const g4gpu::g4::FlatScene& GetScene() const { return scene_; }
  g4gpu::host::TransportEngine<G4double, G4StepHook>& GetEngine() { return *engine_; }

  /// The per-step hook. Set it before BeamOn; see G4StepHook above.
  void SetStepHook(const G4StepHook& h) { engine_->SetStepHook(h); }
  G4StepHook& GetStepHook() { return engine_->GetStepHook(); }

  /// The mass of a scored volume, kg, for converting energy deposit to dose.
  ///
  /// Exact and analytic when nothing overlaps the scored volume. When a higher-layer volume
  /// does overlap it, the overlapping region belongs to that volume and its material, so the
  /// mass is integrated over the region this volume actually wins - by Monte Carlo, with the
  /// count reported. See the implementation for why the two cases are separated rather than
  /// always integrating.
  G4double ScoredMass(G4int scorer_index) const;

 private:
  /// Cached scored masses, one per scorer, filled on first request after Initialize().
  ///
  /// This is a cache because the uncached version was being called from a GUI panel that
  /// redraws every frame, and computing it can involve a 200,000-sample Monte Carlo over the
  /// geometry when a scored volume is overlapped by a higher layer. So every frame ran the
  /// integration and printed its note, which is what filled the terminal with what looked
  /// like a repeating error. The mass depends only on the geometry, and the geometry changes
  /// only in Initialize().
  mutable std::vector<G4double> mass_cache_;

 public:

  /// The gun the primary generator configured, so the messenger and the GUI can read it.
  g4gpu::Source<G4double>& GunSource() { return gun_source_; }
  void SetGun(G4ParticleGun* g) { gun_ = g; }
  G4ParticleGun* GetGun() const { return gun_; }

 private:
  /// Turns the device's per-event scores into the Geant4 event/step action chain.
  class ActionSink : public g4gpu::host::EventSink {
   public:
    ActionSink(G4UserEventAction* ea, G4UserSteppingAction* sa,
               const std::vector<G4PVPlacement*>& slot_pv,
               const std::vector<G4Material*>& slot_mat, G4ParticleDefinition* primary_def,
               G4double primary_ekin)
        : ea_(ea), sa_(sa), slot_pv_(slot_pv), slot_mat_(slot_mat) {
      track_.SetDefinition(primary_def);
      track_.SetKineticEnergy(primary_ekin);
      track_.SetTrackID(-1);
      step_.SetTrack(&track_);
    }

    void Event(int event_id, const double* scores, int n_scorers) override {
      const G4Event event(event_id);
      if (ea_ != nullptr) { ea_->BeginOfEventAction(&event); }

      // The per-event filter hook, for scorers that declare one. Before the stepping action,
      // because a scorer that drops an event should drop it for everything downstream too.
      // Only filtered scorers are touched: an unfiltered one's total comes from the device
      // array summed in a fixed order, and that is the number B1 is validated on.
      {
        auto& scorers = G4SDManager::GetSDMpointer()->Scorers();
        for (int sc = 0; sc < n_scorers && sc < static_cast<int>(scorers.size()); ++sc) {
          if (!scorers[sc]->IsFiltered()) { continue; }
          G4double v = scores[sc];
          if (!scorers[sc]->Accept(v, event_id)) { v = 0; }
          scorers[sc]->filtered_total += v;
          scorers[sc]->filtered_total_sq += v * v;
        }
      }

      if (sa_ != nullptr) {
        for (int sc = 0; sc < n_scorers; ++sc) {
          step_.SetTotalEnergyDeposit(scores[sc] * MeV);
          // Track length and step count come from a G4PSTrackLength or G4PSNofStep primitive
          // on the same detector. Without one there is nothing to report, and zero is what
          // "not measured" has to look like - see g4/G4Step.hh.
          step_.SetStepLength(0);
          step_.SetNumberOfAggregatedSteps(0);
          auto* pv = (sc < static_cast<int>(slot_pv_.size())) ? slot_pv_[sc] : nullptr;
          step_.GetPreStepPoint()->SetTouchable(pv);
          step_.GetPostStepPoint()->SetTouchable(pv);
          step_.GetPreStepPoint()->SetMaterial(
              (sc < static_cast<int>(slot_mat_.size())) ? slot_mat_[sc] : nullptr);
          sa_->UserSteppingAction(&step_);
        }
      }
      if (ea_ != nullptr) { ea_->EndOfEventAction(&event); }
    }

   private:
    G4UserEventAction* ea_;
    G4UserSteppingAction* sa_;
    const std::vector<G4PVPlacement*>& slot_pv_;
    const std::vector<G4Material*>& slot_mat_;
    G4Step step_;
    G4Track track_;
  };

  /// Refuses a species this transport does not carry, by name, before the engine sees it.
  ///
  /// The engine has the same guard - it is the one that cannot be bypassed - but it only has
  /// the species *number*, because a device record carries no names. Saying "proton" rather
  /// than "species 9" is the difference between a message and a puzzle, and this is the last
  /// place the name exists.
  ///
  /// What this replaces: the seeding kernel chose a track buffer by species with the gamma
  /// buffer as its default, so `/gun/particle proton` transported 6 MeV gammas and printed a
  /// dose for them.
  void CheckSpecies(const g4gpu::Source<G4double>& src) const {
    if (src.particle == g4gpu::ParticleType::kGamma
        || src.particle == g4gpu::ParticleType::kElectron
        || src.particle == g4gpu::ParticleType::kPositron
        || src.particle == g4gpu::ParticleType::kProton
        || src.particle == g4gpu::ParticleType::kAlpha) {
      return;
    }
    const G4String name = (gun_ != nullptr) ? gun_->GetParticleName() : G4String("?");
    std::printf(
        "\nFATAL: \"%s\" is not transported.\n"
        "  This transport carries gamma, e-, e+, proton and alpha. Other hadrons and ions\n"
        "  are not stepped, and the reason differs by species:\n"
        "\n"
        "    He3 and generic ions - G4ionIonisation scales a non-alpha ion from a base\n"
        "      particle's table by an effective charge squared and a mass ratio, and that\n"
        "      scaling is not transcribed. Its dE/dx here is wrong by a factor of twelve\n"
        "      (tests/test_hadron_range.cu records the number rather than hiding it).\n"
        "    muons, pions, kaons - dE/dx, delta rays and the radiative processes are all\n"
        "      transcribed and checked (tests/test_muon.cu, test_hadron_radiative.cu), but\n"
        "      no range table is built for them and step_hadron needs one. This is the\n"
        "      smallest gap of the three.\n"
        "    neutrons - need hadronic interactions, which this port does not have at all.\n"
        "\n"
        "  Refusing rather than transporting \"%s\" as something it is not.\n",
        name.c_str(), name.c_str());
    std::exit(2);
  }

  /// The transport's species for a Geant4 particle definition.
  ///
  /// The definition already carries it - G4ParticleTable builds every definition with its
  /// g4gpu::ParticleType - so this is a lookup rather than a second name table that could
  /// disagree with the first. A null definition means a generator built a vertex without
  /// setting a particle; gamma is the gun's own default and the engine's guard would not
  /// catch it, so it is named here instead.
  static g4gpu::ParticleType SpeciesOf(const G4ParticleDefinition* def) {
    if (def == nullptr) {
      std::printf(
          "\nFATAL: a primary vertex has no particle definition.\n"
          "  Call fParticleGun->SetParticleDefinition(...) before GeneratePrimaryVertex, or\n"
          "  set one on the G4PrimaryParticle you built by hand.\n");
      std::exit(2);
    }
    return def->GetType();
  }

  g4gpu::Source<G4double> CurrentSource() const {
    g4gpu::Source<G4double> s = (gun_ != nullptr) ? gun_->GetSource() : gun_source_;
    s.seed = seed_;
    return s;
  }

  /// Copies a loaded spectrum to the device and repoints the source at it.
  void UploadSpectrum(g4gpu::Source<G4double>& src) {
    if (gun_ == nullptr || gun_->SpectrumCdf().empty()) { return; }
    const auto& e = gun_->SpectrumEnergies();
    const auto& c = gun_->SpectrumCdf();
    if (d_spec_e_ == nullptr) {
      cudaMalloc(&d_spec_e_, sizeof(G4double) * e.size());
      cudaMalloc(&d_spec_c_, sizeof(G4double) * c.size());
    }
    cudaMemcpy(d_spec_e_, e.data(), sizeof(G4double) * e.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spec_c_, c.data(), sizeof(G4double) * c.size(), cudaMemcpyHostToDevice);
    src.spectrum.energy = d_spec_e_;
    src.spectrum.cdf = d_spec_c_;
    src.spectrum.n = static_cast<int>(e.size());
  }

  /// One placement and one material per scorer slot, for the aggregated step's touchable.
  void MapScorerSlots();

  std::unique_ptr<G4VUserDetectorConstruction> detector_;
  std::unique_ptr<G4VUserPhysicsList> physics_;
  std::unique_ptr<G4VUserActionInitialization> action_init_;
  std::unique_ptr<G4VUserPrimaryGeneratorAction> primary_;
  std::unique_ptr<G4UserRunAction> run_action_;
  std::unique_ptr<G4UserEventAction> event_action_;
  std::unique_ptr<G4UserSteppingAction> stepping_action_;
  std::unique_ptr<G4UserTrackingAction> tracking_action_;
  std::unique_ptr<G4UserStackingAction> stacking_action_;
  g4gpu::g4::FlatScene scene_;
  /// BEHIND A POINTER, so that sizeof(G4RunManager) does not depend on which step hook the
  /// including translation unit chose.
  ///
  /// Held by value, it did. The engine stores its hook by value, so a project defining
  /// G4STEP_HOOK saw a G4RunManager of one size and every other translation unit - the scene
  /// registry, the viewer, anything including this header without the define - saw another:
  /// measured, 48120 bytes against 48136. One class, two layouts, in one link.
  ///
  /// That is undefined, and the way it would have failed is worth stating because it is not
  /// obvious. Every member here is inline, so the linker keeps ONE copy of each and discards
  /// the rest; `new G4RunManager` sizes its allocation in the translation unit that writes it.
  /// If the surviving copies came from the larger layout while the allocation came from the
  /// smaller, every member after this one - last_stats_, run_, primaries_, slot_pv_ - would be
  /// addressed sixteen bytes too high, past the end. It worked because MSVC keeps the first
  /// COMDAT it sees and the order happened to agree. Nothing guaranteed it.
  ///
  /// A pointer is the same size whatever it points at, so the layout is now identical in every
  /// translation unit and a scene installer calling SetUserInitialization through either copy
  /// reaches the same offset. What is still true, and is the documented rule, is that a
  /// translation unit which CONSTRUCTS a run manager must define the same hook as the one that
  /// instantiated the engine - see the note at the top of transport_run_impl.cuh.
  g4gpu::host::TransportEngine<G4double, G4StepHook>* engine_ = nullptr;
  g4gpu::host::RunStats last_stats_;
  G4Run run_;
  G4ParticleGun* gun_ = nullptr;
  g4gpu::Source<G4double> gun_source_{};
  /// Per-cell voxel deposit for the whole run. The engine zeroes its own copy per call, so a
  /// run split between sources has to accumulate here or keep only the last sub-run.
  std::vector<double> run_voxel_scores_;
  /// The run's primaries, generated per event by the user's GeneratePrimaries. Reused across
  /// runs so a repeated BeamOn does not reallocate for each.
  std::vector<g4gpu::Primary<G4double>> primaries_;
  std::vector<G4PVPlacement*> slot_pv_;
  std::vector<G4Material*> slot_mat_;
  G4double range_cut_mm_ = 0.7;
  G4int batch_ = 0;  // 0 = size it from device memory
  G4int verbose_ = 0;
  G4int print_progress_ = 0;
  unsigned int seed_ = 0xF00Du;
  /// Where the next run starts in the seed's random stream, in primaries.
  ///
  /// Zero at construction, so a process that runs N events gets the same N events every time
  /// it starts - which is what makes a result quotable. Advanced by every run, so the second
  /// run of a process is a different sample of the same physics rather than the first one
  /// again.
  ///
  /// This is the device half of a behaviour whose host half was already there: CLHEP's engine
  /// is constructed once per process with a fixed seed and carries on across runs, so B1's two
  /// random draws per event already made its second run differ. What it did NOT change was the
  /// shower - every track's Philox stream was keyed on (seed, index-within-run), so run two
  /// fired different primaries into identical showers, and a generator that drew nothing
  /// repeated the run exactly. Both halves carry on now.
  long long stream_pos_ = 0;
  g4gpu::ProcessFlags processes_{};
  G4bool initialised_ = false;
  G4double* d_spec_e_ = nullptr;
  G4double* d_spec_c_ = nullptr;
};

// Defined here rather than in G4UserActions.hh because they forward to the run manager, which
// that header cannot see: it is included by this one.
inline void G4VUserActionInitialization::SetUserAction(G4VUserPrimaryGeneratorAction* a) const {
  G4RunManager::GetRunManager()->SetUserAction(a);
}
inline void G4VUserActionInitialization::SetUserAction(G4UserRunAction* a) const {
  G4RunManager::GetRunManager()->SetUserAction(a);
}
inline void G4VUserActionInitialization::SetUserAction(G4UserEventAction* a) const {
  G4RunManager::GetRunManager()->SetUserAction(a);
}
inline void G4VUserActionInitialization::SetUserAction(G4UserSteppingAction* a) const {
  G4RunManager::GetRunManager()->SetUserAction(a);
}
inline void G4VUserActionInitialization::SetUserAction(G4UserTrackingAction* a) const {
  G4RunManager::GetRunManager()->SetUserAction(a);
}
inline void G4VUserActionInitialization::SetUserAction(G4UserStackingAction* a) const {
  G4RunManager::GetRunManager()->SetUserAction(a);
}

inline void G4RunManager::Initialize() {
  if (detector_ == nullptr) {
    std::printf("\nFATAL: G4RunManager::Initialize with no detector construction.\n");
    std::exit(2);
  }
  detector_->Construct();
  detector_->ConstructSDandField();
  if (action_init_ != nullptr) {
    // Geant4 calls Build() once per worker and BuildForMaster() once on the master. There is
    // one of each here, so Build() alone - which is the one that registers every action.
    action_init_->Build();
  }
  scene_ = g4gpu::g4::flatten(range_cut_mm_);
  mass_cache_.clear();  // the geometry changed, so every scored mass has to be recomputed
  MapScorerSlots();
  std::printf("geometry: %d volumes, %d materials, %d scorer(s)\n",
              static_cast<int>(scene_.volumes.size()), scene_.materials.count,
              static_cast<int>(G4SDManager::GetSDMpointer()->Scorers().size()));
  // The production cut, as a range and as the energies it converted to. Geant4 prints its own
  // cuts table at initialisation for the same reason: the range cut is what the user sets and
  // the energies are what the physics actually uses, and a run whose cut did not take effect
  // looks exactly like one whose cut did until these two numbers are put side by side.
  if (scene_.materials.count > 0) {
    std::printf("cuts: %g mm range ->", range_cut_mm_);
    for (int i = 0; i < scene_.materials.count && i < 4; ++i) {
      std::printf(" [%d] e- %.4g MeV, gamma %.4g MeV", i, scene_.materials.m[i].cut_electron,
                  scene_.materials.m[i].cut_gamma);
    }
    std::printf("%s\n", scene_.materials.count > 4 ? " ..." : "");
  }
  engine_->SetProcesses(processes_);
  engine_->Upload(scene_, batch_);
  batch_ = engine_->ChosenBatch();
  initialised_ = true;
}

inline void G4RunManager::MapScorerSlots() {
  const auto& scorers = G4SDManager::GetSDMpointer()->Scorers();
  slot_pv_.assign(scorers.size(), nullptr);
  slot_mat_.assign(scorers.size(), nullptr);
  for (G4PVPlacement* pv : G4PVPlacement::Registry()) {
    G4LogicalVolume* lv = pv->GetLogicalVolume();
    const G4int slot = G4SDManager::GetSDMpointer()->SlotFor(lv);
    if (slot < 0 || slot >= static_cast<G4int>(slot_pv_.size())) { continue; }
    // First placement wins. A scorer summed over several placements has no single touchable;
    // an aggregated step names one of them, and a stepping action that only tests
    // "am I in the scoring volume" - the usual case - is unaffected.
    if (slot_pv_[slot] == nullptr) {
      slot_pv_[slot] = pv;
      slot_mat_[slot] = lv->GetMaterial();
    }
  }
}

/// Generates the run's primaries, one event at a time, and transports them.
///
/// **Everything that runs events goes through here**: BeamOn below, the viewer's Run button,
/// and the model builder's. That is not tidiness. Each of the three used to have its own copy
/// of this, and they disagreed: the viewer's never called GeneratePrimaries at all, so a
/// generated project's viewer fired an unconfigured gun - a point source spraying in every
/// direction - for a model with two beams. Consolidating them also turned up two things each
/// copy would have had to get right separately and only one did: the engine zeroes its
/// per-cell voxel array per call, so a split run has to accumulate it; and a scorer with a
/// per-event filter needs the event sink, which two of the three never created.
///
/// Primary generation is per event and on the host, as Geant4 does it. The loop below calls
/// the user's GeneratePrimaries once for each event with a fresh G4Event, and collects the
/// vertices it built. What that costs is a virtual call and a few random draws per event -
/// on the order of a tenth of a run's time - and what it buys is that a primary can be
/// anything the user's code can compute rather than one of the shapes the gun happens to
/// offer. See G4VUserPrimaryGeneratorAction::GeneratePrimaries.
///
/// @param traj   filled with trajectory segments when non-empty; the viewer wants them
/// @param shares if given, receives the per-source event counts, for a caller that reports them
inline g4gpu::host::RunStats G4RunManager::RunEvents(G4int n_events, std::vector<double>& sums,
                                                     std::vector<double>& sums_sq,
                                                     g4gpu::vis::TrajectoryBuffer traj,
                                                     std::vector<int>* shares) {
  g4gpu::host::RunStats stats{};
  if (shares != nullptr) { shares->clear(); }
  run_voxel_scores_.clear();
  if (primary_ == nullptr || n_events <= 0) { return stats; }
  const auto loop_t0 = std::chrono::steady_clock::now();

  // Only pay for the per-event chain if something wants it: a user action, or a scorer with a
  // filter. The filtered totals are reset here rather than in the sink, because a scorer
  // object outlives a run and its accumulator has to start each run empty.
  bool any_filtered = false;
  for (G4VPrimitiveScorer* ps : G4SDManager::GetSDMpointer()->Scorers()) {
    ps->filtered_total = 0;
    ps->filtered_total_sq = 0;
    if (ps->IsFiltered()) { any_filtered = true; }
  }
  std::unique_ptr<ActionSink> sink;
  if (event_action_ != nullptr || stepping_action_ != nullptr || any_filtered) {
    sink.reset(new ActionSink(
        event_action_.get(), stepping_action_.get(), slot_pv_, slot_mat_,
        (gun_ != nullptr) ? const_cast<G4ParticleDefinition*>(gun_->GetParticleDefinition())
                          : nullptr,
        (gun_ != nullptr) ? gun_->GetParticleEnergy() : 0.0));
  }

  // ---- the primaries
  //
  // One G4Event reused across the run: two million allocations of a vector holding one vertex
  // is a measurable slice of a run and none of it is physics. Reset() clears the vertices.
  primaries_.clear();
  primaries_.reserve(static_cast<std::size_t>(n_events));
  G4Event event(0);
  for (G4int i = 0; i < n_events; ++i) {
    event.Reset(i);
    primary_->GeneratePrimaries(&event);
    for (G4int v = 0; v < event.GetNumberOfPrimaryVertex(); ++v) {
      const G4PrimaryVertex& vtx = event.GetPrimaryVertex(v);
      for (G4int p = 0; p < vtx.GetNumberOfParticle(); ++p) {
        const G4PrimaryParticle& pp = vtx.GetPrimary(p);
        g4gpu::Primary<G4double> rec;
        rec.pos = {vtx.GetPosition().x(), vtx.GetPosition().y(), vtx.GetPosition().z()};
        rec.dir = {pp.GetMomentumDirection().x(), pp.GetMomentumDirection().y(),
                   pp.GetMomentumDirection().z()};
        rec.ekin = pp.GetKineticEnergy();
        rec.particle = SpeciesOf(pp.GetParticleDefinition());
        primaries_.push_back(rec);
      }
    }
  }
  if (primaries_.empty()) {
    std::printf(
        "\nFATAL: the primary generator produced no primaries.\n"
        "  GeneratePrimaries is called once per event and must add at least one vertex -\n"
        "  usually fParticleGun->GeneratePrimaryVertex(anEvent). A generator that only\n"
        "  configures the gun and returns leaves nothing to transport.\n");
    std::exit(2);
  }
  if (shares != nullptr) { shares->assign(1, n_events); }

  // The spectrum, if the gun loaded one, still lives on the device for the sampling the gun
  // does. Uploaded once per run rather than per event.
  {
    g4gpu::Source<G4double> s = CurrentSource();
    UploadSpectrum(s);
  }

  // stream_pos_, and then advanced by what this run consumed. This is the whole of what makes
  // a second BeamOn in one process an independent sample rather than a repeat: the device
  // streams carry on from where the last run left them, exactly as CLHEP's engine does for the
  // primaries on the host. See the note on stream_pos_.
  stats = engine_->BeamOn(static_cast<int>(primaries_.size()), primaries_.data(), seed_, sums,
                         sums_sq, traj, sink.get(), stream_pos_);
  stream_pos_ += static_cast<long long>(primaries_.size());
  const std::vector<double>& v = engine_->voxel_scores();
  if (!v.empty()) { run_voxel_scores_ = v; }

  // A filtered scorer's total is what Accept() accumulated, not what the device summed. Done
  // here rather than in BeamOn so that every caller sees the same number for the same scorer -
  // the builder reads these sums directly and would otherwise show the unfiltered total while
  // the generated project showed the filtered one.
  const auto& scorers = G4SDManager::GetSDMpointer()->Scorers();
  for (std::size_t i = 0; i < scorers.size() && i < sums.size(); ++i) {
    if (!scorers[i]->IsFiltered()) { continue; }
    sums[i] = scorers[i]->filtered_total;
    sums_sq[i] = scorers[i]->filtered_total_sq;
  }
  stats.event_loop_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loop_t0)
          .count();
  return stats;
}



inline void G4RunManager::BeamOn(G4int n_events, const char* macro, G4int n_select) {
  (void)macro;
  (void)n_select;
  if (!initialised_) {
    std::printf("\nFATAL: BeamOn before Initialize.\n");
    std::exit(2);
  }
  if (primary_ == nullptr) {
    std::printf("\nFATAL: BeamOn with no primary generator action.\n");
    std::exit(2);
  }
  // The run record the user actions see. `n_events` on it is what G4Run::GetNumberOfEvent
  // returns, and a RunAction that divides by it - every dose calculation does - reads zero if
  // this is not set. B1's EndOfRunAction returns silently on a zero count, so losing this line
  // made the example exit 0 having printed nothing at all.
  run_.n_events = n_events;
  if (run_action_ != nullptr) { run_action_->BeginOfRunAction(&run_); }

  // The run. RunEvents generates the primaries per event, creates the per-event sink, and
  // accumulates the per-cell voxel array; see it for why all of that lives there.
  const auto stats = RunEvents(n_events, run_.score_sum, run_.score_sum_sq);
  last_stats_ = stats;

  // Hand the totals back to the scorer objects, so a RunAction reads them where a Geant4
  // user expects to. RunEvents has already substituted a filtered scorer's own accumulation
  // into the sums, so this is a copy rather than a choice.
  auto& scorers = G4SDManager::GetSDMpointer()->Scorers();
  for (std::size_t i = 0; i < scorers.size() && i < run_.score_sum.size(); ++i) {
    scorers[i]->total = run_.score_sum[i];
    scorers[i]->total_sq = run_.score_sum_sq[i];
    // A 3D scorer also gets its hits map - the per-cell array the kernels filled, accumulated
    // over every sub-run. The whole pool is handed over rather than the slice belonging to
    // this scorer's volume: the cell index is global across voxel volumes, so a caller keys
    // by the index the geometry gives it and does not have to know about offsets. Cells
    // outside this scorer's volumes are zero, having never been written.
    //
    // Unfiltered by Accept(): the map is per cell and the hook is per event, so there is no
    // correspondence to apply. A filtered 3D scorer's total and its cells therefore describe
    // different things, and the header says so.
    if (auto* d3 = dynamic_cast<G4PSEnergyDeposit3D*>(scorers[i])) {
      d3->cells = run_voxel_scores_;
    }
  }

  if (run_action_ != nullptr) { run_action_->EndOfRunAction(&run_); }
}

inline G4double G4RunManager::ScoredMass(G4int scorer_index) const {
  // Answered from the cache when it can be. Not an optimisation: the GUI asks for this once
  // per scorer per frame to label its readout, the overlapped branch below is a 200,000-sample
  // Monte Carlo, and it prints a note when it runs. Uncached, that note repeated at the frame
  // rate and read like an error.
  const std::size_t si = static_cast<std::size_t>(scorer_index);
  if (scorer_index >= 0 && si < mass_cache_.size() && mass_cache_[si] >= 0) {
    return mass_cache_[si];
  }
  // Volume comes from the solid; density from the material record the flattener already built.
  //
  // Two cases, deliberately not merged. If nothing overlaps the scored volume, its mass is
  // its solid's volume times its density, exactly, with no sampling error - and B1's dose is
  // measured against Geant4 to 0.09 sigma on that exact number, so replacing it with a Monte
  // Carlo estimate would inject noise into a validated result for no gain. If something does
  // overlap it, the analytic product is wrong: the overlapping region belongs to whichever
  // volume has the higher layer, so the mass has to be integrated over the region this volume
  // wins, and that integral has no closed form.
  G4double mass_kg = 0;
  const auto store = scene_.pool.store();

  for (std::size_t i = 0; i < scene_.volumes.size(); ++i) {
    const auto& v = scene_.volumes[i];
    if (v.score_index != scorer_index) { continue; }
    const G4double rho = scene_.materials.m[v.material].density;  // g/cm3
    const G4double half = g4gpu::geom::solid_half_extent(store, v.solid);

    // Does any volume that outranks this one come near it? Bounding spheres, which is
    // conservative: a false positive costs a Monte Carlo integration, a false negative would
    // report a mass that is too large and a dose that is too small.
    bool overlapped = false;
    for (std::size_t j = 0; j < scene_.volumes.size() && !overlapped; ++j) {
      if (j == i) { continue; }
      const auto& w = scene_.volumes[j];
      const bool outranks =
          (w.layer > v.layer) || (w.layer == v.layer && j > i);
      if (!outranks) { continue; }
      const auto d = w.xform.trans - v.xform.trans;
      const G4double sep = std::sqrt(dot(d, d));
      overlapped = sep < half + g4gpu::geom::solid_half_extent(store, w.solid);
    }

    if (!overlapped) {
      const G4double vol_mm3 = g4gpu::geom::solid_volume(store, v.solid);
      mass_kg += vol_mm3 * 1e-3 * rho * 1e-3;  // mm3 -> cm3 -> g -> kg
      continue;
    }

    // Integrate over the region this volume owns. The geometry is the flattened one, so
    // locate() answers exactly the question the transport asks per step.
    g4gpu::geom::Geometry<G4double> g{};
    g.volumes = scene_.volumes.data();
    g.n_volumes = static_cast<int>(scene_.volumes.size());
    g.world = scene_.world;
    g.store = store;
    g.voxels = scene_.pool.voxel_store();

    constexpr int kSamples = 200000;
    unsigned int state = 0x2545F491u + static_cast<unsigned int>(i) * 2654435761u;
    auto u = [&state]() {
      state = state * 1664525u + 1013904223u;
      return static_cast<G4double>(state >> 8) / static_cast<G4double>(1u << 24);
    };
    int owned = 0;
    const G4double* r = v.xform.rot;
    for (int k = 0; k < kSamples; ++k) {
      const G4double lx = (2 * u() - 1) * half;
      const G4double ly = (2 * u() - 1) * half;
      const G4double lz = (2 * u() - 1) * half;
      // Local to world: the stored matrix is world -> local, so its transpose is the inverse.
      const g4gpu::Vec3<G4double> p{
          r[0] * lx + r[3] * ly + r[6] * lz + v.xform.trans.x,
          r[1] * lx + r[4] * ly + r[7] * lz + v.xform.trans.y,
          r[2] * lx + r[5] * ly + r[8] * lz + v.xform.trans.z};
      if (g4gpu::geom::locate(g, p) == static_cast<int>(i)) { ++owned; }
    }
    const G4double box_mm3 = 8.0 * half * half * half;
    const G4double vol_mm3 = box_mm3 * owned / kSamples;
    mass_kg += vol_mm3 * 1e-3 * rho * 1e-3;
    std::printf(
        "volume %d is overlapped by a higher layer, so its scored mass is the region it owns,\n"
        "  not its whole shape: %.6g mm3 of %.6g mm3 (%d of %d samples), +/- %.2f%%.\n"
        "  This is not a problem - a higher layer winning where two volumes overlap is what\n"
        "  the layer model is for - and the dose below already uses this mass.\n",
        static_cast<int>(i), vol_mm3, box_mm3, owned, kSamples,
        (owned > 0) ? 100.0 * std::sqrt(static_cast<G4double>(owned)) / owned : 0.0);
  }
  if (scorer_index >= 0) {
    if (si >= mass_cache_.size()) { mass_cache_.resize(si + 1, -1.0); }
    mass_cache_[si] = mass_kg;
  }
  return mass_kg;
}
