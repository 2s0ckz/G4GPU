// The user action base classes: detector construction, primary generation, run and event
// actions.
//
// Where Geant4 calls a virtual once per event, this calls it once per run and reads off what
// it configured. The reason is throughput - see src/physics/source.cuh - and the classes say
// so at the point where the difference bites, rather than leaving it to be discovered.
#pragma once
#include <vector>
#include "g4/G4LogicalVolume.hh"
#include "Randomize.hh"
#include "g4/G4PVPlacement.hh"
#include "g4/G4ParticleGun.hh"
#include "g4/G4SDManager.hh"
#include "g4/G4Step.hh"

// Geant4 distinguishes the abstract G4VPhysicalVolume from its concrete G4PVPlacement. Here
// there is only placement, so the abstract name is an alias - which keeps Construct()'s
// return type reading the way it does in every Geant4 example.

using G4VPhysicalVolume = G4PVPlacement;

class G4VUserDetectorConstruction {
 public:
  virtual ~G4VUserDetectorConstruction() = default;

  /// Build the geometry and return the world placement.
  virtual G4VPhysicalVolume* Construct() = 0;

  /// Attach sensitive detectors. Called once, after Construct().
  virtual void ConstructSDandField() {}

 protected:
  /// Geant4's helper, with Geant4's signature.
  void SetSensitiveDetector(G4LogicalVolume* lv, G4VSensitiveDetector* sd) {
    G4SDManager::GetSDMpointer()->AddNewDetector(sd);
    G4SDManager::GetSDMpointer()->Attach(lv, sd);
  }
  void SetSensitiveDetector(const G4String& logical_name, G4VSensitiveDetector* sd) {
    for (G4LogicalVolume* lv : G4LogicalVolume::Registry()) {
      if (lv->GetName() == logical_name) {
        SetSensitiveDetector(lv, sd);
        return;
      }
    }
    std::printf("\nFATAL: SetSensitiveDetector: no logical volume named \"%s\".\n",
                logical_name.c_str());
    std::exit(2);
  }
};

/// A stand-in for G4Event. It carries the event id so that a generator can be written the
/// familiar way; it has no primary-vertex list, because primaries are made on the device.
/// One primary particle of a vertex: what it is, where it is going, and how fast.
///
/// Geant4's carries momentum components; this carries a direction and a kinetic energy,
/// because that is what the transport needs and converting between them needs a mass the
/// caller may not have to hand. SetMomentumDirection/SetKineticEnergy are the accessors a
/// generator uses; the momentum form is offered too and normalises for you.
class G4PrimaryParticle {
 public:
  G4PrimaryParticle() = default;
  G4PrimaryParticle(const G4ParticleDefinition* def, const G4ThreeVector& direction,
                    G4double kinetic_energy)
      : definition_(def), direction_(direction), kinetic_energy_(kinetic_energy) {}

  const G4ParticleDefinition* GetParticleDefinition() const { return definition_; }
  void SetParticleDefinition(const G4ParticleDefinition* d) { definition_ = d; }

  const G4ThreeVector& GetMomentumDirection() const { return direction_; }
  void SetMomentumDirection(const G4ThreeVector& d) { direction_ = d.unit(); }

  G4double GetKineticEnergy() const { return kinetic_energy_; }
  void SetKineticEnergy(G4double e) { kinetic_energy_ = e; }

 private:
  const G4ParticleDefinition* definition_ = nullptr;
  G4ThreeVector direction_{0, 0, 1};
  G4double kinetic_energy_ = 0;
};

/// A primary vertex: a place, a time, and the particles that start there.
class G4PrimaryVertex {
 public:
  G4PrimaryVertex() = default;
  G4PrimaryVertex(const G4ThreeVector& position, G4double t0) : position_(position), t0_(t0) {}

  const G4ThreeVector& GetPosition() const { return position_; }
  void SetPosition(const G4ThreeVector& p) { position_ = p; }
  G4double GetT0() const { return t0_; }
  void SetT0(G4double t) { t0_ = t; }

  void SetPrimary(const G4PrimaryParticle& p) { particles_.push_back(p); }
  G4int GetNumberOfParticle() const { return static_cast<G4int>(particles_.size()); }
  const G4PrimaryParticle& GetPrimary(G4int i) const {
    return particles_[static_cast<std::size_t>(i)];
  }

 private:
  G4ThreeVector position_;
  G4double t0_ = 0;
  std::vector<G4PrimaryParticle> particles_;
};

/// One event.
///
/// It carries its primary vertices, because primary generation happens per event here exactly
/// as it does in Geant4: G4VUserPrimaryGeneratorAction::GeneratePrimaries is called once for
/// each event and builds the vertices for it. The run manager collects them and hands them to
/// the device.
///
/// Reused across the events of a run rather than constructed per event - Reset() is what a
/// run manager calls between them - because two million allocations of a vector of one vertex
/// is a measurable fraction of the time a run takes, and none of it is physics.
class G4Event {
 public:
  explicit G4Event(G4int id) : id_(id) {}
  G4int GetEventID() const { return id_; }

  void AddPrimaryVertex(G4PrimaryVertex* v) {
    if (v == nullptr) { return; }
    vertices_.push_back(*v);
    delete v;  // Geant4 takes ownership of the vertex; generators new one per event.
  }
  void AddPrimaryVertex(const G4PrimaryVertex& v) { vertices_.push_back(v); }

  G4int GetNumberOfPrimaryVertex() const { return static_cast<G4int>(vertices_.size()); }
  const G4PrimaryVertex& GetPrimaryVertex(G4int i) const {
    return vertices_[static_cast<std::size_t>(i)];
  }

  /// Clears the vertices and sets a new id, so one object can serve a whole run.
  void Reset(G4int id) {
    id_ = id;
    vertices_.clear();
  }

 private:
  G4int id_;
  std::vector<G4PrimaryVertex> vertices_;
};

class G4VUserPrimaryGeneratorAction {
 public:
  virtual ~G4VUserPrimaryGeneratorAction() = default;

  /// Build this event's primary vertices. Called ONCE PER EVENT, as Geant4 calls it.
  ///
  /// Write whatever you like here. Draw random numbers, read a phase-space file, sample a
  /// Gaussian spot, fire different particles on alternate events - the method runs on the host
  /// once for every event of the run, and the vertices it adds to @p event are what gets
  /// transported. `fParticleGun->GeneratePrimaryVertex(event)` is the usual way to add one.
  ///
  /// This used to be called once per *run*, with the gun's configuration sampled on the device
  /// instead. That was faster and it was not Geant4: the set of things a primary could be was
  /// closed - the beam shapes the gun happened to offer - and anything else, a Gaussian spot
  /// included, could not be written at all. Even this project's own B1 had to be rewritten
  /// away from Geant4's idiom to fit it.
  ///
  /// What it costs: a host call and the primaries' upload, on the order of a tenth of the run.
  /// What it buys is that the method means what it means in Geant4.
  virtual void GeneratePrimaries(G4Event* event) = 0;
};

/// A stand-in for G4Run, carrying what a RunAction normally reads off it.
class G4Run {
 public:
  G4int GetNumberOfEvent() const { return n_events; }

  G4int n_events = 0;
  /// Energy deposited in each scorer, MeV, summed over the run.
  std::vector<G4double> score_sum;
  /// Sum of squares over events, for the uncertainty.
  std::vector<G4double> score_sum_sq;
};

class G4UserRunAction {
 public:
  virtual ~G4UserRunAction() = default;
  virtual void BeginOfRunAction(const G4Run*) {}
  virtual void EndOfRunAction(const G4Run*) {}

  /// True here always: there is one process, so it is both the master and the only worker.
  /// An example that prints a different banner for each gets the master one.
  G4bool IsMaster() const { return true; }
};

class G4UserEventAction {
 public:
  virtual ~G4UserEventAction() = default;
  virtual void BeginOfEventAction(const G4Event*) {}
  virtual void EndOfEventAction(const G4Event*) {}
};

/// The per-step hook. Read the note at the top of g4/G4Step.hh before writing one: the step
/// it is handed is one aggregate per event and per scored volume, not one step.
class G4UserSteppingAction {
 public:
  virtual ~G4UserSteppingAction() = default;
  virtual void UserSteppingAction(const G4Step*) {}
};

/// The per-track hooks. Declared so that an ActionInitialization moved from Geant4 compiles;
/// neither is called, because tracks are created and killed on the device where a host
/// virtual cannot reach them. Registering one prints a warning saying exactly that, rather
/// than accepting it and doing nothing.
class G4UserTrackingAction {
 public:
  virtual ~G4UserTrackingAction() = default;
  virtual void PreUserTrackingAction(const G4Track*) {}
  virtual void PostUserTrackingAction(const G4Track*) {}
};

class G4UserStackingAction {
 public:
  virtual ~G4UserStackingAction() = default;
};

/// Groups the user actions, as G4VUserActionInitialization does.
///
/// Geant4 calls Build() once per worker thread and BuildForMaster() once on the master, which
/// is why an example creates a fresh RunAction in each. There is one process here, so Build()
/// is called once and BuildForMaster() is not called at all - an example that puts its
/// RunAction in both, as B1 does, still gets exactly one.
class G4VUserActionInitialization {
 public:
  virtual ~G4VUserActionInitialization() = default;
  virtual void Build() const = 0;
  /// Not called. Defined so that an override of it compiles.
  virtual void BuildForMaster() const {}

 protected:
  // Defined in G4RunManager.hh, which is where the run manager these forward to is declared.
  void SetUserAction(G4VUserPrimaryGeneratorAction* a) const;
  void SetUserAction(G4UserRunAction* a) const;
  void SetUserAction(G4UserEventAction* a) const;
  void SetUserAction(G4UserSteppingAction* a) const;
  void SetUserAction(G4UserTrackingAction* a) const;
  void SetUserAction(G4UserStackingAction* a) const;
};
/// Samples the gun's configuration and adds the vertex. See the declaration.
///
/// The RNG is Geant4's - G4UniformRand, so a run is reproducible through /random/setSeeds and
/// a user's own G4RandGauss draws interleave with these in one stream, which is what "the same
/// stream" means to somebody debugging a generator.
///
/// `sample_primary` is the same __host__ __device__ function the seeding kernel used to call,
/// so the distribution for a given configuration is unchanged; only where it is evaluated has
/// moved.
inline void G4ParticleGun::GeneratePrimaryVertex(G4Event* event) {
  if (event == nullptr) { return; }

  // An adapter with the interface sample_primary expects. It draws from Geant4's engine, so
  // the numbers come from wherever /random/setSeeds put them.
  //
  // `uniform()` is marked __host__ __device__ so that instantiating the __host__ __device__
  // sample_primary with it does not warn (nvcc 20014, "calling a __host__ function from a
  // __host__ __device__ function"). The device branch is unreachable and says so: nothing
  // instantiates this on the device, because primaries are generated on the host - that is the
  // whole point of the architecture. Suppressing the warning at the call site instead would
  // silence it for every other call too, including one that really was on the device.
  struct HostRng {
    __host__ __device__ G4double uniform() {
#ifdef __CUDA_ARCH__
      // Unreachable: see above. Returning a constant rather than calling the host engine keeps
      // the device path compilable without pretending to be a generator.
      return G4double(0.5);
#else
      return G4UniformRand();
#endif
    }
  } rng;

  g4gpu::Vec3<G4double> pos, dir;
  G4double ekin = 0;
  g4gpu::sample_primary(src_, rng, pos, dir, ekin);

  G4PrimaryVertex vertex(G4ThreeVector(pos.x, pos.y, pos.z), 0.);
  G4PrimaryParticle particle(definition_, G4ThreeVector(dir.x, dir.y, dir.z), ekin);
  vertex.SetPrimary(particle);
  event->AddPrimaryVertex(vertex);
}
