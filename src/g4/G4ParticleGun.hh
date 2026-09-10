// G4ParticleGun and the particle table.
//
// The setters are Geant4's. The additions are the distribution setters - a beam cross-section,
// an angular spread, a shell radius, an imported spectrum - which exist because a
// PrimaryGeneratorAction cannot randomise per event on the CPU at GPU rates. See
// src/physics/source.cuh for why, and for what each distribution means.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "core/particle.cuh"
#include "g4/G4SystemOfUnits.hh"
#include "g4/G4ThreeVector.hh"
#include "g4/G4Types.hh"
#include "physics/source.cuh"

/// A particle definition. Thin: the transport only needs mass, charge and species.
class G4ParticleDefinition {
 public:
  G4ParticleDefinition(const G4String& name, g4gpu::ParticleType type)
      : name_(name), type_(type) {}
  const G4String& GetParticleName() const { return name_; }
  g4gpu::ParticleType GetType() const { return type_; }
  G4double GetPDGMass() const {
    return g4gpu::particle_def<G4double>(type_).mass * MeV;
  }
  G4double GetPDGCharge() const { return g4gpu::particle_def<G4double>(type_).charge; }

 private:
  G4String name_;
  g4gpu::ParticleType type_;
};

class G4ParticleTable {
 public:
  static G4ParticleTable* GetParticleTable() {
    static G4ParticleTable inst;
    return &inst;
  }

  G4ParticleDefinition* FindParticle(const G4String& name) {
    for (auto& p : defs_) {
      if (p->GetParticleName() == name) { return p.get(); }
    }
    std::printf("\nFATAL: unknown particle \"%s\".\n  Known: ", name.c_str());
    for (auto& p : defs_) { std::printf("%s ", p->GetParticleName().c_str()); }
    std::printf("\n");
    std::exit(2);
  }

 private:
  /// Every ParticleType, under Geant4's own name for it.
  ///
  /// Built by walking the enum rather than by a hand-written list of `add` calls, and
  /// g4gpu::particle_name is where the names live. There were three name tables before - this
  /// one, a `type_of` helper in each of six tests, and the prose in
  /// G4RunManager::CheckSpecies - and this one was the shortest: it stopped at GenericIon, so
  /// `/gun/particle neutron` reported "unknown particle" alongside a list that was the set of
  /// names somebody had typed rather than the set of species the port has.
  ///
  /// Being in this table means the name RESOLVES, not that the species is transported. A gun
  /// set to a species with no kernel is refused by G4RunManager::CheckSpecies, which can then
  /// say what is missing for that species - which is a better answer than "unknown particle"
  /// for a particle Geant4 knows perfectly well.
  G4ParticleTable() {
    using g4gpu::ParticleType;
    for (int t = 0; t < static_cast<int>(ParticleType::kNumTypes); ++t) {
      const ParticleType pt = static_cast<ParticleType>(t);
      defs_.emplace_back(new G4ParticleDefinition(g4gpu::particle_name(pt), pt));
    }
  }
  std::vector<std::unique_ptr<G4ParticleDefinition>> defs_;
};

/// Declared here so the gun can take one; defined with the rest of the event API in
/// G4UserActions.hh, which includes this header.
class G4Event;

class G4ParticleGun {
 public:
  explicit G4ParticleGun(G4int n_particle = 1) : n_particle_(n_particle) {}

  // ------------------------------------------------------------ Geant4's setters
  void SetParticleDefinition(G4ParticleDefinition* d) {
    src_.particle = d->GetType();
    particle_name_ = d->GetParticleName();
    definition_ = d;
  }
  void SetParticleEnergy(G4double e) {
    src_.energy_shape = g4gpu::EnergyShape::kMonoenergetic;
    src_.energy = e / MeV;
  }
  /// Where the primary starts. Exactly there.
  ///
  /// In Geant4 this *is* the position: G4ParticleGun has no spatial distribution, so setting it
  /// leaves nothing to sample. The same is true here, and it has to be, or code written the
  /// Geant4 way is silently wrong. This gun does carry optional distributions -
  /// SetBeamCrossSectionRectangular and friends, which have no Geant4 equivalent - so setting a
  /// position clears them: the primary starts at the point you named.
  ///
  /// What that guards against, found by running example B1 interactively. Its vis.mac applied
  /// `/gun/beamRectangular 8 8 cm` so that pressing Run fired what the example fires. Its
  /// generator then drew its own spot in +-8 cm and called this. With the cross-section left in
  /// place the position was randomised *twice*, over +-16 cm against a 20 cm envelope, and a
  /// third of the beam missed the detector entirely: 287 pGy where the batch run - no macro, no
  /// cross-section - gave 429. The batch and interactive answers for one example disagreed by
  /// a third, and nothing in either looked wrong.
  ///
  /// To get a beam spot, set the position first and the cross-section second. That is the order
  /// every caller here already used, and it is the order that reads correctly.
  void SetParticlePosition(const G4ThreeVector& p) {
    src_.pos = g4gpu::Vec3<G4double>{p.x() / mm, p.y() / mm, p.z() / mm};
    src_.shape = g4gpu::SourceShape::kPoint;
  }
  /// The direction the primary travels. Exactly that direction.
  ///
  /// Clears any angular distribution, for the reason SetParticlePosition clears the spatial
  /// one: in Geant4 this is the direction, and code that says so must mean it.
  /// A zero direction falls back to +z rather than being stored.
  ///
  /// G4ThreeVector::unit() returns (0,0,0) for a zero vector - it guards the division, so
  /// there is no NaN - and a track with a zero direction is its own kind of broken: it has a
  /// position and an energy and goes nowhere, so it is transported, deposits nothing, and
  /// makes a run quietly report less dose than it should.
  ///
  /// +z is Geant4's own default direction, so falling back to it is the least surprising
  /// thing available. The guard is here rather than in the builder's form because a direction
  /// also arrives from a loaded project file and from user code calling this directly, and
  /// only one of those three paths has a dialog to complain in.
  void SetParticleMomentumDirection(const G4ThreeVector& d) {
    const G4ThreeVector u = d.unit();
    src_.dir = (u.mag2() > 0) ? g4gpu::Vec3<G4double>{u.x(), u.y(), u.z()}
                              : g4gpu::Vec3<G4double>{0, 0, 1};
    src_.angular = g4gpu::AngularShape::kUnidirectional;
  }
  void SetNumberOfParticles(G4int n) { n_particle_ = n; }
  G4int GetNumberOfParticles() const { return n_particle_; }
  const G4String& GetParticleName() const { return particle_name_; }
  /// The getters a RunAction uses to print the run conditions.
  const G4ParticleDefinition* GetParticleDefinition() const { return definition_; }
  G4double GetParticleEnergy() const { return src_.energy * MeV; }
  G4ThreeVector GetParticlePosition() const {
    return G4ThreeVector(src_.pos.x * mm, src_.pos.y * mm, src_.pos.z * mm);
  }
  G4ThreeVector GetParticleMomentumDirection() const {
    return G4ThreeVector(src_.dir.x, src_.dir.y, src_.dir.z);
  }

  // ------------------------------------------------------------ distributions
  /// A rectangular beam of the given half-widths, normal to the momentum direction.
  void SetBeamCrossSectionRectangular(G4double half_x, G4double half_y) {
    src_.shape = g4gpu::SourceShape::kBeamRectangle;
    src_.hx = half_x / mm;
    src_.hy = half_y / mm;
  }
  /// An elliptical beam with the given semi-axes, normal to the momentum direction.
  void SetBeamCrossSectionElliptical(G4double semi_x, G4double semi_y) {
    src_.shape = g4gpu::SourceShape::kBeamEllipse;
    src_.hx = semi_x / mm;
    src_.hy = semi_y / mm;
  }
  /// A point source with an angular spread: uniform in solid angle within the half-angle.
  void SetAngularSpread(G4double half_angle) {
    src_.angular = g4gpu::AngularShape::kCone;
    src_.cone_half_angle = half_angle / rad;
  }
  void SetIsotropic() { src_.angular = g4gpu::AngularShape::kIsotropic; }
  /// Primaries uniformly over a sphere of this radius, emitted inward.
  void SetIsotropicShell(G4double radius) {
    src_.shape = g4gpu::SourceShape::kIsotropicShell;
    src_.radius = radius / mm;
    src_.angular = g4gpu::AngularShape::kInward;
  }
  /// Primaries uniformly inside a box, for a distributed source.
  void SetSourceVolumeBox(G4double hx, G4double hy, G4double hz) {
    src_.shape = g4gpu::SourceShape::kVolumeBox;
    src_.hx = hx / mm;
    src_.hy = hy / mm;
    src_.hz = hz / mm;
  }

  /// Reads a two-column CSV of energy (MeV) and relative intensity, and builds the cumulative
  /// distribution the device samples. Rows may be in any order; a header line is skipped.
  ///
  /// The intensities are treated as a probability per bin, not a spectral density: that is
  /// what a measured or Monte-Carlo-generated spectrum file normally holds, and interpreting
  /// it as a density would silently reweight every bin by its width.
  G4bool SetEnergySpectrumFromCSV(const G4String& path);

  /// The spectrum, if one was loaded, for the run manager to upload.
  const std::vector<G4double>& SpectrumEnergies() const { return spec_e_; }
  const std::vector<G4double>& SpectrumCdf() const { return spec_cdf_; }

  // ---------------------------------------------------------------- per-event generation

  /// Adds one primary vertex to @p event, sampling whatever the gun is currently configured
  /// for. The Geant4 idiom, and the only way a primary reaches the transport.
  ///
  /// Everything the gun has been told - position, direction, energy, a beam cross-section, an
  /// angular spread, a spectrum - is sampled here, on the host, once per call. Which means
  /// anything you set before calling wins, and anything the gun cannot express you can write
  /// yourself:
  ///
  ///     fParticleGun->SetParticlePosition(G4ThreeVector(G4RandGauss::shoot(0, 3*mm),
  ///                                                     G4RandGauss::shoot(0, 3*mm),
  ///                                                     -20*cm));
  ///     fParticleGun->GeneratePrimaryVertex(anEvent);
  ///
  /// The sampling itself is `g4gpu::sample_primary`, the same function the device used to run
  /// for this - it is __host__ __device__ - so the distributions are identical to what they
  /// were, down to the order the random numbers are drawn in.
  ///
  /// Defined out of line, below G4Event.
  void GeneratePrimaryVertex(G4Event* event);

  /// The description a source is sampled from. Kept for the sampling above and for callers
  /// that want to register several sources on one gun.
  const g4gpu::Source<G4double>& GetSource() const { return src_; }
  g4gpu::Source<G4double>& GetSource() { return src_; }


 private:
  g4gpu::Source<G4double> src_{};
  G4int n_particle_ = 1;
  G4String particle_name_ = "gamma";
  G4ParticleDefinition* definition_ = nullptr;
  std::vector<G4double> spec_e_, spec_cdf_;
};

inline G4bool G4ParticleGun::SetEnergySpectrumFromCSV(const G4String& path) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) {
    std::printf("cannot read energy spectrum %s\n", path.c_str());
    return false;
  }
  std::vector<G4double> e, w;
  char line[512];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    G4double a = 0, b = 0;
    if (std::sscanf(line, "%lf,%lf", &a, &b) != 2
        && std::sscanf(line, "%lf %lf", &a, &b) != 2) {
      continue;  // header or blank
    }
    if (b < 0) { continue; }
    e.push_back(a);
    w.push_back(b);
  }
  std::fclose(f);
  if (e.size() < 2) {
    std::printf("energy spectrum %s has %d usable rows; need at least 2\n", path.c_str(),
                static_cast<int>(e.size()));
    return false;
  }

  // Sort by energy, then integrate. A file out of order is common enough to be worth handling.
  for (std::size_t i = 1; i < e.size(); ++i) {
    for (std::size_t j = i; j > 0 && e[j - 1] > e[j]; --j) {
      std::swap(e[j - 1], e[j]);
      std::swap(w[j - 1], w[j]);
    }
  }
  G4double total = 0;
  for (G4double v : w) { total += v; }
  if (total <= 0) {
    std::printf("energy spectrum %s has zero total intensity\n", path.c_str());
    return false;
  }
  spec_e_ = e;
  spec_cdf_.assign(e.size(), 0.0);
  G4double acc = 0;
  for (std::size_t i = 0; i < w.size(); ++i) {
    acc += w[i];
    spec_cdf_[i] = acc / total;
  }
  spec_cdf_.front() = 0.0;
  spec_cdf_.back() = 1.0;
  src_.energy_shape = g4gpu::EnergyShape::kSpectrum;
  src_.spectrum.n = static_cast<int>(spec_e_.size());
  std::printf("energy spectrum: %d points, %.4g to %.4g MeV\n", src_.spectrum.n,
              spec_e_.front(), spec_e_.back());
  return true;
}
