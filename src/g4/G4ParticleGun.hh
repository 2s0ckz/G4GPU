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
#include "core/track_buffer.cuh"      // ion_za_of: the nuclide's encoding on a track
#include "data/nuclei_mass_ame12.hh"  // G4NucleiProperties::GetNuclearMass, an ion's mass
#include "g4/G4SystemOfUnits.hh"
#include "g4/G4ThreeVector.hh"
#include "g4/G4Types.hh"
#include "physics/source.cuh"

/// A particle definition. Thin: the transport only needs mass, charge and species.
///
/// A REAL NUCLEUS IS A DEFINITION OF ITS OWN (P30), as `G4IonTable::CreateIon` makes one per
/// (Z, A): the species is `kGenericIon` - the one the transport steps every nucleus heavier than
/// an alpha as - and the definition adds the nuclide, which is what the species cannot say.
/// Its mass is `G4IonTable::GetNucleusMass(Z, A)`, i.e. G4NucleiProperties::GetNuclearMass
/// (`data::get_nuclear_mass`, every level P24 transcribed), and its PDG charge Z - CreateIon's
/// `charge = G4double(Z)*eplus`. Built only by G4IonTable::GetIon below.
class G4ParticleDefinition {
 public:
  G4ParticleDefinition(const G4String& name, g4gpu::ParticleType type)
      : name_(name), type_(type) {}
  G4ParticleDefinition(const G4String& name, G4int z, G4int a)
      : name_(name), type_(g4gpu::ParticleType::kGenericIon), z_(z), a_(a),
        ion_za_(g4gpu::ion_za_of(z, a)) {}
  const G4String& GetParticleName() const { return name_; }
  g4gpu::ParticleType GetType() const { return type_; }
  G4double GetPDGMass() const {
    if (ion_za_ != 0) { return g4gpu::data::get_nuclear_mass(a_, z_) * MeV; }
    return g4gpu::particle_def<G4double>(type_).mass * MeV;
  }
  G4double GetPDGCharge() const {
    if (ion_za_ != 0) { return static_cast<G4double>(z_); }
    return g4gpu::particle_def<G4double>(type_).charge;
  }
  /// G4ParticleDefinition::GetAtomicNumber/GetAtomicMass: Z and A for a nucleus made by
  /// G4IonTable, zero for everything else (the port's light nuclei - d, t, He3, alpha - are
  /// species of their own and carry them in their ParticleType, not here).
  G4int GetAtomicNumber() const { return z_; }
  G4int GetAtomicMass() const { return a_; }
  /// The nuclide as `TrackState::ion_za` encodes it, for the run manager to put on the primary
  /// record (`g4gpu::Primary::ion_za`). Zero for every definition that is not a real nucleus.
  unsigned short GetIonZA() const { return ion_za_; }

 private:
  G4String name_;
  g4gpu::ParticleType type_;
  G4int z_ = 0;
  G4int a_ = 0;
  unsigned short ion_za_ = 0;
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

/// G4IonTable, for the one thing a gun asks of it: the definition of a nucleus (Z, A).
///
/// `GetIon(Z, A, E)` is 11.1.1's `G4IonTable::GetIon(Z, A, E, flb, J)` for what this port
/// transports - a ground state:
///
///   - `(A<1) || (Z<=0) || (E<0.0) || (A>999)` returns nullptr, Geant4's own test (G4IonTable.cc
///     :642), and the messenger then prints Geant4's own "Ion with Z=.. A=..is not defined";
///   - Z <= 2 and one of (1,1) (1,2) (1,3) (2,4) (2,3) is `GetLightIon` - the proton, deuteron,
///     triton, alpha or He3 definition, a species of its own here as there;
///   - everything else is a definition named `GetIonName(Z, A, 0)` - "C12", "Fe56" - created
///     once and kept, as CreateIon does.
///
/// AND IT REFUSES BY NAME what the port cannot carry, where Geant4 would make a particle
/// (`RefusalReason` says which, and GetIon returns nullptr for it):
///
///   - an EXCITATION ENERGY: E > 0 makes an excited ion, `C12[4438.900]`, and the transport
///     steps ground states only - a track has no excitation field and no isomer decay;
///   - a (Z, A) whose nuclear mass G4NucleiProperties sets to zero (Z > A): Geant4 warns
///     "Wrong values for A = .. and Z = .." and builds an ion of mass zero, which this transport
///     would step at beta 1 (`step_hadron`'s `kIonWithoutNuclide`);
///   - Z >= 128 or A >= 512, past `ion_za_of`'s encoding (`TrackState::ion_za`). AME2012 stops
///     at Z 110, A 273 and the theoretical table at Z 136, A 339; Geant4 goes to A 999 on the
///     formula, which no B1 or AstroRad beam is.
///
/// A (Z, A) past AME2012 with a positive mass is NOT refused: since P24 (docs/RISK.md V238)
/// `data::get_nuclear_mass` is all four of GetNuclearMass's levels and gives the mass Geant4
/// gives, and a secondary of that nuclide is already transported on it.
class G4IonTable {
 public:
  static G4IonTable* GetIonTable() {
    static G4IonTable inst;
    return &inst;
  }

  /// Why `GetIon(Z, A, E)` returns nullptr when Geant4 would have made a particle, or nullptr
  /// when it would not refuse. Geant4's own illegal-argument test is not a refusal of this
  /// port's and returns nullptr here too - the messenger reports that one in Geant4's words.
  static const char* RefusalReason(G4int z, G4int a, G4double e) {
    if ((a < 1) || (z <= 0) || (e < 0.0) || (a > 999)) { return nullptr; }
    if (e > 0.0) {
      return "an excited ion (E > 0): the transport steps ground states only - a track carries "
             "no excitation energy and no isomer level";
    }
    if (!g4gpu::data::nuclear_mass_known(a, z)) {
      return "G4NucleiProperties::GetNuclearMass(A, Z) is zero for it (Z > A), and a nucleus "
             "of mass zero would be stepped at beta 1";
    }
    if (g4gpu::ion_za_of(z, a) == 0) {
      return "Z >= 128 or A >= 512 is past TrackState::ion_za's encoding";
    }
    return nullptr;
  }

  G4ParticleDefinition* GetIon(G4int z, G4int a, G4double e = 0.0) {
    if ((a < 1) || (z <= 0) || (e < 0.0) || (a > 999)) { return nullptr; }
    if (RefusalReason(z, a, e) != nullptr) { return nullptr; }
    // GetLightIon: the five light nuclei are particles of their own, in Geant4 and here.
    const g4gpu::ParticleType light = g4gpu::particle_type_of_nucleus(z, a);
    if (light != g4gpu::ParticleType::kGenericIon) {
      return G4ParticleTable::GetParticleTable()->FindParticle(g4gpu::particle_name(light));
    }
    const unsigned short za = g4gpu::ion_za_of(z, a);
    for (auto& d : ions_) {
      if (d->GetIonZA() == za) { return d.get(); }
    }
    ions_.emplace_back(new G4ParticleDefinition(GetIonName(z, a), z, a));
    return ions_.back().get();
  }

  /// G4IonTable::GetIonName(Z, A, lvl = 0): the element's symbol and the mass number, or
  /// "E<Z>-<A>" past the 118 elements G4IonTable names (G4IonTable.cc:1229).
  static G4String GetIonName(G4int z, G4int a) {
    // G4IonTable::elementName, 11.1.1, in order.
    static const char* const kElementName[118] = {
        "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
        "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni",
        "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo",
        "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba",
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
        "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",
        "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
        "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
    G4String name;
    if (z > 0 && z <= 118) {
      name = kElementName[z - 1];
    } else if (z > 118) {
      name = "E" + std::to_string(z) + "-";
    } else {
      return "?";
    }
    return name + std::to_string(a);
  }

 private:
  std::vector<std::unique_ptr<G4ParticleDefinition>> ions_;
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
    // G4ParticleGun::SetParticleDefinition: `particle_charge = GetPDGCharge()`, so a gun moved
    // off an ion forgets the ion's charge state as Geant4's does.
    charge_ = d->GetPDGCharge();
  }
  /// G4ParticleGun::SetParticleCharge - the primary's DYNAMIC charge, e+ units. `/gun/ion Z A Q`
  /// sets it after the definition (G4ParticleGunMessenger::IonCommand).
  ///
  /// HELD, NOT TRANSPORTED, AND THAT IS EXACT FOR WHAT THIS PORT RUNS (P30, docs/RISK.md V274).
  /// Geant4 copies it onto the G4PrimaryParticle and G4PrimaryTransformer onto the track's
  /// G4DynamicParticle, and in 11.1.1 the processes QBBC gives a nucleus never read it back:
  /// G4ionIonisation's charge is `G4EmCorrections::EffectiveChargeSquareRatio`, recomputed from
  /// the definition, the material and the energy on every step (`em::hadron_charge_sq_ratio` is
  /// that), msc and the hadronic processes read the definition's PDG charge, and the one reader
  /// of the dynamic charge on this path is G4Transportation's field propagation - and this port
  /// has no magnetic field. A track here carries no dynamic-charge field, so there is nothing to
  /// seed it into; a Q other than Z changes nothing Geant4 would compute for a B1 geometry.
  void SetParticleCharge(G4double q) { charge_ = q; }
  G4double GetParticleCharge() const { return charge_; }

  /// G4ParticleGunMessenger::fShootIon - `/gun/particle ion` sets it and `/gun/ion` requires
  /// it. Geant4 keeps it on the messenger, one per gun; this port's one macro interpreter
  /// (G4UImanager) serves whichever gun is registered, so the flag lives with the gun it
  /// describes rather than with the interpreter.
  void SetShootIon(G4bool on) { shoot_ion_ = on; }
  G4bool GetShootIon() const { return shoot_ion_; }
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
  G4double charge_ = 0;
  G4bool shoot_ion_ = false;
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
