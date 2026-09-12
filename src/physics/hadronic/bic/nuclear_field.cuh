// The nuclear fields the binary cascade propagates in.
//
// Transcribed from G4VNuclearField.{hh,cc}, G4ProtonField.{hh,cc}, G4NeutronField.{hh,cc},
// G4PionPlusField.cc, G4PionMinusField.cc and G4PionZeroField.cc
// (models/binary_cascade) in 11.1.1.
//
// `GetField(position)` is the potential energy a particle of that species has at that point;
// `GetBarrier()` is the energy it has to pay to cross the nuclear surface. G4RKPropagation
// subtracts the field on the way in and adds it on the way out, and `G4KM_NucleonEqRhs` /
// `G4KM_OpticalEqRhs` integrate its gradient. The two families are built differently:
//
// **The nucleon fields are a TABLE, not a formula.** Both constructors precompute the local
// Fermi momentum at r = 0, 0.3, 0.6, ... fm out to `2*GetOuterRadius()`, and `GetField`
// linearly interpolates that table and returns `-p_F^2/(2 m) (+ barrier for the proton)`. The
// grid spacing is 0.3 fm and the interpolation is linear, so the field is piecewise linear in
// p_F and NOT in the potential - which is the V5 lesson of docs/HADRONIC_PLAN.md section 8 in
// its hadronic form: Geant4 does not run the formula, it runs a table built from the formula,
// and the table's grid is part of the answer.
//
// **The table's tail is three entries and two of them are a zero with the wrong units.** After
// the loop the constructor pushes `fermiMom(2R)`, then `0`, then `0` - and the out-of-range
// branch of `GetField` is `if ((index+2) > size) return theFermiMomBuffer.back()`, which
// returns that zero AS A FIELD. So beyond r = 2R + 0.6 fm a proton's field is exactly 0 where
// just inside it is `+theBarrier` (p_F has already fallen to zero there), i.e. the potential
// steps DOWN by the Coulomb barrier - 5.1 MeV for lead - at the edge of the table. Reproduced
// as written; docs/RISK.md V70. The two `G4ThreeVector aPosition` locals in those last two
// blocks are constructed, never read, and are what makes the intent visible: they were meant
// to be evaluated.
//
// **The pion fields are a formula with a sign error in the nucleus mass.** All three compute
//
//     nucleusMass = Z*proton_mass_c2 + (A-Z)*neutron_mass_c2 + bindingEnergy;
//
// and a nucleus's mass is that sum MINUS the binding energy - which is how
// `G4Fancy3DNucleus::GetMass()` writes it, twenty lines of Geant4 away. The mass only enters
// through `reducedMass = m_pi M/(m_pi + M)`, so the error is diluted by `m_pi/M`: 2e-4 relative
// on carbon, 1.2e-5 on lead. Small, real, and present in `G4KM_OpticalEqRhs::SetFactor` too.
// docs/RISK.md V70.
//
// **The two families have different cut-off radii.** The pion fields are zero for
// `r >= radius` with `radius = GetOuterRadius() + 4 fm`, the base class's member; the nucleon
// fields run to `2*GetOuterRadius() + 0.6 fm` through their table. For iron those are 10.2 fm
// and 12.0 fm.
//
// **REFUSED, by name:** the nine other fields `G4RKPropagation::Init` registers -
// G4AntiProtonField, G4KaonPlus/Minus/ZeroField, G4SigmaPlus/Minus/ZeroField. Their optical
// coefficients are recorded below as named constants because they are the only content of
// those classes that is not shared with the pion fields, but no channel this package reaches
// produces an anti-proton, a kaon or a hyperon: QBBC registers BIC for p, n, pi+ and pi- only
// (G4HadronInelasticQBBC.cc:148-186), and `G4BinaryLightIonReaction` feeds it nucleons.
#ifndef G4GPU_BIC_NUCLEAR_FIELD_CUH
#define G4GPU_BIC_NUCLEAR_FIELD_CUH

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/g4pow.hh"
#include "physics/hadronic/bic/nucleus/nucleus_model.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"

namespace g4gpu::bic {

/// The optical coefficients the ten non-nucleon field constructors default to, in fm. The three
/// pion ones are live; the rest are recorded and refused (see the file header).
__host__ __device__ inline constexpr double field_coeff_pion() {
  return 0.042 * 1e-12;  // 0.042 fermi
}
__host__ __device__ inline constexpr double field_coeff_antiproton() { return 1.53 * 1e-12; }
__host__ __device__ inline constexpr double field_coeff_kaon() { return 0.35 * 1e-12; }
__host__ __device__ inline constexpr double field_coeff_sigma() { return 0.36 * 1e-12; }

/// The largest nucleon-field table this port accepts. Geant4's vector has no limit; the table
/// length is `ceil(2*GetOuterRadius()/0.3 fm) + 3`, and the outer radius is a property of the
/// SAMPLED configuration, so it is not bounded by A alone - a U238 whose outermost nucleon
/// landed at 13 fm needs 90 entries. 256 covers 2R up to 76 fm and a caller that needs more is
/// refused rather than truncated.
inline constexpr int kMaxFieldTable = 256;

/// `(1.44/1.14) MeV * Z / (1 + Z13(A))`, written three times in 11.1.1 with three different
/// spellings of the same expression: `G4Fancy3DNucleus::CoulombBarrier` hoists the ratio into a
/// `static const`, `G4ProtonField::GetBarrier` writes it inline, and both pion fields write it
/// inline again. All three are this number.
__host__ __device__ inline double coulomb_barrier_mev(int a, int z) {
  return (1.44 / 1.14) * u::MeV<double>() * static_cast<double>(z) /
         (1.0 + data::g4pow_z13<double>(a));
}

/// G4ProtonField and G4NeutronField, which differ in three things only: the proton adds its
/// Coulomb barrier, divides by the proton mass, and names its radius member `theRadius` where
/// the neutron names it `theR`. One struct, one flag.
struct NucleonField {
  bool is_proton = false;
  int the_a = 0;
  int the_z = 0;
  double barrier = 0.0;    ///< theBarrier; 0 for the neutron
  double the_radius = 0.0; ///< 2*GetOuterRadius()
  double* table = nullptr; ///< theFermiMomBuffer, caller-owned
  int n_table = 0;
  int capacity = 0;
  bool overflow = false;

  /// G4ProtonField::GetField / G4NeutronField::GetField.
  ///
  /// `index = (unsigned)(x/(0.3 fm))` - a truncating cast through `unsigned int`, so a negative
  /// argument would wrap; `x` is a magnitude and cannot be negative. The bound test is
  /// `(index+2) > size`, i.e. the interpolation needs entries `index` and `index+1` to exist,
  /// and the fallback returns the last TABLE entry (a Fermi momentum, always 0) rather than a
  /// field. See the file header.
  __host__ __device__ double field(const Vec3d& position) const {
    const double step = 0.3 * deex::fermi();
    const double x = g4gpu::mag(position);
    const unsigned int index = static_cast<unsigned int>(x / step);
    if ((index + 2) > static_cast<unsigned int>(n_table)) { return table[n_table - 1]; }
    const double y1 = table[index];
    const double y2 = table[index + 1];
    const double x1 = step * static_cast<double>(index);
    const double x2 = step * static_cast<double>(index + 1);
    const double fermi_mom = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
    const double m = is_proton ? u::proton_mass_c2<double>() : u::neutron_mass_c2<double>();
    const double y = -1.0 * (fermi_mom * fermi_mom) / (2.0 * m) + barrier;
    return y;
  }

  /// G4ProtonField::GetBarrier - `bindingEnergy/theA + coulombBarrier` with `bindingEnergy`
  /// a local initialised to 0 and never assigned, under a commented-out line that would have
  /// read it out of `G4NucleiPropertiesTable`. So the `/theA` term is exactly zero and the
  /// barrier is the Coulomb barrier alone. G4NeutronField::GetBarrier is the same shape with
  /// the whole body commented out and `return 0.`.
  __host__ __device__ double get_barrier() const {
    if (!is_proton) { return 0.0; }
    const double binding_energy = 0.0;
    return binding_energy / static_cast<double>(the_a) + coulomb_barrier_mev(the_a, the_z);
  }
};

/// Both nucleon-field constructors. `outer_radius` is `GetOuterRadius()` of the nucleus the
/// field belongs to, which is a property of the sampled configuration.
///
/// The loop bound is `for (G4double aR=0.; aR<theRadius; aR+=0.3*fermi)` - repeated ADDITION of
/// 0.3 fm, not `k*0.3 fm`. Over 60 iterations those differ in the last bits and the difference
/// can change how many entries the table gets, which changes where the out-of-range fallback
/// starts. Reproduced by adding.
__host__ __device__ inline NucleonField make_nucleon_field(bool is_proton,
                                                           const NuclearDensity& density,
                                                           const FermiMomentum& fermi, int a,
                                                           int z, double outer_radius,
                                                           double* table, int capacity) {
  NucleonField f;
  f.is_proton = is_proton;
  f.the_a = a;
  f.the_z = z;
  f.table = table;
  f.capacity = capacity;
  // The proton constructor reads theBarrier BEFORE theRadius; the order does not matter here
  // because neither depends on the other, but GetBarrier() needs theA and theZ, which are set
  // two lines above it in Geant4 as well.
  f.barrier = f.get_barrier();
  f.the_radius = 2.0 * outer_radius;

  int n = 0;
  for (double r = 0.0; r < f.the_radius; r += 0.3 * deex::fermi()) {
    if (n >= capacity - 3) {
      f.overflow = true;
      f.n_table = n;
      return f;
    }
    const Vec3d pos{0.0, 0.0, r};
    table[n++] = fermi.fermi_momentum(density.density(pos));
  }
  {
    const Vec3d pos{0.0, 0.0, f.the_radius};
    table[n++] = fermi.fermi_momentum(density.density(pos));
  }
  // Two zeros, each in a block whose `G4ThreeVector aPosition` is built and not used.
  table[n++] = 0.0;
  table[n++] = 0.0;
  f.n_table = n;
  return f;
}

/// The three pion fields. They differ in the mass they use and in the sign of the barrier;
/// everything else is one formula.
enum PionFieldKind : int { kPionPlusField = 0, kPionMinusField = 1, kPionZeroField = 2 };

struct PionField {
  int kind = kPionPlusField;
  int the_a = 0;
  int the_z = 0;
  double radius = 0.0;   ///< G4VNuclearField::radius = GetOuterRadius() + 4 fm
  double the_coeff = 0.0;
  double pion_mass = 0.0;

  /// G4PionPlusField::GetBarrier / G4PionMinusField::GetBarrier / G4PionZeroField::GetBarrier.
  /// The minus pion's is the NEGATIVE Coulomb barrier - it is attracted - and the neutral
  /// pion's is 0.
  __host__ __device__ double get_barrier() const {
    if (kind == kPionZeroField) { return 0.0; }
    const double cb = coulomb_barrier_mev(the_a, the_z);
    return (kind == kPionPlusField) ? cb : -cb;
  }

  /// G4PionPlusField::GetField and its two siblings.
  ///
  /// `nucleusMass` has the binding energy ADDED; see the file header and docs/RISK.md V70.
  /// The pi0 version does NOT add `GetBarrier()` to the result, where the two charged ones do -
  /// which is the same answer, because its barrier is zero, but it is a different line.
  __host__ __device__ double field(const Vec3d& position,
                                   const NuclearDensity& density) const {
    if (g4gpu::mag(position) >= radius) { return 0.0; }
    const double binding_energy = deex::binding_energy(the_a, the_z);
    const double nucleus_mass = static_cast<double>(the_z) * u::proton_mass_c2<double>() +
                                static_cast<double>(the_a - the_z) *
                                    u::neutron_mass_c2<double>() +
                                binding_energy;
    const double reduced_mass = pion_mass * nucleus_mass / (pion_mass + nucleus_mass);
    const double dens =
        static_cast<double>(the_a) * density.density(position);
    const double nucleon_mass =
        (u::proton_mass_c2<double>() + u::neutron_mass_c2<double>()) / 2.0;
    const double v = 2.0 * u::pi<double>() * u::hbarc<double>() * u::hbarc<double>() /
                     reduced_mass * (1.0 + pion_mass / nucleon_mass) * the_coeff * dens;
    return (kind == kPionZeroField) ? v : (v + get_barrier());
  }
};

/// The PDG masses of the three pions, which are the only particle-table lookups these fields
/// make. These are `core/particle.cuh`'s values, not the PDG's, for the reason P1's comment
/// there gives: `G4PionPlus` is constructed with the literal `0.1395701*GeV`, which is 1.3e-6
/// below the PDG's 139.57018.
///
/// Measured, because this file had the PDG value for one draft: the pi- field on Al27 at
/// 3.6 fm is the near-cancellation of a +3.886 MeV optical potential against a -4.105 MeV
/// Coulomb barrier, so a 1.3e-6 relative error in the pion mass came out as 8.8e-6 relative on
/// the -0.219 MeV field - four hundred times any tolerance, from a mass that was right to six
/// figures. `nuclear_field.cuh` cannot include `core/particle.cuh` (it is P1's and it pulls in
/// the species enum a hadronic model has no business naming), so the two constants are stated
/// here with the cross-reference rather than shared.
__host__ __device__ inline constexpr double pdg_mass_pion_charged() {
  return 139.5701 * u::MeV<double>();
}
__host__ __device__ inline constexpr double pdg_mass_pion_zero() {
  return 134.9766 * u::MeV<double>();
}

__host__ __device__ inline PionField make_pion_field(int kind, int a, int z,
                                                     double outer_radius,
                                                     double coeff = field_coeff_pion()) {
  PionField f;
  f.kind = kind;
  f.the_a = a;
  f.the_z = z;
  // G4VNuclearField's constructor: radius(aNucleus->GetOuterRadius() + 4*fermi).
  f.radius = outer_radius + 4.0 * deex::fermi();
  f.the_coeff = coeff;
  f.pion_mass = (kind == kPionZeroField) ? pdg_mass_pion_zero() : pdg_mass_pion_charged();
  return f;
}

}  // namespace g4gpu::bic

#endif
