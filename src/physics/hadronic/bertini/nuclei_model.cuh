// G4NucleiModel: the shell nucleus Bertini's cascade propagates through.
//
// Transcribed from Geant4 11.1.1, cascade/cascade/src/G4NucleiModel.cc:
//   generateModel, fillBindingEnergies, fillZoneRadii, fillZoneVolumes, fillPotentials,
//   zoneIntegralWoodsSaxon, zoneIntegralGaussian, getFermiKinetic, getPotential, getZone,
//   getRadius, getVolume, generateNucleonMomentum, generateNucleon, generateQuasiDeuteron,
//   useQuasiDeuteron, passFermi, passTrailing, boundaryTransition, worthToPropagate,
//   forceFirst, isProjectile, getRatio, getCurrentDensity, inverseMeanFreePath,
//   generateInteractionLength, absorptionCrossSection, totalCrossSection
//
// Everything the model computes from (A, Z) is deterministic - `generateModel` draws no random
// number - so `ref/oracle/bertini_nuclei.csv` pins the whole zone structure exactly for
// thirteen nuclei chosen to reach every branch:
//
//   A < 5      one zone, a ball of radius `radiusForSmall` (times `radScaleAlpha` for A = 4)
//   5 <= A<12  three zones, GAUSSIAN density, radii at sqrt(-log(alfa3[i])) * gaussRadius
//   12<=A<100  three zones, WOODS-SAXON, radii at nuclearRadius + skinDepth*log((1+e^-R/d)/alfa3[i] - 1)
//   A >= 100   six zones, Woods-Saxon on alfa6
//
// ---------------------------------------------------------------------------------------------
// Five things about this class that a reading does not give you.
//
// **1. `G4cbrt` is two different functions and generateModel calls both.** `nuclearRadius` for
// A > 4 is `radiusScale*G4cbrt(A) + radiusScale2/G4cbrt(A)` with an `G4int A`, which selects the
// INTEGER overload (G4Pow::Z13, a tabulated cube root); `fillPotentials` computes
// `fermiMomentum * G4cbrt(rd)` with a `G4double rd`, which selects the exp/log one. They do not
// agree, and swapping them moves every Fermi momentum.
//
// **2. The two zone integrals are adaptive trapezoid loops with a RELATIVE convergence test**,
// `|(fun-fun1)/fun| <= 1e-3`, capped at 1000 halvings. So the result depends on how many
// iterations it took, not only on the integrand - which is exactly the kind of thing that comes
// out almost right from a transcription and is caught only by an exact oracle. Both are
// reproduced iteration for iteration, including the `jc *= 2` doubling of the inner sample count
// and the `dr1` lag (the inner loop steps by the PREVIOUS dr, not the current one).
//
// **3. `binding_energies` is a pair of ABSOLUTE differences and the source says it should not
// be.** `fillBindingEnergies` stores `fabs(bindingEnergy(A-1,Z-1) - bindingEnergy(A,Z))/GeV` and
// `fabs(bindingEnergy(A-1,Z) - ...)`, with Geant4's own FIXME beside it ("Why is fabs() used
// here instead of the signed difference?"). Those two numbers are added to the nucleon
// potentials in `fillPotentials`, so the sign matters and the absolute value is what runs.
//
// **4. `getPotential` folds 44 type codes into five slots, with two of the folds unreachable.**
// Its body is
//     if (ip == 9 || ip < 0) return 0.0;            // photons and ALL leptons
//     G4int ip0 = ip < 3 ? ip - 1 : 2;              // p -> 0, n -> 1, everything else -> 2
//     if (ip > 10 && ip < 18) ip0 = 3;              // the four kaons
//     if (ip > 20) ip0 = 4;                         // hyperons - and the light nuclei, and
//                                                   // the three dibaryons
// so a deuteron (41), an alpha (47) and a diproton (111) all read the HYPERON potential, 30 MeV,
// and the pion slot (2) is reached by pions, by nothing else below 11, and by codes 18..20 which
// do not exist. The `ip > 20` line is written after the kaon line and overrides nothing, because
// the ranges do not overlap. Reproduced as written; `bertini_potfold.csv` dumps all 44 x 7.
//
// **5. `inverseMeanFreePath` returns a rate, not a length, and `generateInteractionLength`
// returns `large` (1000) rather than the path when nothing interacts.** The 20130701 history
// entry says "Don't average 1/MFP for total interaction; just sum" and the 1000 is a sentinel
// the caller compares against the geometric path. A transcription that returned the path would
// make every particle interact at the zone boundary.
#ifndef G4GPU_BERTINI_NUCLEI_MODEL_CUH
#define G4GPU_BERTINI_NUCLEI_MODEL_CUH

#include <cmath>

#include "core/vec3.cuh"
#include "physics/hadronic/bertini/cascade_interpolator.cuh"
#include "physics/hadronic/bertini/cascade_params.cuh"
#include "physics/hadronic/bertini/channel_tables.cuh"
#include "physics/hadronic/bertini/inucl_particle.cuh"
#include "physics/hadronic/bertini/lorentz_convertor.cuh"

namespace g4gpu::physics::hadronic::bert {

/// G4NucleiModel's static cutoffs and flat potentials.
constexpr double kNmSmall = 1.0e-9;
constexpr double kNmLarge = 1000.0;
constexpr double kPionVp = 0.007;        ///< GeV, A > 4
constexpr double kPionVpSmall = 0.007;   ///< GeV, A <= 4. Identical in 11.1.1; the two names
                                         ///< exist because they were different before the
                                         ///< "REVERT TO OLD NON-PHYSICAL PARAMETERS FOR 9.5"
                                         ///< entry in the file's history.
constexpr double kKaonVp = 0.015;
constexpr double kHyperonVp = 0.030;
constexpr int kMaxZones = 6;

/// `pi*4/3`, which G4NucleiModel holds as a static and its own comment flags
/// ("FIXME: We should not be using this!"). Written to the same digits CLHEP's pi gives.
constexpr double kPiTimes4Thirds = 3.14159265358979323846 * 4.0 / 3.0;

/// The zone boundaries, as fractions of the density at the nuclear radius, from outside in.
__host__ __device__ inline const double* nm_alfa3() {
  static const double v[3] = {0.7, 0.3, 0.01};
  return v;
}
__host__ __device__ inline const double* nm_alfa6() {
  static const double v[6] = {0.9, 0.6, 0.4, 0.2, 0.1, 0.05};
  return v;
}

/// The quasi-deuteron absorption cross section: the deuteron photo-disintegration cross section
/// (gamma + d -> p + n), 2.4 MeV to 500 MeV from data, 0.5 to 3 GeV from an angle integration of
/// the JLab measurement (Mirazita et al., Phys. Rev. C 70, 014005 (2004)), extrapolated above.
/// Its own energy scale, which is NOT any of the three channel-table scales - the first six
/// edges are the deuteron threshold region.
__host__ __device__ inline const double* nm_gamma_qd_bins() {
  static const double v[30] = {
      0.0,  0.0024, 0.0032, 0.0042, 0.0056, 0.0075, 0.01,  0.024, 0.075, 0.1,
      0.13, 0.18,   0.24,   0.32,   0.42,   0.56,   0.75,  1.0,   1.3,   1.8,
      2.4,  3.2,    4.2,    5.6,    7.5,   10.0,   13.0,  18.0,  24.0,  32.0};
  return v;
}
__host__ __device__ inline const double* nm_gamma_qd_xsec() {
  static const double v[30] = {
      0.0,     0.7,    2.0,    2.2,    2.1,    1.8,    1.3,     0.4,     0.098,   0.071,
      0.055,   0.055,  0.065,  0.045,  0.017,  0.007,  2.37e-3, 6.14e-4, 1.72e-4, 4.2e-5,
      1.05e-5, 3.0e-6, 7.0e-7, 1.3e-7, 2.3e-8, 3.2e-9, 4.9e-10, 0.0,     0.0,     0.0};
  return v;
}

/// G4NucleiModel's derived parameters, the ones its constructor builds out of
/// G4CascadeParameters. Separate from CascadeParams because they are the products, and because
/// `skinDepth` is `0.611207*radiusUnits` - a number that appears nowhere else.
struct NucleiModelParams {
  double cross_section_units;   ///< xsecScale
  double radius_units;          ///< radiusScale
  double skin_depth;            ///< 0.611207 * radiusUnits
  double radius_scale;          ///< (useTwoParam ? 1.16 : 1.2) * radiusUnits
  double radius_scale2;         ///< (useTwoParam ? -1.3456 : 0.) * radiusUnits
  double radius_for_small;      ///< radiusSmall
  double rad_scale_alpha;       ///< radiusAlpha
  double fermi_momentum;        ///< fermiScale
  double r_nucleon;             ///< radiusTrailing - zero, so the trailing effect is OFF
  double gamma_qd_scale;        ///< gammaQDScale
  double potential_thickness;   ///< hard-coded 1.0 in the constructor
};

__host__ __device__ inline NucleiModelParams nuclei_model_params(const CascadeParams& p) {
  NucleiModelParams n;
  n.cross_section_units = p.xsec_scale;
  n.radius_units = p.radius_scale;
  n.skin_depth = 0.611207 * p.radius_scale;
  n.radius_scale = (p.use_two_param ? 1.16 : 1.2) * p.radius_scale;
  n.radius_scale2 = (p.use_two_param ? -1.3456 : 0.0) * p.radius_scale;
  n.radius_for_small = p.radius_small;
  n.rad_scale_alpha = p.radius_alpha;
  n.fermi_momentum = p.fermi_scale;
  n.r_nucleon = p.radius_trailing;
  n.gamma_qd_scale = p.gamma_qd_scale;
  n.potential_thickness = 1.0;
  return n;
}

/// The whole nucleus, as a fixed-size struct. A cascade carries hundreds of particles and this
/// is what every one of them reads, so it is passed by pointer and never copied into a local.
struct NucleiModel {
  int a = 0;
  int z = 0;
  int number_of_zones = 0;
  double nuclei_radius = 0.0;
  double nuclei_volume = 0.0;
  double binding_energies[2] = {0.0, 0.0};   ///< [0] proton loss, [1] neutron loss, GeV
  double zone_radii[kMaxZones] = {0, 0, 0, 0, 0, 0};
  double zone_volumes[kMaxZones] = {0, 0, 0, 0, 0, 0};
  /// `nucleon_densities[ip-1][izone]`, `fermi_momenta[ip-1][izone]`, and the five
  /// `zone_potentials` rows in the order fillPotentials/generateModel push them: proton,
  /// neutron, pion, kaon, hyperon.
  double nucleon_densities[2][kMaxZones] = {{0}, {0}};
  double fermi_momenta[2][kMaxZones] = {{0}, {0}};
  double zone_potentials[5][kMaxZones] = {{0}, {0}, {0}, {0}, {0}};
  /// The running state `reset()` keeps: how many nucleons of each kind are left.
  int neutron_number = 0;
  int proton_number = 0;
  int neutron_number_current = 0;
  int proton_number_current = 0;
};

/// G4NucleiModel::reset, without the trailing-effect hit list (which is the caller's, because
/// its capacity is a refusal this module has to name).
__host__ __device__ inline void nm_reset(NucleiModel& m, int n_hit_neutrons = 0,
                                         int n_hit_protons = 0) {
  m.neutron_number_current = m.neutron_number - n_hit_neutrons;
  m.proton_number_current = m.proton_number - n_hit_protons;
}

// =============================================================================================
// The two zone integrals
// =============================================================================================

/// G4NucleiModel::zoneIntegralWoodsSaxon.
///
/// An adaptive trapezoid rule: halve dr, add the new midpoints, compare the new estimate with
/// the old one relatively, stop at 1e-3 or after 1000 halvings. Two details that a rewrite gets
/// wrong: the inner loop steps `r` by `dr1`, which is the PREVIOUS dr and not the current one,
/// and the comparison is `|(fun-fun1)/fun|` with the NEW estimate in the denominator, so it is
/// scale-free but asymmetric.
///
/// The closed-form tail `skinRatio^2 * log((1+e^-r1)/(1+e^-r2))` is added after the loop and is
/// the analytic part of the Woods-Saxon integral; `skinDepth^3` scales the whole thing.
__host__ __device__ inline double nm_zone_integral_woods_saxon(double r1, double r2,
                                                               double nuclear_radius,
                                                               double skin_depth) {
  const double epsilon = 1.0e-3;
  const int itry_max = 1000;
  const double skin_ratio = nuclear_radius / skin_depth;
  const double d2 = 2.0 * skin_ratio;
  double dr = r2 - r1;
  const double fr1 = r1 * (r1 + d2) / (1.0 + std::exp(r1));
  const double fr2 = r2 * (r2 + d2) / (1.0 + std::exp(r2));
  double fi = (fr1 + fr2) / 2.0;
  double fun1 = fi * dr;
  double fun = fun1;
  int jc = 1;
  double dr1 = dr;
  int itry = 0;
  while (itry < itry_max) {
    dr /= 2.0;
    ++itry;
    double r = r1 - dr;
    fi = 0.0;
    for (int i = 0; i < jc; ++i) {
      r += dr1;
      fi += r * (r + d2) / (1.0 + std::exp(r));
    }
    fun = 0.5 * fun1 + fi * dr;
    if (std::fabs((fun - fun1) / fun) <= epsilon) { break; }
    jc *= 2;
    dr1 = dr;
    fun1 = fun;
  }
  const double sd3 = skin_depth * skin_depth * skin_depth;
  return sd3 * (fun + skin_ratio * skin_ratio *
                          std::log((1.0 + std::exp(-r1)) / (1.0 + std::exp(-r2))));
}

/// G4NucleiModel::zoneIntegralGaussian. The same loop with `r^2 exp(-r^2)`, and it recomputes
/// `gaussRadius` from (nucRad, A) rather than taking it as an argument - so the A it uses is the
/// model's own, not the one fillZoneRadii used, and they are the same only because
/// generateModel sets A before calling either.
__host__ __device__ inline double nm_zone_integral_gaussian(double r1, double r2, double nuc_rad,
                                                            int a) {
  const double gauss_radius =
      std::sqrt(nuc_rad * nuc_rad * (1.0 - 1.0 / double(a)) + 6.4);
  const double epsilon = 1.0e-3;
  const int itry_max = 1000;
  double dr = r2 - r1;
  const double fr1 = r1 * r1 * std::exp(-r1 * r1);
  const double fr2 = r2 * r2 * std::exp(-r2 * r2);
  double fi = (fr1 + fr2) / 2.0;
  double fun1 = fi * dr;
  double fun = fun1;
  int jc = 1;
  double dr1 = dr;
  int itry = 0;
  while (itry < itry_max) {
    dr /= 2.0;
    ++itry;
    double r = r1 - dr;
    fi = 0.0;
    for (int i = 0; i < jc; ++i) {
      r += dr1;
      fi += r * r * std::exp(-r * r);
    }
    fun = 0.5 * fun1 + fi * dr;
    if (std::fabs((fun - fun1) / fun) <= epsilon) { break; }
    jc *= 2;
    dr1 = dr;
    fun1 = fun;
  }
  return gauss_radius * gauss_radius * gauss_radius * fun;
}

// =============================================================================================
// generateModel
// =============================================================================================

/// What generate_model could not do. A refusal, not a silent clamp.
enum class NucleiModelRefusal {
  kNone = 0,
  kZoneCountAboveCapacity,   ///< more than six zones: impossible for 11.1.1's thresholds
  kLevelDensityAboveTable    ///< A > 245, where nucleiLevelDensity reads past its array
};

/// G4NucleiModel::generateModel(a, z).
///
/// Note what is NOT here: the `if (a == A && z == Z) { reset(); return; }` shortcut. That is a
/// cache on a stateful object; the caller here holds the struct and decides whether to rebuild.
__host__ __device__ inline NucleiModelRefusal nm_generate_model(NucleiModel& m, int a, int z,
                                                                const NucleiModelParams& p) {
  m.a = a;
  m.z = z;
  m.neutron_number = a - z;
  m.proton_number = z;
  nm_reset(m);

  const double nuclear_radius =
      (a > 4) ? (p.radius_scale * inucl_cbrt_int(a) + p.radius_scale2 / inucl_cbrt_int(a))
              : (p.radius_for_small * ((a == 4) ? p.rad_scale_alpha : 1.0));

  m.number_of_zones = (a < 5) ? 1 : (a < 100) ? 3 : 6;
  if (m.number_of_zones > kMaxZones) { return NucleiModelRefusal::kZoneCountAboveCapacity; }

  // fillBindingEnergies. The absolute values are Geant4's, FIXME and all.
  const double dm = inucl_binding_energy(a, z);
  m.binding_energies[0] = std::fabs(inucl_binding_energy(a - 1, z - 1) - dm) / 1000.0;
  m.binding_energies[1] = std::fabs(inucl_binding_energy(a - 1, z) - dm) / 1000.0;

  // fillZoneRadii, and the `ur[]` skin-depth coordinates the integrals use.
  double ur[kMaxZones + 1];
  const double skin_ratio = nuclear_radius / p.skin_depth;
  const double skin_decay = std::exp(-skin_ratio);
  if (a < 5) {
    m.zone_radii[0] = nuclear_radius;
    ur[0] = 0.0;
    ur[1] = 1.0;
  } else if (a < 12) {
    const double rsq = nuclear_radius * nuclear_radius;
    const double gauss_radius = std::sqrt(rsq * (1.0 - 1.0 / double(a)) + 6.4);
    ur[0] = 0.0;
    for (int i = 0; i < m.number_of_zones; ++i) {
      const double y = std::sqrt(-std::log(nm_alfa3()[i]));
      m.zone_radii[i] = gauss_radius * y;
      ur[i + 1] = y;
    }
  } else {
    const double* alfa = (a < 100) ? nm_alfa3() : nm_alfa6();
    ur[0] = -skin_ratio;
    for (int i = 0; i < m.number_of_zones; ++i) {
      const double y = std::log((1.0 + skin_decay) / alfa[i] - 1.0);
      m.zone_radii[i] = nuclear_radius + p.skin_depth * y;
      ur[i + 1] = y;
    }
  }

  // fillZoneVolumes. `v[]` are the density integrals and `v1[]` the pseudo-volumes (delta r^3);
  // `tot_vol` is the SUM OF THE INTEGRALS and omits the 4pi/3, which the zone_volumes carry.
  double v[kMaxZones];
  double v1[kMaxZones];
  double tot_vol = 0.0;
  if (a < 5) {
    v[0] = 1.0;
    v1[0] = 1.0;
    tot_vol = m.zone_radii[0] * m.zone_radii[0] * m.zone_radii[0];
    m.zone_volumes[0] = tot_vol * kPiTimes4Thirds;
  } else {
    const bool gaussian = (a < 12);
    for (int i = 0; i < m.number_of_zones; ++i) {
      v[i] = gaussian ? nm_zone_integral_gaussian(ur[i], ur[i + 1], nuclear_radius, a)
                      : nm_zone_integral_woods_saxon(ur[i], ur[i + 1], nuclear_radius,
                                                     p.skin_depth);
      tot_vol += v[i];
      v1[i] = m.zone_radii[i] * m.zone_radii[i] * m.zone_radii[i];
      if (i > 0) {
        v1[i] -= m.zone_radii[i - 1] * m.zone_radii[i - 1] * m.zone_radii[i - 1];
      }
      m.zone_volumes[i] = v1[i] * kPiTimes4Thirds;
    }
  }

  // fillPotentials, protons then neutrons - the order matters, because it is the order
  // zone_potentials' rows are pushed in and `getPotential` indexes them by it.
  for (int ip = 1; ip <= 2; ++ip) {
    const double mass = inucl_particle_mass(ip);
    const double dmb = m.binding_energies[ip - 1];
    const int n_nucleons = (ip == kProton) ? m.proton_number : m.neutron_number;
    const double dd0 = double(n_nucleons) / tot_vol / kPiTimes4Thirds;
    for (int i = 0; i < m.number_of_zones; ++i) {
      const double rd = dd0 * v[i] / v1[i];
      m.nucleon_densities[ip - 1][i] = rd;
      const double pff = p.fermi_momentum * inucl_cbrt(rd);
      m.fermi_momenta[ip - 1][i] = pff;
      m.zone_potentials[ip - 1][i] = 0.5 * pff * pff / mass + dmb;
    }
  }
  // The three flat potentials, in generateModel's own order: pion, kaon, hyperon.
  for (int i = 0; i < m.number_of_zones; ++i) {
    m.zone_potentials[2][i] = (a > 4) ? kPionVp : kPionVpSmall;
    m.zone_potentials[3][i] = kKaonVp;
    m.zone_potentials[4][i] = kHyperonVp;
  }

  m.nuclei_radius = m.zone_radii[m.number_of_zones - 1];
  m.nuclei_volume = 0.0;
  for (int i = 0; i < m.number_of_zones; ++i) { m.nuclei_volume += m.zone_volumes[i]; }
  return NucleiModelRefusal::kNone;
}

// =============================================================================================
// The accessors
// =============================================================================================

__host__ __device__ inline double nm_density(const NucleiModel& m, int ip, int izone) {
  return m.nucleon_densities[ip - 1][izone];
}
__host__ __device__ inline double nm_fermi_momentum(const NucleiModel& m, int ip, int izone) {
  return m.fermi_momenta[ip - 1][izone];
}

/// G4NucleiModel::getPotential - the five-slot fold described in note 4 of the header.
__host__ __device__ inline double nm_potential(const NucleiModel& m, int ip, int izone) {
  if (ip == 9 || ip < 0) { return 0.0; }
  int ip0 = (ip < 3) ? ip - 1 : 2;
  if (ip > 10 && ip < 18) { ip0 = 3; }
  if (ip > 20) { ip0 = 4; }
  return (izone < m.number_of_zones) ? m.zone_potentials[ip0][izone] : 0.0;
}

/// G4NucleiModel::getFermiKinetic. Returns 0 for anything but a proton or neutron, and for a
/// zone index at or past the surface - so a nucleon that has left the nucleus has no Fermi
/// energy, which is what `worthToPropagate` relies on.
__host__ __device__ inline double nm_fermi_kinetic(const NucleiModel& m, int ip, int izone) {
  if (ip >= 3 || izone >= m.number_of_zones) { return 0.0; }
  const double pf = m.fermi_momenta[ip - 1][izone];
  const double mass = inucl_particle_mass(ip);
  return std::sqrt(pf * pf + mass * mass) - mass;
}

__host__ __device__ inline double nm_radius(const NucleiModel& m, int izone) {
  return (izone < 0) ? 0.0
                     : (izone < m.number_of_zones) ? m.zone_radii[izone] : m.nuclei_radius;
}
__host__ __device__ inline double nm_volume(const NucleiModel& m, int izone) {
  return (izone < 0) ? 0.0
                     : (izone < m.number_of_zones) ? m.zone_volumes[izone] : m.nuclei_volume;
}

/// G4NucleiModel::getZone - the radius-to-zone search. Strictly `r < zone_radii[iz]`, so a
/// particle exactly on a boundary belongs to the OUTER zone, and a radius at or beyond the
/// surface returns `number_of_zones`, one past the last valid index. Every caller treats that
/// value as "outside".
__host__ __device__ inline int nm_zone(const NucleiModel& m, double r) {
  for (int iz = 0; iz < m.number_of_zones; ++iz) {
    if (r < m.zone_radii[iz]) { return iz; }
  }
  return m.number_of_zones;
}

/// G4NucleiModel::getRatio - the fraction of each nucleon species still available, and the
/// PRODUCT of two of them for a dibaryon. Anything else gives 0, which is the `default:` arm.
__host__ __device__ inline double nm_ratio(const NucleiModel& m, int ip) {
  const double rp = double(m.proton_number_current) / double(m.proton_number);
  const double rn = double(m.neutron_number_current) / double(m.neutron_number);
  switch (ip) {
    case kProton: return rp;
    case kNeutron: return rn;
    case kDiproton: return rp * rp;
    case kUnboundPN: return rp * rn;
    case kDineutron: return rn * rn;
    default: return 0.0;
  }
}

/// G4NucleiModel::getCurrentDensity.
///
/// For a dibaryon the density is the PRODUCT of two nucleon densities times the zone VOLUME -
/// the comment says "remove extra 1/volume term in density product" - and `pn_spec`, the scale
/// factor for pn against pp/nn, is 1.0 with the alternative 0.5 commented out beside it.
__host__ __device__ inline double nm_current_density(const NucleiModel& m, int ip, int izone) {
  const double pn_spec = 1.0;
  double dens = 0.0;
  if (ip < 100) {
    dens = nm_density(m, ip, izone);
  } else {
    if (ip == kDiproton) {
      dens = nm_density(m, kProton, izone) * nm_density(m, kProton, izone);
    } else if (ip == kUnboundPN) {
      dens = nm_density(m, kProton, izone) * nm_density(m, kNeutron, izone) * pn_spec;
    } else if (ip == kDineutron) {
      dens = nm_density(m, kNeutron, izone) * nm_density(m, kNeutron, izone);
    }
    dens *= nm_volume(m, izone);
  }
  return nm_ratio(m, ip) * dens;
}

/// G4NucleiModel::useQuasiDeuteron. A static: which projectiles can be absorbed on which
/// dibaryon. `qdtype == 0` means "any absorptive particle" and is the form the partner
/// generator calls it in; note that the pp arm admits the MUON and the nn arm does not, which
/// is charge conservation (mu- + pp -> n + p + nu, but mu- + nn has nowhere to go).
__host__ __device__ inline bool nm_use_quasideuteron(int ptype, int qdtype = 0) {
  if (qdtype == kUnboundPN || qdtype == 0) {
    return (ptype == kPionZero || ptype == kPionPlus || ptype == kPionMinus ||
            ptype == kPhoton || ptype == kMuonMinus);
  }
  if (qdtype == kDiproton) {
    return (ptype == kPionZero || ptype == kPionMinus || ptype == kPhoton ||
            ptype == kMuonMinus);
  }
  if (qdtype == kDineutron) {
    return (ptype == kPionZero || ptype == kPionPlus || ptype == kPhoton);
  }
  return false;
}

// =============================================================================================
// The two cross sections
// =============================================================================================

/// G4NucleiModel::absorptionCrossSection. Pions (and the muon, "use for muon capture as well")
/// get a two-branch parametrisation with a resonance term; the photon gets the tabulated
/// quasi-deuteron cross section times `gammaQDscale`. Above 1 GeV the pion branch leaves `csec`
/// at ZERO - neither branch covers `ke >= 1.0` - so pion absorption on a dibaryon simply stops
/// there, and that is not a clamp to the last value but a hard zero.
///
/// The `csec < 0` floor is Geant4's and it is reachable: the low-energy form
/// `0.1106/sqrt(ke) - 0.8 + 0.08/((ke-0.123)^2 + 0.0056)` is negative between about 20 and
/// 60 MeV away from the resonance.
///
/// `crossSectionUnits` multiplies the result, so the return value is millibarn times xsecScale
/// and not millibarn.
__host__ __device__ inline double nm_absorption_cross_section(double ke, int type,
                                                              const NucleiModelParams& p,
                                                              bool& refused) {
  refused = !nm_use_quasideuteron(type);
  if (refused) { return 0.0; }
  double csec = 0.0;
  if (type == kPionPlus || type == kPionMinus || type == kPionZero || type == kMuonMinus) {
    if (ke < 0.3) {
      csec = 0.1106 / std::sqrt(ke) - 0.8 +
             0.08 / ((ke - 0.123) * (ke - 0.123) + 0.0056);
    } else if (ke < 1.0) {
      csec = 3.6735 * (1.0 - ke) * (1.0 - ke);
    }
  }
  if (type == kPhoton) {
    csec = interp_value(ke, nm_gamma_qd_bins(), nm_gamma_qd_xsec(), 30) * p.gamma_qd_scale;
  }
  if (csec < 0.0) { csec = 0.0; }
  return p.cross_section_units * csec;
}

/// G4NucleiModel::totalCrossSection - the channel table's inclusive cross section, scaled.
/// `rtype` is the PRODUCT of the two type codes. A missing table is Geant4's
/// "unknown collison type" message and a zero; here it is a reported refusal.
__host__ __device__ inline double nm_total_cross_section(double ke, int rtype,
                                                         const NucleiModelParams& p,
                                                         bool& refused) {
  const ChannelTable t = channel_table(rtype);
  refused = !t.valid();
  if (refused) { return 0.0; }
  return p.cross_section_units * channel_cross_section(t, ke);
}

// =============================================================================================
// The sampling functions
// =============================================================================================

/// G4InuclSpecialFunctions::generateWithRandomAngles(p, mass) - an isotropic three-momentum of
/// the given magnitude, put on the mass shell. Two deviates: cos(theta) then phi, in that
/// order.
template <typename Rng>
__host__ __device__ inline LV inucl_with_random_angles(double p, double mass, Rng& rng) {
  double ct, st;
  inucl_random_cos_sin(rng, ct, st);
  const double phi = inucl_random_phi(rng);
  const double pt = p * st;
  return lv_set_vect_m(Vec3d{pt * std::cos(phi), pt * std::sin(phi), p * ct}, mass);
}

/// G4InuclSpecialFunctions::generateWithFixedTheta(ct, p, mass). One deviate, for phi, and note
/// `pt = p*sqrt(|1 - ct^2|)` - the absolute value lets a |ct| slightly above 1 through instead
/// of producing a NaN.
template <typename Rng>
__host__ __device__ inline LV inucl_with_fixed_theta(double ct, double p, double mass,
                                                     Rng& rng) {
  const double phi = inucl_random_phi(rng);
  const double pt = p * std::sqrt(std::fabs(1.0 - ct * ct));
  return lv_set_vect_m(Vec3d{pt * std::cos(phi), pt * std::sin(phi), p * ct}, mass);
}

/// G4NucleiModel::generateNucleonMomentum. `pf * cbrt(u)` samples |p| uniformly in momentum
/// SPACE below the Fermi surface, and the cube root is the DOUBLE overload of G4cbrt - the
/// exp/log one, not the tabulated integer one. Three deviates in total: one for the magnitude,
/// then two inside generateWithRandomAngles.
template <typename Rng>
__host__ __device__ inline LV nm_generate_nucleon_momentum(const NucleiModel& m, int type,
                                                           int zone, Rng& rng) {
  const double pmod = nm_fermi_momentum(m, type, zone) * inucl_cbrt(rng.uniform());
  return inucl_with_random_angles(pmod, inucl_particle_mass(type), rng);
}

/// G4NucleiModel::generateQuasiDeuteron. Two independent nucleon momenta ADDED, not one
/// dinucleon momentum thrown - Geant4's own FIXME asks why - so it costs six deviates and the
/// resulting four-vector is generally off the dibaryon's mass shell.
template <typename Rng>
__host__ __device__ inline LV nm_generate_quasideuteron(const NucleiModel& m, int type1,
                                                        int type2, int zone, int& dtype,
                                                        Rng& rng) {
  const LV m1 = nm_generate_nucleon_momentum(m, type1, zone, rng);
  const LV m2 = nm_generate_nucleon_momentum(m, type2, zone, rng);
  const int prod = type1 * type2;
  dtype = (prod == kProton * kProton) ? kDiproton
        : (prod == kProton * kNeutron) ? kUnboundPN
        : (prod == kNeutron * kNeutron) ? kDineutron : 0;
  return m1 + m2;
}

/// G4NucleiModel::inverseMeanFreePath. A RATE: cross section times current density. Three
/// special cases return exactly zero - a neutrino, a mu- on a neutron, and a zero or negative
/// cross section - and the zone index is clamped into the nucleus before any array lookup.
///
/// The kinetic energy the tables are looked up at is `getKinEnergyInTheTRS()`, the bullet's
/// energy in the TARGET's rest frame, and the target here is a nucleon with Fermi motion - so
/// the lookup energy is not the lab energy even for the incident projectile.
__host__ __device__ inline double nm_inverse_mean_free_path(const NucleiModel& m, int ptype,
                                                            const LV& bullet_mom,
                                                            int target_type,
                                                            const LV& target_mom, int zone,
                                                            const NucleiModelParams& p,
                                                            bool& refused) {
  refused = false;
  int iz = zone;
  if (iz < 0) { iz = 0; }
  if (iz >= m.number_of_zones) { iz = m.number_of_zones - 1; }
  if (inucl_is_neutrino(ptype)) { return 0.0; }
  if (ptype == kMuonMinus && target_type == kNeutron) { return 0.0; }

  LorentzConvertor lc;
  lc.bullet = bullet_mom;
  lc.target = target_mom;
  lc_to_the_center_of_mass(lc);
  const double ekin = lc_kin_energy_in_trs(lc);

  const double csec = (target_type < 100)
                          ? nm_total_cross_section(ekin, ptype * target_type, p, refused)
                          : nm_absorption_cross_section(ekin, ptype, p, refused);
  if (csec <= 0.0) { return 0.0; }
  return csec * nm_current_density(m, target_type, iz);
}

/// G4CascadParticle::young(cut, cpath) - `(current_path < 1000.) && (cpath < cut)`.
///
/// The two arguments do different jobs and the names invite swapping them. `cpath` is the
/// SAMPLED interaction length being vetted; `current_path` is how far this particle has already
/// travelled. The 1000 is not a distance in any physical sense: it is `G4NucleiModel::large`,
/// the value `initializeCascad` gives the incident projectile's `current_path`, so
/// `current_path < 1000.` is FALSE for the projectile and TRUE for every secondary (created with
/// `cpath = 0`). That is what makes the veto apply to "newly formed secondaries" and not to the
/// beam particle - the two are told apart by a sentinel in a distance field, not by a flag.
__host__ __device__ inline bool cp_young(double young_path_cut, double cpath,
                                         double current_path) {
  return (current_path < 1000.0) && (cpath < young_path_cut);
}

/// G4NucleiModel::generateInteractionLength. Returns `large` (1000) when nothing interacts,
/// which the caller compares against the geometric path - not the path itself.
template <typename Rng>
__host__ __device__ inline double nm_generate_interaction_length(double path, double invmfp,
                                                                 bool force_first,
                                                                 double current_path,
                                                                 Rng& rng) {
  const double young_cut = std::sqrt(10.0) * 0.25;
  const double huge_num = 50.0;
  double spath = kNmLarge;
  if (invmfp < kNmSmall) { return spath; }
  double pw = -path * invmfp;
  if (pw < -huge_num) { pw = -huge_num; }
  pw = 1.0 - std::exp(pw);
  if (force_first || (rng.uniform() < pw)) {
    spath = -std::log(1.0 - pw * rng.uniform()) / invmfp;
    if (cp_young(young_cut, spath, current_path)) { spath = kNmLarge; }
  }
  return spath;
}

/// G4CascadParticle::getPathToTheNextZone(rz_in, rz_out) - the distance to the next zone
/// boundary along the momentum, and it SETS `moving_in` as a side effect, which is what
/// boundaryTransition then reads to decide which way the step went.
///
/// Three things to keep:
///   * `|p|^2 < 1e-9` is "at rest" ("cut-off is 1 eV momentum") and returns a path of zero -
///     and, if the particle is in zone 0, clears `moving_in` so that a stuck particle can
///     escape. A zero path with a non-zero momentum is a different case and the caller
///     distinguishes them.
///   * the choice of which boundary to aim at depends on `current_zone == 0 || rp > 0`, i.e.
///     on whether the particle is in the innermost zone OR moving outward; inside that, the
///     OTHER boundary is tried when the first has no real intersection.
///   * `d2 < 0 && d2 > -1e-6` is snapped to zero "to account for round-off", and a `d2` that
///     stays negative leaves `path` at its initial -1, which the caller treats as an error.
__host__ __device__ inline double cp_path_to_next_zone(const Vec3d& position, const LV& mom,
                                                       int current_zone, double rz_in,
                                                       double rz_out, bool& moving_in) {
  double path = -1.0;
  const double rp = g4gpu::dot(mom.v, position);
  const double rr = g4gpu::mag2(position);
  double pp = g4gpu::mag2(mom.v);
  if (std::fabs(pp) < 1e-9) {
    if (current_zone == 0) { moving_in = false; }
    return 0.0;
  }
  const double ra = rr - rp * rp / pp;
  pp = std::sqrt(pp);
  double ds, d2;
  if (current_zone == 0 || rp > 0.0) {
    d2 = rz_out * rz_out - ra;
    if (d2 > 0.0) {
      ds = 1.0;
      moving_in = false;
    } else {
      d2 = rz_in * rz_in - ra;
      ds = -1.0;
      moving_in = true;
    }
  } else {
    d2 = rz_in * rz_in - ra;
    if (d2 > 0.0) {
      ds = -1.0;
      moving_in = true;
    } else {
      d2 = rz_out * rz_out - ra;
      ds = 1.0;
      moving_in = false;
    }
  }
  if (d2 < 0.0 && d2 > -1e-6) { d2 = 0.0; }
  if (d2 > 0.0) { path = ds * std::sqrt(d2) - rp / pp; }
  return path;
}

/// G4CascadParticle::propagateAlongThePath.
__host__ __device__ inline Vec3d cp_propagate(const Vec3d& position, const LV& mom,
                                              double path) {
  return position + g4gpu::normalize(mom.v) * path;
}

/// G4NucleiModel::passFermi - an outgoing NUCLEON below the local Fermi momentum blocks the
/// whole interaction. Only nucleons are tested; a pion below anything passes.
__host__ __device__ inline bool nm_pass_fermi(const NucleiModel& m, const int* types,
                                              const double* moduli, int n, int zone) {
  for (int i = 0; i < n; ++i) {
    if (!inucl_is_nucleon(types[i])) { continue; }
    if (moduli[i] < m.fermi_momenta[types[i] - 1][zone]) { return false; }
  }
  return true;
}

/// G4NucleiModel::passTrailing - reject an interaction within `R_nucleon` of a previous one.
///
/// **With the dumped parameters this function cannot reject anything.** `R_nucleon` is
/// `G4CascadeParameters::radiusTrailing()`, which is 0 (both the initializer's `0.` and the
/// HDP default `BERT_RAD_TRAILING` are zero), and the test is `dist < R_nucleon` - strict - so
/// it is false even for a repeat at exactly the same point. The trailing effect is OFF in
/// 11.1.1 and this is where that is visible. Transcribed anyway, because the parameter is
/// settable from an environment variable and a run that sets it must be reproducible.
__host__ __device__ inline bool nm_pass_trailing(const Vec3d* hits, int n_hits,
                                                 const Vec3d& hit_position, double r_nucleon) {
  for (int i = 0; i < n_hits; ++i) {
    if (g4gpu::mag(hits[i] - hit_position) < r_nucleon) { return false; }
  }
  return true;
}

/// G4NucleiModel::forceFirst / isProjectile / worthToPropagate.
///
/// `isProjectile` is `generation == 0`, which is why the 20121205 history entry ("daughters
/// should have generation count incremented from parent") matters: before it, a secondary could
/// claim to be the projectile and be forced to interact.
__host__ __device__ inline bool nm_is_projectile(int generation) { return generation == 0; }

__host__ __device__ inline bool nm_force_first(int generation, int ptype) {
  return nm_is_projectile(generation) &&
         (inucl_is_photon(ptype) || inucl_is_muon(ptype));
}

/// G4NucleiModel::worthToPropagate. Only a particle that JUST REFLECTED is tested, and the cut
/// for a non-nucleon is 0 - the `getPotential(ip, zone)` that would have been used is commented
/// out in 11.1.1 ("Temporarily backing out use of potential for non-nucleons"), so a reflected
/// pion is always worth propagating.
__host__ __device__ inline bool nm_worth_to_propagate(const NucleiModel& m, bool reflected_now,
                                                      int ptype, int zone, double kin_energy) {
  if (!reflected_now) { return true; }
  const double ekin_scale = 2.0;
  const double ekin_cut = inucl_is_nucleon(ptype) ? nm_fermi_kinetic(m, ptype, zone) : 0.0;
  return kin_energy / ekin_scale > ekin_cut;
}

/// G4NucleiModel::boundaryTransition.
///
/// The 20141001 "Change sign of dv" entry and the four lines marked `// NAT` are one change:
/// a particle whose radial momentum cannot climb the potential step can still cross it on its
/// TRANSVERSE momentum, if the wall is thick. So there are three arms, not two:
///
///   qv <= 0 and qv+qperp <= 0   reflect; p_r -> -p_r, reflection counter incremented
///   qv > 0                      transmit; p_r -> sqrt(qv) with the sign of p_r
///   otherwise                   transmit on angular momentum; p_r -> 0.001*p_r and p_perp is
///                               RESCALED to `sqrt(pperp^2 + qv - p1r^2)`
///
/// `qv = dv^2 + 2 dv E + p_r^2` uses the TOTAL energy `mom.e()`, with the more-correct
/// `mom.m()` commented out beside it. `potentialThickness` is 1.0 and only enters `qperp`.
__host__ __device__ inline void nm_boundary_transition(const NucleiModel& m, int ptype,
                                                       const Vec3d& pos, LV& mom, int& zone,
                                                       bool moving_inside, int& n_reflections,
                                                       bool& in_zone_zero) {
  in_zone_zero = false;
  if (moving_inside && zone == 0) { in_zone_zero = true; return; }

  const double r = g4gpu::mag(pos);
  const double pmag = g4gpu::mag(mom.v);
  const double pr = g4gpu::dot(pos, mom.v) / r;
  const double pperp2 = pmag * pmag - pr * pr;
  const int next_zone = moving_inside ? zone - 1 : zone + 1;

  const double dv = nm_potential(m, ptype, next_zone) - nm_potential(m, ptype, zone);
  const double qv = dv * dv + 2.0 * dv * mom.e + pr * pr;
  const double qperp = 2.0 * pperp2 * 1.0 / r;    // potentialThickness == 1.0
  const double smallish = 0.001;

  double p1r = 0.0;
  bool adjust_pperp = false;
  if (qv <= 0.0 && qv + qperp <= 0.0) {
    p1r = -pr;
    ++n_reflections;
  } else if (qv > 0.0) {
    p1r = std::sqrt(qv);
    if (pr < 0.0) { p1r = -p1r; }
    zone = next_zone;
    n_reflections = 0;
  } else {
    p1r = smallish * pr;
    adjust_pperp = true;
    zone = next_zone;
    n_reflections = 0;
  }

  const double prr = (p1r - pr) / r;
  if (adjust_pperp) {
    const Vec3d old_pperp = mom.v - pos * (pr / r);
    const double m2 = pperp2 + qv - p1r * p1r;
    const double new_pperp_mag = std::sqrt((m2 > 0.0) ? m2 : 0.0);
    mom.v = old_pperp * (new_pperp_mag / std::sqrt(pperp2));
    mom.v = mom.v + pos * (p1r / r);
  } else {
    mom.v = mom.v + pos * prr;
  }
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_NUCLEI_MODEL_CUH
