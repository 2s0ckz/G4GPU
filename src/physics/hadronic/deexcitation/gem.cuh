// The 60 GEM channels: emission of a nucleus from He6 to Mg28 by the Generalised Evaporation
// Model, with Furihata's width instead of the Weisskopf integral the six light ejectiles use.
//
// Transcribed from G4GEMChannel, G4GEMProbability, G4GEMCoulombBarrier and the 60
// G4Xx GEMProbability subclasses (11.1.1); tables in src/data/gem_levels.hh, extracted by
// tools/extract_gem_tables.pl. JAERI-Data/Code 2001-105 is the paper G4GEMProbability's
// comments cite (S. Furihata), and the alpha/beta parameters are Dostrovsky's as amended by
// the notes added on proof.
//
// This is a different width from evaporation.cuh's, not a heavier version of it. There is no
// inverse cross section and no numerical integral: the width is a closed form in the
// residual's constant-temperature level density, integrated analytically over the ejectile's
// kinetic energy (that is what I0/I1/I3 are - the three integrals of the Fermi-gas and
// constant-temperature forms), times a geometrical cross section pi*Rb^2 with Furihata's Rb.
// So a GEM channel is cheap and an evaporation channel is not, and the default configuration
// uses both: G4EvaporationDefaultGEMFactory gives n, p, d, t, He3 and alpha to
// G4EvaporationChannel and everything from He6 to Mg28 to G4GEMChannel.
//
// **Two of the sixty channels are built out of two nuclides.** A G4GEMChannel holds an (A, Z)
// and a G4GEMProbability holds another one, and they are meant to be the same pair. For
// Be12 the channel is (12, 4) and the probability is (9, 4); for O17 the channel is (17, 8)
// and the probability is (17, 9). Both are 11.1.1 as installed, both are in the default
// channel set, and each one splits the channel in half:
//
//   the channel's pair    decides which nuclide is emitted, its mass, its Coulomb-barrier
//                         object, and the residual (Z, A) the fragment is turned into
//   the probability's     decides the residual the WIDTH is computed for, the nuclear mass in
//                         the spin factor, the alpha/beta parameters, and Rb
//
// So the Be12 channel emits a Be12 with a Be9 channel's width, and the O17 channel emits an
// O17 with the width of a channel whose residual has one proton fewer. They are reproduced
// because they are what runs; RISK.md V37 records them. The two pairs are carried as separate
// fields rather than one, so that a reader cannot use the wrong one by accident.
#ifndef G4GPU_DEEX_GEM_CUH
#define G4GPU_DEEX_GEM_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "data/gem_levels.hh"
#include "data/level_data.cuh"
#include "physics/hadronic/deexcitation/corrections.cuh"
#include "physics/hadronic/deexcitation/coulomb_barrier.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"

namespace g4gpu::deex {

namespace u = g4gpu::units;

constexpr int kNumGemChannels = 60;

/// CLHEP hbar_Planck, MeV ns. CLHEP derives hbarc from it, so it is derived back here rather
/// than pinned again: units.cuh has the two factors and a second copy of a physical constant
/// is how RISK.md V8 happened.
__host__ __device__ inline double hbar_planck() {
  return u::hbarc<double>() / u::c_light<double>();
}

/// G4GEMProbability::fPlanck = hbar_Planck * G4Pow::logZ(2). It is the quantity an excited
/// state's width is compared against to decide whether the state is long-lived enough to
/// contribute: `fPlanck < width*ExcitLifetimes[i]`.
__host__ __device__ inline double gem_fplanck() {
  return hbar_planck() * std::log(2.0);
}

// ---------------------------------------------------------------------------------------------
// G4GEMCoulombBarrier
// ---------------------------------------------------------------------------------------------

/// G4GEMCoulombBarrier::CalcCompoundRadius(ARes) for an ejectile of (theA, theZ).
///
/// Three regimes and they are not continuous with each other: a single nucleon gets
/// 1.7*Ares^(1/3), a light ion up to alpha gets that plus 1.2 fm, and anything heavier gets
/// Furihata's two-body form with a +3.75 fm offset. Every one of the 60 GEM channels is in the
/// third; the first two are here because the barrier object is also built for the GEM n/p/d/t
/// channels of the full-GEM factory, which the default configuration does not use.
__host__ __device__ inline double gem_compound_radius(int ejA, int ARes) {
  const double ares13 = data::g4pow_z13<double>(ARes);
  double r;
  if (ejA == 1) {
    r = 1.7 * ares13;
  } else if (ejA <= 4) {
    r = 1.7 * ares13 + 1.2;
  } else {
    const double aej13 = data::g4pow_z13<double>(ejA);
    r = 1.12 * (ares13 + aej13) - 0.86 * (ares13 + aej13) / (ares13 * aej13) + 3.75;
  }
  return r * fermi();
}

/// G4GEMCoulombBarrier::GetCoulombBarrier(ARes, ZRes, U).
///
/// Not G4CoulombBarrier's: the radius is the compound radius above rather than
/// RadiusCB(res) + 0.4*RadiusCB(ejectile), and the excitation-dependent reduction is applied
/// UNCONDITIONALLY rather than only for U > 0 - which is the same thing for U >= 0 and is
/// written as Geant4 writes it. The barrier-penetration factor is applied only for theA <= 4,
/// i.e. never for the 60 default-GEM channels.
__host__ __device__ inline double gem_coulomb_barrier(int ejA, int ejZ, int ARes, int ZRes,
                                                      double U) {
  if (ejZ <= 0 || ZRes <= 0) { return 0.0; }
  double barrier = elm_coupling() * (static_cast<double>(ejZ) * ZRes) /
                   gem_compound_radius(ejA, ARes);
  if (ejA <= 4) { barrier *= barrier_penetration_factor(ejA, ejZ, ZRes); }
  barrier /= (1.0 + std::sqrt(U / ((2 * ARes) * u::MeV<double>())));
  return barrier;
}

// ---------------------------------------------------------------------------------------------
// G4GEMProbability's analytic integrals
// ---------------------------------------------------------------------------------------------

/// G4Pow::Z23(n) - the cube root of an INTEGER, squared. Not A23, which interpolates a table
/// and differs in the last bits; G4GEMProbability::CalcBetaParam calls Z23 with an integer
/// mass-number difference and the two are not interchangeable.
__host__ __device__ inline double gem_z23(int n) {
  const double x = data::g4pow_z13<double>(n);
  return x * x;
}

/// G4GEMProbability::I0 - the integral of the constant-temperature form.
__host__ __device__ inline double gem_i0(double t) { return std::exp(t) - 1.0; }

/// G4GEMProbability::I1. The `T` factor JMQ's 190709 note says was missing is applied by the
/// CALLER, not here; this is the dimensionless integral.
__host__ __device__ inline double gem_i1(double t, double tx) {
  return (t - tx + 1.0) * std::exp(tx) - t - 1.0;
}

/// G4GEMProbability::I3 - the asymptotic expansion of the Fermi-gas integral, to six terms.
/// The coefficients are Furihata's; they are written out rather than generated because the
/// series is truncated where it is truncated and a closed form would not truncate there.
__host__ __device__ inline double gem_i3(double s0, double sx) {
  const double s2 = s0 * s0;
  const double sx2 = sx * sx;
  const double S = 1.0 / std::sqrt(s0);
  const double S2 = S * S;
  const double Sx = 1.0 / std::sqrt(sx);
  const double Sx2 = Sx * Sx;

  const double p1 = S * (2.0 + S2 * (4.0 + S2 * (13.5 + S2 * (60.0 + S2 * 325.125))));
  double p2 = Sx * Sx2 *
              ((s2 - sx2) +
               Sx2 * ((1.5 * s2 + 0.5 * sx2) +
                      Sx2 * ((3.75 * s2 + 0.25 * sx2) +
                             Sx2 * ((12.875 * s2 + 0.625 * sx2) +
                                    Sx2 * ((59.0625 * s2 + 0.9375 * sx2) +
                                           Sx2 * (324.8 * s2 + 3.28 * sx2))))));
  p2 *= std::exp(sx - s0);
  return p1 - p2;
}

/// G4GEMProbability::CCoeficient(aZ) for an ejectile of mass number theA. Furihata's C, the
/// values Dostrovsky's paper carries as notes added on proof: {20, 0}, {30, -0.06},
/// {40, -0.10}, {50, -0.10}, with a quartic through them in between.
__host__ __device__ inline double gem_c_coefficient(int aZ, int ejA) {
  if (aZ >= 50) { return -0.10 / static_cast<double>(ejA); }
  if (aZ > 20) {
    const double z = static_cast<double>(aZ);
    return (0.123482 - 0.00534691 * z - 0.0000610624 * z * z + 5.93719e-7 * z * z * z +
            1.95687e-8 * z * z * z * z) /
           static_cast<double>(ejA);
  }
  return 0.0;
}

/// G4GEMProbability::CalcAlphaParam. `ejA`/`ejZ` are the PROBABILITY's pair.
__host__ __device__ inline double gem_alpha_param(int fragA, int fragZ, int ejA, int ejZ) {
  if (ejZ == 0) { return 0.76 + 1.93 / data::g4pow_z13<double>(fragA - ejA); }
  return 1.0 + gem_c_coefficient(fragZ - ejZ, ejA);
}

// ---------------------------------------------------------------------------------------------
// G4GEMProbability::CalcProbability - one width
// ---------------------------------------------------------------------------------------------

/// The channel's identity, split the two ways Geant4 splits it.
struct GemChannel {
  int chan_a, chan_z;
  int prob_a, prob_z;
  double spin;          ///< the ground-state spin; an excited state substitutes its own
  int first_level, n_levels;
};

__host__ __device__ inline GemChannel gem_channel(int i) {
  const data::GemChannelEntry& e = data::gem_channels()[i];
  GemChannel c;
  c.chan_a = e.chan_a;
  c.chan_z = e.chan_z;
  c.prob_a = e.prob_a;
  c.prob_z = e.prob_z;
  c.spin = e.spin;
  c.first_level = e.first_level;
  c.n_levels = e.n_levels;
  return c;
}

/// G4GEMProbability::CalcProbability(fragment, MaximalKineticEnergy, V).
///
/// `spin` is passed rather than read from the channel because EmissionProbability overwrites
/// G4GEMProbability::Spin with each excited state's spin and restores it afterwards - the
/// member is a loop variable in Geant4, which is why it is an argument here.
///
/// Every quantity below is labelled RESIDUAL or PARENT the way JMQ's September 2009 comments
/// label them, because the whole history of this function is those two being confused: the
/// residual's level-density parameter is evaluated at `MaxKE + V - delta0` and the parent's at
/// `U - deltaCN`, and the initial level density is the parent's alone.
__host__ __device__ inline double gem_calc_probability(const GemChannel& c, const Fragment& frag,
                                                        double spin, double max_ke, double V,
                                                        const data::LevelTable& lt) {
  const int A = frag.a;
  const int Z = frag.z;
  const int resA = A - c.prob_a;
  const int resZ = Z - c.prob_z;
  const double U = frag.excitation;

  // ComputeGroundStateMass(theZ, theA) with the PROBABILITY's pair.
  const double nuclear_mass_ej = deex::nuclear_mass(c.prob_a, c.prob_z);

  const double alpha = gem_alpha_param(A, Z, c.prob_a, c.prob_z);
  // CalcBetaParam: for a charged ejectile Beta is exactly minus the barrier, so (Beta + V)
  // vanishes in the constant-temperature branch below. Geant4's own comment says so and
  // deletes the neutral-ejectile correction term because of it; the term is not restored here.
  const double beta = (c.prob_z == 0)
                          ? (1.66 / gem_z23(A - c.prob_a) - 0.05) * u::MeV<double>() / alpha
                          : -V;

  //                             *** RESIDUAL ***
  const double delta0 = deex::level_data_pairing_correction(resZ, resA);
  const bool res_levels = (data::find_manager(lt, resZ, resA) >= 0);
  const double a = deex::level_density(resZ, resA, max_ke + V - delta0, res_levels);
  const double Ux = (2.5 + 150.0 / static_cast<double>(resA)) * u::MeV<double>();
  const double Ex = Ux + delta0;
  const double T = 1.0 / (std::sqrt(a / Ux) - 1.5 / Ux);
  const double E0 = Ex - T * (std::log(T) - std::log(a) / 4.0 - 1.25 * std::log(Ux) +
                              2.0 * std::sqrt(a * Ux));

  //                             *** PARENT ***
  const double deltaCN = deex::level_data_pairing_correction(Z, A);
  const bool par_levels = (data::find_manager(lt, Z, A) >= 0);
  const double aCN = deex::level_density(Z, A, U - deltaCN, par_levels);
  const double UxCN = (2.5 + 150.0 / static_cast<double>(A)) * u::MeV<double>();
  const double ExCN = UxCN + deltaCN;
  const double TCN = 1.0 / (std::sqrt(aCN / UxCN) - 1.5 / UxCN);

  double width;
  const double t = max_ke / T;
  if (max_ke < Ex) {
    width = (gem_i1(t, t) * T + (beta + V) * gem_i0(t)) / std::exp(E0 / T);
  } else {
    const double sqrt2 = std::sqrt(2.0);
    const double expE0T = std::exp(E0 / T);
    const double tx = Ex / T;
    double s0 = 2.0 * std::sqrt(a * (max_ke - delta0));
    const double sx = 2.0 * std::sqrt(a * (Ex - delta0));
    // VI's protection against an FPE in exp(s0). It is a clamp on the answer, not on an
    // intermediate: above 350 the width stops growing with the excitation.
    if (s0 > 350.0) { s0 = 350.0; }
    width = gem_i1(t, tx) * T / expE0T + gem_i3(s0, sx) * std::exp(s0) / (sqrt2 * a);
  }

  // hbarc, not hbar_Planck. JMQ's 14/07/2009 fix, and the comment in Geant4 shouts about it:
  // NuclearMass is an energy, so the spin factor needs (hbar c)^2 to come out dimensionless.
  const double gg = (2.0 * spin + 1.0) * nuclear_mass_ej /
                    (pi2() * u::hbarc<double>() * u::hbarc<double>());

  // Furihata's Rb, JAERI-Data/Code 2001-105 p6, on the PROBABILITY's mass number.
  double Rb = 0.0;
  const double Ad = data::g4pow_z13<double>(resA);
  if (c.prob_a > 4) {
    const double Aj = data::g4pow_z13<double>(c.prob_a);
    Rb = (1.12 * (Aj + Ad) - 0.86 * ((Aj + Ad) / (Aj * Ad)) + 2.85) * fermi();
  } else if (c.prob_a > 1) {
    const double Aj = data::g4pow_z13<double>(c.prob_a);
    Rb = 1.5 * (Aj + Ad) * fermi();
  } else {
    Rb = 1.5 * Ad * fermi();
  }
  const double geometrical_xs = u::pi<double>() * Rb * Rb;

  double initial_level_density;
  if (U < ExCN) {
    const double E0CN = ExCN - TCN * (std::log(TCN) - 0.25 * std::log(aCN) -
                                      1.25 * std::log(UxCN) + 2.0 * std::sqrt(aCN * UxCN));
    initial_level_density = (u::pi<double>() / 12.0) * std::exp((U - E0CN) / TCN) / TCN;
  } else {
    const double x = U - deltaCN;
    const double x1 = std::sqrt(aCN * x);
    initial_level_density = (u::pi<double>() / 12.0) * std::exp(2 * x1) / (x * std::sqrt(x1));
  }

  // pi, not sqrt(pi): JMQ's 190709 note, again against Furihata's report.
  width *= u::pi<double>() * gg * geometrical_xs * alpha / (12.0 * initial_level_density);
  return width;
}

/// G4GEMProbability::EmissionProbability - the ground-state width plus one width per excited
/// state of the emitted nuclide whose lifetime is long enough.
///
/// The `fPlanck < width*ExcitLifetimes[i]` test is JMQ's April 2010 condition "added to
/// prevent reported crash", and it is a physical one: a state whose width times its lifetime
/// is below hbar*ln2 is not a state. Note that it gates only the ADDITION - the width is
/// computed either way, and a state above the kinetic limit (Tmax <= 0) is skipped instead.
__host__ __device__ inline double gem_emission_probability(const GemChannel& c,
                                                            const Fragment& frag, double max_ke,
                                                            const data::LevelTable& lt) {
  if (max_ke <= 0.0 || frag.excitation <= 0.0) { return 0.0; }

  // GetCoulombBarrier(fragment): the barrier OBJECT is the channel's (its constructor took the
  // channel's A and Z) but the residual it is asked about is the PROBABILITY's, and the
  // excitation is reduced by the parent's pairing correction. For Be12 and O17 those are two
  // different nuclides; see the file header.
  const double cb = gem_coulomb_barrier(
      c.chan_a, c.chan_z, frag.a - c.prob_a, frag.z - c.prob_z,
      frag.excitation - deex::level_data_pairing_correction(frag.z, frag.a));

  double probability = gem_calc_probability(c, frag, c.spin, max_ke, cb, lt);

  const double fplanck = gem_fplanck();
  for (int i = 0; i < c.n_levels; ++i) {
    const int k = c.first_level + i;
    const double tmax = max_ke - data::gem_level_energy()[k];
    if (tmax <= 0.0) { continue; }
    const double width =
        gem_calc_probability(c, frag, data::gem_level_spin()[k], tmax, cb, lt);
    if (width > 0.0 && fplanck < width * data::gem_level_lifetime()[k]) {
      probability += width;
    }
  }
  return probability;
}

// ---------------------------------------------------------------------------------------------
// G4GEMChannel
// ---------------------------------------------------------------------------------------------

/// What G4GEMChannel::GetEmissionProbability leaves behind for SampleKineticEnergy: the
/// residual it chose, the barrier and the kinetic limit. Geant4 keeps these as members of the
/// channel object and EmittedFragment reads them, so the same coupling exists here.
struct GemState {
  int ch = 0;              ///< index into gem_channels()
  int res_a = 0, res_z = 0;
  double coulomb_barrier = 0.0;
  double max_kinetic_energy = 0.0;
  double emission_probability = 0.0;
};

/// G4GEMChannel::GetEmissionProbability.
///
/// Three conditions and they are NOT the evaporation channel's. `ResidualA >= ResidualZ`,
/// `ResidualZ >= 0` and `ResidualA >= A` - and no exclusion of a residual that is all protons
/// or all neutrons, which G4EvaporationChannel does exclude. So a GEM channel will leave a
/// residual an evaporation channel would refuse to. Reproduced as written.
__host__ __device__ inline double gem_channel_emission_probability(GemState& s,
                                                                    const Fragment& frag,
                                                                    const data::LevelTable& lt) {
  const GemChannel c = gem_channel(s.ch);
  s.emission_probability = 0.0;
  s.res_a = frag.a - c.chan_a;
  s.res_z = frag.z - c.chan_z;
  if (!(s.res_a >= s.res_z && s.res_z >= 0 && s.res_a >= c.chan_a)) { return 0.0; }

  const double ex_energy = frag.excitation - deex::level_data_pairing_correction(frag.z, frag.a);
  if (ex_energy <= 0.0) { return 0.0; }

  const double res_mass = deex::nuclear_mass(s.res_a, s.res_z);
  const double evap_mass = deex::nuclear_mass(c.chan_a, c.chan_z);
  const double etot = frag.ground_state_mass + ex_energy;
  s.coulomb_barrier = gem_coulomb_barrier(c.chan_a, c.chan_z, s.res_a, s.res_z, ex_energy);
  if (etot <= res_mass + evap_mass + s.coulomb_barrier) { return 0.0; }

  s.max_kinetic_energy = ((etot - res_mass) * (etot + res_mass) + evap_mass * evap_mass) /
                             (2.0 * etot) - evap_mass - s.coulomb_barrier;
  if (s.max_kinetic_energy <= 0.0) { return 0.0; }

  s.emission_probability = gem_emission_probability(c, frag, s.max_kinetic_energy, lt);
  return s.emission_probability;
}

/// G4GEMChannel::SampleKineticEnergy.
///
/// A second, independent evaluation of the same physics as CalcProbability - not a call to it.
/// The differences are real and are the reason this is transcribed separately rather than
/// factored: here the residual is the CHANNEL's (so Be12 and O17 sample against a different
/// residual than they integrated over), `Rb` branches on the channel's A, the level-density
/// parameter of the residual is re-evaluated at each trial energy, and the Fermi-gas branch is
/// `exp(2 sqrt(a e) - 0.25 log(a e^5))` rather than the I3 expansion. Factoring the two
/// together would have to pick one of each pair.
///
/// The rejection ceiling is the TOTAL emission probability, excited states included, while the
/// trial value is the ground-state expression alone - so the loop is heavily over-covered and
/// almost always exits by the 100-iteration cap having accepted nothing, returning the last
/// trial energy. That is Geant4's sampler and it is what shapes the GEM spectrum.
template <typename Rng>
__host__ __device__ inline double gem_sample_kinetic_energy(const GemState& s,
                                                             const Fragment& frag,
                                                             const data::LevelTable& lt,
                                                             Rng& rng) {
  const GemChannel c = gem_channel(s.ch);
  const double U = frag.excitation;
  const double alpha = gem_alpha_param(frag.a, frag.z, c.prob_a, c.prob_z);
  // CalcBetaParam again, and again through the PROBABILITY's pair: the barrier it negates is
  // the one GetCoulombBarrier(fragment) returns, not the channel's s.coulomb_barrier.
  const double cb_prob = gem_coulomb_barrier(
      c.chan_a, c.chan_z, frag.a - c.prob_a, frag.z - c.prob_z,
      U - deex::level_data_pairing_correction(frag.z, frag.a));
  const double beta = (c.prob_z == 0)
                          ? (1.66 / gem_z23(frag.a - c.prob_a) - 0.05) * u::MeV<double>() / alpha
                          : -cb_prob;

  //                             *** RESIDUAL *** (the channel's)
  const double delta0 = deex::level_data_pairing_correction(s.res_z, s.res_a);
  const double Ux = (2.5 + 150.0 / static_cast<double>(s.res_a)) * u::MeV<double>();
  const double Ex = Ux + delta0;
  const bool res_levels = (data::find_manager(lt, s.res_z, s.res_a) >= 0);

  //                             *** PARENT ***
  const double deltaCN = deex::level_data_pairing_correction(frag.z, frag.a);
  const bool par_levels = (data::find_manager(lt, frag.z, frag.a) >= 0);
  const double aCN = deex::level_density(frag.z, frag.a, U - deltaCN, par_levels);
  const double UxCN = (2.5 + 150.0 / static_cast<double>(frag.a)) * u::MeV<double>();
  const double ExCN = UxCN + deltaCN;
  const double TCN = 1.0 / (std::sqrt(aCN / UxCN) - 1.5 / UxCN);

  double initial_level_density;
  if (U < ExCN) {
    const double E0CN = ExCN - TCN * (std::log(TCN) - std::log(aCN) / 4.0 -
                                      1.25 * std::log(UxCN) + 2.0 * std::sqrt(aCN * UxCN));
    initial_level_density = (u::pi<double>() / 12.0) * std::exp((U - E0CN) / TCN) / TCN;
  } else {
    const double x = U - deltaCN;
    const double x1 = std::sqrt(aCN * x);
    initial_level_density = (u::pi<double>() / 12.0) * std::exp(2 * x1) / (x * std::sqrt(x1));
  }

  const double evap_mass = deex::nuclear_mass(c.chan_a, c.chan_z);
  const double gg = (2.0 * c.spin + 1.0) * evap_mass /
                    (pi2() * u::hbarc<double>() * u::hbarc<double>());

  double Rb = 0.0;
  const double Ad = data::g4pow_z13<double>(s.res_a);
  if (c.chan_a > 4) {
    const double Aj = data::g4pow_z13<double>(c.chan_a);
    Rb = (1.12 * (Aj + Ad) - 0.86 * ((Aj + Ad) / (Aj * Ad)) + 2.85) * fermi();
  } else if (c.chan_a > 1) {
    const double Aj = data::g4pow_z13<double>(c.chan_a);
    Rb = 1.5 * (Aj + Ad) * fermi();
  } else {
    Rb = 1.5 * Ad * fermi();
  }
  const double geometrical_xs = u::pi<double>() * Rb * Rb;
  const double constant_factor =
      gg * geometrical_xs * alpha * u::pi<double>() / (initial_level_density * 12.0);

  const double the_energy = s.max_kinetic_energy + s.coulomb_barrier;
  double kinetic_energy = 0.0;
  for (int i = 0; i < 100; ++i) {
    kinetic_energy = s.coulomb_barrier + rng.uniform() * s.max_kinetic_energy;
    const double edelta = the_energy - kinetic_energy - delta0;
    double probability = constant_factor * (kinetic_energy + beta);
    const double a = deex::level_density(s.res_z, s.res_a, edelta, res_levels);
    const double T = 1.0 / (std::sqrt(a / Ux) - 1.5 / Ux);
    if (the_energy - kinetic_energy < Ex) {
      const double E0 = Ex - T * (std::log(T) - std::log(a) * 0.25 - 1.25 * std::log(Ux) +
                                  2.0 * std::sqrt(a * Ux));
      probability *= std::exp((the_energy - kinetic_energy - E0) / T) / T;
    } else {
      const double e2 = edelta * edelta;
      probability *= std::exp(2 * std::sqrt(a * edelta) -
                              0.25 * std::log(a * edelta * e2 * e2));
    }
    if (s.emission_probability * rng.uniform() <= probability) { break; }
  }
  return kinetic_energy;
}

/// G4GEMChannel::EmittedFragment. `frag` becomes the residual; the ejectile is returned.
template <typename Rng>
__host__ __device__ inline Fragment gem_emitted_fragment(const GemState& s, Fragment& frag,
                                                          const data::LevelTable& lt, Rng& rng) {
  const GemChannel c = gem_channel(s.ch);
  const double evap_mass = deex::nuclear_mass(c.chan_a, c.chan_z);
  const double ev_energy = gem_sample_kinetic_energy(s, frag, lt, rng) + evap_mass;

  const Vec3d dir = random_direction(rng);
  const double p = std::sqrt((ev_energy - evap_mass) * (ev_energy + evap_mass));
  LorentzVector lv(dir * p, ev_energy);
  LorentzVector lv0 = frag.momentum;
  lv.boost(lv0.boost_vector());

  Fragment ev = make_fragment(c.chan_a, c.chan_z, lv);
  lv0 -= lv;
  frag.set_za_and_momentum(lv0, s.res_z, s.res_a);
  return ev;
}

}  // namespace g4gpu::deex

#endif
