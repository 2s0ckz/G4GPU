// The samplers and the three decisions G4LundStringFragmentation's control flow is made of.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/hadronization/src/G4VLongitudinalStringDecay.cc
//     SampleQuarkFlavor, SampleQuarkPt, CreatePartonPair, QuarkSplitup
//   .../src/G4LundStringFragmentation.cc
//     GetLightConeZ, StopFragmenting, IsItFragmentable, Sample4Momentum, DiQuarkSplitup,
//     lambda
//
// WHY THESE ARE SEPARATE FUNCTIONS AND NOT EXPRESSIONS INSIDE FragmentString. docs/RISK.md
// V52: three of P6's twenty perturbations were not caught because the decision they named was
// an expression inline in `deexcite`, so there was nothing a test could address. The same
// three decisions here - StopFragmenting's Gaussian rejection, IsItFragmentable's
// `|Mmin| < M`, and Sample4Momentum's two-body kinematics - are the ones that decide whether
// a string fragments at all, and each is a function with its own oracle column
// (ref/oracle/ftf_decisions.csv) and its own bucket.
//
// FOUR DEAD LINES, each established by perturbation rather than by reading:
//
//  1. StopFragmenting's second `if (MinimalStringMass < 0.0) return false;` sits inside the
//     `else` of a test that already returned true for `MinimalStringMass < 0.`. It cannot run.
//  2. Sample4Momentum's `if (loopCounter >= maxNumberOfLoops) { AvailablePz2 = 0.0; }` is
//     followed immediately by an unconditional assignment to AvailablePz2. The clamp has no
//     effect; a string whose Pt loop exhausted still gets the Pz the formula gives, which for
//     `MassMt + AntiMassMt > InitialMass` is the square root of a negative number.
//  3. Sample4Momentum applies its `1 - 0.55*((m1+m2)/M)^2` factor to SigmaQT TWICE for a
//     qq-qqbar string: once under `(Mass > 930 || AntiMass > 930)` and again under
//     `(Mass > 930 && AntiMass > 930)`. Not dead - it is the actual width used - but it is not
//     what the surrounding comments describe, and for a heavy pair the factor can go negative,
//     which flips the sampled Pt's sign and is invisible because SampleQuarkPt then puts it
//     through a cosine.
//  4. The `if (Mass < 930. && AntiMass < 930.) {}` and the q-di_q arm's body are empty, with
//     the isotropic `SigmaQT = -1.` commented out. Transcribed as the two no-ops they are, so
//     that the file shows all four mass cases rather than three.
//
// G4Pow, NOT std::pow. GetLightConeZ's `powA(1-z, Alund)` and `powA(u, 1/an)` are G4Pow
// expansions; data/g4pow.hh has them, and the difference from std::pow is 1e-7 - four orders
// above this test's tolerance (docs/HADRONIC_PLAN.md section 8).
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "physics/hadronic/ftf/hadron_builder.cuh"
#include "physics/hadronic/ftf/lund_tables.cuh"
#include "physics/hadronic/ftf/refusal.cuh"

namespace g4gpu::hadronic::ftf {

/// G4VLongitudinalStringDecay::SampleQuarkFlavor.
///
/// One deviate for the heavy-flavour gate, a second for the light flavour. `ProbCB` is
/// ProbCCbar + ProbBBbar = 2.5e-4 in 11.1.1, so the heavy branch fires once in four thousand
/// calls - which is why G4HadronBuilder's charm and bottom substitution tables are reachable
/// from a proton beam and are ported rather than refused.
///
/// `1 + (int)(u/StrangeSuppress)` with StrangeSuppress = 0.44 gives 1, 2 or 3 (d, u, s) at
/// 0.44 : 0.44 : 0.12. Splitup rewrites StrangeSuppress from the string mass before every
/// split, so it is taken from the tables here and the caller is expected to have set it -
/// which is what makes the oracle sweep it.
template <typename real_t, typename Rng>
__host__ __device__ inline int ftf_sample_quark_flavor(const LundTables<real_t>* t, Rng& rng) {
  int quark = 1;
  const real_t ksi = static_cast<real_t>(rng.uniform());
  if (ksi < t->prob_cb) {
    quark = (ksi < t->prob_ccbar) ? 4 : 5;
  } else {
    quark = 1 + static_cast<int>(static_cast<real_t>(rng.uniform()) / t->strange_suppress);
  }
  return quark;
}

/// G4VLongitudinalStringDecay::SampleQuarkPt, with SigmaQT given explicitly because
/// Sample4Momentum temporarily rescales it.
///
/// `G4RandFlat::shoot(ymin, 1.)` is CLHEP's `(b-a)*flat() + a`, so the limited-range arm is
/// `-log((1 - ymin)*u + ymin)` and consumes ONE deviate, as the full-Gaussian arm does. The
/// `q > 20` guard makes `exp(-q*q)` exactly 0 rather than a denormal, which matters because
/// ymin = 0 turns the limited arm back into the unlimited one.
template <typename real_t, typename Rng>
__host__ __device__ inline void ftf_sample_quark_pt_sigma(real_t sigma_qt, real_t pt_max,
                                                          Rng& rng, real_t* px, real_t* py) {
  real_t pt;
  if (pt_max < real_t(0)) {
    pt = -log(static_cast<real_t>(rng.uniform()));
  } else {
    const real_t q = pt_max / sigma_qt;
    const real_t ymin = (q > real_t(20.)) ? real_t(0.0) : exp(-q * q);
    const real_t u = static_cast<real_t>(rng.uniform());
    pt = -log((real_t(1.0) - ymin) * u + ymin);
  }
  pt = sigma_qt * sqrt(pt);
  const real_t phi = real_t(2.) * units::pi<real_t>() * static_cast<real_t>(rng.uniform());
  *px = pt * cos(phi);
  *py = pt * sin(phi);
}

template <typename real_t, typename Rng>
__host__ __device__ inline void ftf_sample_quark_pt(const LundTables<real_t>* t, real_t pt_max,
                                                    Rng& rng, real_t* px, real_t* py) {
  ftf_sample_quark_pt_sigma(t->sigma_qt, pt_max, rng, px, py);
}

/// G4LundStringFragmentation::GetLightConeZ - the Lund symmetric fragmentation function.
///
/// Two arms. The QUARK arm (also taken by a diquark that produces a meson) samples
/// `f(z) = (1-z)^a / z * exp(-b mT^2 / z)` by rejection against its maximum, with a = 1 and
/// b = 0.7 GeV^-2; because a is exactly 1 the mode has the closed form `b mT^2/(b mT^2 + 1)`
/// and the `else` branch that solves the quadratic is unreachable - it is transcribed anyway,
/// because `Alund` is a local that a tune could change. The DIQUARK arm (a diquark producing a
/// baryon) is a power law `z = zmin + (zmax-zmin) u^(1/an)` with `an = 2.0 + pt^2/GeV^2`, and
/// a parton code above 3000 - a diquark containing an s, c or b quark - REFLECTS the result
/// about the middle of the interval.
///
/// The rejection loop is capped at 1000 iterations and then answers `0.5*(zmin+zmax)`, which
/// Geant4's own comment calls "just a value between zmin and zmax, no physics considerations
/// at all". Reproduced, and it is why the loop is not a refusal.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t ftf_get_light_cone_z(const LundTables<real_t>* t, real_t zmin,
                                                       real_t zmax, int parton_pdg,
                                                       int hadron_pdg, real_t px, real_t py,
                                                       Rng& rng) {
  const data::FtfHadron* hd = data::ftf_find_hadron(hadron_pdg);
  const real_t mass = (hd != nullptr) ? static_cast<real_t>(hd->mass) : real_t(0);
  const int hadron_encoding = (hadron_pdg < 0) ? -hadron_pdg : hadron_pdg;
  const int abs_parton = (parton_pdg < 0) ? -parton_pdg : parton_pdg;

  const real_t mt2 = px * px + py * py + mass * mass;
  real_t z = real_t(0.);

  if (!((abs_parton > 1000) && (hadron_encoding > 1000))) {
    // Quark fragmentation, and qq -> meson.
    const real_t alund = real_t(1.);
    const real_t blund = real_t(0.7) / units::GeV<real_t>() / units::GeV<real_t>();
    const real_t bmt2 = blund * mt2;
    real_t z_of_max_yf;
    if (alund == real_t(1.0)) {
      z_of_max_yf = bmt2 / (blund * mt2 + real_t(1.));
    } else {
      const real_t one_minus = real_t(1.0) - bmt2;
      z_of_max_yf = ((real_t(1.0) + bmt2) -
                     sqrt(one_minus * one_minus + real_t(4.0) * bmt2 * alund)) /
                    real_t(2.0) / (real_t(1.) - alund);
    }
    if (z_of_max_yf < zmin) { z_of_max_yf = zmin; }
    if (z_of_max_yf > zmax) { z_of_max_yf = zmax; }
    const real_t max_yf =
        (real_t(1) - z_of_max_yf) / z_of_max_yf * exp(-blund * mt2 / z_of_max_yf);

    const int max_loops = 1000;
    int loop = 0;
    real_t yf = real_t(1.);
    do {
      z = zmin + static_cast<real_t>(rng.uniform()) * (zmax - zmin);
      yf = data::g4pow_pow_a<real_t>(real_t(1.0) - z, alund) / z * exp(-bmt2 / z);
    } while ((static_cast<real_t>(rng.uniform()) * max_yf > yf) && ++loop < max_loops);
    if (loop >= max_loops) { z = real_t(0.5) * (zmin + zmax); }
    return z;
  }

  if (abs_parton > 1000) {
    real_t an = real_t(2.5);
    an += (px * px + py * py) / (units::GeV<real_t>() * units::GeV<real_t>()) - real_t(0.5);
    z = zmin + (zmax - zmin) * data::g4pow_pow_a<real_t>(
                                   static_cast<real_t>(rng.uniform()), real_t(1.) / an);
    if (parton_pdg > 3000) { z = zmin + zmax - z; }
  }
  return z;
}

/// G4LundStringFragmentation::lambda - the two-body phase-space triangle function.
template <typename real_t>
__host__ __device__ inline real_t ftf_lambda(real_t s, real_t m1_sqr, real_t m2_sqr) {
  const real_t d = s - m1_sqr - m2_sqr;
  return d * d - real_t(4.) * m1_sqr * m2_sqr;
}

/// G4LundStringFragmentation::IsItFragmentable - `|MinimalStringMass| < M`. The absolute
/// value is load-bearing: MinimalStringMass is -350 GeV for a string whose final two-body
/// decay has no particle, and without the `abs` that string would look fragmentable.
template <typename real_t>
__host__ __device__ inline bool ftf_is_it_fragmentable(real_t minimal_string_mass,
                                                       real_t string_mass) {
  const real_t a =
      (minimal_string_mass < real_t(0)) ? -minimal_string_mass : minimal_string_mass;
  return a < string_mass;
}

/// G4LundStringFragmentation::StopFragmenting.
///
/// The four-quark arm is a rejection on `exp(-0.0005*(M - Mmin))`, linear in the mass; the
/// ordinary arm on `exp(-0.66e-6*(M^2 - Mmin^2))`, quadratic. Both consume exactly one
/// deviate, and the `MinimalStringMass < 0` arm consumes none - which is what makes the draw
/// count in ref/oracle/ftf_decisions.csv a check on the branch taken and not only on the
/// verdict.
template <typename real_t, typename Rng>
__host__ __device__ inline bool ftf_stop_fragmenting(real_t minimal_string_mass,
                                                     real_t string_mass, bool is_four_quark,
                                                     Rng& rng) {
  if (minimal_string_mass < real_t(0.)) { return true; }
  if (is_four_quark) {
    return static_cast<real_t>(rng.uniform()) <
           exp(real_t(-0.0005) * (string_mass - minimal_string_mass));
  }
  // Geant4 repeats `if (MinimalStringMass < 0.0) return false;` here. It cannot run: the test
  // above already returned true for exactly that condition. Kept as a comment rather than as
  // code, because transcribing it would be transcribing a line with no behaviour - and
  // docs/RISK.md V52's rule is that a dead line needs a perturbation that fires if it runs at
  // all, which for this one is impossible by construction.
  return static_cast<real_t>(rng.uniform()) <
         exp(real_t(-0.66e-6) *
             (string_mass * string_mass - minimal_string_mass * minimal_string_mass));
}

/// What Sample4Momentum fills: the two back-to-back four-momenta in the string's rest frame.
template <typename real_t>
struct TwoBodyMomenta {
  real_t px = 0, py = 0, pz = 0, e = 0;
  real_t apx = 0, apy = 0, apz = 0, ae = 0;
};

/// G4LundStringFragmentation::Sample4Momentum.
///
/// `Pabs` is the exact two-body momentum and is used only as the Pt sampler's upper limit.
/// The 930 MeV tests are a stand-in for "is this a baryon": 930 sits between the eta' (958)
/// and the nucleon (938), so an eta' counts as a baryon here. Said out loud because it is the
/// kind of threshold that looks like a nucleon mass and is not one.
template <typename real_t, typename Rng>
__host__ __device__ inline TwoBodyMomenta<real_t> ftf_sample_4momentum(
    const LundTables<real_t>* t, real_t mass, real_t anti_mass, real_t initial_mass,
    Rng& rng) {
  TwoBodyMomenta<real_t> out;
  const real_t d = initial_mass * initial_mass - mass * mass - anti_mass * anti_mass;
  const real_t two_m = real_t(2.) * mass * anti_mass;
  const real_t r_val = d * d - two_m * two_m;
  const real_t pabs =
      (r_val > real_t(0.)) ? sqrt(r_val) / (real_t(2.) * initial_mass) : real_t(0);

  real_t sigma_qt = t->sigma_qt;
  const real_t sum_over_m = (mass + anti_mass) / initial_mass;
  const real_t factor = real_t(1.0) - real_t(0.55) * sum_over_m * sum_over_m;
  if (mass > real_t(930.) || anti_mass > real_t(930.)) { sigma_qt *= factor; }
  if (mass < real_t(930.) && anti_mass < real_t(930.)) {
    // q-qbar string: Geant4's `{}`. No change.
  }
  if ((mass < real_t(930.) && anti_mass > real_t(930.)) ||
      (mass > real_t(930.) && anti_mass < real_t(930.))) {
    // q-di_q string: Geant4's `{}` with `SigmaQT = -1.` (isotropic decay) commented out.
  }
  if (mass > real_t(930.) && anti_mass > real_t(930.)) {
    sigma_qt *= factor;  // the SECOND application - see the file header
  }

  const int max_loops = 1000;
  int loop = 0;
  real_t mass_mt = real_t(0), anti_mass_mt = real_t(0), px = real_t(0), py = real_t(0);
  do {
    ftf_sample_quark_pt_sigma(sigma_qt, pabs, rng, &px, &py);
    const real_t pt2 = px * px + py * py;
    mass_mt = sqrt(mass * mass + pt2);
    anti_mass_mt = sqrt(anti_mass * anti_mass + pt2);
  } while ((initial_mass < mass_mt + anti_mass_mt) && ++loop < max_loops);

  // Geant4's `if (loopCounter >= maxNumberOfLoops) { AvailablePz2 = 0.0; }` goes here and is
  // overwritten by the next statement. Not transcribed, for the same reason as
  // StopFragmenting's repeated test: it has no behaviour to reproduce.
  const real_t dm =
      initial_mass * initial_mass - mass_mt * mass_mt - anti_mass_mt * anti_mass_mt;
  const real_t prod = mass_mt * anti_mass_mt;
  real_t available_pz2 = dm * dm - real_t(4.) * prod * prod;
  available_pz2 /= (real_t(4.) * initial_mass * initial_mass);
  const real_t available_pz = sqrt(available_pz2);

  out.px = px;
  out.py = py;
  out.pz = available_pz;
  out.e = sqrt(mass_mt * mass_mt + available_pz2);
  out.apx = -px;
  out.apy = -py;
  out.apz = -available_pz;
  out.ae = sqrt(anti_mass_mt * anti_mass_mt + available_pz2);
  return out;
}

/// G4VLongitudinalStringDecay::CreatePartonPair.
///
/// `NeedParticle` is +1 for a particle and -1 for an antiparticle. The diquark arm's PDG code
/// is `(max*1000 + min*100 + spin) * NeedParticle`, and the pair returned is
/// `(FindParticle(-code), FindParticle(code))` - the FIRST of the pair is the ANTI of what was
/// asked for, which is what the comment "first in pair is anti to IsParticle" means and what
/// QuarkSplitup then hands to the hadron builder.
struct PartonPair {
  int first = 0;   ///< goes into the hadron
  int second = 0;  ///< becomes the new string end
  FtfRefusal refused = FtfRefusal::kNone;
};

template <typename real_t, typename Rng>
__host__ __device__ inline PartonPair ftf_create_parton_pair(const LundTables<real_t>* t,
                                                             int need_particle,
                                                             bool allow_diquarks, Rng& rng) {
  PartonPair out;
  if (allow_diquarks && static_cast<real_t>(rng.uniform()) < t->diquark_suppress) {
    const int q1 = ftf_sample_quark_flavor(t, rng);
    const int q2 = ftf_sample_quark_flavor(t, rng);
    const int spin = (q1 != q2 && static_cast<real_t>(rng.uniform()) <= real_t(0.5)) ? 1 : 3;
    const int hi = (q1 > q2) ? q1 : q2;
    const int lo = (q1 < q2) ? q1 : q2;
    const int code = (hi * 1000 + lo * 100 + spin) * need_particle;
    out.first = ftf_existing_code(-code);
    out.second = ftf_existing_code(code);
    return out;
  }
  const int code = ftf_sample_quark_flavor(t, rng) * need_particle;
  out.first = ftf_existing_code(code);
  out.second = ftf_existing_code(-code);
  return out;
}

}  // namespace g4gpu::hadronic::ftf
