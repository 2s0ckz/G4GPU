// Evaporation of n, p, d, t, He3 and alpha: the inverse cross sections, the emission width,
// the integrator that turns it into a probability, and the sampler.
//
// Transcribed from G4KalbachCrossSection, G4ChatterjeeCrossSection, G4VEmissionProbability,
// G4EvaporationProbability, G4EvaporationChannel and the six G4*EvaporationProbability
// subclasses (11.1.1).
//
// The chain, and where each piece lives:
//
//   G4EvaporationChannel::GetEmissionProbability   channel_emission_probability()
//     checks the channel is physically open, computes the Coulomb barrier and the kinetic
//     limits, then calls
//   G4EvaporationProbability::TotalProbability     -> integrate_probability()
//     which with OPTxs = 3 (the 11.1.1 default) is a numerical integral of
//   G4EvaporationProbability::ComputeProbability   compute_probability()
//     = pcoeff * exp(2(sqrt(a1 E1) - sqrt(a0 E0))) * K * sigma_inv(K)
//   G4VEmissionProbability::SampleEnergy           sample_energy()
//     rejection-samples that same expression under a two-region majorant built during the
//     integration.
//
// **OPTxs = 3 means Kalbach, and it also means the OPTxs == 0 branch never runs.** That
// branch is the closed-form Weisskopf width with the Dostrovsky alpha and beta parameters, and
// it is the only consumer of CalcAlphaParam / CalcBetaParam and of
// G4CoulombBarrier::BarrierPenetrationFactor. Those six per-ejectile parameterisations are
// transcribed here for completeness and are marked unreachable; nothing in the default
// configuration evaluates them, and no oracle column can therefore check them.
//
// The other thing worth knowing before reading integrate_probability(): its step size adapts
// during the sweep, and the two side effects it leaves behind - `probmax` and the (fE1, fE2,
// fP2) triple that define the majorant - are consumed by sample_energy(). So the integral is
// not a pure function of its arguments as far as the sampler is concerned; the sampler may
// only be called on the state a matching integration left. That coupling is Geant4's, and it
// is why EvaporationState below is one object rather than free functions.
#ifndef G4GPU_DEEX_EVAPORATION_CUH
#define G4GPU_DEEX_EVAPORATION_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "data/level_data.cuh"
#include "physics/hadronic/deexcitation/corrections.cuh"
#include "physics/hadronic/deexcitation/coulomb_barrier.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"

namespace g4gpu::deex {

namespace u = g4gpu::units;

// ---------------------------------------------------------------------------------------------
// Inverse cross sections. Both return a bare number in millibarn - the unit enters one level
// up, in pcoeff. Dividing these by CLHEP::millibarn is the mistake the oracle dump made first.
// ---------------------------------------------------------------------------------------------

/// G4KalbachCrossSection::paramK. Rows: 0 n, 1 p, 2 d, 3 t, 4 He3, 5 alpha. Columns:
/// p0, p1, p2, lambda0, lambda1, mu0, mu1, nu0, nu1, nu2, ra.
///
/// From subroutine sigpar of PRECO-2000 by Constance Kalbach Walker - the empirical optical
/// model reaction cross sections of Narasimha Murthy, Chatterjee and Gupta going over to the
/// geometrical limit at high energy.
__host__ __device__ inline const double* kalbach_params() {
  static const double p[6][11] = {
    {-312.,  0.,     0.,     12.10,   -11.27, 234.1, 38.26, 1.55,  -106.1, 1280.8, 0.0},
    {15.72,  9.65,  -300.,   0.00437, -16.58, 244.7, 0.503, 273.1, -182.4, -1.872, 0.0},
    {0.798,  420.3, -1651.,  0.00619, -7.54,  583.5, 0.337, 421.8, -474.5, -3.592, 0.8},
    {-21.45, 484.7, -1608.,  0.0186,  -8.9,   686.3, 0.325, 368.9, -522.2, -4.998, 0.8},
    {-2.88,  205.6, -1487.,  0.00459, -8.93,  611.2, 0.35,  473.8, -468.2, -2.225, 0.8},
    {10.95, -85.2,   1146.,  0.0643,  -13.96, 781.2, 0.29, -304.7, -470.0, -8.580, 1.2},
  };
  return &p[0][0];
}

/// G4ChatterjeeCrossSection::paramC. Same shape, different fits - and note that p2 for the
/// proton is -449 here and -300 in Kalbach's table, which G4KalbachCrossSection's comment
/// calls out explicitly ("p2 reduced and global red'n factor introduced below Bc"). The two
/// tables are not two copies of one thing.
__host__ __device__ inline const double* chatterjee_params() {
  static const double p[6][11] = {
    {0.,     0.,     0.,      18.57,   -22.93, 381.7, 24.31, 0.172, -15.39, 804.8, 0.0},
    {15.72,  9.65,  -449.,    0.00437, -16.58, 244.7, 0.503, 273.1, -182.4, -1.872, 0.0},
    {-38.21, 922.6, -2804.,  -0.0323,  -5.48,  336.1, 0.48,  524.3, -371.8, -5.924, 1.2},
    {-11.04, 619.1, -2147.,   0.0426,  -10.33, 601.9, 0.37,  583.0, -546.2,  1.718, 1.2},
    {-3.06,  278.5, -1389.,  -0.00535, -11.16, 555.5, 0.4,   687.4, -476.3,  0.509, 1.2},
    {10.95, -85.2,   1146.,   0.0643,  -13.96, 781.2, 0.29, -304.7, -470.0, -8.58,  1.2},
  };
  return &p[0][0];
}

/// G4KalbachCrossSection::ComputePowerParameter - `powZ(resA, mu1)`, the only place mu1 is
/// used as an exponent rather than a coefficient.
__host__ __device__ inline double kalbach_power_parameter(int resA, int idx) {
  return g4pow_pow_z(resA, kalbach_params()[idx * 11 + 6]);
}

/// G4KalbachCrossSection::ComputeCrossSection.
///
/// `Z` here is the EJECTILE's charge and `A` its mass number, `resA` the residual's; `cb` is
/// the energy the threshold branch pivots on, and G4EvaporationProbability passes 0.6 times
/// the Coulomb barrier rather than the barrier itself. The `signor` factors are Kalbach's own
/// per-target-mass renormalisations and they are asymmetric: the neutron is scaled down below
/// A = 40 and UP above A = 210, the proton down below A = 100 only.
__host__ __device__ inline double kalbach_cross_section(double K, double cb, double resA13,
                                                        double amu1, int idx, int Z, int A,
                                                        int resA) {
  const double* p = kalbach_params() + idx * 11;
  double sig = 0.0;
  double signor = 1.0;
  double lambda, mu, nu;
  double ec = 0.5;
  if (Z > 0) { ec = cb; }
  const double ecsq = ec * ec;
  const double elab = K * (A + resA) / static_cast<double>(resA);

  if (idx == 0) {
    if (resA < 40) { signor = 0.7 + resA * 0.0075; }
    else if (resA > 210) { signor = 1.0 + (resA - 210) * 0.004; }
    lambda = p[3] / resA13 + p[4];
    mu = (p[5] + p[6] * resA13) * resA13;
    // The absolute value is JMQ's fix of 20.11.2008: without it the neutron cross section goes
    // through zero at very low energy for A above about 60.
    nu = std::fabs((p[7] * resA + p[8] * resA13) * resA13 + p[9]);
  } else {
    if (idx == 1) {
      if (resA <= 60) { signor = 0.92; }
      else if (resA < 100) { signor = 0.8 + resA * 0.002; }
    }
    lambda = p[3] * resA + p[4];
    mu = p[5] * amu1;
    nu = amu1 * (p[7] + p[8] * ec + p[9] * ecsq);
  }

  if (elab < ec) {
    // Threshold branch: a parabola through the high-energy form's value and slope at ec,
    // cut off at the root below which it would be negative.
    double pp = p[0];
    if (Z > 0) { pp += p[1] / ec + p[2] / ecsq; }
    const double aa = -2 * pp * ec + lambda - nu / ecsq;
    const double bb = pp * ecsq + mu + 2 * nu / ec;
    double ecut;
    const double det = aa * aa - 4 * pp * bb;
    if (det > 0.0) { ecut = (std::sqrt(det) - aa) / (2 * pp); }
    else { ecut = -aa / (2 * pp); }

    if (idx == 0) {
      sig = (lambda * ec + mu + nu / ec) * signor * std::sqrt(elab / ec);
    } else if (elab >= ecut) {
      sig = (pp * elab * elab + aa * elab + bb) * signor;
      if (idx == 1) {
        // The extra proton correction: a Fermi function in (ec - elab), scaled down where ec
        // itself is small, i.e. for light targets.
        const double cc = (3.15 < ec * 0.5) ? 3.15 : ec * 0.5;
        const double signor2 = (ec - elab - cc) * 3.15 / (0.7 * cc);
        sig /= (1.0 + std::exp(signor2));
      }
    }
  } else {
    double etest = 32.0;
    double xnulam = 1.0;
    const double flow = 1.e-18;
    const double spill = 1.e+18;
    if (Z > 0) {
      etest = 0.0;
      xnulam = nu / lambda;
      if (xnulam > spill) { xnulam = spill; }
      if (xnulam >= flow) {
        if (idx == 1) { etest = std::sqrt(xnulam) + 7.0; }
        else { etest = 1.2 * std::sqrt(xnulam); }
      }
    }
    sig = (lambda * elab + mu + nu / elab) * signor;
    if (xnulam >= flow && elab >= etest) {
      double geom = std::sqrt(static_cast<double>(A) * K);
      geom = 1.23 * resA13 + p[10] + 4.573 / geom;
      geom = 31.416 * geom * geom;
      if (geom > sig) { sig = geom; }
    }
  }
  return (sig > 0.0) ? sig : 0.0;
}

/// G4ChatterjeeCrossSection::ComputeCrossSection. Reached only when OPTxs <= 2, which the
/// default configuration never selects (fDeexType = 3). Ported and checked against the oracle
/// anyway - it is a deterministic function of its arguments, so it costs nothing to prove
/// right, and OPTxs is a one-line change away.
__host__ __device__ inline double chatterjee_cross_section(double K, double cb, double resA13,
                                                           double amu1, int idx, int Z,
                                                           int resA) {
  const double* p = chatterjee_params() + idx * 11;
  const double emax = 50.0 * u::MeV<double>();
  const double Kc = (K < emax) ? K : emax;
  double sig;
  if (Z == 0) {
    const double lambda = p[3] / resA13 + p[4];
    const double mu = (p[5] + p[6] * resA13) * resA13;
    const double nu = std::fabs((p[7] * resA + p[8] * resA13) * resA13 + p[9]);
    sig = lambda * Kc + mu + nu / Kc;
  } else {
    const double ec = cb;
    const double ecsq = ec * ec;
    const double pp = p[0] + p[1] / ec + p[2] / ecsq;
    const double lambda = p[3] * resA + p[4];
    const double mu = p[5] * amu1;
    const double nu = amu1 * (p[7] + p[8] * ec + p[9] * ecsq);
    const double q = lambda - nu / ecsq - 2 * pp * ec;
    const double r = mu + 2 * nu / ec + pp * ecsq;
    const double ji = (Kc > ec) ? Kc : ec;
    if (Kc < ec) { sig = pp * Kc * Kc + q * Kc + r; }
    else { sig = pp * (Kc - ji) * (Kc - ji) + lambda * Kc + mu + nu * (2 - Kc / ji) / ji; }
  }
  return (sig > 0.0) ? sig : 0.0;
}

// ---------------------------------------------------------------------------------------------
// The six ejectiles the default channel set gives a G4EvaporationChannel.
// ---------------------------------------------------------------------------------------------

/// In G4EvaporationDefaultGEMFactory's order: n, p, d, t, He3, alpha. `gamma` is the spin
/// factor (2J+1)-ish coefficient each G4*EvaporationProbability passes to its base.
struct Ejectile {
  int a, z;
  double gamma;
  int index;   ///< the paramK/paramC row: 0 for a neutron, theA for Z == 1, theA + 1 above
};

__host__ __device__ inline const Ejectile* evaporation_ejectiles() {
  // index is not a table order: G4EvaporationProbability computes it as
  //   Z == 0 -> 0 ;  Z == 1 -> A ;  else -> A + 1
  // which happens to enumerate n, p, d, t, He3, alpha as 0..5. Written out rather than
  // recomputed so the coincidence is visible.
  static const Ejectile e[6] = {
    {1, 0, 2.0, 0},   // G4NeutronEvaporationProbability(1, 0, 2.0)
    {1, 1, 2.0, 1},   // G4ProtonEvaporationProbability(1, 1, 2.0)
    {2, 1, 3.0, 2},   // G4DeuteronEvaporationProbability(2, 1, 3.0)
    {3, 1, 2.0, 3},   // G4TritonEvaporationProbability(3, 1, 2.0)
    {3, 2, 2.0, 4},   // G4He3EvaporationProbability(3, 2, 2.0)
    {4, 2, 1.0, 5},   // G4AlphaEvaporationProbability(4, 2, 1.0)
  };
  return e;
}

/// The closed-form Weisskopf emission width - the OPTxs == 0 branch of
/// G4EvaporationProbability::TotalProbability, with the Dostrovsky alpha and beta parameters
/// of the six G4*EvaporationProbability subclasses (Phys. Rev. 116 (1959) 683) and
/// G4CoulombBarrier::BarrierPenetrationFactor under it.
///
/// **REFUSED, by name.** It is the only consumer of CalcAlphaParam, CalcBetaParam and the
/// penetration factor, and fDeexType is 3 in 11.1.1, so nothing in QBBC's configuration
/// evaluates it. It is also unobservable: Geant4 exposes no public path that reaches it with
/// OPTxs at its default, so there is no oracle column to check a transcription against and
/// writing one would be transcription without validation. If a caller ever sets
/// deex_type = 0, channel_emission_probability() reports this rather than returning the
/// Kalbach answer under the wrong name.
__host__ __device__ inline const char* refused_weisskopf_width() {
  return "G4EvaporationProbability::TotalProbability OPTxs==0 (Weisskopf width with "
         "Dostrovsky alpha/beta and BarrierPenetrationFactor)";
}

// ---------------------------------------------------------------------------------------------
// G4VEmissionProbability / G4EvaporationProbability, as one state object.
// ---------------------------------------------------------------------------------------------

/// Everything TotalProbability leaves behind for ComputeProbability and SampleEnergy. The
/// members are named after Geant4's so a transcription can be read against the source.
struct EvaporationState {
  // set by the channel
  int ej = 0;               ///< index into evaporation_ejectiles()
  int res_z = 0, res_a = 0;
  double p_mass = 0.0;      ///< the decaying fragment's total mass (M + E*)
  double p_evap_mass = 0.0; ///< the ejectile's ground-state mass
  double p_res_mass = 0.0;  ///< the residual's ground-state mass
  double b_coulomb = 0.0;

  // set by TotalProbability
  double a0 = 0.0;          ///< level density of the PARENT at its own excitation
  double free_u = 0.0;      ///< E* minus the parent's pairing correction
  double delta1 = 0.0;      ///< the RESIDUAL's pairing correction
  double res_a13 = 0.0;
  double muu = 0.0;         ///< Kalbach's power parameter for this residual
  int last_a = 0;

  // set by IntegrateProbability, consumed by SampleEnergy
  double probability = 0.0;
  double emin = 0.0, emax = 0.0, e_coulomb = 0.0;
  double probmax = 0.0;
  double fE1 = 0.0, fE2 = 0.0, fP2 = 0.0;
  double elimit = 0.0;      ///< the integrator's starting step, per ejectile
  double accuracy = 0.0;

  // set by SampleEnergy through FindRecoilExcitation
  double exc_res = 0.0;
  double exc = 0.0;         ///< the ejectile's own excitation; always 0 for these six
};

/// G4EvaporationProbability's constructor: pcoeff, and the integrator reset that differs
/// between the neutron and everything else.
__host__ __device__ inline double evaporation_pcoeff(int ej) {
  const Ejectile& e = evaporation_ejectiles()[ej];
  const double m = deex::nuclear_mass(e.a, e.z);
  const double x = u::pi<double>() * u::hbarc<double>();
  return e.gamma * m * millibarn() / (x * x);
}

/// G4VEmissionProbability::ResetIntegrator as the constructor calls it: (30, 0.25 MeV, 0.02)
/// for the neutron and (30, 0.5 MeV, 0.03) for a charged ejectile. The bin count is ignored -
/// ResetIntegrator's first argument is unused in 11.1.1.
__host__ __device__ inline void reset_integrator(EvaporationState& s) {
  if (evaporation_ejectiles()[s.ej].z == 0) {
    s.elimit = 0.25 * u::MeV<double>();
    s.accuracy = 0.02;
  } else {
    s.elimit = 0.5 * u::MeV<double>();
    s.accuracy = 0.03;
  }
}

/// G4EvaporationProbability::CrossSection with OPTxs = 3: Kalbach, pivoted at 0.6 of the
/// Coulomb barrier, times the barrier-penetration factor (1 - elim/K).
///
/// The `resA != lastA` cache is Geant4's and is reproduced because it is observable: `muu` is
/// recomputed only when the residual's mass number changes, so a channel evaluated for two
/// residuals of the same A but different Z reuses the first one's power parameter. For these
/// six ejectiles A determines the row and Z does not enter powZ, so the reuse is correct - but
/// it is correct by accident and the cache is what makes it so.
__host__ __device__ inline double evaporation_cross_section(EvaporationState& s, double K,
                                                            double CB) {
  const Ejectile& e = evaporation_ejectiles()[s.ej];
  if (s.res_a != s.last_a) {
    s.last_a = s.res_a;
    if (e.index > 0) { s.muu = kalbach_power_parameter(s.res_a, e.index); }
  }
  const int optxs = deex_params().deex_type;
  if (optxs <= 2) {
    return chatterjee_cross_section(K, CB, s.res_a13, s.muu, e.index, e.z, s.res_a);
  }
  const double elim = 0.6 * CB;
  if (K <= elim) { return 0.0; }
  double res = kalbach_cross_section(K, elim, s.res_a13, s.muu, e.index, e.z, e.a, s.res_a);
  res *= (1.0 - elim / K);
  return res;
}

/// G4EvaporationProbability::ComputeProbability - the integrand.
///
/// The residual's excitation is computed relativistically from the invariant mass rather than
/// as E* - K - Q, which is why pMass, pEvapMass and pResMass all appear: `mres` is the
/// invariant mass of the residual after a two-body decay of `pMass` into the ejectile with
/// kinetic energy K.
__host__ __device__ inline double compute_probability(const EvaporationState& s, double K,
                                                      double CB, double pcoeff,
                                                      const data::LevelTable& lt) {
  const double E0 = s.free_u;
  if (s.p_mass < s.p_evap_mass + s.p_res_mass) { return 0.0; }

  const double m02 = s.p_mass * s.p_mass;
  const double m12 = s.p_evap_mass * s.p_evap_mass;
  const double mres = std::sqrt(m02 + m12 - 2.0 * s.p_mass * (s.p_evap_mass + K));

  const double exc_res = mres - s.p_res_mass;
  const double E1 = exc_res - s.delta1;
  if (E1 <= 0.0) { return 0.0; }
  const bool has_levels = (data::find_manager(lt, s.res_z, s.res_a) >= 0);
  const double a1 = deex::level_density(s.res_z, s.res_a, exc_res, has_levels);
  EvaporationState tmp = s;   // CrossSection mutates the muu cache; the integrand does not
  const double xs = evaporation_cross_section(tmp, K, CB);
  return pcoeff * std::exp(2.0 * (std::sqrt(a1 * E1) - std::sqrt(s.a0 * E0))) *
         K * xs;
}

/// G4VEmissionProbability::IntegrateProbability.
///
/// A trapezoid sweep from emin to emax whose step adapts: shrunk by 0.7 when one trapezoid
/// carries more than 80% of the running total, grown by 1.5 when it carries less than 10%, and
/// clamped to [0.2, 2] MeV. It stops early when a trapezoid contributes less than `accuracy`
/// of the total, so the "nbin = 5 * ibin" cap is a guard rather than the loop's length.
///
/// Its three side effects are the majorant the sampler needs: `probmax` is the largest
/// integrand value seen, `fE1` the first energy at which the integrand has fallen below half
/// of probmax, and (fE2, fP2) a second point above fE1 used to fit the exponential tail.
__host__ __device__ inline double integrate_probability(EvaporationState& s, double elow,
                                                        double ehigh, double cb, double pcoeff,
                                                        const data::LevelTable& lt) {
  s.probability = 0.0;
  if (elow >= ehigh) { return s.probability; }

  s.emin = elow;
  s.emax = ehigh;
  s.e_coulomb = cb;

  const double edeltamin = 0.2 * u::MeV<double>();
  const double edeltamax = 2.0 * u::MeV<double>();
  double edelta = s.elimit;
  if (edelta > edeltamax) { edelta = edeltamax; }
  if (edelta < edeltamin) { edelta = edeltamin; }
  const double xbin = (s.emax - s.emin) / edelta + 1.0;
  int ibin = static_cast<int>(xbin);
  if (ibin < 4) { ibin = 4; }

  const int nbin = ibin * 5;
  edelta = (s.emax - s.emin) / ibin;

  double x = s.emin;
  double y = 0.0;
  const double edelmicro = edelta * 0.02;
  s.probmax = compute_probability(s, x + edelmicro, s.e_coulomb, pcoeff, lt);
  double problast = s.probmax;
  s.fE1 = s.fE2 = s.fP2 = 0.0;
  const double emax0 = s.emax - edelmicro;
  bool endpoint = false;
  for (int i = 0; i < nbin; ++i) {
    x += edelta;
    if (x >= emax0) {
      x = emax0;
      endpoint = true;
    }
    y = compute_probability(s, x, s.e_coulomb, pcoeff, lt);
    if (y >= s.probmax) {
      s.probmax = y;
    } else if (0.0 == s.fE1 && 2 * y < s.probmax) {
      s.fE1 = x;
    }
    const double del = (y + problast) * edelta * 0.5;
    s.probability += del;
    if (del < s.accuracy * s.probability || endpoint) { break; }
    problast = y;
    if (del != s.probability && del > 0.8 * s.probability && 0.7 * edelta > edeltamin) {
      edelta *= 0.7;
    } else if (del < 0.1 * s.probability && 1.5 * edelta < edeltamax) {
      edelta *= 1.5;
    }
  }
  if (s.fE1 > s.emin && s.fE1 < s.emax) {
    s.fE2 = 0.5 * (s.fE1 + s.emax);
    const double alt = s.emax - edelta;
    if (alt > s.fE2) { s.fE2 = alt; }
    s.fP2 = 2 * compute_probability(s, s.fE2, s.e_coulomb, pcoeff, lt);
  }
  return s.probability;
}

/// G4VEmissionProbability::FindRecoilExcitation - the residual's excitation after the ejectile
/// took kinetic energy `e`, snapped to a discrete level when one is close enough.
///
/// Three outcomes, and the return value is a kinetic energy in all three:
///   * below the 10 eV tolerance the residual is put in its ground state and the kinetic
///     energy is recomputed from the two-body kinematics, so energy is conserved exactly
///     rather than approximately;
///   * with fFD false, or no level data, or an excitation above the highest known level, `e`
///     is returned unchanged;
///   * otherwise, if the nearest level is within tolerance AND the decay is kinematically
///     allowed to it, the residual is put on that level and the kinetic energy recomputed.
///
/// The `pMass > mass + pResMass + elevel` test is what stops the snap from inventing energy.
__host__ __device__ inline double find_recoil_excitation(EvaporationState& s, double e,
                                                         const data::LevelTable& lt) {
  const double mass = s.p_evap_mass + s.exc;
  const double m02 = s.p_mass * s.p_mass;
  const double m12 = mass * mass;
  const double m22 = s.p_res_mass * s.p_res_mass;
  const double mres = std::sqrt(m02 + m12 - 2.0 * s.p_mass * (mass + e));

  s.exc_res = mres - s.p_res_mass;
  const double tolerance = deex_params().min_excitation;

  if (s.exc_res < tolerance) {
    s.exc_res = 0.0;
    const double v = 0.5 * (m02 + m12 - m22) / s.p_mass - mass;
    return (v > 0.0) ? v : 0.0;
  }
  if (!deex_params().discrete_excitation_flag) { return e; }

  const int m = data::find_manager(lt, s.res_z, s.res_a);
  if (m < 0) { return e; }
  if (s.exc_res > data::read_max_level_energy(lt, m) + tolerance) { return e; }

  const double elevel = data::level_energy(lt, m, data::nearest_level_index(lt, m, s.exc_res));
  if (s.p_mass > mass + s.p_res_mass + elevel && std::fabs(elevel - s.exc_res) <= tolerance) {
    const double massR = s.p_res_mass + elevel;
    const double mr2 = massR * massR;
    s.exc_res = elevel;
    const double v = 0.5 * (m02 + m12 - mr2) / s.p_mass - mass;
    return (v > 0.0) ? v : 0.0;
  }
  return e;
}

/// G4VEmissionProbability::SampleEnergy - rejection sampling of the integrand under a
/// two-region majorant: flat at 1.05 * probmax up to fE1, then an exponential fitted through
/// (fE1, probmax) and (fE2, fP2).
///
/// The 1000-iteration cap is Geant4's, and so is what happens at it: the loop exits with
/// whatever `ekin` it last drew, accepted or not. That is a real bias and it is kept, because
/// removing it would change the answer in exactly the cases the majorant is worst - which are
/// the cases a comparison would notice.
template <typename Rng>
__host__ __device__ inline double sample_energy(EvaporationState& s, double pcoeff,
                                                const data::LevelTable& lt, Rng& rng) {
  const double fact = 1.05;
  const double alim = 0.05;
  const double blim = 20.0;
  s.probmax *= fact;

  double del = s.emax - s.emin;
  double p1 = 1.0;
  double p2 = 0.0;
  double a0 = 0.0;
  double a1 = 1.0;
  double x;
  if (s.fE1 > 0.0 && s.fP2 > 0.0 && s.fP2 < 0.5 * s.probmax) {
    a0 = std::log(s.probmax / s.fP2) / (s.fE2 - s.fE1);
    del = s.fE1 - s.emin;
    p1 = del;
    x = a0 * (s.emax - s.fE1);
    if (x < blim) {
      a1 = (x > alim) ? 1.0 - std::exp(-x) : x * (1.0 - 0.5 * x);
    }
    p2 = a1 / a0;
    p1 /= (p1 + p2);
    p2 = 1.0 - p1;
  }

  const int nmax = 1000;
  double ekin = 0.0, g = 0.0, gmax = 0.0;
  int n = 0;
  do {
    ++n;
    const double q = rng.uniform();
    if (q <= p1) {
      gmax = s.probmax;
      ekin = del * q / p1 + s.emin;
    } else {
      ekin = s.fE1 - std::log(1.0 - (q - p1) * a1 / p2) / a0;
      x = a0 * (ekin - s.fE1);
      gmax = s.fP2;
      if (x < blim) {
        gmax = s.probmax * ((x > alim) ? std::exp(-x)
                                       : 1.0 - x * (1.0 - 0.5 * x));
      }
    }
    g = compute_probability(s, ekin, s.e_coulomb, pcoeff, lt);
  } while (gmax * rng.uniform() > g && n < nmax);
  return find_recoil_excitation(s, ekin, lt);
}

// ---------------------------------------------------------------------------------------------
// G4EvaporationChannel
// ---------------------------------------------------------------------------------------------

/// G4EvaporationChannel::GetEmissionProbability. Returns 0 for a closed channel and leaves
/// `s` untouched in that case - which is why the caller must not sample from a channel whose
/// probability came back zero. G4Evaporation obeys that by construction; this port's handler
/// asserts it.
__host__ __device__ inline double channel_emission_probability(EvaporationState& s,
                                                               const Fragment& frag,
                                                               const data::LevelTable& lt) {
  const Ejectile& e = evaporation_ejectiles()[s.ej];
  s.probability = 0.0;
  const int fragA = frag.a;
  const int fragZ = frag.z;
  s.res_a = fragA - e.a;
  s.res_z = fragZ - e.z;

  // "Only channels which are physically allowed": the residual must be at least as heavy as
  // the ejectile, must have a non-negative and physical charge, and - the last clause - must
  // not be a nucleus of only protons or only neutrons unless it is a single nucleon.
  if (s.res_a < e.a || s.res_a < s.res_z || s.res_z < 0 ||
      (s.res_a == e.a && s.res_z < e.z) ||
      ((s.res_a > 1) && (s.res_a == s.res_z || s.res_z == 0))) {
    return 0.0;
  }

  const double ex_energy = frag.excitation;
  const double delta0 = deex::level_data_pairing_correction(fragZ, fragA);
  if (ex_energy < delta0) { return 0.0; }

  const double frag_mass = frag.ground_state_mass;
  s.p_mass = frag_mass + ex_energy;
  s.p_evap_mass = deex::nuclear_mass(e.a, e.z);
  s.p_res_mass = deex::nuclear_mass(s.res_a, s.res_z);
  const double evap_mass2 = s.p_evap_mass * s.p_evap_mass;
  const double ekinmax =
      0.5 * ((s.p_mass - s.p_res_mass) * (s.p_mass + s.p_res_mass) + evap_mass2) / s.p_mass -
      s.p_evap_mass;

  double elim = 0.0;
  s.b_coulomb = 0.0;
  if (e.z > 0) {
    s.b_coulomb = deex::coulomb_barrier(e.a, e.z, s.res_a, s.res_z, 0.0);
    // With OPTxs != 0 the barrier is only 60% enforced, because the Kalbach cross section
    // already carries a penetration factor below it.
    elim = (deex_params().deex_type != 0) ? s.b_coulomb * 0.6 : s.b_coulomb;
  }
  if (s.p_mass <= s.p_res_mass + s.p_evap_mass + elim) { return 0.0; }

  double ekinmin = 0.0;
  if (elim > 0.0) {
    const double resM = s.p_mass - s.p_evap_mass - elim;
    const double v =
        0.5 * ((s.p_mass - resM) * (s.p_mass + resM) + evap_mass2) / s.p_mass - s.p_evap_mass;
    ekinmin = (v > 0.0) ? v : 0.0;
  }
  if (ekinmax <= ekinmin) { return 0.0; }

  // G4EvaporationProbability::TotalProbability, OPTxs != 0 branch.
  const bool parent_levels = (data::find_manager(lt, fragZ, fragA) >= 0);
  s.a0 = deex::level_density(fragZ, fragA, ex_energy, parent_levels);
  s.free_u = ex_energy - delta0;
  s.delta1 = deex::level_data_pairing_correction(s.res_z, s.res_a);
  s.res_a13 = data::g4pow_z13<double>(s.res_a);
  s.exc = 0.0;
  reset_integrator(s);
  const double pcoeff = evaporation_pcoeff(s.ej);
  return integrate_probability(s, ekinmin, ekinmax, s.b_coulomb, pcoeff, lt);
}

/// G4EvaporationChannel::EmittedFragment. `frag` is modified in place into the residual, and
/// the ejectile is returned.
///
/// `resA > 4` is the condition for sampling at all: for a residual of four nucleons or fewer
/// the kinetic energy is taken at its maximum instead, because the level structure there is
/// not what the continuum integrand describes.
template <typename Rng>
__host__ __device__ inline Fragment channel_emitted_fragment(EvaporationState& s,
                                                             Fragment& frag,
                                                             const data::LevelTable& lt,
                                                             Rng& rng) {
  const Ejectile& e = evaporation_ejectiles()[s.ej];
  const double evap_mass = s.p_evap_mass;
  const double evap_mass2 = evap_mass * evap_mass;
  double ekin = 0.5 * ((s.p_mass - s.p_res_mass) * (s.p_mass + s.p_res_mass) + evap_mass2) /
                    s.p_mass - evap_mass;
  if (s.res_a > 4 && s.probability > 0.0) {
    const double pcoeff = evaporation_pcoeff(s.ej);
    ekin = sample_energy(s, pcoeff, lt, rng);
  }
  if (ekin < 0.0) { ekin = 0.0; }

  LorentzVector lv0 = frag.momentum;
  const Vec3d dir = random_direction(rng);
  const double p = std::sqrt(ekin * (ekin + 2.0 * evap_mass));
  LorentzVector lv(dir * p, ekin + evap_mass);
  lv.boost(lv0.boost_vector());

  Fragment ev = make_fragment(e.a, e.z, lv);
  lv0 -= lv;
  frag.set_za_and_momentum(lv0, s.res_z, s.res_a);
  return ev;
}

}  // namespace g4gpu::deex

#endif
