// G4AntiNuclElastic - the elastic final state of the anti-nucleon and the light anti-nuclei
// above 100 MeV, in QBBC's hadElastic for the six species of G4HadParticles::GetLightAntiIons().
//
// Transcribed from source/processes/hadronic/models/coherent_elastic/src/G4AntiNuclElastic.cc
// (11.1.1): SampleInvariantT, CalculateParticleBeta, CalculateZommerfeld, CalculateAm,
// DampFactor, BesselJzero, BesselJone, BesselOneByArg and GetcosTeta1. SampleThetaCMS and
// SampleThetaLab are not called by anything in a physics list - `G4HadronElastic::ApplyYourself`
// calls only the virtual `SampleInvariantT` - and are not ported; nothing reaches them.
//
// THE MODEL IS A SAMPLER INSIDE G4HadronElastic::ApplyYourself. It overrides SampleInvariantT and
// nothing else, so the kinematics around it (the boost, phi, the recoil, the resample with the
// BASE class's SampleInvariantT when t falls outside [0, pLocalTmax]) are P5's
// `hadron_elastic_apply_yourself`; this file is the sampler handed to it.
//
// WHAT IT DRAWS, in order - and the draw count is part of the answer:
//
//   Plab < |B| * 100 MeV   one uniform, t = fTmax * u. Never reached from QBBC's wiring for a
//                          kinetic energy above 100 MeV (an anti-proton at 100 MeV has
//                          Plab = 444.6 MeV), but `G4EnergyRangeManager` hands this model the
//                          100-100.1 MeV overlap and an anti-alpha's |B| is 4; transcribed.
//   otherwise              one uniform against the Coulomb probability; then either the Coulomb
//                          arm (two more: phi, then Ksi - phi is drawn and only enters the
//                          transverse momentum, so t does not depend on it but the stream does) or
//                          the strong arm's rejection loop, two uniforms per trial, at most 10000
//                          trials (`loopCounter`; past it the sampler returns t = 0).
//
// The cross sections it reads are the AntiAGlauber component's (`xs/anti_nucl_nuclear_xs.cuh`) at
// the ISOTOPE's integer A (`(G4double)A`), not at the NIST mean mass the process's data set uses -
// two different A's for the same scatter, as in Geant4.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "data/nuclei_mass_ame12.hh"
#include "physics/hadronic/xs/anti_nucl_nuclear_xs.cuh"
#include "physics/hadronic/xs/projectile.cuh"

namespace g4gpu::physics::hadronic::elastic {

namespace axs = g4gpu::hadronic::xs;

/// The six G4AntiNuclElastic projectiles by PDG code, as the xs component reads them. A code that
/// is not one of the six gives pdg 0.
template <typename real_t>
__host__ __device__ inline axs::Projectile<real_t> anuc_projectile(int pdg) {
  switch (pdg) {
    case -2212: return axs::anti_proton<real_t>();
    case -2112: return axs::anti_neutron<real_t>();
    case -1000010020: return axs::anti_deuteron<real_t>();
    case -1000010030: return axs::anti_triton<real_t>();
    case -1000020030: return axs::anti_he3<real_t>();
    case -1000020040: return axs::anti_alpha<real_t>();
    default: return axs::Projectile<real_t>{};
  }
}

/// G4AntiNuclElastic::DampFactor.
template <typename real_t>
__host__ __device__ inline real_t anuc_damp_factor(real_t x) {
  const real_t f3 = real_t(6.);
  if (fabs(x) < real_t(0.01)) { return real_t(1.) / (real_t(1.) + x * x / f3); }
  return x / sinh(x);
}

/// G4AntiNuclElastic::BesselJzero - the Numerical Recipes rational approximation, as written.
template <typename real_t>
__host__ __device__ inline real_t anuc_bessel_j0(real_t value) {
  const real_t modvalue = fabs(value);
  if (value < real_t(8.0) && value > real_t(-8.0)) {
    const real_t value2 = value * value;
    const real_t fact1 =
        real_t(57568490574.0) +
        value2 * (real_t(-13362590354.0) +
                  value2 * (real_t(651619640.7) +
                            value2 * (real_t(-11214424.18) +
                                      value2 * (real_t(77392.33017) +
                                                value2 * (real_t(-184.9052456))))));
    const real_t fact2 =
        real_t(57568490411.0) +
        value2 * (real_t(1029532985.0) +
                  value2 * (real_t(9494680.718) +
                            value2 * (real_t(59272.64853) +
                                      value2 * (real_t(267.8532712) + value2 * real_t(1.0)))));
    return fact1 / fact2;
  }
  const real_t arg = real_t(8.0) / modvalue;
  const real_t value2 = arg * arg;
  const real_t shift = modvalue - real_t(0.785398164);
  const real_t fact1 =
      real_t(1.0) +
      value2 * (real_t(-0.1098628627e-2) +
                value2 * (real_t(0.2734510407e-4) +
                          value2 * (real_t(-0.2073370639e-5) + value2 * real_t(0.2093887211e-6))));
  const real_t fact2 =
      real_t(-0.1562499995e-1) +
      value2 * (real_t(0.1430488765e-3) +
                value2 * (real_t(-0.6911147651e-5) +
                          value2 * (real_t(0.7621095161e-6) - value2 * real_t(0.934945152e-7))));
  return sqrt(real_t(0.636619772) / modvalue) * (cos(shift) * fact1 - arg * sin(shift) * fact2);
}

/// G4AntiNuclElastic::BesselJone.
template <typename real_t>
__host__ __device__ inline real_t anuc_bessel_j1(real_t value) {
  const real_t modvalue = fabs(value);
  if (modvalue < real_t(8.0)) {
    const real_t value2 = value * value;
    const real_t fact1 =
        value * (real_t(72362614232.0) +
                 value2 * (real_t(-7895059235.0) +
                           value2 * (real_t(242396853.1) +
                                     value2 * (real_t(-2972611.439) +
                                               value2 * (real_t(15704.48260) +
                                                         value2 * (real_t(-30.16036606)))))));
    const real_t fact2 =
        real_t(144725228442.0) +
        value2 * (real_t(2300535178.0) +
                  value2 * (real_t(18583304.74) +
                            value2 * (real_t(99447.43394) +
                                      value2 * (real_t(376.9991397) + value2 * real_t(1.0)))));
    return fact1 / fact2;
  }
  const real_t arg = real_t(8.0) / modvalue;
  const real_t value2 = arg * arg;
  const real_t shift = modvalue - real_t(2.356194491);
  const real_t fact1 =
      real_t(1.0) +
      value2 * (real_t(0.183105e-2) +
                value2 * (real_t(-0.3516396496e-4) +
                          value2 * (real_t(0.2457520174e-5) + value2 * (real_t(-0.240337019e-6)))));
  const real_t fact2 =
      real_t(0.04687499995) +
      value2 * (real_t(-0.2002690873e-3) +
                value2 * (real_t(0.8449199096e-5) +
                          value2 * (real_t(-0.88228987e-6) + value2 * real_t(0.105787412e-6))));
  real_t bessel =
      sqrt(real_t(0.636619772) / modvalue) * (cos(shift) * fact1 - arg * sin(shift) * fact2);
  if (value < real_t(0.0)) { bessel = -bessel; }
  return bessel;
}

/// G4AntiNuclElastic::BesselOneByArg - J1(x)/x with its small-argument series.
template <typename real_t>
__host__ __device__ inline real_t anuc_bessel_one_by_arg(real_t x) {
  if (fabs(x) < real_t(0.01)) {
    x *= real_t(0.5);
    const real_t x2 = x * x;
    return (real_t(2.) - x2 + x2 * x2 / real_t(6.)) / real_t(4.);
  }
  return anuc_bessel_j1<real_t>(x) / x;
}

/// G4AntiNuclElastic::GetcosTeta1. `p0 = 1.*hbarc/fermi` with CLHEP's fermi = 1e-15 m.
template <typename real_t>
__host__ __device__ inline real_t anuc_cos_teta1(real_t plab, int A) {
  const real_t fermi = real_t(1.e-15) * units::m<real_t>();
  const real_t p0 = real_t(1.) * units::hbarc<real_t>() / fermi;
  real_t cteta1 = real_t(1.0) - p0 * p0 / real_t(2.0) * axs::g4pow_z23<real_t>(A)
                                    / (plab * plab);
  if (cteta1 < real_t(-1.)) { cteta1 = real_t(-1.0); }
  return cteta1;
}

/// G4AntiNuclElastic::CalculateAm. `G4Pow::A13(Z)` takes Z as a DOUBLE here (the parameter is
/// `G4double Z`), so it is the interpolated A13 and not the integer table Z13.
template <typename real_t>
__host__ __device__ inline real_t anuc_am(real_t momentum, real_t n, real_t Z) {
  const real_t k = momentum / units::hbarc<real_t>();
  const real_t ch = real_t(1.13) + real_t(3.76) * n * n;
  const real_t zn = real_t(1.77) * k / data::g4pow_a13<real_t>(Z) * units::bohr_radius<real_t>();
  const real_t zn2 = zn * zn;
  return ch / zn2;
}

/// Everything SampleInvariantT computes before its first uniform, for a test to compare the
/// deterministic pieces one by one (the members fptot, fTmax, fBeta, fZommerfeld, fAm, fRa,
/// fRef, fceff, and the locals ctet1, S, the two hadronic cross sections, CoulombProb, rho,
/// sig_pbarp, Qmax).
template <typename real_t>
struct AnucState {
  bool low_momentum = false;  ///< Plab < |B|*100 MeV: t = fTmax*u and nothing else
  real_t ctet1 = 0, S = 0, ptot = 0, tmax = 0, beta = 0, n = 0, am = 0;
  real_t xs_coulomb = 0, xs_el_mb = 0, xs_tot_mb = 0, coulomb_prob = 0;
  real_t qmax = 0, sig_pbarp = 0, ra = 0, ref = 0, ceff = 0, rho = 0;
  // The CMS projectile four-vector after the boost, which the Coulomb arm's t is built from.
  real_t pz_cms = 0, e_cms = 0, mproj = 0;
};

/// The deterministic head of G4AntiNuclElastic::SampleInvariantT, up to the Coulomb draw.
template <typename real_t>
__host__ __device__ inline AnucState<real_t> anuc_prepare(int pdg, real_t Plab, int Z, int A) {
  AnucState<real_t> s;
  const axs::Projectile<real_t> pj = anuc_projectile<real_t>(pdg);
  const real_t Mproj = pj.mass;
  s.mproj = Mproj;
  // G4LorentzVector Pproj(0.,0.,Plab,sqrt(Plab*Plab+Mproj*Mproj));
  const real_t pe = sqrt(Plab * Plab + Mproj * Mproj);
  s.ctet1 = anuc_cos_teta1<real_t>(Plab, A);
  const real_t energy = pe - Mproj;
  const real_t TargMass = data::nuclear_mass<real_t>(A, Z);
  // lv = (0,0,0,TargMass) + Pproj; S = lv.mag2()/GeV^2; bst = lv.boostVector() = p/E.
  const real_t lv_e = TargMass + pe;
  const real_t lv_pz = Plab;
  const real_t gev = units::GeV<real_t>();
  s.S = (lv_e * lv_e - lv_pz * lv_pz) / (gev * gev);
  // Pproj.boost(-bst): CLHEP HepLorentzVector::boost(bx,by,bz) with b = -lv_pz/lv_e on z.
  {
    const real_t bz = -(lv_pz / lv_e);
    const real_t b2 = bz * bz;
    const real_t ggamma = real_t(1.0) / sqrt(real_t(1.0) - b2);
    const real_t bp = bz * Plab;
    const real_t gamma2 = (b2 > real_t(0)) ? (ggamma - real_t(1.0)) / b2 : real_t(0.0);
    s.pz_cms = Plab + gamma2 * bp * bz + ggamma * bz * pe;
    s.e_cms = ggamma * (pe + bp);
  }
  s.ptot = fabs(s.pz_cms);
  s.tmax = real_t(4.0) * s.ptot * s.ptot;
  const int ab = (pj.baryon_number < 0) ? -pj.baryon_number : pj.baryon_number;
  if (Plab < static_cast<real_t>(ab * 100) * units::MeV<real_t>()) {
    s.low_momentum = true;
    return s;
  }
  const real_t PlabPerN = Plab / static_cast<real_t>(ab);
  const real_t NucleonMass =
      real_t(0.5) * (units::proton_mass_c2<real_t>() + units::neutron_mass_c2<real_t>());
  const real_t PrNucleonMass = (ab == 1) ? Mproj : NucleonMass;
  real_t energyPerN = sqrt(PlabPerN * PlabPerN + PrNucleonMass * PrNucleonMass);
  energyPerN -= PrNucleonMass;
  const real_t Z1 = pj.charge;
  const real_t Z2 = static_cast<real_t>(Z);
  {
    const real_t a = s.ptot / Mproj;
    s.beta = a / sqrt(real_t(1) + a * a);
  }
  s.n = units::fine_structure_const<real_t>() * Z1 * Z2 / s.beta;
  s.am = anuc_am<real_t>(s.ptot, s.n, Z2);
  const real_t fWaveVector = s.ptot;
  const real_t mevToBarn = real_t(0.38938e+6);
  const real_t pi = units::pi<real_t>();
  s.xs_coulomb = mevToBarn * (s.n / fWaveVector) * (s.n / fWaveVector) * pi * (real_t(1) + s.ctet1)
                 / (real_t(1.) + s.am) / (real_t(1.) + real_t(2.) * s.am - s.ctet1);
  const real_t mb = axs::anu_millibarn<real_t>();
  s.xs_el_mb = axs::anu_elastic_element<real_t>(pj, energy, Z, static_cast<real_t>(A)).value / mb;
  s.xs_tot_mb = axs::anu_total_element<real_t>(pj, energy, Z, static_cast<real_t>(A)).value / mb;
  s.coulomb_prob = s.xs_coulomb / (s.xs_coulomb + s.xs_el_mb);

  // The strong arm's parameters. Computed here even though only that arm reads them, because
  // none of it draws and a test wants them for every point.
  s.qmax = real_t(2.) * s.ptot / real_t(197.33);
  s.sig_pbarp = axs::anu_nucleon_total<real_t>(axs::anti_proton<real_t>(), energyPerN).total_mb;
  const real_t z13 = data::g4pow_z13<real_t>(A);
  real_t fRa = real_t(1.113) * z13 - real_t(0.227) / z13;
  if (A == 3) { fRa = real_t(1.81); }
  if (A == 4) { fRa = real_t(1.37); }
  if ((A >= 12.) && (A < 27)) { fRa = fRa * real_t(0.85); }
  if ((A >= 27.) && (A < 48)) { fRa = fRa * real_t(0.90); }
  if ((A >= 48.) && (A < 65)) { fRa = fRa * real_t(0.95); }
  s.ra = fRa;
  real_t Ref2 = s.xs_tot_mb / real_t(10.) / real_t(2.) / pi;
  real_t ceff2 = real_t(0.0);
  real_t rho = real_t(0.0);
  const real_t sg = s.sig_pbarp;
  // theTargetDef, by (Z, A): proton, deuteron, triton, He3, alpha, or none.
  const bool tP = (Z == 1 && A == 1), tD = (Z == 1 && A == 2), tT = (Z == 1 && A == 3);
  const bool tH3 = (Z == 2 && A == 3), tA = (Z == 2 && A == 4);
  const bool pN = (pdg == -2212 || pdg == -2112);
  if (pN) {
    if (tP) {
      if (Plab < real_t(610.)) {
        rho = real_t(1.3347) - real_t(10.342) * Plab / real_t(1000.)
              + real_t(22.277) * Plab / real_t(1000.) * Plab / real_t(1000.)
              - real_t(13.634) * Plab / real_t(1000.) * Plab / real_t(1000.) * Plab / real_t(1000.);
      }
      if ((Plab < real_t(5500.)) && (Plab >= real_t(610.))) { rho = real_t(0.22); }
      if ((Plab >= real_t(5500.)) && (Plab < real_t(12300.))) { rho = real_t(-0.32); }
      if (Plab >= real_t(12300.)) { rho = real_t(0.135) - real_t(2.26) / (sqrt(s.S)); }
      Ref2 = real_t(0.35) + real_t(0.9) / sqrt(sqrt(s.S - real_t(4.) * real_t(0.88)))
             + real_t(0.04) * log(s.S);
      ceff2 = real_t(0.375) - real_t(2.) / s.S
              + real_t(0.44) / ((s.S - real_t(4.)) * (s.S - real_t(4.)) + real_t(1.5));
      Ref2 = Ref2 * Ref2;
      ceff2 = ceff2 * ceff2;
    }
    if ((Z == 1) && (A == 2)) {
      Ref2 = fRa * fRa - real_t(0.28) + real_t(0.019) * sg + real_t(2.06e-6) * sg * sg;
      ceff2 = real_t(0.297) + real_t(7.853e-04) * sg + real_t(0.2899) * exp(real_t(-0.03) * sg);
    }
    if ((Z == 1) && (A == 3)) {
      Ref2 = fRa * fRa - real_t(1.36) + real_t(0.025) * sg - real_t(3.69e-7) * sg * sg;
      ceff2 = real_t(0.149) + real_t(7.091e-04) * sg + real_t(0.3743) * exp(real_t(-0.03) * sg);
    }
    if ((Z == 2) && (A == 3)) {
      Ref2 = fRa * fRa - real_t(1.36) + real_t(0.025) * sg - real_t(3.69e-7) * sg * sg;
      ceff2 = real_t(0.149) + real_t(7.091e-04) * sg + real_t(0.3743) * exp(real_t(-0.03) * sg);
    }
    if ((Z == 2) && (A == 4)) {
      Ref2 = fRa * fRa - real_t(0.46) + real_t(0.03) * sg - real_t(2.98e-6) * sg * sg;
      ceff2 = real_t(0.078) + real_t(6.657e-4) * sg + real_t(0.3359) * exp(real_t(-0.03) * sg);
    }
    if (Z > 2) {
      Ref2 = fRa * fRa + real_t(2.48) * real_t(0.01) * sg * fRa
             - real_t(2.23e-6) * sg * sg * fRa * fRa;
      ceff2 = real_t(0.16) + real_t(3.3e-4) * sg + real_t(0.35) * exp(real_t(-0.03) * sg);
    }
  }
  if (pdg == -1000010020) {
    if (tP) { ceff2 = real_t(0.297) + real_t(7.853e-04) * sg + real_t(0.2899) * exp(real_t(-0.03) * sg); }
    if (tD) { ceff2 = real_t(0.65) + real_t(3.0e-4) * sg + real_t(0.55) * exp(real_t(-0.03) * sg); }
    if (tT || tH3) { ceff2 = real_t(0.57) + real_t(2.5e-4) * sg + real_t(0.65) * exp(real_t(-0.02) * sg); }
    if (tA) { ceff2 = real_t(0.40) + real_t(3.5e-4) * sg + real_t(0.45) * exp(real_t(-0.02) * sg); }
    if (Z > 2) { ceff2 = real_t(0.38) + real_t(2.0e-4) * sg + real_t(0.5) * exp(real_t(-0.03) * sg); }
  }
  if (pdg == -1000020030 || pdg == -1000010030) {
    if (tP) { ceff2 = real_t(0.149) + real_t(7.091e-04) * sg + real_t(0.3743) * exp(real_t(-0.03) * sg); }
    if (tD) { ceff2 = real_t(0.57) + real_t(2.5e-4) * sg + real_t(0.65) * exp(real_t(-0.02) * sg); }
    if (tT || tH3) { ceff2 = real_t(0.39) + real_t(2.7e-4) * sg + real_t(0.7) * exp(real_t(-0.02) * sg); }
    if (tA) { ceff2 = real_t(0.24) + real_t(3.5e-4) * sg + real_t(0.75) * exp(real_t(-0.03) * sg); }
    if (Z > 2) { ceff2 = real_t(0.26) + real_t(2.2e-4) * sg + real_t(0.33) * exp(real_t(-0.03) * sg); }
  }
  // `(theParticle == theAAlpha) || (ceff2 == 0.0)`: the anti-alpha's row, and the fall-back for
  // any projectile whose own block left ceff2 at zero (an anti-nucleon on Z = 2 with A other
  // than 3 or 4, say) - a fall-back by VALUE, transcribed as such.
  if ((pdg == -1000020040) || (ceff2 == real_t(0.0))) {
    if (tP) { ceff2 = real_t(0.078) + real_t(6.657e-4) * sg + real_t(0.3359) * exp(real_t(-0.03) * sg); }
    if (tD) { ceff2 = real_t(0.40) + real_t(3.5e-4) * sg + real_t(0.45) * exp(real_t(-0.02) * sg); }
    if (tT || tH3) { ceff2 = real_t(0.24) + real_t(3.5e-4) * sg + real_t(0.75) * exp(real_t(-0.03) * sg); }
    if (tA) { ceff2 = real_t(0.17) + real_t(3.5e-4) * sg + real_t(0.45) * exp(real_t(-0.03) * sg); }
    if (Z > 2) { ceff2 = real_t(0.22) + real_t(2.0e-4) * sg + real_t(0.2) * exp(real_t(-0.03) * sg); }
  }
  s.ref = sqrt(Ref2);
  s.ceff = sqrt(ceff2);
  s.rho = rho;
  return s;
}

/// The strong arm's acceptance function at momentum transfer Q (fm^-1) - BracFunct, including
/// its final `*Q`. A test samples this at a grid of Q against Geant4's own functions.
template <typename real_t>
__host__ __device__ inline real_t anuc_brac_funct(const AnucState<real_t>& s, real_t Q) {
  const real_t Amag = real_t(1.0);
  const real_t SlopeMag = real_t(0.5);
  const real_t x = s.ref * Q;
  const real_t b1 = anuc_bessel_one_by_arg<real_t>(x);
  const real_t b0 = s.rho / real_t(2.) * anuc_bessel_j0<real_t>(x);
  const real_t d = anuc_damp_factor<real_t>(units::pi<real_t>() * s.ceff * Q);
  real_t BracFunct = ((b1 * b1 + b0 * b0) * (d * d)) / (Amag * exp(-SlopeMag * Q));
  return BracFunct * Q;
}

/// G4AntiNuclElastic::SampleInvariantT, MeV^2. `draws`, when given, counts the uniforms.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t anuc_sample_invariant_t(int pdg, real_t Plab, int Z, int A,
                                                          Rng& rng, int* loops = nullptr) {
  const AnucState<real_t> s = anuc_prepare<real_t>(pdg, Plab, Z, A);
  if (s.low_momentum) { return s.tmax * rng.uniform(); }
  if (rng.uniform() < s.coulomb_prob) {
    // Simulation of Coulomb scattering.
    const real_t phi = units::twopi<real_t>() * rng.uniform();
    const real_t Ksi = rng.uniform();
    const real_t par1 = real_t(2.) * (real_t(1.) + s.am) / (real_t(1.) + s.ctet1);
    const real_t cosThetaCMS =
        (par1 * s.ctet1 - Ksi * (real_t(1.) + real_t(2.) * s.am)) / (par1 - Ksi);
    const real_t PtZ = s.ptot * cosThetaCMS;
    const real_t PtProjCMS = s.ptot * sqrt(real_t(1.0) - cosThetaCMS * cosThetaCMS);
    const real_t PtX = PtProjCMS * cos(phi);
    const real_t PtY = PtProjCMS * sin(phi);
    const real_t Fe = sqrt(PtX * PtX + PtY * PtY + PtZ * PtZ + s.mproj * s.mproj);
    // T = -(Pproj - Fproj).mag2(), with Pproj the CMS projectile (0, 0, pz_cms, e_cms).
    const real_t dx = -PtX, dy = -PtY, dz = s.pz_cms - PtZ, de = s.e_cms - Fe;
    return -(de * de - (dx * dx + dy * dy + dz * dz));
  }
  // Strong interaction scattering.
  const real_t SlopeMag = real_t(0.5);
  const int maxNumberOfLoops = 10000;
  int loopCounter = 0;
  real_t Q = real_t(0.0);
  real_t BracFunct;
  do {
    Q = -log(real_t(1.) - (real_t(1.) - exp(-SlopeMag * s.qmax)) * rng.uniform()) / SlopeMag;
    BracFunct = anuc_brac_funct<real_t>(s, Q);
  } while ((rng.uniform() > BracFunct) && ++loopCounter < maxNumberOfLoops);
  if (loops != nullptr) { *loops = loopCounter; }
  if (loopCounter >= maxNumberOfLoops) { return real_t(0.0); }
  real_t T = Q * Q;
  T *= real_t(3.893913e+4);  // fm^(-2) -> MeV^2
  return T;
}

}  // namespace g4gpu::physics::hadronic::elastic
