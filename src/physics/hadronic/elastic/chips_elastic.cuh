// G4ChipsElasticModel - the elastic final state QBBC gives protons and neutrons - and the two
// CHIPS cross-section classes that carry its t-distribution machinery.
//
// Transcribed from Geant4 11.1.1:
//   processes/hadronic/models/coherent_elastic/src/G4ChipsElasticModel.cc
//     G4ChipsElasticModel::SampleInvariantT
//   processes/hadronic/cross_sections/src/G4ChipsProtonElasticXS.cc
//     GetChipsCrossSection, CalculateCrossSection, GetPTables, GetTabValues, GetQ2max,
//     GetExchangeT, GetSlope
//   processes/hadronic/cross_sections/src/G4ChipsNeutronElasticXS.cc
//     the same seven, plus the 417-row per-isotope low-energy table extracted into
//     chips_neutron_lowe.hh by tools/extract_chips_neutron.pl
//
// ---------------------------------------------------------------------------------------------
// The CHIPS classes are stateful, and the state is a pure memoisation - except in one place.
//
// G4ChipsProtonElasticXS keeps an "Associative Memory DB": per (Z, N) it allocates ten
// 128-element tables over log(p) from -8 to 8, fills them lazily from the lowest bin up to
// whatever momentum has been asked for, and interpolates linearly in log(p). Nothing in the fill
// depends on the order of the calls - `GetTabValues(lp)` is a pure function of (lp, Z, N) once
// `lastPAR` is set - so a STATELESS port that computes the two bracketing bins on demand gives
// bit-identical answers, and that is what this file does. Two exceptions, both recorded because
// they are the only places where Geant4's answer depends on history:
//
//   1. `lastTH`, the threshold. GetChipsCrossSection does
//        `if(lastCS<=0. && pEn>lastTH) lastTH=pEn;`
//      and a LATER call with `pEn <= lastTH` returns 0 without computing anything. So once the
//      parameterisation has gone non-positive at some momentum, every lower momentum for that
//      isotope returns zero for the rest of the run. For pA the returned expression is a sum of
//      positive terms and cannot go non-positive; for pp it can, because
//      `(par1 + par2*dl1^2 + par4/p)/(1 + 0.425*lp)/(...)` has a pole and a sign change at
//      lp = -1/0.425, i.e. p = 95 MeV/c. `chips_cross_section` below returns the computed value
//      and reports `non_positive`, leaving the latch to the caller: a device port has no
//      per-isotope run history, and silently omitting the latch would be a different physics
//      near 100 MeV/c on hydrogen. See docs/RISK.md.
//   2. the `lastLP == lastPIN` branch of CalculateCrossSection, which reads a single bin instead
//      of interpolating. `lastPIN` is `lPMin + fin*dlnP` with `fin = int((LP-lPMin)/dlnP)+1`, so
//      it is strictly greater than LP unless it was clamped to the last bin - which happens only
//      at exactly `lp == lPMax == 8` (p = e^8 GeV = 2981 GeV/c). At that single point Geant4
//      prints `G4QEleastCS::CCS:b=127,127` and uses bin 127. Reproduced below.
//
// ---------------------------------------------------------------------------------------------
// What the model does with all of it.
//
// G4ChipsElasticModel::SampleInvariantT calls GetChipsCrossSection first, purely to make the
// class compute its parameters and its (-t)max, and then GetExchangeT to sample. If the cross
// section comes back <= 0 it falls back to G4HadronElastic::SampleInvariantT. It also remaps two
// light targets before asking: (Z=1, N=2) becomes N=1 and (Z=2, N=1) becomes N=2 - so tritium is
// treated as deuterium and He3 as He4 in the CHIPS tables. That is a deliberate remap in
// G4ChipsElasticModel.cc, not a rounding, and it happens before the isotope search.
#pragma once

#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/elastic/chips_neutron_lowe.hh"
#include "physics/hadronic/elastic/hadron_elastic.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::elastic {

/// The tabulation grid both CHIPS classes use: 128 points in log(p/GeV) from -8 to 8.
struct ChipsGrid {
  static constexpr int kNPoints = 128;
  static constexpr int kNLast = kNPoints - 1;
  static constexpr double kLPMin = -8.0;
  static constexpr double kLPMax = 8.0;
  static constexpr double kDlnP = (kLPMax - kLPMin) / double(kNLast);
};

/// The nine differential-cross-section parameters CHIPS interpolates alongside the total, and
/// the total itself. Geant4 holds them in ten parallel 128-element tables (CST, SST, S1T, B1T,
/// S2T, B2T, S3T, B3T, S4T, B4T) and copies the interpolated values into theSS, theS1, ... .
template <typename real_t>
struct ChipsTabValues {
  real_t cs = real_t(0);
  real_t ss = real_t(0);
  real_t s1 = real_t(0), b1 = real_t(0);
  real_t s2 = real_t(0), b2 = real_t(0);
  real_t s3 = real_t(0), b3 = real_t(0);
  real_t s4 = real_t(0), b4 = real_t(0);
};

/// `lastPAR` for the proton: 52 parameters, indices 0..51.
///
/// G4ChipsProtonElasticXS::GetPTables writes 0..8 for the total cross section, then 9..44 (A<6.5)
/// or 9..50 (A>=6.5) for the differential parameterisation, then 51 for the low-energy neutron
/// term. Note that the A<6.5 branch leaves 45..50 untouched, and GetTabValues' A<6.5 branch does
/// not read them - the two branches are separate parameterisations sharing one array, so an
/// index read in the wrong branch is a stale number and not a compile error. They are
/// zero-initialised here for that reason.
template <typename real_t>
struct ChipsProtonPars {
  real_t p[52] = {};
  bool is_pp = false;  ///< target is a free proton (Z=1, N=0): a different parameterisation
};

/// G4ChipsProtonElasticXS::GetPTables, the parameter-setting half.
///
/// The two hadron-hadron rows come straight from the source's `np_el[24]` and `pp_el[32]`
/// literals; the nuclear rows are the `peh_fit.f` / `pel_*` fits, piecewise at A = 6.5.
///
/// `tgZ == 0 && tgN == 1` (a free neutron target) takes the np row, and `tgZ == 1 && tgN == 0`
/// (a free proton) the pp row - so for the PROTON class the pp row is the like-particle one.
/// Anything else is the nuclear branch, including (Z=1, N=1) deuterium.
template <typename real_t>
__host__ __device__ ChipsProtonPars<real_t> chips_proton_pars(int tg_z, int tg_n) {
  ChipsProtonPars<real_t> out;
  // np_el[24] and pp_el[32], verbatim.
  const real_t np_el[24] = {real_t(12.),  real_t(.05),  real_t(.0001), real_t(5.),
                            real_t(.35),  real_t(6.75), real_t(.14),   real_t(19.),
                            real_t(.6),   real_t(6.75), real_t(.14),   real_t(13.),
                            real_t(.14),  real_t(.6),   real_t(.00013), real_t(75.),
                            real_t(.001), real_t(7.2),  real_t(4.32),  real_t(.012),
                            real_t(2.5),  real_t(0.0),  real_t(12.),   real_t(.34)};
  const real_t pp_el[32] = {
      real_t(2.865), real_t(18.9),  real_t(.6461), real_t(3.),    real_t(9.),    real_t(.425),
      real_t(.4276), real_t(.0022), real_t(5.),    real_t(74.),   real_t(3.),    real_t(3.4),
      real_t(.2),    real_t(.17),   real_t(.001),  real_t(8.),    real_t(.055),  real_t(3.64),
      real_t(5.e-5), real_t(4000.), real_t(1500.), real_t(.46),   real_t(1.2e6), real_t(3.5e6),
      real_t(5.e-5), real_t(1.e10), real_t(8.5e8), real_t(1.e10), real_t(1.1),   real_t(3.4e6),
      real_t(6.8e6), real_t(0.)};

  if (tg_z == 0 && tg_n == 1) {
    for (int i = 0; i < 24; ++i) { out.p[i] = np_el[i]; }
    return out;
  }
  if (tg_z == 1 && tg_n == 0) {
    for (int i = 0; i < 32; ++i) { out.p[i] = pp_el[i]; }
    out.is_pp = true;
    return out;
  }

  const real_t a = real_t(tg_z + tg_n);
  const real_t sa = sqrt(a), ssa = sqrt(sa), asa = a * sa;
  const real_t a2 = a * a, a3 = a2 * a, a4 = a3 * a, a5 = a4 * a, a6 = a4 * a2;
  const real_t a7 = a6 * a, a8 = a7 * a, a9 = a8 * a, a10 = a5 * a5;
  const real_t a12 = a6 * a6, a14 = a7 * a7, a16 = a8 * a8, a17 = a16 * a;
  const real_t a20 = a16 * a4, a32 = a16 * a16;

  real_t* P = out.p;
  P[0] = real_t(5.) / (real_t(1.) + real_t(22.) / asa);
  P[1] = real_t(4.8) * pow(a, real_t(1.14)) / (real_t(1.) + real_t(3.6) / a3);
  P[2] = real_t(1.) / (real_t(1.) + real_t(4.E-3) * a4) +
         real_t(2.E-6) * a3 / (real_t(1.) + real_t(1.3E-6) * a3);
  P[3] = real_t(1.3) * a;
  P[4] = real_t(3.E-8) * a3 / (real_t(1.) + real_t(4.E-7) * a4);
  P[5] = real_t(.07) * asa / (real_t(1.) + real_t(.009) * a2);
  P[6] = (real_t(3.) + real_t(3.E-16) * a20) /
         (real_t(1.) + a20 * (real_t(2.E-16) / a + real_t(3.E-19) * a));
  P[7] = (real_t(5.E-9) * a4 * sa + real_t(.27) / a) / (real_t(1.) + real_t(5.E16) / a20) /
             (real_t(1.) + real_t(6.E-9) * a4) +
         real_t(.015) / a2;
  P[8] = (real_t(.001) * a + real_t(.07) / a) /
             (real_t(1.) + real_t(5.E13) / a16 + real_t(5.E-7) * a3) +
         real_t(.0003) / sa;

  if (a < real_t(6.5)) {
    const real_t a28 = a16 * a12;
    P[9] = real_t(4000) * a;
    P[10] = real_t(1.2e7) * a8 + real_t(380) * a17;
    P[11] = real_t(.7) / (real_t(1.) + real_t(4.e-12) * a16);
    P[12] = real_t(2.5) / a8 / (a4 + real_t(1.e-16) * a32);
    P[13] = real_t(.28) * a;
    P[14] = real_t(1.2) * a2 + real_t(2.3);
    P[15] = real_t(3.8) / a;
    P[16] = real_t(.01) / (real_t(1.) + real_t(.0024) * a5);
    P[17] = real_t(.2) * a;
    P[18] = real_t(9.e-7) / (real_t(1.) + real_t(.035) * a5);
    P[19] = (real_t(42.) + real_t(2.7e-11) * a16) / (real_t(1.) + real_t(.14) * a);
    P[20] = real_t(2.25) * a3;
    P[21] = real_t(18.);
    P[22] = real_t(2.4e-3) * a8 / (real_t(1.) + real_t(2.6e-4) * a7);
    P[23] = real_t(3.5e-36) * a32 * a8 / (real_t(1.) + real_t(5.e-15) * a32 / a);
    P[24] = real_t(1.e5) / (a8 + real_t(2.5e12) / a16);
    P[25] = real_t(8.e7) / (a12 + real_t(1.e-27) * a28 * a28);
    P[26] = real_t(.0006) * a3;
    P[27] = real_t(10.) + real_t(4.e-8) * a12 * a;
    P[28] = real_t(.114);
    P[29] = real_t(.003);
    P[30] = real_t(2.e-23);
    P[31] = real_t(1.) / (real_t(1.) + real_t(.0001) * a8);
    P[32] = real_t(1.5e-4) / (real_t(1.) + real_t(5.e-6) * a12);
    P[33] = real_t(.03);
    P[34] = a / real_t(2);
    P[35] = real_t(2.e-7) * a4;
    P[36] = real_t(4.);
    P[37] = real_t(64.) / a3;
    P[38] = real_t(1.e8) * exp(real_t(.32) * asa);
    P[39] = real_t(20.) * exp(real_t(.45) * asa);
    P[40] = real_t(7.e3) + real_t(2.4e6) / a5;
    P[41] = real_t(2.5e5) * exp(real_t(.085) * a3);
    P[42] = real_t(2.5) * a;
    P[43] = real_t(920.) + real_t(.03) * a8 * a3;
    P[44] = real_t(93.) + real_t(.0023) * a12;
  } else {
    const real_t p1a10 = real_t(2.2e-28) * a10;
    const real_t r4a16 = real_t(6.e14) / a16;
    const real_t s4a16 = r4a16 * r4a16;
    P[9] = real_t(4.5) * pow(a, real_t(1.15));
    P[10] = real_t(.06) * pow(a, real_t(.6));
    P[11] = real_t(.6) * a / (real_t(1.) + real_t(2.e15) / a16);
    P[12] = real_t(.17) / (a + real_t(9.e5) / a3 + real_t(1.5e33) / a32);
    P[13] = (real_t(.001) + real_t(7.e-11) * a5) / (real_t(1.) + real_t(4.4e-11) * a5);
    P[14] = (p1a10 * p1a10 + real_t(2.e-29)) / (real_t(1.) + real_t(2.e-22) * a12);
    P[15] = real_t(400.) / a12 + real_t(2.e-22) * a9;
    P[16] = real_t(1.e-32) * a12 / (real_t(1.) + real_t(5.e22) / a14);
    P[17] = real_t(1000.) / a2 + real_t(9.5) * sa * ssa;
    P[18] = real_t(4.e-6) * a * asa + real_t(1.e11) / a16;
    P[19] = (real_t(120.) / a + real_t(.002) * a2) / (real_t(1.) + real_t(2.e14) / a16);
    P[20] = real_t(9.) + real_t(100.) / a;
    P[21] = real_t(.002) * a3 + real_t(3.e7) / a6;
    P[22] = real_t(7.e-15) * a4 * asa;
    P[23] = real_t(9000.) / a4;
    P[24] = real_t(.0011) * asa / (real_t(1.) + real_t(3.e34) / a32 / a4);
    P[25] = real_t(1.e-5) * a2 + real_t(2.e14) / a16;
    P[26] = real_t(1.2e-11) * a2 / (real_t(1.) + real_t(1.5e19) / a12);
    P[27] = real_t(.016) * asa / (real_t(1.) + real_t(5.e16) / a16);
    P[28] = real_t(.002) * a4 / (real_t(1.) + real_t(7.e7) / pow(a - real_t(6.83), real_t(14)));
    P[29] = real_t(2.e6) / a6 + real_t(7.2) / pow(a, real_t(.11));
    P[30] = real_t(11.) * a3 / (real_t(1.) + real_t(7.e23) / a16 / a8);
    P[31] = real_t(100.) / asa;
    P[32] = (real_t(.1) + real_t(4.4e-5) * a2) / (real_t(1.) + real_t(5.e5) / a4);
    P[33] = real_t(3.5e-4) * a2 / (real_t(1.) + real_t(1.e8) / a8);
    P[34] = real_t(1.3) + real_t(3.e5) / a4;
    P[35] = real_t(500.) / (a2 + real_t(50.)) + real_t(3);
    P[36] = real_t(1.e-9) / a + s4a16 * s4a16;
    P[37] = real_t(.4) * asa + real_t(3.e-9) * a6;
    P[38] = real_t(.0005) * a5;
    P[39] = real_t(.002) * a5;
    P[40] = real_t(10.);
    P[41] = real_t(.05) + real_t(.005) * a;
    P[42] = real_t(7.e-8) / sa;
    P[43] = real_t(.8) * sa;
    P[44] = real_t(.02) * sa;
    P[45] = real_t(1.e8) / a3;
    P[46] = real_t(3.e32) / (a32 + real_t(1.e32));
    P[47] = real_t(24.);
    P[48] = real_t(20.) / sa;
    P[49] = real_t(7.e3) * a / (sa + real_t(1.));
    P[50] = real_t(900.) * sa / (real_t(1.) + real_t(500.) / a3);
  }
  P[51] = real_t(1.e15) + real_t(2.e27) / a4 / (real_t(1.) + real_t(2.e-18) * a16);
  return out;
}

/// G4ChipsProtonElasticXS::GetTabValues. `lp` is log(p/GeV).
///
/// Returns the total elastic cross section in mb and fills the nine differential parameters.
/// A negative Z is refused by Geant4 with a warning and a zero return; `is_heavy_element_allowed`
/// is `true` in 11.1.1 (the AR-24Apr2018 switch) so Z > 92 is allowed. A Z of 0 is converted to
/// a proton target, which is the branch that makes `tgZ==1 && tgN==0` reachable from a neutron
/// target - and it is done by MUTATING tgZ/tgN, so it changes which branch runs below too.
template <typename real_t>
__host__ __device__ ChipsTabValues<real_t> chips_proton_tab_values(
    const ChipsProtonPars<real_t>& pars, real_t lp, int tg_z, int tg_n, bool* refused) {
  ChipsTabValues<real_t> v;
  if (refused) { *refused = false; }
  if (tg_z < 0) {
    if (refused) { *refused = true; }
    return v;
  }
  if (tg_z - 1 < 0) { tg_z = 1; tg_n = 0; }

  const real_t* P = pars.p;
  const real_t p = exp(lp), sp = sqrt(p);
  const real_t p2 = p * p, p3 = p2 * p, p4 = p3 * p;

  if (tg_z == 1 && tg_n == 0) {  // pp
    const real_t p2s = p2 * sp;
    const real_t dl2 = lp - P[8];
    v.ss = P[31];
    v.s1 = (P[9] + P[10] * dl2 * dl2) / (real_t(1.) + P[11] / p4 / p) +
           (P[12] / p2 + P[13] * p) / (p4 + P[14] * sp);
    v.b1 = P[15] * pow(p, P[16]) / (real_t(1.) + P[17] / p3);
    v.s2 = P[18] + P[19] / (p4 + P[20] * p);
    v.b2 = P[21] + P[22] / (p4 + P[23] / sp);
    v.s3 = P[24] + P[25] / (p4 * p4 + P[26] * p2 + P[27]);
    v.b3 = P[28] + P[29] / (p4 + P[30]);
    v.s4 = real_t(0.);
    v.b4 = real_t(0.);
    const real_t dl1 = lp - P[3];
    v.cs = P[0] / p2s / (real_t(1.) + P[7] / p2s) +
           (P[1] + P[2] * dl1 * dl1 + P[4] / p) / (real_t(1.) + P[5] * lp) /
               (real_t(1.) + P[6] / p4);
    return v;
  }

  const real_t p5 = p4 * p, p6 = p5 * p, p8 = p6 * p2, p10 = p8 * p2;
  const real_t p12 = p10 * p2, p16 = p8 * p8;
  const real_t dl = lp - real_t(5.);
  const real_t a = real_t(tg_z + tg_n);
  if (a < real_t(6.5)) {
    const real_t pah = pow(p, a / real_t(2));
    const real_t pa = pah * pah;
    const real_t pa2 = pa * pa;
    v.s1 = P[9] / (real_t(1.) + P[10] * p4 * pa) + P[11] / (p4 + P[12] * p4 / pa2) +
           (P[13] * dl * dl + P[14]) / (real_t(1.) + P[15] / p2);
    v.b1 = (P[16] + P[17] * p2) / (p4 + P[18] / pah) + P[19];
    v.ss = P[20] / (real_t(1.) + P[21] / p2) + P[22] / (p6 / pa + P[23] / p16);
    v.s2 = P[24] / (pa / p2 + P[25] / p4) + P[26];
    v.b2 = P[27] * pow(p, P[28]) + P[29] / (p8 + P[30] / p16);
    v.s3 = P[31] / (pa * p + P[32] / pa) + P[33];
    v.b3 = P[34] / (p3 + P[35] / p6) + P[36] / (real_t(1.) + P[37] / p2);
    v.s4 = p2 * (pah * P[38] * exp(-pah * P[39]) +
                 P[40] / (real_t(1.) + P[41] * pow(p, P[42])));
    v.b4 = P[43] * pa / p2 / (real_t(1.) + pa * P[44]);
  } else {
    v.s1 = P[9] / (real_t(1.) + P[10] / p4) + P[11] / (p4 + P[12] / p2) +
           P[13] / (p5 + P[14] / p16);
    v.b1 = (P[15] / p8 + P[19]) / (p + P[16] / pow(p, P[20])) +
           P[17] / (real_t(1.) + P[18] / p4);
    v.ss = P[21] / (p4 / pow(p, P[23]) + P[22] / p4);
    v.s2 = P[24] / p4 / (pow(p, P[25]) + P[26] / p12) + P[27];
    v.b2 = P[28] / pow(p, P[29]) + P[30] / pow(p, P[31]);
    v.s3 = P[32] / pow(p, P[35]) / (real_t(1.) + P[36] / p12) +
           P[33] / (real_t(1.) + P[34] / p6);
    v.b3 = P[37] / p8 + P[38] / p2 + P[39] / (real_t(1.) + P[40] / p8);
    v.s4 = (P[41] / p4 + P[46] / p) / (real_t(1.) + P[42] / p10) +
           (P[43] + P[44] * dl * dl) / (real_t(1.) + P[45] / p12);
    v.b4 = P[47] / (real_t(1.) + P[48] / p) + P[49] * p4 / (real_t(1.) + P[50] * p5);
  }
  v.cs = (P[0] * dl * dl + P[1]) / (real_t(1.) + P[2] / p + P[5] / p6) +
         P[3] / (p3 + P[4] / p3) + P[7] / (p4 + pow(P[8] / p, P[6]));
  return v;
}

/// `lastPAR` for the neutron: 58 parameters, indices 0..57. The layout is NOT the proton's -
/// the neutron's total-cross-section fit (`na_el.f`) uses 0..12 with a seven-entry per-isotope
/// block at 4, 7, 8, 9, 10, 11, 12, and the differential block starts at 15.
template <typename real_t>
struct ChipsNeutronPars {
  real_t p[58] = {};
  bool is_np = false;  ///< target is a free proton (Z=1, N=0)
};

/// Geant4's fallback row when the isotope is not in the CHIPS list.
template <typename real_t>
__host__ __device__ inline void chips_neutron_default_lowe(real_t* P) {
  P[4] = real_t(5.2E-7);
  P[7] = real_t(22.);
  P[8] = real_t(.00026);
  P[9] = real_t(1.3E-9);
  P[10] = real_t(2.7);
  P[11] = real_t(4.E-5);
  P[12] = real_t(.005);
}

/// G4ChipsNeutronElasticXS::GetPTables, the parameter-setting half.
///
/// The isotope search is a LINEAR walk over the per-Z list taking the FIRST entry whose neutron
/// number equals tgN, which is why chips_neutron_lowe.hh preserves the list order and its
/// duplicate slots. A Z above 98 has no list at all - Geant4 would index Pars[tgZ] out of bounds;
/// here it is refused by name through `*refused`, since reading past the array is not a physics
/// choice.
///
/// `lastPAR[1]` is `4.8*exp(ala*1.14)` with `ala = log(a)` - the same value as the proton's
/// `4.8*std::pow(a,1.14)` up to the difference between exp(1.14*log(a)) and pow(a,1.14), which
/// is not zero in double precision. Written as the source writes it.
template <typename real_t>
__host__ __device__ ChipsNeutronPars<real_t> chips_neutron_pars(int tg_z, int tg_n,
                                                               bool* refused) {
  ChipsNeutronPars<real_t> out;
  if (refused) { *refused = false; }
  const real_t np_el[24] = {real_t(12.),  real_t(.05),  real_t(.0001), real_t(5.),
                            real_t(.35),  real_t(6.75), real_t(.14),   real_t(19.),
                            real_t(.6),   real_t(6.75), real_t(.14),   real_t(13.),
                            real_t(.14),  real_t(.6),   real_t(.00013), real_t(75.),
                            real_t(.001), real_t(7.2),  real_t(4.32),  real_t(.012),
                            real_t(2.5),  real_t(0.0),  real_t(12.),   real_t(.34)};
  const real_t pp_el[32] = {
      real_t(2.865), real_t(18.9),  real_t(.6461), real_t(3.),    real_t(9.),    real_t(.425),
      real_t(.4276), real_t(.0022), real_t(5.),    real_t(74.),   real_t(3.),    real_t(3.4),
      real_t(.2),    real_t(.17),   real_t(.001),  real_t(8.),    real_t(.055),  real_t(3.64),
      real_t(5.e-5), real_t(4000.), real_t(1500.), real_t(.46),   real_t(1.2e6), real_t(3.5e6),
      real_t(5.e-5), real_t(1.e10), real_t(8.5e8), real_t(1.e10), real_t(1.1),   real_t(3.4e6),
      real_t(6.8e6), real_t(0.)};

  if (tg_z == 1 && tg_n == 0) {  // np
    for (int i = 0; i < 24; ++i) { out.p[i] = np_el[i]; }
    out.is_np = true;
    return out;
  }
  if (tg_z == 0 && tg_n == 1) {  // nn: the pp row
    for (int i = 0; i < 32; ++i) { out.p[i] = pp_el[i]; }
    return out;
  }
  if (tg_z < 0 || tg_z >= chips_data::kNeutronLowENZ) {
    if (refused) { *refused = true; }
    return out;
  }

  const real_t a = real_t(tg_z + tg_n);
  const real_t ala = log(a);
  const real_t sa = sqrt(a), ssa = sqrt(sa), asa = a * sa;
  const real_t a2 = a * a, a3 = a2 * a, a4 = a3 * a, a5 = a4 * a, a6 = a4 * a2;
  const real_t a7 = a6 * a, a8 = a7 * a, a9 = a8 * a, a10 = a5 * a5;
  const real_t a12 = a6 * a6, a14 = a7 * a7, a16 = a8 * a8, a17 = a16 * a, a32 = a16 * a16;

  real_t* P = out.p;
  P[0] = real_t(5.) / (real_t(1.) + real_t(22.) / asa);
  P[1] = real_t(4.8) * exp(ala * real_t(1.14)) / (real_t(1.) + real_t(3.6) / a3);
  P[2] = real_t(1.) / (real_t(1.) + real_t(.004) * a4) +
         real_t(2.E-6) * a3 / (real_t(1.) + real_t(1.3E-6) * a3);
  P[3] = real_t(.07) * asa / (real_t(1.) + real_t(.009) * a2);
  P[5] = real_t(1.7) * a;
  P[6] = real_t(5.5E-6) * exp(ala * real_t(1.3));
  P[13] = real_t(0.);
  P[14] = real_t(0.);

  const int nn = chips_data::neutron_lowe_niso()[tg_z];
  const chips_data::NeutronLowERow* rows = chips_data::neutron_lowe_pars(tg_z);
  bool nfound = true;
  if (rows == nullptr) {
    if (refused) { *refused = true; }
    return out;
  }
  for (int in = 0; in < nn; ++in) {
    if (rows[in].n == tg_n) {
      P[4] = real_t(rows[in].p[0]);
      P[7] = real_t(rows[in].p[1]);
      P[8] = real_t(rows[in].p[2]);
      P[9] = real_t(rows[in].p[3]);
      P[10] = real_t(rows[in].p[4]);
      P[11] = real_t(rows[in].p[5]);
      P[12] = real_t(rows[in].p[6]);
      nfound = false;
      break;
    }
  }
  if (nfound) { chips_neutron_default_lowe<real_t>(P); }

  if (a < real_t(6.5)) {
    const real_t a28 = a16 * a12;
    P[15] = real_t(4000) * a;
    P[16] = real_t(1.2e7) * a8 + real_t(380) * a17;
    P[17] = real_t(.7) / (real_t(1.) + real_t(4.e-12) * a16);
    P[18] = real_t(2.5) / a8 / (a4 + real_t(1.e-16) * a32);
    P[19] = real_t(.28) * a;
    P[20] = real_t(1.2) * a2 + real_t(2.3);
    P[21] = real_t(3.8) / a;
    P[22] = real_t(.01) / (real_t(1.) + real_t(.0024) * a5);
    P[23] = real_t(.2) * a;
    P[24] = real_t(9.e-7) / (real_t(1.) + real_t(.035) * a5);
    P[25] = (real_t(42.) + real_t(2.7e-11) * a16) / (real_t(1.) + real_t(.14) * a);
    P[26] = real_t(2.25) * a3;
    P[27] = real_t(18.);
    P[28] = real_t(.0024) * a8 / (real_t(1.) + real_t(2.6e-4) * a7);
    P[29] = real_t(3.5e-36) * a32 * a8 / (real_t(1.) + real_t(5.e-15) * a32 / a);
    P[30] = real_t(1.e5) / (a8 + real_t(2.5e12) / a16);
    P[31] = real_t(8.e7) / (a12 + real_t(1.e-27) * a28 * a28);
    P[32] = real_t(.0006) * a3;
    P[33] = real_t(10.) + real_t(4.e-8) * a12 * a;
    P[34] = real_t(.114);
    P[35] = real_t(.003);
    P[36] = real_t(2.e-23);
    P[37] = real_t(1.) / (real_t(1.) + real_t(.0001) * a8);
    P[38] = real_t(1.5e-4) / (real_t(1.) + real_t(5.e-6) * a12);
    P[39] = real_t(.03);
    P[40] = a / real_t(2);
    P[41] = real_t(2.e-7) * a4;
    P[42] = real_t(4.);
    P[43] = real_t(64.) / a3;
    P[44] = real_t(1.e8) * exp(real_t(.32) * asa);
    P[45] = real_t(20.) * exp(real_t(.45) * asa);
    P[46] = real_t(7.e3) + real_t(2.4e6) / a5;
    P[47] = real_t(2.5e5) * exp(real_t(.085) * a3);
    P[48] = real_t(2.5) * a;
    P[49] = real_t(920.) + real_t(.03) * a8 * a3;
    P[50] = real_t(93.) + real_t(.0023) * a12;
  } else {
    const real_t p1a10 = real_t(2.2e-28) * a10;
    const real_t r4a16 = real_t(6.e14) / a16;
    const real_t s4a16 = r4a16 * r4a16;
    P[15] = real_t(4.5) * pow(a, real_t(1.15));
    P[16] = real_t(.06) * pow(a, real_t(.6));
    P[17] = real_t(.6) * a / (real_t(1.) + real_t(2.e15) / a16);
    P[18] = real_t(.17) / (a + real_t(9.e5) / a3 + real_t(1.5e33) / a32);
    P[19] = (real_t(.001) + real_t(7.e-11) * a5) / (real_t(1.) + real_t(4.4e-11) * a5);
    P[20] = (p1a10 * p1a10 + real_t(2.e-29)) / (real_t(1.) + real_t(2.e-22) * a12);
    P[21] = real_t(400.) / a12 + real_t(2.e-22) * a9;
    P[22] = real_t(1.e-32) * a12 / (real_t(1.) + real_t(5.e22) / a14);
    P[23] = real_t(1000.) / a2 + real_t(9.5) * sa * ssa;
    P[24] = real_t(4.e-6) * a * asa + real_t(1.e11) / a16;
    P[25] = (real_t(120.) / a + real_t(.002) * a2) / (real_t(1.) + real_t(2.e14) / a16);
    P[26] = real_t(9.) + real_t(100.) / a;
    P[27] = real_t(.002) * a3 + real_t(3.e7) / a6;
    P[28] = real_t(7.e-15) * a4 * asa;
    P[29] = real_t(9000.) / a4;
    P[30] = real_t(.0011) * asa / (real_t(1.) + real_t(3.e34) / a32 / a4);
    P[31] = real_t(1.e-5) * a2 + real_t(2.e14) / a16;
    P[32] = real_t(1.2e-11) * a2 / (real_t(1.) + real_t(1.5e19) / a12);
    P[33] = real_t(.016) * asa / (real_t(1.) + real_t(5.e16) / a16);
    P[34] = real_t(.002) * a4 / (real_t(1.) + real_t(7.e7) / pow(a - real_t(6.83), real_t(14)));
    P[35] = real_t(2.e6) / a6 + real_t(7.2) / pow(a, real_t(.11));
    P[36] = real_t(11.) * a3 / (real_t(1.) + real_t(7.e23) / a16 / a8);
    P[37] = real_t(100.) / asa;
    P[38] = (real_t(.1) + real_t(4.4e-5) * a2) / (real_t(1.) + real_t(5.e5) / a4);
    P[39] = real_t(3.5e-4) * a2 / (real_t(1.) + real_t(1.e8) / a8);
    P[40] = real_t(1.3) + real_t(3.e5) / a4;
    P[41] = real_t(500.) / (a2 + real_t(50.)) + real_t(3);
    P[42] = real_t(1.e-9) / a + s4a16 * s4a16;
    P[43] = real_t(.4) * asa + real_t(3.e-9) * a6;
    P[44] = real_t(.0005) * a5;
    P[45] = real_t(.002) * a5;
    P[46] = real_t(10.);
    P[47] = real_t(.05) + real_t(.005) * a;
    P[48] = real_t(7.e-8) / sa;
    P[49] = real_t(.8) * sa;
    P[50] = real_t(.02) * sa;
    P[51] = real_t(1.e8) / a3;
    P[52] = real_t(3.e32) / (a32 + real_t(1.e32));
    P[53] = real_t(24.);
    P[54] = real_t(20.) / sa;
    P[55] = real_t(7.e3) * a / (sa + real_t(1.));
    P[56] = real_t(900.) * sa / (real_t(1.) + real_t(500.) / a3);
  }
  P[57] = real_t(1.e15) + real_t(2.e27) / a4 / (real_t(1.) + real_t(2.e-18) * a16);
  return out;
}

/// G4ChipsNeutronElasticXS::GetTabValues. `lp` is log(p/GeV).
template <typename real_t>
__host__ __device__ ChipsTabValues<real_t> chips_neutron_tab_values(
    const ChipsNeutronPars<real_t>& pars, real_t lp, int tg_z, int tg_n, bool* refused) {
  ChipsTabValues<real_t> v;
  if (refused) { *refused = false; }
  if (tg_z < 0) { if (refused) { *refused = true; } return v; }
  if (tg_z - 1 < 0) { tg_z = 1; tg_n = 0; }

  const real_t* P = pars.p;
  const real_t p = exp(lp), sp = sqrt(p);
  const real_t p2 = p * p, p3 = p2 * p, p4 = p3 * p;

  if (tg_z == 1 && tg_n == 0) {  // np
    const real_t ssp = sqrt(sp);
    const real_t p2s = p2 * sp;
    const real_t dl1 = lp - P[3];
    v.ss = P[27];
    v.s1 = (P[9] + P[10] * dl1 * dl1 + P[11] / p) / (real_t(1.) + P[12] / p4) +
           P[13] / (p4 + P[14]);
    v.b1 = (P[17] + P[18] / (p4 * p4 + P[19] * p3)) / (real_t(1.) + P[20] / p4);
    v.s2 = (P[15] + P[16] / p4 / p) / p3;
    v.b2 = P[22] / (p * sp + P[23]);
    v.s3 = real_t(0.);
    v.b3 = real_t(0.);
    v.s4 = real_t(0.);
    v.b4 = real_t(0.);
    v.cs = P[0] / (p2s + P[1] * p + P[2] / ssp) + P[4] / p +
           (P[5] + P[6] * dl1 * dl1 + P[7] / p) / (real_t(1.) + P[8] / p4);
    return v;
  }

  const real_t p5 = p4 * p, p6 = p5 * p, p8 = p6 * p2, p10 = p8 * p2;
  const real_t p12 = p10 * p2, p16 = p8 * p8;
  const real_t dl = lp - real_t(5.);
  const real_t a = real_t(tg_z + tg_n);
  if (a < real_t(6.5)) {
    const real_t pah = pow(p, a / real_t(2));
    const real_t pa = pah * pah;
    const real_t pa2 = pa * pa;
    v.s1 = P[15] / (real_t(1.) + P[16] * p4 * pa) + P[17] / (p4 + P[18] * p4 / pa2) +
           (P[19] * dl * dl + P[20]) / (real_t(1.) + P[21] / p2);
    v.b1 = (P[22] + P[23] * p2) / (p4 + P[24] / pah) + P[25];
    v.ss = P[26] / (real_t(1.) + P[27] / p2) + P[28] / (p6 / pa + P[29] / p16);
    v.s2 = P[30] / (pa / p2 + P[31] / p4) + P[32];
    v.b2 = P[33] * pow(p, P[34]) + P[35] / (p8 + P[36] / p16);
    v.s3 = P[37] / (pa * p + P[38] / pa) + P[39];
    v.b3 = P[40] / (p3 + P[41] / p6) + P[42] / (real_t(1.) + P[43] / p2);
    v.s4 = p2 * (pah * P[44] * exp(-pah * P[45]) +
                 P[46] / (real_t(1.) + P[47] * pow(p, P[48])));
    v.b4 = P[49] * pa / p2 / (real_t(1.) + pa * P[50]);
  } else {
    v.s1 = P[15] / (real_t(1.) + P[16] / p4) + P[17] / (p4 + P[18] / p2) +
           P[19] / (p5 + P[20] / p16);
    v.b1 = (P[21] / p8 + P[25]) / (p + P[22] / pow(p, P[26])) +
           P[23] / (real_t(1.) + P[24] / p4);
    v.ss = P[27] / (p4 / pow(p, P[29]) + P[28] / p4);
    v.s2 = P[30] / p4 / (pow(p, P[31]) + P[32] / p12) + P[33];
    v.b2 = P[34] / pow(p, P[35]) + P[36] / pow(p, P[37]);
    v.s3 = P[38] / pow(p, P[41]) / (real_t(1.) + P[42] / p12) +
           P[39] / (real_t(1.) + P[40] / p6);
    v.b3 = P[43] / p8 + P[44] / p2 + P[45] / (real_t(1.) + P[46] / p8);
    v.s4 = (P[47] / p4 + P[52] / p) / (real_t(1.) + P[48] / p10) +
           (P[49] + P[50] * dl * dl) / (real_t(1.) + P[51] / p12);
    v.b4 = P[53] / (real_t(1.) + P[54] / p) + P[55] * p4 / (real_t(1.) + P[56] * p5);
  }
  v.cs = (P[0] * dl * dl + P[1]) / (real_t(1.) + P[2] / p + P[3] / p4) +
         P[5] / (p3 + P[6] / p3) +
         P[7] / (p2 + P[4] / (p2 + P[8]) + P[9] / p) + P[10] / (p5 + P[11] / p2) + P[12] / p;
  return v;
}

/// G4ChipsProtonElasticXS::GetQ2max / G4ChipsNeutronElasticXS::GetQ2max, in GeV^2.
///
/// The like-particle case (pp for the proton, nn for the neutron) is `2*(sqrt(p^2+m^2)*m - m^2)`,
/// which is twice the CMS 90-degree value - Geant4's comment says so. Everything else is the
/// Mandelstam form with the TARGET NUCLEUS mass from G4IonTable::GetIon(Z, Z+N, 0), which is
/// G4NucleiProperties::GetNuclearMass; it comes in as a functor for the reason given in
/// hadron_elastic.cuh.
///
/// The neutron's version has one extra branch: `mt = mProt` unless `tgN || tgZ > 1`. So a
/// neutron on (Z=1, N=0) - a free proton - uses the proton mass directly instead of looking up
/// GetIon(1,1,0); the two are the same number, because G4NucleiProperties::GetNuclearMass(1,1)
/// returns the proton's PDG mass, so the branch only avoids the lookup. It is written out below
/// anyway, because "the same number" is a claim about P3's table and this file should not
/// depend on it.
///
/// Both classes raise a FatalException when `tgZ == 0 && tgN == 0`. The proton class also
/// reaches G4IonTable::GetIon(0, 1, 0) for a free-NEUTRON target, which has no ion - so a proton
/// on Z=0 is a crash in Geant4 and is refused by name here. No material has a Z=0 element, which
/// is why nobody has hit it.
///
/// `hadron_hadron` here is the test GetQ2max makes, which for the NEUTRON class is `tgZ==0 &&
/// tgN==1` and not the `tgZ==1 && tgN==0` its three other functions ask - see the note in
/// `chips_sample_invariant_t`. The caller passes the right one; this function does not decide.
template <typename real_t>
__host__ __device__ real_t chips_q2max_gev2(real_t p_gev, real_t projectile_mass_gev,
                                            real_t target_mass_gev, bool hadron_hadron) {
  const real_t pp2 = p_gev * p_gev;
  const real_t m = projectile_mass_gev, m2 = m * m;
  if (hadron_hadron) {
    const real_t t_mid = sqrt(pp2 + m2) * m - m2;
    return t_mid + t_mid;
  }
  const real_t mt = target_mass_gev;
  const real_t dmt = mt + mt;
  const real_t mds = dmt * sqrt(pp2 + m2) + m2 + mt * mt;
  return dmt * dmt * pp2 / mds;
}

/// The interpolated CHIPS state at one momentum: the total cross section in mb, the nine
/// differential parameters, (-t)max in GeV^2, and log(p/GeV).
template <typename real_t>
struct ChipsState {
  ChipsTabValues<real_t> v;
  real_t last_tm = real_t(0);  ///< (-t)max, GeV^2
  real_t last_lp = real_t(0);  ///< log(p/GeV)
  bool refused = false;        ///< Z outside the tabulated range
  bool non_positive_cs = false;  ///< the parameterisation returned <= 0; see the header note (1)
};

/// G4Chips*ElasticXS::CalculateCrossSection, stateless.
///
/// `tab(lp)` must be the class's GetTabValues bound to its parameters and (Z, N).
///
/// The three branches of the original, in order:
///   - `lastLP > lPMin && lastLP < lastPIN`: linear interpolation between the two bracketing
///     bins of the 128-point log(p) grid, with `blast` clamped to [0, nLast-1].
///   - `lastLP == lastPIN`: a single bin. Reachable only at exactly lp == lPMax, see the header.
///   - otherwise (lp <= lPMin, or lp >= lPMax): GetTabValues called directly at lp.
/// and finally `if(lastSIG<0.) lastSIG = 0.` - the total is floored at zero while the nine
/// differential parameters are not.
template <typename real_t, typename TabFn>
__host__ __device__ ChipsState<real_t> chips_calculate(const TabFn& tab, real_t p_mev,
                                                       real_t last_tm_gev2) {
  ChipsState<real_t> st;
  const real_t p_gev = p_mev / units::GeV<real_t>();
  const real_t lp = log(p_gev);
  st.last_lp = lp;
  st.last_tm = last_tm_gev2;

  const real_t lpmin = real_t(ChipsGrid::kLPMin), lpmax = real_t(ChipsGrid::kLPMax);
  const real_t dlnp = real_t(ChipsGrid::kDlnP);

  if (lp > lpmin && lp < lpmax) {
    real_t shift = (lp - lpmin) / dlnp;
    int blast = static_cast<int>(shift);
    if (blast < 0) { blast = 0; }
    if (blast >= ChipsGrid::kNLast) { blast = ChipsGrid::kNLast - 1; }
    shift -= real_t(blast);
    bool r0 = false, r1 = false;
    const ChipsTabValues<real_t> lo = tab(lpmin + real_t(blast) * dlnp, &r0);
    const ChipsTabValues<real_t> hi = tab(lpmin + real_t(blast + 1) * dlnp, &r1);
    st.refused = r0 || r1;
    st.v.cs = lo.cs + shift * (hi.cs - lo.cs);
    st.v.ss = lo.ss + shift * (hi.ss - lo.ss);
    st.v.s1 = lo.s1 + shift * (hi.s1 - lo.s1);
    st.v.b1 = lo.b1 + shift * (hi.b1 - lo.b1);
    st.v.s2 = lo.s2 + shift * (hi.s2 - lo.s2);
    st.v.b2 = lo.b2 + shift * (hi.b2 - lo.b2);
    st.v.s3 = lo.s3 + shift * (hi.s3 - lo.s3);
    st.v.b3 = lo.b3 + shift * (hi.b3 - lo.b3);
    st.v.s4 = lo.s4 + shift * (hi.s4 - lo.s4);
    st.v.b4 = lo.b4 + shift * (hi.b4 - lo.b4);
  } else if (lp == lpmax) {
    // The `lastLP == lastPIN` branch: bin 127, the top of the table. Geant4 also prints
    // "G4QEleastCS::CCS:b=127,127" here because its own bounds check fires.
    bool r = false;
    st.v = tab(lpmin + real_t(ChipsGrid::kNLast) * dlnp, &r);
    st.refused = r;
  } else {
    bool r = false;
    st.v = tab(lp, &r);
    st.refused = r;
  }
  if (st.v.cs < real_t(0)) { st.v.cs = real_t(0); }
  st.non_positive_cs = !(st.v.cs > real_t(0));
  return st;
}

/// G4ChipsProtonElasticXS::GetExchangeT and G4ChipsNeutronElasticXS::GetExchangeT.
///
/// Returns -t in MeV^2. `hadron_hadron` selects the pp (proton class) or np (neutron class)
/// two-channel branch; everything else takes the four-channel nuclear branch. The two hadron-
/// hadron branches are DIFFERENT between the two classes and both are here:
///   pp: three channels, R2 uses exp(-E2^3), I1 = R1*S1/B1, I2 = R2*S2, I3 = R3*S3, and channel
///       2 returns pow(q2, 1/3)/B2.
///   np: two channels, I1 = R1*S1, I2 = R2*S2/B2, and channel 2 is a u-channel exchange,
///       `q2 = tmax + log(1-ran)/B2`, which is where charge exchange in np elastic comes from.
///
/// Below lp = -4.3 (p < 13.5 MeV/c, kinetic energy under 0.1 MeV) both classes return
/// `tmax * uniform` - pure S-wave, isotropic in the CMS - and consume exactly one uniform.
///
/// The nuclear branch's channel 1 inverts a quadratic when the effective slope `ss` is non-zero:
/// `q2 = (sqrt(B1*(B1 + 4*ss*q2)) - B1)/(2*ss)`, guarded by `|2*ss| > 1e-7`. Channel 4 for
/// A < 6.5 is a u-channel too (`q2 = tmax - q2`), and the powers in channels 2 and 3 change at
/// A = 6.5: 1/3 and 1 below, 1/5 and 1/7 above.
///
/// Every `1 - exp(-E)` is a truncated-exponential normalisation, and `ran = R*uniform` is
/// clamped at 1 before the log. The final value is clamped to [0, tmax].
///
/// `hadron_hadron` is `tgZ == 1 && tgN == 0` for BOTH classes - line 631 of
/// G4ChipsProtonElasticXS.cc ("===> p+p=p+p") and line 1878 of G4ChipsNeutronElasticXS.cc
/// ("===> n+p=n+p"). It is NOT the same test as the one GetQ2max makes; see the note on
/// `chips_sample_invariant_t` below, which is where the two are kept apart.
template <typename real_t, typename Rng>
__host__ __device__ real_t chips_exchange_t(const ChipsState<real_t>& st, int tg_z, int tg_n,
                                            bool hadron_hadron, bool neutron_class, Rng& rng) {
  const real_t gev_sq = units::GeV<real_t>() * units::GeV<real_t>();
  const real_t third = real_t(1) / real_t(3);
  const real_t fifth = real_t(1) / real_t(5);
  const real_t sevth = real_t(1) / real_t(7);
  const real_t tmax = st.last_tm;

  if (st.last_lp < real_t(-4.3)) { return tmax * gev_sq * rng.uniform(); }

  const ChipsTabValues<real_t>& v = st.v;
  real_t q2 = real_t(0);

  if (hadron_hadron) {
    if (!neutron_class) {
      // pp
      const real_t E1 = tmax * v.b1;
      const real_t R1 = real_t(1) - exp(-E1);
      const real_t E2 = tmax * v.b2;
      const real_t R2 = real_t(1) - exp(-E2 * E2 * E2);
      const real_t E3 = tmax * v.b3;
      const real_t R3 = real_t(1) - exp(-E3);
      const real_t I1 = R1 * v.s1 / v.b1;
      const real_t I2 = R2 * v.s2;
      const real_t I3 = R3 * v.s3;
      const real_t I12 = I1 + I2;
      const real_t rand = (I12 + I3) * rng.uniform();
      if (rand < I1) {
        real_t ran = R1 * rng.uniform();
        if (ran > real_t(1)) { ran = real_t(1); }
        q2 = -log(real_t(1) - ran) / v.b1;
      } else if (rand < I12) {
        real_t ran = R2 * rng.uniform();
        if (ran > real_t(1)) { ran = real_t(1); }
        q2 = -log(real_t(1) - ran);
        if (q2 < real_t(0)) { q2 = real_t(0); }
        q2 = pow(q2, third) / v.b2;
      } else {
        real_t ran = R3 * rng.uniform();
        if (ran > real_t(1)) { ran = real_t(1); }
        q2 = -log(real_t(1) - ran) / v.b3;
      }
    } else {
      // np
      const real_t E1 = tmax * v.b1;
      const real_t R1 = real_t(1) - exp(-E1);
      const real_t E2 = tmax * v.b2;
      const real_t R2 = real_t(1) - exp(-E2);
      const real_t I1 = R1 * v.s1;
      const real_t I2 = R2 * v.s2 / v.b2;
      const real_t I12 = I1 + I2;
      const real_t rand = I12 * rng.uniform();
      if (rand < I1) {
        real_t ran = R1 * rng.uniform();
        if (ran > real_t(1)) { ran = real_t(1); }
        q2 = -log(real_t(1) - ran) / v.b1;  // t-channel
      } else {
        real_t ran = R2 * rng.uniform();
        if (ran > real_t(1)) { ran = real_t(1); }
        q2 = tmax + log(real_t(1) - ran) / v.b2;  // u-channel, charge exchange
      }
    }
  } else {
    const real_t a = real_t(tg_z + tg_n);
    const real_t E1 = tmax * (v.b1 + tmax * v.ss);
    const real_t R1 = real_t(1) - exp(-E1);
    const real_t tss = v.ss + v.ss;
    const real_t tm2 = tmax * tmax;
    real_t E2 = tmax * tm2 * v.b2;
    if (a > real_t(6.5)) { E2 *= tm2; }
    const real_t R2 = real_t(1) - exp(-E2);
    real_t E3 = tmax * v.b3;
    if (a > real_t(6.5)) { E3 *= tm2 * tm2 * tm2; }
    const real_t R3 = real_t(1) - exp(-E3);
    const real_t E4 = tmax * v.b4;
    const real_t R4 = real_t(1) - exp(-E4);
    const real_t I1 = R1 * v.s1;
    const real_t I2 = R2 * v.s2;
    const real_t I3 = R3 * v.s3;
    const real_t I4 = R4 * v.s4;
    const real_t I12 = I1 + I2;
    const real_t I13 = I12 + I3;
    const real_t rand = (I13 + I4) * rng.uniform();
    if (rand < I1) {
      real_t ran = R1 * rng.uniform();
      if (ran > real_t(1)) { ran = real_t(1); }
      q2 = -log(real_t(1) - ran) / v.b1;
      const real_t atss = (tss > real_t(0)) ? tss : -tss;
      if (atss > real_t(1.e-7)) {
        q2 = (sqrt(v.b1 * (v.b1 + (tss + tss) * q2)) - v.b1) / tss;
      }
    } else if (rand < I12) {
      real_t ran = R2 * rng.uniform();
      if (ran > real_t(1)) { ran = real_t(1); }
      q2 = -log(real_t(1) - ran) / v.b2;
      if (q2 < real_t(0)) { q2 = real_t(0); }
      q2 = (a < real_t(6.5)) ? pow(q2, third) : pow(q2, fifth);
    } else if (rand < I13) {
      real_t ran = R3 * rng.uniform();
      if (ran > real_t(1)) { ran = real_t(1); }
      q2 = -log(real_t(1) - ran) / v.b3;
      if (q2 < real_t(0)) { q2 = real_t(0); }
      if (a > real_t(6.5)) { q2 = pow(q2, sevth); }
    } else {
      real_t ran = R4 * rng.uniform();
      if (ran > real_t(1)) { ran = real_t(1); }
      q2 = -log(real_t(1) - ran) / v.b4;
      if (a < real_t(6.5)) { q2 = tmax - q2; }  // u reduced for light A
    }
  }
  if (q2 < real_t(0)) { q2 = real_t(0); }
  if (q2 > tmax) { q2 = tmax; }
  return q2 * gev_sq;
}

/// G4Chips*ElasticXS::GetSlope - the first diffraction slope, in MeV^-2. Below lp = -4.3 it is
/// zero (S-wave). Not used by the elastic model; G4ChargeExchangeProcess reads it.
template <typename real_t>
__host__ __device__ real_t chips_slope(const ChipsState<real_t>& st) {
  if (st.last_lp < real_t(-4.3)) { return real_t(0); }
  const real_t b1 = (st.v.b1 < real_t(0)) ? real_t(0) : st.v.b1;
  return b1 / (units::GeV<real_t>() * units::GeV<real_t>());
}

/// G4ChipsElasticModel::SampleInvariantT, for a proton or a neutron projectile.
///
/// `pdg` must be 2212 or 2112: those are the only two G4HadronElasticPhysics registers a
/// G4ChipsElasticModel for. The model's own dispatch also handles pbar, pi+-, K+- through
/// G4ChipsAntiBaryonElasticXS / G4ChipsPionPlusElasticXS / G4ChipsPionMinusElasticXS /
/// G4ChipsKaonPlusElasticXS / G4ChipsKaonMinusElasticXS, which are five more classes of the same
/// size and are NOT ported: QBBC gives pions G4ElasticHadrNucleusHE and kaons
/// G4HadronicBuilder::BuildElastic, so no QBBC particle reaches them. `*unsupported_pdg` says so
/// by name rather than returning a plausible number.
///
/// The (Z,N) remap comes first: (1,2) -> N=1 and (2,1) -> N=2. Then the cross section, and only
/// if it is positive is GetExchangeT called; otherwise Geant4 falls back to
/// G4HadronElastic::SampleInvariantT, which is what the caller must do when this returns
/// `used_fallback`.
template <typename real_t, typename NuclearMassFn, typename Rng>
__host__ __device__ real_t chips_sample_invariant_t(int pdg, real_t plab, int z, int a,
                                                    real_t p_local_tmax,
                                                    const NuclearMassFn& nuclear_mass,
                                                    Rng& rng, bool* used_fallback,
                                                    bool* unsupported_pdg) {
  if (used_fallback) { *used_fallback = false; }
  if (unsupported_pdg) { *unsupported_pdg = false; }

  int n = a - z;
  if (z == 1 && n == 2) { n = 1; }
  else if (z == 2 && n == 1) { n = 2; }

  const bool is_proton = (pdg == 2212);
  const bool is_neutron = (pdg == 2112);
  if (!is_proton && !is_neutron) {
    if (unsupported_pdg) { *unsupported_pdg = true; }
    if (used_fallback) { *used_fallback = true; }
    return hadron_elastic_sample_invariant_t<real_t>(pdg, plab, z, a, p_local_tmax, rng);
  }

  const real_t m_gev = (is_proton ? units::proton_mass_c2<real_t>()
                                  : units::neutron_mass_c2<real_t>()) *
                       real_t(0.001);
  // ------------------------------------------------------------------------------------------
  // The neutron class asks a DIFFERENT (Z,N) question in GetQ2max than in its other three
  // functions, and the disagreement is Geant4's, not a transcription slip.
  //
  //   G4ChipsNeutronElasticXS::GetQ2max      line 2101:  if(tgZ==0 && tgN==1)   <- a free NEUTRON
  //   G4ChipsNeutronElasticXS::GetPTables    line 1611:  if(tgZ==1 && tgN==0)   <- a free PROTON
  //   G4ChipsNeutronElasticXS::GetTabValues  line 2021:  if(tgZ==1 && tgN==0)      "  np "
  //   G4ChipsNeutronElasticXS::GetExchangeT  line 1878:  if(tgZ==1 && tgN==0)      "n+p=n+p"
  //
  // So for a neutron on free hydrogen (Z=1, N=0) Geant4 takes the two-channel np parameter row
  // and the two-channel np t-sampling, but computes (-t)max from the nuclear/Mandelstam
  // expression with mt = m_proton; and for the (unphysical in any material) free-neutron target
  // (Z=0, N=1) it does the opposite. The proton class is consistent - all four ask (1,0).
  //
  // One flag for both would be wrong for the neutron whichever way it was set: setting it from
  // GetQ2max sends n+p through the four-channel nuclear sampler, which is what this port did
  // until the oracle caught it at Z=1 N=0, p = 3 GeV/c (a factor-of-20 error in -t and a 38-sigma
  // histogram). Two flags, one per function, reproduce both branches.
  const bool hh_q2max = is_proton ? (z == 1 && n == 0) : (z == 0 && n == 1);
  const bool hh_exchange = (z == 1 && n == 0);
  if (z == 0 && !hh_q2max) {
    // G4ChipsProtonElasticXS::GetQ2max would ask G4IonTable for GetIon(0, 1, 0). REFUSED by name
    // rather than substituted: a Z=0 target does not exist in any material, and a plausible mass
    // here would be an invention.
    if (unsupported_pdg) { *unsupported_pdg = true; }
    if (used_fallback) { *used_fallback = true; }
    return hadron_elastic_sample_invariant_t<real_t>(pdg, plab, z, a, p_local_tmax, rng);
  }
  const real_t target_mass_gev =
      (is_neutron && n == 0 && z <= 1)
          ? units::proton_mass_c2<real_t>() * real_t(0.001)          // the `mt = mProt` branch
          : nuclear_mass(z, z + n) * real_t(0.001);
  const real_t q2max = chips_q2max_gev2<real_t>(plab / units::GeV<real_t>(), m_gev,
                                                target_mass_gev, hh_q2max);

  ChipsState<real_t> st;
  if (is_proton) {
    const ChipsProtonPars<real_t> pars = chips_proton_pars<real_t>(z, n);
    auto tab = [&](real_t lp, bool* ref) {
      return chips_proton_tab_values<real_t>(pars, lp, z, n, ref);
    };
    st = chips_calculate<real_t>(tab, plab, q2max);
  } else {
    bool refused = false;
    const ChipsNeutronPars<real_t> pars = chips_neutron_pars<real_t>(z, n, &refused);
    auto tab = [&](real_t lp, bool* ref) {
      return chips_neutron_tab_values<real_t>(pars, lp, z, n, ref);
    };
    st = chips_calculate<real_t>(tab, plab, q2max);
    st.refused = st.refused || refused;
  }

  if (st.refused || st.non_positive_cs) {
    if (used_fallback) { *used_fallback = true; }
    return hadron_elastic_sample_invariant_t<real_t>(pdg, plab, z, a, p_local_tmax, rng);
  }
  return chips_exchange_t<real_t>(st, z, n, hh_exchange, is_neutron, rng);
}

}  // namespace g4gpu::physics::hadronic::elastic
