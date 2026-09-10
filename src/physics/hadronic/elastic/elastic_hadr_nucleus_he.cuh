// G4ElasticHadrNucleusHE - the elastic final state QBBC gives pi+ and pi-.
//
// Transcribed from Geant4 11.1.1,
//   processes/hadronic/models/coherent_elastic/src/G4ElasticHadrNucleusHE.cc
//     G4ElasticData::G4ElasticData, G4ElasticData::DefineNucleusParameters
//     G4ElasticHadrNucleusHE::G4ElasticHadrNucleusHE (the energy grid), InitialiseModel,
//       SampleInvariantT, FillData, FillFq2, HadrNucDifferCrSec, DefineHadronValues,
//       InterpolateHN, GetQ2_2, HadronNucleusQ2_2, GetFt, HadronProtonQ2
//   processes/hadronic/models/coherent_elastic/include/G4ElasticHadrNucleusHE.hh
//     NHADRONS=26, ONQ2=102, NENERGY=24, ZMAX=93, LineInterpol
//
// ---------------------------------------------------------------------------------------------
// This model runs a table, and the table is built at initialisation.
//
// `G4HadronicProcessStore::PrintInfo` calls
// `G4HadronicInteractionRegistry::Instance()->InitialiseModels()` once the last particle's
// physics table is built, and that reaches `G4ElasticHadrNucleusHE::InitialiseModel()`, which
// walks the material-cuts-couple table and calls `FillData(pi+/pi-, idx, Z)` for every element Z
// that appears in any material. So by the time a pion is transported there is, per element, a
// 24-energy set of cumulative Q2 distributions of up to 102 points each. `SampleInvariantT` also
// calls FillData on demand if it meets a Z that initialisation missed, so the tables are the same
// whether they were built early or late.
//
// One asymmetry in InitialiseModel worth knowing: for Z > 1 the pi- entry is made to POINT AT the
// pi+ table (`fElasticData[1][Z] = fElasticData[0][Z]`) rather than being built. That is only a
// memory saving - for Z > 1 `DefineHadronValues` case 2 (pi+) and case 3 (pi-) are the same block
// and the two pions have the same PDG mass, so the two tables would be identical. The per-pion
// difference lives entirely in the `if(Z != 1) return;` tail, which is the free-proton branch, and
// that branch is sampled by `HadronProtonQ2` rather than from a table. The port therefore builds
// per (pion, Z) and gets the sharing for free.
//
// The A each table is built for is `G4lrint(G4NistManager::GetAtomicMassAmu(Z))` - the element's
// mean atomic mass ROUNDED TO AN INTEGER, not the isotope A that `SampleZandA` selected. So an
// oxygen table is an A=16 table whether the isotope drawn was O-16, O-17 or O-18, and the isotope
// only enters through the tmax that `G4HadronElastic::ApplyYourself` computes from the real
// target mass. That is Geant4's, and it is the reason `he_fill_data` takes A separately from the
// (Z, A) of the target.
//
// ---------------------------------------------------------------------------------------------
// What is refused, by name.
//
//   - `GetLightFq2` and the 240x240 binomial table `Binom()` fills for it. GetLightFq2 is the
//     only caller of GetBinomCof, and nothing calls GetLightFq2 - it is dead in 11.1.1. It is
//     also the A^4 quadruple Glauber sum (207^4 ~ 2e9 terms for lead), so this is not a
//     convenience: there is no oracle to check it against, because the oracle's Geant4 never
//     evaluates it.
//   - `fStoreToFile` / `fRetrieveFromFile`, both false by default, and the `hedata/<name><Z>.dat`
//     files under G4LEDATA they would read. With retrieval off the tables are computed, which is
//     what is reproduced here; a run with G4ElasticHadrNucleusHE::SetRetrieveFromFile(true) would
//     read numbers this port does not have.
//   - the hadron list beyond pi+/pi-: the 26-entry code/type tables ARE transcribed and all eight
//     `DefineHadronValues` cases with them, because they are pure parameterisation and the
//     alternative is having them missing when P11's FTFP secondaries arrive. What is not verified
//     is any hadron QBBC does not send here, and PORTED.md says so.
#pragma once

#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/elastic/hadron_elastic.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::elastic {

// ---------------------------------------------------------------------------------------------
// Constants, verbatim from the file scope of G4ElasticHadrNucleusHE.cc and its header.
// ---------------------------------------------------------------------------------------------

struct HeDims {
  static constexpr int kNHadrons = 26;
  static constexpr int kOnQ2 = 102;
  static constexpr int kNEnergy = 24;
  static constexpr int kZMax = 93;
};

/// mb -> GeV^-2. Geant4's own 2.568, not 1/(hbarc^2) recomputed - the two differ in the fourth
/// digit and this one is what the parameterisation was fitted with.
template <typename real_t>
__host__ __device__ constexpr real_t he_mb_to_gev2() { return real_t(2.568); }

/// fHadronCode, fHadronType, fHadronType1 - the PDG codes this model knows and the two type
/// indices they map to. Order matters: `SampleInvariantT` uses the INDEX into this array as the
/// first subscript of fElasticData, so pi+ is table row 0 and pi- row 1.
__host__ __device__ inline const int* he_hadron_code() {
  static const int v[HeDims::kNHadrons] = {
      211, -211, 2112, 2212, 321, -321, 130, 310, 311, -311,
      3122, 3222, 3112, 3212, 3312, 3322, 3334,
      -2212, -2112, -3122, -3222, -3112, -3212, -3312, -3322, -3334};
  return v;
}
__host__ __device__ inline const int* he_hadron_type() {
  static const int v[HeDims::kNHadrons] = {2, 3, 6, 0, 4, 5, 4, 4, 4, 5,
                                           0, 0, 0, 0, 0, 0, 0,
                                           1, 7, 1, 1, 1, 1, 1, 1, 1};
  return v;
}
__host__ __device__ inline const int* he_hadron_type1() {
  static const int v[HeDims::kNHadrons] = {3, 4, 1, 0, 5, 6, 5, 5, 5, 6,
                                           0, 0, 0, 0, 0, 0, 0,
                                           2, 2, 2, 2, 2, 2, 2, 2, 2};
  return v;
}

/// BoundaryP / BoundaryTG / BoundaryTL, set in the constructor. BoundaryTL[0], [1], [3], [4] and
/// [5] are OVERWRITTEN with tmax by HadronProtonQ2 and by HadrNucDifferCrSec's A==1 branch, so
/// only [2] = 1.5 and [6] = 3.0 survive from the constructor. Both are kept here because the
/// overwrite is per call and the constructor value is what indices 2 and 6 use.
template <typename real_t>
struct HeBoundary {
  real_t p[7] = {real_t(9.0), real_t(20.0), real_t(5.0), real_t(8.0),
                 real_t(7.0), real_t(5.0),  real_t(5.0)};
  real_t tg[7] = {real_t(5.0), real_t(1.5), real_t(1.0), real_t(3.0),
                  real_t(3.0), real_t(2.0), real_t(1.5)};
  real_t tl[7] = {real_t(0.0), real_t(0.0), real_t(1.5), real_t(0.0),
                  real_t(0.0), real_t(0.0), real_t(3.0)};
};

/// The lower limit below which the model hands over to Gheisha: ekinLowLimit = 400 MeV.
template <typename real_t>
__host__ __device__ constexpr real_t he_ekin_low_limit() {
  return real_t(400) * units::MeV<real_t>();
}

/// fEnergy and fLowEdgeEnergy, in GeV, exactly as the constructor builds them: four hand-set
/// points and then a decade-per-five-bins geometric grid with `f = 10^0.1`,
/// `fEnergy[i] = e`, `fLowEdgeEnergy[i] = e/f`, `e *= f*f`, starting from `e = f*f`.
///
/// Written as a filled struct rather than a constexpr array because `G4Exp(G4Log(10.)*0.1)` is
/// not `pow(10,0.1)` to the last bit and the grid is what the table's energy bin edges are.
template <typename real_t>
struct HeEnergyGrid {
  real_t energy[HeDims::kNEnergy];
  real_t low_edge[HeDims::kNEnergy];

  __host__ __device__ HeEnergyGrid() {
    energy[0] = real_t(0.4);
    energy[1] = real_t(0.6);
    energy[2] = real_t(0.8);
    energy[3] = real_t(1.0);
    low_edge[0] = real_t(0.0);
    low_edge[1] = real_t(0.5);
    low_edge[2] = real_t(0.7);
    low_edge[3] = real_t(0.9);
    const real_t f = exp(log(real_t(10.)) * real_t(0.1));
    real_t e = f * f;
    for (int i = 4; i < HeDims::kNEnergy; ++i) {
      energy[i] = e;
      low_edge[i] = e / f;
      e *= f * f;
    }
  }
};

// ---------------------------------------------------------------------------------------------
// G4ElasticData
// ---------------------------------------------------------------------------------------------

/// G4ElasticData::DefineNucleusParameters - the nuclear form-factor parameters, a switch on A
/// with a default fit.
///
/// The switch cases pair A with A+1 (207/208, 237/238, 90/91, 58/59, 48/47, 40/41, 28/29) but 16,
/// 12, 11, 9, 4 and 1 stand alone, so A = 14 (nitrogen) or A = 27 (aluminium) take the DEFAULT
/// branch even though 12 and 28 are tabulated. That is not an oversight to tidy: the default
/// branch's R1 is `4.45*exp(log(A-1)*0.309)*0.9`, which for A = 1 would be log(0) - the A = 1
/// case exists to keep that from happening.
template <typename real_t>
struct HeNucleusParams {
  real_t r1 = real_t(0), r2 = real_t(0), pnucl = real_t(0), aeff = real_t(0);
};

template <typename real_t>
__host__ __device__ HeNucleusParams<real_t> he_define_nucleus_parameters(int a) {
  HeNucleusParams<real_t> p;
  switch (a) {
    case 207: case 208: p.r1 = real_t(20.5); p.r2 = real_t(15.74); p.pnucl = real_t(0.4);
                        p.aeff = real_t(0.7); break;
    case 237: case 238: p.r1 = real_t(21.7); p.r2 = real_t(16.5);  p.pnucl = real_t(0.4);
                        p.aeff = real_t(0.7); break;
    case 90:  case 91:  p.r1 = real_t(16.5); p.r2 = real_t(11.62); p.pnucl = real_t(0.4);
                        p.aeff = real_t(0.7); break;
    case 58:  case 59:  p.r1 = real_t(15.75); p.r2 = real_t(9.9);  p.pnucl = real_t(0.45);
                        p.aeff = real_t(0.85); break;
    case 48:  case 47:  p.r1 = real_t(14.0); p.r2 = real_t(9.26);  p.pnucl = real_t(0.31);
                        p.aeff = real_t(0.75); break;
    case 40:  case 41:  p.r1 = real_t(13.3); p.r2 = real_t(9.26);  p.pnucl = real_t(0.31);
                        p.aeff = real_t(0.75); break;
    case 28:  case 29:  p.r1 = real_t(12.0); p.r2 = real_t(7.64);  p.pnucl = real_t(0.253);
                        p.aeff = real_t(0.8); break;
    case 16:            p.r1 = real_t(10.50); p.r2 = real_t(5.5);  p.pnucl = real_t(0.7);
                        p.aeff = real_t(0.98); break;
    case 12:            p.r1 = real_t(9.3936); p.r2 = real_t(4.63); p.pnucl = real_t(0.7);
                        p.aeff = real_t(1.0); break;
    case 11:            p.r1 = real_t(9.0);  p.r2 = real_t(5.42);  p.pnucl = real_t(0.19);
                        p.aeff = real_t(0.9); break;
    case 9:             p.r1 = real_t(9.9);  p.r2 = real_t(6.5);   p.pnucl = real_t(0.690);
                        p.aeff = real_t(0.95); break;
    case 4:             p.r1 = real_t(5.3);  p.r2 = real_t(3.7);   p.pnucl = real_t(0.4);
                        p.aeff = real_t(0.75); break;
    case 1:             p.r1 = real_t(4.5);  p.r2 = real_t(2.3);   p.pnucl = real_t(0.177);
                        p.aeff = real_t(0.9); break;
    default:
      p.r1 = real_t(4.45) * exp(log(real_t(a - 1)) * real_t(0.309)) * real_t(0.9);
      p.r2 = real_t(2.3) * exp(log(real_t(a)) * real_t(0.36));
      p.pnucl = (a < 100 && a > 3) ? real_t(0.176) + real_t(0.00275) * real_t(a) : real_t(0.4);
      if (a >= 100)                { p.aeff = real_t(0.7); }
      else if (a < 100 && a > 75)  { p.aeff = real_t(1.5) - real_t(0.008) * real_t(a); }
      else                         { p.aeff = real_t(0.9); }
      break;
  }
  return p;
}

/// The hadron-nucleon quantities `DefineHadronValues` produces, in the units its comments give.
template <typename real_t>
struct HeHadronValues {
  real_t hadr_tot = real_t(0);    ///< mb
  real_t hadr_slope = real_t(0);  ///< GeV^-2
  real_t hadr_re_im = real_t(0);
  real_t ddsect2 = real_t(0);     ///< mb GeV^-2
  real_t ddsect3 = real_t(0);
  real_t tot_p = real_t(0);       ///< the proton-target total, kept because it is a member
  // The free-proton (Z == 1) extras.
  real_t coeff0 = real_t(0), coeff1 = real_t(0), coeff2 = real_t(0);
  real_t slope0 = real_t(1), slope1 = real_t(1), slope2 = real_t(5);
};

/// G4ElasticHadrNucleusHE::LineInterpol.
template <typename real_t>
__host__ __device__ inline real_t he_line_interpol(real_t p1, real_t p2, real_t c1, real_t c2,
                                                   real_t p) {
  return c1 + (p - p1) * (c2 - c1) / (p2 - p1);
}

/// G4ElasticHadrNucleusHE::InterpolateHN - a linear interpolation in lab momentum over four
/// parallel tables, with the index found by the first `hLabMomentum <= EnP[i]` from i = 1 and
/// clamped to n-1. Note that below EnP[0] it EXTRAPOLATES backwards through the i = 1 pair, and
/// above EnP[n-1] through the last pair; the caller guards the upper end with BoundaryP.
template <typename real_t>
__host__ __device__ void he_interpolate_hn(int n, const real_t* enp, const real_t* c0p,
                                           const real_t* c1p, const real_t* b0p,
                                           const real_t* b1p, real_t h_lab_momentum,
                                           HeHadronValues<real_t>* v) {
  int i;
  for (i = 1; i < n; ++i) { if (h_lab_momentum <= enp[i]) { break; } }
  if (i == n) { i = n - 1; }
  v->coeff0 = he_line_interpol<real_t>(enp[i], enp[i - 1], c0p[i], c0p[i - 1], h_lab_momentum);
  v->coeff1 = he_line_interpol<real_t>(enp[i], enp[i - 1], c1p[i], c1p[i - 1], h_lab_momentum);
  v->slope0 = he_line_interpol<real_t>(enp[i], enp[i - 1], b0p[i], b0p[i - 1], h_lab_momentum);
  v->slope1 = he_line_interpol<real_t>(enp[i], enp[i - 1], b1p[i], b1p[i - 1], h_lab_momentum);
}

/// G4ElasticHadrNucleusHE::DefineHadronValues.
///
/// `i_hadron` is fHadronType[idx] and `i_hadron1` is fHadronType1[idx]; `i_hadr_code` is the PDG
/// code, which the hyperon multipliers test on. All eight `iHadron` cases are transcribed.
///
/// `h_lab_momentum`, `h_lab_momentum2`, `hadr_energy` and `h_mass2` are in GeV / GeV^2 - the
/// comment at the top of HadrNucDifferCrSec ("All external kinematical variables are in MeV, but
/// internal in GeV") is the whole file's convention.
///
/// `hadr_energy` is an ARGUMENT rather than `sqrt(h_mass2 + p^2)` computed here, because Geant4's
/// two call sites set the member differently: `FillData` uses `HadrEnergy = hMass + T` and
/// `HadronProtonQ2` uses `HadrEnergy = sqrt(hMass2 + hLabMomentum2)`. Those are the same number
/// in exact arithmetic - hLabMomentum2 is T*(T+2m) - and not the same double. Computing it here
/// would silently pick one of the two for both.
///
/// Two details: `logE` is log(HadrEnergy) and `logS` is log(sHadr) with
/// `sHadr = 2*E*mp + mp^2 + m^2`, so the nucleon-nucleon Mandelstam s uses the PROTON mass for
/// the target whatever the nucleus is. And in the nucleon case the `hLabMomentum > 10` branch
/// sets TotP = TotN together, so the low-momentum proton/neutron split below it is unreachable
/// above 10 GeV/c.
template <typename real_t>
__host__ __device__ HeHadronValues<real_t> he_define_hadron_values(
    int z, int i_hadron, int i_hadron1, int i_hadr_code, real_t h_mass, real_t h_lab_momentum,
    real_t hadr_energy, const HeBoundary<real_t>& bnd) {
  HeHadronValues<real_t> v;
  const real_t proton_m = units::proton_mass_c2<real_t>() * real_t(1e-3);
  const real_t proton_m2 = proton_m * proton_m;
  const real_t h_mass2 = h_mass * h_mass;
  const real_t h_lab_momentum2 = h_lab_momentum * h_lab_momentum;

  const real_t s_hadr = real_t(2) * hadr_energy * proton_m + proton_m2 + h_mass2;
  const real_t sqr_s = sqrt(s_hadr);
  real_t tot_n = real_t(0);
  const real_t log_e = log(hadr_energy);
  const real_t log_s = log(s_hadr);
  real_t tot_p = real_t(0);

  switch (i_hadron) {
    case 0:  // proton
    case 6:  // neutron
      if (h_lab_momentum > real_t(10)) {
        tot_p = tot_n = real_t(7.5) * log_e - real_t(40.12525) +
                        real_t(103) * exp(-log_s * real_t(0.165));
      } else {
        if (h_lab_momentum > real_t(1.4)) {
          tot_n = real_t(33.3) + real_t(15.2) * (h_lab_momentum2 - real_t(1.35)) /
                                     (exp(log(h_lab_momentum) * real_t(2.37)) + real_t(0.95));
        } else if (h_lab_momentum > real_t(0.8)) {
          const real_t a0 = log_e + real_t(0.0513);
          tot_n = real_t(33.0) + real_t(25.5) * a0 * a0;
        } else {
          const real_t a0 = log_e - real_t(0.2634);
          tot_n = real_t(33.0) + real_t(30.) * a0 * a0 * a0 * a0;
        }
        if (h_lab_momentum >= real_t(1.05)) {
          tot_p = real_t(39.0) + real_t(75.) * (h_lab_momentum - real_t(1.2)) /
                                     (h_lab_momentum2 * h_lab_momentum + real_t(0.15));
        } else if (h_lab_momentum >= real_t(0.7)) {
          const real_t a0 = log_e + real_t(0.3147);
          tot_p = real_t(23.0) + real_t(40.) * a0 * a0;
        } else {
          tot_p = real_t(23.) +
                  real_t(50.) * exp(log(log(real_t(0.73) / h_lab_momentum)) * real_t(3.5));
        }
      }
      v.hadr_tot = real_t(0.5) * (tot_p + tot_n);
      if (h_lab_momentum >= real_t(2.)) {
        v.hadr_slope = real_t(5.44) + real_t(0.88) * log_s;
      } else if (h_lab_momentum >= real_t(0.5)) {
        v.hadr_slope = real_t(3.73) * h_lab_momentum - real_t(0.37);
      } else {
        v.hadr_slope = real_t(1.5);
      }
      if (h_lab_momentum >= real_t(1.2)) {
        v.hadr_re_im = real_t(0.13) * (log_s - real_t(5.8579332)) * exp(-log_s * real_t(0.18));
      } else if (h_lab_momentum >= real_t(0.6)) {
        v.hadr_re_im = real_t(-75.5) * (exp(log(h_lab_momentum) * real_t(0.25)) - real_t(0.95)) /
                       (exp(log(real_t(3) * h_lab_momentum) * real_t(2.2)) + real_t(1));
      } else {
        v.hadr_re_im = real_t(15.5) * h_lab_momentum /
                       (real_t(27) * h_lab_momentum2 * h_lab_momentum + real_t(2));
      }
      v.ddsect2 = real_t(2.2);
      v.ddsect3 = real_t(0.6);
      if (i_hadr_code == 3122) { v.hadr_tot *= real_t(0.88); v.hadr_slope *= real_t(0.85); }
      else if (i_hadr_code == 3222) { v.hadr_tot *= real_t(0.81); v.hadr_slope *= real_t(0.85); }
      else if (i_hadr_code == 3112 || i_hadr_code == 3212) {
        v.hadr_tot *= real_t(0.88); v.hadr_slope *= real_t(0.85);
      } else if (i_hadr_code == 3312 || i_hadr_code == 3322) {
        v.hadr_tot *= real_t(0.77); v.hadr_slope *= real_t(0.75);
      } else if (i_hadr_code == 3334) {
        v.hadr_tot *= real_t(0.78); v.hadr_slope *= real_t(0.7);
      }
      break;

    case 1:  // antiproton
    case 7:  // antineutron
      v.hadr_tot = real_t(5.2) + real_t(5.2) * log_e + real_t(123.2) / sqr_s;
      v.hadr_slope = real_t(8.32) + real_t(0.57) * log_s;
      if (hadr_energy < real_t(1000)) {
        v.hadr_re_im = real_t(0.06) * (sqr_s - real_t(2.236)) * (sqr_s - real_t(14.14)) *
                       exp(-log_s * real_t(0.8));
      } else {
        v.hadr_re_im = real_t(0.6) * (log_s - real_t(5.8579332)) * exp(-log_s * real_t(0.25));
      }
      v.ddsect2 = real_t(11);
      v.ddsect3 = real_t(3);
      if (i_hadr_code == -3122) { v.hadr_tot *= real_t(0.88); v.hadr_slope *= real_t(0.85); }
      else if (i_hadr_code == -3222) { v.hadr_tot *= real_t(0.81); v.hadr_slope *= real_t(0.85); }
      else if (i_hadr_code == -3112 || i_hadr_code == -3212) {
        v.hadr_tot *= real_t(0.88); v.hadr_slope *= real_t(0.85);
      } else if (i_hadr_code == -3312 || i_hadr_code == -3322) {
        v.hadr_tot *= real_t(0.77); v.hadr_slope *= real_t(0.75);
      } else if (i_hadr_code == -3334) {
        v.hadr_tot *= real_t(0.78); v.hadr_slope *= real_t(0.7);
      }
      break;

    case 2:  // pi+
    case 3:  // pi-
      if (h_lab_momentum >= real_t(3.5)) {
        tot_p = real_t(10.6) + real_t(2.) * log_e + real_t(25.) * exp(-log_e * real_t(0.43));
      } else if (h_lab_momentum >= real_t(1.15)) {
        const real_t x = (h_lab_momentum - real_t(2.55)) / real_t(0.55);
        const real_t y = (h_lab_momentum - real_t(1.47)) / real_t(0.225);
        tot_p = real_t(3.2) * exp(-x * x) + real_t(12.) * exp(-y * y) + real_t(27.5);
      } else if (h_lab_momentum >= real_t(0.4)) {
        tot_p = real_t(88) * (log_e + real_t(0.2877)) * (log_e + real_t(0.2877)) + real_t(14.0);
      } else {
        const real_t x = (h_lab_momentum - real_t(0.29)) / real_t(0.085);
        tot_p = real_t(20.) + real_t(180.) * exp(-x * x);
      }
      if (h_lab_momentum >= real_t(3.0)) {
        tot_n = real_t(10.6) + real_t(2.) * log_e + real_t(30.) * exp(-log_e * real_t(0.43));
      } else if (h_lab_momentum >= real_t(1.3)) {
        const real_t x = (h_lab_momentum - real_t(2.1)) / real_t(0.4);
        const real_t y = (h_lab_momentum - real_t(1.4)) / real_t(0.12);
        tot_n = real_t(36.1) + real_t(0.079) - real_t(4.313) * log_e + real_t(3.) * exp(-x * x) +
                real_t(1.5) * exp(-y * y);
      } else if (h_lab_momentum >= real_t(0.65)) {
        const real_t x = (h_lab_momentum - real_t(0.72)) / real_t(0.06);
        const real_t y = (h_lab_momentum - real_t(1.015)) / real_t(0.075);
        tot_n = real_t(36.1) + real_t(10.) * exp(-x * x) + real_t(24) * exp(-y * y);
      } else if (h_lab_momentum >= real_t(0.37)) {
        const real_t x = log(h_lab_momentum / real_t(0.48));
        tot_n = real_t(26.) + real_t(110.) * x * x;
      } else {
        const real_t x = (h_lab_momentum - real_t(0.29)) / real_t(0.07);
        tot_n = real_t(28.0) + real_t(40.) * exp(-x * x);
      }
      v.hadr_tot = (tot_p + tot_n) * real_t(0.5);
      v.hadr_slope = real_t(7.28) + real_t(0.245) * log_s;
      v.hadr_re_im = real_t(0.2) * (log_s - real_t(4.6051702)) * exp(-log_s * real_t(0.15));
      v.ddsect2 = real_t(0.7);
      v.ddsect3 = real_t(0.27);
      break;

    case 4:  // K+
      v.hadr_tot = real_t(10.6) + real_t(1.8) * log_e + real_t(9.0) * exp(-log_e * real_t(0.55));
      if (hadr_energy > real_t(100)) { v.hadr_slope = real_t(15.0); }
      else { v.hadr_slope = real_t(1.0) + real_t(1.76) * log_s - real_t(2.84) / sqr_s; }
      v.hadr_re_im = real_t(0.4) * (s_hadr - real_t(20)) * (s_hadr - real_t(150)) *
                     exp(-log(s_hadr + real_t(50)) * real_t(2.1));
      v.ddsect2 = real_t(0.7);
      v.ddsect3 = real_t(0.21);
      break;

    case 5:  // K-
      v.hadr_tot = real_t(10) + real_t(1.8) * log_e + real_t(25.) / sqr_s;
      v.hadr_slope = real_t(6.98) + real_t(0.127) * log_s;
      v.hadr_re_im = real_t(0.4) * (s_hadr - real_t(20)) * (s_hadr - real_t(20)) *
                     exp(-log(s_hadr + real_t(50)) * real_t(2.1));
      v.ddsect2 = real_t(0.7);
      v.ddsect3 = real_t(0.27);
      break;
    default:
      // `iHadron < 0` is caught by SampleInvariantT before this is reached; a value outside
      // 0..7 cannot come out of fHadronType. Left with the zero-initialised values so that a
      // future type shows up as a zero cross section rather than as stale numbers.
      break;
  }
  v.tot_p = tot_p;

  if (z != 1) { return v; }

  // ------- the free-proton branch: Coeff0/1/2 and Slope0/1/2 -------
  v.coeff0 = v.coeff1 = v.coeff2 = real_t(0);
  v.slope0 = v.slope1 = real_t(1);
  v.slope2 = real_t(5);

  const real_t EnP0[6] = {real_t(1.5), real_t(3.0), real_t(5.0), real_t(9.0), real_t(14.0),
                          real_t(19.0)};
  const real_t C0P0[6] = {real_t(0.15), real_t(0.02), real_t(0.06), real_t(0.08),
                          real_t(0.0003), real_t(0.0002)};
  const real_t C1P0[6] = {real_t(0.05), real_t(0.02), real_t(0.03), real_t(0.025), real_t(0.0),
                          real_t(0.0)};
  const real_t B0P0[6] = {real_t(1.5), real_t(2.5), real_t(3.0), real_t(4.5), real_t(1.4),
                          real_t(1.25)};
  const real_t B1P0[6] = {real_t(5.0), real_t(1.0), real_t(3.5), real_t(4.0), real_t(4.8),
                          real_t(4.8)};

  const real_t EnN[5] = {real_t(1.5), real_t(5.0), real_t(10.0), real_t(14.0), real_t(20.0)};
  const real_t C0N[5] = {real_t(0.0), real_t(0.0), real_t(0.02), real_t(0.02), real_t(0.01)};
  const real_t C1N[5] = {real_t(0.06), real_t(0.008), real_t(0.0015), real_t(0.001),
                         real_t(0.0003)};
  const real_t B0N[5] = {real_t(1.5), real_t(2.5), real_t(3.8), real_t(3.8), real_t(3.5)};
  const real_t B1N[5] = {real_t(1.5), real_t(2.2), real_t(3.6), real_t(4.5), real_t(4.8)};

  const real_t EnP[2] = {real_t(1.5), real_t(4.0)};
  const real_t C0P[2] = {real_t(0.001), real_t(0.0005)};
  const real_t C1P[2] = {real_t(0.003), real_t(0.001)};
  const real_t B0P[2] = {real_t(2.5), real_t(4.5)};
  const real_t B1P[2] = {real_t(1.0), real_t(4.0)};

  const real_t EnPP[4] = {real_t(1.0), real_t(2.0), real_t(3.0), real_t(4.0)};
  const real_t C0PP[4] = {real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0)};
  const real_t C1PP[4] = {real_t(0.15), real_t(0.08), real_t(0.02), real_t(0.01)};
  const real_t B0PP[4] = {real_t(1.5), real_t(2.8), real_t(3.8), real_t(3.8)};
  const real_t B1PP[4] = {real_t(0.8), real_t(1.6), real_t(3.6), real_t(4.6)};

  const real_t EnPPN[4] = {real_t(1.0), real_t(2.0), real_t(3.0), real_t(4.0)};
  const real_t C0PPN[4] = {real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0)};
  const real_t C1PPN[4] = {real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0)};
  const real_t B0PPN[4] = {real_t(1.5), real_t(2.8), real_t(3.8), real_t(3.8)};
  const real_t B1PPN[4] = {real_t(0.8), real_t(1.6), real_t(3.6), real_t(4.6)};

  const real_t EnK[4] = {real_t(1.4), real_t(2.33), real_t(3.0), real_t(5.0)};
  const real_t C0K[4] = {real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0)};
  const real_t C1K[4] = {real_t(0.01), real_t(0.007), real_t(0.005), real_t(0.003)};
  const real_t B0K[4] = {real_t(1.5), real_t(2.0), real_t(3.8), real_t(3.8)};
  const real_t B1K[4] = {real_t(1.6), real_t(1.6), real_t(1.6), real_t(1.6)};

  const real_t EnKM[2] = {real_t(1.4), real_t(4.0)};
  const real_t C0KM[2] = {real_t(0.006), real_t(0.002)};
  const real_t C1KM[2] = {real_t(0.00), real_t(0.00)};
  const real_t B0KM[2] = {real_t(2.5), real_t(3.5)};
  const real_t B1KM[2] = {real_t(1.6), real_t(1.6)};

  switch (i_hadron) {
    case 0:
      if (h_lab_momentum < bnd.p[0]) {
        he_interpolate_hn<real_t>(6, EnP0, C0P0, C1P0, B0P0, B1P0, h_lab_momentum, &v);
      }
      v.coeff2 = real_t(0.8) / h_lab_momentum2;
      break;
    case 6:
      if (h_lab_momentum < bnd.p[1]) {
        he_interpolate_hn<real_t>(5, EnN, C0N, C1N, B0N, B1N, h_lab_momentum, &v);
      }
      v.coeff2 = real_t(0.8) / h_lab_momentum2;
      break;
    case 1:
    case 7:
      if (h_lab_momentum < bnd.p[2]) {
        he_interpolate_hn<real_t>(2, EnP, C0P, C1P, B0P, B1P, h_lab_momentum, &v);
      }
      break;
    case 2:
      if (h_lab_momentum < bnd.p[3]) {
        he_interpolate_hn<real_t>(4, EnPP, C0PP, C1PP, B0PP, B1PP, h_lab_momentum, &v);
      }
      v.coeff2 = real_t(0.02) / h_lab_momentum;
      break;
    case 3:
      if (h_lab_momentum < bnd.p[4]) {
        he_interpolate_hn<real_t>(4, EnPPN, C0PPN, C1PPN, B0PPN, B1PPN, h_lab_momentum, &v);
      }
      v.coeff2 = real_t(0.02) / h_lab_momentum;
      break;
    case 4:
      if (h_lab_momentum < bnd.p[5]) {
        he_interpolate_hn<real_t>(4, EnK, C0K, C1K, B0K, B1K, h_lab_momentum, &v);
      }
      v.coeff2 = (h_lab_momentum < real_t(1)) ? real_t(0.34)
                                              : real_t(0.34) / (h_lab_momentum2 * h_lab_momentum);
      break;
    case 5:
      if (h_lab_momentum < bnd.p[6]) {
        he_interpolate_hn<real_t>(2, EnKM, C0KM, C1KM, B0KM, B1KM, h_lab_momentum, &v);
      }
      v.coeff2 = (h_lab_momentum < real_t(1)) ? real_t(0.01)
                                              : real_t(0.01) / (h_lab_momentum2 * h_lab_momentum);
      break;
    default:
      break;
  }
  (void)i_hadron1;  // used only by HadronProtonQ2's BoundaryP/TG/TL lookup
  return v;
}

/// G4ElasticHadrNucleusHE::HadrNucDifferCrSec - d(sigma)/d|t| in mb GeV^-2.
///
/// For A == 1 it is the free-proton form with Coeff0/1/2 and Slope0/1/2; note `valueConstU` here
/// is `m^2 + mp^2 - 2*mp*E - Q2`, which is NOT the `ConstU` that GetFt uses
/// (`2*mp^2 + 2*m^2 - s`), and that the second exponential is `exp(Slope2*(valueConstU) + aQ2)`
/// - a bare `+aQ2` outside the Slope2 factor. Both are as written.
///
/// For A > 1 it is the Glauber series: an O(A) sum over the multiple-scattering order i with an
/// inner O(i) binomial sum over the two-Gaussian form factor, plus a two-step inelastic
/// screening correction summed to A-2. Both sums break early on a 1e-6 relative test, which is
/// part of the answer - `if(std::abs(Prod1*N/ImElasticAmpl0) < 0.000001) break;` uses the running
/// total, so the number of terms depends on the order they are added in.
///
/// `Norm` includes Aeff in the nuclear branch (`(R1^3 - Pnucl*R2^3)*Aeff`) and does NOT in
/// GetLightFq2 (`R1^3 - Pnucl*R2^3`, with the comment `// HP->Aeff;`). The nuclear branch is the
/// one that runs.
template <typename real_t>
__host__ __device__ real_t he_hadr_nuc_differ_cr_sec(int a, real_t aq2, real_t q2max,
                                                     const HeNucleusParams<real_t>& np,
                                                     const HeHadronValues<real_t>& hv,
                                                     real_t h_mass2, real_t hadr_energy) {
  const real_t proton_m = units::proton_mass_c2<real_t>() * real_t(1e-3);
  const real_t proton_m2 = proton_m * proton_m;
  const real_t mb_to_gev2 = he_mb_to_gev2<real_t>();
  const real_t twopi_ = units::twopi<real_t>();
  const real_t pi_ = units::pi<real_t>();

  if (a == 1) {
    const real_t sqr_q2 = sqrt(aq2);
    const real_t value_const_u =
        h_mass2 + proton_m2 - real_t(2) * proton_m * hadr_energy - aq2;
    (void)q2max;  // Geant4 assigns BoundaryTL[0,1,3,4,5] = Q2max here; nothing below reads them
    const real_t d =
        hv.hadr_tot * hv.hadr_tot * (real_t(1) + hv.hadr_re_im * hv.hadr_re_im) *
        (hv.coeff1 * exp(-hv.slope1 * sqr_q2) +
         hv.coeff2 * exp(hv.slope2 * (value_const_u) + aq2) +
         (real_t(1) - hv.coeff1 - hv.coeff0) * exp(-hv.hadr_slope * aq2) +
         hv.coeff0 * exp(-hv.slope0 * aq2)) *
        real_t(2.568) / (real_t(16) * pi_);
    return d;
  }

  const real_t stot = hv.hadr_tot * mb_to_gev2;
  const real_t bhad = hv.hadr_slope;
  const real_t asq = real_t(1) + hv.hadr_re_im * hv.hadr_re_im;
  const real_t rho2 = sqrt(asq);
  const real_t r1 = np.r1, r2 = np.r2;
  const real_t r12 = r1 * r1, r22 = r2 * r2;
  const real_t r12b = r12 + real_t(2) * bhad;
  const real_t r22b = r22 + real_t(2) * bhad;
  const real_t r12ap = r12 + real_t(20);
  const real_t r22ap = r22 + real_t(20);
  const real_t r13ap = r12 * r1 / r12ap;
  const real_t r23ap = r22 * r2 * np.pnucl / r22ap;
  const real_t r23dr13 = r23ap / r13ap;
  const real_t r12apd = real_t(2) / r12ap;
  const real_t r22apd = real_t(2) / r22ap;
  const real_t r12apdr22ap = real_t(0.5) * (r12apd + r22apd);

  const real_t ddsec1p = hv.ddsect2 + hv.ddsect3 * log(real_t(0.53) * hadr_energy / r1);
  const real_t ddsec2p =
      hv.ddsect2 +
      hv.ddsect3 * log(real_t(0.53) * hadr_energy / sqrt((r12 + r22) * real_t(0.5)));
  const real_t ddsec3p = hv.ddsect2 + hv.ddsect3 * log(real_t(0.53) * hadr_energy / r2);

  const real_t norm = (r12 * r1 - np.pnucl * r22 * r2) * np.aeff;
  const real_t r13 = r12 * r1 / r12b;
  const real_t r23 = np.pnucl * r22 * r2 / r22b;
  const real_t unucl = stot / (twopi_ * norm) * r13;
  const real_t unucl_scr = stot / (twopi_ * norm) * r13ap;
  const real_t sin_fi = hv.hadr_re_im / rho2;
  const real_t fi_h = asin(sin_fi);
  real_t n = real_t(-1);
  const real_t n2 = r23 / r13;

  real_t im_ampl0 = real_t(0), re_ampl0 = real_t(0);
  real_t exp1;

  for (int i = 1; i <= a; ++i) {
    n *= (-unucl * rho2 * real_t(a - i + 1) / real_t(i));
    real_t n4 = real_t(1);
    real_t med_tot = r12b / real_t(i);
    real_t prod1 = exp(-aq2 * r12b / real_t(4 * i)) * med_tot;
    for (int l = 1; l <= i; ++l) {
      exp1 = real_t(l) / r22b + real_t(i - l) / r12b;
      n4 *= (-n2 * real_t(i - l + 1) / real_t(l));
      const real_t expn4 = n4 / exp1;
      prod1 += expn4 * exp(-aq2 / (exp1 * real_t(4)));
      med_tot += expn4;
    }
    const real_t dcos = n * cos(fi_h * real_t(i));
    re_ampl0 += prod1 * n * sin(fi_h * real_t(i));
    im_ampl0 += prod1 * dcos;
    const real_t rel = prod1 * n / im_ampl0;
    if (((rel > real_t(0)) ? rel : -rel) < real_t(0.000001)) { break; }
  }

  const real_t pi25 = pi_ / real_t(2.568);
  im_ampl0 *= pi25;
  re_ampl0 *= pi25;

  const real_t c1 = r13ap * r13ap * real_t(0.5) * ddsec1p;
  const real_t c2 = real_t(2) * r23ap * r13ap * real_t(0.5) * ddsec2p;
  const real_t c3 = r23ap * r23ap * real_t(0.5) * ddsec3p;

  real_t n1p = real_t(1);
  real_t din1 = real_t(0.5) * (c1 * exp(-aq2 / real_t(8) * r12ap) / real_t(2) * r12ap -
                               c2 / r12apdr22ap * exp(-aq2 / (real_t(4) * r12apdr22ap)) +
                               c3 * r22ap / real_t(2) * exp(-aq2 / real_t(8) * r22ap));
  real_t dtot1 = real_t(0.5) * (c1 * real_t(0.5) * r12ap - c2 / r12apdr22ap +
                                c3 * r22ap * real_t(0.5));

  for (int i = 1; i <= a - 2; ++i) {
    n1p *= (-unucl_scr * rho2 * real_t(a - i - 1) / real_t(i));
    real_t n2p = real_t(1);
    real_t din2 = real_t(0);
    real_t dmed_tot = real_t(0);
    real_t bin_coeff = real_t(1);
    for (int l = 0; l <= i; ++l) {
      if (l > 0) { bin_coeff *= real_t(i - l + 1) / real_t(l); }
      exp1 = real_t(l) / r22b + real_t(i - l) / r12b;
      const real_t exp1p = exp1 + r12apd;
      const real_t exp2p = exp1 + r12apdr22ap;
      const real_t exp3p = exp1 + r22apd;
      din2 += n2p * bin_coeff * (c1 / exp1p * exp(-aq2 / (real_t(4) * exp1p)) -
                                 c2 / exp2p * exp(-aq2 / (real_t(4) * exp2p)) +
                                 c3 / exp3p * exp(-aq2 / (real_t(4) * exp3p)));
      dmed_tot += n2p * bin_coeff * (c1 / exp1p - c2 / exp2p + c3 / exp3p);
      n2p *= -r23dr13;
    }
    const real_t dcos = n1p * cos(fi_h * real_t(i)) / real_t((i + 2) * (i + 1));
    din1 += din2 * dcos;
    dtot1 += dmed_tot * dcos;
    const real_t rel = din2 * n1p / din1;
    if (((rel > real_t(0)) ? rel : -rel) < real_t(0.000001)) { break; }
  }
  const real_t gg = real_t(a * (a - 1) * 4) / (norm * norm);
  din1 *= (-gg);
  dtot1 *= real_t(5) * gg;
  (void)dtot1;  // Dtot11, a diagnostic member; nothing in the sampling path reads it

  const real_t diff = (re_ampl0 * re_ampl0 + (im_ampl0 + din1) * (im_ampl0 + din1)) / twopi_;
  return diff;
}

/// One (hadron index, Z) table: what G4ElasticData holds, with the cumulative arrays inline.
template <typename real_t>
struct HeElasticData {
  real_t r1 = real_t(0), r2 = real_t(0), pnucl = real_t(0), aeff = real_t(0);
  real_t dq2 = real_t(0);
  real_t mass_a = real_t(0), mass_a2 = real_t(0);   ///< target nuclear mass, GeV
  real_t max_q2[HeDims::kNEnergy] = {};
  int len[HeDims::kNEnergy] = {};                   ///< fCumProb[i].size()
  real_t cum[HeDims::kNEnergy][HeDims::kOnQ2] = {};
  int a = 0;                                        ///< the rounded NIST A the table was built at
  int z = 0;
};

/// G4ElasticHadrNucleusHE::FillFq2 - integrate d(sigma)/d|t| onto the Q2 grid and return the
/// number of cumulative points.
///
/// Each of the 100 possible bins is a 10-point midpoint rule over `dQ2` with step `dQ2*0.1`,
/// EXCEPT that the inner loop breaks as soon as a sample would land at or beyond Q2max, and the
/// bin width `del` then becomes `Q2max - Q2l` while the factor `0.1` stays - so the last bin is a
/// partial sum scaled by the full bin's factor. The outer loop stops when a bin contributes less
/// than 1e-4 of the running total, or when Q2max is reached.
///
/// After the loop a tail is added analytically: `curSec*(1 - exp(-R1*(Q2max - Q2l)))/R1`,
/// evaluated at the last grid point, with `exp` replaced by 0 when the exponent exceeds 20. That
/// tail is what makes the last cumulative point 1.0 and it is also the assumption `GetQ2_2`'s
/// `kk == kmax-1` branch inverts.
///
/// `line_f[0]` is never written by Geant4 - the static array's zero survives - and the caller
/// only reads indices 1..ii+1, so it is zeroed here for the same reason.
template <typename real_t>
__host__ __device__ int he_fill_fq2(int a, real_t dq2, real_t q2max,
                                    const HeNucleusParams<real_t>& np,
                                    const HeHadronValues<real_t>& hv, real_t h_mass2,
                                    real_t hadr_energy, real_t* line_f) {
  real_t cur_q2 = real_t(0), cur_sec = real_t(0);
  real_t cur_sum = real_t(0), tot_sum = real_t(0);
  const real_t ddq2 = dq2 * real_t(0.1);
  real_t q2l = real_t(0);
  for (int k = 0; k < HeDims::kOnQ2; ++k) { line_f[k] = real_t(0); }

  int ii = 0;
  for (ii = 1; ii < HeDims::kOnQ2 - 1; ++ii) {
    cur_sum = cur_sec = real_t(0);
    for (int jj = 0; jj < 10; ++jj) {
      cur_q2 = q2l + (real_t(jj) + real_t(0.5)) * ddq2;
      if (cur_q2 >= q2max) { break; }
      cur_sec = he_hadr_nuc_differ_cr_sec<real_t>(a, cur_q2, q2max, np, hv, h_mass2, hadr_energy);
      cur_sum += cur_sec;
    }
    const real_t del = (cur_q2 >= q2max) ? (q2max - q2l) : dq2;
    q2l += del;
    cur_sum *= del * real_t(0.1);
    tot_sum += cur_sum;
    line_f[ii] = tot_sum;
    if (tot_sum * real_t(1.e-4) > cur_sum || q2l >= q2max) { break; }
  }
  if (ii > HeDims::kOnQ2 - 2) { ii = HeDims::kOnQ2 - 2; }
  cur_q2 = q2l;
  real_t xx = np.r1 * (q2max - cur_q2);
  if (xx > real_t(0)) {
    xx = (xx > real_t(20)) ? real_t(0) : exp(-xx);
    cur_sec = he_hadr_nuc_differ_cr_sec<real_t>(a, cur_q2, q2max, np, hv, h_mass2, hadr_energy);
    tot_sum += cur_sec * (real_t(1) - xx) / np.r1;
  }
  line_f[ii + 1] = tot_sum;
  return ii + 2;
}

/// G4ElasticData::G4ElasticData plus G4ElasticHadrNucleusHE::FillData: build the whole 24-energy
/// table for one (hadron, Z).
///
/// `a` is the rounded NIST atomic mass for Z, `mass_a_mev` is
/// G4NucleiProperties::GetNuclearMass(a, z), `h_mass_mev` is the projectile's PDG mass.
/// `is_proton_projectile` selects the `if(Z == 1 && p == G4Proton::Proton()) Q2m *= 0.5` halving
/// of maxQ2, which applies to a proton on hydrogen only - Geant4 compares the definition
/// pointer, so it is the proton and not "any nucleon".
template <typename real_t>
__host__ inline HeElasticData<real_t> he_fill_data(int z, int a, real_t mass_a_mev,
                                                   real_t h_mass_mev, int i_hadron,
                                                   int i_hadron1, int i_hadr_code,
                                                   bool is_proton_projectile,
                                                   const HeEnergyGrid<real_t>& grid,
                                                   const HeBoundary<real_t>& bnd) {
  HeElasticData<real_t> d;
  d.z = z;
  d.a = a;
  const real_t inv_gev = real_t(1) / units::GeV<real_t>();
  const real_t h_mass = h_mass_mev * inv_gev;
  const real_t h_mass2 = h_mass * h_mass;

  const HeNucleusParams<real_t> np = he_define_nucleus_parameters<real_t>(a);
  d.r1 = np.r1; d.r2 = np.r2; d.pnucl = np.pnucl; d.aeff = np.aeff;
  const real_t limit_q2 = real_t(35) / (np.r1 * np.r1);
  d.mass_a = mass_a_mev * inv_gev;
  d.mass_a2 = d.mass_a * d.mass_a;
  for (int kk = 0; kk < HeDims::kNEnergy; ++kk) {
    const real_t t = grid.energy[kk];
    const real_t elab = t + h_mass;
    const real_t plab2 = t * (t + real_t(2) * h_mass);
    real_t q2m = real_t(4) * plab2 * d.mass_a2 /
                 (h_mass2 + d.mass_a2 + real_t(2) * d.mass_a * elab);
    if (z == 1 && is_proton_projectile) { q2m *= real_t(0.5); }
    d.max_q2[kk] = q2m;
  }
  d.dq2 = limit_q2 / real_t(HeDims::kOnQ2 - 2);

  real_t line_f[HeDims::kOnQ2];
  for (int i = 0; i < HeDims::kNEnergy; ++i) {
    const real_t t = grid.energy[i];
    const real_t h_lab_momentum2 = t * (t + real_t(2) * h_mass);
    const real_t h_lab_momentum = sqrt(h_lab_momentum2);
    const real_t hadr_energy = h_mass + t;
    const HeHadronValues<real_t> hv = he_define_hadron_values<real_t>(
        z, i_hadron, i_hadron1, i_hadr_code, h_mass, h_lab_momentum, hadr_energy, bnd);
    const real_t q2max = d.max_q2[i];
    const int length =
        he_fill_fq2<real_t>(a, d.dq2, q2max, np, hv, h_mass2, hadr_energy, line_f);
    const real_t norm = real_t(1) / line_f[length - 1];
    d.len[i] = length;
    d.cum[i][0] = real_t(0);
    for (int ii = 1; ii < length - 1; ++ii) { d.cum[i][ii] = line_f[ii] * norm; }
    d.cum[i][length - 1] = real_t(1);
  }
  return d;
}

/// G4ElasticHadrNucleusHE::GetQ2_2 - invert the cumulative array at `ran_uni`.
///
/// Three branches. The last bin (`kk == kmax-1`) inverts the analytic exponential tail that
/// FillFq2 appended. Otherwise a quadratic is fitted through three (F, X) points - the two before
/// kk and kk itself, or the first three when kk is 0 or 1 - and evaluated at ran_uni; if the
/// quadratic's determinant `D0` is below 1e-9 it degenerates to a straight line between the last
/// two points. Note the quadratic is X as a function of F, i.e. it interpolates the INVERSE CDF,
/// so no root finding is needed.
template <typename real_t>
__host__ __device__ real_t he_get_q2_2(int kk, int kmax, const real_t* f, real_t ran_uni,
                                       real_t dq2, real_t q2max, real_t r1) {
  if (kk == kmax - 1) {
    const real_t x1 = dq2 * real_t(kk);
    const real_t f1 = f[kk - 1];
    const real_t x2 = q2max;
    real_t xx = r1 * (x2 - x1);
    xx = (xx > real_t(20)) ? real_t(0) : exp(-xx);
    const real_t y =
        x1 - log(real_t(1) - (ran_uni - f1) * (real_t(1) - xx) / (real_t(1) - f1)) / r1;
    return y;
  }
  real_t f1, f2, f3, x1, x2, x3;
  if (kk == 1 || kk == 0) {
    f1 = f[0]; f2 = f[1]; f3 = f[2];
    x1 = real_t(0); x2 = dq2; x3 = dq2 * real_t(2);
  } else {
    f1 = f[kk - 2]; f2 = f[kk - 1]; f3 = f[kk];
    x1 = dq2 * real_t(kk - 2); x2 = dq2 * real_t(kk - 1); x3 = dq2 * real_t(kk);
  }
  const real_t f12 = f1 * f1, f22 = f2 * f2, f32 = f3 * f3;
  const real_t d0 = f12 * f2 + f1 * f32 + f3 * f22 - f32 * f2 - f22 * f1 - f12 * f3;
  const real_t ad0 = (d0 > real_t(0)) ? d0 : -d0;
  if (ad0 < real_t(1.e-9)) { return x2 + (ran_uni - f2) * (x3 - x2) / (f3 - f2); }
  const real_t da = x1 * f2 + x3 * f1 + x2 * f3 - x3 * f2 - x1 * f3 - x2 * f1;
  const real_t db = x2 * f12 + x1 * f32 + x3 * f22 - x2 * f32 - x3 * f12 - x1 * f22;
  const real_t dc = x3 * f2 * f12 + x2 * f1 * f32 + x1 * f3 * f22 - x1 * f2 * f32 -
                    x2 * f3 * f12 - x3 * f1 * f22;
  return (da * ran_uni * ran_uni + db * ran_uni + dc) / d0;
}

/// G4ElasticHadrNucleusHE::HadronNucleusQ2_2 - sample Q2 in GeV^2 from a built table.
///
/// The energy bin is the first `idx` in 0..NENERGY-2 with `ekin <= fLowEdgeEnergy[idx+1]`, and
/// NENERGY-1 if none matches - so the top bin covers everything above the last low edge, up to
/// the model's 100 TeV maximum, with the 10^4.2 GeV table. That is the whole high-energy
/// behaviour of the model: it stops resolving energy at about 16 TeV.
///
/// The sampled Q2 is then RESCALED, `Q2 *= tmax/Q2max`, from the table's kinematic limit (built
/// at the bin's node energy and the element's rounded A) to the actual kinematic limit of this
/// track and isotope. So the isotope and the exact energy enter only through that ratio.
///
/// One uniform is consumed.
template <typename real_t, typename Rng>
__host__ __device__ real_t he_hadron_nucleus_q2_2(const HeElasticData<real_t>& d, real_t h_mass,
                                                  real_t plab, real_t tmax,
                                                  const HeEnergyGrid<real_t>& grid, Rng& rng) {
  const real_t h_mass2 = h_mass * h_mass;
  const real_t ekin = sqrt(h_mass2 + plab * plab) - h_mass;
  int idx = 0;
  for (idx = 0; idx < HeDims::kNEnergy - 1; ++idx) {
    if (ekin <= grid.low_edge[idx + 1]) { break; }
  }
  const real_t r1 = d.r1;
  const real_t dq2 = d.dq2;
  const real_t q2max = d.max_q2[idx];
  const int length = d.len[idx];
  const real_t rand = rng.uniform();

  int i_numb_q2 = 0;
  for (i_numb_q2 = 1; i_numb_q2 < length; ++i_numb_q2) {
    if (rand <= d.cum[idx][i_numb_q2]) { break; }
  }
  if (i_numb_q2 > length - 1) { i_numb_q2 = length - 1; }
  real_t q2 = he_get_q2_2<real_t>(i_numb_q2, length, d.cum[idx], rand, dq2, q2max, r1);
  if (q2 > q2max) { q2 = q2max; }
  q2 *= tmax / q2max;
  return q2;
}

/// G4ElasticHadrNucleusHE::GetFt - the cumulative t-distribution for a free-proton target.
template <typename real_t>
__host__ __device__ real_t he_get_ft(real_t q2, const HeHadronValues<real_t>& hv,
                                     real_t const_u) {
  const real_t sqr_q2 = sqrt(q2);
  return (real_t(1) - hv.coeff1 - hv.coeff0) / hv.hadr_slope *
             (real_t(1) - exp(-hv.hadr_slope * q2)) +
         hv.coeff0 * (real_t(1) - exp(-hv.slope0 * q2)) +
         hv.coeff2 / hv.slope2 * exp(hv.slope2 * const_u) * (exp(hv.slope2 * q2) - real_t(1)) +
         real_t(2) * hv.coeff1 / hv.slope1 *
             (real_t(1) / hv.slope1 -
              (real_t(1) / hv.slope1 + sqr_q2) * exp(-hv.slope1 * sqr_q2));
}

/// G4ElasticHadrNucleusHE::HadronProtonQ2 - sample Q2 in GeV^2 off a free proton by bisecting
/// `GetFt`.
///
/// The upper limit is `MaxTR = (plab < BoundaryP[iHadron1]) ? BoundaryTL[iHadron1]
///                                                          : BoundaryTG[iHadron1]`,
/// with BoundaryTL[0], [1], [3], [4] and [5] having just been overwritten with tmax. For pi+
/// iHadron1 is 3 and for pi- it is 4, so both use `plab < 8` (pi+) or `plab < 7` (pi-) to choose
/// between tmax and 3 GeV^2 - i.e. above that momentum the sampling is CUT at 3 GeV^2 of momentum
/// transfer regardless of what the kinematics allow.
///
/// The bisection: DDD0 starts at MaxTR/2 with the bracket [0, MaxTR], and steps until
/// `|GetFt(DDD0)/GetFt(MaxTR) - rand| <= 1e-4` or 10000 iterations, at which point it returns
/// **0.0** - not the last value. That failure return is reproduced. Exactly one uniform is drawn,
/// so the stream is unaffected by how many bisection steps were needed.
template <typename real_t, typename Rng>
__host__ __device__ real_t he_hadron_proton_q2(real_t plab, real_t tmax, real_t h_mass,
                                              int i_hadron, int i_hadron1, int i_hadr_code,
                                              const HeBoundary<real_t>& bnd_in, Rng& rng) {
  const real_t proton_m = units::proton_mass_c2<real_t>() * real_t(1e-3);
  const real_t proton_m2 = proton_m * proton_m;
  const real_t h_mass2 = h_mass * h_mass;
  const real_t hadr_energy = sqrt(h_mass2 + plab * plab);
  const HeHadronValues<real_t> hv = he_define_hadron_values<real_t>(
      1, i_hadron, i_hadron1, i_hadr_code, h_mass, plab, hadr_energy, bnd_in);

  const real_t sh = real_t(2) * proton_m * hadr_energy + proton_m2 + h_mass2;
  const real_t const_u = real_t(2) * proton_m2 + real_t(2) * h_mass2 - sh;

  HeBoundary<real_t> bnd = bnd_in;
  bnd.tl[0] = tmax; bnd.tl[1] = tmax; bnd.tl[3] = tmax; bnd.tl[4] = tmax; bnd.tl[5] = tmax;
  const real_t max_tr = (plab < bnd.p[i_hadron1]) ? bnd.tl[i_hadron1] : bnd.tg[i_hadron1];

  const real_t rand = rng.uniform();
  real_t d0 = max_tr * real_t(0.5), d1 = real_t(0), d2 = max_tr;
  const real_t norm = real_t(1) / he_get_ft<real_t>(max_tr, hv, const_u);
  real_t delta = he_get_ft<real_t>(d0, hv, const_u) * norm - rand;

  const int max_loops = 10000;
  int loop_counter = -1;
  real_t adelta = (delta > real_t(0)) ? delta : -delta;
  while (adelta > real_t(0.0001) && ++loop_counter < max_loops) {
    if (delta > real_t(0)) { d2 = d0; d0 = (d0 + d1) * real_t(0.5); }
    else if (delta < real_t(0)) { d1 = d0; d0 = (d0 + d2) * real_t(0.5); }
    delta = he_get_ft<real_t>(d0, hv, const_u) * norm - rand;
    adelta = (delta > real_t(0)) ? delta : -delta;
  }
  return (loop_counter >= max_loops) ? real_t(0) : d0;
}

/// G4ElasticHadrNucleusHE::SampleInvariantT. Returns -t in MeV^2.
///
/// Order of the guards, which is the order of the source:
///   1. `kine = sqrt(p^2 + m^2) - m <= 400 MeV` -> G4HadronElastic::SampleInvariantT. The caller
///      is told through `*used_fallback` and must do the fall-back itself, so that the random
///      stream is the caller's.
///   2. Z is clamped to ZMAX-1 = 92, so a Z above 92 uses uranium's table. That is Geant4's
///      `std::min(iZ, ZMAX-1)`, not a refusal, and it is reproduced.
///   3. the PDG code is looked up in fHadronCode; if it is not there the function returns
///      **0.0**, a forward scatter with no fall-back. `*unknown_hadron` says so.
///   4. Z == 1 -> HadronProtonQ2; otherwise the table.
///
/// `tmax` is `pLocalTmax * invGeV2`, i.e. the caller's 4*pcms^2 in GeV^2.
template <typename real_t, typename Rng>
__host__ __device__ real_t he_sample_invariant_t(int pdg, real_t plab_mev, int iz, int /*a*/,
                                                 real_t h_mass_mev, real_t p_local_tmax,
                                                 const HeElasticData<real_t>* table,
                                                 const HeEnergyGrid<real_t>& grid,
                                                 const HeBoundary<real_t>& bnd, Rng& rng,
                                                 bool* used_fallback, bool* unknown_hadron,
                                                 int* hadron_index) {
  if (used_fallback) { *used_fallback = false; }
  if (unknown_hadron) { *unknown_hadron = false; }
  if (hadron_index) { *hadron_index = -1; }

  const real_t kine = sqrt(plab_mev * plab_mev + h_mass_mev * h_mass_mev) - h_mass_mev;
  if (kine <= he_ekin_low_limit<real_t>()) {
    if (used_fallback) { *used_fallback = true; }
    return real_t(0);
  }
  const int z = (iz < HeDims::kZMax - 1) ? iz : HeDims::kZMax - 1;

  int idx = -1, i_hadron = -1, i_hadron1 = -1;
  const int* codes = he_hadron_code();
  for (int i = 0; i < HeDims::kNHadrons; ++i) {
    if (pdg == codes[i]) {
      idx = i;
      i_hadron = he_hadron_type()[i];
      i_hadron1 = he_hadron_type1()[i];
      break;
    }
  }
  if (i_hadron < 0) {
    if (unknown_hadron) { *unknown_hadron = true; }
    return real_t(0);
  }
  if (hadron_index) { *hadron_index = idx; }

  const real_t inv_gev = real_t(1) / units::GeV<real_t>();
  const real_t gev2 = units::GeV<real_t>() * units::GeV<real_t>();
  const real_t h_mass = h_mass_mev * inv_gev;
  const real_t h_mass2 = h_mass * h_mass;
  const real_t plab = plab_mev * inv_gev;
  const real_t tmax = p_local_tmax / gev2;

  real_t q2;
  if (z == 1) {
    q2 = he_hadron_proton_q2<real_t>(plab, tmax, h_mass, i_hadron, i_hadron1, pdg, bnd, rng);
  } else {
    q2 = he_hadron_nucleus_q2_2<real_t>(*table, h_mass, plab, tmax, grid, rng);
  }
  (void)h_mass2;
  return q2 * gev2;
}

}  // namespace g4gpu::physics::hadronic::elastic
