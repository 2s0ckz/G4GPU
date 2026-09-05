// Shell correction to the Bethe-Bloch stopping power, transcribed from
// G4EmCorrections::ShellCorrection / KShell / LShell (11.1.1).
//
// This is the term that dominates the error in the plain Bethe-Bloch formula below about
// 10 MeV per nucleon: without it the port was 10% off there.
//
// The correction is built per element from K, L, M and N shell contributions, each an
// interpolation over a two-dimensional (theta, eta) table. The tables themselves live in
// data/em_correction_tables.cuh and are extracted from the Geant4 source programmatically;
// the CK and CL grids are not tabulated in Geant4 either but assembled at initialisation
// from SK/TK/SL/TL and the bk/bls/bll tables, which is what build_shell_tables does here.
#pragma once
#include <cmath>
#include "core/particle.cuh"
#include "core/units.cuh"
#include "data/em_correction_tables.cuh"
#include "data/materials.cuh"

namespace g4gpu::em {

/// theta grids KShell and LShell interpolate over.
template <typename real_t> __host__ __device__ inline const real_t* the_k() {
  static const real_t v[20] = {real_t(0.64), real_t(0.65), real_t(0.66), real_t(0.68),
                               real_t(0.70), real_t(0.72), real_t(0.74), real_t(0.75),
                               real_t(0.76), real_t(0.78), real_t(0.80), real_t(0.82),
                               real_t(0.84), real_t(0.85), real_t(0.86), real_t(0.88),
                               real_t(0.90), real_t(0.92), real_t(0.94), real_t(0.95)};
  return v;
}
template <typename real_t> __host__ __device__ inline const real_t* the_l() {
  static const real_t v[26] = {real_t(0.24), real_t(0.26), real_t(0.28), real_t(0.30),
                               real_t(0.32), real_t(0.34), real_t(0.35), real_t(0.36),
                               real_t(0.38), real_t(0.40), real_t(0.42), real_t(0.44),
                               real_t(0.45), real_t(0.46), real_t(0.48), real_t(0.50),
                               real_t(0.52), real_t(0.54), real_t(0.55), real_t(0.56),
                               real_t(0.58), real_t(0.60), real_t(0.62), real_t(0.64),
                               real_t(0.65), real_t(0.66)};
  return v;
}

/// The CK / CL grids and the ZK / VL edge coefficients, assembled exactly as
/// G4EmCorrections::Initialise does.
template <typename real_t>
struct ShellTables {
  real_t ck[data::emcorr::kNK][data::emcorr::kNEtaK];
  real_t cl[data::emcorr::kNL][data::emcorr::kNEtaL];
  real_t zk[data::emcorr::kNK];
  real_t vl[data::emcorr::kNL];
};

template <typename real_t>
__host__ __device__ inline void build_shell_tables(ShellTables<real_t>& t) {
  using namespace data::emcorr;
  const real_t* Eta = eta<real_t>();
  const real_t* SK = sk<real_t>();
  const real_t* TK = tk<real_t>();
  const real_t* SL = sl<real_t>();
  const real_t* TL = tl<real_t>();
  const real_t* UK = uk<real_t>();
  const real_t* VK = vk<real_t>();
  const real_t* UL = ul<real_t>();
  const real_t* BK1 = bk1<real_t>();
  const real_t* BK2 = bk2<real_t>();
  const real_t* BLS1 = bls1<real_t>();
  const real_t* BLS2 = bls2<real_t>();
  const real_t* BLS3 = bls3<real_t>();
  const real_t* BLL1 = bll1<real_t>();
  const real_t* BLL2 = bll2<real_t>();
  const real_t* BLL3 = bll3<real_t>();

  for (int i = 0; i < kNEtaK; ++i) {
    const real_t et = Eta[i];
    const real_t loget = log(et);
    for (int j = 0; j < kNK; ++j) {
      const real_t b = (j < 10) ? BK2[i * 11 + (10 - j)] : BK1[i * 11 + (20 - j)];
      t.ck[j][i] = SK[j] * loget + TK[j] - b;
      if (i == kNEtaK - 1) {
        t.zk[j] = et * (et * et * t.ck[j][i] - et * UK[j] - VK[j]);
      }
    }
    if (i < kNEtaL) {
      for (int j = 0; j < kNL; ++j) {
        real_t bs, b;
        if (j < 8) {
          bs = BLS3[i * 9 + (8 - j)];
          b = BLL3[i * 9 + (8 - j)];
        } else if (j < 17) {
          bs = BLS2[i * 10 + (17 - j)];
          b = BLL2[i * 10 + (17 - j)];
        } else {
          bs = BLS1[i * 10 + (26 - j)];
          b = BLL1[i * 10 + (26 - j)];
        }
        const real_t c = SL[j] * loget + TL[j];
        t.cl[j][i] = c - bs - real_t(3) * b;
        if (i == kNEtaL - 1) { t.vl[j] = et * (et * t.cl[j][i] - UL[j]); }
      }
    }
  }
}

template <typename real_t>
__host__ __device__ inline int corr_index(real_t x, const real_t* v, int n) {
  int idx = n - 2;
  for (int i = 1; i < n - 1; ++i) {
    if (x <= v[i]) { idx = i - 1; break; }
  }
  return idx;
}
template <typename real_t>
__host__ __device__ inline real_t corr_value(real_t xv, real_t x1, real_t x2, real_t y1,
                                             real_t y2) {
  return y1 + (y2 - y1) * (xv - x1) / (x2 - x1);
}
template <typename real_t>
__host__ __device__ inline real_t corr_value2(real_t xv, real_t yv, real_t x1, real_t x2,
                                              real_t y1, real_t y2, real_t z11, real_t z21,
                                              real_t z12, real_t z22) {
  return (z11 * (x2 - xv) * (y2 - yv) + z22 * (xv - x1) * (yv - y1)
          + z12 * (x2 - xv) * (yv - y1) + z21 * (xv - x1) * (y2 - yv))
         / ((x2 - x1) * (y2 - y1));
}

/// Verbatim from G4EmCorrections::KShell.
template <typename real_t>
__host__ __device__ inline real_t k_shell(const ShellTables<real_t>& t, real_t tet,
                                          real_t eta_in) {
  using namespace data::emcorr;
  const real_t* TheK = the_k<real_t>();
  const real_t* Eta = eta<real_t>();
  real_t x = tet;
  int itet = 0;
  if (tet < TheK[0]) {
    x = TheK[0];
  } else if (tet > TheK[kNK - 1]) {
    x = TheK[kNK - 1];
    itet = kNK - 2;
  } else {
    itet = corr_index(x, TheK, kNK);
  }
  if (eta_in >= Eta[kNEtaK - 1]) {
    const real_t* UK = uk<real_t>();
    const real_t* VK = vk<real_t>();
    return (corr_value(x, TheK[itet], TheK[itet + 1], UK[itet], UK[itet + 1])
            + corr_value(x, TheK[itet], TheK[itet + 1], VK[itet], VK[itet + 1]) / eta_in
            + corr_value(x, TheK[itet], TheK[itet + 1], t.zk[itet], t.zk[itet + 1])
                  / (eta_in * eta_in))
           / eta_in;
  }
  real_t y = eta_in;
  int ieta = 0;
  if (eta_in < Eta[0]) { y = Eta[0]; } else { ieta = corr_index(y, Eta, kNEtaK); }
  return corr_value2(x, y, TheK[itet], TheK[itet + 1], Eta[ieta], Eta[ieta + 1],
                     t.ck[itet][ieta], t.ck[itet + 1][ieta], t.ck[itet][ieta + 1],
                     t.ck[itet + 1][ieta + 1]);
}

/// Verbatim from G4EmCorrections::LShell.
template <typename real_t>
__host__ __device__ inline real_t l_shell(const ShellTables<real_t>& t, real_t tet,
                                          real_t eta_in) {
  using namespace data::emcorr;
  const real_t* TheL = the_l<real_t>();
  const real_t* Eta = eta<real_t>();
  const real_t* UL = ul<real_t>();
  real_t x = tet;
  int itet = 0;
  if (tet < TheL[0]) {
    x = TheL[0];
  } else if (tet > TheL[kNL - 1]) {
    x = TheL[kNL - 1];
    itet = kNL - 2;
  } else {
    itet = corr_index(x, TheL, kNL);
  }
  if (eta_in >= Eta[kNEtaL - 1]) {
    return (corr_value(x, TheL[itet], TheL[itet + 1], UL[itet], UL[itet + 1])
            + corr_value(x, TheL[itet], TheL[itet + 1], t.vl[itet], t.vl[itet + 1]) / eta_in)
           / eta_in;
  }
  real_t y = eta_in;
  int ieta = 0;
  if (eta_in < Eta[0]) { y = Eta[0]; } else { ieta = corr_index(y, Eta, kNEtaL); }
  return corr_value2(x, y, TheL[itet], TheL[itet + 1], Eta[ieta], Eta[ieta + 1],
                     t.cl[itet][ieta], t.cl[itet + 1][ieta], t.cl[itet][ieta + 1],
                     t.cl[itet + 1][ieta + 1]);
}

/// Interpolates the tabulated theta_K / theta_L against Z.
template <typename real_t>
__host__ __device__ inline real_t theta_interp(const real_t* xs, const real_t* ys, int n,
                                               real_t z) {
  if (z <= xs[0]) { return ys[0]; }
  if (z >= xs[n - 1]) { return ys[n - 1]; }
  for (int i = 1; i < n; ++i) {
    if (z <= xs[i]) {
      return ys[i - 1] + (ys[i] - ys[i - 1]) * (z - xs[i - 1]) / (xs[i] - xs[i - 1]);
    }
  }
  return ys[n - 1];
}

/// Shell correction for one material, verbatim from G4EmCorrections::ShellCorrection.
/// The result is subtracted twice from the Bethe-Bloch bracket.
template <typename real_t>
__host__ __device__ inline real_t shell_correction(const ShellTables<real_t>& t,
                                                   const data::Material<real_t>& m,
                                                   const ParticleDef<real_t>& pd,
                                                   real_t kinetic) {
  using namespace data::emcorr;
  constexpr real_t alpha2 = units::fine_structure_const<real_t>() * units::fine_structure_const<real_t>();
  const real_t tau = kinetic / pd.mass;
  const real_t gam = tau + real_t(1);
  const real_t bg2 = tau * (tau + real_t(2));
  const real_t beta2 = bg2 / (gam * gam);
  const real_t ba2 = beta2 / alpha2;

  static const real_t HM[53] = {
      real_t(12.0), real_t(12.0), real_t(12.0), real_t(12.0), real_t(11.9), real_t(11.7),
      real_t(11.5), real_t(11.2), real_t(10.8), real_t(10.4), real_t(10.0), real_t(9.51),
      real_t(8.97), real_t(8.52), real_t(8.03), real_t(7.46), real_t(6.95), real_t(6.53),
      real_t(6.18), real_t(5.87), real_t(5.61), real_t(5.39), real_t(5.19), real_t(5.01),
      real_t(4.86), real_t(4.72), real_t(4.62), real_t(4.53), real_t(4.44), real_t(4.38),
      real_t(4.32), real_t(4.26), real_t(4.20), real_t(4.15), real_t(4.1),  real_t(4.04),
      real_t(4.00), real_t(3.95), real_t(3.93), real_t(3.91), real_t(3.90), real_t(3.89),
      real_t(3.89), real_t(3.88), real_t(3.88), real_t(3.88), real_t(3.88), real_t(3.88),
      real_t(3.89), real_t(3.89), real_t(3.90), real_t(3.92), real_t(3.93)};
  static const real_t HN[31] = {
      real_t(75.5), real_t(61.9), real_t(52.2), real_t(45.1), real_t(39.6), real_t(35.4),
      real_t(31.9), real_t(29.1), real_t(27.2), real_t(25.8), real_t(24.5), real_t(23.6),
      real_t(22.7), real_t(22.0), real_t(21.4), real_t(20.9), real_t(20.5), real_t(20.2),
      real_t(19.9), real_t(19.7), real_t(19.5), real_t(19.3), real_t(19.2), real_t(19.1),
      real_t(18.4), real_t(18.8), real_t(18.7), real_t(18.6), real_t(18.5), real_t(18.4),
      real_t(18.2)};

  const real_t* ZD = zd<real_t>();
  const int* nsh = n_shells();
  const int* ish = shell_index();
  const int* nel = n_electrons();

  real_t term = real_t(0);
  real_t total_atoms = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) { total_atoms += m.n_atoms[i]; }

  for (int i = 0; i < m.n_elements; ++i) {
    const real_t Z = m.z[i];
    const int iz = static_cast<int>(Z + real_t(0.5));
    real_t Z2 = (Z - real_t(0.3)) * (Z - real_t(0.3));
    real_t f = real_t(1);
    if (iz == 1) {
      f = real_t(0.5);
      Z2 = real_t(1);
    }
    real_t eta_v = ba2 / Z2;
    real_t tet = Z2 * (real_t(1) + Z2 * real_t(0.25) * alpha2);
    if (iz > 11) { tet = theta_interp(xzk<real_t>(), yzk<real_t>(), 34, Z); }
    real_t res = f * k_shell(t, tet, eta_v);

    if (iz > 2) {
      const real_t Zeff = (iz < 10) ? Z - ZD[iz] : Z - ZD[10];
      Z2 = Zeff * Zeff;
      eta_v = ba2 / Z2;
      f = real_t(0.125);
      tet = theta_interp(xzl<real_t>(), yzl<real_t>(), 36, Z);
      const int ntot = (iz >= 0 && iz <= 104) ? nsh[iz] : 0;
      const int nmax = (ntot < 4) ? ntot : 4;
      real_t norm = real_t(0), eshell = real_t(0);
      for (int j = 1; j < nmax; ++j) {
        const int ne = nel[ish[iz] + j];
        if (iz <= 15) {
          tet = (j < 3) ? real_t(0.25) * Z2 * (real_t(1) + real_t(5) * Z2 * alpha2 / real_t(16))
                        : real_t(0.25) * Z2 * (real_t(1) + Z2 * alpha2 / real_t(16));
        }
        norm += real_t(ne);
        eshell += tet * real_t(ne);
        res += f * real_t(ne) * l_shell(t, tet, eta_v);
      }
      if (ntot > nmax && norm > real_t(0)) {
        eshell /= norm;
        if (iz < 28) {
          res += f * real_t(iz - 10) * l_shell(t, eshell, HM[iz - 11] * eta_v);
        } else {
          res += f * real_t(18) * l_shell(t, eshell, HM[52] * eta_v);
        }
        if (iz > 32) {
          if (iz < 60) {
            res += f * real_t(iz - 28) * l_shell(t, eshell, HN[iz - 33] * eta_v);
          } else if (iz < 63) {
            res += real_t(4) * l_shell(t, eshell, HN[iz - 33] * eta_v);
          } else {
            res += real_t(4) * l_shell(t, eshell, HN[30] * eta_v);
          }
          if (iz > 60) {
            res += f * real_t(iz - 60) * l_shell(t, eshell, real_t(150) * eta_v);
          }
        }
      }
    }
    term += res * m.n_atoms[i] / Z;
  }
  return (total_atoms > real_t(0)) ? term / total_atoms : real_t(0);
}


/// Barkas correction interpolation table, from G4EmCorrections::Initialise fTable[47][2].
template <typename real_t> __host__ __device__ inline const real_t* barkas_w() {
  static const real_t v[47] = {
      real_t(0.02), real_t(0.03), real_t(0.04), real_t(0.05), real_t(0.06), real_t(0.07)
      , real_t(0.08), real_t(0.09), real_t(0.1), real_t(0.2), real_t(0.3), real_t(0.4)
      , real_t(0.5), real_t(0.6), real_t(0.7), real_t(0.8), real_t(0.9), real_t(1.0)
      , real_t(1.2), real_t(1.3), real_t(1.4), real_t(1.5), real_t(1.6), real_t(1.7)
      , real_t(1.8), real_t(1.9), real_t(2.0), real_t(2.1), real_t(2.4), real_t(3.0)
      , real_t(3.08), real_t(3.1), real_t(3.3), real_t(3.5), real_t(3.8), real_t(4.0)
      , real_t(4.1), real_t(4.8), real_t(5.0), real_t(5.1), real_t(6.0), real_t(6.5)
      , real_t(7.0), real_t(7.1), real_t(8.0), real_t(9.0), real_t(10.0)};
  return v;
}
template <typename real_t> __host__ __device__ inline const real_t* barkas_v() {
  static const real_t v[47] = {
      real_t(21.5), real_t(20.0), real_t(18.0), real_t(15.6), real_t(15.0), real_t(14.0)
      , real_t(13.5), real_t(13.), real_t(12.2), real_t(9.25), real_t(7.0), real_t(6.0)
      , real_t(4.5), real_t(3.5), real_t(3.0), real_t(2.5), real_t(2.0), real_t(1.7)
      , real_t(1.2), real_t(1.0), real_t(0.86), real_t(0.7), real_t(0.61), real_t(0.52)
      , real_t(0.5), real_t(0.43), real_t(0.42), real_t(0.3), real_t(0.2), real_t(0.13)
      , real_t(0.1), real_t(0.09), real_t(0.08), real_t(0.07), real_t(0.06), real_t(0.051)
      , real_t(0.04), real_t(0.03), real_t(0.024), real_t(0.02), real_t(0.013), real_t(0.01)
      , real_t(0.009), real_t(0.008), real_t(0.006), real_t(0.0032), real_t(0.0025)};
  return v;
}

/// Verbatim from G4EmCorrections::BarkasCorrection. Odd in the charge, which is why it
/// matters most for antiprotons and why omitting it made them worse than omitting nothing.
template <typename real_t>
__host__ __device__ inline real_t barkas_correction(const data::Material<real_t>& m,
                                                    const ParticleDef<real_t>& pd,
                                                    real_t kinetic) {
  constexpr real_t alpha2 = units::fine_structure_const<real_t>() * units::fine_structure_const<real_t>();
  const real_t tau = kinetic / pd.mass;
  const real_t gam = tau + real_t(1);
  const real_t bg2 = tau * (tau + real_t(2));
  const real_t beta2 = bg2 / (gam * gam);
  const real_t beta = sqrt(beta2);
  const real_t ba2 = beta2 / alpha2;
  const real_t* bw = barkas_w<real_t>();
  const real_t* bv = barkas_v<real_t>();

  real_t term = real_t(0), total_atoms = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) { total_atoms += m.n_atoms[i]; }
  for (int i = 0; i < m.n_elements; ++i) {
    const real_t Z = m.z[i];
    const int iz = static_cast<int>(Z + real_t(0.5));
    if (iz == 47) {
      term += m.n_atoms[i] * real_t(0.006812) * exp(-log(beta) * real_t(0.9));
    } else if (iz >= 64) {
      term += m.n_atoms[i] * real_t(0.002833) * exp(-log(beta) * real_t(1.2));
    } else {
      const real_t X = ba2 / Z;
      real_t b = real_t(1.3);
      // G4_lH2 gets b = 0.6; no material here is liquid hydrogen, so the 1.8 branch stands.
      if (iz == 1)       { b = real_t(1.8); }
      else if (iz == 2)  { b = real_t(0.6); }
      else if (iz <= 10) { b = real_t(1.8); }
      else if (iz <= 17) { b = real_t(1.4); }
      else if (iz == 18) { b = real_t(1.8); }
      else if (iz <= 25) { b = real_t(1.4); }
      else if (iz <= 50) { b = real_t(1.35); }
      const real_t W = b / sqrt(X);
      real_t val = theta_interp(bw, bv, 47, W);
      if (W > bw[46]) { val *= bw[46] / W; }
      term += val * m.n_atoms[i] / (sqrt(Z * X) * X);
    }
  }
  return (total_atoms > real_t(0))
             ? term * real_t(1.29) * pd.charge / total_atoms
             : real_t(0);
}

/// Verbatim from G4EmCorrections::BlochCorrection.
template <typename real_t>
__host__ __device__ inline real_t bloch_correction(const ParticleDef<real_t>& pd,
                                                   real_t kinetic) {
  constexpr real_t alpha2 = units::fine_structure_const<real_t>() * units::fine_structure_const<real_t>();
  const real_t tau = kinetic / pd.mass;
  const real_t gam = tau + real_t(1);
  const real_t bg2 = tau * (tau + real_t(2));
  const real_t beta2 = bg2 / (gam * gam);
  const real_t ba2 = beta2 / alpha2;
  const real_t y2 = pd.charge * pd.charge / ba2;
  real_t term = real_t(1) / (real_t(1) + y2);
  real_t del, j = real_t(1);
  for (int n = 0; n < 10000; ++n) {
    j += real_t(1);
    del = real_t(1) / (j * (j * j + y2));
    term += del;
    if (del <= real_t(0.01) * term) { break; }
  }
  return -y2 * term;
}

/// Verbatim from G4EmCorrections::MottCorrection.
template <typename real_t>
__host__ __device__ inline real_t mott_correction(const ParticleDef<real_t>& pd,
                                                  real_t kinetic) {
  constexpr real_t alpha = units::fine_structure_const<real_t>();
  const real_t tau = kinetic / pd.mass;
  const real_t gam = tau + real_t(1);
  const real_t bg2 = tau * (tau + real_t(2));
  const real_t beta = sqrt(bg2 / (gam * gam));
  return real_t(3.14159265358979323846) * alpha * beta * pd.charge;
}


/// Effective charge of an ion, and the charge correction that goes with it.
///
/// Transcribed from `G4ionEffectiveCharge::EffectiveCharge` (11.1.1) in full - both branches.
///
/// Geant4 substitutes this for the nominal charge in the Bethe-Bloch *correction* terms
/// (`G4EmCorrections::SetupKinematics` does so whenever |q| > 1.5); the main Bethe-Bloch term
/// keeps the nominal charge, which `G4BetheBlochModel::SetupParameters` sets from the PDG
/// value.
///
/// Three cases, in Geant4's order:
///
///   Zi <= 1    the nominal charge, unchanged. Protons, pions, kaons, muons - and every
///              negative particle, because Zi is the *signed* charge in eplus, not its
///              magnitude. An anti-alpha takes this branch in Geant4, so it takes it here.
///   Zi <= 2    the six-coefficient helium fit. Alpha and He3, which are the only ions
///              G4EmStandardPhysics registers by name.
///   Zi > 2     Ziegler's heavy-ion form, which needs the material's Fermi energy - the
///              reason this branch went untranscribed until Material grew one.
///
/// 11.5.0 writes the first two gates as `effCharge <= 1.5` and `Zi == 2`; for an integer ion
/// charge those are the same tests, and 11.1.1's form is used here because 11.1.1 is what the
/// oracle links (docs/RISK.md O7).
///
/// @param[out] charge_correction  Geant4's second output, which is 1 for everything but the
///                                heavy-ion branch.
///
/// Where the two outputs go in Geant4, which is not symmetric:
///
///   - the **charge** is what `G4EmCorrections::SetupKinematics` substitutes for the nominal
///     one (`if (charge > 1.5) charge = effCharge.EffectiveCharge(...)`), giving the `q2` that
///     scales the whole high-order correction;
///   - the **correction** is used only through `EffectiveChargeSquareRatio`, i.e. as
///     `(charge * correction)^2`, and only inside `G4EmCorrections::BuildCorrectionVector` -
///     the ICRU73/experimental-stopping ratio table for named ions.
///
/// The port has no consumer for the correction yet, because it has no ion transport and so no
/// correction vector to build. It is transcribed and measured anyway: the alternative is to
/// write it later from a formula nothing has ever checked, and `tests/test_ion_charge.cu`
/// compares it against Geant4's own `EffectiveChargeSquareRatio` today.
template <typename real_t>
__host__ __device__ inline real_t ion_effective_charge(const data::Material<real_t>& m,
                                                       const ParticleDef<real_t>& pd,
                                                       real_t kinetic,
                                                       real_t& charge_correction) {
  charge_correction = real_t(1);
  real_t eff = pd.charge;
  // G4lrint(effCharge*inveplus): the signed charge in eplus, rounded. Not |charge|.
  const int Zi = static_cast<int>(floor(eff + real_t(0.5)));
  if (Zi <= 1) { return eff; }

  // CLHEP 11.1.1's values, from units.cuh - see the note there on why they are pinned.
  constexpr real_t kProtonMass = units::proton_mass_c2<real_t>();
  constexpr real_t kEnergyHighLimit = real_t(20.0);  // 20 MeV
  constexpr real_t kEnergyLowLimit = real_t(1e-3);   // 1 keV
  constexpr real_t kEnergyBohr = real_t(25e-3);      // 25 keV
  // amu_c2 / (proton_mass_c2 * keV): the argument of the helium branch's logarithm is a
  // reduced energy in those units, and getting the factor wrong shifts the whole curve.
  constexpr real_t kMassFactor = units::amu_c2<real_t>() / (kProtonMass * real_t(1e-3));
  constexpr real_t kMinCharge = real_t(1.0);

  real_t reduced = kinetic * kProtonMass / pd.mass;
  if (reduced > eff * kEnergyHighLimit) { return eff; }
  const real_t z = m.z_eff;
  reduced = fmax(reduced, kEnergyLowLimit);

  if (Zi <= 2) {
    constexpr real_t c[6] = {real_t(0.2865),  real_t(0.1266),   real_t(-0.001429),
                             real_t(0.02402), real_t(-0.01135), real_t(0.001475)};
    const real_t Q = fmax(real_t(0), log(reduced * kMassFactor));
    real_t x = c[0], y = real_t(1);
    for (int i = 1; i < 6; ++i) {
      y *= Q;
      x += y * c[i];
    }
    // Series expansions where the closed form would lose precision; Geant4 does the same.
    const real_t ex =
        (x < real_t(0.2)) ? x * (real_t(1) - real_t(0.5) * x) : real_t(1) - exp(-x);
    const real_t tq = real_t(7.6) - Q;
    const real_t tq2 = tq * tq;
    real_t tt = (real_t(0.007) + real_t(0.00005) * z);
    if (tq2 < real_t(0.2)) {
      tt *= (real_t(1) - tq2 + real_t(0.5) * tq2 * tq2);
    } else {
      tt *= exp(-tq2);
    }
    return eff * (real_t(1) + tt) * sqrt(ex);
  }

  // Heavy ion. Ziegler, Biersack and Littmark, "The Stopping and Range of Ions in Matter"
  // vol. 1 (1985), with the screening length from Ziegler and Manoyan, NIM B35 (1988) 215.
  const real_t zi13 = data::g4pow_z13<real_t>(Zi);
  const real_t zi23 = zi13 * zi13;

  // v1sq is the ion's kinetic energy in units of the material's Fermi energy, and vFsq the
  // Fermi energy in units of the Bohr energy - so vF is a velocity in Fermi-velocity units.
  const real_t eF = m.fermi_energy;
  const real_t v1sq = reduced / eF;
  const real_t vFsq = eF / kEnergyBohr;
  const real_t vF = sqrt(eF / kEnergyBohr);

  const real_t y = (v1sq > real_t(1))
                       // Faster than the Fermi velocity.
                       ? vF * sqrt(v1sq) * (real_t(1) + real_t(0.2) / v1sq) / zi23
                       // Slower: a series in v1sq, which is Geant4's own expansion.
                       : real_t(0.692308) * vF
                             * (real_t(1) + real_t(0.666666) * v1sq + v1sq * v1sq / real_t(15))
                             / zi23;

  const real_t y3 = exp(real_t(0.3) * log(y));
  const real_t q = fmax(real_t(1)
                            - exp(real_t(0.803) * y3 - real_t(1.3167) * y3 * y3
                                  - real_t(0.38157) * y - real_t(0.008983) * y * y),
                        kMinCharge / eff);

  // compute charge correction
  const real_t tq = real_t(7.6) - log(reduced / real_t(1e-3));  // reduced energy in keV
  const real_t tq2 = tq * tq;
  const real_t sq =
      real_t(1)
      + (real_t(0.18) + real_t(0.0015) * z) * exp(-tq2) / (static_cast<real_t>(Zi) * Zi);

  // g4pow_a23, not pow(1-q, 2/3): G4Pow::A23 is a Taylor expansion of the cube root and the
  // exact power is ~1e-5 away from it, which is 10000 times the tolerance this is checked at.
  const real_t lambda =
      real_t(10) * vF * data::g4pow_a23<real_t>(real_t(1) - q) / (zi13 * (real_t(6) + q));
  const real_t lambda2 = lambda * lambda;
  const real_t xx = (real_t(0.5) / q - real_t(0.5)) * log(real_t(1) + lambda2) / vFsq;

  charge_correction = sq * (real_t(1) + xx);
  return eff * q;
}

/// The charge alone, for callers that do not need the correction.
template <typename real_t>
__host__ __device__ inline real_t ion_effective_charge(const data::Material<real_t>& m,
                                                       const ParticleDef<real_t>& pd,
                                                       real_t kinetic) {
  real_t unused = real_t(1);
  return ion_effective_charge(m, pd, kinetic, unused);
}

/// The whole high-order term, before the twopi_mc2_rcl2 * q2 * n_e / beta^2 prefactor that
/// G4EmCorrections::HighOrderCorrections applies. Returned in the same units as the
/// Bethe-Bloch bracket so the caller can add it there.
template <typename real_t>
__host__ __device__ inline real_t high_order_bracket(const data::Material<real_t>& m,
                                                     const ParticleDef<real_t>& pd,
                                                     real_t kinetic) {
  if (kinetic <= real_t(0)) { return real_t(0); }
  return real_t(2) * (barkas_correction(m, pd, kinetic) + bloch_correction(pd, kinetic))
         + mott_correction(pd, kinetic);
}

}  // namespace g4gpu::em
