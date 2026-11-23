// Muon radiative losses: bremsstrahlung and direct pair production.
//
// Transcribed from G4MuBremsstrahlungModel and G4MuPairProductionModel (11.1.1).
// G4EmBuilder::ConstructCharged registers both for mu+ and mu- whenever the physics list
// reaches high energy, which G4EmStandardPhysics does (MaxKinEnergy defaults to 100 TeV).
//
// A muon radiates far less than an electron - the cross section scales as 1/m^2 - and the two
// models return nothing below their own floors: G4MuBremsstrahlungModel's lowestKinEnergy is
// 100 MeV, G4MuPairProductionModel's max(850 MeV, 8 m). Above those they are DISCRETE
// processes in every B1 material: the restricted (sub-cut) loss is 4e-10 MeV/mm for the brem
// in water and exactly zero for the pair, whose 4 m_e threshold is above every B1 cut.
//
// P46 (docs/RISK.md V346-V349) added the half that makes them processes rather than numbers:
// both models' SampleSecondaries, G4ModifiedMephi's angles, and the pair model's sampling
// tables (MakeSamplingTables, G4Physics2DVector's Value and FindLinearX). Until then the muon was
// transported without either, and at 10 GeV in B1's water that was 0.13% of the trapezoid's
// dose (Geant4's own tape: 0.0107 of 8.076 MeV an event).
#pragma once
#include <cfloat>
#include <cmath>
#include "core/particle.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/materials.cuh"

namespace g4gpu::em {

/// 6-point Gauss-Legendre nodes both muon models integrate on.
template <typename real_t> __host__ __device__ inline const real_t* mu_rad_xgi() {
  static const real_t v[6] = {real_t(0.03377), real_t(0.16940), real_t(0.38069),
                              real_t(0.61931), real_t(0.83060), real_t(0.96623)};
  return v;
}
template <typename real_t> __host__ __device__ inline const real_t* mu_rad_wgi() {
  static const real_t v[6] = {real_t(0.08566), real_t(0.18038), real_t(0.23396),
                              real_t(0.23396), real_t(0.18038), real_t(0.08566)};
  return v;
}

/// A^0.27, as G4NistManager::GetA27 returns it. Geant4 tabulates the standard atomic mass
/// per element; using the same A(Z) the material table already carries keeps the two
/// consistent.
template <typename real_t>
__host__ __device__ inline real_t nist_a27(int z) {
  return pow(data::atomic_mass<real_t>(z), real_t(0.27));
}

/// fDN[Z], from G4MuBremsstrahlungModel::Initialise:
///   dn = 1.54 * A^0.27;  fDN[1] = dn;  fDN[Z>1] = dn / dn^(1/Z)
template <typename real_t>
__host__ __device__ inline real_t mu_brem_dn(int z) {
  if (z < 1) { z = 1; }
  if (z > 92) { z = 92; }
  const real_t dn = real_t(1.54) * nist_a27<real_t>(z);
  return (z == 1) ? dn : dn / pow(dn, real_t(1) / real_t(z));
}

/// Differential cross section per atom for muon bremsstrahlung.
/// Verbatim from G4MuBremsstrahlungModel::ComputeDMicroscopicCrossSection.
template <typename real_t>
__host__ __device__ inline real_t mu_brem_dxs(real_t tkin, real_t mass, int z,
                                              real_t gamma_energy) {
  if (gamma_energy > tkin || gamma_energy <= real_t(0)) { return real_t(0); }
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t sqrte = sqrt(exp(real_t(1)));
  const real_t rmass = mass / me;
  const real_t cc = units::classic_electron_radius<real_t>() / rmass;
  const real_t coeff = real_t(16) * units::fine_structure_const<real_t>() * cc * cc / real_t(3);

  const real_t E = tkin + mass;
  const real_t v = gamma_energy / E;
  const real_t delta = real_t(0.5) * mass * mass * v / (E - gamma_energy);
  const real_t rab0 = delta * sqrte;
  int iz = z;
  if (iz < 1) { iz = 1; }
  if (iz > 92) { iz = 92; }
  const real_t z13 = real_t(1) / pow(real_t(iz), real_t(1) / real_t(3));
  const real_t dnstar = mu_brem_dn<real_t>(iz);

  // Hydrogen uses the Bethe-Heitler constants, everything else Thomas-Fermi.
  const real_t b = (iz == 1) ? real_t(202.4) : real_t(183.0);
  const real_t b1 = (iz == 1) ? real_t(446.0) : real_t(1429.0);

  const real_t rab1 = b * z13;
  real_t fn = log(rab1 / (dnstar * (me + rab0 * rab1))
                  * (mass + delta * (dnstar * sqrte - real_t(2))));
  fn = fmax(fn, real_t(0));

  const real_t epmax1 = E / (real_t(1) + real_t(0.5) * mass * rmass / E);
  real_t fe = real_t(0);
  if (gamma_energy < epmax1) {
    const real_t rab2 = b1 * z13 * z13;
    fe = log(rab2 * mass
             / ((real_t(1) + delta * rmass / (me * sqrte)) * (me + rab0 * rab2)));
    fe = fmax(fe, real_t(0));
  }
  const real_t dxs = coeff * (real_t(1) - v * (real_t(1) - real_t(0.75) * v)) * real_t(z)
                     * (fn * real_t(z) + fe) / gamma_energy;
  return fmax(dxs, real_t(0));
}

/// Restricted radiative energy loss per atom.
/// Verbatim from G4MuBremsstrahlungModel::ComputMuBremLoss.
template <typename real_t>
__host__ __device__ inline real_t mu_brem_loss(int z, real_t tkin, real_t mass, real_t cut) {
  const real_t total = mass + tkin;
  constexpr real_t ak1 = real_t(0.05);
  constexpr int k2 = 5;
  const real_t vcut = cut / total;
  int kkk = static_cast<int>(vcut / ak1) + k2;
  if (kkk > 8) { kkk = 8; }
  if (kkk < 1) { kkk = 1; }
  const real_t hhh = vcut / real_t(kkk);
  const real_t* xgi = mu_rad_xgi<real_t>();
  const real_t* wgi = mu_rad_wgi<real_t>();
  real_t loss = real_t(0), aa = real_t(0);
  for (int l = 0; l < kkk; ++l) {
    for (int i = 0; i < 6; ++i) {
      const real_t ep = (aa + xgi[i] * hhh) * total;
      loss += ep * wgi[i] * mu_brem_dxs(tkin, mass, z, ep);
    }
    aa += hhh;
  }
  return loss * hhh * total;
}

/// Cross section per atom above the photon production cut.
/// Verbatim from G4MuBremsstrahlungModel::ComputeMicroscopicCrossSection.
template <typename real_t>
__host__ __device__ inline real_t mu_brem_xs_per_atom(int z, real_t tkin, real_t mass,
                                                      real_t cut) {
  if (cut >= tkin) { return real_t(0); }
  const real_t total = tkin + mass;
  constexpr real_t ak1 = real_t(2.3);
  constexpr int k2 = 4;
  const real_t aaa = log(cut / total);
  const real_t bbb = log(tkin / total);
  int kkk = static_cast<int>((bbb - aaa) / ak1) + k2;
  if (kkk > 8) { kkk = 8; }
  if (kkk < 1) { kkk = 1; }
  const real_t hhh = (bbb - aaa) / real_t(kkk);
  const real_t* xgi = mu_rad_xgi<real_t>();
  const real_t* wgi = mu_rad_wgi<real_t>();
  real_t cross = real_t(0), aa = aaa;
  for (int l = 0; l < kkk; ++l) {
    for (int i = 0; i < 6; ++i) {
      const real_t ep = exp(aa + xgi[i] * hhh) * total;
      cross += ep * wgi[i] * mu_brem_dxs(tkin, mass, z, ep);
    }
    aa += hhh;
  }
  return cross * hhh;
}

/// Below this the model returns zero (G4MuBremsstrahlungModel::lowestKinEnergy).
template <typename real_t> __host__ __device__ constexpr real_t kMuBremLowest() {
  return real_t(100.0);  // G4MuBremsstrahlungModel::lowestKinEnergy = 0.1 GeV
}
/// Smallest photon energy the model will consider (minThreshold).
template <typename real_t> __host__ __device__ constexpr real_t kMuBremMinThreshold() {
  return real_t(0.9e-3);  // 0.9 keV
}

/// Muon bremsstrahlung dE/dx per volume, MeV/mm.
/// Verbatim from G4MuBremsstrahlungModel::ComputeDEDXPerVolume.
template <typename real_t>
__host__ __device__ inline real_t mu_brem_dedx(const data::Material<real_t>& m,
                                               ParticleType type, real_t kinetic,
                                               real_t cut) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (kinetic <= kMuBremLowest<real_t>()) { return real_t(0); }
  real_t c = fmax(cut, kMuBremMinThreshold<real_t>());
  c = fmin(c, kinetic);
  real_t dedx = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    dedx += mu_brem_loss(z, kinetic, pd.mass, c) * m.n_atoms[i];
  }
  return fmax(dedx, real_t(0));
}

/// Muon bremsstrahlung cross section per volume, 1/mm.
template <typename real_t>
__host__ __device__ inline real_t mu_brem_xs(const data::Material<real_t>& m,
                                             ParticleType type, real_t kinetic, real_t cut) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (kinetic <= kMuBremLowest<real_t>()) { return real_t(0); }
  const real_t c = fmax(cut, kMuBremMinThreshold<real_t>());
  if (c >= kinetic) { return real_t(0); }
  real_t xs = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    xs += mu_brem_xs_per_atom(z, kinetic, pd.mass, c) * m.n_atoms[i];
  }
  return fmax(xs, real_t(0));
}


// ------------------------------------------------------------- direct pair production

/// 8-point Gauss-Legendre nodes G4MuPairProductionModel uses (a different set from the
/// 6-point rule the bremsstrahlung model integrates on).
template <typename real_t> __host__ __device__ inline const real_t* mu_pair_xgi() {
  static const real_t v[8] = {
      real_t(0.0198550717512320), real_t(0.1016667612931865), real_t(0.2372337950418355),
      real_t(0.4082826787521750), real_t(0.5917173212478250), real_t(0.7627662049581645),
      real_t(0.8983332387068135), real_t(0.9801449282487680)};
  return v;
}
template <typename real_t> __host__ __device__ inline const real_t* mu_pair_wgi() {
  static const real_t v[8] = {
      real_t(0.0506142681451880), real_t(0.1111905172266870), real_t(0.1568533229389435),
      real_t(0.1813418916891810), real_t(0.1813418916891810), real_t(0.1568533229389435),
      real_t(0.1111905172266870), real_t(0.0506142681451880)};
  return v;
}

/// Smallest e+e- pair energy: 4 m_e (G4MuPairProductionModel::minPairEnergy).
template <typename real_t> __host__ __device__ inline real_t kMinPairEnergy() {
  return real_t(4) * units::electron_mass_c2<real_t>();
}
/// G4MuPairProductionModel::lowestKinEnergy, raised to 8 * mass in Initialise.
template <typename real_t> __host__ __device__ inline real_t mu_pair_lowest(real_t mass) {
  return fmax(real_t(850.0), mass * real_t(8));
}

/// alpha^2 * r_e^2 * 4 / (3 pi).
template <typename real_t> __host__ __device__ inline real_t mu_pair_factor() {
  constexpr real_t alpha = units::fine_structure_const<real_t>();
  const real_t re = units::classic_electron_radius<real_t>();
  return alpha * alpha * re * re * real_t(4)
         / (real_t(3) * real_t(3.14159265358979323846));
}

/// Largest pair energy for this element.
/// Verbatim from G4MuPairProductionModel::MaxSecondaryEnergyForElement.
template <typename real_t>
__host__ __device__ inline real_t mu_pair_max_energy(real_t kinetic, real_t mass, int z) {
  const real_t sqrte = sqrt(exp(real_t(1)));
  const real_t z13 = pow(real_t(z), real_t(1) / real_t(3));
  return kinetic + mass * (real_t(1) - real_t(0.75) * sqrte * z13);
}

/// Differential cross section for muon direct pair production.
/// Verbatim from G4MuPairProductionModel::ComputeDMicroscopicCrossSection - the Kelner,
/// Kokoulin and Petrukhin formula, with an inner 8-point rule over the pair asymmetry rho.
template <typename real_t>
__host__ __device__ inline real_t mu_pair_dxs(real_t tkin, real_t mass, int z,
                                              real_t pair_energy) {
  constexpr real_t bbbtf = real_t(183.0), bbbh = real_t(202.4);
  constexpr real_t g1tf = real_t(1.95e-5), g2tf = real_t(5.3e-5);
  constexpr real_t g1h = real_t(4.4e-5), g2h = real_t(4.8e-5);
  const real_t me = units::electron_mass_c2<real_t>();
  if (pair_energy <= kMinPairEnergy<real_t>()) { return real_t(0); }

  const real_t sqrte = sqrt(exp(real_t(1)));
  const real_t z13 = pow(real_t(z), real_t(1) / real_t(3));
  const real_t z23 = z13 * z13;
  const real_t total = tkin + mass;
  const real_t resid = total - pair_energy;
  if (resid <= real_t(0.75) * sqrte * z13 * mass) { return real_t(0); }

  const real_t a0 = real_t(1) / (total * resid);
  const real_t alf = real_t(4) * me / pair_energy;
  const real_t rt = sqrt(real_t(1) - alf);
  const real_t delta = real_t(6) * mass * mass * a0;
  const real_t tmnexp = alf / (real_t(1) + rt) + delta * rt;
  if (tmnexp >= real_t(1)) { return real_t(0); }
  const real_t tmn = log(tmnexp);
  const real_t massratio = mass / me;
  const real_t massratio2 = massratio * massratio;
  const real_t inv_massratio2 = real_t(1) / massratio2;

  const real_t bbb = (real_t(z) < real_t(1.5)) ? bbbh : bbbtf;
  const real_t g1 = (real_t(z) < real_t(1.5)) ? g1h : g1tf;
  const real_t g2 = (real_t(z) < real_t(1.5)) ? g2h : g2tf;

  real_t zeta = real_t(0);
  const real_t z1exp = total / (mass + g1 * z23 * total);
  if (z1exp > real_t(35.221047195922)) {
    const real_t z2exp = total / (mass + g2 * z13 * total);
    zeta = (real_t(0.073) * log(z1exp) - real_t(0.26))
           / (real_t(0.058) * log(z2exp) - real_t(0.14));
  }
  const real_t z2 = real_t(z) * (real_t(z) + zeta);
  const real_t screen0 = real_t(2) * me * sqrte * bbb / (z13 * pair_energy);
  const real_t beta = real_t(0.5) * pair_energy * pair_energy * a0;
  const real_t xi0 = real_t(0.5) * massratio2 * beta;

  const real_t* xgi = mu_pair_xgi<real_t>();
  const real_t* wgi = mu_pair_wgi<real_t>();
  const real_t b40 = real_t(4) * beta;
  const real_t b62 = real_t(6) * beta + real_t(2);

  real_t sum = real_t(0);
  for (int i = 0; i < 8; ++i) {
    const real_t rho = exp(tmn * xgi[i]) - real_t(1);  // rho = -asymmetry
    const real_t rho2 = rho * rho;
    const real_t xi = xi0 * (real_t(1) - rho2);
    const real_t xi1 = real_t(1) + xi;
    const real_t xii = real_t(1) / xi;

    const real_t yeu = (b40 + real_t(5)) + (b40 - real_t(1)) * rho2;
    const real_t yed =
        b62 * log(real_t(3) + xii) + (real_t(2) * beta - real_t(1)) * rho2 - b40;
    const real_t ymu = b62 * (real_t(1) + rho2) + real_t(6);
    const real_t ymd = (b40 + real_t(3)) * (real_t(1) + rho2) * log(real_t(3) + xi)
                       + real_t(2) - real_t(3) * rho2;
    const real_t ye1 = real_t(1) + yeu / yed;
    const real_t ym1 = real_t(1) + ymu / ymd;

    real_t be, bm;
    if (xi <= real_t(1000)) {
      be = ((real_t(2) + rho2) * (real_t(1) + beta) + xi * (real_t(3) + rho2))
               * log(real_t(1) + xii)
           + (real_t(1) - rho2 - beta) / xi1 - (real_t(3) + rho2);
    } else {
      be = real_t(0.5) * (real_t(3) - rho2 + real_t(2) * beta * (real_t(1) + rho2)) * xii;
    }
    if (xi >= real_t(0.001)) {
      const real_t a10 = (real_t(1) + real_t(2) * beta) * (real_t(1) - rho2);
      bm = ((real_t(1) + rho2) * (real_t(1) + real_t(1.5) * beta) - a10 * xii) * log(xi1)
           + xi * (real_t(1) - rho2 - beta) / xi1 + a10;
    } else {
      bm = real_t(0.5) * (real_t(5) - rho2 + beta * (real_t(3) + rho2)) * xi;
    }

    const real_t screen = screen0 * xi1 / (real_t(1) - rho2);
    const real_t ale = log(bbb / z13 * sqrt(xi1 * ye1) / (real_t(1) + screen * ye1));
    const real_t cre =
        real_t(0.5) * log(real_t(1) + real_t(2.25) * z23 * xi1 * ye1 * inv_massratio2);
    real_t fe = (ale - cre) * be;
    fe = fmax(fe, real_t(0));
    const real_t alm_crm =
        log(bbb * massratio / (real_t(1.5) * z23 * (real_t(1) + screen * ym1)));
    const real_t fm = fmax(alm_crm * bm, real_t(0)) * inv_massratio2;
    sum += wgi[i] * (real_t(1) + rho) * (fe + fm);
  }
  return -tmn * sum * mu_pair_factor<real_t>() * z2 * resid / (total * pair_energy);
}

/// Cross section per atom above the pair-energy cut.
/// Verbatim from G4MuPairProductionModel::ComputeMicroscopicCrossSection. G4lrint rounds to
/// nearest, which is why the subinterval count uses floor(x + 0.5) rather than a cast.
template <typename real_t>
__host__ __device__ inline real_t mu_pair_xs_per_atom(int z, real_t tkin, real_t mass,
                                                      real_t cut_energy) {
  constexpr real_t ak1 = real_t(6.9), ak2 = real_t(1.0);
  const real_t tmax = mu_pair_max_energy(tkin, mass, z);
  const real_t cut = fmax(cut_energy, kMinPairEnergy<real_t>());
  if (tmax <= cut) { return real_t(0); }
  const real_t aaa = log(cut), bbb = log(tmax);
  int kkk = static_cast<int>(floor((bbb - aaa) / ak1 + ak2 + real_t(0.5)));
  if (kkk > 8) { kkk = 8; }
  if (kkk < 1) { kkk = 1; }
  const real_t hhh = (bbb - aaa) / real_t(kkk);
  const real_t* xgi = mu_pair_xgi<real_t>();
  const real_t* wgi = mu_pair_wgi<real_t>();
  real_t cross = real_t(0), x = aaa;
  for (int l = 0; l < kkk; ++l) {
    for (int i = 0; i < 8; ++i) {
      const real_t ep = exp(x + xgi[i] * hhh);
      cross += ep * wgi[i] * mu_pair_dxs(tkin, mass, z, ep);
    }
    x += hhh;
  }
  return fmax(cross * hhh, real_t(0));
}

/// Restricted energy loss per atom.
/// Verbatim from G4MuPairProductionModel::ComputMuPairLoss.
template <typename real_t>
__host__ __device__ inline real_t mu_pair_loss(int z, real_t tkin, real_t mass,
                                               real_t cut_energy, real_t tmax) {
  constexpr real_t ak1 = real_t(6.9), ak2 = real_t(1.0);
  const real_t cut = fmin(cut_energy, tmax);
  if (cut <= kMinPairEnergy<real_t>()) { return real_t(0); }
  const real_t aaa = log(kMinPairEnergy<real_t>()), bbb = log(cut);
  int kkk = static_cast<int>(floor((bbb - aaa) / ak1 + ak2 + real_t(0.5)));
  if (kkk > 8) { kkk = 8; }
  if (kkk < 1) { kkk = 1; }
  const real_t hhh = (bbb - aaa) / real_t(kkk);
  const real_t* xgi = mu_pair_xgi<real_t>();
  const real_t* wgi = mu_pair_wgi<real_t>();
  real_t loss = real_t(0), x = aaa;
  for (int l = 0; l < kkk; ++l) {
    for (int i = 0; i < 8; ++i) {
      const real_t ep = exp(x + xgi[i] * hhh);
      loss += wgi[i] * ep * ep * mu_pair_dxs(tkin, mass, z, ep);
    }
    x += hhh;
  }
  return fmax(loss * hhh, real_t(0));
}

/// Muon pair-production dE/dx per volume, MeV/mm.
/// Verbatim from G4MuPairProductionModel::ComputeDEDXPerVolume. This is zero unless the
/// electron production cut exceeds 4 m_e = 2.04 MeV, which it does not for any of the four
/// B1 materials: there every pair is produced as a discrete secondary instead.
template <typename real_t>
__host__ __device__ inline real_t mu_pair_dedx(const data::Material<real_t>& m,
                                               ParticleType type, real_t kinetic,
                                               real_t cut) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (cut <= kMinPairEnergy<real_t>() || kinetic <= mu_pair_lowest(pd.mass)) {
    return real_t(0);
  }
  real_t dedx = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    const real_t tmax = mu_pair_max_energy(kinetic, pd.mass, z);
    dedx += mu_pair_loss(z, kinetic, pd.mass, cut, tmax) * m.n_atoms[i];
  }
  return fmax(dedx, real_t(0));
}

/// Muon pair-production cross section per volume, 1/mm.
template <typename real_t>
__host__ __device__ inline real_t mu_pair_xs(const data::Material<real_t>& m,
                                             ParticleType type, real_t kinetic, real_t cut) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (kinetic <= mu_pair_lowest(pd.mass)) { return real_t(0); }
  real_t xs = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    xs += mu_pair_xs_per_atom(z, kinetic, pd.mass, cut) * m.n_atoms[i];
  }
  return fmax(xs, real_t(0));
}


// ------------------------------------------------------------- hadron bremsstrahlung

/// Differential cross section for hadron bremsstrahlung.
/// Verbatim from G4hBremsstrahlungModel::ComputeDMicroscopicCrossSection.
///
/// G4hBremsstrahlungModel derives from G4MuBremsstrahlungModel and overrides only this one
/// function, so everything else - the loss and cross section integrals, the node sets, the
/// energy limits - is shared. It differs from the muon form in three ways: the nuclear
/// form-factor length dn is mass * A^0.27 / 70 MeV rather than the tabulated fDN, there is
/// no scattering-off-atomic-electrons term, and the spin factor is applied only for
/// particles with non-zero spin (so pions and kaons take a different numerator).
template <typename real_t>
__host__ __device__ inline real_t h_brem_dxs(real_t tkin, real_t mass, real_t spin, int z,
                                             real_t gamma_energy) {
  if (gamma_energy > tkin || gamma_energy <= real_t(0)) { return real_t(0); }
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t sqrte = sqrt(exp(real_t(1)));
  const real_t rmass = mass / me;
  const real_t cc = units::classic_electron_radius<real_t>() / rmass;
  const real_t coeff = real_t(16) * units::fine_structure_const<real_t>() * cc * cc / real_t(3);

  const real_t E = tkin + mass;
  const real_t v = gamma_energy / E;
  const real_t delta = real_t(0.5) * mass * mass * v / (E - gamma_energy);
  const real_t rab0 = delta * sqrte;
  const int iz = (z < 1) ? 1 : z;
  const real_t z13 = real_t(1) / pow(real_t(iz), real_t(1) / real_t(3));
  const real_t dn = mass * nist_a27<real_t>(iz) / real_t(70.0);  // 70 MeV
  const real_t b = (iz == 1) ? real_t(202.4) : real_t(183.0);
  const real_t rab1 = b * z13;
  real_t fn = log(rab1 / (dn * (me + rab0 * rab1)) * (mass + delta * (dn * sqrte - real_t(2))));
  fn = fmax(fn, real_t(0));
  real_t x = real_t(1) - v;
  if (spin != real_t(0)) { x += real_t(0.75) * v * v; }
  return coeff * x * real_t(z) * real_t(z) * fn / gamma_energy;
}

/// Restricted radiative energy loss per atom, hadron form.
/// Same integral as G4MuBremsstrahlungModel::ComputMuBremLoss, which G4hBremsstrahlungModel
/// inherits unchanged.
template <typename real_t>
__host__ __device__ inline real_t h_brem_loss(int z, real_t tkin, real_t mass, real_t spin,
                                              real_t cut) {
  const real_t total = mass + tkin;
  constexpr real_t ak1 = real_t(0.05);
  constexpr int k2 = 5;
  const real_t vcut = cut / total;
  int kkk = static_cast<int>(vcut / ak1) + k2;
  if (kkk > 8) { kkk = 8; }
  if (kkk < 1) { kkk = 1; }
  const real_t hhh = vcut / real_t(kkk);
  const real_t* xgi = mu_rad_xgi<real_t>();
  const real_t* wgi = mu_rad_wgi<real_t>();
  real_t loss = real_t(0), aa = real_t(0);
  for (int l = 0; l < kkk; ++l) {
    for (int i = 0; i < 6; ++i) {
      const real_t ep = (aa + xgi[i] * hhh) * total;
      loss += ep * wgi[i] * h_brem_dxs(tkin, mass, spin, z, ep);
    }
    aa += hhh;
  }
  return loss * hhh * total;
}

/// Cross section per atom above the photon production cut, hadron form.
template <typename real_t>
__host__ __device__ inline real_t h_brem_xs_per_atom(int z, real_t tkin, real_t mass,
                                                     real_t spin, real_t cut) {
  if (cut >= tkin) { return real_t(0); }
  const real_t total = tkin + mass;
  constexpr real_t ak1 = real_t(2.3);
  constexpr int k2 = 4;
  const real_t aaa = log(cut / total);
  const real_t bbb = log(tkin / total);
  int kkk = static_cast<int>((bbb - aaa) / ak1) + k2;
  if (kkk > 8) { kkk = 8; }
  if (kkk < 1) { kkk = 1; }
  const real_t hhh = (bbb - aaa) / real_t(kkk);
  const real_t* xgi = mu_rad_xgi<real_t>();
  const real_t* wgi = mu_rad_wgi<real_t>();
  real_t cross = real_t(0), aa = aaa;
  for (int l = 0; l < kkk; ++l) {
    for (int i = 0; i < 6; ++i) {
      const real_t ep = exp(aa + xgi[i] * hhh) * total;
      cross += ep * wgi[i] * h_brem_dxs(tkin, mass, spin, z, ep);
    }
    aa += hhh;
  }
  return cross * hhh;
}

/// Hadron bremsstrahlung dE/dx per volume, MeV/mm.
template <typename real_t>
__host__ __device__ inline real_t h_brem_dedx(const data::Material<real_t>& m,
                                              ParticleType type, real_t kinetic, real_t cut) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (kinetic <= kMuBremLowest<real_t>()) { return real_t(0); }
  real_t c = fmax(cut, kMuBremMinThreshold<real_t>());
  c = fmin(c, kinetic);
  real_t dedx = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    dedx += h_brem_loss(z, kinetic, pd.mass, pd.spin, c) * m.n_atoms[i];
  }
  return fmax(dedx, real_t(0));
}

/// Hadron bremsstrahlung cross section per volume, 1/mm.
template <typename real_t>
__host__ __device__ inline real_t h_brem_xs(const data::Material<real_t>& m, ParticleType type,
                                            real_t kinetic, real_t cut) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (kinetic <= kMuBremLowest<real_t>()) { return real_t(0); }
  const real_t c = fmax(cut, kMuBremMinThreshold<real_t>());
  if (c >= kinetic) { return real_t(0); }
  real_t xs = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    xs += h_brem_xs_per_atom(z, kinetic, pd.mass, pd.spin, c) * m.n_atoms[i];
  }
  return fmax(xs, real_t(0));
}

// G4hPairProductionModel derives from G4MuPairProductionModel and overrides nothing, so
// mu_pair_dedx / mu_pair_xs already cover hadrons: they take the mass from the ParticleDef.


// ===================================================================== P46: the final states
//
// Everything below is what the two muon processes DO when they fire, transcribed from
// G4MuBremsstrahlungModel::SampleSecondaries, G4MuPairProductionModel::MakeSamplingTables /
// FindScaledEnergy / SampleSecondaries, G4ModifiedMephi and G4Physics2DVector (all 11.1.1).
// tests/test_mu_radiative.cu holds each to Geant4's own calls deviate for deviate
// (ref/dump/dump_mu_radiative.cc), and the pair tables to their nodes.
//
// TWO THINGS ARE THE PORT'S CONVENTION AND NOT GEANT4'S, both stated where the stepper uses them
// and both shared with every other discrete EM process here (docs/PORTED.md 1.6, docs/RISK.md
// V78): the element is drawn from the partial cross sections evaluated at the energy rather
// than from G4EmElementSelector's table (one uniform either way, the same comparison), and the
// interaction length is the model's cross section at the pre-step energy rather than the
// integral approach against the lambda table.

/// `Hep3Vector::unit()`: multiplies by the inverse norm (CLHEP), where `normalize` divides.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> mu_unit(const Vec3<real_t>& v) {
  const real_t tot = mag2(v);
  if (!(tot > real_t(0))) { return v; }
  const real_t s = real_t(1) / sqrt(tot);
  return Vec3<real_t>{v.x * s, v.y * s, v.z * s};
}

/// `G4ModifiedMephi::SampleCosTheta`: one uniform.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t modified_mephi_cos_theta(real_t prim_kin_energy,
                                                           real_t g_energy, real_t mass,
                                                           Rng& rng) {
  constexpr real_t kHalfPi = real_t(1.57079632679489661923);  // CLHEP::halfpi
  const real_t gam = real_t(1) + prim_kin_energy / mass;
  const real_t rmax = gam * kHalfPi * fmin(real_t(1), gam * mass / g_energy - real_t(1));
  const real_t rmax2 = rmax * rmax;
  const real_t x = rng.uniform() * rmax2 / (real_t(1) + rmax2);
  return cos(sqrt(x / (real_t(1) - x)) / gam);
}

/// What either muon process leaves behind. `n` secondaries (1 for the brem's photon, 2 for the
/// pair's electron then positron, 0 when the kinematic window is shut and nothing happened -
/// SampleSecondaries returns without touching the particle change, so the muon is unchanged).
template <typename real_t>
struct MuRadFinalState {
  int n = 0;
  ParticleType sec_type[2] = {ParticleType::kGamma, ParticleType::kGamma};
  real_t sec_ekin[2] = {real_t(0), real_t(0)};
  Vec3<real_t> sec_dir[2] = {};
  real_t primary_ekin = real_t(0);
  Vec3<real_t> primary_dir{};
};

/// The element a muon radiative interaction happens on: `G4VEmModel::SelectRandomAtom`'s one
/// uniform (none for a one-element material) against the cumulative partial cross sections,
/// computed at the energy (the port's convention, see the block header). `pair` picks the
/// model. Returns the element index.
template <typename real_t, typename Rng>
__host__ __device__ inline int mu_rad_select_element(const data::Material<real_t>& m, bool pair,
                                                     real_t kinetic, real_t mass, real_t cut,
                                                     Rng& rng) {
  if (m.n_elements <= 1) { return 0; }
  real_t part[data::kMaxElements];
  real_t sum = real_t(0);
  const real_t bcut = fmax(cut, kMuBremMinThreshold<real_t>());
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    const real_t xs = pair ? mu_pair_xs_per_atom(z, kinetic, mass, cut)
                           : ((bcut < kinetic) ? mu_brem_xs_per_atom(z, kinetic, mass, bcut)
                                               : real_t(0));
    sum += m.n_atoms[i] * xs;
    part[i] = sum;
  }
  // G4EmElementSelector::SelectRandomAtom: x = u, the first element whose normalised
  // cumulative cross section is >= x, the last one otherwise.
  const real_t x = rng.uniform();
  if (!(sum > real_t(0))) { return m.n_elements - 1; }
  for (int i = 0; i < m.n_elements - 1; ++i) {
    if (x <= part[i] / sum) { return i; }
  }
  return m.n_elements - 1;
}

/// `G4MuBremsstrahlungModel::SampleSecondaries` on element `z`, the photon's direction by
/// `G4ModifiedMephi::SampleDirection`. `cut` is the photon production threshold (the couple's
/// gamma cut, as G4VEnergyLossProcess::PostStepDoIt passes it); maxEnergy is DBL_MAX.
///
/// The do/while is Geant4's, unbounded there; here it gives up after 100000 trials with no
/// interaction, a bound the acceptance (func1 is the integrand at its lower edge, and
/// E*dsigma/dE falls monotonically) makes unreachable.
template <typename real_t, typename Rng>
__host__ __device__ inline MuRadFinalState<real_t> sample_mu_brem(int z, real_t kinetic,
                                                                  real_t mass, real_t cut,
                                                                  const Vec3<real_t>& dir,
                                                                  Rng& rng) {
  MuRadFinalState<real_t> out;
  out.primary_ekin = kinetic;
  out.primary_dir = dir;
  const real_t min_threshold = kMuBremMinThreshold<real_t>();
  const real_t tmax = kinetic;  // min(kineticEnergy, maxEnergy = DBL_MAX)
  real_t tmin = fmin(kinetic, cut);
  tmin = fmax(tmin, min_threshold);
  if (tmin >= tmax) { return out; }

  const real_t func1 = tmin * mu_brem_dxs(kinetic, mass, z, tmin);
  const real_t xmin = log(tmin / min_threshold);
  const real_t xmax = log(tmax / tmin);
  real_t g_energy = real_t(0);
  bool accepted = false;
  for (int it = 0; it < 100000 && !accepted; ++it) {
    g_energy = min_threshold * exp(xmin + rng.uniform() * xmax);
    const real_t func2 = g_energy * mu_brem_dxs(kinetic, mass, z, g_energy);
    accepted = !(func2 < func1 * rng.uniform());
  }
  if (!accepted) { return out; }

  // G4ModifiedMephi::SampleDirection: the polar angle first, then phi.
  const real_t cost = modified_mephi_cos_theta(kinetic, g_energy, mass, rng);
  const real_t sint = sqrt((real_t(1) - cost) * (real_t(1) + cost));
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  const Vec3<real_t> gdir =
      rotate_uz(Vec3<real_t>{sint * cos(phi), sint * sin(phi), cost}, dir);

  const real_t tot_momentum = sqrt(kinetic * (kinetic + real_t(2) * mass));
  out.n = 1;
  out.sec_type[0] = ParticleType::kGamma;
  out.sec_ekin[0] = g_energy;
  out.sec_dir[0] = gdir;
  // SecondaryThreshold() is MuHadBremsstrahlungTh = MaxKinEnergy (100 TeV) in option0, so the
  // primary always survives; the fStopAndKill branch is unreachable below 100 TeV.
  out.primary_dir = mu_unit(tot_momentum * dir - g_energy * gdir);
  out.primary_ekin = kinetic - g_energy;
  return out;
}

// ------------------------------------------------------------- the pair model's tables

/// G4MuPairProductionModel's static `zdat`: the five elements it tabulates.
constexpr int kMuPairNZ = 5;
__host__ __device__ inline int mu_pair_zdat(int i) {
  switch (i) {
    case 0: return 1;
    case 1: return 4;
    case 2: return 13;
    case 3: return 29;
    default: return 92;
  }
}
/// `nbiny` (the x axis: 1000 bins, 1001 nodes) and the largest `nbine` + 1 a table holds. For the
/// muon nbine = size_t(4 * log10(1e8 / 850)) = 20 (`nYBinPerDecade` = 4, emin = 850 MeV, emax =
/// MaxKinEnergy = 100 TeV). A heavier projectile (G4hPairProductionModel, not P46's) has a
/// higher emin and fewer.
constexpr int kMuPairNBinY = 1000;
constexpr int kMuPairNX = kMuPairNBinY + 1;
constexpr int kMuPairMaxNE = 24;

/// The `G4ElementData` 2D vectors `MakeSamplingTables` fills, one per `zdat` element, and the
/// model scalars sampling reads. X is the scaled pair energy y = log(E_pair/T)/coef in
/// [ymin, 0]; Y is log(T/MeV) on nbine + 1 nodes; the value is the running sum of
/// E_pair * dsigma/dE_pair over x (not multiplied by the bin width - "table will be used only
/// for sampling").
template <typename real_t>
struct MuPairSamplingTable {
  int ny = 0;          ///< nbine + 1
  real_t mass = 0;     ///< the projectile mass the table was built for
  real_t ymin = 0;     ///< G4Log(minPairEnergy / emin)
  real_t x[kMuPairNX];
  real_t y[kMuPairMaxNE];
  real_t v[kMuPairNZ][kMuPairMaxNE][kMuPairNX];
};

/// `G4MuPairProductionModel::Initialise` (the table scale) and `MakeSamplingTables`, host only,
/// in double. `emin` is max(lowestKinEnergy, LowEnergyLimit()) = max(850 MeV, 8 m, 100 eV);
/// `emax` max(HighEnergyLimit() = 100 TeV, 2 emin). Every expression is the source's, in its
/// order, so the nodes are Geant4's to rounding (`G4Log`/`G4Exp` are std::log/std::exp on this
/// platform) - including x accumulated as `x += dy` rather than computed as ymin + i*dy.
template <typename real_t>
inline void build_mu_pair_sampling_table(MuPairSamplingTable<real_t>& t, double mass) {
  const double min_pair = 4.0 * units::electron_mass_c2<double>();
  const double lowest = fmax(850.0, mass * 8.0);
  const double emin = fmax(lowest, 1e-4);
  const double emax = fmax(1e8, emin * 2);
  std::size_t nbine = static_cast<std::size_t>(4 * std::log10(emax / emin));
  if (nbine < 3) { nbine = 3; }
  if (static_cast<int>(nbine) + 1 > kMuPairMaxNE) { nbine = kMuPairMaxNE - 1; }  // never: muon 20
  const double ymin = std::log(min_pair / emin);
  const double dy = -ymin / static_cast<double>(kMuPairNBinY);
  t.ny = static_cast<int>(nbine) + 1;
  t.mass = static_cast<real_t>(mass);
  t.ymin = static_cast<real_t>(ymin);
  const double factore = std::exp(std::log(emax / emin) / static_cast<double>(nbine));
  const double sqrte = std::sqrt(std::exp(1.0));
  for (int iz = 0; iz < kMuPairNZ; ++iz) {
    const int z = mu_pair_zdat(iz);
    double kin = emin;
    for (std::size_t it = 0; it <= nbine; ++it) {
      t.y[it] = static_cast<real_t>(std::log(kin));
      // MaxSecondaryEnergyForElement, with the z13 the differential cross section also uses.
      const double max_pair = kin + mass * (1.0 - 0.75 * sqrte * std::pow(double(z), 1.0 / 3.0));
      const double coef = std::log(min_pair / kin) / ymin;
      const double ymax = std::log(max_pair / kin) / coef;
      double fac = (ymax - ymin) / dy;
      const std::size_t imax = static_cast<std::size_t>(fac);
      fac -= static_cast<double>(imax);
      double xsec = 0.0;
      double x = ymin;
      t.v[iz][it][0] = real_t(0);
      if (it == 0) { t.x[kMuPairNBinY] = real_t(0); }
      for (std::size_t i = 0; i < static_cast<std::size_t>(kMuPairNBinY); ++i) {
        if (it == 0) { t.x[i] = static_cast<real_t>(x); }
        if (i < imax) {
          const double ep = kin * std::exp(coef * (x + dy * 0.5));
          xsec += ep * mu_pair_dxs<double>(kin, mass, z, ep);
        } else if (i == imax) {
          const double ep = kin * std::exp(coef * (x + fac * dy * 0.5));
          xsec += ep * fac * mu_pair_dxs<double>(kin, mass, z, ep);
        }
        t.v[iz][it][i + 1] = static_cast<real_t>(xsec);
        x += dy;
      }
      kin *= factore;
      if (it + 1 == nbine) { kin = emax; }
    }
  }
}

/// `G4Physics2DVector::FindBin` with the cached index 0, as `Value(x, y)` and `FindLinearX(r, y)`
/// call it: the `lower_bound` over the whole axis when z is inside (v[1], v[idxmax]).
template <typename real_t>
__host__ __device__ inline int mu_pair_find_bin(real_t z, const real_t* v, int n, int idxmax) {
  if (z <= v[1]) { return 0; }
  if (z >= v[idxmax]) { return idxmax; }
  int lo = 0, hi = n;  // std::lower_bound: the first node not less than z
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (v[mid] < z) { lo = mid + 1; } else { hi = mid; }
  }
  return lo - 1;
}

/// `G4Physics2DVector::Value(x, y)`, bilinear (useBicubic is false), clamped to the table.
template <typename real_t>
__host__ __device__ inline real_t mu_pair_table_value(const MuPairSamplingTable<real_t>& t,
                                                      int iz, real_t xx, real_t yy) {
  const real_t x = fmin(fmax(xx, t.x[0]), t.x[kMuPairNX - 1]);
  const real_t y = fmin(fmax(yy, t.y[0]), t.y[t.ny - 1]);
  const int idx = mu_pair_find_bin(x, t.x, kMuPairNX, kMuPairNX - 2);
  const int idy = mu_pair_find_bin(y, t.y, t.ny, t.ny - 2);
  const real_t x1 = t.x[idx], x2 = t.x[idx + 1];
  const real_t y1 = t.y[idy], y2 = t.y[idy + 1];
  const real_t v11 = t.v[iz][idy][idx];
  const real_t v12 = t.v[iz][idy][idx + 1];
  const real_t v21 = t.v[iz][idy + 1][idx];
  const real_t v22 = t.v[iz][idy + 1][idx + 1];
  return ((y2 - y) * (v11 * (x2 - x) + v12 * (x - x1))
          + ((y - y1) * (v21 * (x2 - x) + v22 * (x - x1))))
         / ((x2 - x1) * (y2 - y1));
}

/// `G4Physics2DVector::InterpolateLinearX` on row `row` of element `iz`.
template <typename real_t>
__host__ __device__ inline real_t mu_pair_interpolate_x(const MuPairSamplingTable<real_t>& t,
                                                        int iz, int row, real_t rand) {
  const real_t* v = t.v[iz][row];
  int n1 = 0;
  int n2 = kMuPairNX / 2;
  int n3 = kMuPairNX - 1;
  const real_t y = rand * v[n3];
  while (n1 + 1 != n3) {
    if (y > v[n2]) { n1 = n2; } else { n3 = n2; }
    n2 = (n3 + n1 + 1) / 2;
  }
  real_t res = t.x[n1];
  const real_t del = v[n3] - v[n1];
  if (del > real_t(0)) { res += (y - v[n1]) * (t.x[n3] - res) / del; }
  return res;
}

/// `G4Physics2DVector::FindLinearX(rand, y)`.
template <typename real_t>
__host__ __device__ inline real_t mu_pair_find_linear_x(const MuPairSamplingTable<real_t>& t,
                                                        int iz, real_t rand, real_t yy) {
  const real_t y = fmin(fmax(yy, t.y[0]), t.y[t.ny - 1]);
  const int idy = mu_pair_find_bin(y, t.y, t.ny, t.ny - 2);
  const real_t x1 = mu_pair_interpolate_x(t, iz, idy, rand);
  const real_t x2 = mu_pair_interpolate_x(t, iz, idy + 1, rand);
  real_t res = x1;
  const real_t del = t.y[idy + 1] - t.y[idy];
  if (del != real_t(0)) { res += (x2 - x1) * (y - t.y[idy]) / del; }
  return res;
}

/// `G4MuPairProductionModel::FindScaledEnergy` for the table of `zdat` index `iz`.
template <typename real_t>
__host__ __device__ inline real_t mu_pair_find_scaled_energy(const MuPairSamplingTable<real_t>& t,
                                                             int iz, real_t rand,
                                                             real_t log_tkin, real_t yymin,
                                                             real_t yymax) {
  const real_t pmin = mu_pair_table_value(t, iz, yymin, log_tkin);
  const real_t pmax = mu_pair_table_value(t, iz, yymax, log_tkin);
  const real_t p0 = mu_pair_table_value(t, iz, real_t(0), log_tkin);
  // p0 <= 0 is DataCorrupted, a FatalException in Geant4; the table built above never has it.
  if (!(p0 > real_t(0))) { return yymin; }
  return mu_pair_find_linear_x(t, iz, (pmin + rand * (pmax - pmin)) / p0, log_tkin);
}

/// `G4MuPairProductionModel::SampleSecondaries` on element `z`, the pair's directions by
/// `G4ModifiedMephi::SamplePairDirections`. `cut` is the couple's POSITRON cut (the process's
/// secondary particle), floored at 4 m_e inside; maxEnergy is DBL_MAX. Returns the electron as
/// secondary 0 and the positron as secondary 1, in Geant4's push order.
template <typename real_t, typename Rng>
__host__ __device__ inline MuRadFinalState<real_t> sample_mu_pair(
    const MuPairSamplingTable<real_t>& t, int z, real_t kinetic, real_t mass, real_t cut,
    const Vec3<real_t>& dir, Rng& rng) {
  MuRadFinalState<real_t> out;
  out.primary_ekin = kinetic;
  out.primary_dir = dir;
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t min_pair = kMinPairEnergy<real_t>();
  const real_t total_energy = kinetic + mass;
  const real_t total_momentum = sqrt(kinetic * (kinetic + real_t(2) * mass));

  const real_t max_pair = mu_pair_max_energy(kinetic, mass, z);
  const real_t max_energy = max_pair;  // min(tmax = DBL_MAX, maxPairEnergy)
  const real_t min_energy = fmax(cut, min_pair);
  if (min_energy >= max_energy) { return out; }

  const real_t coeff = log(min_pair / kinetic) / t.ymin;
  const real_t yymin = log(min_energy / kinetic) / coeff;
  const real_t yymax = log(max_energy / kinetic) / coeff;
  const real_t log_tkin = log(kinetic);

  // The two tabulated elements around Z, as G4 keys them: an exact zdat match, the bracketing
  // pair, or uranium's table for Z > 92.
  int i1 = -1, i2 = -1;
  for (int iz = 0; iz < kMuPairNZ; ++iz) {
    if (z == mu_pair_zdat(iz)) {
      i1 = i2 = iz;
      break;
    } else if (z < mu_pair_zdat(iz)) {
      i2 = iz;
      i1 = (iz > 0) ? iz - 1 : iz;
      break;
    }
  }
  if (i1 < 0) { i1 = i2 = kMuPairNZ - 1; }

  real_t pair_energy = real_t(0);
  int count = 0;
  do {
    ++count;
    const real_t rand = rng.uniform();
    real_t x = mu_pair_find_scaled_energy(t, i1, rand, log_tkin, yymin, yymax);
    if (i1 != i2) {
      const real_t x2 = mu_pair_find_scaled_energy(t, i2, rand, log_tkin, yymin, yymax);
      // nist->GetLOGZ: G4Pow's logZ table, std::log of the integer.
      const real_t lz1 = log(real_t(mu_pair_zdat(i1)));
      const real_t lz2 = log(real_t(mu_pair_zdat(i2)));
      x += (x2 - x) * (log(real_t(z)) - lz1) / (lz2 - lz1);
    }
    pair_energy = kinetic * exp(x * coeff);
  } while ((pair_energy < min_energy || pair_energy > max_energy) && 10 > count);

  // r = (E+ - E-)/E_pair, uniform within its kinematic limit.
  const real_t rmax =
      (real_t(1) - real_t(6) * mass * mass / (total_energy * (total_energy - pair_energy)))
      * sqrt(real_t(1) - min_pair / pair_energy);
  const real_t r = rmax * (real_t(-1) + real_t(2) * rng.uniform());
  real_t e_energy = (real_t(1) - r) * pair_energy * real_t(0.5);
  real_t p_energy = pair_energy - e_energy;

  // G4ModifiedMephi::SamplePairDirections: phi, then the electron's polar angle, then the
  // positron's, both from the pair's total energy.
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  const real_t sinp = sin(phi);
  const real_t cosp = cos(phi);
  const real_t etwo = e_energy + p_energy;
  real_t cost = modified_mephi_cos_theta(kinetic, etwo, mass, rng);
  real_t sint = sqrt((real_t(1) - cost) * (real_t(1) + cost));
  const Vec3<real_t> edir = rotate_uz(Vec3<real_t>{sint * cosp, sint * sinp, cost}, dir);
  cost = modified_mephi_cos_theta(kinetic, etwo, mass, rng);
  sint = sqrt((real_t(1) - cost) * (real_t(1) + cost));
  const Vec3<real_t> pdir = rotate_uz(Vec3<real_t>{-sint * cosp, -sint * sinp, cost}, dir);

  e_energy = fmax(e_energy - me, real_t(0));
  p_energy = fmax(p_energy - me, real_t(0));
  out.n = 2;
  out.sec_type[0] = ParticleType::kElectron;
  out.sec_ekin[0] = e_energy;
  out.sec_dir[0] = edir;
  out.sec_type[1] = ParticleType::kPositron;
  out.sec_ekin[1] = p_energy;
  out.sec_dir[1] = pdir;

  // The muon: G4DynamicParticle::GetMomentum is direction * sqrt(T (T + 2 m)) for each lepton.
  const real_t pe = sqrt(e_energy * (e_energy + real_t(2) * me));
  const real_t pp = sqrt(p_energy * (p_energy + real_t(2) * me));
  const Vec3<real_t> pfin = total_momentum * dir - (pe * edir + pp * pdir);
  out.primary_dir = mu_unit(pfin);
  out.primary_ekin = kinetic - pair_energy;
  return out;
}

/// Both muon radiative cross sections per volume at the pre-step energy, for `step_hadron`.
/// `__noinline__` for the reason `had::inelastic_length` gives: the pair integral is 8 x 8 x 8
/// logarithms per element, and inlined into the muon kernels' `step_hadron` it would be one more
/// frame of locals in the function whose ptxas this project has fought hardest (V55, V195).
template <typename real_t>
__host__ __device__ __noinline__ void mu_radiative_xs(const data::Material<real_t>& m,
                                                      ParticleType type, real_t kinetic,
                                                      real_t gamma_cut, real_t positron_cut,
                                                      real_t& brem_xs, real_t& pair_xs) {
  brem_xs = mu_brem_xs(m, type, kinetic, gamma_cut);
  pair_xs = mu_pair_xs(m, type, kinetic, positron_cut);
}

/// The PostStepDoIt of whichever of the two fired: the element, then the final state.
/// `__noinline__` for the same reason as `mu_radiative_xs`.
template <typename real_t, typename Rng>
__host__ __device__ __noinline__ MuRadFinalState<real_t> mu_radiative_sample(
    const MuPairSamplingTable<real_t>& table, const data::Material<real_t>& m, bool pair,
    real_t kinetic, real_t mass, real_t gamma_cut, real_t positron_cut, const Vec3<real_t>& dir,
    Rng& rng) {
  const real_t cut = pair ? positron_cut : gamma_cut;
  const int ie = mu_rad_select_element(m, pair, kinetic, mass, cut, rng);
  const int z = static_cast<int>(m.z[ie] + real_t(0.5));
  return pair ? sample_mu_pair(table, z, kinetic, mass, cut, dir, rng)
              : sample_mu_brem(z, kinetic, mass, cut, dir, rng);
}

}  // namespace g4gpu::em
