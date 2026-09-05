// Muon radiative losses: bremsstrahlung and direct pair production.
//
// Transcribed from G4MuBremsstrahlungModel and G4MuPairProductionModel (11.1.1).
// G4EmBuilder::ConstructCharged registers both for mu+ and mu- whenever the physics list
// reaches high energy, which G4EmStandardPhysics does (MaxKinEnergy defaults to 100 TeV).
//
// A muon radiates far less than an electron - the cross section scales as 1/m^2 - so these
// only matter above roughly 100 GeV, which is why G4MuBremsstrahlungModel sets its lowest
// kinetic energy to exactly that.
#pragma once
#include <cmath>
#include "core/particle.cuh"
#include "core/units.cuh"
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

}  // namespace g4gpu::em
