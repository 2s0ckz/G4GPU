// Livermore Rayleigh (coherent) scattering: G4EMLOW epics2017/rayl.
//
// Transcribed from G4LivermoreRayleighModel::ComputeCrossSectionPerAtom. The data files
// store E^2 * sigma, so the cross section is table(E)/E^2, flat-extrapolated above the last
// point exactly as Geant4 does.
//
// Rayleigh transfers no energy - it only redirects the photon - so it affects the dose only
// through attenuation and the resulting flux. The angular distribution is the Cullen
// three-term form-factor fit that G4RayleighAngularGenerator uses, for every Z from 1 to 100.
#pragma once
#include <cmath>
#include <string>
#include <vector>
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/photoelectric_data.cuh"  // reuses load_cs_file and interp_block
#include "data/rayleigh_angular_tables.cuh"

namespace g4gpu::data {

constexpr int kRayleighMaxElements = 16;

template <typename real_t>
struct RayleighElement {
  int z;
  int offset, count;
};

template <typename real_t>
struct RayleighTable {
  RayleighElement<real_t> elements[kRayleighMaxElements];
  int n_elements;
  int z_to_index[101];
  const real_t* table_e;
  const real_t* table_v;
  int table_n;
};

template <typename real_t>
__host__ inline bool load_rayleigh(const std::string& dir, const int* zs, int n_z,
                                   RayleighTable<real_t>& out, std::vector<real_t>& te,
                                   std::vector<real_t>& tv) {
  for (int i = 0; i <= 100; ++i) { out.z_to_index[i] = -1; }
  out.n_elements = 0;
  te.clear();
  tv.clear();
  for (int i = 0; i < n_z; ++i) {
    const int z = zs[i];
    if (out.z_to_index[z] >= 0) { continue; }
    if (out.n_elements >= kRayleighMaxElements) { return false; }
    RayleighElement<real_t>& e = out.elements[out.n_elements];
    e.z = z;
    // The files are already in Geant4 internal units, so no barn rescaling: pass 1.
    if (!load_cs_file_scaled<real_t>(dir + "/re-cs-" + std::to_string(z) + ".dat", te, tv,
                                     e.offset, e.count, real_t(1))) {
      std::printf("rayleigh: cannot read re-cs-%d.dat in %s\n", z, dir.c_str());
      return false;
    }
    out.z_to_index[z] = out.n_elements;
    ++out.n_elements;
  }
  out.table_n = static_cast<int>(te.size());
  return true;
}

/// Rayleigh cross section per atom, mm^2.
template <typename real_t>
__host__ __device__ inline real_t rayleigh_xs_per_atom(const RayleighTable<real_t>& t, int z,
                                                       real_t energy) {
  if (z < 1 || z > 100 || energy <= real_t(0)) { return real_t(0); }
  const int i = t.z_to_index[z];
  if (i < 0) { return real_t(0); }
  const RayleighElement<real_t>& e = t.elements[i];
  if (e.count <= 0) { return real_t(0); }

  const real_t e_last = t.table_e[e.offset + e.count - 1];
  const real_t e_first = t.table_e[e.offset];
  if (energy >= e_last) { return t.table_v[e.offset + e.count - 1] / (energy * energy); }
  if (energy >= e_first) {
    return interp_block(t.table_e, t.table_v, e.offset, e.count, energy) / (energy * energy);
  }
  return real_t(0);
}

/// Cullen fit parameters PP0..PP8, indexed by Z, from src/data/rayleigh_angular_tables.cuh.
/// Extraction was checked against the sum rule PP0+PP1+PP2 = Z^2.
struct RayleighParams {
  double p0, p1, p2, p3, p4, p5, p6, p7, p8;
};

/// fFactor = 0.5 * (cm / (h_Planck * c_light))^2, in 1/MeV^2.
///
/// DERIVED, not a literal, and the reason is that the literal was wrong. It read
/// `h*c = 1.23984193e-18 MeV*mm`; the true value is 2*pi*hbarc = 1.23984e-9 MeV*mm, nine
/// orders of magnitude larger. That made `xx` 1e18 too big, so `x/(b*xx)` underflowed to zero,
/// so `cost = 1 - 0` and every Rayleigh scatter came out perfectly forward. Rayleigh was
/// consuming steps and deflecting nothing.
///
/// It hid for as long as it did because coherent scattering transfers no energy: the process
/// only turns a photon, so a broken deflection costs a fraction of a per cent of dose and
/// nothing else. build_all.bat had been printing the evidence every run -
/// `rayleigh off: ... (+0.0017, 0.0 sigma)` - a process whose removal changes the answer by
/// zero sigma is a process that is not doing anything.
///
/// CLHEP: h_Planck = 2*pi*hbar_Planck and hbarc = hbar_Planck*c_light, so
/// h_Planck*c_light = 2*pi*hbarc. Written that way here so it cannot drift from units.cuh.
template <typename real_t> __host__ __device__ constexpr real_t rayleigh_factor() {
  constexpr real_t hc = units::twopi<real_t>() * units::hbarc<real_t>();  // MeV*mm
  constexpr real_t cm = real_t(10);                                      // mm
  return real_t(0.5) * (cm / hc) * (cm / hc);
}

/// Coherent scattering deflection, transcribed from
/// G4RayleighAngularGenerator::SampleDirection (the Cullen three-term fit).
template <typename real_t, typename Rng>
__host__ __device__ inline Vec3<real_t> sample_rayleigh_direction(int z, real_t gamma_energy,
                                                                  const Vec3<real_t>& dir,
                                                                  Rng& rng) {
  if (z < 1 || z >= kRaylPPSize) { return dir; }  // outside the fitted set: leave undeflected
  const RayleighParams params{rayl_pp0<double>()[z], rayl_pp1<double>()[z],
                              rayl_pp2<double>()[z], rayl_pp3<double>()[z],
                              rayl_pp4<double>()[z], rayl_pp5<double>()[z],
                              rayl_pp6<double>()[z], rayl_pp7<double>()[z],
                              rayl_pp8<double>()[z]};
  const RayleighParams* P = &params;

  const real_t xx = rayleigh_factor<real_t>() * gamma_energy * gamma_energy;
  const real_t n0 = real_t(P->p6) - real_t(1), n1 = real_t(P->p7) - real_t(1),
               n2 = real_t(P->p8) - real_t(1);
  const real_t b0 = real_t(P->p3), b1 = real_t(P->p4), b2 = real_t(P->p5);
  constexpr real_t numlim = real_t(0.02);

  real_t x = real_t(2) * xx * b0;
  const real_t w0 = (x < numlim)
                        ? n0 * x * (real_t(1) - real_t(0.5) * (n0 - real_t(1)) * x
                                                    * (real_t(1) - (n0 - real_t(2)) * x / real_t(3)))
                        : real_t(1) - exp(-n0 * log(real_t(1) + x));
  x = real_t(2) * xx * b1;
  const real_t w1 = (x < numlim)
                        ? n1 * x * (real_t(1) - real_t(0.5) * (n1 - real_t(1)) * x
                                                    * (real_t(1) - (n1 - real_t(2)) * x / real_t(3)))
                        : real_t(1) - exp(-n1 * log(real_t(1) + x));
  x = real_t(2) * xx * b2;
  const real_t w2 = (x < numlim)
                        ? n2 * x * (real_t(1) - real_t(0.5) * (n2 - real_t(1)) * x
                                                    * (real_t(1) - (n2 - real_t(2)) * x / real_t(3)))
                        : real_t(1) - exp(-n2 * log(real_t(1) + x));

  const real_t x0 = w0 * real_t(P->p0) / (b0 * n0);
  const real_t x1 = w1 * real_t(P->p1) / (b1 * n1);
  const real_t x2 = w2 * real_t(P->p2) / (b2 * n2);

  real_t cost = real_t(1);
  for (int guard = 0; guard < 1000; ++guard) {
    real_t w = w0, nn = n0, b = b0;
    x = rng.uniform() * (x0 + x1 + x2);
    if (x > x0) {
      x -= x0;
      if (x <= x1) { w = w1; nn = n1; b = b1; }
      else         { w = w2; nn = n2; b = b2; }
    }
    nn = real_t(1) / nn;
    const real_t y = w * rng.uniform();
    x = (y < numlim) ? y * nn * (real_t(1) + real_t(0.5) * (nn + real_t(1)) * y
                                                 * (real_t(1) - (nn + real_t(2)) * y / real_t(3)))
                     : exp(-nn * log(real_t(1) - y)) - real_t(1);
    cost = real_t(1) - x / (b * xx);
    if (real_t(2) * rng.uniform() <= real_t(1) + cost * cost && cost >= real_t(-1)) { break; }
  }
  if (cost < real_t(-1)) { cost = real_t(-1); }
  if (cost > real_t(1)) { cost = real_t(1); }
  const real_t sint = sqrt(fmax(real_t(0), (real_t(1) - cost) * (real_t(1) + cost)));
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  return normalize(rotate_uz(Vec3<real_t>{sint * cos(phi), sint * sin(phi), cost}, dir));
}

}  // namespace g4gpu::data
