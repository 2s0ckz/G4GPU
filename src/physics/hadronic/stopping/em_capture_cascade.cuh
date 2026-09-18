// G4EmCaptureCascade: the atomic cascade a stopped negative particle makes on its way down to
// the K shell of the mesonic atom, and the electrons and gammas it emits doing it.
//
// Transcribed from Geant4 11.1.1,
//   processes/hadronic/stopping/src/G4EmCaptureCascade.cc  (the constructor's K-level table and
//   ApplyYourself), and the inline AddNewParticle in the header.
//
// ---------------------------------------------------------------------------------------------
// **Every captured particle is treated as a muon.** The constructor takes
// `fMuMass = G4MuonMinus::MuonMinus()->GetPDGMass()` and `ApplyYourself` never looks at the
// projectile at all: the reduced mass, the level energies and therefore every gamma and every
// Auger electron are computed as if a muon had been captured, whether the stopped particle is a
// mu-, a pi-, a K-, a Sigma-, a Xi- or an Omega-. A pi- is 0.74 muon masses and a K- is 4.7, so
// the Bohr energies of a real pionic or kaonic atom differ from these by those factors; Geant4
// does not model that and neither does this port. It is not an approximation introduced here -
// it is what the class does - and it is stated because a reader comparing a kaonic-atom X-ray
// table against this code would otherwise conclude the port had a bug.
//
// **The K-shell energy is tabulated and everything above it is hydrogenic.** 28 measured
// (Z, E_K) pairs are stored, and the constructor fills the gaps by interpolating E/Z^2 LINEARLY
// in Z between neighbouring entries and multiplying back by Z^2 - so the stored quantity is the
// scaled one and the interpolation is in the scaled variable, which is what makes a 28-point
// table cover 92 elements. Levels 2 through 14 are `e/(i+1)^2` with
// `e = 13.6 eV * Z^2 * m_reduced/m_electron`; only the K shell gets the finite-nuclear-size
// correction the table carries.
//
// **The cascade starts on level 14 and ends on level 1, and the first thing it emits is an
// electron carrying the whole 14th-level energy.** Then, per step, either
//
//   * an AUGER electron, if `nAuger < nElec` and `(Z^4 + 10000) * u < 10000` - i.e. with
//     probability 10000/(Z^4 + 10000), which is 91% at Z = 1 and 0.004% at Z = 82, so light
//     elements de-excite by Auger emission and heavy ones radiate; it steps down exactly one
//     level; or
//   * a PHOTON, whose destination level is drawn as `var = (10 + nLevel - 1) * u` and then
//     `iLevel = nLevel - 1 - (int(var - 10) + 1)` when `var > 10` - a 10/(9 + nLevel) chance of
//     the single-step transition and the rest spread over the longer jumps.
//
// The loop ends when nLevel reaches 0, so it runs at most 13 times and emits at most 14
// secondaries. That bound is exact, not an estimate, and it is what sizes the buffer.
//
// **The local energy deposit is not a deposit.** `edep` accumulates every transition energy AND
// every one of those energies is also given to a real secondary, so the sum is not energy that
// stays behind - it is the total binding energy released. `G4HadronStoppingProcess` reads it as
// `ebound` and hands it to `thePro.SetBoundEnergy(ebound)` for the bound-decay model to use as
// the muon's kinetic energy in the K shell; it is NOT added to the step's local deposit. Calling
// it `SetLocalEnergyDeposit` is Geant4's naming, and a port that treated it as a deposit would
// double count the entire cascade.
#ifndef G4GPU_STOPPING_EM_CAPTURE_CASCADE_CUH
#define G4GPU_STOPPING_EM_CAPTURE_CASCADE_CUH

#include <cmath>

#include "core/vec3.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::stopping {

/// The most secondaries one EM cascade can emit: the level-14 electron plus one per step of a
/// loop that starts at nLevel = 13 and strictly decreases. Exact.
constexpr int kMaxEmCascadeSecondaries = 14;

/// One emitted particle of the atomic cascade.
struct EmCascadeProduct {
  int pdg = 0;              ///< 11 (e-) or 22 (gamma)
  double kin_energy = 0.0;  ///< MeV
  Vec3<double> direction{0.0, 0.0, 1.0};
};

/// What one cascade produced.
struct EmCascadeResult {
  int n = 0;
  double e_bound = 0.0;     ///< the `edep` Geant4 calls a local deposit; see the header
  EmCascadeProduct p[kMaxEmCascadeSecondaries];
};

/// The 28 measured K-shell energies, MeV, and the Z they belong to.
///
/// `G4EmCaptureCascade`'s constructor builds a 93-entry table from these once and keeps it; here
/// the table is built on demand by `k_level_energy(Z)` because the arithmetic is four lines and
/// a per-thread 93-double array is not worth carrying. The interpolation is reproduced exactly,
/// including that it is linear in E/Z^2 and not in E.
__host__ __device__ inline int em_k_level_count() { return 28; }

__host__ __device__ inline int em_k_level_z(int i) {
  const int listK[28] = {1,  2,  4,  6,  8,  11, 14, 17, 18, 21, 24, 26, 29, 32,
                         38, 40, 41, 44, 49, 53, 55, 60, 65, 70, 75, 81, 85, 92};
  return listK[i];
}

__host__ __device__ inline double em_k_level_value(int i) {
  const double listKEnergy[28] = {0.00275, 0.011,  0.043,  0.098,  0.173,  0.326,  0.524,
                                  0.765,   0.853,  1.146,  1.472,  1.708,  2.081,  2.475,
                                  3.323,   3.627,  3.779,  4.237,  5.016,  5.647,  5.966,
                                  6.793,   7.602,  8.421,  9.249,  10.222, 10.923, 11.984};
  return listKEnergy[i];
}

/// `fKLevelEnergy[Z]` after the constructor has run, for one Z. MeV.
///
/// Z = 0 is 0.0 (the constructor writes it and nothing reads it, since Z >= 1 for any element),
/// Z = 1 is the first table value, a tabulated Z is its own value, and a Z between two entries
/// is `(y1 + (y2-y1)*(Z-z1)/(z2-z1)) * Z^2` with `y = E/z^2` at the neighbours. Above 92 the
/// caller clamps, exactly as `fKLevelEnergy[std::min(Z, 92)]` does.
__host__ __device__ inline double k_level_energy(int z_in) {
  const int z = (z_in < 0) ? 0 : ((z_in > 92) ? 92 : z_in);
  if (z == 0) { return 0.0; }
  if (z == 1) { return em_k_level_value(0); }
  // The constructor walks the list with `idx` trailing `i`, filling the gap between them. A
  // tabulated Z is written directly; the loop below finds the same answer by looking for the
  // bracketing pair.
  int idx = 0;
  for (int i = 1; i < em_k_level_count(); ++i) {
    const int z1 = em_k_level_z(idx);
    const int z2 = em_k_level_z(i);
    if (z == z2) { return em_k_level_value(i); }
    if (z > z1 && z < z2) {
      const double dz = double(z2 - z1);
      const double y1 = em_k_level_value(idx) / double(z1 * z1);
      const double y2 = em_k_level_value(i) / double(z2 * z2);
      return (y1 + (y2 - y1) * double(z - z1) / dz) * double(z) * double(z);
    }
    idx = i;
  }
  return em_k_level_value(em_k_level_count() - 1);
}

/// The muon mass Geant4 uses for EVERY captured particle, MeV. G4MuonMinus's PDG mass, the same
/// value `physics/decay/decay_tables.hh` carries as `0.1056583715 * gev()`.
__host__ __device__ inline constexpr double em_cascade_muon_mass() { return 105.6583715; }

/// CLHEP's electron mass, MeV - the same number `core/units.cuh` pins.
__host__ __device__ inline constexpr double em_cascade_electron_mass() { return 0.510998910; }

/// `G4EmCaptureCascade::AddNewParticle`: one secondary, isotropic, with that kinetic energy.
///
/// The overflow is a REFUSAL by construction rather than a report: the loop that calls this is
/// bounded at 14 by its own arithmetic (see the header), so a full buffer would mean the bound is
/// wrong, and dropping a particle silently is the one thing that must not happen. The test pins
/// the bound by running every element from Z = 1 to 92 and asserting the count never reaches it.
template <typename Rng>
__host__ __device__ inline void add_em_product(EmCascadeResult& out, int pdg, double kin_energy,
                                               Rng& rng) {
  if (out.n >= kMaxEmCascadeSecondaries) { return; }
  EmCascadeProduct& q = out.p[out.n++];
  q.pdg = pdg;
  q.kin_energy = kin_energy;
  const deex::Vec3d d = deex::random_direction(rng);
  q.direction = Vec3<double>{d.x, d.y, d.z};
}

/// G4EmCaptureCascade::ApplyYourself.
///
/// `nuclear_mass_MeV` is `G4NucleiProperties::GetNuclearMass(A, Z)` for the captured-on nucleus;
/// it enters only through the reduced mass. `rng` is drawn once per step of the cascade, plus
/// two per emitted particle for `G4RandomDirection`.
template <typename Rng>
__host__ __device__ inline void em_capture_cascade(int z, double nuclear_mass_MeV, Rng& rng,
                                                   EmCascadeResult& out) {
  out.n = 0;
  out.e_bound = 0.0;

  const double mu = em_cascade_muon_mass();
  const double reduced = mu * nuclear_mass_MeV / (mu + nuclear_mass_MeV);
  // 13.6 eV in MeV is 13.6e-6; Geant4 writes `13.6 * eV` with eV = 1e-6 MeV.
  const double e = 13.6e-6 * double(z) * double(z) * reduced / em_cascade_electron_mass();

  double level[14];
  level[0] = k_level_energy(z);
  for (int i = 1; i < 14; ++i) { level[i] = e / double((i + 1) * (i + 1)); }

  const int n_elec = z;
  int n_auger = 1;
  int n_level = 13;
  const double p_gamma = double(z) * double(z) * double(z) * double(z);

  // "Capture on 14-th level": an electron carrying the whole level-14 energy.
  double edep = level[13];
  add_em_product(out, 11, level[13], rng);

  do {
    double delta_e;
    if ((n_auger < n_elec) && ((p_gamma + 10000.0) * double(rng.uniform()) < 10000.0)) {
      ++n_auger;
      delta_e = level[n_level - 1] - level[n_level];
      --n_level;
      add_em_product(out, 11, delta_e, rng);
    } else {
      // Wu and Wilets, Ann. Rev. Nuclear Sci. 19 (1969) 527, as Geant4 spells the draw.
      const double var = (10.0 + double(n_level - 1)) * double(rng.uniform());
      int i_level = n_level - 1;
      if (var > 10.0) { i_level -= int(var - 10.0) + 1; }
      if (i_level < 0) { i_level = 0; }
      delta_e = level[i_level] - level[n_level];
      n_level = i_level;
      add_em_product(out, 22, delta_e, rng);
    }
    edep += delta_e;
  } while (n_level > 0);

  out.e_bound = edep;
}

}  // namespace g4gpu::physics::hadronic::stopping

#endif  // G4GPU_STOPPING_EM_CAPTURE_CASCADE_CUH
