// G4MuonVDNuclearModel's sampling table: its constants, the 5 x 73 x 801 table and the host
// function that builds it - and nothing of the model that samples it.
//
// SPLIT OUT OF lepton_vd.cuh FOR THE INCLUDE GRAPH, not for taste. `host/hadronic_upload.cuh`
// builds this table at Upload, and until docs/RISK.md V233 it included `lepton_vd.cuh` to do so -
// which includes `bertini/cascade_interface.cuh`, the gamma chain and the light-target collider,
// so Bertini reached every translation unit that included the upload header: every stepping
// unit, the drivers, every test that uploads a scene. A table is data; this header holds the
// table and the model header includes it. `tests/test_emextra_models.cu` and the lepton models'
// own `muon_vd_apply` read `MuVdTable` exactly as before.
#pragma once

#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/xs/kokoulin_muon_xs.cuh"

namespace g4gpu::physics::hadronic::emextra {

// ---------------------------------------------------------------------------------------------
// G4MuonVDNuclearModel
// ---------------------------------------------------------------------------------------------

/// NBIN, nzdat, ntdat and the `zdat`/`adat`/`tdat` tables of G4MuonVDNuclearModel.
inline constexpr int kMuVdNBin = 800;
inline constexpr int kMuVdNZ = 5;
inline constexpr int kMuVdNT = 73;

__host__ __device__ inline const int* mu_vd_zdat() {
  static const int v[kMuVdNZ] = {1, 4, 13, 29, 92};
  return v;
}
__host__ __device__ inline const double* mu_vd_adat() {
  static const double v[kMuVdNZ] = {1.01, 9.01, 26.98, 63.55, 238.03};
  return v;
}
__host__ __device__ inline const double* mu_vd_tdat() {
  static const double v[kMuVdNT] = {
      1.e3,  2.e3,  3.e3,  4.e3,  5.e3,  6.e3,  7.e3,  8.e3,  9.e3,
      1.e4,  2.e4,  3.e4,  4.e4,  5.e4,  6.e4,  7.e4,  8.e4,  9.e4,
      1.e5,  2.e5,  3.e5,  4.e5,  5.e5,  6.e5,  7.e5,  8.e5,  9.e5,
      1.e6,  2.e6,  3.e6,  4.e6,  5.e6,  6.e6,  7.e6,  8.e6,  9.e6,
      1.e7,  2.e7,  3.e7,  4.e7,  5.e7,  6.e7,  7.e7,  8.e7,  9.e7,
      1.e8,  2.e8,  3.e8,  4.e8,  5.e8,  6.e8,  7.e8,  8.e8,  9.e8,
      1.e9,  2.e9,  3.e9,  4.e9,  5.e9,  6.e9,  7.e9,  8.e9,  9.e9,
      1.e10, 2.e10, 3.e10, 4.e10, 5.e10, 6.e10, 7.e10, 8.e10, 9.e10, 1.e11};
  return v;
}

/// `CutFixed` - the model's OWN member, set to `0.2*CLHEP::GeV` in its constructor and equal to
/// the cross section class's `CutFixed` by coincidence rather than by sharing.
__host__ __device__ inline constexpr double mu_vd_cut_fixed() { return 200.0; }

/// `g/mole` in Geant4's internal units - 6.2415090744607617e21, NOT 1.
///
/// `MakeSamplingTable` multiplies `adat[iz]` by it before handing the result to
/// `ComputeDDMicroscopicCrossSection`, which uses it as a pure mass number. Derived the way
/// CLHEP derives it rather than pasted, for the reason core/units.cuh gives for `barn`:
///
///   electronvolt = 1e-6 (MeV being 1),  e_SI = 1.602176634e-19,  joule = eV/e_SI
///   kilogram = joule*second^2/meter^2,  gram = 1e-3*kilogram,  mole = 1
///
/// with second = 1e9 ns and meter = 1000 mm. `emextra_kokoulin.csv`'s `gmole` row carries
/// CLHEP's own value and `tests/test_emextra_xs.cu` compares against it, so the derivation is
/// checked rather than trusted - the 2019 SI value of e is what makes this 6.24150907e21 and
/// not the 6.24150648e21 the older 1.602176487e-19 gives, a difference in the seventh digit.
__host__ __device__ inline double g_per_mole() {
  const double e_SI = 1.602176634e-19;
  const double joule = 1.0e-6 / e_SI;                      // electronvolt / e_SI
  const double second = 1.0e9 * units::ns<double>();
  const double meter = 1.0e3 * units::mm<double>();
  const double kilogram = joule * second * second / (meter * meter);
  return 1.0e-3 * kilogram;                                // gram, with mole = 1
}

/// The 5 x 73 x 801 table `MakeSamplingTable` builds, as a caller-owned struct.
///
/// `value[z][it][iy]` is the normalised cumulative cross section and `x[iy]` the logarithmic
/// abscissa, which is the same for every (z, it) and is therefore stored once.
/// `x[NBIN] = 0` is written by `pv->PutX(NBIN, 0.)` AFTER the loop and is the upper edge
/// ymax = 0, not a hole: the sampler reads `GetX(iy+1)` with iy at most NBIN-1.
struct MuVdTable {
  double x[kMuVdNBin + 1] = {0.0};
  double value[kMuVdNZ][kMuVdNT][kMuVdNBin] = {{{0.0}}};
};

/// G4MuonVDNuclearModel::MakeSamplingTable, on the host.
///
/// `AtomicWeight = adat[iz]*(g/mole)`, and `g/mole` is 6.24150934e21 in Geant4's internal
/// units - so the double-differential cross section is evaluated for a nucleus of A = 6.3e21
/// rather than A = 1.01. It changes nothing: A enters only through
/// `aeff = 0.22*A + 0.78*A^0.89`, which does not depend on the energy loss, so it factors out
/// of the integral and cancels against the `CrossSection` the table is normalised by. The
/// factor is applied anyway, because "this cancels" is a claim a test measures rather than a
/// licence to drop a line - `emextra_kokoulin.csv` dumps the cross section with BOTH arguments
/// and `tests/test_emextra_lepton.cu` compares the normalised tables built from each.
__host__ inline void mu_vd_make_sampling_table(MuVdTable& t, double muon_mass,
                                               double g_per_mole) {
  const double ymin = -5.0;
  const double ymax = 0.0;
  const double dy = (ymax - ymin) / kMuVdNBin;
  for (int iz = 0; iz < kMuVdNZ; ++iz) {
    const double A = mu_vd_adat()[iz] * g_per_mole;
    for (int it = 0; it < kMuVdNT; ++it) {
      const double KineticEnergy = mu_vd_tdat()[it];
      const double TotalEnergy = KineticEnergy + muon_mass;
      const double Maxep = TotalEnergy - 0.5 * units::proton_mass_c2<double>();
      double CrossSection = 0.0;
      const double c = std::log(Maxep / mu_vd_cut_fixed());
      int nbin = -1;
      double y = ymin - 0.5 * dy;
      double yy = ymin - dy;
      for (int i = 0; i < kMuVdNBin; ++i) {
        y += dy;
        const double x = std::exp(y);
        yy += dy;
        const double dx = std::exp(yy + dy) - std::exp(yy);
        const double ep = mu_vd_cut_fixed() * std::exp(c * x);
        CrossSection +=
            ep * dx * g4gpu::hadronic::xs::kokoulin::dd_microscopic_xs<double>(KineticEnergy, A, ep, muon_mass);
        if (nbin < kMuVdNBin) {
          ++nbin;
          t.value[iz][it][nbin] = CrossSection;
          t.x[nbin] = y;
        }
      }
      t.x[kMuVdNBin] = 0.0;
      if (CrossSection > 0.0) {
        for (int ib = 0; ib <= nbin && ib < kMuVdNBin; ++ib) {
          t.value[iz][it][ib] /= CrossSection;
        }
      }
    }
  }
}

}  // namespace g4gpu::physics::hadronic::emextra
