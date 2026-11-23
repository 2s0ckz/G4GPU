// P46: the muon's three discrete EM final states and the pair model's sampling tables, against
// QBBC's own model objects (ref/dump/dump_mu_radiative.cc; docs/RISK.md V346-V347).
//
// THREE CHECKS, EACH EXACT.
//
//   1. mu_pair_table.csv - every node of G4MuPairProductionModel's five G4Physics2DVectors
//      (MakeSamplingTables, 5 x 21 x 1001) against `em::build_mu_pair_sampling_table`: the axes
//      and the running sums, to a relative 1e-12. They are the same arithmetic in the same order;
//      what is left is std::pow against G4Pow's cube-root table, a few ulps.
//
//   2. mu_rad_xs.csv - the rates the stepper draws its lengths from, at the couples' own cuts:
//      `hadron_delta_xs` (the TRANSPORT's call, which since P46 is G4MuBetheBlochModel's for a
//      muon above 200 keV), `mu_brem_xs` at the photon cut and `mu_pair_xs` at the positron cut,
//      against the models' CrossSectionPerVolume, to 1e-8. The lambda tables Geant4's integral
//      approach reads instead are printed beside them, not asserted: that difference is the
//      port's convention (docs/RISK.md V78) and this file measures its size.
//
//   3. mu_tape.csv - SampleSecondaries replayed deviate for deviate. Every deviate each Geant4
//      call drew is in the file; the port's sampler is handed the same sequence and must draw
//      exactly as many, and produce the same secondaries and the same primary to 1e-12. The
//      element is Geant4's (its first deviate, in a compound, is G4EmElementSelector's table
//      draw, which the port makes against partials at the energy instead); the port's own
//      choice from that same deviate is counted and printed, and asserted to agree on all but a
//      handful - the two differ only where the table's interpolation moves a boundary past the
//      deviate.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/particle.cuh"
#include "data/materials.cuh"
#include "physics/em/hadron_delta.cuh"
#include "physics/em/hadron_range.cuh"
#include "physics/em/muon_radiative.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::size_t b = 0;
  for (;;) {
    const std::size_t e = s.find(c, b);
    out.push_back(s.substr(b, e == std::string::npos ? std::string::npos : e - b));
    if (e == std::string::npos) { break; }
    b = e + 1;
  }
  return out;
}

std::vector<std::vector<std::string>> read_csv(const std::string& path) {
  std::vector<std::vector<std::string>> rows;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return rows; }
  std::string line;
  char buf[65536];
  bool header = true;
  while (std::fgets(buf, sizeof buf, f) != nullptr) {
    line = buf;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) { line.pop_back(); }
    if (header) {
      header = false;
      continue;
    }
    if (!line.empty()) { rows.push_back(split(line, ',')); }
  }
  std::fclose(f);
  return rows;
}

int material_of(const std::string& n) {
  if (n == "G4_AIR") { return data::kAir; }
  if (n == "G4_WATER") { return data::kWater; }
  if (n == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (n == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;
}

/// Hands out a recorded sequence; past its end, 0.5 and a count of the overrun.
struct ReplayRng {
  const double* u = nullptr;
  std::size_t size = 0;
  std::size_t n = 0;
  __host__ __device__ double uniform() {
    const double r = (n < size) ? u[n] : 0.5;
    ++n;
    return r;
  }
};

double rel(double a, double b) {
  const double d = std::fabs(a - b);
  const double s = std::fmax(std::fabs(a), std::fabs(b));
  return (s > 0) ? d / s : 0.0;
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  static em::MuPairSamplingTable<real_t> table;
  const double mu_mass = particle_def<real_t>(ParticleType::kMuonMinus).mass;
  em::build_mu_pair_sampling_table<real_t>(table, mu_mass);
  int fails = 0;

  // ---------------------------------------------------------------- 1. the pair tables
  {
    const auto rows = read_csv(dir + "/mu_pair_table.csv");
    if (rows.empty()) {
      std::printf("cannot read %s/mu_pair_table.csv - run the mu_radiative dump\n", dir.c_str());
      return 1;
    }
    double wx = 0, wy = 0, wv = 0;
    long n = 0, ny_seen = 0;
    for (const auto& r : rows) {
      const int z = std::atoi(r[0].c_str());
      const int iy = std::atoi(r[1].c_str());
      const int ix = std::atoi(r[3].c_str());
      int iz = -1;
      for (int k = 0; k < em::kMuPairNZ; ++k) {
        if (em::mu_pair_zdat(k) == z) { iz = k; }
      }
      if (iz < 0 || iy >= table.ny || ix >= em::kMuPairNX) {
        std::printf("  FAIL: table node Z=%d iy=%d ix=%d is outside the port's %d x %d\n", z, iy,
                    ix, table.ny, em::kMuPairNX);
        ++fails;
        continue;
      }
      if (iy + 1 > ny_seen) { ny_seen = iy + 1; }
      wy = std::fmax(wy, rel(table.y[iy], std::atof(r[2].c_str())));
      wx = std::fmax(wx, rel(table.x[ix], std::atof(r[4].c_str())));
      const double g = std::atof(r[5].c_str());
      // The first nodes are ~1e-34 against sums of ~1e-25: relative to the row's last value.
      const double scale = std::fmax(std::fabs(g), 1e-12 * table.v[iz][iy][em::kMuPairNX - 1]);
      wv = std::fmax(wv, std::fabs(table.v[iz][iy][ix] - g) / scale);
      ++n;
    }
    const bool ok = (n == static_cast<long>(em::kMuPairNZ) * table.ny * em::kMuPairNX)
                    && (ny_seen == table.ny) && wx < 1e-12 && wy < 1e-12 && wv < 1e-12;
    std::printf("1. pair sampling tables: %ld nodes (%d energies), worst rel  x %.2e  y %.2e  "
                "value %.2e  %s\n",
                n, table.ny, wx, wy, wv, ok ? "ok" : "FAIL");
    if (!ok) { ++fails; }
  }

  // ---------------------------------------------------------------- 2. the rates
  {
    const auto rows = read_csv(dir + "/mu_rad_xs.csv");
    if (rows.empty()) {
      std::printf("cannot read %s/mu_rad_xs.csv\n", dir.c_str());
      return 1;
    }
    double w[3] = {0, 0, 0}, t[3] = {0, 0, 0};
    std::string ww[3], tw[3];
    int n = 0;
    for (const auto& r : rows) {
      const int mi = material_of(r[0]);
      if (mi < 0) { continue; }
      const ParticleType pt = (r[1] == "mu-") ? ParticleType::kMuonMinus : ParticleType::kMuonPlus;
      const double e = std::atof(r[2].c_str());
      const double gc = std::atof(r[3].c_str()), ec = std::atof(r[4].c_str()),
                   pc = std::atof(r[5].c_str());
      const data::Material<real_t>& m = mats[mi];
      // The cuts the stepper passes are the material's own; they must be the couple's.
      if (rel(m.cut_gamma, gc) > 1e-8 || rel(m.cut_electron, ec) > 1e-8
          || rel(m.cut_positron, pc) > 1e-8) {
        std::printf("  FAIL: %s cuts (%g %g %g) are not the couple's (%g %g %g)\n", r[0].c_str(),
                    m.cut_gamma, m.cut_electron, m.cut_positron, gc, ec, pc);
        ++fails;
        break;
      }
      double bx = 0, px = 0;
      em::mu_radiative_xs<real_t>(m, pt, e, m.cut_gamma, m.cut_positron, bx, px);
      const double ours[3] = {em::hadron_delta_xs<real_t>(m, pt, e, m.cut_electron, 1e30), bx, px};
      for (int k = 0; k < 3; ++k) {
        const double g = std::atof(r[6 + k].c_str());
        const double tab = std::atof(r[9 + k].c_str());
        char b[160];
        std::snprintf(b, sizeof b, "%s %s %.4g MeV (ours %.10g, G4 %.10g)", r[1].c_str(),
                      r[0].c_str(), e, ours[k], g);
        const double d = (g > 0) ? rel(ours[k], g) : std::fabs(ours[k]);
        if (d > w[k]) { w[k] = d; ww[k] = b; }
        if (g > 0 && tab > 0 && rel(tab, g) > t[k]) {
          t[k] = rel(tab, g);
          std::snprintf(b, sizeof b, "%s %s %.4g MeV", r[1].c_str(), r[0].c_str(), e);
          tw[k] = b;
        }
      }
      ++n;
    }
    const char* names[3] = {"muIoni (G4MuBetheBlochModel)", "muBrems", "muPairProd"};
    std::printf("2. rates per volume, %d rows (100 MeV - 100 TeV, four materials, both muons):\n",
                n);
    for (int k = 0; k < 3; ++k) {
      // 1e-8 for the ionisation: its rate is per ELECTRON times the material's electron density,
      // which data/materials.cuh carries to 1.6e-9 of Geant4's in bone (measured here); the two
      // radiative rates are per atom times the atom densities and agree to 3e-10 and 5e-16.
      const bool ok = w[k] < 1e-8;
      std::printf("   %-30s model worst %.2e %s  [Geant4's own table vs its model: %.2e at %s]\n",
                  names[k], w[k], ok ? "ok" : "FAIL", t[k], tw[k].c_str());
      if (!ok) {
        std::printf("      at %s\n", ww[k].c_str());
        ++fails;
      }
    }
  }

  // ---------------------------------------------------------------- 3. the tape
  {
    const auto rows = read_csv(dir + "/mu_tape.csv");
    if (rows.empty()) {
      std::printf("cannot read %s/mu_tape.csv\n", dir.c_str());
      return 1;
    }
    struct Acc {
      int n = 0, bad = 0, el_n = 0, el_diff = 0;
      double el_emax = 0;
      double we = 0, wd = 0;
    } acc[3];
    const char* procs[3] = {"muIoni", "muBrems", "muPairProd"};
    for (const auto& r : rows) {
      int k = -1;
      for (int j = 0; j < 3; ++j) {
        if (r[0] == procs[j]) { k = j; }
      }
      const int mi = material_of(r[2]);
      if (k < 0 || mi < 0 || r.size() < 23) { continue; }
      const ParticleType pt = (r[1] == "mu-") ? ParticleType::kMuonMinus : ParticleType::kMuonPlus;
      const double e = std::atof(r[3].c_str());
      const double cut = std::atof(r[4].c_str());
      const int z = std::atoi(r[5].c_str());
      const std::size_t nu = static_cast<std::size_t>(std::atol(r[6].c_str()));
      std::vector<double> u;
      if (r[7] != "-") {
        for (const auto& s : split(r[7], ';')) { u.push_back(std::atof(s.c_str())); }
      }
      const int nsec = std::atoi(r[8].c_str());
      const data::Material<real_t>& m = mats[mi];
      const Vec3<real_t> dir0 = normalize(Vec3<real_t>{0.3, -0.4, 0.866});
      // CLHEP's unit(), which the dump used: the same as mu_unit.
      const Vec3<real_t> d0 = em::mu_unit(Vec3<real_t>{0.3, -0.4, 0.866});
      (void)dir0;

      // The port's answer, on the same deviates.
      ReplayRng rng;
      rng.u = u.data();
      rng.size = u.size();
      int n_out = 0;
      double se[2] = {0, 0};
      Vec3<real_t> sd[2] = {};
      int st[2] = {-1, -1};
      double pe = e;
      Vec3<real_t> pdir = d0;
      if (k == 0) {
        const auto d = em::sample_hadron_delta<real_t>(pt, e, cut, d0, rng);
        if (d.produced) {
          n_out = 1;
          st[0] = 1;
          se[0] = d.delta_ekin;
          sd[0] = d.delta_dir;
          pe = d.primary_ekin;
          pdir = d.primary_dir;
        }
      } else {
        const double mass = particle_def<real_t>(pt).mass;
        // The element: Geant4's, after its selector's deviate where there is one; and, from
        // that same deviate, the port's own choice, counted.
        if (m.n_elements > 1) {
          ReplayRng r0;
          r0.u = u.data();
          r0.size = u.size();
          const int ie = em::mu_rad_select_element<real_t>(m, k == 2, e, mass, cut, r0);
          ++acc[k].el_n;
          if (static_cast<int>(m.z[ie] + 0.5) != z) {
            ++acc[k].el_diff;
            acc[k].el_emax = std::fmax(acc[k].el_emax, e);
          }
          rng.n = 1;
        }
        const auto fs = (k == 2) ? em::sample_mu_pair<real_t>(table, z, e, mass, cut, d0, rng)
                                 : em::sample_mu_brem<real_t>(z, e, mass, cut, d0, rng);
        n_out = fs.n;
        for (int j = 0; j < fs.n; ++j) {
          st[j] = (fs.sec_type[j] == ParticleType::kGamma)      ? 0
                  : (fs.sec_type[j] == ParticleType::kElectron) ? 1
                                                                : 2;
          se[j] = fs.sec_ekin[j];
          sd[j] = fs.sec_dir[j];
        }
        pe = fs.primary_ekin;
        pdir = fs.primary_dir;
      }

      // Against Geant4's.
      bool bad = (rng.n != nu) || (n_out != nsec);
      double we = 0, wd = 0;
      for (int j = 0; j < 2 && !bad; ++j) {
        const int gt = std::atoi(r[9 + 5 * j].c_str());
        if (j >= nsec) { break; }
        if (gt != st[j]) { bad = true; break; }
        we = std::fmax(we, rel(se[j], std::atof(r[10 + 5 * j].c_str())));
        wd = std::fmax(wd, std::fabs(sd[j].x - std::atof(r[11 + 5 * j].c_str())));
        wd = std::fmax(wd, std::fabs(sd[j].y - std::atof(r[12 + 5 * j].c_str())));
        wd = std::fmax(wd, std::fabs(sd[j].z - std::atof(r[13 + 5 * j].c_str())));
      }
      if (!bad) {
        we = std::fmax(we, rel(pe, std::atof(r[19].c_str())));
        wd = std::fmax(wd, std::fabs(pdir.x - std::atof(r[20].c_str())));
        wd = std::fmax(wd, std::fabs(pdir.y - std::atof(r[21].c_str())));
        wd = std::fmax(wd, std::fabs(pdir.z - std::atof(r[22].c_str())));
      }
      if (bad || we > 1e-12 || wd > 1e-12) {
        if (acc[k].bad < 5) {
          std::printf("  FAIL: %s %s %s %.6g MeV: deviates %zu (G4 %zu), secondaries %d (G4 %d), "
                      "worst energy %.2e direction %.2e\n",
                      r[0].c_str(), r[1].c_str(), r[2].c_str(), e, rng.n, nu, n_out, nsec, we,
                      wd);
        }
        ++acc[k].bad;
      }
      ++acc[k].n;
      acc[k].we = std::fmax(acc[k].we, we);
      acc[k].wd = std::fmax(acc[k].wd, wd);
    }
    std::printf("3. SampleSecondaries replayed deviate for deviate:\n");
    for (int k = 0; k < 3; ++k) {
      const bool ok = acc[k].n > 0 && acc[k].bad == 0;
      std::printf("   %-11s %4d calls, %d off; worst energy %.2e, direction %.2e  %s", procs[k],
                  acc[k].n, acc[k].bad, acc[k].we, acc[k].wd, ok ? "ok" : "FAIL");
      if (acc[k].el_n > 0) {
        std::printf("   [element from the same deviate: %d of %d differ from G4's table, all at "
                    "or below %.4g MeV]",
                    acc[k].el_diff, acc[k].el_n, acc[k].el_emax);
      }
      std::printf("\n");
      if (!ok) { ++fails; }
      // The partials-at-the-energy convention may disagree with the table where a boundary sits
      // within interpolation error of the deviate; more than 2% of calls would be a real error.
      if (acc[k].el_n > 0 && acc[k].el_diff * 50 > acc[k].el_n) {
        std::printf("   FAIL: the element draw disagrees too often\n");
        ++fails;
      }
    }
  }

  std::printf("\n%s\n", fails == 0 ? "PASS" : "FAIL");
  return fails == 0 ? 0 : 1;
}
