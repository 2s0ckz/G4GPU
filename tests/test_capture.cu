// G4NeutronRadCapture, against ref/oracle/capture_*.csv.
//
// Three disciplines, in the order they can fail usefully.
//
//   1. THE MASS BALANCE, exactly. `capture_masses.csv` carries
//      G4NucleiProperties::GetNuclearMass for every target and its compound, read out of the
//      public function. The capture Q value is the difference of two ~2e5 MeV numbers, so it is
//      the one place in this model where a last-bit error in a mass table becomes a per-cent
//      error in a gamma energy: on Pb208 the Q is 3.94 MeV out of 194 GeV, a cancellation of
//      five decades. Checked first because everything else is built on it.
//
//   2. THE WHOLE FINAL STATE, exactly, under a prescribed uniform cycle. `capture_det.csv` runs
//      ApplyYourself with an engine that returns eight known values in rotation, so the model is
//      a deterministic function of (target, energy, phase) and the comparison is not statistical
//      at all - gamma energies, directions, the residual's recoil, the cascade's time, and the
//      NUMBER OF UNIFORMS CONSUMED. The draw count is the control-flow check: a port that took a
//      different branch of GenerateGamma reads a different uniform from then on, and the count
//      says so in one integer instead of in twenty disagreeing doubles.
//
//   3. THE DISTRIBUTIONS, statistically. `capture_stat.csv` is 20,000 captures per (target,
//      energy) with the real engine at a fixed seed. The port cannot follow HepJamesRandom's
//      stream, so this compares moments and histograms with a stated tolerance - and it is not
//      redundant with (2), which exercises eight paths where this exercises every branch weight
//      in the level scheme.
//
// WHAT THE PORT DOES NOT REPRODUCE, AND IS MEASURED HERE RATHER THAN ASSERTED
//
//   * The residual ion's PDG CODE. Geant4 asks G4IonTable::GetIon(Z, A, E*, noFloat, 0), which
//     returns a code whose last digit is the isomer index - `1000260572` for Fe57 at 136 keV -
//     and the port returns the ground-state encoding `1000260570`. The code is not compared;
//     (Z, A) and the MASS are, which is what a kinetic energy is built from.
//   * The SNAPPING of the residual's excitation onto G4ENSDFSTATE. That one is measured: the
//     port carries the raw PhotonEvaporation5.7 level energy the cascade stopped on, Geant4
//     carries whatever G4NuclideTable matched it to, and this test counts how many of the two
//     agree. docs/RISK.md V40 is why that is a question and not an assumption - a compiled table
//     and a data file that answer the same question disagreed for 952 nuclides once already.
//   * The photon-evaporation state carried ACROSS captures. See the `persistent` argument of
//     neutron_rad_capture_apply: Geant4 keeps one G4PhotonEvaporation for the run and its
//     `fIndex` survives, which can change a NearestLevelIndex answer after a cascade that ended
//     on an isomer. The deterministic oracle uses a fresh model per call so (2) is about the
//     transcription; the statistical oracle uses one model for all 20,000, so (3) is where that
//     difference would show, and the tolerance it passes at is the bound on it.
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/capture/capture_process.cuh"
#include "physics/hadronic/deexcitation/corrections.cuh"

using namespace g4gpu;
using namespace g4gpu::physics::hadronic;
using real_t = double;

namespace {

int g_fails = 0;

void fail(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::printf("  FAIL: ");
  std::vprintf(fmt, ap);
  va_end(ap);
  std::printf("\n");
  ++g_fails;
}

/// One comparison bucket: how many points, the worst relative deviation and where.
struct Cell {
  const char* name = "";
  int n = 0;
  double worst = 0;
  std::string where;
};

double deviation(double ours, double g4) {
  const double floor = 1e-300;
  if (std::fabs(g4) > floor) { return std::fabs(ours - g4) / std::fabs(g4); }
  return (std::fabs(ours) > floor) ? 1.0 : 0.0;
}

void cmp(Cell& c, double ours, double g4, const std::string& what) {
  ++c.n;
  const double d = deviation(ours, g4);
  if (d > c.worst) {
    c.worst = d;
    char buf[512];
    std::snprintf(buf, sizeof buf, "%s (ours %.17g, G4 %.17g)", what.c_str(), ours, g4);
    c.where = buf;
  }
}

int report(Cell& c, double tol) {
  if (c.n == 0) {
    std::printf("  %-34s NO POINTS\n", c.name);
    return 1;
  }
  const bool ok = (c.worst <= tol);
  std::printf("  %-34s %7d points, worst %.3e %s\n", c.name, c.n, c.worst,
              ok ? "" : "  <-- FAIL");
  if (!ok) { std::printf("      %s\n", c.where.c_str()); }
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------------------------
// The prescribed uniform cycle, mirroring ref/dump/dump_capture.cc's CycleEngine.
// ---------------------------------------------------------------------------------------------

/// The eight values dump_capture.cc's CycleEngine returns, in its order. Retyped here rather
/// than shared, for the reason the dump gives: the two are separate translation units and the
/// only thing that has to agree is the numbers. `draws` is what makes the control flow
/// checkable.
struct CycleRng {
  int phase = 0;
  int draws = 0;
  __host__ __device__ double uniform() {
    static const double seq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
    const double v = seq[(draws + phase) % 8];
    ++draws;
    return v;
  }
};

// ---------------------------------------------------------------------------------------------
// CSV reading.
// ---------------------------------------------------------------------------------------------

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\r' && c != '\n') {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

std::vector<std::vector<std::string>> read_csv(const std::string& path, bool& ok) {
  std::vector<std::vector<std::string>> rows;
  FILE* f = std::fopen(path.c_str(), "r");
  ok = (f != nullptr);
  if (!ok) {
    std::printf("FAIL: cannot read %s - run ref/oracle/run.bat tables in this worktree first\n",
                path.c_str());
    return rows;
  }
  char line[4096];
  bool first = true;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    if (first) {
      first = false;
      continue;
    }
    if (line[0] == '\0' || line[0] == '\n') { continue; }
    rows.push_back(split(line));
  }
  std::fclose(f);
  return rows;
}

double num(const std::string& s) { return std::atof(s.c_str()); }
int inum(const std::string& s) { return std::atoi(s.c_str()); }

// ---------------------------------------------------------------------------------------------

HadProjectile<real_t> make_neutron(real_t ekin) {
  HadProjectile<real_t> p;
  p.pdg = 2112;
  p.baryon_number = 1;
  p.charge = real_t(0);
  // G4HadProjectile takes theMass from the DEFINITION's PDG mass, so this is G4Neutron's
  // 939.56536 MeV and not a dynamic mass. See particle.cuh's kNeutron row for the transposed
  // digits in G4Neutron.cc's comment that this is not.
  p.mass = units::neutron_mass_c2<real_t>();
  p.kin_energy = ekin;
  return p;
}

/// The final state capacity. A thermal capture on a heavy nuclide emits up to about ten gammas
/// plus the residual; 32 is generous and its overflow is reported rather than silent, which is
/// the property that matters.
constexpr int kCap = 32;

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";

  const std::string pe = host::g4photon_evaporation_dir();
  if (pe.empty()) {
    std::printf("FAIL: PhotonEvaporation dataset not found - g4data.cuh could not resolve "
                "G4LEVELGAMMADATA\n");
    return 1;
  }
  data::LevelTableStorage lts;
  data::read_all_level_data(
      lts, pe, data::kLevelZMax,
      [](int Z, int A) { return deex::shell_correction(A, Z); },
      [](int Z, int A) { return deex::level_manager_level_density(Z, A); });
  const data::LevelTable lt = lts.view();
  std::printf("level table: %d managers\n", lt.n_managers);

  // ------------------------------------------------------------------ 1. the mass balance
  {
    bool ok = false;
    const auto rows = read_csv(dir + "/capture_masses.csv", ok);
    if (!ok) { return 1; }
    Cell target{"target mass"}, compound{"compound mass"}, q{"capture Q"}, mn{"neutron mass"};
    Cell minexc{"min excitation"};
    for (const auto& r : rows) {
      if (r.size() < 8) { continue; }
      const int z = inum(r[1]), a = inum(r[2]);
      const std::string tag = r[0];
      cmp(target, deex::nuclear_mass(a, z), num(r[3]), tag + " target");
      cmp(compound, deex::nuclear_mass(a + 1, z), num(r[4]), tag + " compound");
      cmp(mn, static_cast<double>(units::neutron_mass_c2<double>()), num(r[5]), tag + " n");
      cmp(q, deex::nuclear_mass(a, z) + units::neutron_mass_c2<double>()
                 - deex::nuclear_mass(a + 1, z),
          num(r[6]), tag + " Q");
      cmp(minexc, capture::capture_min_excitation(), num(r[7]), tag + " minExcitation");
    }
    std::printf("\n1. mass balance, exact\n");
    // The Q is a difference of two 2e5 MeV masses and a 1e-16 relative error in either is a
    // 1e-11 relative error in a 4 MeV Q on lead. The tolerance is on the Q itself and it is
    // met at zero, because P3's AME2012 table is bit-exact and the arithmetic here is the same
    // subtraction Geant4 does - so this is not a loose gate, it is a tight one that passes.
    g_fails += report(target, 0.0);
    g_fails += report(compound, 0.0);
    g_fails += report(mn, 0.0);
    g_fails += report(q, 1e-14);
    g_fails += report(minexc, 0.0);
  }

  // ------------------------------------------------------------------ 2. deterministic
  {
    bool ok = false;
    const auto rows = read_csv(dir + "/capture_det.csv", ok);
    if (!ok) { return 1; }

    Cell nsec{"secondary count"}, draws{"uniforms consumed"}, ekin{"secondary Ekin"};
    Cell dirc{"secondary direction"}, massc{"secondary PDG mass"};
    Cell dync{"secondary dynamical mass"}, timec{"secondary time"};
    Cell energy_change{"primary energy change"};
    int pdg_mismatch = 0, isomer_digit = 0, za_mismatch = 0, kind_mismatch = 0;
    int exc_agree = 0, exc_differ = 0;
    double worst_exc = 0;
    std::string worst_exc_where;
    int calls = 0;

    // The rows are grouped: every secondary of one call shares (name, z, a, ekin, phase). Walk
    // them in order, re-running the port once per group.
    std::size_t i = 0;
    while (i < rows.size()) {
      const auto& r0 = rows[i];
      if (r0.size() < 22) { ++i; continue; }
      const int z = inum(r0[1]), a = inum(r0[2]);
      const double e = num(r0[3]);
      const int phase = inum(r0[4]);
      const int g4_draws = inum(r0[5]);
      const int g4_nsec = inum(r0[6]);
      const double g4_energy_change = num(r0[8]);
      const std::string tag = r0[0] + " E=" + r0[3] + " ph=" + r0[4];

      HadFinalState<real_t, kCap> fs;
      CycleRng rng;
      rng.phase = phase;
      const auto info = capture::neutron_rad_capture_apply<real_t, kCap>(
          make_neutron(static_cast<real_t>(e)), HadNucleus{z, a, 0}, lt, rng, &fs);
      ++calls;
      if (info.refused == capture::CaptureRefusal::kSecondaryOverflow) {
        fail("%s: final state overflowed at capacity %d", tag.c_str(), kCap);
      }
      cmp(nsec, double(fs.n_secondaries), double(g4_nsec), tag + " nsec");
      cmp(draws, double(rng.draws), double(g4_draws), tag + " draws");
      cmp(energy_change, double(fs.energy_change), g4_energy_change, tag + " energy_change");

      // Walk this call's rows.
      std::size_t j = i;
      while (j < rows.size() && rows[j].size() >= 22 && rows[j][0] == r0[0]
             && rows[j][3] == r0[3] && rows[j][4] == r0[4]) {
        const auto& r = rows[j];
        const int idx = inum(r[10]);
        ++j;
        if (idx < 0) { continue; }   // the "emitted nothing" row
        if (idx >= fs.n_secondaries) { continue; }
        const HadSecondary<real_t>& s = fs.secondaries[idx];
        const int g4_pdg = inum(r[11]);
        const int g4_z = inum(r[12]), g4_a = inum(r[13]);
        char what[256];
        std::snprintf(what, sizeof what, "%s sec %d", tag.c_str(), idx);
        // A gamma or a conversion electron has (Z, A) = (0, 0) and a real PDG code; a nucleus
        // has (Z, A) and an encoded one whose isomer digit the port does not reproduce.
        const bool nucleus = (g4_a > 0);
        if (nucleus != (s.a > 0)) {
          ++kind_mismatch;
          fail("%s: Geant4 made %s, port made %s", what, nucleus ? "a nucleus" : "a particle",
               (s.a > 0) ? "a nucleus" : "a particle");
          continue;
        }
        if (!nucleus) {
          if (s.pdg != g4_pdg) { ++pdg_mismatch; }
        } else {
          if (s.z != g4_z || s.a != g4_a) { ++za_mismatch; }
          if (s.pdg != g4_pdg) {
            // Expected exactly when Geant4's code carries a non-zero isomer digit.
            if ((g4_pdg % 10) != 0 && (g4_pdg / 10) == (s.pdg / 10)) {
              ++isomer_digit;
            } else {
              ++pdg_mismatch;
            }
          }
          // The excitation the mass was built from: raw here, snapped there.
          const double g4_exc = num(r[21]);
          const double d = deviation(double(info.residual_excitation), g4_exc);
          if (d <= 1e-12) {
            ++exc_agree;
          } else {
            ++exc_differ;
            if (d > worst_exc) {
              worst_exc = d;
              char b[512];
              std::snprintf(b, sizeof b, "%s (port %.17g, G4 %.17g)", what,
                            double(info.residual_excitation), g4_exc);
              worst_exc_where = b;
            }
          }
        }
        // TWO masses, and comparing only one of them would hide the finding. `sec_mass_MeV` is
        // the DEFINITION's mass, which the port recomputes from (pdg, Z, A) because
        // HadSecondary has one field; `sec_dynmass_MeV` is the dynamical one the four-momentum
        // implies, which is what that field holds. On the A <= 4 branch they differ by an ulp
        // and neither is the other.
        cmp(massc,
            capture::capture_secondary_pdg_mass<real_t>(
                s, double(info.residual_excitation)),
            num(r[14]), std::string(what) + " PDG mass");
        cmp(dync, double(s.mass), num(r[15]), std::string(what) + " dynamical mass");
        cmp(ekin, double(s.kin_energy), num(r[16]), what);
        cmp(dirc, double(s.direction.x), num(r[17]), std::string(what) + " dx");
        cmp(dirc, double(s.direction.y), num(r[18]), std::string(what) + " dy");
        cmp(dirc, double(s.direction.z), num(r[19]), std::string(what) + " dz");
        cmp(timec, double(s.time), num(r[20]), std::string(what) + " t");
      }
      i = j;
    }

    std::printf("\n2. the whole final state under the prescribed cycle, %d calls\n", calls);
    g_fails += report(nsec, 0.0);
    g_fails += report(draws, 0.0);
    g_fails += report(energy_change, 0.0);
    g_fails += report(massc, 0.0);
    g_fails += report(dync, 0.0);
    g_fails += report(ekin, 0.0);
    g_fails += report(dirc, 1e-15);
    g_fails += report(timec, 0.0);
    std::printf("  residual excitation: %d agree with G4ENSDFSTATE's snapped value, %d differ",
                exc_agree, exc_differ);
    if (exc_differ > 0) {
      std::printf(" (worst %.3e)\n      %s\n", worst_exc, worst_exc_where.c_str());
    } else {
      std::printf("\n");
    }
    std::printf("  residual PDG code:   %d differ only in the isomer digit (expected), "
                "%d otherwise\n", isomer_digit, pdg_mismatch);
    if (pdg_mismatch != 0) { fail("%d secondaries have an unexplained PDG code", pdg_mismatch); }
    if (za_mismatch != 0) { fail("%d residuals have the wrong (Z, A)", za_mismatch); }
    if (kind_mismatch != 0) { fail("%d secondaries are the wrong kind", kind_mismatch); }
  }

  // ------------------------------------------------------------------ 3. statistical
  {
    bool ok = false;
    const auto rows = read_csv(dir + "/capture_stat.csv", ok);
    if (!ok) { return 1; }
    constexpr int kNMult = 16;
    constexpr int kNSpec = 20;
    constexpr double kSpecMax = 10.0;
    // A z-score, not a relative tolerance: what is compared is two finite samples of the same
    // distribution. 4 sigma on 13 targets x 5 energies x (2 moments + 36 bins) is about 2500
    // comparisons, so one 4-sigma outlier is expected roughly once in twenty runs of this file
    // and the gate is stated at 5.
    const double kMaxSigma = 5.0;
    double worst_sigma = 0;
    std::string worst_where;
    int npoints = 0;
    int over = 0;

    std::printf("\n3. distributions, 20,000 captures per point\n");
    for (const auto& r : rows) {
      if (r.size() < std::size_t(14 + kNMult + kNSpec)) { continue; }
      const int z = inum(r[1]), a = inum(r[2]);
      const double e = num(r[3]);
      const int n = inum(r[4]);
      const std::string tag = r[0] + " E=" + r[3];

      // The port's own sample, from Philox at a fixed key. A different stream from Geant4's by
      // construction - that is what makes this a statistical comparison and not an exact one.
      double sg = 0, sgg = 0, se = 0, seg = 0, segg = 0, setot = 0, sres = 0, stime = 0;
      long long ngam_total = 0;
      int isomer = 0;
      int mult[kNMult] = {0};
      int spec[kNSpec] = {0};
      int overflow = 0;
      for (int i = 0; i < n; ++i) {
        Philox<double> rng(static_cast<unsigned int>(z * 1000 + a),
                           static_cast<unsigned int>(i), 0xC0DEu);
        HadFinalState<real_t, kCap> fs;
        const auto info = capture::neutron_rad_capture_apply<real_t, kCap>(
            make_neutron(static_cast<real_t>(e)), HadNucleus{z, a, 0}, lt, rng, &fs);
        if (info.refused == capture::CaptureRefusal::kSecondaryOverflow) { ++overflow; }
        int ng = 0, ne = 0;
        double etot = 0;
        for (int j = 0; j < fs.n_secondaries; ++j) {
          const HadSecondary<real_t>& s = fs.secondaries[j];
          if (s.pdg == 22) {
            ++ng;
            etot += double(s.kin_energy);
            seg += double(s.kin_energy);
            segg += double(s.kin_energy) * double(s.kin_energy);
            int b = int(double(s.kin_energy) / kSpecMax * double(kNSpec));
            if (b < 0) { b = 0; }
            if (b >= kNSpec) { b = kNSpec - 1; }
            ++spec[b];
          } else if (s.pdg == 11) {
            ++ne;
          } else {
            sres += double(s.kin_energy);
            stime += double(s.time);
            if (double(info.residual_excitation) > 0.0) { ++isomer; }
          }
        }
        sg += ng;
        sgg += double(ng) * double(ng);
        se += ne;
        setot += etot;
        ngam_total += ng;
        int mb = ng;
        if (mb >= kNMult) { mb = kNMult - 1; }
        ++mult[mb];
      }
      if (overflow > 0) {
        fail("%s: %d of %d captures overflowed the final state", tag.c_str(), overflow, n);
      }
      const double mg = sg / n;
      const double vg = sgg / n - mg * mg;
      const double g4_mg = num(r[5]), g4_vg = num(r[6]);

      auto check_mean = [&](double m1, double v1, double m2, double v2, long long nn,
                            const char* what) {
        if (nn < 2) { return; }
        const double sd = std::sqrt((std::fabs(v1) + std::fabs(v2)) / double(nn));
        if (!(sd > 0.0)) { return; }
        const double zscore = std::fabs(m1 - m2) / sd;
        ++npoints;
        if (zscore > worst_sigma) {
          worst_sigma = zscore;
          char b[512];
          std::snprintf(b, sizeof b, "%s %s (port %.6g, G4 %.6g, %.2f sigma)", tag.c_str(),
                        what, m1, m2, zscore);
          worst_where = b;
        }
        if (zscore > kMaxSigma) { ++over; }
      };
      check_mean(mg, vg, g4_mg, g4_vg, n, "mean gamma multiplicity");
      const double meg = (ngam_total > 0) ? seg / double(ngam_total) : 0.0;
      const double veg = (ngam_total > 0) ? segg / double(ngam_total) - meg * meg : 0.0;
      check_mean(meg, veg, num(r[8]), num(r[9]), ngam_total, "mean gamma energy");

      // The two histograms, bin by bin, as two multinomial samples. sqrt(n1 + n2) is the right
      // sigma here and not sqrt(n1): both sides are finite samples.
      for (int b = 0; b < kNMult; ++b) {
        const double n1 = double(mult[b]), n2 = num(r[14 + b]);
        if (n1 + n2 < 20.0) { continue; }   // a bin with a handful of counts is not a test
        const double zscore = std::fabs(n1 - n2) / std::sqrt(n1 + n2);
        ++npoints;
        if (zscore > worst_sigma) {
          worst_sigma = zscore;
          char bb[512];
          std::snprintf(bb, sizeof bb, "%s mult bin %d (port %.0f, G4 %.0f, %.2f sigma)",
                        tag.c_str(), b, n1, n2, zscore);
          worst_where = bb;
        }
        if (zscore > kMaxSigma) { ++over; }
      }
      for (int b = 0; b < kNSpec; ++b) {
        const double n1 = double(spec[b]), n2 = num(r[14 + kNMult + b]);
        if (n1 + n2 < 20.0) { continue; }
        const double zscore = std::fabs(n1 - n2) / std::sqrt(n1 + n2);
        ++npoints;
        if (zscore > worst_sigma) {
          worst_sigma = zscore;
          char bb[512];
          std::snprintf(bb, sizeof bb, "%s spectrum bin %d (port %.0f, G4 %.0f, %.2f sigma)",
                        tag.c_str(), b, n1, n2, zscore);
          worst_where = bb;
        }
        if (zscore > kMaxSigma) { ++over; }
      }
    }
    std::printf("  %d comparisons, worst %.2f sigma (limit %.1f), %d over\n", npoints,
                worst_sigma, kMaxSigma, over);
    if (worst_sigma > 0.0) { std::printf("      %s\n", worst_where.c_str()); }
    if (over > 0) { g_fails += over; }
  }

  std::printf("\n%s (%d failures)\n", (g_fails == 0) ? "PASSED" : "FAILED", g_fails);
  return (g_fails == 0) ? 0 : 1;
}
