// The evaporation and photon-evaporation probability layer against Geant4's own answers.
//
// Two oracles, and they are deliberately at two different levels of composition:
//
//   deex_invxs.csv   G4KalbachCrossSection::ComputeCrossSection and
//                    G4ChatterjeeCrossSection::ComputeCrossSection called directly, on a
//                    (ejectile, residual A, K) grid, together with the two arguments a channel
//                    would feed them - G4KalbachCrossSection::ComputePowerParameter and
//                    G4CoulombBarrier::GetCoulombBarrier. Straight functions of their
//                    arguments, so they are compared at machine precision.
//   deex_probs.csv   G4EvaporationChannel::GetEmissionProbability for all six default
//                    ejectiles and G4PhotonEvaporation::GetEmissionProbability, on a
//                    (Z, A, E*) grid of 14 nuclides x 8 excitations. This is the composite:
//                    a Coulomb barrier, the kinematic limits, the level density, the pairing
//                    correction and an ADAPTIVE trapezoid integral of the product. A
//                    disagreement in it is ambiguous, which is exactly why the previous file
//                    exists.
//
// The emission probability is the number that decides which channel fires, so it is the
// number the whole statistical layer rests on. It is also the hardest to get right, because
// G4VEmissionProbability::IntegrateProbability changes its own step size as it sweeps: a
// trapezoid carrying more than 80% of the running total shrinks the step by 0.7, one carrying
// less than 10% grows it by 1.5, and the loop stops when a trapezoid adds less than
// `accuracy` (2% for a neutron, 3% for a charged ejectile) of the total. So the integral is
// not the integral of the integrand - it is the integral this particular walk produces, and
// reproducing the walk step for step is the only way to reproduce the number. A port that
// integrated the same function more accurately would fail this test, and should.
//
// The two derived columns are compared only where the oracle wrote them. `inv_xs_mb_at_5MeV`
// and `prob_at_5MeV` come from G4VEvaporationChannel's ComputeInverseXSection and
// ComputeProbability, which read `resA13`, `a0`, `freeU` and `delta1` - members that
// TotalProbability sets. For a CLOSED channel TotalProbability never runs, those members hold
// the previous fragment's values, and `resA13 = 0` makes Kalbach's `lambda = p3/resA13 + p4`
// infinite. The dump writes NaN there rather than an infinity for the port to chase; this
// test skips those rows and counts them, so "not compared" is visible instead of silent.
//
// Anti-vacuity: the run recorded in the commit message perturbed, one at a time, the Kalbach
// signor renormalisations, the 0.6 barrier pivot, the adaptive step's 0.7/1.5 factors, the
// integrator's per-ejectile (elimit, accuracy) pair, the neutron's absolute value in `nu`,
// and the GR data's float storage. What each one reported is in the commit body.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/deexcitation/evaporation.cuh"
#include "physics/hadronic/deexcitation/photon_evaporation.cuh"

using namespace g4gpu;

namespace {

struct Cell {
  int n = 0;
  double worst = 0;
  std::string where;
};

void note(Cell& c, double dev, const char* what) {
  ++c.n;
  if (dev > c.worst) {
    c.worst = dev;
    c.where = what;
  }
}

int fails = 0;

void report(const Cell& c, double tol, const char* label) {
  const bool ok = (c.n > 0 && c.worst <= tol);
  std::printf("  %-40s %7d pts  worst %.3g  %s\n", label, c.n, c.worst, ok ? "OK" : "FAIL");
  if (!ok) {
    ++fails;
    std::printf("      worst at %s\n", c.where.c_str());
  }
}

/// Relative deviation, with an absolute floor below which a value counts as zero. The
/// probabilities here span 1e-30 to 1e+5 - they are widths in Geant4's internal units - so a
/// relative measure is the only one that means anything, and the floor has to be far below
/// the smallest value the oracle actually wrote rather than a round number.
double dev_of(double ours, double g4, double floor_abs) {
  if (std::fabs(g4) > floor_abs) { return std::fabs(ours - g4) / std::fabs(g4); }
  return (std::fabs(ours) > floor_abs) ? 1.0 : 0.0;
}

std::vector<std::string> read_lines(const std::string& path) {
  std::vector<std::string> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[4096];
  while (std::fgets(line, sizeof line, f) != nullptr) { out.push_back(line); }
  std::fclose(f);
  return out;
}

char buf[256];
const char* at(int a, int b, int c, double d) {
  std::snprintf(buf, sizeof buf, "(%d,%d,%d) x=%.4g", a, b, c, d);
  return buf;
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";

  // --------------------------------------------------------------------- inverse cross sections
  //
  // No level data needed: both parameterisations are closed forms of (K, cb, resA13, mu).
  {
    const auto lines = read_lines(dir + "/deex_invxs.csv");
    if (lines.size() < 2) {
      std::printf("cannot read %s/deex_invxs.csv - run ref/oracle/run.bat tables\n",
                  dir.c_str());
      return 1;
    }
    Cell mu_c, cb_c, kal, chat;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      int idx = 0, ejZ = 0, ejA = 0, resA = 0;
      double K = 0, cb = 0, mu = 0, kmb = 0, cmb = 0;
      if (std::sscanf(lines[i].c_str(), "%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf", &idx, &ejZ, &ejA,
                      &resA, &K, &cb, &mu, &kmb, &cmb) != 9) {
        continue;
      }
      const int resZ = resA / 2;
      // The dump's own two inputs, checked before what they feed: a wrong mu or a wrong
      // barrier would otherwise show up as a cross-section error of unknown origin.
      const double our_mu = (idx > 0) ? deex::kalbach_power_parameter(resA, idx) : 0.0;
      const double our_cb = deex::coulomb_barrier(ejA, ejZ, resA, resZ, 0.0);
      note(mu_c, dev_of(our_mu, mu, 1e-30), at(idx, resA, 0, K));
      note(cb_c, dev_of(our_cb, cb, 1e-30), at(idx, resA, resZ, K));

      const double resA13 = data::g4pow_z13<double>(resA);
      // Kalbach is pivoted at 0.6 of the barrier by G4EvaporationProbability::CrossSection;
      // Chatterjee takes the barrier itself. The dump passes exactly those two.
      const double our_k =
          deex::kalbach_cross_section(K, 0.6 * cb, resA13, our_mu, idx, ejZ, ejA, resA);
      const double our_c = deex::chatterjee_cross_section(K, cb, resA13, our_mu, idx, ejZ, resA);
      note(kal, dev_of(our_k, kmb, 1e-30), at(idx, resA, ejZ, K));
      note(chat, dev_of(our_c, cmb, 1e-30), at(idx, resA, ejZ, K));
    }
    std::printf("inverse cross sections (deex_invxs.csv)\n");
    report(mu_c, 1e-15, "Kalbach power parameter mu");
    report(cb_c, 1e-14, "G4CoulombBarrier at U = 0");
    report(kal, 1e-13, "G4KalbachCrossSection");
    report(chat, 1e-13, "G4ChatterjeeCrossSection");
  }

  // --------------------------------------------------------------------- emission probabilities
  //
  // These need the level table: G4EvaporationProbability::ComputeProbability asks
  // G4NuclearLevelData::GetLevelDensity for the residual (which with fLD = true does not read
  // it, but the manager lookup happens anyway) and G4PhotonEvaporation's threshold does the
  // same for the parent. Building the whole table rather than the sampled nuclides, because
  // the grid reaches 14 nuclides and their residuals, and the table is what a device kernel
  // would carry.
  const std::string pe = host::g4photon_evaporation_dir();
  if (pe.empty()) {
    std::printf("PhotonEvaporation dataset not found - g4data.cuh could not resolve "
                "G4LEVELGAMMADATA\n");
    return 1;
  }
  data::LevelTableStorage st;
  data::read_all_level_data(
      st, pe, data::kLevelZMax,
      [](int Z, int A) { return deex::shell_correction(A, Z); },
      [](int Z, int A) { return deex::level_manager_level_density(Z, A); });
  const data::LevelTable lt = st.view();
  std::printf("level table: %d managers, %d levels\n", lt.n_managers, lt.n_levels);

  {
    const auto lines = read_lines(dir + "/deex_probs.csv");
    if (lines.size() < 2) {
      std::printf("cannot read %s/deex_probs.csv - run ref/oracle/run.bat tables\n",
                  dir.c_str());
      return 1;
    }
    Cell evap[6], photon, invxs, pk;
    int open_channels = 0, closed_channels = 0, nan_rows = 0;
    static const char* ejname[6] = {"neutron", "proton", "deuteron", "triton", "He3", "alpha"};

    for (std::size_t i = 1; i < lines.size(); ++i) {
      int Z = 0, A = 0, chan = 0, ejZ = 0, ejA = 0;
      double eexc = 0, prob = 0, xs = 0, p5 = 0;
      if (std::sscanf(lines[i].c_str(), "%d,%d,%lf,%d,%d,%d,%lf,%lf,%lf", &Z, &A, &eexc, &chan,
                      &ejZ, &ejA, &prob, &xs, &p5) != 9) {
        continue;
      }
      deex::Fragment frag = deex::make_excited_fragment(Z, A, eexc, 0.0);

      if (chan == 100) {
        deex::PhotonEvaporationState ps;
        const double p = deex::photon_emission_probability(ps, frag, lt);
        note(photon, dev_of(p, prob, 1e-40), at(Z, A, 0, eexc));
        continue;
      }
      if (chan < 0 || chan > 5) { continue; }
      // The dump's channel order is G4EvaporationDefaultGEMFactory's: n, p, d, t, He3, alpha.
      // Asserted against the file's own ejectile columns rather than assumed, because a
      // reordering there would otherwise compare the deuteron against the triton and only
      // show up as a tolerance failure.
      const deex::Ejectile& e = deex::evaporation_ejectiles()[chan];
      if (e.z != ejZ || e.a != ejA) {
        std::printf("  channel %d is (Z=%d,A=%d) in the oracle but (Z=%d,A=%d) here\n", chan,
                    ejZ, ejA, e.z, e.a);
        ++fails;
        break;
      }

      deex::EvaporationState s;
      s.ej = chan;
      const double p = deex::channel_emission_probability(s, frag, lt);
      note(evap[chan], dev_of(p, prob, 1e-40), at(Z, A, chan, eexc));
      if (prob > 0.0) { ++open_channels; } else { ++closed_channels; }

      // ComputeInverseXSection and ComputeProbability read what TotalProbability left behind,
      // so they are only meaningful on an open channel - which is what the NaN in the oracle
      // records. Counted, not skipped quietly.
      if (std::isnan(xs) || std::isnan(p5)) {
        ++nan_rows;
        continue;
      }
      const double K = 5.0 * units::MeV<double>();
      const double pcoeff = deex::evaporation_pcoeff(chan);
      const double our_xs = deex::evaporation_cross_section(s, K, s.b_coulomb);
      const double our_p5 = deex::compute_probability(s, K, s.b_coulomb, pcoeff, lt);
      note(invxs, dev_of(our_xs, xs, 1e-30), at(Z, A, chan, eexc));
      note(pk, dev_of(our_p5, p5, 1e-40), at(Z, A, chan, eexc));
    }

    std::printf("emission probabilities (deex_probs.csv): %d open, %d closed, %d rows with "
                "no derived columns\n", open_channels, closed_channels, nan_rows);
    for (int c = 0; c < 6; ++c) {
      char lab[64];
      std::snprintf(lab, sizeof lab, "G4EvaporationChannel %s", ejname[c]);
      report(evap[c], 1e-12, lab);
    }
    report(photon, 1e-13, "G4PhotonEvaporation probability");
    report(invxs, 1e-13, "ComputeInverseXSection at 5 MeV");
    report(pk, 1e-12, "ComputeProbability at 5 MeV");

    // A test in which every channel is closed would pass every comparison above and prove
    // nothing. The grid was chosen so that it does not, and that has to be asserted rather
    // than hoped for.
    if (open_channels < 100) {
      std::printf("  only %d channels are open on this grid - the integrator is barely "
                  "exercised\n", open_channels);
      ++fails;
    }
  }

  std::printf("%s\n", (fails == 0) ? "PASS" : "FAIL");
  return (fails == 0) ? 0 : 1;
}
