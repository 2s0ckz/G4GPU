// The three model layers between the probability layer and the handler: the 60 GEM channels,
// competitive fission, and the Fermi break-up fragment pool.
//
// Four oracle files, all deterministic:
//
//   deex_gem.csv               G4GEMChannel::GetEmissionProbability for all 60 GEM channels
//                              and G4CompetitiveFission::GetEmissionProbability, over
//                              12 nuclides x 6 excitations, taken through
//                              G4EvaporationDefaultGEMFactory so that the CHANNEL INDEX is
//                              compared too. The index is observable - G4Evaporation stops its
//                              channel loop early on it - so a port that agreed on every
//                              probability but ordered the channels differently would still be
//                              wrong, and this file catches that rather than the physics.
//   deex_fission_barrier.csv   G4FissionBarrier::FissionBarrier and
//                              G4FissionLevelDensityParameter over 4,963 (Z, A) at two U.
//                              This is the ONLY oracle-visible consumer of
//                              G4CameronShellPlusPairingCorrections: that table's accessor is
//                              inline in a header whose symbol this Windows Geant4 does not
//                              export, so dump_corrections() cannot read it directly and
//                              tests/test_deex_nuclear.cu had to mark it not-comparable. The
//                              barrier reads it, so this file checks it.
//   deex_fission_prob.csv      the fission channel probability and the five
//                              G4FissionParameters the mass distribution is built from, over
//                              15 nuclides x 7 excitations - which reaches every branch of
//                              DefineParameters (Z >= 90, Z == 89, 82 <= Z < 89, Z < 82) and
//                              the A < 227 boost.
//   deex_fermi_pool.csv        the whole G4FermiFragmentsPoolVI: for every (Z, A) in the
//                              maxZ = 9 / maxA = 17 window and 12 excitations, what
//                              ClosestChannels selects, and every pair in that channel set
//                              with its cumulative probability. 2,802 rows.
//
// The pool file is the one that matters most, because Fermi break-up is not a small
// correction: it REPLACES evaporation for every fragment with Z < 9 and A < 17, which is most
// of what a de-excitation cascade ends up as. If the pool has one fragment too many the
// cascade takes a different route and no amount of agreement on evaporation widths shows it.
//
// Anti-vacuity: the run recorded in the commit message perturbed, one at a time, the two
// (A, Z) pairs of the Be12 and O17 GEM channels (unified to the channel's), the GEM excited
// states' lifetime gate, Z23 replaced by A23 in the GEM beta parameter, the Cameron
// shell-plus-pairing subtraction in the fission barrier, the fission level density's Z
// scaling, the pool's float truncation of the level energies, and the pool's
// `MaxLevelEnergy == 0 && LifeTime == 0` skip.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/deexcitation/fermi_breakup.cuh"
#include "physics/hadronic/deexcitation/fission.cuh"
#include "physics/hadronic/deexcitation/gem.cuh"

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
  std::printf("  %-42s %7d pts  worst %.3g  %s\n", label, c.n, c.worst, ok ? "OK" : "FAIL");
  if (!ok) {
    ++fails;
    std::printf("      worst at %s\n", c.where.c_str());
  }
}

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
  std::snprintf(buf, sizeof buf, "(Z=%d,A=%d,#%d) x=%.4g", a, b, c, d);
  return buf;
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
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

  // ------------------------------------------------------------------------ GEM and fission
  {
    const auto lines = read_lines(dir + "/deex_gem.csv");
    if (lines.size() < 2) {
      std::printf("cannot read %s/deex_gem.csv - run ref/oracle/run.bat tables\n", dir.c_str());
      return 1;
    }
    Cell gem, fiss;
    int open_gem = 0, open_fiss = 0, bad_index = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      int Z = 0, A = 0, chan = 0;
      double eexc = 0, prob = 0;
      if (std::sscanf(lines[i].c_str(), "%d,%d,%lf,%d,%lf", &Z, &A, &eexc, &chan, &prob) != 5) {
        continue;
      }
      deex::Fragment frag = deex::make_excited_fragment(Z, A, eexc, 0.0);
      if (chan == 1) {
        deex::FissionState fs;
        const double p = deex::fission_emission_probability(fs, frag, lt);
        note(fiss, dev_of(p, prob, 1e-60), at(Z, A, chan, eexc));
        if (prob > 0.0) { ++open_fiss; }
        continue;
      }
      // The factory's layout: 0 photon, 1 fission, 2..7 the light ejectiles, 8..67 the GEM
      // nuclei. Asserted rather than assumed.
      const int g = chan - 8;
      if (g < 0 || g >= deex::kNumGemChannels) {
        ++bad_index;
        continue;
      }
      deex::GemState gs;
      gs.ch = g;
      const double p = deex::gem_channel_emission_probability(gs, frag, lt);
      note(gem, dev_of(p, prob, 1e-60), at(Z, A, chan, eexc));
      if (prob > 0.0) { ++open_gem; }
    }
    std::printf("GEM and fission channels (deex_gem.csv): %d GEM open, %d fission open\n",
                open_gem, open_fiss);
    if (bad_index > 0) {
      std::printf("  %d rows had a channel index outside 1 and 8..67\n", bad_index);
      ++fails;
    }
    report(gem, 1e-11, "G4GEMChannel emission probability");
    report(fiss, 1e-12, "G4CompetitiveFission probability");
    // A grid on which every channel were closed would pass every comparison above and prove
    // nothing, so the count is asserted. The fission threshold is low here on purpose: this
    // grid has only three nuclides heavy enough to fission and deex_fission_prob.csv is where
    // fission is actually covered.
    if (open_gem < 500 || open_fiss < 10) {
      std::printf("  too few open channels (%d GEM, %d fission) to exercise the widths\n",
                  open_gem, open_fiss);
      ++fails;
    }
  }

  // ------------------------------------------------------------- the fission barrier, widely
  {
    const auto lines = read_lines(dir + "/deex_fission_barrier.csv");
    if (lines.size() < 2) {
      std::printf("cannot read %s/deex_fission_barrier.csv\n", dir.c_str());
      return 1;
    }
    Cell bar, ldp;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      int Z = 0, A = 0;
      double U = 0, b = 0, l = 0;
      if (std::sscanf(lines[i].c_str(), "%d,%d,%lf,%lf,%lf", &Z, &A, &U, &b, &l) != 5) {
        continue;
      }
      note(bar, dev_of(deex::fission_barrier(A, Z, U), b, 1e-30), at(Z, A, 0, U));
      const bool has = (data::find_manager(lt, Z, A) >= 0);
      note(ldp, dev_of(deex::fission_level_density(A, Z, U, has), l, 1e-30), at(Z, A, 1, U));
    }
    std::printf("fission barrier (deex_fission_barrier.csv)\n");
    report(bar, 1e-13, "G4FissionBarrier (Cameron shell+pairing)");
    report(ldp, 1e-14, "G4FissionLevelDensityParameter");
  }

  // --------------------------------------------------- the fission probability and parameters
  {
    const auto lines = read_lines(dir + "/deex_fission_prob.csv");
    if (lines.size() < 2) {
      std::printf("cannot read %s/deex_fission_prob.csv\n", dir.c_str());
      return 1;
    }
    Cell cp, fp, bf, mk, pAs, pS1, pS2, pSS, pW;
    int open = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      int Z = 0, A = 0;
      double e = 0, o_cp = 0, o_fp = 0, o_bf = 0, o_mk = 0;
      double o_as = 0, o_s1 = 0, o_s2 = 0, o_ss = 0, o_w = 0;
      if (std::sscanf(lines[i].c_str(),
                      "%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &Z, &A, &e, &o_cp,
                      &o_fp, &o_bf, &o_mk, &o_as, &o_s1, &o_s2, &o_ss, &o_w) != 12) {
        continue;
      }
      deex::Fragment frag = deex::make_excited_fragment(Z, A, e, 0.0);
      deex::FissionState fs;
      note(cp, dev_of(deex::fission_emission_probability(fs, frag, lt), o_cp, 1e-60),
           at(Z, A, 0, e));

      const double ex = e - deex::fission_pairing_correction(A, Z);
      double b = 0.0, m = 0.0, p = 0.0;
      deex::FissionParameters par;
      if (ex > 0.0 && A >= 65 && Z > 16) {
        b = deex::fission_barrier(A, Z, ex);
        m = ex - b;
        p = deex::fission_probability(frag, m, lt);
        par.define(A, Z, ex, b);
      }
      note(bf, dev_of(b, o_bf, 1e-30), at(Z, A, 1, e));
      note(mk, dev_of(m, o_mk, 1e-30), at(Z, A, 2, e));
      note(fp, dev_of(p, o_fp, 1e-60), at(Z, A, 3, e));
      note(pAs, dev_of(par.As, o_as, 1e-30), at(Z, A, 4, e));
      note(pS1, dev_of(par.Sigma1, o_s1, 1e-30), at(Z, A, 5, e));
      note(pS2, dev_of(par.Sigma2, o_s2, 1e-30), at(Z, A, 6, e));
      note(pSS, dev_of(par.SigmaS, o_ss, 1e-30), at(Z, A, 7, e));
      note(pW, dev_of(par.w, o_w, 1e-30), at(Z, A, 8, e));
      if (o_cp > 0.0) { ++open; }
    }
    std::printf("fission probability and parameters (deex_fission_prob.csv): %d open of %d\n",
                open, static_cast<int>(lines.size()) - 1);
    if (open < 40) {
      std::printf("  only %d rows have a non-zero fission probability\n", open);
      ++fails;
    }
    report(cp, 1e-12, "G4CompetitiveFission probability");
    report(bf, 1e-13, "barrier at the reduced excitation");
    report(mk, 1e-12, "maximal kinetic energy");
    report(fp, 1e-12, "G4FissionProbability");
    report(pAs, 1e-15, "G4FissionParameters As");
    report(pS1, 1e-15, "G4FissionParameters Sigma1");
    report(pS2, 1e-15, "G4FissionParameters Sigma2");
    report(pSS, 1e-14, "G4FissionParameters SigmaS");
    report(pW, 1e-12, "G4FissionParameters w");
  }

  // ------------------------------------------------------------------- the Fermi break-up pool
  {
    deex::FermiPoolStorage ps;
    deex::build_fermi_pool(ps, lt);
    const deex::FermiPool pool = ps.view();
    std::printf("Fermi pool: %d fragments, %d pairs, %d channels\n", pool.n_frag, pool.n_pairs,
                pool.n_channels);

    const auto lines = read_lines(dir + "/deex_fermi_pool.csv");
    if (lines.size() < 2) {
      std::printf("cannot read %s/deex_fermi_pool.csv\n", dir.c_str());
      return 1;
    }
    int flag_bad = 0, nch_bad = 0, pair_bad = 0, rows = 0, pairs_seen = 0;
    Cell exc, mass, cum;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      int Z = 0, A = 0, app = 0, hc = 0, ip = 0, nch = 0, k = 0;
      int Z1 = 0, A1 = 0, Z2 = 0, A2 = 0;
      double e = 0, ch_exc = 0, ch_mass = 0, e1 = 0, e2 = 0, prob = 0;
      if (std::sscanf(lines[i].c_str(),
                      "%d,%d,%lf,%d,%d,%d,%d,%lf,%lf,%d,%d,%d,%lf,%d,%d,%lf,%lf", &Z, &A, &e,
                      &app, &hc, &ip, &nch, &ch_exc, &ch_mass, &k, &Z1, &A1, &e1, &Z2, &A2,
                      &e2, &prob) != 17) {
        continue;
      }
      ++rows;
      const double exc_in = e * units::MeV<double>();
      const int our_app = deex::fermi_is_applicable(pool, Z, A, exc_in) ? 1 : 0;
      const int our_hc = deex::fermi_has_channels(pool, Z, A, exc_in) ? 1 : 0;
      const int our_ip = deex::fermi_is_physical(pool, Z, A) ? 1 : 0;
      if (our_app != app || our_hc != hc || our_ip != ip) {
        if (flag_bad < 5) {
          std::printf("  flags differ at Z=%d A=%d E*=%g: applicable %d/%d has %d/%d "
                      "physical %d/%d\n", Z, A, e, our_app, app, our_hc, hc, our_ip, ip);
        }
        ++flag_bad;
      }

      // ClosestChannels is called on the fragment's TOTAL energy, which is what
      // G4FermiBreakUpVI::SampleDecay passes: ground-state mass plus excitation.
      const double gmass = deex::nuclear_mass(A, Z);
      const int slot = deex::fermi_closest_channels(pool, Z, A, gmass + exc_in);
      if (nch < 0) {
        if (slot >= 0) { ++nch_bad; }
        continue;
      }
      if (slot < 0) {
        ++nch_bad;
        continue;
      }
      if (pool.ch_count[slot] != nch) {
        if (nch_bad < 5) {
          std::printf("  channel count differs at Z=%d A=%d E*=%g: %d vs %d\n", Z, A, e,
                      pool.ch_count[slot], nch);
        }
        ++nch_bad;
        continue;
      }
      const deex::FermiFragment& f = pool.frag[pool.by_a[slot]];
      note(exc, dev_of(f.excitation, ch_exc, 1e-30), at(Z, A, -1, e));
      // G4FermiChannels::GetMass() is `excitation + ground_mass` where ground_mass was set to
      // the fragment's TOTAL energy - so it is mass + 2*excitation, not mass + excitation.
      // Reproduced because it is dumped and because it is the number the class returns.
      note(mass, dev_of(f.excitation + f.total_energy(), ch_mass, 1e-30), at(Z, A, -2, e));
      if (nch == 0 || k < 0) { continue; }
      ++pairs_seen;
      const deex::FermiPair& p = pool.pairs[pool.pair_of_channel[pool.ch_offset[slot] + k]];
      const deex::FermiFragment& g1 = pool.frag[p.f1];
      const deex::FermiFragment& g2 = pool.frag[p.f2];
      if (g1.z != Z1 || g1.a != A1 || g2.z != Z2 || g2.a != A2) {
        if (pair_bad < 5) {
          std::printf("  pair %d of Z=%d A=%d E*=%g: (%d,%d)+(%d,%d) vs (%d,%d)+(%d,%d)\n", k,
                      Z, A, e, g1.z, g1.a, g2.z, g2.a, Z1, A1, Z2, A2);
        }
        ++pair_bad;
        continue;
      }
      note(exc, dev_of(g1.excitation, e1, 1e-30), at(Z1, A1, k, e));
      note(exc, dev_of(g2.excitation, e2, 1e-30), at(Z2, A2, k, e));
      note(cum, dev_of(pool.cum_prob[pool.ch_offset[slot] + k], prob, 1e-30), at(Z, A, k, e));
    }
    std::printf("Fermi pool (deex_fermi_pool.csv): %d rows, %d with pairs\n", rows,
                pairs_seen);
    if (flag_bad > 0) {
      std::printf("  %-42s %7d mismatches  FAIL\n", "IsApplicable/HasChannels/IsPhysical",
                  flag_bad);
      ++fails;
    } else {
      std::printf("  %-42s %7d pts  0 mismatches  OK\n",
                  "IsApplicable/HasChannels/IsPhysical", rows);
    }
    if (nch_bad > 0) {
      std::printf("  %-42s %7d mismatches  FAIL\n", "ClosestChannels / channel count", nch_bad);
      ++fails;
    } else {
      std::printf("  %-42s %7d pts  0 mismatches  OK\n", "ClosestChannels / channel count",
                  rows);
    }
    if (pair_bad > 0) {
      std::printf("  %-42s %7d mismatches  FAIL\n", "pair (Z, A) and order", pair_bad);
      ++fails;
    } else {
      std::printf("  %-42s %7d pts  0 mismatches  OK\n", "pair (Z, A) and order", pairs_seen);
    }
    report(exc, 1e-15, "pool excitation energies");
    report(mass, 1e-15, "G4FermiChannels::GetMass");
    report(cum, 1e-13, "static cumulative probabilities");
  }

  std::printf("%s\n", (fails == 0) ? "PASS" : "FAIL");
  return (fails == 0) ? 0 : 1;
}
