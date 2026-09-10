// G4ExcitationHandler::BreakItUp, statistically, against 20,000 Geant4 events per campaign.
//
// This is the only test in the package that cannot be exact, and the reason is not the physics:
// this port's engine is Philox and Geant4's is HepJamesRandom, so no sampler in the module can
// be compared draw for draw even in principle. What CAN be compared is the distribution, and
// with 20,000 events on each side the comparison is sharp - a channel that fires 10% too often
// is 30 standard deviations away.
//
// Three oracle files over the same 17 campaigns, plus two that isolate one step of the cascade:
//
//   deex_breakup.csv           per species, the number produced in 20,000 events and the first
//                              two moments of its kinetic energy
//   deex_breakup_residual.csv  per event, the (Z, A) of the heaviest product - the residual
//                              distribution, which is the shape of the whole cascade in one
//                              histogram
//   deex_spectra.csv           the kinetic-energy SPECTRUM in 0.25 MeV bins, for the eight
//                              species whose rest mass is a constant. A mean and a variance are
//                              two numbers; a gamma cascade is a set of discrete lines and an
//                              evaporation spectrum has an edge at the Coulomb barrier, and two
//                              distributions can match both moments while disagreeing about
//                              every line. This file is where a line in the wrong place shows.
//
// The campaigns are C12, Al27, Ca40, Fe56, Pb208 and U238 at excitations from 1 to 200 MeV,
// plus one Fe56 with 500 MeV/c of momentum so that the boost is exercised and not only the
// rest frame. C12 at any of these excitations is inside the Fermi break-up window; Fe56
// crosses from photon evaporation through nucleon evaporation into the GEM channels; Pb208 and
// U238 add fission.
//
// **What is compared by (Z, A) and not by PDG code, and why.** The oracle keys its species by
// `G4ReactionProduct::GetDefinition()->GetPDGEncoding()`, and for a heavy ion that encoding's
// last digit is an ISOMER LEVEL that G4IonTable assigns when it first creates the ion - taken
// from the G4ENSDFSTATE table if the excitation matches an entry there, and numbered in
// creation order otherwise. It is a property of the run's ion table, not of the fragment. So
// this test folds the oracle's counts onto (Z, A), which is what the port produces, and 2.3%
// of the products in the file are the ones that folding merges.
//
// The same boundary limits the ENERGY comparison. `GetKineticEnergy()` is
// `totalEnergy - mass` with the mass of the particle definition, and for an isomer that mass
// is the excitation SNAPPED to G4ENSDFSTATE. The port does not snap - the snapping is refused
// by name in excitation_handler.cuh and belongs to whoever owns core/particle.cuh - so the
// kinetic energy is compared only for the eight species whose mass is a fixed constant:
// gamma, e-, n, p, d, t, He3 and alpha. Those carry the great majority of the emitted energy.
// For everything heavier the COUNT is compared and the energy is not, and the count is the
// number that says whether the cascade took the same route.
//
// Anti-vacuity: sixteen perturbations of the port, one at a time, each rebuilt and rerun over
// the whole grid; the commit message lists every one with what it moved. Seven fail. Nine
// change the output by less than five sigma and are labelled there rather than counted as
// checks, because a check that cannot fail is not a check - and two of those nine are worth
// knowing about while reading this file:
//
//   * SortSecondaryFragment's natural-isotope release is REDUNDANT. Removing it leaves the
//     output byte-for-byte identical, because the fragment then goes onto the evaporation list
//     and the identical test at the top of G4Evaporation::BreakFragment releases it on the next
//     pass. The two tests differ only in `<` versus `<=` at fMinExcitation.
//   * The forced-at-threshold branch of G4UnstableFragmentBreakUp is never reached here, so
//     CLHEP's zero-vector `unit()` cannot be tested by this grid: making clhep_unit() return
//     (0, 0, 1) for EVERY vector, not only the zero one, also changes nothing.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/deexcitation/excitation_handler.cuh"

using namespace g4gpu;

/// The deliverable is a DEVICE-callable entry point, and a `__host__ __device__` function that
/// is only ever called from the host is never compiled for the device at all - nvcc instantiates
/// a template on demand, so every device-side restriction in it stays invisible. This kernel is
/// never launched (this package is validated on the host; the machine's GPU belongs to the
/// integration builds) but it forces the instantiation, which is what proves `deexcite` is
/// device code: no host-only header, no `std::` call without a device overload, no local static
/// that nvcc rejects. Deleting it would leave the whole module compiling for the host only.
__global__ void deex_device_instantiation_probe(const deex::Fragment* in, data::LevelTable lt,
                                                deex::FermiPool pool, deex::DeexWorkspace ws,
                                                deex::DeexStatus* out) {
  Philox<double> rng(1u, 2u, 3u);
  out[0] = deex::deexcite(in[0], lt, pool, ws, rng);
}

namespace {

struct Campaign {
  int Z, A;
  double eexc, pz;
  int N;
};

/// One species' tally, in whichever key the caller chose. `sum_k2` is filled only on the port's
/// side, where the per-event multiplicity is visible - see multiplicity_z().
struct Tally {
  long long count = 0;
  double sum_e = 0.0;
  double sum_e2 = 0.0;
  double sum_k2 = 0.0;
};

int fails = 0;

/// The two-sample z-score for two total counts over N events each.
///
/// The naive form - |n1 - n2| / sqrt(n1 + n2) - assumes the per-event multiplicity is Poisson,
/// and for a de-excitation cascade it is not: Pb208 at 200 MeV emits 14 neutrons per event on
/// average, and energy conservation makes that distribution NARROWER than Poisson, so the
/// naive sigma is too LARGE and the naive z too small. Getting this backwards would hide a
/// real disagreement, so the variance is measured rather than assumed: `var` is the port's own
/// per-event multiplicity variance, and it is used for both samples.
///
/// Using the port's variance for Geant4's sample is the standard move and it is sound here for
/// a reason worth stating: the variance is a second-order property of the same distribution
/// whose mean is under test, so if it disagreed enough to matter the mean would already have
/// failed by a wide margin.
double multiplicity_z(long long n1, long long n2, long long N, double var) {
  if (N < 2) { return 0.0; }
  const double s2 = 2.0 * static_cast<double>(N) * var;
  if (!(s2 > 0.0)) {
    // A species with zero variance is produced exactly the same number of times in every
    // event, so any difference at all is infinitely significant - reported as a large finite z
    // rather than as a NaN.
    return (n1 == n2) ? 0.0 : 1.e9;
  }
  return std::fabs(static_cast<double>(n1 - n2)) / std::sqrt(s2);
}

/// The two-sample z-score for a per-event CATEGORICAL count, which is what the residual
/// distribution is: every event contributes exactly one residual (Z, A), so the count in a bin
/// is Binomial(N, p) and its variance is N p (1 - p), not N p. The difference is the whole
/// comparison for a dominant bin - Pb208 at 1 MeV leaves Pb208 in every single event, where a
/// Poisson sigma of sqrt(20000) = 141 would accept a 600-event disagreement.
double binomial_z(long long n1, long long n2, long long N) {
  if (N < 2) { return 0.0; }
  const double p1 = static_cast<double>(n1) / static_cast<double>(N);
  const double p2 = static_cast<double>(n2) / static_cast<double>(N);
  const double s2 = static_cast<double>(N) * (p1 * (1.0 - p1) + p2 * (1.0 - p2));
  if (!(s2 > 0.0)) { return (n1 == n2) ? 0.0 : 1.e9; }
  return std::fabs(static_cast<double>(n1 - n2)) / std::sqrt(s2);
}

/// The relative difference of two counts over the same number of trials. Used where a channel
/// is asked for a fragment N times and should deliver one every time - the emitted count is
/// then not a random variable at all, only the photon channel's is, since GenerateGamma can
/// decline to emit.
double count_relative(long long a, long long b) {
  const double m = static_cast<double>((a > b) ? a : b);
  return (m > 0.0) ? std::fabs(static_cast<double>(a - b)) / m : 0.0;
}

/// The z-score for two means with known second moments.
double mean_z(double m1, double v1, long long n1, double m2, double v2, long long n2) {
  if (n1 < 2 || n2 < 2) { return 0.0; }
  const double s2 = v1 / static_cast<double>(n1) + v2 / static_cast<double>(n2);
  if (!(s2 > 0.0)) { return 0.0; }
  return std::fabs(m1 - m2) / std::sqrt(s2);
}

/// mean_z where the quantity CAN be deterministic, which a per-event mean of an integer often
/// is: G4Evaporation::BreakFragment emits exactly zero products for every Fermi-applicable
/// fragment, so the mean product count of the C12 campaigns is 0 with variance 0 in both
/// samples. mean_z returns 0 for a zero variance - it has to, because a discrete gamma line
/// gives two samples the same energy in every event and a last-bit difference must not be
/// infinitely significant - and here that 0 would accept ANY disagreement. So the degenerate
/// case is decided by the means themselves.
double moment_z(double m1, double v1, long long n1, double m2, double v2, long long n2) {
  if (!(v1 > 0.0) && !(v2 > 0.0)) {
    return (std::fabs(m1 - m2) <= 1.e-12) ? 0.0 : 1.e9;
  }
  return mean_z(m1, v1, n1, m2, v2, n2);
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

/// (Z, A) from an oracle PDG code, folding the isomer digit away. The light species keep their
/// own codes; a nuclear code is 10LZZZAAAI.
bool pdg_to_za(int pdg, int& Z, int& A) {
  if (pdg == 22) { Z = 0; A = 0; return true; }        // gamma
  if (pdg == 11) { Z = -1; A = 0; return true; }       // e-, distinguished from the gamma by Z
  if (pdg == 2112) { Z = 0; A = 1; return true; }
  if (pdg == 2212) { Z = 1; A = 1; return true; }
  if (pdg > 1000000000) {
    A = (pdg / 10) % 1000;
    Z = (pdg / 10000) % 1000;
    return true;
  }
  return false;
}

/// The port's own key, in the same encoding.
int za_key(int Z, int A) { return Z * 1000 + A + 500000; }

// The kinetic-energy histogram's grid, which must be the dump's: 0.25 MeV per bin from zero,
// bin 200 holding everything at or above 50 MeV. See dump_breakup() in dump_deexcitation.cc.
const double kBinWidth = 0.25;
const int kOverflowBin = 200;

/// The eight species whose rest mass is a constant, and therefore the eight whose kinetic
/// energy this port can reproduce at all - the same list the dump histograms.
bool is_spectrum_pdg(int pdg) {
  switch (pdg) {
    case 22: case 11: case 2112: case 2212:
    case 1000010020: case 1000010030: case 1000020030: case 1000020040:
      return true;
    default:
      return false;
  }
}

long long spectrum_key(int pdg, double ekin) {
  int bin = static_cast<int>(ekin / kBinWidth);
  if (bin < 0) { bin = 0; }
  if (bin > kOverflowBin) { bin = kOverflowBin; }
  return static_cast<long long>(pdg) * 1000 + bin;
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
  data::LevelTableStorage lts;
  data::read_all_level_data(
      lts, pe, data::kLevelZMax,
      [](int Z, int A) { return deex::shell_correction(A, Z); },
      [](int Z, int A) { return deex::level_manager_level_density(Z, A); });
  const data::LevelTable lt = lts.view();
  deex::FermiPoolStorage ps;
  deex::build_fermi_pool(ps, lt);
  const deex::FermiPool pool = ps.view();
  std::printf("level table: %d managers; Fermi pool: %d fragments, %d channels\n",
              lt.n_managers, pool.n_frag, pool.n_channels);

  // The oracle's campaigns, read out of the file rather than retyped, so the two cannot drift.
  const auto lines = read_lines(dir + "/deex_breakup.csv");
  const auto rlines = read_lines(dir + "/deex_breakup_residual.csv");
  if (lines.size() < 2 || rlines.size() < 2) {
    std::printf("cannot read %s/deex_breakup*.csv - run ref/oracle/run.bat tables\n",
                dir.c_str());
    return 1;
  }

  // Campaign key -> (species key -> tally), for Geant4.
  std::map<std::string, std::map<int, Tally>> g4;
  std::map<std::string, std::map<int, long long>> g4res;
  std::vector<Campaign> campaigns;
  std::map<std::string, int> seen;
  char key[128];

  for (std::size_t i = 1; i < lines.size(); ++i) {
    int Z = 0, A = 0, N = 0, pdg = 0;
    long long cnt = 0;
    double eexc = 0, pz = 0, m1 = 0, m2 = 0;
    if (std::sscanf(lines[i].c_str(), "%d,%d,%lf,%lf,%d,%d,%lld,%lf,%lf", &Z, &A, &eexc, &pz,
                    &N, &pdg, &cnt, &m1, &m2) != 9) {
      continue;
    }
    std::snprintf(key, sizeof key, "%d_%d_%g_%g", Z, A, eexc, pz);
    if (seen.find(key) == seen.end()) {
      seen[key] = 1;
      campaigns.push_back(Campaign{Z, A, eexc, pz, N});
    }
    int pz2 = 0, pa = 0;
    if (!pdg_to_za(pdg, pz2, pa)) {
      std::printf("  unrecognised PDG code %d in the oracle\n", pdg);
      ++fails;
      continue;
    }
    Tally& t = g4[key][za_key(pz2, pa)];
    t.count += cnt;
    t.sum_e += m1 * static_cast<double>(cnt);
    t.sum_e2 += m2 * static_cast<double>(cnt);
  }
  for (std::size_t i = 1; i < rlines.size(); ++i) {
    int Z = 0, A = 0, N = 0, rz = 0, ra = 0;
    long long cnt = 0;
    double eexc = 0, pz = 0;
    if (std::sscanf(rlines[i].c_str(), "%d,%d,%lf,%lf,%d,%d,%d,%lld", &Z, &A, &eexc, &pz, &N,
                    &rz, &ra, &cnt) != 8) {
      continue;
    }
    std::snprintf(key, sizeof key, "%d_%d_%g_%g", Z, A, eexc, pz);
    g4res[key][za_key(rz, ra)] += cnt;
  }

  // The SPECTRA. A mean and a variance are two numbers per species and a de-excitation
  // spectrum is not a two-parameter family: the gamma output is a set of discrete lines whose
  // positions come from the level data, and an evaporation spectrum is a Maxwellian with an
  // edge at the Coulomb barrier. Two distributions can agree on both moments and disagree about
  // where every line sits, so deex_spectra.csv holds the histogram itself - 0.25 MeV bins - for
  // the eight species whose rest mass is a constant.
  const auto splines = read_lines(dir + "/deex_spectra.csv");
  if (splines.size() < 2) {
    std::printf("cannot read %s/deex_spectra.csv - run ref/oracle/run.bat tables\n",
                dir.c_str());
    return 1;
  }
  std::map<std::string, std::map<long long, long long>> g4spec;
  for (std::size_t i = 1; i < splines.size(); ++i) {
    int Z = 0, A = 0, N = 0, bin = 0;
    long long pdg = 0, cnt = 0;
    double eexc = 0, pz = 0, lo = 0;
    if (std::sscanf(splines[i].c_str(), "%d,%d,%lf,%lf,%d,%lld,%d,%lf,%lld", &Z, &A, &eexc, &pz,
                    &N, &pdg, &bin, &lo, &cnt) != 9) {
      continue;
    }
    std::snprintf(key, sizeof key, "%d_%d_%g_%g", Z, A, eexc, pz);
    g4spec[key][pdg * 1000 + bin] += cnt;
  }
  std::printf("oracle: %d campaigns, %d spectrum bins\n", static_cast<int>(campaigns.size()),
              static_cast<int>(splines.size()) - 1);

  // The species whose mass is a fixed constant, so their kinetic energy can be compared. Keyed
  // the same way as everything else.
  std::map<int, const char*> named;
  named[za_key(0, 0)] = "gamma";
  named[za_key(-1, 0)] = "e-";
  named[za_key(0, 1)] = "neutron";
  named[za_key(1, 1)] = "proton";
  named[za_key(1, 2)] = "deuteron";
  named[za_key(1, 3)] = "triton";
  named[za_key(2, 3)] = "He3";
  named[za_key(2, 4)] = "alpha";

  // Buffers. The evaporation list is the one that grows; 1024 is above Geant4's own 1000-step
  // guard on the loop over it, so the guard fires before the buffer does.
  std::vector<deex::Fragment> evap(1024), results(1024), step(512);
  std::vector<deex::DeexProduct> products(1024);
  deex::DeexWorkspace ws;
  ws.evap_list = evap.data();
  ws.evap_capacity = static_cast<int>(evap.size());
  ws.results = results.data();
  ws.results_capacity = static_cast<int>(results.size());
  ws.step = step.data();
  ws.step_capacity = static_cast<int>(step.size());
  ws.products = products.data();
  ws.products_capacity = static_cast<int>(products.size());

  double worst_count_z = 0.0, worst_mean_z = 0.0, worst_res_z = 0.0, worst_hist_z = 0.0;
  std::string worst_count_at, worst_mean_at, worst_res_at, worst_hist_at;
  int count_bad = 0, mean_bad = 0, res_bad = 0, refusals = 0, hist_bins = 0, hist_bad = 0;
  long long total_products = 0;

  // 5 sigma on every comparison. With 20,000 events a channel branching ratio is known to
  // about 1% absolute, so 5 sigma is roughly a 5% relative disagreement on a common species
  // and much tighter on an abundant one - and the z is reported so that a systematic shift is
  // visible as its size rather than only as a pass or a fail.
  const double kZLimit = 5.0;
  // (declared before the per-channel section, which uses it too)
  // Counts below this are dominated by their own noise and their z is meaningless; they are
  // still summed into the total so a species the port never makes at all is not hidden.
  const long long kMinCount = 40;

  std::printf("\n%-22s %-10s %8s %8s %7s %8s\n", "campaign", "species", "geant4", "port",
              "z(N)", "z(<E>)");
  for (const Campaign& c : campaigns) {
    std::snprintf(key, sizeof key, "%d_%d_%g_%g", c.Z, c.A, c.eexc, c.pz);
    const std::string k = key;
    std::map<int, Tally> mine;
    std::map<int, long long> myres;
    std::map<int, int> per_event;
    std::map<long long, Tally> myspec;
    std::map<long long, int> per_event_bin;
    deex::DeexStatus worst_status;

    for (int n = 0; n < c.N; ++n) {
      per_event.clear();
      per_event_bin.clear();
      // One stream per (campaign, event). Philox is counter-based, so a distinct (event,
      // track) pair is an independent stream without seeding cost - which is why the port can
      // run 20,000 events reproducibly without a global generator.
      Philox<double> rng(static_cast<uint32_t>(1000 * c.Z + c.A),
                         static_cast<uint32_t>(n), static_cast<uint32_t>(c.eexc + 1));
      deex::Fragment frag = deex::make_excited_fragment(c.Z, c.A, c.eexc, c.pz);
      const deex::DeexStatus st = deex::deexcite(frag, lt, pool, ws, rng);
      if (st.any_refusal()) {
        ++refusals;
        worst_status = st;
      }
      int heaviest_a = -1, hz = 0, ha = 0;
      for (int i = 0; i < st.n_products; ++i) {
        const deex::DeexProduct& p = products[static_cast<std::size_t>(i)];
        // The gamma and the electron are both (Z=0, A=0) fragments; the oracle separates them
        // by PDG code and so does this key, through Z = -1 for the electron.
        const int pz2 = (p.a == 0) ? ((p.pdg == deex::kPdgElectron) ? -1 : 0) : p.z;
        Tally& t = mine[za_key(pz2, p.a)];
        const double ek = deex::deex_kinetic_energy(p);
        ++t.count;
        t.sum_e += ek;
        t.sum_e2 += ek * ek;
        ++per_event[za_key(pz2, p.a)];
        ++total_products;
        if (is_spectrum_pdg(p.pdg)) {
          const long long bk = spectrum_key(p.pdg, ek);
          ++myspec[bk].count;
          ++per_event_bin[bk];
        }
        if (p.a > heaviest_a) {
          heaviest_a = p.a;
          hz = p.z;
          ha = p.a;
        }
      }
      // The per-event multiplicity, squared and summed, so the count comparison can use a
      // measured variance instead of a Poisson assumption. Done per BIN as well as per species,
      // because a bin holding the whole of a discrete line has exactly zero variance - Pb208 at
      // 1 MeV puts one gamma in one bin in every single event - and a Poisson sigma of sqrt(n)
      // there would accept a 400-count disagreement.
      for (const auto& kv : per_event) {
        mine[kv.first].sum_k2 += static_cast<double>(kv.second) * kv.second;
      }
      for (const auto& kv : per_event_bin) {
        myspec[kv.first].sum_k2 += static_cast<double>(kv.second) * kv.second;
      }
      ++myres[za_key(hz, ha)];
    }

    // Species counts, and the mean energy where the mass is unambiguous.
    std::map<int, Tally>& og = g4[k];
    for (const auto& kv : og) {
      const long long gn = kv.second.count;
      const long long mn = mine.count(kv.first) ? mine[kv.first].count : 0;
      const double mu = static_cast<double>(mn) / static_cast<double>(c.N);
      const double k2 = mine.count(kv.first) ? mine[kv.first].sum_k2 : 0.0;
      double var = k2 / static_cast<double>(c.N) - mu * mu;
      if (var < 0.0) { var = 0.0; }
      const double z = multiplicity_z(gn, mn, c.N, var);
      const bool big = (gn + mn) >= kMinCount;
      if (big && z > worst_count_z) {
        worst_count_z = z;
        worst_count_at = k + " sp" + std::to_string(kv.first - 500000);
      }
      if (big && z > kZLimit) { ++count_bad; }

      double mz = 0.0;
      const auto it = named.find(kv.first);
      if (it != named.end() && mn >= 2 && gn >= 2) {
        const double gm = kv.second.sum_e / static_cast<double>(gn);
        const double gv = kv.second.sum_e2 / static_cast<double>(gn) - gm * gm;
        const Tally& mt = mine[kv.first];
        const double mm = mt.sum_e / static_cast<double>(mn);
        const double mv = mt.sum_e2 / static_cast<double>(mn) - mm * mm;
        mz = mean_z(gm, gv, gn, mm, mv, mn);
        if (mz > worst_mean_z) {
          worst_mean_z = mz;
          worst_mean_at = k + " " + it->second;
        }
        if (mz > kZLimit) { ++mean_bad; }
      }
      if (it != named.end() && big) {
        std::printf("%-22s %-10s %8lld %8lld %7.2f %8.2f\n", k.c_str(), it->second, gn, mn, z,
                    mz);
      }
    }
    // A species the port makes and Geant4 never did is as much a failure as the reverse.
    for (const auto& kv : mine) {
      if (og.find(kv.first) != og.end()) { continue; }
      if (kv.second.count < kMinCount) { continue; }
      std::printf("%-22s sp%-8d %8d %8lld  PORT-ONLY\n", k.c_str(), kv.first - 500000, 0,
                  kv.second.count);
      ++count_bad;
    }

    // The residual (Z, A) distribution.
    std::map<int, long long>& orr = g4res[k];
    for (const auto& kv : orr) {
      const long long gn = kv.second;
      const long long mn = myres.count(kv.first) ? myres[kv.first] : 0;
      if (gn + mn < kMinCount) { continue; }
      const double z = binomial_z(gn, mn, c.N);
      if (z > worst_res_z) {
        worst_res_z = z;
        worst_res_at = k + " res" + std::to_string(kv.first - 500000);
      }
      if (z > kZLimit) { ++res_bad; }
    }

    // The kinetic-energy histograms, bin by bin, in both directions: a bin Geant4 fills and
    // the port does not is a missing line, and a bin the port fills and Geant4 does not is an
    // invented one.
    std::map<long long, long long>& osp = g4spec[k];
    for (const auto& kv : osp) {
      const long long gn = kv.second;
      const long long mn = myspec.count(kv.first) ? myspec[kv.first].count : 0;
      if (gn + mn < kMinCount) { continue; }
      const double mu = static_cast<double>(mn) / static_cast<double>(c.N);
      double var = (myspec.count(kv.first) ? myspec[kv.first].sum_k2 : 0.0) /
                       static_cast<double>(c.N) - mu * mu;
      if (var < 0.0) { var = 0.0; }
      const double z = multiplicity_z(gn, mn, c.N, var);
      ++hist_bins;
      if (z > worst_hist_z) {
        worst_hist_z = z;
        char b[96];
        std::snprintf(b, sizeof b, "%s pdg%lld bin%d (%.2f MeV)", k.c_str(), kv.first / 1000,
                      static_cast<int>(kv.first % 1000),
                      static_cast<double>(kv.first % 1000) * kBinWidth);
        worst_hist_at = b;
      }
      if (z > kZLimit) { ++hist_bad; }
    }
    for (const auto& kv : myspec) {
      if (osp.find(kv.first) != osp.end()) { continue; }
      if (kv.second.count < kMinCount) { continue; }
      std::printf("%-22s pdg%lld bin%d (%.2f MeV) %8d %8lld  PORT-ONLY BIN\n", k.c_str(),
                  kv.first / 1000, static_cast<int>(kv.first % 1000),
                  static_cast<double>(kv.first % 1000) * kBinWidth, 0, kv.second.count);
      ++hist_bad;
      ++hist_bins;
    }
    if (worst_status.any_refusal()) {
      std::printf("%-22s refusal: mf=%d hyper=%d unstable=%d fission=%d cap=%d loop=%d "
                  "at Z=%d A=%d\n", k.c_str(), worst_status.refused_multifragmentation,
                  worst_status.refused_hyper_fragment,
                  worst_status.refused_unstable_no_channel,
                  worst_status.refused_fission_split, worst_status.refused_capacity,
                  worst_status.refused_loop_limit, worst_status.refused_z,
                  worst_status.refused_a);
    }
  }

  // --------------------------------------------------------------- one channel's sampler, alone
  //
  // deex_channel_spectrum.csv fixes the channel and samples only its EmittedFragment, so each
  // of the three kinetic-energy samplers in the module is compared on its own: the rejection
  // against a two-region majorant for the six light ejectiles, G4GEMChannel's rejection
  // against the total width for the sixty GEM nuclei, and G4PhotonEvaporation's cumulative
  // array for the gammas.
  //
  // This is the file that found the GEM barrier defect. The 60 GEM channels' emission
  // probabilities are exact (tests/test_deex_models.cu, worst 0) and their mean kinetic
  // energies were 9% to 36% high, which is only possible if a quantity that cancels in the
  // width does not cancel in the sampler - and `Beta` is exactly that quantity. See
  // gem_probability_coulomb_barrier() in gem.cuh.
  {
    const auto slines = read_lines(dir + "/deex_channel_spectrum.csv");
    if (slines.size() < 2) {
      std::printf("cannot read %s/deex_channel_spectrum.csv - run ref/oracle/run.bat tables\n",
                  dir.c_str());
      return 1;
    }
    double worst_cnt = 0.0, worst_mean = 0.0;
    std::string worst_cnt_at, worst_mean_at;
    int rows = 0, chan_bad = 0, spec_bad = 0;
    double worst_by_kind[3] = {0.0, 0.0, 0.0};
    static const char* kind_name[3] = {"light ejectile", "GEM nucleus", "photon"};
    for (std::size_t i = 1; i < slines.size(); ++i) {
      int Z = 0, A = 0, E = 0, chan = 0, pz = 0, pa = 0, N = 0;
      long long gn = 0;
      double gm = 0, gm2 = 0;
      if (std::sscanf(slines[i].c_str(), "%d,%d,%d,%d,%d,%d,%d,%lld,%lf,%lf", &Z, &A, &E,
                      &chan, &pz, &pa, &N, &gn, &gm, &gm2) != 10) {
        continue;
      }
      ++rows;
      long long mn = 0;
      double sum = 0.0, sum2 = 0.0;
      int seen_z = -1000, seen_a = -1000;
      for (int n = 0; n < N; ++n) {
        Philox<double> rng(static_cast<uint32_t>(1000 * Z + A), static_cast<uint32_t>(n),
                           static_cast<uint32_t>(100 * chan + E));
        deex::Fragment frag = deex::make_excited_fragment(Z, A, static_cast<double>(E), 0.0);
        deex::Fragment out;
        bool have = false;
        if (chan == deex::kChannelPhoton) {
          deex::PhotonEvaporationState pst;
          deex::photon_emission_probability(pst, frag, lt);
          const deex::GammaEmission ge = deex::generate_gamma(pst, frag, lt, rng);
          have = ge.emitted;
          out = ge.product;
        } else if (chan < deex::kChannelFirstGem) {
          deex::EvaporationState s;
          s.ej = chan - deex::kChannelFirstEvaporation;
          deex::channel_emission_probability(s, frag, lt);
          out = deex::channel_emitted_fragment(s, frag, lt, rng);
          have = true;
        } else {
          deex::GemState s;
          s.ch = chan - deex::kChannelFirstGem;
          deex::gem_channel_emission_probability(s, frag, lt);
          out = deex::gem_emitted_fragment(s, frag, lt, rng);
          have = true;
        }
        if (!have) { continue; }
        if (out.z != pz || out.a != pa) {
          seen_z = out.z;
          seen_a = out.a;
          continue;
        }
        const double ek = out.momentum.e - out.ground_state_mass;
        ++mn;
        sum += ek;
        sum2 += ek * ek;
      }
      if (mn == 0) {
        std::printf("  channel %d of Z=%d A=%d E*=%d produced no (Z=%d,A=%d) - saw "
                    "(Z=%d,A=%d)\n", chan, Z, A, E, pz, pa, seen_z, seen_a);
        ++chan_bad;
        continue;
      }
      const int kind = (chan == 0) ? 2 : (chan < deex::kChannelFirstGem ? 0 : 1);
      const double mm = sum / static_cast<double>(mn);
      const double mv = sum2 / static_cast<double>(mn) - mm * mm;
      const double gv = gm2 - gm * gm;
      std::snprintf(key, sizeof key, "%d_%d_%d ch%d", Z, A, E, chan);
      const double rel = count_relative(gn, mn);
      if (rel > worst_cnt) {
        worst_cnt = rel;
        worst_cnt_at = key;
      }
      const double z = mean_z(gm, gv, gn, mm, mv, mn);
      if (z > worst_mean) {
        worst_mean = z;
        worst_mean_at = key;
      }
      if (rel > 0.02 || z > kZLimit) { ++spec_bad; }
      if (z > worst_by_kind[kind]) { worst_by_kind[kind] = z; }
    }
    std::printf("\nper-channel samplers (deex_channel_spectrum.csv): %d channel-fragment "
                "combinations\n", rows);
    for (int k = 0; k < 3; ++k) {
      std::printf("  worst z, mean kinetic energy, %-16s %6.2f\n", kind_name[k],
                  worst_by_kind[k]);
    }
    std::printf("  worst relative count difference   %8.4f  at %s\n", worst_cnt,
                worst_cnt_at.c_str());
    std::printf("  worst z, mean kinetic energy      %8.2f  at %s\n", worst_mean,
                worst_mean_at.c_str());
    if (spec_bad > 0) {
      std::printf("  %d of %d channel-fragment combinations above tolerance\n", spec_bad,
                  rows);
      ++fails;
    }
    if (chan_bad > 0) {
      std::printf("  %d channels emitted the wrong species\n", chan_bad);
      ++fails;
    }
  }

  // ---------------------------------------------------------------- one evaporation step, alone
  //
  // A whole cascade cannot say WHERE it disagrees: twenty-two emissions each 0.03% off look
  // exactly like one emission 0.7% off. deex_firststep.csv is one call of
  // G4Evaporation::BreakFragment with only its FIRST product recorded - the channel that fired
  // and the kinetic energy it took - and deex_firststep_chain.csv is that same call's product
  // count and the residual it left. Between them they separate the step from the chain and the
  // chain from G4ExcitationHandler's loop.
  //
  // The kinetic energy here is `momentum.e() - GetGroundStateMass()`, deliberately NOT
  // `G4ReactionProduct::GetKineticEnergy()`: the latter subtracts the particle definition's
  // mass and would bring G4IonTable's excitation snapping back into a comparison this file
  // exists to keep clean of it.
  {
    const auto flines = read_lines(dir + "/deex_firststep.csv");
    const auto clines = read_lines(dir + "/deex_firststep_chain.csv");
    if (flines.size() < 2 || clines.size() < 2) {
      std::printf("cannot read %s/deex_firststep*.csv - run ref/oracle/run.bat tables\n",
                  dir.c_str());
      return 1;
    }
    struct Step {
      int Z, A, N;
      double eexc;
    };
    std::vector<Step> steps;
    std::map<std::string, std::map<int, Tally>> gfirst;
    std::map<std::string, int> sseen;
    // The campaign list comes from the CHAIN file, not from the first-product file, and that is
    // the difference between checking the Fermi gate inside G4Evaporation::BreakFragment and
    // not checking it. C12 at 20 MeV is Fermi-applicable, so BreakFragment returns having
    // emitted nothing and contributes no row at all to deex_firststep.csv - it appears only in
    // deex_firststep_chain.csv, with mean_nprod = 0. Taking the campaigns from the first-product
    // file would silently drop every campaign whose answer is "no products", which is exactly
    // the answer that gate exists to give.
    for (std::size_t i = 1; i < clines.size(); ++i) {
      int Z = 0, A = 0, N = 0;
      double e = 0, np = 0, rz = 0, ra = 0, rx = 0;
      if (std::sscanf(clines[i].c_str(), "%d,%d,%lf,%d,%lf,%lf,%lf,%lf", &Z, &A, &e, &N, &np,
                      &rz, &ra, &rx) != 8) {
        continue;
      }
      std::snprintf(key, sizeof key, "%d_%d_%g", Z, A, e);
      if (sseen.find(key) == sseen.end()) {
        sseen[key] = 1;
        steps.push_back(Step{Z, A, N, e});
      }
    }
    for (std::size_t i = 1; i < flines.size(); ++i) {
      int Z = 0, A = 0, N = 0, fz = 0, fa = 0;
      long long cnt = 0;
      double e = 0, m1 = 0, m2 = 0;
      if (std::sscanf(flines[i].c_str(), "%d,%d,%lf,%d,%d,%d,%lld,%lf,%lf", &Z, &A, &e, &N,
                      &fz, &fa, &cnt, &m1, &m2) != 9) {
        continue;
      }
      std::snprintf(key, sizeof key, "%d_%d_%g", Z, A, e);
      Tally& t = gfirst[key][za_key(fz, fa)];
      t.count = cnt;
      t.sum_e = m1 * static_cast<double>(cnt);
      t.sum_e2 = m2 * static_cast<double>(cnt);
    }

    std::printf("\n%-16s %10s %10s %10s %10s %8s %8s %8s %8s\n", "one step", "nprod g4",
                "nprod port", "resA g4", "resA port", "z(nprod)", "z(resA)", "z(resZ)",
                "z(Eres)");
    double worst_step_z = 0.0, worst_first_z = 0.0, worst_first_e_z = 0.0;
    std::string worst_step_at, worst_first_at, worst_first_e_at;
    int step_bad = 0;
    for (const Step& s : steps) {
      std::snprintf(key, sizeof key, "%d_%d_%g", s.Z, s.A, s.eexc);
      const std::string k = key;
      double g_nprod = 0, g_rz = 0, g_ra = 0, g_rexc = 0;
      bool found = false;
      for (std::size_t i = 1; i < clines.size(); ++i) {
        int Z = 0, A = 0, N = 0;
        double e = 0, np = 0, rz = 0, ra = 0, rx = 0;
        if (std::sscanf(clines[i].c_str(), "%d,%d,%lf,%d,%lf,%lf,%lf,%lf", &Z, &A, &e, &N, &np,
                        &rz, &ra, &rx) != 8) {
          continue;
        }
        if (Z == s.Z && A == s.A && e == s.eexc) {
          g_nprod = np;
          g_rz = rz;
          g_ra = ra;
          g_rexc = rx;
          found = true;
          break;
        }
      }
      if (!found) { continue; }

      std::map<int, Tally> mfirst;
      double nprod = 0, sn2 = 0, rz = 0, rz2 = 0, ra = 0, ra2 = 0, rexc = 0, rexc2 = 0;
      for (int n = 0; n < s.N; ++n) {
        Philox<double> rng(static_cast<uint32_t>(1000 * s.Z + s.A), static_cast<uint32_t>(n),
                           static_cast<uint32_t>(s.eexc + 1));
        deex::Fragment frag = deex::make_excited_fragment(s.Z, s.A, s.eexc, 0.0);
        deex::FragmentList out{ws.step, ws.step_capacity, 0};
        deex::PhotonEvaporationState pst;
        deex::DeexStatus st;
        deex::evaporation_break_fragment(frag, out, lt, pool, pst, st, rng);
        nprod += out.n;
        sn2 += static_cast<double>(out.n) * out.n;
        rz += frag.z;
        rz2 += static_cast<double>(frag.z) * frag.z;
        ra += frag.a;
        ra2 += static_cast<double>(frag.a) * frag.a;
        rexc += frag.excitation;
        rexc2 += frag.excitation * frag.excitation;
        if (out.n > 0) {
          const deex::Fragment& p = out.v[0];
          Tally& t = mfirst[za_key(p.z, p.a)];
          const double ek = p.momentum.e - p.ground_state_mass;
          ++t.count;
          t.sum_e += ek;
          t.sum_e2 += ek * ek;
        }
      }
      const double dN = static_cast<double>(s.N);
      const double mnp = nprod / dN;
      const double vnp = sn2 / dN - mnp * mnp;
      const double mra = ra / dN;
      const double vra = ra2 / dN - mra * mra;
      const double mrz = rz / dN;
      const double vrz = rz2 / dN - mrz * mrz;
      const double mrx = rexc / dN;
      const double znp = moment_z(g_nprod, vnp, s.N, mnp, vnp, s.N);
      const double zra = moment_z(g_ra, vra, s.N, mra, vra, s.N);
      const double zrz = moment_z(g_rz, vrz, s.N, mrz, vrz, s.N);
      if (znp > worst_step_z) {
        worst_step_z = znp;
        worst_step_at = k + " nprod";
      }
      if (zra > worst_step_z) {
        worst_step_z = zra;
        worst_step_at = k + " resA";
      }
      if (zrz > worst_step_z) {
        worst_step_z = zrz;
        worst_step_at = k + " resZ";
      }
      // The residual's mean EXCITATION - the number that says the chain stopped in the same
      // place - is the one quantity here whose relative difference is NOT a criterion: it is
      // the mean of a heavily skewed distribution that is zero in most events and a fraction of
      // an MeV in the rest, so a 4% difference in it is one and a half sigma, not a
      // disagreement. Pb208 at 200 MeV leaves 0.524 MeV on average and U238 at 30 MeV leaves
      // 0.182 MeV; a flat 2% tolerance failed both and meant nothing. Compared as a z, with the
      // port's own measured variance standing in for both samples exactly as the product count
      // and the residual A are.
      const double vrx = rexc2 / dN - mrx * mrx;
      const double zrx = moment_z(g_rexc, vrx, s.N, mrx, vrx, s.N);
      if (zrx > worst_step_z) {
        worst_step_z = zrx;
        worst_step_at = k + " resExc";
      }
      if (znp > kZLimit || zra > kZLimit || zrz > kZLimit || zrx > kZLimit) { ++step_bad; }
      std::printf("%-16s %10.5f %10.5f %10.4f %10.4f %8.2f %8.2f %8.2f %8.2f\n", k.c_str(),
                  g_nprod, mnp, g_ra, mra, znp, zra, zrz, zrx);

      for (const auto& kv : gfirst[k]) {
        const long long gn = kv.second.count;
        const long long mn = mfirst.count(kv.first) ? mfirst[kv.first].count : 0;
        if (gn + mn < kMinCount) { continue; }
        // The first product is one categorical draw per event, so its count is binomial.
        const double z = binomial_z(gn, mn, s.N);
        if (z > worst_first_z) {
          worst_first_z = z;
          worst_first_at = k + " sp" + std::to_string(kv.first - 500000);
        }
        if (z > kZLimit) { ++step_bad; }
        if (mn >= 2 && gn >= 2) {
          const double gm = kv.second.sum_e / static_cast<double>(gn);
          const double gv = kv.second.sum_e2 / static_cast<double>(gn) - gm * gm;
          const Tally& mt = mfirst[kv.first];
          const double mm = mt.sum_e / static_cast<double>(mn);
          const double mv = mt.sum_e2 / static_cast<double>(mn) - mm * mm;
          const double ez = mean_z(gm, gv, gn, mm, mv, mn);
          if (ez > worst_first_e_z) {
            worst_first_e_z = ez;
            worst_first_e_at = k + " sp" + std::to_string(kv.first - 500000);
          }
          if (ez > kZLimit) { ++step_bad; }
        }
      }
    }
    std::printf("  worst z, one-step nprod/resA %6.2f  at %s\n", worst_step_z,
                worst_step_at.c_str());
    std::printf("  worst z, first product count %6.2f  at %s\n", worst_first_z,
                worst_first_at.c_str());
    std::printf("  worst z, first product <E>   %6.2f  at %s\n", worst_first_e_z,
                worst_first_e_at.c_str());
    if (step_bad > 0) {
      std::printf("  %d one-step comparisons above %.0f sigma\n", step_bad, kZLimit);
      ++fails;
    }
  }

  std::printf("\n%lld products over %d campaigns, %d events with a refusal\n", total_products,
              static_cast<int>(campaigns.size()), refusals);
  std::printf("  worst z, species count   %6.2f  at %s\n", worst_count_z,
              worst_count_at.c_str());
  std::printf("  worst z, mean energy     %6.2f  at %s\n", worst_mean_z, worst_mean_at.c_str());
  std::printf("  worst z, residual (Z, A) %6.2f  at %s\n", worst_res_z, worst_res_at.c_str());
  std::printf("  worst z, spectrum bin    %6.2f  at %s\n", worst_hist_z, worst_hist_at.c_str());
  std::printf("  above %.0f sigma: %d counts, %d means, %d residuals, %d of %d spectrum bins\n",
              kZLimit, count_bad, mean_bad, res_bad, hist_bad, hist_bins);

  if (count_bad > 0 || mean_bad > 0 || res_bad > 0 || hist_bad > 0) { ++fails; }
  // A histogram comparison over an empty set of bins would pass everything above.
  if (hist_bins < 500) {
    std::printf("  only %d spectrum bins compared - the histograms are not being read\n",
                hist_bins);
    ++fails;
  }
  if (refusals > 0) {
    std::printf("  %d events hit a refusal - none is expected on this grid\n", refusals);
    ++fails;
  }
  // A comparison over an empty tally would pass everything above.
  if (total_products < 1000000) {
    std::printf("  only %lld products - the grid is not being run\n", total_products);
    ++fails;
  }

  std::printf("%s\n", (fails == 0) ? "PASS" : "FAIL");
  return (fails == 0) ? 0 : 1;
}
