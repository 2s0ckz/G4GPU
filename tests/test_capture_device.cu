// P3's nuclear level data on the device, and the capture cascade that walks it.
//
// `tests/test_capture.cu` checks `G4NeutronRadCapture` and `G4PhotonEvaporation::BreakUpChain`
// against the oracle to the last bit - on the HOST, out of `std::vector`s. What nothing checked
// is whether the same functions give the same answers when the level scheme they walk is 10 MB
// of device memory rather than host memory, which is the form the transport needs it in and the
// second of the two tables P8 named as blocking the neutron general process.
//
// So this test:
//
//   1. Reads PhotonEvaporation5.7 once (3108 managers) and uploads it through
//      `host/upload_level_data`, printing the device cost.
//   2. Runs a fixed grid of captures - `capture_final_state`, which is
//      `G4NeutronCaptureProcess::PostStepDoIt` around `G4NeutronRadCapture::ApplyYourself` -
//      on the HOST table and on the DEVICE table with the same prescribed uniforms, and
//      compares every gamma, the residual, the cascade time and the secondary count.
//   3. Asserts the level table's own accessors agree between the two, nuclide by nuclide, over
//      every manager: the level count, every level energy, every lifetime and every
//      transition's packed target and cumulative probability. That is the upload itself rather
//      than the physics on top of it, and it is where a wrong stride or a missing array would
//      show.
//
// WHY THE GRID STRADDLES A MASS NUMBER OF 3
//
// `G4NeutronRadCapture::ApplyYourself` increments A and then tests `A <= 4`, so the split
// between the two-body branch and photon evaporation is at a TARGET mass number of 3 - and only
// the second branch reads the level data at all. H1 and H2 take the first; everything from He4
// up takes the second. Both are here so that a broken upload fails on the rows that read it and
// passes on the rows that do not, which is the difference between "the table is wrong" and
// "the model is wrong". Iron's four stable isotopes are in the grid for the reason
// `ref/dump/dump_capture.cc` gives: the level scheme is per isotope, so a port that got the
// isotope right and the manager wrong has to fail on one of Fe54/56/57/58.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/rng.cuh"
#include "data/level_data.cuh"
#include "host/level_upload.cuh"
#include "physics/hadronic/capture/capture_process.cuh"

using real_t = double;
using namespace g4gpu;

namespace {

int g_fails = 0;
void fail(const char* what) {
  std::printf("  FAIL: %s\n", what);
  ++g_fails;
}

constexpr int kSecCap = 32;

/// THE UPLOAD IS COMPARED BIT FOR BIT. THE CASCADE ON TOP OF IT IS COMPARED IN ABSOLUTE TERMS,
/// AND THAT IS THE NATURAL SCALE RATHER THAN A CONCESSION.
///
/// The table read is exact: `digest_manager` sums the same doubles in the same order on both
/// sides and nothing in it is arithmetic on them, so one differing bit means the upload changed
/// a number. 174,411 levels and 268,190 transitions, worst 0.
///
/// The cascade cannot be exact - nvcc contracts multiply-adds by default and the host compiler
/// does not, the same reason `tests/test_step_hadron.cu` states at length - and a RELATIVE gate
/// on it is the wrong instrument. Every energy here is a difference of four-momenta at the
/// NUCLEAR MASS scale: `sample_gamma_transition_kinematics` boosts the fragment and takes the
/// gamma out of the boosted energy, and the residual's kinetic energy is `t - sqrt(t^2-|p|^2)`
/// rather than `t - M(Z,A)` (see capture/neutron_rad_capture.cuh). One ulp of an Fe-57 mass is
/// 7.3e-12 MeV and of a U-239 mass 3.3e-11 MeV, so:
///
///   * a 0.5 MeV gamma off iron differs by 7.3e-12 MeV, which a relative gate reads as 1.5e-11;
///   * a 8e-5 MeV recoil off uranium differs by 2.6e-11 MeV, which the same gate reads as
///     3.2e-07 - four orders apart for the same number of wrong bits.
///
/// So energies are gated at `|diff| <= 1e-8 MeV` - 0.01 eV, three hundred ulps of a uranium
/// mass and eleven orders below the smallest transition in the dataset - with a relative gate
/// kept as a backstop for anything large enough for it to be the tighter of the two. Direction
/// components are components of a UNIT vector, so their scale is absolute too: a nearly-forward
/// residual has a transverse component of 1e-3 whose last bit is 1e-19, and a relative gate
/// reads the same vector as 1e-16 in z and 1e-10 in x.
///
/// Measured with these: worst 5.82e-11 MeV absolute on an energy (3.20e-07 relative, on a
/// uranium recoil) and 2.23e-11 absolute on a direction component, over 1783 secondaries of
/// 448 captures.
constexpr double kEnergyAbsTol = 1e-8;   ///< MeV
constexpr double kTol = 1e-6;            ///< relative backstop
constexpr double kDirTol = 1e-9;

/// A prescribed uniform cycle, so a capture is a pure function of (target, energy, phase) and
/// the two sides consume identical numbers. The same eight values `ref/dump/dump_capture.cc`
/// uses, spread over (0,1) and away from both endpoints.
struct CycleRng {
  int i = 0;
  int phase = 0;
  int draws = 0;
  __host__ __device__ real_t uniform() {
    const real_t v[8] = {real_t(0.137), real_t(0.291), real_t(0.408), real_t(0.523),
                         real_t(0.661), real_t(0.744), real_t(0.859), real_t(0.932)};
    const real_t x = v[(i + phase) & 7];
    ++i;
    ++draws;
    return x;
  }
};

/// One capture, flattened so a host and a device run can be compared field by field.
struct CaptureOut {
  int n_sec = 0;
  int draws = 0;
  int status = -1;
  real_t edep = 0;
  int sec_pdg[kSecCap] = {};
  real_t sec_ekin[kSecCap] = {};
  real_t sec_dx[kSecCap] = {}, sec_dy[kSecCap] = {}, sec_dz[kSecCap] = {};
  real_t sec_time[kSecCap] = {};
};

struct Case {
  int z, a;
  real_t ekin;   ///< MeV
  int phase;
};

/// `G4HadProjectile` for a neutron, as `tests/test_capture.cu` builds one: the mass is the
/// DEFINITION's PDG mass, because that is what G4HadProjectile takes.
__host__ __device__ physics::hadronic::HadProjectile<real_t> make_neutron(real_t ekin) {
  physics::hadronic::HadProjectile<real_t> p;
  p.pdg = 2112;
  p.baryon_number = 1;
  p.charge = real_t(0);
  p.mass = units::neutron_mass_c2<real_t>();
  p.kin_energy = ekin;
  return p;
}

/// The whole of one capture, written once so the host and the device run the same source.
///
/// `neutron_rad_capture_apply` and not `capture_final_state`, because the second adds the
/// generic process's resample loop and the energy-momentum check around the first and it is
/// the FIRST that reads the level data. A test of the upload wants the function that walks the
/// levels, not the one that wraps it.
__host__ __device__ void one_capture(const data::LevelTable& lt, const Case& c,
                                     CaptureOut* out) {
  using namespace physics::hadronic;
  CycleRng rng{0, c.phase, 0};
  HadFinalState<real_t, kSecCap> fs;
  const auto info = capture::neutron_rad_capture_apply<real_t, kSecCap>(
      make_neutron(c.ekin), HadNucleus{c.z, c.a, 0}, lt, rng, &fs);
  out->n_sec = fs.n_secondaries;
  out->draws = rng.draws;
  out->status = static_cast<int>(info.refused);
  out->edep = fs.local_energy_deposit;
  for (int i = 0; i < fs.n_secondaries && i < kSecCap; ++i) {
    out->sec_pdg[i] = fs.secondaries[i].pdg;
    out->sec_ekin[i] = fs.secondaries[i].kin_energy;
    out->sec_dx[i] = fs.secondaries[i].direction.x;
    out->sec_dy[i] = fs.secondaries[i].direction.y;
    out->sec_dz[i] = fs.secondaries[i].direction.z;
    out->sec_time[i] = fs.secondaries[i].time;
  }
}

__global__ void RunCaptures(data::LevelTable lt, const Case* cases, int n, CaptureOut* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  out[i] = CaptureOut{};
  one_capture(lt, cases[i], &out[i]);
}

/// What the level table itself says about one manager, read through the accessors the cascade
/// uses. Compared host against device so the UPLOAD is checked and not only the physics.
struct ManagerDigest {
  int n_levels = 0;
  int n_trans_total = 0;
  double energy_sum = 0;
  double lifetime_sum = 0;
  double cumprob_sum = 0;
  long long trans_code_sum = 0;
  int spin_sum = 0;
};

__host__ __device__ void digest_manager(const data::LevelTable& t, int m, ManagerDigest* d) {
  const data::LevelManagerEntry& e = t.managers[m];
  d->n_levels = e.n_levels;
  for (int i = 0; i < e.n_levels; ++i) {
    d->energy_sum += data::level_energy(t, m, i);
    d->lifetime_sum += data::level_lifetime(t, m, i);
    d->spin_sum += data::level_spin_two(t, m, i);
    const int nt = data::level_ntrans(t, m, i);
    d->n_trans_total += nt;
    for (int j = 0; j < nt; ++j) {
      const data::LevelTransition& tr = data::level_transition(t, m, i, j);
      d->trans_code_sum += tr.trans;
      d->cumprob_sum += static_cast<double>(tr.cum_prob);
    }
  }
}

__global__ void RunDigests(data::LevelTable lt, int n, ManagerDigest* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  out[i] = ManagerDigest{};
  digest_manager(lt, i, &out[i]);
}

}  // namespace

int main() {
  const cudaError_t lim = cudaDeviceSetLimit(cudaLimitStackSize, 16384);
  if (lim != cudaSuccess) {
    std::printf("FATAL: cudaDeviceSetLimit: %s\n", cudaGetErrorString(lim));
    return 2;
  }
  std::printf("== PhotonEvaporation5.7 on the device, and the capture cascade on it ==\n\n");

  // ---- 1. read and upload.
  data::LevelTableStorage storage;
  auto owner = host::upload_level_data(storage, data::kLevelZMax, /*verbose=*/true);
  if (owner.view.managers == nullptr) {
    fail("the level table did not upload - G4LEVELGAMMADATA could not be resolved");
    std::printf("\nFAILED (%d failures)\n", g_fails);
    return 1;
  }
  const data::LevelTable host_lt = storage.view();
  std::printf("            %d managers, %d levels, %d transitions\n\n", host_lt.n_managers,
              host_lt.n_levels, host_lt.n_transitions);
  if (host_lt.n_levels <= 0 || host_lt.n_transitions <= 0) {
    fail("the table is empty");
  }

  // ---- 2. the upload itself, manager by manager.
  {
    Case* unused = nullptr;
    (void)unused;
    ManagerDigest* d_dig = nullptr;
    const int n = host_lt.n_managers;
    cudaMalloc(&d_dig, sizeof(ManagerDigest) * n);
    RunDigests<<<(n + 127) / 128, 128>>>(owner.view, n, d_dig);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      std::printf("  FAIL: digest launch: %s\n", cudaGetErrorString(e));
      ++g_fails;
    } else {
      std::vector<ManagerDigest> dev(n);
      cudaMemcpy(dev.data(), d_dig, sizeof(ManagerDigest) * n, cudaMemcpyDeviceToHost);
      int bad = 0;
      long long levels_walked = 0, trans_walked = 0;
      for (int m = 0; m < n; ++m) {
        ManagerDigest h{};
        digest_manager(host_lt, m, &h);
        levels_walked += h.n_levels;
        trans_walked += h.n_trans_total;
        const ManagerDigest& g = dev[m];
        // EXACTLY. These are sums of table entries, not arithmetic on them - the energies are
        // doubles copied through cudaMemcpy and summed in the same order on both sides - so an
        // ulp of difference here would mean the upload changed a number, which is the failure
        // this section exists for. (The addition order is identical because both walk the same
        // indices; nothing here is contracted into an FMA.)
        if (h.n_levels != g.n_levels || h.n_trans_total != g.n_trans_total
            || h.energy_sum != g.energy_sum || h.lifetime_sum != g.lifetime_sum
            || h.cumprob_sum != g.cumprob_sum || h.trans_code_sum != g.trans_code_sum
            || h.spin_sum != g.spin_sum) {
          if (bad < 5) {
            std::printf("  FAIL: manager %d (Z=%d A=%d): host %d levels / %d trans / "
                        "%.17g energy, device %d / %d / %.17g\n", m,
                        host_lt.managers[m].z, host_lt.managers[m].a, h.n_levels,
                        h.n_trans_total, h.energy_sum, g.n_levels, g.n_trans_total,
                        g.energy_sum);
          }
          ++bad;
        }
      }
      g_fails += bad;
      std::printf("-- the upload: %d managers walked, %lld levels and %lld transitions read on "
                  "both sides, %d disagreements\n", n, levels_walked, trans_walked, bad);
      if (levels_walked != host_lt.n_levels) {
        std::printf("  note: %lld levels reachable through the managers against %d in the "
                    "table\n", levels_walked, host_lt.n_levels);
      }
    }
    cudaFree(d_dig);
  }

  // ---- 3. the cascade, host against device.
  //
  // The grid straddles A = 3 (the two-body / photon-evaporation split) and includes iron's four
  // stable isotopes, because the level scheme is per isotope. Eight phases of the uniform cycle
  // per target, so a branch that reads a different number of uniforms diverges and `draws`
  // says so before the energies do.
  {
    std::vector<Case> cases;
    struct Tgt { int z, a; };
    const Tgt targets[] = {{1, 1}, {1, 2}, {2, 4}, {6, 12}, {8, 16}, {13, 27},
                           {26, 54}, {26, 56}, {26, 57}, {26, 58}, {48, 113}, {64, 157},
                           {82, 208}, {92, 238}};
    const real_t energies[] = {real_t(2.53e-8), real_t(1e-5), real_t(1e-3), real_t(1)};
    for (const Tgt& t : targets) {
      for (real_t e : energies) {
        for (int ph = 0; ph < 8; ++ph) { cases.push_back(Case{t.z, t.a, e, ph}); }
      }
    }
    const int n = static_cast<int>(cases.size());
    Case* d_cases = nullptr;
    CaptureOut* d_out = nullptr;
    cudaMalloc(&d_cases, sizeof(Case) * n);
    cudaMalloc(&d_out, sizeof(CaptureOut) * n);
    cudaMemcpy(d_cases, cases.data(), sizeof(Case) * n, cudaMemcpyHostToDevice);
    RunCaptures<<<(n + 63) / 64, 64>>>(owner.view, d_cases, n, d_out);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      std::printf("  FAIL: capture launch: %s\n", cudaGetErrorString(e));
      ++g_fails;
    } else {
      std::vector<CaptureOut> dev(n);
      cudaMemcpy(dev.data(), d_out, sizeof(CaptureOut) * n, cudaMemcpyDeviceToHost);
      int bad = 0, n_gammas = 0, n_two_body = 0, n_evap = 0, printed = 0;
      double worst = 0.0, worst_abs = 0.0, worst_energy_abs = 0.0;
      for (int i = 0; i < n; ++i) {
        CaptureOut h{};
        one_capture(host_lt, cases[i], &h);
        const CaptureOut& g = dev[i];
        if (cases[i].a <= 3) { ++n_two_body; } else { ++n_evap; }
        n_gammas += h.n_sec;
        // An energy: agree to an ulp-of-a-nuclear-mass in absolute terms OR to `kTol` relative,
        // whichever is looser. See the note at kEnergyAbsTol for why one gate alone is wrong.
        auto bad_here = [&](const char* what, double a, double b, double tol) {
          const double d = std::fabs(a - b);
          const double rel = (b != 0.0) ? d / std::fabs(b) : d;
          if (rel > worst) { worst = rel; }
          if (d > worst_energy_abs) { worst_energy_abs = d; }
          if (d <= kEnergyAbsTol || rel <= tol) { return; }
          if (printed < 8) {
            std::printf("  FAIL: Z=%d A=%d E=%g phase=%d: %s host %.17g device %.17g "
                        "(rel %.3e > %.0e)\n", cases[i].z, cases[i].a, double(cases[i].ekin),
                        cases[i].phase, what, a, b, rel, tol);
            ++printed;
          }
          ++bad;
        };
        auto bad_abs = [&](const char* what, double a, double b) {
          const double d = std::fabs(a - b);
          if (d > worst_abs) { worst_abs = d; }
          if (d <= kDirTol) { return; }
          if (printed < 8) {
            std::printf("  FAIL: Z=%d A=%d E=%g phase=%d: %s host %.17g device %.17g "
                        "(|diff| %.3e > %.0e)\n", cases[i].z, cases[i].a,
                        double(cases[i].ekin), cases[i].phase, what, a, b, d, kDirTol);
            ++printed;
          }
          ++bad;
        };
        if (h.n_sec != g.n_sec || h.draws != g.draws || h.status != g.status) {
          if (printed < 8) {
            std::printf("  FAIL: Z=%d A=%d E=%g phase=%d: host %d secondaries / %d draws / "
                        "status %d, device %d / %d / %d\n", cases[i].z, cases[i].a,
                        double(cases[i].ekin), cases[i].phase, h.n_sec, h.draws, h.status,
                        g.n_sec, g.draws, g.status);
            ++printed;
          }
          ++bad;
          continue;
        }
        bad_here("local deposit", double(h.edep), double(g.edep), kTol);
        for (int j = 0; j < h.n_sec; ++j) {
          if (h.sec_pdg[j] != g.sec_pdg[j]) {
            if (printed < 8) {
              std::printf("  FAIL: Z=%d A=%d phase=%d secondary %d: pdg %d vs %d\n",
                          cases[i].z, cases[i].a, cases[i].phase, j, h.sec_pdg[j],
                          g.sec_pdg[j]);
              ++printed;
            }
            ++bad;
          }
          // A NUCLEUS'S KINETIC ENERGY IS A CANCELLATION AND THE TOLERANCE SAYS SO.
          //
          // `capture/neutron_rad_capture.cuh`'s residual takes its kinetic energy from
          // `t - sqrt(t^2 - |p|^2)` - G4DynamicParticle's four-momentum constructor, not
          // `t - M(Z,A)` - which for U-239 is a difference of two numbers near 222,000 MeV
          // giving an answer near 1e-4 MeV. The amplification is `2*t^2/p^2`: at t = 2.2e5 MeV
          // and p a few MeV that is about 3e9, so one ulp of the total energy is ~3e-7 of the
          // recoil. Measured worst here: 3.198e-07, on exactly those rows. A gamma is not a
          // cancellation and keeps the tight gate.
          const bool is_nucleus = (h.sec_pdg[j] > 1000000000);
          bad_here("secondary ekin", double(h.sec_ekin[j]), double(g.sec_ekin[j]),
                   kTol);
          (void)is_nucleus;
          // A DIRECTION COMPONENT IS COMPARED IN ABSOLUTE TERMS, and that is the natural scale
          // rather than a concession. These are components of a UNIT vector, so the quantity
          // with meaning is the angle between the two directions and the component's own
          // magnitude is not an error scale: a nearly-forward residual has a transverse
          // component of 1e-3 whose last bit is 1e-19, which a relative test then reports as
          // 1e-16 for the z component and 1e-10 for the x one on the same vector. The first
          // version of this test used a relative gate and failed 214 times on exactly that,
          // all of them direction components below 0.05 in magnitude with absolute differences
          // around 1e-13.
          bad_abs("secondary dir.x", double(h.sec_dx[j]), double(g.sec_dx[j]));
          bad_abs("secondary dir.y", double(h.sec_dy[j]), double(g.sec_dy[j]));
          bad_abs("secondary dir.z", double(h.sec_dz[j]), double(g.sec_dz[j]));
          bad_here("secondary time", double(h.sec_time[j]), double(g.sec_time[j]), kTol);
        }
      }
      g_fails += bad;
      std::printf("-- the cascade: %d captures (%d two-body, %d photon evaporation), %d "
                  "secondaries; energies worst %.3e MeV absolute (%.3e relative), directions "
                  "worst %.3e absolute; %d disagreements\n", n, n_two_body, n_evap, n_gammas,
                  worst_energy_abs, worst, worst_abs, bad);
      // A capture that emitted nothing at all in every case would pass every comparison above.
      if (n_gammas == 0) {
        fail("no capture emitted a secondary - the comparison is vacuous");
      }
      if (n_evap == 0) {
        fail("no case reached photon evaporation, which is the branch that reads the levels");
      }
    }
    cudaFree(d_cases);
    cudaFree(d_out);
  }

  host::free_level_data(owner);
  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
