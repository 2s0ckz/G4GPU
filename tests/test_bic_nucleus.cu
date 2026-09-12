// The binary cascade's nucleus model, nuclear fields and Runge-Kutta propagation against
// ref/oracle/bic_*.csv.
//
// Five of the six comparisons here are EXACT, and the reason is a replay rather than a
// prescribed random engine. `G4Fancy3DNucleus::Init` cannot be driven by a cycle engine - its
// hard-core exclusion rejects every repeat of a position, so a periodic stream places at most
// eight nucleons and then exhausts its 1000*A attempt budget - and no seed makes HepJamesRandom
// and Philox agree. So `ref/dump/dump_bic.cc` dumps five complete sampled configurations
// (`bic_nucleons.csv`) and everything downstream is evaluated on THOSE: the outer radius, the
// nuclear mass, the five nuclear fields and the whole RK trajectory become deterministic
// functions of the configuration, and the sampling itself is what the statistical file checks.
//
//   bic_limits.csv       the two models' energy windows and energy-momentum check levels as
//                        QBBC and G4IonPhysicsXS leave them. Exact.
//   bic_density.csv      rho0, GetRelativeDensity, GetDensity, GetDeriv on a radial grid for
//                        twelve nuclides either side of the A < 17 dispatch. Exact.
//   bic_density_radius   GetRadius over ten relative densities including both ends of its
//                        guard, where it returns DBL_MAX. Exact.
//   bic_fermi.csv        GetFermiMomentum over (A, density), including density = 0. Exact.
//   bic_nucleus.csv      the (A, Z)-only scalars: both radii, GetMass, CoulombBarrier. Exact.
//   bic_nucleons.csv     the replay set. Checks the invariants Init guarantees - one binding
//                        energy per nucleon, zero total three-momentum, the hard-core
//                        exclusion, the proton/neutron counts - and the outer radius, exactly.
//   bic_field.csv        the five fields and their barriers on the replayed nuclei. Exact.
//   bic_rk.csv           G4RKPropagation::Transport, eleven initial states x five nuclei x
//                        twelve steps, position, both momenta, cascade state and the
//                        accumulated momentum transfer after each step. Exact.
//   bic_nucleus_stats    the port's OWN sampling, 20,000 nuclei per nuclide, against Geant4's:
//   bic_nucleus_moments  radial and momentum histograms with a binomial sigma per bin, and
//                        five moments with the two-sample mean test.
//
// **What the momentum-transfer column needs.** `G4RKPropagation::Transport` resets
// `theMomentumTranfer` at the TOP of every call, so the dumped value is per-step and not
// cumulative - the oracle's own rows show 41.5 MeV on the step that crosses the nucleus and
// exactly zero on every step after it. `RkPropagation::transport_one` leaves the reset to the
// caller, because Geant4 resets once per call and then loops over the whole active list, so
// this test resets before each step to compare like for like.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "physics/hadronic/bic/bic_params.cuh"
#include "physics/hadronic/bic/rk_propagation.cuh"

using namespace g4gpu;

/// The deliverables are device-callable, and a `__host__ __device__` template that is only ever
/// instantiated from the host is never compiled for the device at all. These kernels are never
/// launched - the machine's GPU belongs to the integration builds - but instantiating them is
/// what proves the module is device code, and `-Xptxas -v` on this translation unit is what
/// reports its register and stack cost. The cascade's workspace is a struct of pointers for
/// exactly this reason: a nucleus is 250 nucleons of 72 bytes and does not belong in a frame.
__global__ void bic_nucleus_probe(bic::Nucleus3D nuc, bic::Nucleus3DScratch sc, int a, int z,
                                  bic::NucleusReport* out) {
  Philox<double> rng(1u, 2u, 3u);
  out[0] = bic::nucleus_init(nuc, sc, a, z, rng);
}

__global__ void bic_rk_probe(bic::RkPropagation prop, bic::KineticTrack* kt, double dt,
                             bic::RkAdvanceReport* rep) {
  prop.transport_one(kt[0], dt, rep[0]);
}

namespace {

int fails = 0;

// ---------------------------------------------------------------------------------------------
// Comparison bookkeeping - the same shape tests/test_precompound.cu uses
// ---------------------------------------------------------------------------------------------

struct Bucket {
  const char* name;
  long long n = 0;
  double worst = 0.0;
  std::string where;
  double tol = 1e-13;
};

std::vector<Bucket> buckets;

int new_bucket(const char* name, double tol) {
  Bucket b;
  b.name = name;
  b.tol = tol;
  buckets.push_back(b);
  return static_cast<int>(buckets.size()) - 1;
}

void cmp(int bi, double got, double want, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  const double scale = (std::fabs(want) > 0.0) ? std::fabs(want) : 1.0;
  const double rel = std::fabs(got - want) / scale;
  if (rel > b.worst) {
    b.worst = rel;
    b.where = where;
  }
}

/// `cmp` with an absolute floor on the denominator, for a quantity that is the DIFFERENCE of
/// two terms of a known natural size.
///
/// The proton field is `-p_F^2/(2m) + barrier` and the pi- field is `V - barrier`, and both
/// cross zero inside the nucleus - the proton's at 4.8 fm on C12, where the two terms are 2.3
/// MeV each and their difference is 1e-13 MeV. A relative comparison there divides a rounding
/// by a cancellation and reports 8e-14 as a failure of a 1e-14 test. `scale` is what the
/// quantity is made of (1 MeV for a nuclear potential), so this is a relative test everywhere
/// the answer is not a cancellation and an absolute one where it is.
void cmp_scaled(int bi, double got, double want, double scale, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  const double denom = (std::fabs(want) > scale) ? std::fabs(want) : scale;
  const double rel = std::fabs(got - want) / denom;
  if (rel > b.worst) {
    b.worst = rel;
    b.where = where;
  }
}

/// An ABSOLUTE comparison, for a quantity whose own value carries no scale.
///
/// `theMomentumTranfer` is `p_before - p_after` with both around 1000 MeV, so once the track
/// has left the nucleus it is the rounding of that subtraction - the oracle's own rows read
/// 5.7e-14 and -1.4e-14 MeV on a step where nothing happened. Dividing by those reports a
/// factor of six as a failure. The bucket's tolerance is then in MeV, and 1e-9 MeV is four
/// orders below anything the cascade can notice.
void cmp_abs(int bi, double got, double want, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  const double d = std::fabs(got - want);
  if (d > b.worst) {
    b.worst = d;
    b.where = where;
  }
}

void cmp_int(int bi, long long got, long long want, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  if (got != want) {
    b.worst = 1.0;
    if (b.where.empty()) {
      b.where = where + " got " + std::to_string(got) + " want " + std::to_string(want);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// CSV reading
// ---------------------------------------------------------------------------------------------

std::string oracle_dir() {
  const char* env = std::getenv("G4GPU_ORACLE");
  return (env != nullptr) ? std::string(env) : std::string("ref/oracle");
}

std::vector<std::vector<std::string>> read_csv(const std::string& name) {
  std::vector<std::vector<std::string>> rows;
  const std::string path = oracle_dir() + "/" + name;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) {
    std::printf("MISSING %s\n", path.c_str());
    ++fails;
    return rows;
  }
  char line[8192];
  bool header = true;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    if (header) { header = false; continue; }
    std::vector<std::string> f2;
    std::string cur;
    for (const char* p = line; *p != '\0'; ++p) {
      if (*p == ',') { f2.push_back(cur); cur.clear(); }
      else if (*p != '\n' && *p != '\r') { cur.push_back(*p); }
    }
    f2.push_back(cur);
    if (!f2.empty()) { rows.push_back(f2); }
  }
  std::fclose(f);
  return rows;
}

double dv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::atof(f[i].c_str()) : 0.0;
}
int iv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::atoi(f[i].c_str()) : 0;
}
std::string sv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? f[i] : std::string();
}

// ---------------------------------------------------------------------------------------------
// Statistics, as tests/test_deex_breakup.cu and tests/test_precompound.cu define them
// ---------------------------------------------------------------------------------------------

/// A per-nucleon histogram bin is Binomial(N*A, p), not Poisson: every nucleus contributes
/// exactly A entries to the radial histogram, so the total is fixed and a Poisson sigma of
/// sqrt(n) would be too large by the completeness factor.
double binomial_z(long long n1, long long n2, long long ntot) {
  if (ntot < 2) { return 0.0; }
  const double p1 = static_cast<double>(n1) / static_cast<double>(ntot);
  const double p2 = static_cast<double>(n2) / static_cast<double>(ntot);
  const double s2 =
      static_cast<double>(ntot) * (p1 * (1.0 - p1) + p2 * (1.0 - p2));
  if (!(s2 > 0.0)) { return (n1 == n2) ? 0.0 : 1.e9; }
  return std::fabs(static_cast<double>(n1 - n2)) / std::sqrt(s2);
}

double mean_z(double m1, double v1, long long n1, double m2, double v2, long long n2) {
  if (n1 < 2 || n2 < 2) { return 0.0; }
  const double s2 = v1 / static_cast<double>(n1) + v2 / static_cast<double>(n2);
  if (!(s2 > 0.0)) {
    return (std::fabs(m1 - m2) <= 1.e-9 * (1.0 + std::fabs(m2))) ? 0.0 : 1.e9;
  }
  return std::fabs(m1 - m2) / std::sqrt(s2);
}

// ---------------------------------------------------------------------------------------------
// Storage. Static, not local: a Pb208 nucleus plus its scratch is 60 kB and the statistical
// campaign needs it for the whole run.
// ---------------------------------------------------------------------------------------------

bic::Nucleon g_nucleons[bic::kMaxNucleons];
Vec3<double> g_momentum[bic::kMaxNucleons];
double g_fermi_p[bic::kMaxNucleons];
bic::NucleusSortEntry g_test_sums[bic::kMaxNucleons];
double g_flat[bic::kFlatBlock];
/// ONE PAIR OF FIELD TABLES PER REPLAY NUCLEUS, and not one pair shared.
///
/// `NucleonField` holds a POINTER into caller storage, so five propagations built on one pair of
/// tables all read whichever nucleus was built last - and the table length differs per nucleus,
/// because it runs to `2*GetOuterRadius()`. Sharing them made the O16 neutron field read
/// Pb208's table and the comparison came out 1.0e4 relative, which looked exactly like a wrong
/// interpolation. Six slots for five nuclei, so adding one to the replay set is a compile
/// error rather than an overwrite.
constexpr int kMaxReplay = 6;
double g_ptable[kMaxReplay][bic::kMaxFieldTable];
double g_ntable[kMaxReplay][bic::kMaxFieldTable];

bic::Nucleus3D make_nucleus_shell() {
  bic::Nucleus3D n;
  n.nucleons = g_nucleons;
  n.capacity = bic::kMaxNucleons;
  return n;
}
bic::Nucleus3DScratch make_scratch() {
  bic::Nucleus3DScratch s;
  s.momentum = g_momentum;
  s.fermi_p = g_fermi_p;
  s.test_sums = g_test_sums;
  s.flat_block = g_flat;
  s.capacity = bic::kMaxNucleons;
  return s;
}

/// One replayed nucleus: (A, Z) and the dumped nucleon list, assembled into the port's own
/// `Nucleus3D` so that every function the cascade calls on a nucleus can be called on it.
struct Replay {
  std::string name;
  int a = 0, z = 0;
  double outer_radius = 0.0;   ///< the oracle's, for the exact comparison
  double mass = 0.0;
  std::vector<bic::Nucleon> nucleons;
};

/// Fills a `Nucleus3D` from a replayed configuration. Everything except the nucleon array is a
/// deterministic function of (A, Z) and is rebuilt by the same code `nucleus_init` uses, so
/// this checks that dispatch too: a nucleus whose density or hard-core distance came out wrong
/// would give the wrong outer radius and the wrong field.
void load_replay(const Replay& r, bic::Nucleus3D& nuc) {
  nuc.my_a = r.a;
  nuc.my_z = r.z;
  nuc.my_l = 0;
  nuc.current = -1;
  nuc.excitation = 0.0;
  nuc.nucleondistance = 0.8 * deex::fermi();
  if (r.a < 17) {
    nuc.density = bic::make_shell_model_density(r.a, r.z);
    if (r.a == 12) { nuc.nucleondistance = 0.9 * deex::fermi(); }
  } else {
    nuc.density = bic::make_fermi_density(r.a, r.z);
  }
  nuc.fermi.init(r.a, r.z);
  for (int i = 0; i < r.a; ++i) { nuc.nucleons[i] = r.nucleons[i]; }
}

}  // namespace

int main() {
  // -------------------------------------------------------------------------------------------
  // 1. The two models' limits
  // -------------------------------------------------------------------------------------------
  {
    const int b = new_bucket("Limits", 0.0);
    const auto rows = read_csv("bic_limits.csv");
    for (const auto& row : rows) {
      const std::string model = sv(row, 0);
      const double emin = dv(row, 2), emax = dv(row, 3);
      if (model == "G4BinaryCascade") {
        cmp(b, bic::bic_qbbc_min_energy(), emin, "BIC emin");
        cmp(b, bic::bic_qbbc_max_energy(), emax, "BIC emax");
        cmp(b, bic::bic_ep_check_relative(), dv(row, 4), "BIC ep rel");
        cmp(b, bic::bic_ep_check_absolute(), dv(row, 5), "BIC ep abs");
      } else if (model == "G4BinaryCascade_ctor") {
        cmp(b, bic::bic_ctor_max_energy(), emax, "BIC ctor emax");
      } else if (model == "G4BinaryLightIonReaction") {
        cmp(b, bic::blir_qbbc_min_energy(), emin, "BLIR emin");
        cmp(b, bic::blir_qbbc_max_energy(), emax, "BLIR emax");
        // DBL_MAX on both, which is the default G4HadronicInteraction leaves and which
        // G4BinaryLightIonReaction never overrides - so its energy-momentum check is off where
        // G4BinaryCascade's is (1%, 1 MeV). Asserted on the ORACLE side as well as the port's,
        // so a release that starts checking fails this test loudly.
        cmp(b, bic::blir_ep_check_relative(), dv(row, 4), "BLIR ep rel");
        cmp(b, bic::blir_ep_check_absolute(), dv(row, 5), "BLIR ep abs");
        if (!(dv(row, 4) > 1e300)) {
          std::printf("ORACLE CHANGED: G4BinaryLightIonReaction now has an ep check level %g\n",
                      dv(row, 4));
          ++fails;
        }
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 2. The two nuclear densities
  // -------------------------------------------------------------------------------------------
  {
    const int b_rho = new_bucket("DensityRho0", 1e-15);
    const int b_rel = new_bucket("DensityRelative", 1e-15);
    const int b_abs = new_bucket("DensityAbsolute", 1e-15);
    const int b_der = new_bucket("DensityDeriv", 1e-15);
    const int b_kind = new_bucket("DensityKind", 0.0);
    for (const auto& row : read_csv("bic_density.csv")) {
      const int a = iv(row, 1), z = iv(row, 2), kind = iv(row, 3);
      const bic::NuclearDensity d = (kind == 0) ? bic::make_shell_model_density(a, z)
                                                : bic::make_fermi_density(a, z);
      // The dispatch itself: the port must choose the same class for this A.
      cmp_int(b_kind, (a < 17) ? 0 : 1, kind, sv(row, 0));
      const std::string w = sv(row, 0) + " r=" + sv(row, 5);
      cmp(b_rho, d.rho0, dv(row, 4), w);
      const Vec3<double> pos{0.0, 0.0, dv(row, 5) * deex::fermi()};
      cmp(b_rel, d.relative_density(pos), dv(row, 6), w);
      cmp(b_abs, d.density(pos), dv(row, 7), w);
      cmp(b_der, d.deriv(pos), dv(row, 8), w);
    }
    const int b_rad = new_bucket("DensityRadius", 1e-15);
    for (const auto& row : read_csv("bic_density_radius.csv")) {
      const int a = iv(row, 1), z = iv(row, 2), kind = iv(row, 3);
      const bic::NuclearDensity d = (kind == 0) ? bic::make_shell_model_density(a, z)
                                                : bic::make_fermi_density(a, z);
      cmp(b_rad, d.radius(dv(row, 4)), dv(row, 5), sv(row, 0) + " mrd=" + sv(row, 4));
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3. G4FermiMomentum
  // -------------------------------------------------------------------------------------------
  {
    // 1e-14 and not 1e-15, for a reason that is not this package's: `src/data/g4pow.hh`'s
    // `g4pow_a13_high` writes the above-table branch as `exp(log(a)/3)` where G4Pow writes
    // `G4Exp(G4Log(a)*onethird)` with `onethird = 1.0/3.0`, and a division by 3 is not the same
    // double as a multiplication by 1/3. Measured: A13(4e32) comes out 73680629972.807739 here
    // and 73680629972.807632 in Geant4, 1.5e-15 relative, and `GetFermiMomentum` is
    // `constofpmax * A13(density*A)` so it inherits exactly that. Every density above 1.9e2
    // in these units takes the branch. Not fixed here - g4pow.hh is shared with the EM port,
    // whose tests sit near their own tolerances - and recorded in docs/RISK.md V73.
    const int b = new_bucket("FermiMomentum", 1e-14);
    for (const auto& row : read_csv("bic_fermi.csv")) {
      bic::FermiMomentum fm;
      fm.init(iv(row, 0), iv(row, 1));
      cmp(b, fm.fermi_momentum(dv(row, 2)), dv(row, 3),
          "A=" + sv(row, 0) + " rho=" + sv(row, 2));
    }
  }

  // -------------------------------------------------------------------------------------------
  // 4. The (A, Z)-only scalars of G4Fancy3DNucleus
  // -------------------------------------------------------------------------------------------
  {
    const int b_r = new_bucket("NucleusRadius", 1e-15);
    const int b_m = new_bucket("NucleusMass", 1e-15);
    const int b_c = new_bucket("CoulombBarrier", 1e-15);
    const int b_be = new_bucket("BindingEnergy", 1e-15);
    bic::Nucleus3D nuc = make_nucleus_shell();
    bic::NucleusReport rep;
    for (const auto& row : read_csv("bic_nucleus.csv")) {
      const int a = iv(row, 1), z = iv(row, 2);
      nuc.my_a = a;
      nuc.my_z = z;
      nuc.my_l = 0;
      nuc.density = (a < 17) ? bic::make_shell_model_density(a, z)
                             : bic::make_fermi_density(a, z);
      const std::string w = sv(row, 0);
      cmp(b_r, nuc.nuclear_radius(), dv(row, 4), w + " R(0.5)");
      cmp(b_r, nuc.nuclear_radius(0.001), dv(row, 5), w + " R(0.001)");
      cmp(b_m, nuc.mass(rep), dv(row, 7), w + " mass");
      cmp(b_c, nuc.coulomb_barrier(), dv(row, 8), w + " barrier");
      cmp(b_be, deex::binding_energy(a, z), dv(row, 9), w + " BE");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 5. The replay set, and the invariants Init guarantees
  // -------------------------------------------------------------------------------------------
  std::vector<Replay> replays;
  {
    std::map<std::string, Replay> byname;
    for (const auto& row : read_csv("bic_nucleons.csv")) {
      const std::string name = sv(row, 0);
      Replay& r = byname[name];
      r.name = name;
      r.a = iv(row, 1);
      r.z = iv(row, 2);
      r.outer_radius = dv(row, 13);
      r.mass = dv(row, 14);
      bic::Nucleon n;
      const int pdg = iv(row, 4);
      n.type = (pdg == 2212) ? bic::kProton : ((pdg == 2112) ? bic::kNeutron : bic::kLambda);
      n.position = Vec3<double>{dv(row, 5), dv(row, 6), dv(row, 7)};
      n.momentum = deex::LorentzVector(Vec3<double>{dv(row, 8), dv(row, 9), dv(row, 10)},
                                       dv(row, 11));
      n.binding_energy = dv(row, 12);
      r.nucleons.push_back(n);
    }
    for (auto& kv : byname) { replays.push_back(kv.second); }

    const int b_router = new_bucket("ReplayOuterRadius", 1e-15);
    const int b_mass = new_bucket("ReplayMass", 1e-15);
    const int b_be = new_bucket("ReplayNucleonBinding", 1e-15);
    const int b_e = new_bucket("ReplayNucleonEnergy", 1e-15);
    const int b_inv = new_bucket("ReplayInvariants", 0.0);
    const int b_psum = new_bucket("ReplayMomentumSum", 0.0);
    bic::Nucleus3D nuc = make_nucleus_shell();
    bic::NucleusReport rep;
    for (const Replay& r : replays) {
      if (static_cast<int>(r.nucleons.size()) != r.a) {
        std::printf("REPLAY %s: %d nucleons for A = %d\n", r.name.c_str(),
                    static_cast<int>(r.nucleons.size()), r.a);
        ++fails;
        continue;
      }
      load_replay(r, nuc);
      // The outer radius is the furthest nucleon plus the hard-core distance, so it checks the
      // A == 12 nucleondistance branch at the same time - C12's is 0.9 fm and every other
      // nuclide's is 0.8 fm, a 0.1 fm difference this comparison would catch at 1e-2.
      cmp(b_router, nuc.outer_radius(), r.outer_radius, r.name + " outer radius");
      cmp(b_mass, nuc.mass(rep), r.mass, r.name + " mass");

      // Init writes ONE binding energy onto every nucleon: GetBindingEnergy(A,Z)/A.
      const double eb = deex::binding_energy(r.a, r.z) / static_cast<double>(r.a);
      int n_proton = 0;
      double dmin2 = 1e30;
      Vec3<double> psum{0.0, 0.0, 0.0};
      for (int i = 0; i < r.a; ++i) {
        cmp(b_be, eb, r.nucleons[i].binding_energy, r.name + " nucleon binding");
        // The nucleon's ENERGY, which `ChooseFermiMomenta`'s last loop sets to
        // `GetPDGMass() - BindingEnergy()/A` - the line that puts every nucleon off its mass
        // shell downwards and selects the QGS arm of preco::propagate_residual (RISK V50).
        //
        // This comparison exists because its absence was a hole. The port's own sampled
        // energies are read by nothing else: the off-shell invariant below is asserted on the
        // ORACLE's nucleons, and the statistical campaign histograms only |p| and r. So
        // perturbing the sign of `BindingEnergy()/A` in `choose_fermi_momenta` - putting every
        // nucleon 8 MeV ABOVE its mass shell instead of below it - passed the whole test.
        // Perturbation 24 of this package's campaign. It compares the port's FORMULA against
        // Geant4's dumped energy, so it is exact and it cannot be satisfied by a copy.
        cmp(b_e, r.nucleons[i].pdg_mass() - eb, r.nucleons[i].momentum.e,
            r.name + " nucleon energy");
        if (r.nucleons[i].type == bic::kProton) { ++n_proton; }
        psum = psum + r.nucleons[i].momentum.v;
        for (int j = i + 1; j < r.a; ++j) {
          const Vec3<double> d = r.nucleons[i].position - r.nucleons[j].position;
          const double d2 = g4gpu::mag2(d);
          if (d2 < dmin2) { dmin2 = d2; }
        }
      }
      cmp_int(b_inv, n_proton, r.z, r.name + " proton count");
      // The hard-core exclusion: no pair closer than `nucleondistance`. C12's cluster branch
      // uses `<=` where the general branch uses `>`, so the bound is the same number.
      cmp_int(b_inv, (std::sqrt(dmin2) >= nuc.nucleondistance * (1.0 - 1e-12)) ? 1 : 0, 1,
              r.name + " hard core (dmin = " + std::to_string(std::sqrt(dmin2) / 1e-12) +
                  " fm, nd = " + std::to_string(nuc.nucleondistance / 1e-12) + " fm)");
      // ReduceSum's postcondition. The sum is not exactly zero in Geant4 either - the last
      // nucleon is assigned `-sum` or `delta-sum` in double arithmetic - so the bound is the
      // rounding of a sum of A momenta of a few hundred MeV: A * 1e-13 MeV.
      const double psum_mag = g4gpu::mag(psum);
      const double bound = static_cast<double>(r.a) * 1e-12;
      cmp_int(b_psum, (psum_mag < bound) ? 1 : 0, 1,
              r.name + " |sum p| = " + std::to_string(psum_mag) + " MeV");

      // Every nucleon is off its mass shell downwards - the fact RISK V50 turns on. Asserted,
      // because a caller that assumes otherwise hands preco::propagate_residual a nucleus that
      // selects its QGS arm.
      for (int i = 0; i < r.a; ++i) {
        const double m = r.nucleons[i].momentum.mag();
        cmp_int(b_inv, (m < r.nucleons[i].pdg_mass()) ? 1 : 0, 1,
                r.name + " nucleon off shell downwards");
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 6. The five nuclear fields on the replayed nuclei
  // -------------------------------------------------------------------------------------------
  {
    // 1e-13 and not 1e-15, and the reason is docs/RISK.md V73 again: the nucleon field is
    // `-p_F^2/(2m) + barrier` and p_F comes through `g4pow_a13`, which is 1.5e-15 off G4Pow's
    // own A13 for these arguments. Squaring doubles that, and p_F^2/(2m) reaches 5 MeV, so the
    // floor is 1.5e-14 MeV - measured at 2.4e-14 on Al27 at 5.4 fm, where the barrier nearly
    // cancels the well.
    const int b_f = new_bucket("NuclearField", 1e-13);
    const int b_b = new_bucket("NuclearBarrier", 1e-15);
    bic::Nucleus3D nuc = make_nucleus_shell();
    bic::NucleusReport rep;
    std::map<std::string, bic::RkPropagation> props;
    if (static_cast<int>(replays.size()) > kMaxReplay) {
      std::printf("more replay nuclei (%d) than field-table slots (%d)\n",
                  static_cast<int>(replays.size()), kMaxReplay);
      ++fails;
    }
    int slot = 0;
    for (const Replay& r : replays) {
      load_replay(r, nuc);
      props[r.name] = bic::make_rk_propagation(nuc, rep, g_ptable[slot], g_ntable[slot],
                                               bic::kMaxFieldTable);
      if (props[r.name].proton.nucleon.overflow) {
        std::printf("FIELD TABLE OVERFLOW for %s\n", r.name.c_str());
        ++fails;
      }
      ++slot;
    }
    for (const auto& row : read_csv("bic_field.csv")) {
      const std::string name = sv(row, 0);
      auto it = props.find(name);
      if (it == props.end()) { continue; }
      const bic::RkPropagation& p = it->second;
      const int pdg = iv(row, 3);
      const bic::SpeciesField* sf = p.find_field(pdg);
      if (sf == nullptr) { continue; }
      const Vec3<double> pos{0.0, 0.0, dv(row, 4) * deex::fermi()};
      const std::string w = name + " pdg=" + sv(row, 3) + " r=" + sv(row, 4);
      // 1 MeV is the scale of a nuclear potential, so this is relative except where the field
      // crosses zero. See cmp_scaled.
      cmp_scaled(b_f, sf->field(pos, p.density), dv(row, 5), 1.0, w);
      cmp(b_b, sf->barrier(), dv(row, 6), w + " barrier");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 7. G4RKPropagation::Transport, step by step
  // -------------------------------------------------------------------------------------------
  {
    const int b_pos = new_bucket("RkPosition", 1e-11);
    const int b_mom = new_bucket("RkMomentum", 1e-11);
    const int b_tr = new_bucket("RkTrackingMomentum", 1e-11);
    const int b_st = new_bucket("RkState", 0.0);
    // ABSOLUTE, in MeV - see cmp_abs. The transfer is a difference of two 1000 MeV momenta.
    const int b_mt = new_bucket("RkMomentumTransfer(MeV)", 1e-9);

    // The eleven cases dump_bic.cc drives, in the same order - `case` is the index.
    struct RkCase { int pdg; double kin; double b_fm; double z0_fm; double dt; int n; };
    const double mp = units::proton_mass_c2<double>();
    const double mn = units::neutron_mass_c2<double>();
    const double mpi = bic::pdg_mass_pion_charged();
    const double mpi0 = bic::pdg_mass_pion_zero();
    const RkCase kCases[] = {
      {2212, 200.0, 1.0, -20.0, 1.0e-5, 12}, {2212, 50.0, 3.0, -20.0, 1.0e-5, 12},
      {2212,  20.0, 0.5, -20.0, 2.0e-5, 12}, {2112, 200.0, 1.0, -20.0, 1.0e-5, 12},
      {2112,  30.0, 4.0, -20.0, 2.0e-5, 12}, {2112,   5.0, 0.0, -20.0, 5.0e-5, 12},
      { 211, 150.0, 2.0, -20.0, 1.0e-5, 12}, {-211, 150.0, 2.0, -20.0, 1.0e-5, 12},
      { 111, 150.0, 2.0, -20.0, 1.0e-5, 12}, {2212, 120.0, 0.0,   0.0, 1.0e-5, 12},
      {2112,  80.0, 0.0,   0.0, 1.0e-5, 12},
    };
    const int n_cases = static_cast<int>(sizeof kCases / sizeof kCases[0]);

    // Index the oracle by (name, case, step).
    std::map<std::string, std::vector<std::string>> oracle;
    for (const auto& row : read_csv("bic_rk.csv")) {
      const std::string key = sv(row, 0) + "/" + sv(row, 3) + "/" + sv(row, 5);
      oracle[key] = row;
    }

    bic::Nucleus3D nuc = make_nucleus_shell();
    bic::NucleusReport nrep;
    int n_quick = 0, n_quick_cases = 0;
    int rslot = 0;
    for (const Replay& r : replays) {
      load_replay(r, nuc);
      bic::RkPropagation prop = bic::make_rk_propagation(nuc, nrep, g_ptable[rslot],
                                                         g_ntable[rslot],
                                                         bic::kMaxFieldTable);
      ++rslot;
      for (int ic = 0; ic < n_cases; ++ic) {
        const RkCase& c = kCases[ic];
        double mass = mp;
        if (c.pdg == 2112) { mass = mn; }
        else if (c.pdg == 211 || c.pdg == -211) { mass = mpi; }
        else if (c.pdg == 111) { mass = mpi0; }

        bic::KineticTrack kt;
        kt.pdg = c.pdg;
        kt.pdg_mass = mass;
        kt.charge = (c.pdg == 2212 || c.pdg == 211) ? 1 : ((c.pdg == -211) ? -1 : 0);
        kt.baryon_number = (c.pdg == 2212 || c.pdg == 2112) ? 1 : 0;
        kt.formation_time = 0.0;
        kt.position = Vec3<double>{c.b_fm * deex::fermi(), 0.0, c.z0_fm * deex::fermi()};
        const double pmag = std::sqrt(c.kin * (c.kin + 2.0 * mass));
        kt.set4_momentum(
            deex::LorentzVector(Vec3<double>{0.0, 0.0, pmag}, c.kin + mass));
        kt.state = (c.z0_fm < -1.0) ? bic::kOutside : bic::kInside;

        bic::RkAdvanceReport arep;
        for (int s = 0; s < c.n; ++s) {
          // Transport resets theMomentumTranfer at the top of every call; see the header.
          prop.momentum_transfer = Vec3<double>{0.0, 0.0, 0.0};
          prop.transport_one(kt, c.dt, arep);
          const std::string key =
              r.name + "/" + std::to_string(ic) + "/" + std::to_string(s);
          auto it = oracle.find(key);
          if (it == oracle.end()) { continue; }
          const std::vector<std::string>& row = it->second;
          const std::string w = key + " pdg=" + std::to_string(c.pdg);
          cmp_int(b_st, kt.state, iv(row, 6), w + " state");
          cmp(b_pos, kt.position.x, dv(row, 7), w + " pos.x");
          cmp(b_pos, kt.position.y, dv(row, 8), w + " pos.y");
          cmp(b_pos, kt.position.z, dv(row, 9), w + " pos.z");
          cmp(b_mom, kt.momentum4().v.x, dv(row, 10), w + " p.x");
          cmp(b_mom, kt.momentum4().v.y, dv(row, 11), w + " p.y");
          cmp(b_mom, kt.momentum4().v.z, dv(row, 12), w + " p.z");
          cmp(b_mom, kt.momentum4().e, dv(row, 13), w + " p.e");
          cmp(b_tr, kt.tracking_momentum().v.x, dv(row, 14), w + " tr.x");
          cmp(b_tr, kt.tracking_momentum().v.y, dv(row, 15), w + " tr.y");
          cmp(b_tr, kt.tracking_momentum().v.z, dv(row, 16), w + " tr.z");
          cmp(b_tr, kt.tracking_momentum().e, dv(row, 17), w + " tr.e");
          cmp_abs(b_mt, prop.momentum_transfer.x, dv(row, 18), w + " mt.x");
          cmp_abs(b_mt, prop.momentum_transfer.y, dv(row, 19), w + " mt.y");
          cmp_abs(b_mt, prop.momentum_transfer.z, dv(row, 20), w + " mt.z");
        }
        n_quick += arep.n_quick_advance;
        if (arep.n_quick_advance > 0) { ++n_quick_cases; }
        if (arep.step_became_zero) {
          std::printf("RK: integration step became zero for case %d on %s\n", ic,
                      r.name.c_str());
          ++fails;
        }
      }
    }
    // Reported rather than asserted: it is a count of how often the driver's overshoot clamp
    // left a sub-minimum step, which is a property of the eleven cases and not of the port.
    // It is printed because this header said for one draft that the number was zero. BOTH
    // numbers are printed because one draft of rk_propagation.cuh's header quoted them as if
    // they were one number: the sub-steps are what the arm ran, the (nucleus, case) pairs are
    // how many of the 55 trajectories contain one.
    std::printf("  RK: %d sub-steps in %d of %d (nucleus, case) pairs took the QuickAdvance "
                "arm (h <= fMinimumStep)\n",
                n_quick, n_quick_cases, static_cast<int>(replays.size()) * n_cases);
  }

  // -------------------------------------------------------------------------------------------
  // 8. The sampling itself, statistically
  // -------------------------------------------------------------------------------------------
  {
    const long kN = 20000;
    const int kNRad = 40, kNMom = 40;
    const double kRadMax = 20.0, kMomMax = 400.0;

    struct StatCase { int a, z; const char* name; };
    const StatCase kStat[] = {{12, 6, "C12"}, {16, 8, "O16"}, {27, 13, "Al27"},
                              {56, 26, "Fe56"}, {208, 82, "Pb208"}};

    std::map<std::string, std::vector<long long>> oracle_hist;
    long long oracle_n = 0;
    for (const auto& row : read_csv("bic_nucleus_stats.csv")) {
      const std::string key = sv(row, 0) + "/" + sv(row, 4);
      auto& v = oracle_hist[key];
      const int bin = iv(row, 5);
      if (static_cast<int>(v.size()) <= bin) { v.resize(bin + 1, 0); }
      v[bin] = std::atoll(row[8].c_str());
      oracle_n = std::atoll(row[3].c_str());
    }
    std::map<std::string, std::pair<double, double>> oracle_mom;
    for (const auto& row : read_csv("bic_nucleus_moments.csv")) {
      oracle_mom[sv(row, 0) + "/" + sv(row, 4)] = {dv(row, 5), dv(row, 6)};
    }

    const int b_hist = new_bucket("SamplingHistogram(sigma)", 5.0);
    const int b_mom = new_bucket("SamplingMoments(sigma)", 5.0);
    const int b_rep = new_bucket("SamplingReports", 0.0);

    bic::Nucleus3D nuc = make_nucleus_shell();
    bic::Nucleus3DScratch sc = make_scratch();
    for (const StatCase& s : kStat) {
      std::vector<long long> hr(kNRad, 0), hp(kNMom, 0);
      double sum_router = 0, sum_router2 = 0, sum_psum = 0, sum_psum2 = 0;
      double sum_dmin = 0, sum_dmin2 = 0, sum_r = 0, sum_r2 = 0, sum_p = 0, sum_p2 = 0;
      long long n_nucleons = 0;
      int n_fatal = 0, n_reduce_failed = 0, n_zeroed = 0;
      long long n_on_shell = 0;

      for (long ev = 0; ev < kN; ++ev) {
        Philox<double> rng(static_cast<unsigned>(s.a), static_cast<unsigned>(ev), 0x9E37u);
        const bic::NucleusReport rep = bic::nucleus_init(nuc, sc, s.a, s.z, rng);
        if (rep.fatal()) { ++n_fatal; continue; }
        if (rep.reduce_sum_failed) { ++n_reduce_failed; }
        n_zeroed += rep.proton_momentum_zeroed;
        Vec3<double> psum{0.0, 0.0, 0.0};
        double dmin2 = 1e30;
        for (int i = 0; i < s.a; ++i) {
          const double r = g4gpu::mag(nuc.nucleons[i].position) / deex::fermi();
          const double p = g4gpu::mag(nuc.nucleons[i].momentum.v);
          // Asserted on the PORT's own nucleons and not only on the replayed ones, so that a
          // sign error in the energy is caught by the sampling too - see the note on
          // `b_e` above. `n_on_shell` counts a nucleon that came out on or above its mass
          // shell, which would silently select the QGS arm of preco::propagate_residual.
          if (!(nuc.nucleons[i].momentum.mag() < nuc.nucleons[i].pdg_mass())) {
            ++n_on_shell;
          }
          psum = psum + nuc.nucleons[i].momentum.v;
          sum_r += r; sum_r2 += r * r;
          sum_p += p; sum_p2 += p * p;
          ++n_nucleons;
          int ib = static_cast<int>(r / (kRadMax / kNRad));
          if (ib >= kNRad) { ib = kNRad - 1; }
          ++hr[ib];
          int jb = static_cast<int>(p / (kMomMax / kNMom));
          if (jb >= kNMom) { jb = kNMom - 1; }
          ++hp[jb];
          for (int j = i + 1; j < s.a; ++j) {
            const Vec3<double> d = nuc.nucleons[i].position - nuc.nucleons[j].position;
            const double d2 = g4gpu::mag2(d);
            if (d2 < dmin2) { dmin2 = d2; }
          }
        }
        const double router = nuc.outer_radius() / deex::fermi();
        sum_router += router; sum_router2 += router * router;
        const double pm = g4gpu::mag(psum);
        sum_psum += pm; sum_psum2 += pm * pm;
        const double dmin = (s.a > 1) ? std::sqrt(dmin2) / deex::fermi() : 0.0;
        sum_dmin += dmin; sum_dmin2 += dmin * dmin;
      }

      cmp_int(b_rep, n_fatal, 0, std::string(s.name) + " fatal reports");
      cmp_int(b_rep, n_on_shell, 0, std::string(s.name) + " nucleons NOT off shell downwards");
      if (n_reduce_failed > 0 || n_zeroed > 0) {
        std::printf("  %-6s reports: ReduceSum failed %d, proton momentum zeroed %d (of %ld "
                    "nuclei)\n", s.name, n_reduce_failed, n_zeroed, kN);
      }

      const long long ntot_r = n_nucleons;
      auto& orr = oracle_hist[std::string(s.name) + "/radius"];
      auto& orp = oracle_hist[std::string(s.name) + "/momentum"];
      for (int i = 0; i < kNRad && i < static_cast<int>(orr.size()); ++i) {
        cmp(b_hist, binomial_z(hr[i], orr[i], ntot_r), 0.0,
            std::string(s.name) + " radius bin " + std::to_string(i));
      }
      for (int i = 0; i < kNMom && i < static_cast<int>(orp.size()); ++i) {
        cmp(b_hist, binomial_z(hp[i], orp[i], ntot_r), 0.0,
            std::string(s.name) + " momentum bin " + std::to_string(i));
      }

      auto moment = [&](const char* what, double s1, double s2, long long nn) {
        const double m = s1 / static_cast<double>(nn);
        const double v = s2 / static_cast<double>(nn) - m * m;
        const auto it = oracle_mom.find(std::string(s.name) + "/" + what);
        if (it == oracle_mom.end()) { return; }
        cmp(b_mom, mean_z(m, (v > 0.0) ? v : 0.0, nn, it->second.first, it->second.second,
                          oracle_n),
            0.0, std::string(s.name) + " " + what + " (port " + std::to_string(m) +
                     " oracle " + std::to_string(it->second.first) + ")");
      };
      moment("radius", sum_r, sum_r2, n_nucleons);
      moment("momentum", sum_p, sum_p2, n_nucleons);
      moment("outer_radius", sum_router, sum_router2, kN);
      moment("total_momentum", sum_psum, sum_psum2, kN);
      moment("min_pair_distance", sum_dmin, sum_dmin2, kN);
    }
  }

  // -------------------------------------------------------------------------------------------
  std::printf("\n%-34s %10s %14s  %s\n", "bucket", "points", "worst", "where");
  for (const Bucket& b : buckets) {
    const bool bad = b.worst > b.tol;
    if (bad) { ++fails; }
    std::printf("%-34s %10lld %14.4g  %s%s\n", b.name, b.n, b.worst,
                bad ? "FAIL " : "", b.where.c_str());
  }
  std::printf("\ntest_bic_nucleus: %s\n", (fails == 0) ? "PASS" : "FAIL");
  return (fails == 0) ? 0 : 1;
}
