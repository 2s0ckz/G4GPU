// The im_r_matrix collision tree's cross sections and angular distributions, against
// ref/oracle/bic_imr_*.csv.
//
// Everything here is EXACT, and not because the quantities are deterministic - two of the three
// are samplers - but because the samplers are driven by a PRESCRIBED uniform sequence on both
// sides. `ref/dump/dump_bic.cc` installs an eight-value cycle engine into CLHEP and the port
// uses the same eight in the same order, which turns `G4AngularDistributionNP::CosTheta` and
// `G4AngularDistribution::CosTheta` into functions of their arguments. That is a much stronger
// check than a histogram: a histogram of 20,000 angles would not see a table read one row off,
// and this does, at the first row it reaches.
//
//   bic_imr_xsec.csv     ten cross-section sources - the four G4CrossSectionPatch composites
//                        and every arm under them - on 240 values of sqrt(s) for each of the
//                        four nucleon pairs. 6,280 points. The grid includes both sides of both
//                        patch boundaries at one part in 1e9, because which arm answers at
//                        3 GeV and at 5 GeV is decided by a `<` against a `<=`.
//   bic_imr_angular_sweep  the same two tables driven by 199 values of the uniform at every one
//                        of their 39 and 40 tabulated energies and at the midpoint between each
//                        pair - 31,044 points, which is what makes the bisection walk the whole
//                        cumulative. The eight-phase file below reaches eight points of each row
//                        and was MEASURED not to notice one table entry moved by 1e-5.
//   bic_imr_angular.csv  G4AngularDistributionNP::CosTheta, G4AngularDistributionPP::CosTheta,
//                        G4AngularDistribution::CosTheta in both its symmetric and asymmetric
//                        forms, and Phi, at 25 energies x 4 mass pairs x 8 phases. 4,000 points,
//                        and the number of uniforms each call consumed is compared too - a
//                        sampler that draws the wrong number of randoms gives the right answer
//                        here and the wrong one in a cascade.
//   bic_imr_obe.csv      G4AngularDistribution::DifferentialCrossSection on a cos(theta) grid,
//                        so that a disagreement in the one-boson-exchange formula is separated
//                        from a disagreement in the twelve halvings that invert it. 4,200 points.
//   bic_imr_collision    G4CollisionNN::IsInCharge and its recast total, plus the two elastic
//                        channels' own IsInCharge and cross sections, over four nucleon pairs x
//                        two mass shells x three kinematic tilts x 157 energies.
//   bic_imr_elastic_fs   G4VElasticCollision::FinalState - both outgoing four-momenta and the
//                        number of uniforms consumed, same grid x 8 phases. The TILT is what
//                        makes the rotations in it observable at all: with both tracks along +z
//                        they are the identity (docs/RISK.md V75).
//   bic_imr_scatterer    G4Scatterer::GetTimeToInteraction and GetCrossSection over 24 impact
//                        parameters that straddle all three distance thresholds from both sides.
//                        The grid matters more than the energies: five of the six exits are
//                        decided by a distance, and the first version of it put the thresholds a
//                        factor of pi off - one millibarn is 0.1 fm^2 - after which four of six
//                        perturbations passed because nothing crossed a gate.
//   bic_imr_manager      G4CollisionManager replayed over a prescribed add/remove sequence with
//                        two exact ties, compared by WHICH collision won rather than by its time.
//   bic_imr_restab       the six resonance-production tables through their own CrossSectionTable(),
//                        on every one of the 121 tabulated energies and the midpoint between each
//                        pair. 12,100 points. Five of the six halve their cross section and the
//                        sixth does not: docs/RISK.md V94.
//   bic_imr_dbi          G4DetailedBalancePhaseSpaceIntegral for all 25 resonances, fine from 1 to
//                        4 GeV and coarse to 61 GeV - the coarse arm because the class searches
//                        `ie < 119` and above 48.109 GeV extrapolates rather than clamps, which a
//                        grid stopping at 4 GeV cannot see.
//   bic_imr_clebsch      G4Pow::logfactorial, G4Clebsch::TriangleCoeff, ClebschGordanCoeff,
//                        ClebschGordan and Weight over the whole (2J1, 2M1, 2J2, 2M2, 2J) box,
//                        with the M values two units past their J so the guards that reject them
//                        are exercised. 39,291 points.
//   bic_imr_meson        the meson-baryon ELASTIC channel - G4XAqmTotal, G4XAqmElastic,
//                        G4XMesonBaryonElastic and G4CollisionMesonBaryonElastic - over pion-
//                        nucleon AND pion-resonance pairs, plus a nucleon pair the parton-count
//                        test must reject. The cross section is zero for every pion-nucleon row:
//                        docs/RISK.md V107.
//   bic_imr_meson_fs     the same channel's FinalState, 8 phases per point. The only place in the
//                        cascade where the one-boson-exchange formula's asymmetric branch runs.
//   bic_imr_resxsec      G4XResonance::CrossSection through 29 concrete channels from all six
//                        families, built as their own constructors build them.
//   bic_imr_species      the isospin, spin, mass, width and IsShortLived flag of every species
//                        those families put in or out, from Geant4's own definitions - which is
//                        what makes the port's isospin list a checked table and not a copy.
//   bic_imr_nnbuffer     the 32 nodes of every resonance component's cross-section buffer, and
//                        the grid they sit on. Separate from the interpolated value because the
//                        interpolation amplifies a grid error by (y2-y1)/(x2-x1), which for the
//                        N-Delta channel above threshold is 0.18 mb per MeV.
//   bic_imr_nnpartial    G4CollisionNN's eight partial cross sections, the ones FinalState
//                        selects on, with the input four-momenta so nothing has to be rebuilt.
//   bic_imr_nnselect     which component one prescribed uniform lands in, 8 phases per energy.
//   bic_imr_resfs        G4VScatteringCollision::FinalState through twelve concrete channels from
//                        all six families - the two outgoing four-momenta with the resonance
//                        masses sampled from a Breit-Wigner, and the number of uniforms drawn.
//   bic_imr_annihfs      G4VAnnihilationCollision::FinalState, which draws none at all.
//   bic_imr_annih        G4XAnnihilationChannel over all 25 pion-nucleon resonance channels and
//                        two pion charges, with the two mass-dependent widths dumped beside the
//                        cross section so a disagreement says which of the three factors it is.
//   bic_imr_mbpartial    G4CollisionMesonBaryon's two partials and its buffered TOTAL, over five
//                        pion-nucleon pairs x 300 kinetic energies from 10 MeV to 3 GeV. The
//                        range runs to 3 GeV because the elastic partial is identically zero
//                        below 1865 MeV (docs/RISK.md V107) and a sweep that stopped at 1.5 GeV
//                        compared a column of zeros - MEASURED: buffering the elastic partial,
//                        which is wrong, passed 1,500 of 1,500 on the short range.
//   bic_imr_mbselect     which of the two components one prescribed uniform lands in, 8 phases
//                        per energy.
//   bic_imr_decaytable   every decay channel of all 94 species in the cascade's transitive
//                        closure, with G4SampleResonance::GetMinimumMass beside each. This is
//                        also the source tools/extract_bic_decay.pl generates the port's table
//                        from, so the comparison is of a TRANSFORM and not of a copy.
//   bic_imr_actualwidth  theActualWidth[] of every channel of every species with a decay table,
//                        at nine actual masses spread six widths either side of the pole.
//   bic_imr_k0flip       the K0 -> K0S/K0L coin flip in the G4KineticTrack constructor, 8 phases
//                        and both signs, with the number of uniforms it consumes.
//   bic_imr_lifetime     SampleResidualLifetime over 15 resonances x 3 masses x 3 momenta x 8
//                        phases. The momenta are what make the Lorentz dilation observable: at
//                        rest the factor is 1 and dropping it entirely would agree everywhere.
//   bic_imr_decayfs      G4KineticTrack::Decay over 28 resonances x 3 masses x 2 momenta x 8
//                        phases - every product's identity, four-momentum and position in the
//                        returned list, plus the number of uniforms the whole decay consumed.
//                        The list ORDER is compared because it is the order the cascade's track
//                        list takes and therefore the order of every later random: V151.
//   bic_imr_capture      G4BinaryCascade::Capture's decision over 4 nuclei x 8 synthetic
//                        secondary lists, with the per-entry nuclear field in
//                        bic_imr_capturefield.csv so the test drives the gate with the field
//                        Geant4 computed. Two of the lists are tuned to land BETWEEN 9 and 45
//                        MeV, which is the only window where the 0.2 factor decides anything.
//   bic_imr_absorbcut    G4Absorber::WillBeAbsorbed at the threshold Geant4 sets (zero, so
//                        nothing is ever absorbed) and at one that is not, so the predicate is
//                        shown to be live and the CONSTANT shown to be what kills it.
//   bic_imr_pauli        G4BinaryCascade::CheckPauliPrinciple over 5 nuclei x 12 radii x 2
//                        nucleons x 10 momenta. The function is private, so the dump
//                        re-expresses it from the three public ingredients the source names -
//                        G4FermiMomentum, the nuclear density and the Coulomb barrier - in the
//                        arithmetic the source writes. 366 of the 1,200 rows are blocked.
//   bic_imr_ionmass      G4IonTable::GetIonMass over the whole (Z, A) triangle a cascade can
//                        leave behind - 6,670 cells within 15 of the stable line up to A = 240.
//                        It is the first arm of G4BinaryCascade::GetIonMass; the other three are
//                        branches on (Z, A) and are asserted against the source's arithmetic.
//   bic_imr_scatter      G4Scatterer::Scatter end to end over 9 pairs x 9 energies from 50 MeV
//                        to 5 GeV x 8 phases - the product identities name which of the 306
//                        concrete channels the three nested uniforms landed on, and the draw
//                        count names how deep the path went.
//   bic_imr_scatterlife  four cross sections on one pair, before and after a G4Scatterer is
//                        destroyed. It is not a comparison - the port has no such static - it is
//                        the measurement behind docs/RISK.md V155.
//   bic_imr_kdecay       G4DecayKineticTracks at the LIST level, through the contract header
//                        bic/kinetic_decay.cuh, over 7 lists x 8 phases. What this file checks
//                        that no per-decay comparison can is the ORDER the survivors come back
//                        in, the RECURSION (a daughter that is itself short-lived), and the
//                        parent bookkeeping Geant4 stamps on every daughter.
//   bic_imr_absorb       G4MesonAbsorption's scheduling over 8 pairs x 7 energies x 61 impact
//                        parameters in 0.025 fm steps. The cross section is private, so the scan
//                        measures it through the verdict it decides - and the pair (proton, pi-)
//                        is in the list to reach the out-of-bounds table read of V153, with the
//                        candidate given a momentum so that its t is not zero for the honest
//                        reason as well as the buggy one.
//   bic_imr_absorbcluster  FindAndFillCluster over three five-candidate sets built so that the
//                        nearest candidate and the most-opposite one are different tracks.
//   bic_imr_absorbfs     GetFinalState over 9 (pion, nucleon, nucleon) combinations x 3 energies
//                        x 8 phases, with a tilted projectile so the rotation to z is not the
//                        identity, and the draw count because the charge fixing draws a SECOND
//                        uniform only when both nucleons sit on the same side of the pion.
//
// **Why the tolerance is 1e-15 and not zero.** The port and Geant4 evaluate the same expressions
// in the same order in double, so most of these agree bitwise; what they do not share is
// `G4Pow::powA`, whose port (src/data/g4pow.hh) computes its two tables with `std::log`/`std::exp`
// where Geant4 stores them, and `millibarn`, which both derive from `1e-28*m^2` but through
// different products. Both are last-bit effects. The buckets are set where the measurement puts
// them and the commit message records what moves when a term is perturbed.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "physics/hadronic/bic/im_r/angular.cuh"
#include "physics/hadronic/bic/im_r/channels.cuh"
#include "physics/hadronic/bic/im_r/clebsch.cuh"
#include "physics/hadronic/bic/im_r/collision_final_state.cuh"
#include "physics/hadronic/bic/im_r/collision_meson.cuh"
#include "physics/hadronic/bic/im_r/absorption.cuh"
#include "physics/hadronic/bic/im_r/decay.cuh"
#include "physics/hadronic/bic/cascade_capture.cuh"
#include "physics/hadronic/bic/cascade_collision.cuh"
#include "physics/hadronic/bic/cascade_state.cuh"
#include "physics/hadronic/bic/kinetic_decay.cuh"
#include "physics/hadronic/bic/im_r/collision_nn.cuh"
#include "physics/hadronic/bic/im_r/resonance_fs.cuh"
#include "physics/hadronic/bic/im_r/xsec_annihilation.cuh"
#include "physics/hadronic/bic/im_r/resonance_tables.cuh"
#include "physics/hadronic/bic/im_r/scatterer.cuh"
#include "physics/hadronic/bic/im_r/xsec_nn.cuh"

using namespace g4gpu;
namespace imr = g4gpu::bic::imr;

/// The device probe. The deliverables are `__host__ __device__` templates, and a template that
/// is only ever instantiated from the host is never compiled for the device at all - so this
/// kernel, which is never launched, is what proves the module is device code and what makes
/// `-Xptxas -v` report its register and stack cost.
struct CycleRngDev {
  int n = 0;
  int phase = 0;
  __host__ __device__ double uniform();
};

__global__ void bic_imr_probe(double* out, double s, double m1, double m2) {
  CycleRngDev rng;
  imr::AngularRefusal ref;
  imr::XsecRefusal xref;
  const imr::AngularObeConstants k = imr::angular_obe_constants();
  out[0] = imr::angular_np_cos_theta(s, m1, m2, rng, ref);
  out[1] = imr::angular_pp_cos_theta(s, m1, m2, rng, ref);
  out[2] = imr::angular_obe_cos_theta(k, true, s, m1, m2, rng, ref);
  out[3] = imr::x_nn_total(2212, 2212, m1, m2, std::sqrt(s), xref);
  out[4] = imr::x_nn_elastic(2212, 2212, m1, m2, std::sqrt(s), xref);
  out[5] = imr::x_np_elastic(2112, 2212, m1, m2, std::sqrt(s), xref);
}

/// The meson-baryon composite's own probe. It takes the buffer by pointer rather than building
/// one, because `build_meson_baryon_buffers` is the cascade's setup step and a 32 x 3 array of
/// doubles on a kernel stack would report a stack frame this module does not actually need at
/// collision time.
__global__ void bic_imr_meson_probe(double* out, const imr::MesonBaryonBuffers* buf, double s,
                                    double m_pion, double m_baryon) {
  CycleRngDev rng;
  imr::XsecRefusal xref;
  imr::MesonRefusal mref;
  const double e1 = (s - m_pion * m_pion - m_baryon * m_baryon) / (2.0 * m_baryon);
  const imr::LorentzVector p1(deex::Vec3d{0.0, 0.0, std::sqrt(e1 * e1 - m_pion * m_pion)}, e1);
  const imr::LorentzVector p2(deex::Vec3d{0.0, 0.0, 0.0}, m_baryon);
  double partial[imr::kMesonBaryonChannelCount];
  imr::meson_baryon_partials(*buf, 211, 2212, m_pion, m_baryon, p1, p2, m_pion, m_baryon,
                             partial, xref);
  out[0] = partial[0];
  out[1] = partial[1];
  out[2] = imr::meson_baryon_cross_section(211, 2212, std::sqrt(s), *buf, mref);
  out[3] = imr::meson_baryon_select(partial, rng);
}

/// The decay machinery's probe. `compute_min_mass` is deliberately NOT in it: it is a recursion,
/// the cascade never calls it - it reads the cached column, as Geant4 does after its first call -
/// and putting it here would report a stack frame the kernel does not pay for.
__global__ void bic_imr_decay_probe(double* out, int pdg, double actual_mass) {
  CycleRngDev rng;
  imr::DecayRefusal ref;
  double w[16];
  const int n = imr::kinetic_track_actual_widths(imr::kinetic_track_substitute_k0(pdg, rng),
                                                 actual_mass, w, 16, ref);
  const double total = imr::evaluate_total_actual_width(w, n);
  out[0] = total;
  out[1] = imr::sample_residual_lifetime(total, 1.5, rng);
  out[2] = imr::species_min_mass(pdg, ref);
  out[3] = static_cast<double>(n);
  const imr::LorentzVector parent(deex::Vec3d{0.0, 0.0, 700.0},
                                  std::sqrt(700.0 * 700.0 + actual_mass * actual_mass));
  const imr::DecayResult d = imr::kinetic_track_decay(pdg, actual_mass, parent, w, n, rng, ref);
  out[4] = static_cast<double>(d.n);
  out[5] = (d.n > 0) ? d.prod[0].p.e : 0.0;
  out[6] = (d.n > 0) ? d.prod[0].p.v.z : 0.0;
  out[7] = static_cast<double>(d.channel);
}

/// `G4Scatterer::Scatter` on the device: the whole dispatch, from two tracks to their products.
/// The channel table and the buffers come in by pointer because they are the caller's - which is
/// also why the port cannot have docs/RISK.md V155's lifetime bug.
__global__ void bic_imr_scatter_probe(double* out, const imr::ConcreteChannel* chans, int n_chan,
                                      const imr::NNChannelBuffers* buf, double t) {
  CycleRngDev rng;
  imr::ScatterRefusal ref;
  const double mp = 938.272013;
  const double e1 = t + mp;
  const imr::LorentzVector p1(deex::Vec3d{0.0, 0.0, std::sqrt(e1 * e1 - mp * mp)}, e1);
  const imr::LorentzVector p2(deex::Vec3d{0.0, 0.0, 0.0}, mp);
  const imr::ScatterFinalState fs = imr::nn_scatter_final_state(
      chans, n_chan, 2212, 2212, p1, p2, mp, mp, mp, mp, mp, 939.56536, 139.5701, *buf, rng, ref);
  out[0] = static_cast<double>(fs.n);
  out[1] = static_cast<double>(fs.component);
  out[2] = static_cast<double>(fs.concrete);
  out[3] = (fs.n > 0) ? fs.p[0].e : 0.0;
}

/// The CONTRACT header's own probe, so that the entry point an outside caller will use is proved
/// to be device code and measured on its own rather than through this package's internals.
__global__ void bic_kinetic_decay_probe(double* out, bic::DecayTrack* list, int n, int capacity) {
  CycleRngDev rng;
  bic::KineticDecayRefusal ref;
  int m = n;
  out[0] = static_cast<double>(bic::decay_kinetic_tracks(list, m, capacity, rng, ref));
  out[1] = (m > 0) ? list[0].momentum.e : 0.0;
  out[2] = static_cast<double>(ref.any() ? 1 : 0);
}

namespace {

int fails = 0;


// ---------------------------------------------------------------------------------------------
// Comparison bookkeeping - the same shape tests/test_bic_nucleus.cu uses
// ---------------------------------------------------------------------------------------------

struct Bucket {
  const char* name;
  long long n = 0;
  double worst = 0.0;
  std::string where;
  double tol = 1e-15;
};

std::vector<Bucket> buckets;

int new_bucket(const char* name, double tol) {
  Bucket b;
  b.name = name;
  b.tol = tol;
  buckets.push_back(b);
  return static_cast<int>(buckets.size()) - 1;
}

/// The plain relative `cmp` other tests in this package have is deliberately absent: every
/// quantity here crosses zero somewhere - a cross section below its threshold, a cosine at 90
/// degrees, the symmetric cumulative at cos = 0 where its two halves cancel exactly - and a
/// relative comparison at those points divides a rounding by a cancellation. Only the floored
/// form below is used.

/// `cmp` with an absolute floor, for a quantity that crosses zero. `cos(theta)` does, and so
/// does the symmetric OBE cumulative at cos(theta) = 0 where it is exactly 0.5 by construction
/// and its two halves cancel to a rounding.
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
  return (i < f.size()) ? std::strtod(f[i].c_str(), nullptr) : 0.0;
}
int iv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::atoi(f[i].c_str()) : 0;
}
std::string sv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? f[i] : std::string();
}

// ---------------------------------------------------------------------------------------------
// The prescribed uniform cycle, host side: the same eight values in the same order as
// ref/dump/dump_bic.cc's ImrCycleEngine, and a counter so the test can assert that the port
// drew as many uniforms as Geant4 did.
// ---------------------------------------------------------------------------------------------

struct CycleRng {
  int n = 0;
  int phase = 0;
  void reset(int p) { phase = p; n = 0; }
  // `__host__ __device__` although this one only ever runs on the host: the samplers it is
  // handed to are `__host__ __device__` templates, and nvcc warns 20011 about every
  // instantiation otherwise - a warning in every build that would hide a real one.
  __host__ __device__ double uniform() {
    // The eight are written inside the function rather than read from the namespace-scope
    // `kSeq`: a host `const double[]` referenced from `__host__ __device__` code is warning
    // 20014, and the point of the annotation was to have no warnings.
    const double seq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
    const double v = seq[(n + phase) % 8];
    ++n;
    return v;
  }
};

}  // namespace

/// The device probe's RNG, out of line so the struct above stays a plain aggregate.
__host__ __device__ double CycleRngDev::uniform() {
  const double seq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
  const double v = seq[(n + phase) % 8];
  ++n;
  return v;
}

int main() {
  const double mp = 938.27201300000002;  // G4Proton's PDG mass, as the oracle prints it
  const double mn = 939.56536000000002;  // G4Neutron's

  // -------------------------------------------------------------------------------------------
  // 1. The cross sections.
  // -------------------------------------------------------------------------------------------
  const int b_patch = new_bucket("XsecPatch (the four composites)", 1e-15);
  const int b_lowe = new_bucket("XsecLowE (the four tabulated arms)", 1e-15);
  const int b_pdg = new_bucket("XsecPDG (the two fits)", 2e-15);

  {
    const auto rows = read_csv("bic_imr_xsec.csv");
    for (const auto& r : rows) {
      const std::string pair = sv(r, 0);
      const std::string src = sv(r, 1);
      const double sqrt_s = dv(r, 2);
      const double want = dv(r, 3);  // millibarn
      int pdg1 = 0, pdg2 = 0;
      double m1 = 0.0, m2 = 0.0;
      if (pair == "pp") { pdg1 = pdg2 = 2212; m1 = m2 = mp; }
      else if (pair == "nn") { pdg1 = pdg2 = 2112; m1 = m2 = mn; }
      else if (pair == "np") { pdg1 = 2112; pdg2 = 2212; m1 = mn; m2 = mp; }
      else { pdg1 = 2212; pdg2 = 2112; m1 = mp; m2 = mn; }

      imr::XsecRefusal ref;
      double got = 0.0;
      int bucket = b_patch;
      if (src == "XNNTotal") { got = imr::x_nn_total(pdg1, pdg2, m1, m2, sqrt_s, ref); }
      else if (src == "XNNElastic") { got = imr::x_nn_elastic(pdg1, pdg2, m1, m2, sqrt_s, ref); }
      else if (src == "XnpElastic") { got = imr::x_np_elastic(pdg1, pdg2, m1, m2, sqrt_s, ref); }
      else if (src == "XnpTotal") { got = imr::x_np_total(pdg1, pdg2, m1, m2, sqrt_s, ref); }
      else if (src == "XNNTotalLowE") {
        got = imr::x_nn_total_lowe(pdg1, pdg2, sqrt_s, ref);
        bucket = b_lowe;
      } else if (src == "XNNElasticLowE") {
        got = imr::x_nn_elastic_lowe(pdg1, pdg2, sqrt_s, ref);
        bucket = b_lowe;
      } else if (src == "XnpElasticLowE") {
        got = imr::x_np_elastic_lowe(pdg1, pdg2, sqrt_s);
        bucket = b_lowe;
      } else if (src == "XnpTotalLowE") {
        got = imr::x_np_total_lowe(pdg1, pdg2, sqrt_s);
        bucket = b_lowe;
      } else if (src == "XPDGTotal") {
        got = imr::x_pdg_total(pdg1, pdg2, m1, m2, sqrt_s, ref);
        bucket = b_pdg;
      } else if (src == "XPDGElastic") {
        got = imr::x_pdg_elastic(pdg1, pdg2, m1, m2, sqrt_s, ref);
        bucket = b_pdg;
      } else {
        continue;
      }
      // Every one of these four pairs is one the arms answer for, so a refusal here is a
      // transcription error and not a species the port declines.
      if (ref.any()) {
        std::printf("REFUSED %s %s at sqrt(s)=%.6g\n", pair.c_str(), src.c_str(), sqrt_s);
        ++fails;
        continue;
      }
      cmp_scaled(bucket, got / imr::millibarn(), want, 1e-6,
                 pair + " " + src + " sqrt(s)=" + std::to_string(sqrt_s));
    }
  }

  // -------------------------------------------------------------------------------------------
  // 2. The angular distributions, under the prescribed cycle.
  // -------------------------------------------------------------------------------------------
  const int b_tbl = new_bucket("AngularTable (NP and PP CosTheta)", 1e-15);
  const int b_obe = new_bucket("AngularOBE (CosTheta, both sym)", 1e-15);
  const int b_phi = new_bucket("AngularPhi", 1e-15);
  const int b_draws = new_bucket("AngularDraws (uniforms consumed)", 0.0);

  {
    const imr::AngularObeConstants k = imr::angular_obe_constants();
    const auto rows = read_csv("bic_imr_angular.csv");
    for (const auto& r : rows) {
      const std::string dist = sv(r, 0);
      const double s = dv(r, 1);
      const double m1 = dv(r, 2);
      const double m2 = dv(r, 3);
      const int phase = iv(r, 4);
      const double want = dv(r, 5);
      const int want_draws = iv(r, 6);
      CycleRng rng;
      rng.reset(phase);
      imr::AngularRefusal ref;
      double got = 0.0;
      int bucket = b_tbl;
      const std::string where =
          dist + " s=" + std::to_string(s) + " m1=" + std::to_string(m1) +
          " m2=" + std::to_string(m2) + " phase=" + std::to_string(phase);
      if (dist == "NP") { got = imr::angular_np_cos_theta(s, m1, m2, rng, ref); }
      else if (dist == "PP") { got = imr::angular_pp_cos_theta(s, m1, m2, rng, ref); }
      else if (dist == "OBEsym") {
        got = imr::angular_obe_cos_theta(k, true, s, m1, m2, rng, ref);
        bucket = b_obe;
      } else if (dist == "OBEasym") {
        got = imr::angular_obe_cos_theta(k, false, s, m1, m2, rng, ref);
        bucket = b_obe;
      } else if (dist == "Phi") {
        got = imr::angular_phi(rng);
        bucket = b_phi;
      } else {
        continue;
      }
      if (ref.bisection_budget) {
        std::printf("REFUSED bisection budget: %s\n", where.c_str());
        ++fails;
      }
      cmp_scaled(bucket, got, want, 1e-3, where);
      cmp_int(b_draws, rng.n, want_draws, where);
    }
  }

  // -------------------------------------------------------------------------------------------
  // 2b. The table sweep - the same two samplers driven by 199 values of the uniform at each of
  //     the 39 and 40 tabulated energies and at the midpoint between each pair, which is what
  //     makes the bisection walk the WHOLE cumulative instead of eight points of it.
  //
  //     This block exists because the eight-phase block above was measured not to be enough:
  //     moving one entry of the 7,020-value NP table by one part in 65,000 changed none of its
  //     1,600 angles. It changes this block's.
  // -------------------------------------------------------------------------------------------
  const int b_sweep = new_bucket("AngularSweep (whole cumulative)", 1e-15);
  {
    const auto rows = read_csv("bic_imr_angular_sweep.csv");
    for (const auto& r : rows) {
      const std::string dist = sv(r, 0);
      const double s = dv(r, 1);
      const double m1 = dv(r, 2);
      const double m2 = dv(r, 3);
      const double sample = dv(r, 4);
      const double want = dv(r, 5);
      // One prescribed value, served to every draw - the port's mirror of ImrFixedEngine.
      struct FixedRng {
        double v;
        __host__ __device__ double uniform() { return v; }
      } rng{sample};
      imr::AngularRefusal ref;
      double got = 0.0;
      if (dist == "NP") { got = imr::angular_np_cos_theta(s, m1, m2, rng, ref); }
      else if (dist == "PP") { got = imr::angular_pp_cos_theta(s, m1, m2, rng, ref); }
      else { continue; }
      cmp_scaled(b_sweep, got, want, 1e-3,
                 dist + " s=" + std::to_string(s) + " sample=" + std::to_string(sample));
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3. The one-boson-exchange cumulative itself.
  // -------------------------------------------------------------------------------------------
  const int b_dsig = new_bucket("ObeDifferentialCrossSection", 1e-14);
  {
    const imr::AngularObeConstants k = imr::angular_obe_constants();
    const auto rows = read_csv("bic_imr_obe.csv");
    for (const auto& r : rows) {
      const bool sym = iv(r, 0) != 0;
      const double s = dv(r, 1);
      const double m1 = dv(r, 2);
      const double m2 = dv(r, 3);
      const double ct = dv(r, 4);
      const double want = dv(r, 5);
      const double got = imr::angular_obe_dsigma(k, sym, s, m1, m2, ct);
      cmp_scaled(b_dsig, got, want, 1e-6,
                 std::string(sym ? "sym" : "asym") + " s=" + std::to_string(s) +
                     " cos=" + std::to_string(ct));
    }
  }

  // -------------------------------------------------------------------------------------------
  // -------------------------------------------------------------------------------------------
  // 3b. The two elastic collision channels: who is in charge, the three cross sections a pair is
  //     asked for, and the two outgoing four-momenta. Half the rows have particle 1 thirty MeV
  //     off its mass shell downwards, which is what the nuclear potential does to a cascade
  //     nucleon - and three separate things in this path read a mass (the composite's recast, the
  //     angular distribution, the outgoing momenta), so an on-shell-only comparison would make
  //     all three agree and check none of them.
  //
  //     The INPUT four-momenta come from the oracle rather than being reconstructed from
  //     sqrt(s). They have to: in an off-shell row the energy is lowered at fixed three-momentum,
  //     so the invariant mass no longer determines the pair, and inverting it gives a different
  //     configuration with the same sqrt(s). The first version of this block did invert it, and
  //     it disagreed by 2.49 relative on the np total - which was the reconstruction and not the
  //     port.
  // -------------------------------------------------------------------------------------------
  const int b_incharge = new_bucket("CollisionIsInCharge", 0.0);
  const int b_csec = new_bucket("CollisionCrossSection", 1e-15);
  const int b_fsempty = new_bucket("ElasticFinalStateEmpty", 0.0);
  // Three buckets, by how much of the transform is non-trivial. With both tracks along +z the
  // two rotations are the identity and the boost is along one axis; tilting the projectile makes
  // both rotations real; giving the target a momentum of its own makes the boost general. The
  // three worst errors then say how many ulps each step of the chain costs, which is the only
  // way to tell accumulation from a transcription error - and inside a cascade every collision
  // is the third kind.
  const int b_fs0 = new_bucket("ElasticFinalState along +z", 1e-15);
  const int b_fs1 = new_bucket("ElasticFinalState tilted projectile", 3e-15);
  const int b_fs2 = new_bucket("ElasticFinalState general boost", 5e-15);
  const int b_fsdraws = new_bucket("ElasticFinalStateDraws", 0.0);

  // The pair the two collision files describe, rebuilt from the dumped four-momenta.
  struct InPair {
    int pdg1, pdg2;
    double m1, m2;          ///< the PDG masses
    imr::LorentzVector p1, p2;
    double actual1, actual2;  ///< G4KineticTrack::GetActualMass(), sqrt(|p.mag2()|)
  };
  auto make_pair = [&](const std::string& pair, const std::vector<std::string>& r) -> InPair {
    InPair q;
    if (pair == "pp") { q.pdg1 = q.pdg2 = 2212; q.m1 = q.m2 = mp; }
    else if (pair == "nn") { q.pdg1 = q.pdg2 = 2112; q.m1 = q.m2 = mn; }
    else if (pair == "np") { q.pdg1 = 2112; q.pdg2 = 2212; q.m1 = mn; q.m2 = mp; }
    else { q.pdg1 = 2212; q.pdg2 = 2112; q.m1 = mp; q.m2 = mn; }
    // Columns 4..7 are particle 1's four-momentum and 8..11 particle 2's, as the dump wrote
    // them - all three spatial components, because the tilted rows have all three.
    q.p1 = imr::LorentzVector(deex::Vec3d{dv(r, 4), dv(r, 5), dv(r, 6)}, dv(r, 7));
    q.p2 = imr::LorentzVector(deex::Vec3d{dv(r, 8), dv(r, 9), dv(r, 10)}, dv(r, 11));
    // `GetActualMass()` is `sqrt(|the4Momentum.mag2()|)` - the absolute value, so a spacelike
    // tracking momentum gives a positive mass. bic/kinetic_track.cuh says why.
    q.actual1 = std::sqrt(std::fabs(q.p1.e * q.p1.e - g4gpu::mag2(q.p1.v)));
    q.actual2 = std::sqrt(std::fabs(q.p2.e * q.p2.e - g4gpu::mag2(q.p2.v)));
    return q;
  };

  {
    const auto rows = read_csv("bic_imr_collision.csv");
    for (const auto& r : rows) {
      const std::string pair = sv(r, 0);
      const InPair q = make_pair(pair, r);
      const std::string where = pair + " off=" + sv(r, 1) + " tilt=" + sv(r, 2) +
                                " sqrt(s)=" + std::to_string(dv(r, 3));

      cmp_int(b_incharge, imr::collision_nn_is_in_charge(q.pdg1, q.pdg2) ? 1 : 0, iv(r, 12),
              where + " nn");
      cmp_int(b_incharge, imr::nn_elastic_is_in_charge(q.pdg1, q.pdg2) ? 1 : 0, iv(r, 13),
              where + " nnEl");
      cmp_int(b_incharge, imr::np_elastic_is_in_charge(q.pdg1, q.pdg2) ? 1 : 0, iv(r, 14),
              where + " npEl");

      imr::XsecRefusal xref;
      cmp_scaled(b_csec,
                 imr::collision_nn_cross_section(q.pdg1, q.pdg2, q.p1, q.p2, q.actual1,
                                                 q.actual2, q.m1, q.m2, xref) /
                     imr::millibarn(),
                 dv(r, 15), 1e-6, where + " total");
      if (imr::nn_elastic_is_in_charge(q.pdg1, q.pdg2)) {
        cmp_scaled(b_csec,
                   imr::nn_elastic_cross_section(q.pdg1, q.pdg2, q.p1, q.p2, q.m1, q.m2, xref) /
                       imr::millibarn(),
                   dv(r, 16), 1e-6, where + " nnEl");
      }
      if (imr::np_elastic_is_in_charge(q.pdg1, q.pdg2)) {
        cmp_scaled(b_csec,
                   imr::np_elastic_cross_section(q.pdg1, q.pdg2, q.p1, q.p2, q.m1, q.m2, xref) /
                       imr::millibarn(),
                   dv(r, 17), 1e-6, where + " npEl");
      }
      if (xref.any()) {
        std::printf("REFUSED cross section: %s\n", where.c_str());
        ++fails;
      }
    }
  }
  {
    const auto rows = read_csv("bic_imr_elastic_fs.csv");
    for (const auto& r : rows) {
      const std::string pair = sv(r, 0);
      const InPair q = make_pair(pair, r);
      const int phase = iv(r, 12);
      const int want_empty = iv(r, 13);
      CycleRng rng;
      rng.reset(phase);
      imr::AngularRefusal aref;
      const bool is_np = imr::np_elastic_is_in_charge(q.pdg1, q.pdg2);
      const imr::ElasticFinalState fs = imr::elastic_final_state(
          is_np ? imr::kAngularNp : imr::kAngularPp, q.p1, q.p2, q.actual1, q.actual2, q.m1,
          q.m2, rng, aref);
      const std::string where = pair + " off=" + sv(r, 1) + " tilt=" + sv(r, 2) + " sqrt(s)=" +
                                std::to_string(dv(r, 3)) + " phase=" + std::to_string(phase);
      cmp_int(b_fsempty, fs.empty ? 1 : 0, want_empty, where);
      // note: b_fs is chosen below, after the two scales are known.
      cmp_int(b_fsdraws, rng.n, iv(r, 22), where);
      if (want_empty != 0 || fs.empty) { continue; }
      // The scale a component is compared against is the MAGNITUDE of its own three-momentum,
      // not 1 MeV and not the component. A component is `p_cm * sin(theta) * cos(phi)` rotated
      // into the lab, so `|p|` is what it is built from and a component near zero is a
      // cancellation between terms of that size. The worst row in this file is a p2y of
      // -0.114 MeV beside a p2x of -285 MeV: against a 1 MeV floor it reads as 1.14e-13 and
      // against its own vector's 308 MeV as 3.7e-16, which is one and a half ulps and is what
      // it is. Measured, not assumed - the first version of this block used the 1 MeV floor and
      // reported exactly that row.
      const double s1 = std::sqrt(dv(r, 14) * dv(r, 14) + dv(r, 15) * dv(r, 15) +
                                  dv(r, 16) * dv(r, 16));
      const double s2 = std::sqrt(dv(r, 18) * dv(r, 18) + dv(r, 19) * dv(r, 19) +
                                  dv(r, 20) * dv(r, 20));
      const int tl = iv(r, 2);
      const int b_fs = (tl == 0) ? b_fs0 : ((tl == 1) ? b_fs1 : b_fs2);
      cmp_scaled(b_fs, fs.p1.v.x, dv(r, 14), s1, where + " p1x");
      cmp_scaled(b_fs, fs.p1.v.y, dv(r, 15), s1, where + " p1y");
      cmp_scaled(b_fs, fs.p1.v.z, dv(r, 16), s1, where + " p1z");
      cmp_scaled(b_fs, fs.p1.e, dv(r, 17), s1, where + " p1e");
      cmp_scaled(b_fs, fs.p2.v.x, dv(r, 18), s2, where + " p2x");
      cmp_scaled(b_fs, fs.p2.v.y, dv(r, 19), s2, where + " p2y");
      cmp_scaled(b_fs, fs.p2.v.z, dv(r, 20), s2, where + " p2z");
      cmp_scaled(b_fs, fs.p2.e, dv(r, 21), s2, where + " p2e");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3c. The scheduler: G4Scatterer::GetTimeToInteraction and GetCrossSection over a grid of
  //     impact parameters that brackets both distance thresholds, and G4CollisionManager's
  //     ordering over a prescribed sequence of adds and removals.
  //
  //     The impact parameters matter more than the energies here: five of this function's six
  //     exits are decided by a distance, and 0.795 fm and 1.262 fm - sqrt(200 mb/pi) and
  //     sqrt(500 mb/pi) - are on the grid from both sides.
  // -------------------------------------------------------------------------------------------
  const int b_ttime = new_bucket("ScattererTimeToInteraction", 1e-16);
  const int b_tverdict = new_bucket("ScattererCollisionOrNot", 0.0);
  const int b_tsigma = new_bucket("ScattererCrossSection", 1e-15);
  const int b_mgr = new_bucket("CollisionManagerOrder", 0.0);
  {
    // The meson-baryon composite's buffer, one per pair of DEFINITIONS, built up front because
    // a device kernel cannot build it inside the call the way G4CollisionComposite does.
    const double m_pip_s = 139.5701;
    struct MbPair { const char* name; int pion; int baryon; double m_baryon; int iso3_pion;
                    int iso3_baryon; };
    const MbPair kMb[] = {{"pip_p", 211, 2212, mp, 2, 1},
                          {"pim_p", -211, 2212, mp, -2, 1},
                          {"pip_n", 211, 2112, mn, 2, -1},
                          {"p_pip", 211, 2212, mp, 2, 1}};
    static imr::MesonBaryonBuffers sbuf[4];
    for (int k = 0; k < 4; ++k) {
      imr::AnnihRefusal aref;
      imr::build_meson_baryon_buffers(kMb[k].pion, kMb[k].baryon, m_pip_s, kMb[k].m_baryon,
                                      kMb[k].iso3_pion, kMb[k].iso3_baryon, m_pip_s, mp,
                                      sbuf[k], aref);
      if (aref.unknown_resonance) {
        std::printf("REFUSED scatterer meson buffer for %s\n", kMb[k].name);
        ++fails;
      }
    }
    const auto rows = read_csv("bic_imr_scatterer.csv");
    for (const auto& r : rows) {
      const std::string pair = sv(r, 0);
      int pdg1 = 0, pdg2 = 0, c1 = 0, c2 = 0;
      double m1 = 0.0, m2 = 0.0;
      const imr::MesonBaryonBuffers* mb = nullptr;
      if (pair == "pp") { pdg1 = pdg2 = 2212; m1 = m2 = mp; c1 = c2 = 1; }
      else if (pair == "nn") { pdg1 = pdg2 = 2112; m1 = m2 = mn; c1 = c2 = 0; }
      else if (pair == "np") { pdg1 = 2112; pdg2 = 2212; m1 = mn; m2 = mp; c1 = 0; c2 = 1; }
      else if (pair == "pn") { pdg1 = 2212; pdg2 = 2112; m1 = mp; m2 = mn; c1 = 1; c2 = 0; }
      else if (pair == "pip_p") {
        pdg1 = 211; pdg2 = 2212; m1 = m_pip_s; m2 = mp; c1 = 1; c2 = 1; mb = &sbuf[0];
      } else if (pair == "pim_p") {
        pdg1 = -211; pdg2 = 2212; m1 = m_pip_s; m2 = mp; c1 = -1; c2 = 1; mb = &sbuf[1];
      } else if (pair == "pip_n") {
        pdg1 = 211; pdg2 = 2112; m1 = m_pip_s; m2 = mn; c1 = 1; c2 = 0; mb = &sbuf[2];
      } else {  // p_pip: the same pair with the tracks the other way round
        pdg1 = 2212; pdg2 = 211; m1 = mp; m2 = m_pip_s; c1 = 1; c2 = 1; mb = &sbuf[3];
      }
      const double b_fm = dv(r, 3);
      const double dz_fm = dv(r, 4);
      const imr::LorentzVector p1v(deex::Vec3d{dv(r, 5), dv(r, 6), dv(r, 7)}, dv(r, 8));
      const imr::LorentzVector p2v(deex::Vec3d{dv(r, 9), dv(r, 10), dv(r, 11)}, dv(r, 12));
      const double want_time = dv(r, 13);   // -1 for no collision
      const double want_sigma = dv(r, 14);  // millibarn
      // CLHEP's fermi is 1e-12 mm, derived the way core/units.cuh derives its lengths.
      const double fm = 1.e-12;
      const deex::Vec3d x1{0.0, 0.0, 0.0};
      const deex::Vec3d x2{b_fm * fm, 0.0, dz_fm * fm};
      const double a1 = std::sqrt(std::fabs(p1v.e * p1v.e - g4gpu::mag2(p1v.v)));
      const double a2 = std::sqrt(std::fabs(p2v.e * p2v.e - g4gpu::mag2(p2v.v)));
      imr::ScatterRefusal sref;
      const imr::TimeToInteraction tt = imr::scatterer_time_to_interaction(
          pdg1, pdg2, c1, c2, x1, x2, p1v, p1v, p2v, a1, a2, m1, m2, sref, mb);
      const std::string where = pair + " alongz=" + sv(r, 1) + " T=" + sv(r, 2) +
                                " b=" + sv(r, 3) + " dz=" + sv(r, 4);
      const bool got_collision = (tt.time < DBL_MAX);
      cmp_int(b_tverdict, got_collision ? 1 : 0, (want_time >= 0.0) ? 1 : 0, where);
      if (want_time >= 0.0 && got_collision) {
        // A time in ns. The one at b = 0, dz = 2 fm is 3.28e-14 ns, so the comparison is
        // relative with a floor of one attosecond - below which the number is the rounding of
        // a 2 fm chord divided by c.
        cmp_scaled(b_ttime, tt.time, want_time, 1e-18, where);
      }
      cmp_scaled(b_tsigma,
                 imr::scatterer_cross_section(pdg1, pdg2, p1v, p2v, a1, a2, m1, m2, sref, mb) /
                     imr::millibarn(),
                 want_sigma, 1e-6, where);
      if (sref.any()) {
        std::printf("REFUSED scatterer: %s\n", where.c_str());
        ++fails;
      }
    }
  }
  {
    // The manager replayed against the same prescribed sequence. `next_index` in the oracle is
    // the earliest time in microseconds-as-an-integer, which is a stable way to name WHICH
    // collision won a tie without depending on a pointer.
    const auto rows = read_csv("bic_imr_manager.csv");
    imr::CollisionInitialState storage[16];
    imr::CollisionList list;
    list.items = storage;
    list.capacity = 16;
    const double times[8] = {2.0, 0.5, 1.7, 0.9, 0.5, 3.1, 0.9, 1.2};
    imr::ScatterRefusal sref;
    int add_i = 0;
    for (const auto& r : rows) {
      const std::string op = sv(r, 1);
      if (op == "add") {
        list.add(times[add_i], add_i, (add_i + 1) % 8, 0, sref);
        ++add_i;
      } else if (op == "remove_next") {
        list.remove(list.next_collision());
      } else if (op == "remove_tracks") {
        const int caned[2] = {3, 5};
        list.remove_tracks(caned, 2);
      } else if (op == "clear") {
        list.clear();
      }
      const int nxt = list.next_collision();
      const double nt = (nxt >= 0) ? list.items[nxt].collision_time : -1.0;
      cmp_int(b_mgr, list.size(), iv(r, 3), "step " + sv(r, 0) + " " + op + " size");
      // WHICH collision won, by its primary track index - not its time. See the note in
      // ref/dump/dump_bic.cc: with ties in the list the time is the same either way and the
      // tie-break is invisible.
      cmp_int(b_mgr, (nxt >= 0) ? list.items[nxt].primary : -1, iv(r, 4),
              "step " + sv(r, 0) + " " + op + " next");
      (void)nt;
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3d. The six resonance-production cross-section tables through their own accessor, and the
  //     detailed-balance phase-space integral for every resonance it knows.
  //
  //     The energy grid is every one of the 121 tabulated energies AND the midpoint between each
  //     pair: on a node the lookup returns the table entry and a wrong column would still be
  //     wrong, but between nodes the straight line is what catches a grid read one index off.
  // -------------------------------------------------------------------------------------------
  const int b_restab = new_bucket("ResonanceCrossSectionTables", 1e-15);
  const int b_dbi = new_bucket("DetailedBalancePhaseSpaceIntegral", 1e-15);
  {
    const auto rows = read_csv("bic_imr_restab.csv");
    for (const auto& r : rows) {
      const std::string tag = sv(r, 0);
      int which = -1;
      if (tag == "nd") { which = imr::kResNDelta; }
      else if (tag == "dd") { which = imr::kResDeltaDelta; }
      else if (tag == "ndstar") { which = imr::kResNDeltastar; }
      else if (tag == "ddstar") { which = imr::kResDeltaDeltastar; }
      else if (tag == "nnstar") { which = imr::kResNNstar; }
      else if (tag == "dnstar") { which = imr::kResDeltaNstar; }
      else { continue; }
      const int mass = iv(r, 1);
      const double sqrt_s = dv(r, 2);
      imr::ResonanceTableRefusal rref;
      const double got = imr::resonance_cross_section_table(which, mass, sqrt_s, rref);
      if (rref.no_column) {
        std::printf("REFUSED resonance column: %s %d\n", tag.c_str(), mass);
        ++fails;
        continue;
      }
      cmp_scaled(b_restab, got / imr::millibarn(), dv(r, 3), 1e-9,
                 tag + " " + std::to_string(mass) + " sqrt(s)=" + std::to_string(sqrt_s));
    }
  }
  {
    // "delta+" is the ground-state Delta(1232); only the excited states carry a mass in their
    // name. The port keys by nominal mass, so the map is from the Geant4 name to that.
    const auto rows = read_csv("bic_imr_dbi.csv");
    for (const auto& r : rows) {
      const std::string nm = sv(r, 0);
      int column = -1;
      if (nm == "delta+") {
        column = imr::kDbiDelta1232;
      } else {
        // "delta(1600)+" or "N(1440)+" - the four digits between the parentheses are the mass,
        // and the column is found by searching the port's own list in the class's own order.
        const std::size_t a = nm.find('(');
        const std::size_t b = nm.find(')');
        if (a == std::string::npos || b == std::string::npos) { continue; }
        const int mass = std::atoi(nm.substr(a + 1, b - a - 1).c_str());
        const bool is_delta = (nm[0] == 'd');
        // N(1700) and delta(1700) share a mass, and N(1900) and delta(1900) do too, so the
        // search has to know which half of the list to look in: columns 0-9 are the Deltas and
        // 10-24 the N*. Getting that wrong swaps two columns and nothing else, which is exactly
        // the kind of thing a grid of only distinct masses would not catch.
        const int lo = is_delta ? 0 : 10;
        const int hi = is_delta ? 10 : imr::kDbiColumns;
        for (int i = lo; i < hi; ++i) {
          if (imr::dbi_mass(i) == mass) { column = i; break; }
        }
      }
      if (column < 0) {
        std::printf("REFUSED detailed-balance column: %s\n", nm.c_str());
        ++fails;
        continue;
      }
      const double sqrt_s = dv(r, 1);
      cmp_scaled(b_dbi, imr::dbi_phase_space_integral(column, sqrt_s), dv(r, 2), 1e-9,
                 nm + " sqrt(s)=" + std::to_string(sqrt_s));
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3e. G4Clebsch - the isospin bookkeeping every resonance cross section is scaled by - and
  //     G4Pow's log-factorial table underneath it.
  //
  //     The M values in the sweep run two units PAST their J on purpose: the two guard `if`s at
  //     the top of ClebschGordanCoeff are what reject those, and a port that dropped either
  //     would return a number where Geant4 returns zero. 39,291 points.
  // -------------------------------------------------------------------------------------------
  const int b_logfact = new_bucket("LogFactorialTable", 1e-15);
  const int b_tri = new_bucket("ClebschTriangleCoeff", 1e-15);
  const int b_cgc = new_bucket("ClebschGordanCoeff", 1e-14);
  const int b_cg = new_bucket("ClebschGordan (the square)", 1e-14);
  const int b_wgt = new_bucket("ClebschWeight", 1e-14);
  long long clebsch_range_hits = 0;
  {
    const auto rows = read_csv("bic_imr_clebsch.csv");
    for (const auto& r : rows) {
      const std::string kind = sv(r, 0);
      const int j1 = iv(r, 1), m1 = iv(r, 2), j2 = iv(r, 3), m2 = iv(r, 4);
      const int j3 = iv(r, 5), j4 = iv(r, 6);
      const double want = dv(r, 7);
      imr::ClebschRefusal cref;
      const std::string where = kind + "(" + sv(r, 1) + "," + sv(r, 2) + "," + sv(r, 3) + "," +
                                sv(r, 4) + "," + sv(r, 5) + "," + sv(r, 6) + ")";
      if (kind == "logfact") {
        cmp_scaled(b_logfact, imr::log_factorial(j1), want, 1e-12, where);
      } else if (kind == "triangle") {
        cmp_scaled(b_tri, imr::clebsch_triangle_coeff(j1, j2, j3), want, 1e-12, where);
      } else if (kind == "coeff") {
        // The coefficient crosses zero and is of order 1, so the floor is 1e-12 - below the
        // smallest non-zero value any of these take and far above the rounding of a sum of
        // exponentials.
        cmp_scaled(b_cgc, imr::clebsch_gordan_coeff(j1, m1, j2, m2, j3, cref), want, 1e-12,
                   where);
      } else if (kind == "cg") {
        cmp_scaled(b_cg, imr::clebsch_gordan(j1, m1, j2, m2, j3, cref), want, 1e-12, where);
      } else if (kind == "weight") {
        cmp_scaled(b_wgt, imr::clebsch_weight(j1, m1, j2, m2, j3, j4, cref), want, 1e-12, where);
      }
      if (cref.coefficient_range) { ++clebsch_range_hits; }
    }
  }
  {
    const int b_guard = new_bucket("ClebschDeadGuards", 0.0);
    // ClebschGordanCoeff's three `JustWarning; return 0` exits - kMin < 0, kMax < kMin and
    // kMax >= 512 - are UNREACHABLE, and this is the assertion that says so. Geant4 agrees from
    // the other side: the same 18,225-point sweep raised zero G4Exceptions in the oracle run,
    // which was measured by installing a handler that counted them (it counted zero, and then
    // crashed the run for an unrelated reason - see ref/dump/dump_bic.cc).
    //
    // The reason is structural: kMin is max(0, -sum2, -sum4) and kMax is min(sum1, sum3, sum5),
    // and once the two M checks and the triangle inequality have passed, the Racah bounds
    // guarantee kMax >= kMin. kMin < 0 cannot happen at all - it starts at 0 and only grows. So
    // the guards cost three comparisons per call and catch nothing, and asserting ZERO hits is
    // what would break if the port's bounds arithmetic were wrong in either direction.
    cmp_int(b_guard, clebsch_range_hits, 0, "no ClebschGordanCoeff range guard is ever reached");

    // GenerateIso3's two answerable branches, and the refusal of the third.
    imr::ClebschRefusal cref2;
    const imr::Iso3Pair p = imr::clebsch_generate_iso3(1, 1, 1, -1, 3, 3, cref2);
    cmp_int(b_guard, cref2.generate_iso3 ? 1 : 0, 1, "GenerateIso3 refuses its sampling branch");
    cmp_int(b_guard, p.valid ? 1 : 0, 0, "and returns nothing");
    imr::ClebschRefusal cref3;
    const imr::Iso3Pair q = imr::clebsch_generate_iso3(1, 1, 1, -1, 0, 3, cref3);
    cmp_int(b_guard, (cref3.generate_iso3 || !q.valid) ? 1 : 0, 0,
            "but answers when an outgoing isospin is zero");
    cmp_int(b_guard, static_cast<long long>(q.m2), 0, "with m2 = twoM1 + twoM2");
  }

  // -------------------------------------------------------------------------------------------
  // 3f. The meson-baryon ELASTIC channel - the first one in this package a pion reaches.
  //     G4XAqmTotal, G4XAqmElastic, G4XMesonBaryonElastic, and G4CollisionMesonBaryonElastic's
  //     IsInCharge, CrossSection and FinalState.
  //
  //     The pairs include a pion on a Delta(1232) and on an N(1440), because the channel's
  //     IsInCharge is by PARTON COUNT and accepts them - and they turn out to be the only pairs
  //     for which the cross section is not identically zero. docs/RISK.md V107.
  // -------------------------------------------------------------------------------------------
  const int b_mincharge = new_bucket("MesonBaryonIsInCharge", 0.0);
  const int b_aqm = new_bucket("AqmCrossSections", 1e-15);
  const int b_mbx = new_bucket("MesonBaryonElasticCrossSection", 1e-15);
  const int b_mbzero = new_bucket("MesonBaryonElasticIsZeroBelow2GeVpLab", 0.0);
  const int b_mbfs = new_bucket("MesonBaryonElasticFinalState", 5e-15);
  const int b_mbdraws = new_bucket("MesonBaryonElasticDraws", 0.0);
  {
    const double m_pip = 139.5701;
    const auto rows = read_csv("bic_imr_meson.csv");
    for (const auto& r : rows) {
      const int pdg1 = iv(r, 1);
      const int pdg2 = iv(r, 2);
      const double m1 = dv(r, 3);
      const double m2 = dv(r, 4);
      const imr::LorentzVector p1v(deex::Vec3d{dv(r, 6), dv(r, 7), dv(r, 8)}, dv(r, 9));
      const imr::LorentzVector p2v(deex::Vec3d{dv(r, 10), dv(r, 11), dv(r, 12)}, dv(r, 13));
      const std::string where = sv(r, 0) + " sqrt(s)=" + std::to_string(dv(r, 5));
      imr::XsecRefusal xref;
      const bool in_charge = imr::meson_baryon_elastic_is_in_charge(pdg1, pdg2, xref);
      cmp_int(b_mincharge, in_charge ? 1 : 0, iv(r, 14), where);

      int nq1 = 0, ns1 = 0, nq2 = 0, ns2 = 0;
      imr::XsecRefusal pref;
      if (imr::parton_counts(pdg1, nq1, ns1, pref) &&
          imr::parton_counts(pdg2, nq2, ns2, pref)) {
        cmp_scaled(b_aqm, imr::x_aqm_total(nq1, ns1, nq2, ns2) / imr::millibarn(), dv(r, 15),
                   1e-9, where + " aqm total");
        bool exceeds = false;
        cmp_scaled(b_aqm, imr::x_aqm_elastic(nq1, ns1, nq2, ns2, exceeds), dv(r, 16), 1e-40,
                   where + " aqm elastic");
        // The `if (sigma > sigmaTot) throw` in G4XAqmElastic can never fire - the elastic form
        // raises an area in mm^2 to the power 1.5 and lands nine orders below the total.
        cmp_int(b_mbzero, exceeds ? 1 : 0, 0, where + " aqm elastic never exceeds the total");
      } else {
        std::printf("REFUSED parton counts: %s\n", where.c_str());
        ++fails;
      }
      if (!in_charge) { continue; }
      const double got = imr::meson_baryon_elastic_cross_section(pdg1, pdg2, m1, m2, p1v, p2v,
                                                                 m_pip, mp, xref);
      cmp_scaled(b_mbx, got / imr::millibarn(), dv(r, 17), 1e-9, where);
      // A pion on a NUCLEON is zero over the whole of QBBC's BIC window, and the assertion says
      // so from the port's side as well as the oracle's: pLab never reaches the 2 GeV the pi+p
      // PDG fit starts at. Only the two resonance pairs, whose larger sqrt(s) pushes the dummy
      // pLab past it, are ever non-zero.
      const bool baryon_is_nucleon = (pdg2 == 2212 || pdg2 == 2112 || pdg1 == 2212 ||
                                      pdg1 == 2112);
      if (baryon_is_nucleon) {
        cmp_int(b_mbzero, (got == 0.0) ? 1 : 0, 1, where + " pion-nucleon elastic is zero");
      }
    }
  }
  {
    const double m_pip = 139.5701;
    const auto rows = read_csv("bic_imr_meson_fs.csv");
    for (const auto& r : rows) {
      const imr::LorentzVector p1v(deex::Vec3d{dv(r, 2), dv(r, 3), dv(r, 4)}, dv(r, 5));
      const imr::LorentzVector p2v(deex::Vec3d{dv(r, 6), dv(r, 7), dv(r, 8)}, dv(r, 9));
      const int phase = iv(r, 10);
      const int want_empty = iv(r, 11);
      // The two PDG masses are recovered from the corresponding cross-section row's pair name;
      // here only the actual masses and the outgoing PDG masses are needed, and both come from
      // the four-momenta and the pair.
      const double a1 = std::sqrt(std::fabs(p1v.e * p1v.e - g4gpu::mag2(p1v.v)));
      const double a2 = std::sqrt(std::fabs(p2v.e * p2v.e - g4gpu::mag2(p2v.v)));
      // Every pair in this file is built on shell, so the actual masses ARE the PDG masses.
      CycleRng rng;
      rng.reset(phase);
      imr::AngularRefusal aref;
      const imr::ElasticFinalState fs =
          imr::meson_baryon_elastic_final_state(p1v, p2v, a1, a2, a1, a2, rng, aref);
      const std::string where =
          sv(r, 0) + " sqrt(s)=" + std::to_string(dv(r, 1)) + " phase=" + std::to_string(phase);
      cmp_int(b_mbdraws, fs.empty ? 1 : 0, want_empty, where + " empty");
      cmp_int(b_mbdraws, rng.n, iv(r, 20), where + " draws");
      if (want_empty != 0 || fs.empty) { continue; }
      const double s1 = std::sqrt(dv(r, 12) * dv(r, 12) + dv(r, 13) * dv(r, 13) +
                                  dv(r, 14) * dv(r, 14));
      const double s2 = std::sqrt(dv(r, 16) * dv(r, 16) + dv(r, 17) * dv(r, 17) +
                                  dv(r, 18) * dv(r, 18));
      cmp_scaled(b_mbfs, fs.p1.v.x, dv(r, 12), s1, where + " p1x");
      cmp_scaled(b_mbfs, fs.p1.v.y, dv(r, 13), s1, where + " p1y");
      cmp_scaled(b_mbfs, fs.p1.v.z, dv(r, 14), s1, where + " p1z");
      cmp_scaled(b_mbfs, fs.p1.e, dv(r, 15), s1, where + " p1e");
      cmp_scaled(b_mbfs, fs.p2.v.x, dv(r, 16), s2, where + " p2x");
      cmp_scaled(b_mbfs, fs.p2.v.y, dv(r, 17), s2, where + " p2y");
      cmp_scaled(b_mbfs, fs.p2.v.z, dv(r, 18), s2, where + " p2z");
      cmp_scaled(b_mbfs, fs.p2.e, dv(r, 19), s2, where + " p2e");
      (void)m_pip;
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3g. G4XResonance::CrossSection through the concrete channels that construct it, and the
  //     isospin quantum numbers it is scaled by.
  //
  //     `resonance_iso` is a list of PDG codes with an isospin attached, which is exactly the
  //     kind of copy that could be wrong in a way nothing notices - so it is checked against what
  //     Geant4's own particle definitions answer, species by species, and the `IsShortLived` flag
  //     is checked with it, because that flag is what decides whether the dead detailed-balance
  //     branch would run.
  // -------------------------------------------------------------------------------------------
  const int b_species = new_bucket("ResonanceSpeciesIsospin", 0.0);
  const int b_resx = new_bucket("XResonanceCrossSection", 1e-14);
  const int b_resinch = new_bucket("ConcreteChannelIsInCharge", 0.0);
  {
    const auto rows = read_csv("bic_imr_species.csv");
    for (const auto& r : rows) {
      const int pdg = iv(r, 0);
      if (sv(r, 1) == "MISSING") {
        std::printf("MISSING species in the oracle: %d\n", pdg);
        ++fails;
        continue;
      }
      imr::ResonanceTableRefusal rref;
      const int got = imr::resonance_iso(pdg, rref);
      if (rref.no_cross_section) {
        std::printf("REFUSED isospin for %d (%s)\n", pdg, sv(r, 1).c_str());
        ++fails;
        continue;
      }
      cmp_int(b_species, got, iv(r, 4), sv(r, 1) + " 2I");
      // Every one of these except the proton and the neutron is short-lived, and that is what
      // makes G4VXResonance::DetailedBalance dead: the only entrance pairs G4XResonance ever
      // sees are two NUCLEONS, and neither of those is short-lived.
      const bool expect_short = (pdg != 2212 && pdg != 2112);
      cmp_int(b_species, expect_short ? 1 : 0, iv(r, 7), sv(r, 1) + " IsShortLived");
    }
  }
  {
    const auto rows = read_csv("bic_imr_resxsec.csv");
    for (const auto& r : rows) {
      const std::string family = sv(r, 0);
      int which = -1;
      if (family == "nd") { which = imr::kResNDelta; }
      else if (family == "dd") { which = imr::kResDeltaDelta; }
      else if (family == "ndstar") { which = imr::kResNDeltastar; }
      else if (family == "ddstar") { which = imr::kResDeltaDeltastar; }
      else if (family == "nnstar") { which = imr::kResNNstar; }
      else if (family == "dnstar") { which = imr::kResDeltaNstar; }
      else { continue; }
      const int mass = iv(r, 1);
      const int in1 = iv(r, 2), in2 = iv(r, 3), out1 = iv(r, 4), out2 = iv(r, 5);
      const double sqrt_s = dv(r, 6);
      const std::string where = family + " " + std::to_string(mass) + " " + sv(r, 2) + "+" +
                                sv(r, 3) + "->" + sv(r, 4) + "+" + sv(r, 5) +
                                " sqrt(s)=" + std::to_string(sqrt_s);
      // Every entrance pair here is two nucleons, which is the only kind
      // G4ConcreteNNTwoBodyResonance::IsInCharge accepts.
      const bool in_charge = (in1 == 2212 || in1 == 2112) && (in2 == 2212 || in2 == 2112);
      cmp_int(b_resinch, in_charge ? 1 : 0, iv(r, 7), where);
      if (!in_charge) { continue; }
      imr::ResonanceTableRefusal rref;
      const int iso_out1 = imr::resonance_iso(out1, rref);
      const int iso_out2 = imr::resonance_iso(out2, rref);
      // 2I3 is +1 for a proton and -1 for a neutron; 2I is 1 for both.
      const int iso3_1 = (in1 == 2212) ? 1 : -1;
      const int iso3_2 = (in2 == 2212) ? 1 : -1;
      const double got = imr::x_resonance_cross_section(which, mass, 1, iso3_1, 1, iso3_2,
                                                        iso_out1, iso_out2, sqrt_s, rref);
      if (rref.no_column || rref.no_cross_section) {
        std::printf("REFUSED resonance cross section: %s\n", where.c_str());
        ++fails;
        continue;
      }
      cmp_scaled(b_resx, got / imr::millibarn(), dv(r, 8), 1e-9, where);
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3h. The eight partial cross sections G4CollisionComposite::FinalState selects on, and the
  //     selection. 306 concrete channels, nine middle-layer buffers twice over, and the 0.01 mb
  //     floor applied once per buffer.
  // -------------------------------------------------------------------------------------------
  const int b_chancount = new_bucket("ConcreteChannelEnumeration", 0.0);
  const int b_partial = new_bucket("NNPartialCrossSections", 1e-14);
  const int b_select = new_bucket("NNChannelSelection", 0.0);
  {
    static imr::ConcreteChannel chans[imr::kConcreteChannelCount];
    const int n_chan = imr::build_concrete_channels(chans, imr::kConcreteChannelCount);
    cmp_int(b_chancount, n_chan, imr::kConcreteChannelCount, "306 concrete channels");
    // Charge and baryon number balance on every one of them - the check
    // G4CollisionComposite::Resolve does at construction time and prints to G4cerr.
    imr::ResonanceTableRefusal cref;
    for (int i = 0; i < n_chan; ++i) {
      const imr::ConcreteChannel& c = chans[i];
      auto charge = [](int pdg) -> int {
        // The charge of a nucleon or a non-strange baryon resonance, from its PDG code: the
        // three quark digits, with u = +2/3 and d = -1/3, times three.
        int q = (pdg < 0) ? -pdg : pdg;
        q /= 10;
        const int d3 = q % 10, d2 = (q / 10) % 10, d1 = (q / 100) % 10;
        auto qc = [](int d) { return (d == 2) ? 2 : -1; };
        return (qc(d1) + qc(d2) + qc(d3)) / 3;
      };
      cmp_int(b_chancount, charge(c.in1) + charge(c.in2), charge(c.out1) + charge(c.out2),
              "channel " + std::to_string(i) + " charge balance");
      // Every entrance is a nucleon pair and every exit two baryons.
      cmp_int(b_chancount,
              ((c.in1 == 2212 || c.in1 == 2112) && (c.in2 == 2212 || c.in2 == 2112)) ? 1 : 0, 1,
              "channel " + std::to_string(i) + " entrance is two nucleons");
      imr::resonance_iso(c.out1, cref);
      imr::resonance_iso(c.out2, cref);
    }
    if (cref.no_cross_section) {
      std::printf("REFUSED isospin for an outgoing species in the channel list\n");
      ++fails;
    }

    // The buffer's own 32 nodes, before anything interpolates between them. The grid is checked
    // separately from the values because the interpolation amplifies an error in the grid by
    // `(y2-y1)/(x2-x1)`, which for the N-Delta channel just above threshold is 0.18 per MeV - so
    // one ulp of a 2.1 GeV node comes out as 4e-14 of the answer, and a test that compared only
    // the interpolated value could not tell that from a wrong cross section.
    const int b_grid = new_bucket("NNBufferGrid", 1e-15);
    const int b_node = new_bucket("NNBufferNodes", 1e-14);

    // The partials, per pair, with the buffers built once as Geant4 builds them once.
    static imr::NNChannelBuffers buffers[4];
    const auto rows = read_csv("bic_imr_nnpartial.csv");
    std::map<std::string, int> pair_index;
    pair_index["pp"] = 0; pair_index["nn"] = 1; pair_index["np"] = 2; pair_index["pn"] = 3;
    for (int k = 0; k < 4; ++k) {
      int pdg1 = 0, pdg2 = 0;
      double m1 = 0.0, m2 = 0.0;
      if (k == 0) { pdg1 = pdg2 = 2212; m1 = m2 = mp; }
      else if (k == 1) { pdg1 = pdg2 = 2112; m1 = m2 = mn; }
      else if (k == 2) { pdg1 = 2112; pdg2 = 2212; m1 = mn; m2 = mp; }
      else { pdg1 = 2212; pdg2 = 2112; m1 = mp; m2 = mn; }
      imr::ResonanceTableRefusal bref;
      imr::build_nn_channel_buffers(chans, n_chan, pdg1, pdg2, m1, m2, buffers[k], bref);
      if (bref.no_column || bref.no_cross_section) {
        std::printf("REFUSED while building the buffers for pair %d\n", k);
        ++fails;
      }
    }
    {
      const auto brows = read_csv("bic_imr_nnbuffer.csv");
      for (const auto& r : brows) {
        const auto it = pair_index.find(sv(r, 0));
        if (it == pair_index.end()) { continue; }
        const int k = it->second;
        const int point = iv(r, 1);
        const int comp = iv(r, 4);
        const std::string where = sv(r, 0) + " node " + sv(r, 1) + " comp " + sv(r, 4);
        cmp_scaled(b_grid, buffers[k].grid[point], dv(r, 3), 1.0, where + " grid");
        if (comp >= imr::kNNToNDelta) {
          // The oracle's value at a node is what `comps[i]->CrossSection` returns there, which
          // for these six is the BUFFERED value - so it carries the 0.01 mb floor and, at the
          // LAST node, the fall-through that makes the buffer return zero above its own grid.
          // Comparing the raw node sum against it would be comparing two different things: the
          // first version of this block did, and reported 2.0e9 at node 31 where Geant4 returns
          // exactly zero and the sum is 2.018 mb.
          const double got = imr::buffered_cross_section(
              buffers[k].grid, buffers[k].top[comp], imr::kBufferPoints, buffers[k].grid[point]);
          cmp_scaled(b_node, got / imr::millibarn(), dv(r, 5), 1e-9, where + " node");
        }
      }
    }
    for (const auto& r : rows) {
      const std::string pair = sv(r, 0);
      const auto it = pair_index.find(pair);
      if (it == pair_index.end()) { continue; }
      const int k = it->second;
      int pdg1 = 0, pdg2 = 0;
      double m1 = 0.0, m2 = 0.0;
      if (k == 0) { pdg1 = pdg2 = 2212; m1 = m2 = mp; }
      else if (k == 1) { pdg1 = pdg2 = 2112; m1 = m2 = mn; }
      else if (k == 2) { pdg1 = 2112; pdg2 = 2212; m1 = mn; m2 = mp; }
      else { pdg1 = 2212; pdg2 = 2112; m1 = mp; m2 = mn; }
      const double sqrt_s = dv(r, 1);
      const int comp = iv(r, 5);
      // The pair comes from the oracle, not from inverting sqrt(s): see the note in
      // ref/dump/dump_bic.cc. The buffer's slope for the N-Delta channel just above threshold is
      // 0.18 mb per MeV, so an ulp of reconstruction is 4e-14 of the answer - which is exactly
      // what the first version of this block reported.
      const imr::LorentzVector p1v(deex::Vec3d{0.0, 0.0, dv(r, 2)}, dv(r, 3));
      const imr::LorentzVector p2v(deex::Vec3d{0.0, 0.0, 0.0}, dv(r, 4));
      double partial[imr::kNNChannelCount];
      imr::XsecRefusal xref;
      imr::nn_partial_cross_sections(pdg1, pdg2, p1v, p2v, m1, m2, buffers[k], partial, xref);
      cmp_scaled(b_partial, partial[comp] / imr::millibarn(), dv(r, 6), 1e-9,
                 pair + " comp " + std::to_string(comp) + " sqrt(s)=" + std::to_string(sqrt_s));
    }
    {
      const auto srows = read_csv("bic_imr_nnselect.csv");
      for (const auto& r : srows) {
        const std::string pair = sv(r, 0);
        const auto it = pair_index.find(pair);
        if (it == pair_index.end()) { continue; }
        const int k = it->second;
        int pdg1 = 0, pdg2 = 0;
        double m1 = 0.0, m2 = 0.0;
        if (k == 0) { pdg1 = pdg2 = 2212; m1 = m2 = mp; }
        else if (k == 1) { pdg1 = pdg2 = 2112; m1 = m2 = mn; }
        else if (k == 2) { pdg1 = 2112; pdg2 = 2212; m1 = mn; m2 = mp; }
        else { pdg1 = 2212; pdg2 = 2112; m1 = mp; m2 = mn; }
        const double sqrt_s = dv(r, 1);
        const imr::LorentzVector p1v(deex::Vec3d{0.0, 0.0, dv(r, 2)}, dv(r, 3));
        const imr::LorentzVector p2v(deex::Vec3d{0.0, 0.0, 0.0}, dv(r, 4));
        double partial[imr::kNNChannelCount];
        imr::XsecRefusal xref;
        imr::nn_partial_cross_sections(pdg1, pdg2, p1v, p2v, m1, m2, buffers[k], partial, xref);
        CycleRng rng;
        rng.reset(iv(r, 5));
        const int got = imr::nn_select_channel(partial, rng);
        cmp_int(b_select, got, iv(r, 6),
                pair + " sqrt(s)=" + std::to_string(sqrt_s) + " phase=" + sv(r, 5));
        cmp_int(b_select, rng.n, iv(r, 7), pair + " draws");
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3i. What comes OUT of a resonance channel: G4VScatteringCollision::FinalState with the
  //     resonance masses sampled from a Breit-Wigner, and G4VAnnihilationCollision::FinalState.
  //
  //     The number of uniforms each call draws is compared too, and it varies - a stable product
  //     draws none for its mass, a short-lived one draws exactly one - so a port that resampled
  //     a nucleon would agree on nothing after the first collision of an event.
  // -------------------------------------------------------------------------------------------
  const int b_resspec = new_bucket("ResonanceMassAndWidth", 0.0);
  const int b_resfs = new_bucket("ScatteringFinalState", 5e-15);
  const int b_resfsd = new_bucket("ScatteringFinalStateDraws", 0.0);
  const int b_annih = new_bucket("AnnihilationFinalState", 1e-15);
  {
    // The port's mass and width table against Geant4's own definitions, species by species.
    const auto srows = read_csv("bic_imr_species.csv");
    for (const auto& r : srows) {
      const int pdg = iv(r, 0);
      if (sv(r, 1) == "MISSING") { continue; }
      const imr::SpeciesProperties s = imr::species_properties(pdg, mp, mn);
      if (!s.known) {
        std::printf("REFUSED species properties for %d (%s)\n", pdg, sv(r, 1).c_str());
        ++fails;
        continue;
      }
      cmp_int(b_resspec, (std::fabs(s.mass - dv(r, 2)) < 1e-9) ? 1 : 0, 1, sv(r, 1) + " mass");
      cmp_int(b_resspec, (std::fabs(s.width - dv(r, 3)) < 1e-9) ? 1 : 0, 1, sv(r, 1) + " width");
      cmp_int(b_resspec, s.short_lived ? 1 : 0, iv(r, 7), sv(r, 1) + " IsShortLived");
      cmp_int(b_resspec, s.two_spin, iv(r, 6), sv(r, 1) + " 2J from the PDG code's last digit");
    }
  }
  {
    const auto rows = read_csv("bic_imr_resfs.csv");
    const double m_pip = 139.5701;
    for (const auto& r : rows) {
      const int in1 = iv(r, 1), in2 = iv(r, 2), out1 = iv(r, 3), out2 = iv(r, 4);
      const double m1 = (in1 == 2212) ? mp : mn;
      const double m2 = (in2 == 2212) ? mp : mn;
      const imr::LorentzVector p1v(deex::Vec3d{0.0, 0.0, dv(r, 6)}, dv(r, 7));
      const imr::LorentzVector p2v(deex::Vec3d{0.0, 0.0, 0.0}, dv(r, 8));
      const int phase = iv(r, 9);
      const int nprod = iv(r, 10);
      const imr::SpeciesProperties s1 = imr::species_properties(out1, mp, mn);
      const imr::SpeciesProperties s2 = imr::species_properties(out2, mp, mn);
      CycleRng rng;
      rng.reset(phase);
      imr::ResonanceFsRefusal fref;
      const imr::ElasticFinalState fs = imr::scattering_final_state(
          p1v, p2v, m1, m2, s1, s2, mn, m_pip, rng, fref);
      const std::string where = sv(r, 0) + " " + sv(r, 3) + "+" + sv(r, 4) + " sqrt(s)=" +
                                std::to_string(dv(r, 5)) + " phase=" + std::to_string(phase);
      cmp_int(b_resfsd, fs.empty ? 0 : 2, nprod, where + " product count");
      cmp_int(b_resfsd, rng.n, iv(r, 19), where + " draws");
      // Two branches of SampleResonanceMass are unreachable for the species this cascade
      // produces, and the assertions say so rather than leaving them untested. The narrowest
      // short-lived width here is 100 MeV, nine orders above the `gamma < 1e-10*GeV` shortcut;
      // and the mass window never closes, which Geant4 agrees with - it prints
      // "SampleResonanceMass: particle out of mass range" when it does, and the oracle run
      // printed it zero times over these 3,744 calls.
      cmp_int(b_resfsd, fref.mass_window_empty ? 1 : 0, 0, where + " mass window never closes");
      cmp_int(b_resfsd, fref.mass_window_zeroed ? 1 : 0, 0, where + " and is never zeroed");
      if (nprod < 2 || fs.empty) { continue; }
      const double sc1 = std::sqrt(dv(r, 11) * dv(r, 11) + dv(r, 12) * dv(r, 12) +
                                   dv(r, 13) * dv(r, 13));
      const double sc2 = std::sqrt(dv(r, 15) * dv(r, 15) + dv(r, 16) * dv(r, 16) +
                                   dv(r, 17) * dv(r, 17));
      cmp_scaled(b_resfs, fs.p1.v.x, dv(r, 11), sc1, where + " p1x");
      cmp_scaled(b_resfs, fs.p1.v.y, dv(r, 12), sc1, where + " p1y");
      cmp_scaled(b_resfs, fs.p1.v.z, dv(r, 13), sc1, where + " p1z");
      cmp_scaled(b_resfs, fs.p1.e, dv(r, 14), sc1, where + " p1e");
      cmp_scaled(b_resfs, fs.p2.v.x, dv(r, 15), sc2, where + " p2x");
      cmp_scaled(b_resfs, fs.p2.v.y, dv(r, 16), sc2, where + " p2y");
      cmp_scaled(b_resfs, fs.p2.v.z, dv(r, 17), sc2, where + " p2z");
      cmp_scaled(b_resfs, fs.p2.e, dv(r, 18), sc2, where + " p2e");
    }
  }
  {
    const auto rows = read_csv("bic_imr_annihfs.csv");
    for (const auto& r : rows) {
      const imr::LorentzVector p1v(deex::Vec3d{0.0, 0.0, dv(r, 2)}, dv(r, 3));
      const imr::LorentzVector p2v(deex::Vec3d{0.0, 0.0, 0.0}, dv(r, 4));
      const imr::LorentzVector got = imr::annihilation_final_state(p1v, p2v);
      const std::string where = sv(r, 0) + " sqrt(s)=" + std::to_string(dv(r, 1));
      const double sc = std::sqrt(dv(r, 5) * dv(r, 5) + dv(r, 6) * dv(r, 6) +
                                  dv(r, 7) * dv(r, 7));
      cmp_scaled(b_annih, got.v.x, dv(r, 5), sc, where + " px");
      cmp_scaled(b_annih, got.v.y, dv(r, 6), sc, where + " py");
      cmp_scaled(b_annih, got.v.z, dv(r, 7), sc, where + " pz");
      cmp_scaled(b_annih, got.e, dv(r, 8), sc, where + " e");
      // It draws no uniform at all.
      cmp_int(b_annih, 0, iv(r, 9), where + " draws");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3j. G4XAnnihilationChannel - the pion-nucleon resonance production cross section, which is
  //     the only thing a pion in the binary cascade can do (docs/RISK.md V107). All 25 channels
  //     G4CollisionMesonBaryonToResonance builds, over 77 energies each.
  //
  //     The two mass-dependent widths are compared separately from the cross section, because
  //     two of the 25 have a BROKEN map key and fall back to a constant, and the only way to see
  //     that in the product would be to notice it is the wrong shape.
  // -------------------------------------------------------------------------------------------
  const int b_annw = new_bucket("AnnihilationWidths", 1e-14);
  const int b_annx = new_bucket("AnnihilationCrossSection", 1e-13);
  const int b_annk = new_bucket("AnnihilationBrokenKeys", 0.0);
  {
    const auto rows = read_csv("bic_imr_annih.csv");
    long long n1700_distinct = 0;
    long long n2250_distinct = 0;
    double n1700_first = -1.0;
    double n2250_first = -1.0;
    for (const auto& r : rows) {
      const std::string label = sv(r, 0);
      if (label.size() < 5) { continue; }
      const bool is_delta = (label[0] == 'D');
      const int mass = std::atoi(label.substr(1, 4).c_str());
      const int res_pdg = iv(r, 1);
      const int pion_pdg = iv(r, 2);
      const double sqrt_s = dv(r, 3);
      const imr::LorentzVector p1v(deex::Vec3d{0.0, 0.0, dv(r, 4)}, dv(r, 5));
      const imr::LorentzVector p2v(deex::Vec3d{0.0, 0.0, 0.0}, dv(r, 6));
      const std::string where = label + " sqrt(s)=" + std::to_string(sqrt_s);
      const imr::SpeciesProperties res = imr::species_properties(res_pdg, mp, mn);
      if (!res.known) {
        std::printf("REFUSED resonance properties for %d (%s)\n", res_pdg, label.c_str());
        ++fails;
        continue;
      }
      // The two widths.
      const double got_w =
          imr::annih_variable_width(mass, is_delta, res.width, sqrt_s);
      const double got_pw =
          imr::annih_variable_partial_width(mass, is_delta, res.width, sqrt_s);
      cmp_scaled(b_annw, got_w, dv(r, 9), 1e-9, where + " total width");
      cmp_scaled(b_annw, got_pw, dv(r, 10), 1e-9, where + " partial width");
      if (label == "N1700_Npi") {
        if (n1700_first < 0.0) { n1700_first = dv(r, 10); }
        if (dv(r, 10) != n1700_first) { ++n1700_distinct; }
      }
      if (label == "N2250_Npi") {
        if (n2250_first < 0.0) { n2250_first = dv(r, 9); }
        if (dv(r, 9) != n2250_first) { ++n2250_distinct; }
      }
      // The cross section.
      imr::AnnihRefusal aref;
      const int iso31 = (pion_pdg == 211) ? 2 : ((pion_pdg == -211) ? -2 : 0);
      // The pion MASS comes from its own dumped four-momentum, not from a constant: a pi0 is
      // 134.977 MeV where a pi+ is 139.570, and p_CM squared is in the denominator. Passing the
      // charged mass for every row put the pi0 cross sections 44% out.
      const double m_pion = std::sqrt(dv(r, 5) * dv(r, 5) - dv(r, 4) * dv(r, 4));
      const double got = imr::x_annihilation_channel(
          0, m_pion, 2, iso31,         // the pion: spin 0, isospin 1
          1, mp, 1, 1,                 // the proton: 2J = 1, 2I = 1, 2I3 = +1
          mass, is_delta, res.two_spin, res.mass, res.width,
          is_delta ? 3 : 1, sqrt_s, aref);
      // The cross section is a product of a Breit-Wigner, a branching ratio and a Clebsch-Gordan
      // ratio, each of which is itself a quotient, so the bucket is a little looser than the
      // widths it is built from - and it crosses zero where the isospin forbids the channel.
      cmp_scaled(b_annx, got / imr::millibarn(), dv(r, 8), 1e-9, where);
      if (aref.unknown_resonance) {
        std::printf("REFUSED annihilation cross section: %s\n", where.c_str());
        ++fails;
      }
    }
    // The two broken keys, asserted from the oracle's own data: a working channel's width varies
    // with sqrt(s) over these 77 points and these two do not, because their map key is missing.
    cmp_int(b_annk, n1700_distinct, 0, "N(1700)'s partial width is constant - the key is gone");
    cmp_int(b_annk, n2250_distinct, 0, "N(2250)'s total width is constant - no key at all");
    cmp_int(b_annk, (n1700_first == 150.0) ? 1 : 0, 1, "and it is the PDG width, 150 MeV");
    cmp_int(b_annk, (n2250_first == 500.0) ? 1 : 0, 1, "and it is the PDG width, 500 MeV");
    cmp_int(b_annk, imr::partial_width_column(1700, false), -1, "no N1700_Npi column");
    cmp_int(b_annk, imr::total_width_column(2250, false), -1, "no N(2250) column");
    cmp_int(b_annk, (imr::partial_width_column(1700, true) >= 0) ? 1 : 0, 1,
            "but D1700_Npi is there");
    cmp_int(b_annk, (imr::total_width_column(2220, false) >= 0) ? 1 : 0, 1,
            "and N(2220) is there");
  }

  // -------------------------------------------------------------------------------------------
  // 3k. G4CollisionMesonBaryon's two partials and the selection between them - the whole of what
  //     a pion does in the cascade once G4Scatterer has found the channel.
  // -------------------------------------------------------------------------------------------
  const int b_mbpart = new_bucket("MesonBaryonPartials", 1e-13);
  const int b_mbtot = new_bucket("MesonBaryonBufferedTotal", 1e-13);
  const int b_mbsel = new_bucket("MesonBaryonSelection", 0.0);
  {
    const double m_pip = 139.5701;
    struct MPair { const char* name; int pion; int baryon; };
    // `pip_dpp` is a pion on a Delta++: in charge of the ELASTIC component by parton count and
    // not of the to-resonance one, whose IsInCharge compares G4ParticleTypeConverter generic
    // types and sees D1232 rather than NUCLEON. Its component 0 is identically zero in all 300
    // oracle rows, which is what makes the gate in `build_meson_baryon_buffers` observable.
    const MPair kPairs[] = {{"pip_p", 211, 2212},  {"pim_p", -211, 2212},
                            {"pi0_p", 111, 2212},  {"pip_n", 211, 2112},
                            {"pim_n", -211, 2112}, {"pip_dpp", 211, 2224}};
    static imr::MesonBaryonBuffers mbuf[6];
    std::map<std::string, int> idx;
    for (int k = 0; k < 6; ++k) {
      idx[kPairs[k].name] = k;
      const double m_pion = (kPairs[k].pion == 111) ? 134.9766 : m_pip;
      const imr::SpeciesProperties bar = imr::species_properties(kPairs[k].baryon, mp, mn);
      if (!bar.known) {
        std::printf("REFUSED baryon properties for %d\n", kPairs[k].baryon);
        ++fails;
      }
      const double m_bar = bar.mass;
      const int iso3_pion = (kPairs[k].pion == 211) ? 2 : ((kPairs[k].pion == -211) ? -2 : 0);
      const int iso3_bar = (kPairs[k].baryon == 2212) ? 1 : -1;
      imr::AnnihRefusal aref;
      imr::build_meson_baryon_buffers(kPairs[k].pion, kPairs[k].baryon, m_pion, m_bar, iso3_pion,
                                      iso3_bar, m_pip, mp, mbuf[k], aref);
      if (aref.unknown_resonance) {
        std::printf("REFUSED while building meson-baryon buffers for %s\n", kPairs[k].name);
        ++fails;
      }
    }
    // The tracks come from the CSV, not from sqrt(s): the elastic partial is evaluated on the
    // tracks as given and reconstructing them by inverting the invariant mass loses digits.
    auto tracks = [&](const std::vector<std::string>& r, imr::LorentzVector& p1,
                      imr::LorentzVector& p2) {
      p1 = imr::LorentzVector(deex::Vec3d{0.0, 0.0, dv(r, 2)}, dv(r, 3));
      p2 = imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0}, dv(r, 4));
    };
    const auto rows = read_csv("bic_imr_mbpartial.csv");
    for (const auto& r : rows) {
      const auto it = idx.find(sv(r, 0));
      if (it == idx.end()) { continue; }
      const int k = it->second;
      const double m_pion = (kPairs[k].pion == 111) ? 134.9766 : m_pip;
      const double m_bar = imr::species_properties(kPairs[k].baryon, mp, mn).mass;
      imr::LorentzVector p1, p2;
      tracks(r, p1, p2);
      const int comp = iv(r, 5);
      double partial[imr::kMesonBaryonChannelCount];
      imr::XsecRefusal xref;
      imr::meson_baryon_partials(mbuf[k], kPairs[k].pion, kPairs[k].baryon, m_pion, m_bar, p1, p2,
                                 m_pip, mp, partial, xref);
      cmp_scaled(b_mbpart, partial[comp] / imr::millibarn(), dv(r, 6), 1e-9,
                 sv(r, 0) + " comp " + sv(r, 5) + " sqrt(s)=" + sv(r, 1));
      imr::MesonRefusal mref;
      const double tot = imr::meson_baryon_cross_section(kPairs[k].pion, kPairs[k].baryon,
                                                         (p1 + p2).mag(), mbuf[k], mref);
      if (mref.any()) {
        std::printf("REFUSED meson-baryon total for %s\n", sv(r, 0).c_str());
        ++fails;
      }
      cmp_scaled(b_mbtot, tot / imr::millibarn(), dv(r, 7), 1e-9,
                 sv(r, 0) + " total sqrt(s)=" + sv(r, 1));
    }
    // The two-level nesting is invisible in the numbers above and this is why. MEASURED with a
    // probe on pi+ p: the child buffer reproduces its own node exactly at 31 of the 32 nodes,
    // and the one it does not is the LAST - `G4CrossSectionBuffer::CrossSection`'s search never
    // finds a grid point above sqrt(s) there, so x1,y1 keep their initialisers 1 and 0 and the
    // 0.01 mb floor on y1 forces the result to zero. So the composite's 32nd node is ELASTIC
    // ONLY, for every pair, whatever the resonance sum is there. For pi+ p that sum is 7.946e-3
    // mb at sqrt(s) = 13.74 GeV, below the floor in its own right; the point is that it would be
    // discarded at any size. Nothing in the 10-3000 MeV sweep reaches that node, which is why
    // rebuilding the parent from the raw child sum instead of the buffered one passed 3,000 of
    // 3,000 - so the nesting is asserted here rather than inferred from a comparison.
    for (int k = 0; k < 5; ++k) {
      const double top = imr::buffered_cross_section(mbuf[k].grid, mbuf[k].to_resonance,
                                                     imr::kBufferPoints,
                                                     mbuf[k].grid[imr::kBufferPoints - 1]);
      cmp_int(b_mbtot, (top == 0.0) ? 1 : 0, 1,
              std::string(kPairs[k].name) + " child buffer is zero at its own last node");
      cmp_int(b_mbtot, (mbuf[k].to_resonance[imr::kBufferPoints - 1] > 0.0) ? 1 : 0, 1,
              std::string(kPairs[k].name) + " though the raw sum there is not");
      for (int t = 0; t < imr::kBufferPoints - 1; ++t) {
        const double at = imr::buffered_cross_section(mbuf[k].grid, mbuf[k].to_resonance,
                                                      imr::kBufferPoints, mbuf[k].grid[t]);
        cmp_int(b_mbtot, (at == mbuf[k].to_resonance[t]) ? 1 : 0, 1,
                std::string(kPairs[k].name) + " node " + std::to_string(t) + " is the identity");
      }
    }
    const auto srows = read_csv("bic_imr_mbselect.csv");
    for (const auto& r : srows) {
      const auto it = idx.find(sv(r, 0));
      if (it == idx.end()) { continue; }
      const int k = it->second;
      const double m_pion = (kPairs[k].pion == 111) ? 134.9766 : m_pip;
      const double m_bar = imr::species_properties(kPairs[k].baryon, mp, mn).mass;
      imr::LorentzVector p1, p2;
      tracks(r, p1, p2);
      double partial[imr::kMesonBaryonChannelCount];
      imr::XsecRefusal xref;
      imr::meson_baryon_partials(mbuf[k], kPairs[k].pion, kPairs[k].baryon, m_pion, m_bar, p1, p2,
                                 m_pip, mp, partial, xref);
      CycleRng rng;
      rng.reset(iv(r, 5));
      const int selected = imr::meson_baryon_select(partial, rng);
      cmp_int(b_mbsel, selected, iv(r, 6),
              sv(r, 0) + " sqrt(s)=" + sv(r, 1) + " phase=" + sv(r, 5));
      cmp_int(b_mbsel, rng.n, iv(r, 7), sv(r, 0) + " draws");
    }
  }

  // -------------------------------------------------------------------------------------------
  // -------------------------------------------------------------------------------------------
  // 3l. G4KineticTrack's resonance machinery: the decay tables themselves, G4SampleResonance's
  //     minimum-mass recursion, theActualWidth[] with its Simpson integrals, and
  //     SampleResidualLifetime - which is what schedules every G4BCDecay in the cascade.
  // -------------------------------------------------------------------------------------------
  const int b_dtab = new_bucket("DecayTableContents", 0.0);
  const int b_dmin = new_bucket("SampleResonanceMinMass", 1e-15);
  const int b_dwid = new_bucket("KineticTrackActualWidth", 1e-13);
  const int b_dk0 = new_bucket("KaonZeroSubstitution", 0.0);
  const int b_dlife = new_bucket("ResidualLifetime", 1e-15);
  {
    // The generated table against the dump it came from, field by field. This is not circular:
    // tools/extract_bic_decay.pl transforms the CSV - it reorders channels into one flat array,
    // drops the placeholder rows and resolves each species to an index - and a transformation is
    // exactly the kind of thing that silently loses a row.
    const auto rows = read_csv("bic_imr_decaytable.csv");
    for (const auto& r : rows) {
      const int pdg = iv(r, 0);
      const int idx = imr::decay_species_index(pdg);
      cmp_int(b_dtab, (idx >= 0) ? 1 : 0, 1, "species " + sv(r, 0) + " is in the table");
      if (idx < 0) { continue; }
      const std::string where = sv(r, 1) + " (" + sv(r, 0) + ")";
      cmp_scaled(b_dtab, imr::decay_species_mass()[idx], dv(r, 2), 1e-12, where + " mass");
      cmp_scaled(b_dtab, imr::decay_species_width()[idx], dv(r, 3), 1e-30, where + " width");
      cmp_int(b_dtab, imr::decay_species_shortlived()[idx], iv(r, 4), where + " shortlived");
      cmp_int(b_dtab, imr::decay_species_charge()[idx], iv(r, 5), where + " charge");
      cmp_int(b_dtab, imr::decay_species_baryon()[idx], iv(r, 6), where + " baryon");
      cmp_int(b_dtab, imr::decay_species_n_channels()[idx], iv(r, 8), where + " n_channels");
      const int c = iv(r, 9);
      if (c < 0) { continue; }
      const int ch = imr::decay_species_first_channel()[idx] + c;
      cmp_scaled(b_dtab, imr::decay_channel_br()[ch], dv(r, 10), 1e-12,
                 where + " channel " + sv(r, 9) + " br");
      cmp_int(b_dtab, imr::decay_channel_n_daughters()[ch], iv(r, 11),
              where + " channel " + sv(r, 9) + " n_daughters");
      for (int j = 0; j < iv(r, 11); ++j) {
        cmp_int(b_dtab, imr::decay_channel_daughters()[ch * imr::kDecayMaxDaughters + j],
                iv(r, 12 + j), where + " channel " + sv(r, 9) + " daughter "
                                     + std::to_string(j));
      }
      // The recursion, against the column Geant4 cached. Every species, stable ones included.
      imr::DecayRefusal dref;
      const double got_min = imr::compute_min_mass(pdg, dref);
      cmp_scaled(b_dmin, got_min, dv(r, 7), 1e-12, where + " GetMinimumMass");
      if (dref.any()) {
        std::printf("REFUSED min mass for %s (pdg %d)\n", where.c_str(), dref.refused_pdg);
        ++fails;
      }
    }
  }
  {
    // theActualWidth[], per channel and summed. The two-resonance branch runs a Simpson inside a
    // Simpson - 201 x 201 = 40,401 evaluations per channel - so this is the most arithmetic in
    // the file behind a single number.
    const auto rows = read_csv("bic_imr_actualwidth.csv");
    int last_pdg = 0;
    double last_mass = -1.0;
    double w[16];
    int nw = 0;
    for (const auto& r : rows) {
      const int pdg = iv(r, 0);
      const double am = dv(r, 1);
      if (pdg != last_pdg || am != last_mass) {
        imr::DecayRefusal dref;
        nw = imr::kinetic_track_actual_widths(pdg, am, w, 16, dref);
        last_pdg = pdg;
        last_mass = am;
        if (dref.any()) {
          std::printf("REFUSED actual widths for %d\n", dref.refused_pdg);
          ++fails;
          nw = -1;
        }
      }
      if (nw < 0) { continue; }
      const std::string where = sv(r, 0) + " m=" + sv(r, 1) + " ch=" + sv(r, 3);
      cmp_int(b_dwid, nw, iv(r, 2), where + " n_channels");
      const int c = iv(r, 3);
      if (c >= 0 && c < nw) {
        cmp_scaled(b_dwid, w[c], dv(r, 4), 1e-12, where + " actual width");
      }
      cmp_scaled(b_dwid, imr::evaluate_total_actual_width(w, nw), dv(r, 5), 1e-12,
                 where + " total");
    }
  }
  {
    const auto rows = read_csv("bic_imr_k0flip.csv");
    for (const auto& r : rows) {
      CycleRng rng;
      rng.reset(iv(r, 1));
      const int got = imr::kinetic_track_substitute_k0(iv(r, 0), rng);
      cmp_int(b_dk0, got, iv(r, 2), "K0 phase " + sv(r, 1));
      cmp_int(b_dk0, rng.n, iv(r, 4), "K0 phase " + sv(r, 1) + " draws");
      const int idx = imr::decay_species_index(got);
      cmp_int(b_dk0, (idx >= 0) ? imr::decay_species_n_channels()[idx] : -1, iv(r, 3),
              "K0 phase " + sv(r, 1) + " channels after the flip");
    }
  }
  {
    const auto rows = read_csv("bic_imr_lifetime.csv");
    for (const auto& r : rows) {
      CycleRng rng;
      rng.reset(iv(r, 5));
      // The Lorentz factor is e/mass, which is what G4LorentzVector::gamma() returns; the test
      // takes it from the dumped four-momentum rather than the dumped gamma so that the round
      // trip through the energy is the port's own.
      const double gamma = dv(r, 3) / dv(r, 1);
      const double got = imr::sample_residual_lifetime(dv(r, 7), gamma, rng);
      cmp_scaled(b_dlife, got, dv(r, 6), 1e-30,
                 sv(r, 0) + " m=" + sv(r, 1) + " pz=" + sv(r, 2) + " phase=" + sv(r, 5));
      cmp_int(b_dlife, rng.n, iv(r, 8), sv(r, 0) + " lifetime draws");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3m. G4KineticTrack::Decay - the channel draw, the daughter masses and the phase-space
  //     kinematics, boosted into the lab and returned in Geant4's own order.
  // -------------------------------------------------------------------------------------------
  const int b_dfs = new_bucket("DecayFinalState", 5e-15);
  const int b_dfsn = new_bucket("DecayProductsAndOrder", 0.0);
  const int b_dfsd = new_bucket("DecayDraws", 0.0);
  {
    const auto rows = read_csv("bic_imr_decayfs.csv");
    int last_pdg = 0;
    double last_mass = -1.0, last_pz = -1.0;
    int last_phase = -1;
    imr::DecayResult res;
    for (const auto& r : rows) {
      const int pdg = iv(r, 0);
      const double am = dv(r, 1);
      const double pz = dv(r, 2);
      const int phase = iv(r, 3);
      if (pdg != last_pdg || am != last_mass || pz != last_pz || phase != last_phase) {
        last_pdg = pdg; last_mass = am; last_pz = pz; last_phase = phase;
        double w[16];
        imr::DecayRefusal dref;
        CycleRng rng;
        rng.reset(phase);
        const int n = imr::kinetic_track_actual_widths(pdg, am, w, 16, dref);
        const imr::LorentzVector parent(deex::Vec3d{0.0, 0.0, pz},
                                        std::sqrt(pz * pz + am * am));
        res = imr::kinetic_track_decay(pdg, am, parent, w, n, rng, dref);
        res.loops = rng.n;  // reuse the field to carry the draw count for the comparison below
        if (dref.any()) {
          std::printf("REFUSED decay of %d at %g\n", pdg, am);
          ++fails;
        }
      }
      const std::string where = sv(r, 0) + " m=" + sv(r, 1) + " pz=" + sv(r, 2) +
                                " phase=" + sv(r, 3) + " i=" + sv(r, 5);
      cmp_int(b_dfsn, res.n, iv(r, 4), where + " n_products");
      cmp_int(b_dfsd, res.loops, iv(r, 11), where + " draws");
      const int i = iv(r, 5);
      if (i < 0 || i >= res.n) { continue; }
      cmp_int(b_dfsn, res.prod[i].pdg, iv(r, 6), where + " product pdg");
      // Each component is judged against the product's own ENERGY, not against itself. A decay
      // product's lab momentum crosses zero as the boost cancels the CM momentum - N(1520)+ at
      // 1666.5 MeV with pz = 700 gives its proton a lab pz of -1.7376 MeV out of terms of about
      // 700 - and a relative comparison there divides an ulp of the operands by the
      // cancellation. Normalised to the four-vector's own size, that row is 1.1e-16.
      const double scale = dv(r, 10);
      cmp_scaled(b_dfs, res.prod[i].p.v.x, dv(r, 7), scale, where + " px");
      cmp_scaled(b_dfs, res.prod[i].p.v.y, dv(r, 8), scale, where + " py");
      cmp_scaled(b_dfs, res.prod[i].p.v.z, dv(r, 9), scale, where + " pz");
      cmp_scaled(b_dfs, res.prod[i].p.e, dv(r, 10), scale, where + " e");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3n. G4MesonAbsorption: the scheduling through GetTimeToAbsorption and AbsorptionCrossSection,
  //     the cluster partner FindAndFillCluster picks, and the two outgoing nucleons.
  // -------------------------------------------------------------------------------------------
  const int b_absch = new_bucket("AbsorptionSchedule", 0.0);
  const int b_abstime = new_bucket("AbsorptionTime", 1e-16);
  const int b_absclu = new_bucket("AbsorptionCluster", 0.0);
  const int b_absfs = new_bucket("AbsorptionFinalState", 5e-15);
  const int b_absdr = new_bucket("AbsorptionDraws", 0.0);
  {
    const double fm = 1.e-12;
    const double m_pip = 139.5701;
    const double m_pi0 = 134.9766;
    auto mass_of = [&](int pdg) {
      if (pdg == 2212) { return mp; }
      if (pdg == 2112) { return mn; }
      if (pdg == 111) { return m_pi0; }
      return m_pip;
    };
    auto charge_of = [](int pdg) {
      if (pdg == 2212 || pdg == 211) { return 1; }
      if (pdg == -211) { return -1; }
      return 0;
    };
    struct APair { const char* name; int pro; int c0; };
    const APair kPairs[] = {{"pip_p", 211, 2212},  {"pim_p", -211, 2212}, {"pim_n", -211, 2112},
                            {"p_pim", 2212, -211}, {"n_pim", 2112, -211}, {"pi0_p", 111, 2212},
                            {"pip_n", 211, 2112},  {"p_pip", 2212, 211}};
    const auto rows = read_csv("bic_imr_absorb.csv");
    for (const auto& r : rows) {
      int pro = 0, c0 = 0;
      for (const APair& q : kPairs) {
        if (sv(r, 0) == q.name) { pro = q.pro; c0 = q.c0; }
      }
      if (pro == 0) { continue; }
      const double ekin = dv(r, 1);
      const double b = dv(r, 2);
      const double m1 = mass_of(pro);
      const double e1 = ekin + m1;
      const imr::LorentzVector q1(deex::Vec3d{0.0, 0.0, std::sqrt(e1 * e1 - m1 * m1)}, e1);
      const double m2 = mass_of(c0);
      const deex::Vec3d k2{150.0, -80.0, 60.0};
      const imr::LorentzVector q2(k2, std::sqrt(g4gpu::mag2(k2) + m2 * m2));
      const deex::Vec3d x0{0.0, 0.0, 0.0};
      const deex::Vec3d x1{b * fm, 0.0, 4.0 * fm};
      imr::AbsorptionRefusal aref;
      const double t = imr::time_to_absorption(pro, c0, charge_of(pro), charge_of(c0), x0, x1, q1,
                                               q1, q2, m1, m2, aref);
      // The candidate list the dump builds: the pair's partner, then a neutron spectator, then a
      // proton one. `find_and_fill_cluster` is asked about the first of them.
      const int charges[3] = {charge_of(c0), 0, 1};
      const deex::Vec3d positions[3] = {x1,
                                        deex::Vec3d{-1.0 * fm, 2.0 * fm, 3.0 * fm},
                                        deex::Vec3d{2.0 * fm, -1.0 * fm, -3.0 * fm}};
      const int partner =
          imr::find_and_fill_cluster(charge_of(pro), charge_of(c0), x1, charges, positions, 3, 0);
      const int scheduled = (t < DBL_MAX && partner >= 0) ? 1 : 0;
      const std::string where = sv(r, 0) + " T=" + sv(r, 1) + " b=" + sv(r, 2);
      cmp_int(b_absch, scheduled, iv(r, 3), where);
      if (scheduled == 1 && iv(r, 3) == 1) {
        cmp_scaled(b_abstime, t, dv(r, 4), 1e-18, where + " time");
        cmp_int(b_absclu, iv(r, 5), 2, where + " two targets");
        cmp_int(b_absclu, c0, iv(r, 6), where + " first target");
        cmp_int(b_absclu, (partner == 1) ? 2112 : 2212, iv(r, 7), where + " partner");
      }
    }
  }
  {
    // FindAndFillCluster on its own. The sets are the dump's, and the port is asked the same
    // question for every candidate as the first target - which is what GetCollisions does.
    struct Cand { double x, y, z; int charge; };
    const Cand kSets[3][5] = {
        {{0.2, 0.0, 4.1, 1}, {-0.2, 0.0, -4.1, 1}, {3.0, 1.0, 1.0, 0}, {-3.0, -1.0, -1.0, 0},
         {0.1, 0.1, 4.0, 0}},
        {{1.0, 1.0, 1.0, 1}, {-1.0, -1.0, -1.0, 1}, {0.5, 0.5, 0.5, 1}, {-0.5, -0.5, -0.5, 1},
         {2.0, 0.0, 0.0, 0}},
        {{0.0, 0.0, 5.0, 0}, {0.0, 0.0, -5.0, 0}, {1.0, 0.0, 4.5, 1}, {-1.0, 0.0, -4.5, 1},
         {0.0, 3.0, 0.0, 0}}};
    const double fm = 1.e-12;
    const auto rows = read_csv("bic_imr_absorbcluster.csv");
    for (const auto& r : rows) {
      const int set = iv(r, 0);
      const int first = iv(r, 2);
      if (set < 0 || set > 2 || first < 0) { continue; }
      int charges[5];
      deex::Vec3d positions[5];
      for (int k = 0; k < 5; ++k) {
        charges[k] = kSets[set][k].charge;
        positions[k] = deex::Vec3d{kSets[set][k].x * fm, kSets[set][k].y * fm,
                                   kSets[set][k].z * fm};
      }
      const int partner = imr::find_and_fill_cluster(1, charges[first], positions[first], charges,
                                                     positions, 5, first);
      cmp_int(b_absclu, partner, iv(r, 3),
              "cluster set " + sv(r, 0) + " first " + sv(r, 2) + " partner");
    }
  }
  {
    const double fm = 1.e-12;
    const double m_pip = 139.5701;
    const double m_pi0 = 134.9766;
    struct FsCase { const char* name; int pro; int a; int b; };
    const FsCase kCases[] = {{"pip_pp", 211, 2212, 2212},  {"pip_pn", 211, 2212, 2112},
                             {"pip_np", 211, 2112, 2212},  {"pip_nn", 211, 2112, 2112},
                             {"pim_pp", -211, 2212, 2212}, {"pim_pn", -211, 2212, 2112},
                             {"pim_np", -211, 2112, 2212}, {"pim_nn", -211, 2112, 2112},
                             {"pi0_pn", 111, 2212, 2112}};
    (void)fm;
    const auto rows = read_csv("bic_imr_absorbfs.csv");
    for (const auto& r : rows) {
      const FsCase* c = nullptr;
      for (const FsCase& q : kCases) {
        if (sv(r, 0) == q.name) { c = &q; }
      }
      if (c == nullptr) { continue; }
      const double ekin = dv(r, 1);
      const double m1 = (c->pro == 111) ? m_pi0 : m_pip;
      const double e1 = ekin + m1;
      // The same tilt the dump applies: rotateY(0.4) then rotateZ(0.9) on a +z momentum.
      const double pmag = std::sqrt(e1 * e1 - m1 * m1);
      double vx = pmag * std::sin(0.4);
      double vz = pmag * std::cos(0.4);
      double vy = 0.0;
      const double cz = std::cos(0.9), sz = std::sin(0.9);
      const double nx = vx * cz - vy * sz;
      const double ny = vx * sz + vy * cz;
      vx = nx;
      vy = ny;
      const imr::LorentzVector pro(deex::Vec3d{vx, vy, vz}, e1);
      const double ma = (c->a == 2212) ? mp : mn;
      const double mb = (c->b == 2212) ? mp : mn;
      const deex::Vec3d k1{31.0, -17.0, 44.0};
      const deex::Vec3d k2{-23.0, 51.0, -12.0};
      const imr::LorentzVector t1(k1, std::sqrt(g4gpu::mag2(k1) + ma * ma));
      const imr::LorentzVector t2(k2, std::sqrt(g4gpu::mag2(k2) + mb * mb));
      CycleRng rng;
      rng.reset(iv(r, 2));
      const imr::AbsorptionFinalState fs = imr::meson_absorption_final_state(
          c->pro, static_cast<double>((c->pro == 211) ? 1 : ((c->pro == -211) ? -1 : 0)), pro, t1,
          t2, c->a, c->b, mp, mn, rng);
      const std::string where = sv(r, 0) + " T=" + sv(r, 1) + " phase=" + sv(r, 2);
      cmp_int(b_absdr, fs.pdg1, iv(r, 3), where + " d1");
      cmp_int(b_absdr, fs.pdg2, iv(r, 4), where + " d2");
      cmp_int(b_absdr, rng.n, iv(r, 13), where + " draws");
      const double s1 = dv(r, 8);
      const double s2 = dv(r, 12);
      cmp_scaled(b_absfs, fs.p1.v.x, dv(r, 5), s1, where + " p1x");
      cmp_scaled(b_absfs, fs.p1.v.y, dv(r, 6), s1, where + " p1y");
      cmp_scaled(b_absfs, fs.p1.v.z, dv(r, 7), s1, where + " p1z");
      cmp_scaled(b_absfs, fs.p1.e, dv(r, 8), s1, where + " p1e");
      cmp_scaled(b_absfs, fs.p2.v.x, dv(r, 9), s2, where + " p2x");
      cmp_scaled(b_absfs, fs.p2.v.y, dv(r, 10), s2, where + " p2y");
      cmp_scaled(b_absfs, fs.p2.v.z, dv(r, 11), s2, where + " p2z");
      cmp_scaled(b_absfs, fs.p2.e, dv(r, 12), s2, where + " p2e");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3o. G4DecayKineticTracks through the CONTRACT header - the list-level pass both the binary
  //     cascade and the Fritiof string model open with. See src/physics/hadronic/bic/
  //     kinetic_decay.cuh; this block is the only place the port is exercised the way an outside
  //     caller will use it, list in and list out.
  // -------------------------------------------------------------------------------------------
  const int b_kdn = new_bucket("KineticDecayListShape", 0.0);
  const int b_kdp = new_bucket("KineticDecayProducts", 5e-15);
  const int b_kdb = new_bucket("KineticDecayBookkeeping", 0.0);
  {
    struct Entry { int pdg; double pz; };
    const Entry kSets[7][6] = {
        {{2212, 500}, {2214, 300}, {2124, 100}, {0, 0}, {0, 0}, {0, 0}},
        {{2214, 0}, {2212, 900}, {1114, 250}, {2112, 50}, {0, 0}, {0, 0}},
        {{211, 400}, {113, 600}, {223, 200}, {-211, 100}, {0, 0}, {0, 0}},
        {{225, 800}, {115, 300}, {2212, 0}, {0, 0}, {0, 0}, {0, 0}},
        {{333, 150}, {313, 450}, {3214, 200}, {321, 300}, {0, 0}, {0, 0}},
        {{2224, 700}, {12212, 350}, {32124, 150}, {2112, 250}, {22124, 80}, {2212, 600}},
        {{2212, 100}, {2112, 200}, {211, 300}, {0, 0}, {0, 0}, {0, 0}}};
    const auto rows = read_csv("bic_imr_kdecay.csv");
    int last_set = -1, last_phase = -1;
    bic::DecayTrack list[64];
    int n_out = 0;
    int draws = 0;
    for (const auto& r : rows) {
      const int set = iv(r, 0);
      const int phase = iv(r, 1);
      if (set != last_set || phase != last_phase) {
        last_set = set;
        last_phase = phase;
        int n = 0;
        for (int k = 0; k < 6; ++k) {
          const int pdg = kSets[set][k].pdg;
          if (pdg == 0) { continue; }
          const int si = imr::decay_species_index(pdg);
          if (si < 0) {
            std::printf("REFUSED kdecay species %d\n", pdg);
            ++fails;
            continue;
          }
          const double m = imr::decay_species_mass()[si];
          const double pz = kSets[set][k].pz;
          list[n] = bic::DecayTrack{};
          list[n].pdg = pdg;
          list[n].momentum = imr::LorentzVector(deex::Vec3d{0.0, 0.0, pz},
                                                std::sqrt(pz * pz + m * m));
          list[n].creator_model_id = 70 + k;
          ++n;
        }
        CycleRng rng;
        rng.reset(phase);
        bic::KineticDecayRefusal kref;
        n_out = bic::decay_kinetic_tracks(list, n, 64, rng, kref);
        draws = rng.n;
        if (kref.any()) {
          std::printf("REFUSED kdecay set %d phase %d (pdg %d)\n", set, phase, kref.refused_pdg);
          ++fails;
        }
      }
      const std::string where = "set " + sv(r, 0) + " phase " + sv(r, 1) + " i " + sv(r, 4);
      cmp_int(b_kdn, n_out, iv(r, 3), where + " n_out");
      cmp_int(b_kdn, draws, iv(r, 13), where + " draws");
      const int i = iv(r, 4);
      if (i < 0 || i >= n_out) { continue; }
      cmp_int(b_kdn, list[i].pdg, iv(r, 5), where + " pdg");
      const double scale = dv(r, 9);
      cmp_scaled(b_kdp, list[i].momentum.v.x, dv(r, 6), scale, where + " px");
      cmp_scaled(b_kdp, list[i].momentum.v.y, dv(r, 7), scale, where + " py");
      cmp_scaled(b_kdp, list[i].momentum.v.z, dv(r, 8), scale, where + " pz");
      cmp_scaled(b_kdp, list[i].momentum.e, dv(r, 9), scale, where + " e");
      cmp_int(b_kdb, list[i].parent_resonance_pdg, iv(r, 10), where + " parent pdg");
      cmp_int(b_kdb, list[i].parent_resonance_id, iv(r, 11), where + " parent id");
      cmp_int(b_kdb, list[i].creator_model_id, iv(r, 12), where + " creator model id");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3p. G4Scatterer::Scatter - the whole collision tree from two tracks to their products. The
  //     leaves are compared on their own above; this is the DISPATCH, and what it checks that no
  //     leaf comparison can is which component, which middle composite and which of the 306
  //     concrete channels each uniform lands in, and how many uniforms the path consumed.
  // -------------------------------------------------------------------------------------------
  const int b_scfs = new_bucket("ScatterFinalState", 5e-15);
  const int b_scid = new_bucket("ScatterProductsAndOrder", 0.0);
  const int b_scdr = new_bucket("ScatterDraws", 0.0);
  {
    const double m_pip = 139.5701;
    const double m_pi0 = 134.9766;
    static imr::ConcreteChannel chans[imr::kConcreteChannelCount];
    imr::ResonanceTableRefusal tref0;
    const int n_chan = imr::build_concrete_channels(chans, imr::kConcreteChannelCount);
    auto mass_of = [&](int pdg) {
      if (pdg == 2212) { return mp; }
      if (pdg == 2112) { return mn; }
      if (pdg == 111) { return m_pi0; }
      return m_pip;
    };
    auto iso3_of = [](int pdg) {
      if (pdg == 2212) { return 1; }
      if (pdg == 2112) { return -1; }
      if (pdg == 211) { return 2; }
      if (pdg == -211) { return -2; }
      return 0;
    };
    struct SPair { const char* name; int a; int b; };
    const SPair kPairs[] = {{"pp", 2212, 2212},  {"nn", 2112, 2112},  {"np", 2112, 2212},
                            {"pn", 2212, 2112},  {"pip_p", 211, 2212}, {"pim_p", -211, 2212},
                            {"pi0_p", 111, 2212}, {"pip_n", 211, 2112}, {"pim_n", -211, 2112}};
    // One buffer set per pair of definitions, exactly as G4CollisionComposite caches one.
    static imr::NNChannelBuffers nbuf[4];
    static imr::MesonBaryonBuffers mbuf[5];
    for (int k = 0; k < 4; ++k) {
      imr::ResonanceTableRefusal tr;
      imr::build_nn_channel_buffers(chans, n_chan, kPairs[k].a, kPairs[k].b, mass_of(kPairs[k].a),
                                    mass_of(kPairs[k].b), nbuf[k], tr);
    }
    for (int k = 0; k < 5; ++k) {
      const SPair& q = kPairs[4 + k];
      imr::AnnihRefusal ar;
      imr::build_meson_baryon_buffers(q.a, q.b, mass_of(q.a), mass_of(q.b), iso3_of(q.a),
                                      iso3_of(q.b), m_pip, mp, mbuf[k], ar);
    }
    const auto rows = read_csv("bic_imr_scatter.csv");
    std::string last_key;
    imr::ScatterFinalState fs;
    int draws = 0;
    for (const auto& r : rows) {
      int idx = -1;
      for (int k = 0; k < 9; ++k) {
        if (sv(r, 0) == kPairs[k].name) { idx = k; }
      }
      if (idx < 0) { continue; }
      const std::string key = sv(r, 0) + "|" + sv(r, 1) + "|" + sv(r, 2);
      if (key != last_key) {
        last_key = key;
        const int pa = kPairs[idx].a;
        const int pb = kPairs[idx].b;
        const double m1 = mass_of(pa);
        const double m2 = mass_of(pb);
        const double e1 = dv(r, 1) + m1;
        // The dump's tilt: rotateY(0.35) then rotateZ(1.1) on a +z momentum.
        const double pmag = std::sqrt(e1 * e1 - m1 * m1);
        double vx = pmag * std::sin(0.35);
        double vy = 0.0;
        const double vz = pmag * std::cos(0.35);
        const double cz = std::cos(1.1), sz = std::sin(1.1);
        const double nx = vx * cz - vy * sz;
        const double ny = vx * sz + vy * cz;
        vx = nx;
        vy = ny;
        const imr::LorentzVector q1(deex::Vec3d{vx, vy, vz}, e1);
        const deex::Vec3d v2{37.0, -29.0, 19.0};
        const imr::LorentzVector q2(v2, std::sqrt(g4gpu::mag2(v2) + m2 * m2));
        CycleRng rng;
        rng.reset(iv(r, 2));
        imr::ScatterRefusal sref;
        if (idx < 4) {
          fs = imr::nn_scatter_final_state(chans, n_chan, pa, pb, q1, q2, m1, m2, m1, m2, mp, mn,
                                           m_pip, nbuf[idx], rng, sref);
        } else {
          fs = imr::meson_scatter_final_state(pa, pb, q1, q2, m1, m2, m1, m2, m1, m2, m_pip, mp,
                                              iso3_of(pa), iso3_of(pb), mbuf[idx - 4], rng, sref);
        }
        draws = rng.n;
        if (sref.any()) {
          std::printf("REFUSED scatter %s\n", key.c_str());
          ++fails;
        }
      }
      const std::string where = sv(r, 0) + " T=" + sv(r, 1) + " phase=" + sv(r, 2) +
                                " i=" + sv(r, 4);
      cmp_int(b_scid, fs.n, iv(r, 3), where + " n");
      cmp_int(b_scdr, draws, iv(r, 10), where + " draws");
      const int i = iv(r, 4);
      if (i < 0 || i >= fs.n) { continue; }
      cmp_int(b_scid, fs.pdg[i], iv(r, 5), where + " pdg");
      const double scale = dv(r, 9);
      cmp_scaled(b_scfs, fs.p[i].v.x, dv(r, 6), scale, where + " px");
      cmp_scaled(b_scfs, fs.p[i].v.y, dv(r, 7), scale, where + " py");
      cmp_scaled(b_scfs, fs.p[i].v.z, dv(r, 8), scale, where + " pz");
      cmp_scaled(b_scfs, fs.p[i].e, dv(r, 9), scale, where + " e");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3q. G4BinaryCascade::GetIonMass and the cascade state's bookkeeping. The first arm reads
  //     G4IonTable; the other three are branches on (Z, A) and are asserted against the
  //     arithmetic the source writes, because there is no table behind them to compare with.
  // -------------------------------------------------------------------------------------------
  const int b_ionm = new_bucket("CascadeIonMass", 1e-15);
  const int b_ionb = new_bucket("CascadeIonMassBranches", 0.0);
  {
    const auto rows = read_csv("bic_imr_ionmass.csv");
    for (const auto& r : rows) {
      const int z = iv(r, 0);
      const int a = iv(r, 1);
      cmp_scaled(b_ionm, bic::get_ion_mass(z, a, mn), dv(r, 2), 1e-9,
                 "Z=" + sv(r, 0) + " A=" + sv(r, 1));
    }
    // `Z > A` returns the mass of A PROTONS - "will happen for light nuclei with pions involved".
    cmp_scaled(b_ionb, bic::get_ion_mass(3, 2, mn), bic::get_ion_mass(2, 2, mn), 1e-9,
               "Z>A falls back to (A,A)");
    cmp_scaled(b_ionb, bic::get_ion_mass(0, 5, mn), 5.0 * mn, 1e-9, "neutral remnant is A m_n");
    cmp_scaled(b_ionb, bic::get_ion_mass(0, 0, mn), 0.0, 1e-9, "empty nucleus is zero");
    cmp_scaled(b_ionb, bic::get_ion_mass(-1, 3, mn), 3.0 * mn, 1e-9,
               "negative charge takes the neutral arm, not the throw");
    // theCutOnP: the mass-in-MeV reading, which is docs/RISK.md V72 - every nucleus that exists
    // is above 120 MeV, so the constant is always 45 and the three branches above it are dead.
    cmp_scaled(b_ionb, bic::cut_on_p(bic::get_ion_mass(6, 12, mn)), 45.0, 1e-12, "C12 cut is 45");
    cmp_scaled(b_ionb, bic::cut_on_p(bic::get_ion_mass(82, 208, mn)), 45.0, 1e-12,
               "Pb208 cut is 45");
    cmp_scaled(b_ionb, bic::cut_on_p(mn), 45.0, 1e-12, "even one neutron is above 120 MeV");
  }

  // -------------------------------------------------------------------------------------------
  // 3r. G4BinaryCascade::CheckPauliPrinciple, re-expressed in the dump from the three public
  //     ingredients the source names - see ref/dump/dump_bic.cc's write_imr_pauli.
  // -------------------------------------------------------------------------------------------
  const int b_pau = new_bucket("PauliEnergyThreshold", 1e-14);
  const int b_pauv = new_bucket("PauliVerdict", 0.0);
  const int b_pauf = new_bucket("FermiCorrectionSpecies", 0.0);
  {
    const auto rows = read_csv("bic_imr_pauli.csv");
    int last_a = -1, last_z = -1;
    bic::NuclearDensity density;
    bic::FermiMomentum fermi;
    double coulomb = 0.0;
    for (const auto& r : rows) {
      const int a = iv(r, 0);
      const int z = iv(r, 1);
      if (a != last_a || z != last_z) {
        last_a = a;
        last_z = z;
        // `G4Fancy3DNucleus::Init` picks the shell-model density below A = 17 and the Fermi one
        // at and above it; the Pauli block reads whichever the nucleus was built with.
        density = (a < 17) ? bic::make_shell_model_density(a, z) : bic::make_fermi_density(a, z);
        fermi.init(a, z);
        coulomb = (1.44 / 1.14) * g4gpu::units::MeV<double>() * static_cast<double>(z) /
                  (1.0 + g4gpu::data::g4pow_z13<double>(a));
      }
      const int pdg = iv(r, 3);
      const double pz = dv(r, 4);
      const double m = (pdg == 2212) ? mp : mn;
      bic::CascadeTrack t;
      t.pdg = pdg;
      t.charge = (pdg == 2212) ? 1 : 0;
      t.position = deex::Vec3d{dv(r, 2) * 1.e-12, 0.0, 0.0};
      t.momentum = imr::LorentzVector(deex::Vec3d{0.0, 0.0, pz}, std::sqrt(pz * pz + m * m));
      const std::string where = "A=" + sv(r, 0) + " r=" + sv(r, 2) + " pdg=" + sv(r, 3) +
                                " pz=" + sv(r, 4);
      cmp_scaled(b_pau, density.density(t.position), dv(r, 5), 1e30, where + " density");
      cmp_scaled(b_pau, fermi.fermi_momentum(density.density(t.position)), dv(r, 6), 1e-6,
                 where + " pFermi");
      cmp_scaled(b_pau, coulomb, dv(r, 7), 1e-9, where + " Coulomb barrier");
      const bool blocked = !bic::check_pauli_principle(&t, 1, a, z, density, coulomb);
      cmp_int(b_pauv, blocked ? 1 : 0, iv(r, 9), where + " blocked");
    }
    // The species test the two Fermi corrections branch on, which is on the CODE and not on
    // IsShortLived: abs(pdg) > 1000 and neither nucleon. A pion is not one; an ANTI-nucleon is,
    // because the two exclusions are the positive codes only; a lambda is.
    cmp_int(b_pauf, bic::is_shortlived_for_fermi(2214) ? 1 : 0, 1, "delta+ is short-lived");
    cmp_int(b_pauf, bic::is_shortlived_for_fermi(12212) ? 1 : 0, 1, "N(1440)+ is");
    cmp_int(b_pauf, bic::is_shortlived_for_fermi(2212) ? 1 : 0, 0, "a proton is not");
    cmp_int(b_pauf, bic::is_shortlived_for_fermi(2112) ? 1 : 0, 0, "a neutron is not");
    cmp_int(b_pauf, bic::is_shortlived_for_fermi(211) ? 1 : 0, 0, "a pi+ is not");
    cmp_int(b_pauf, bic::is_shortlived_for_fermi(-2212) ? 1 : 0, 1,
            "an ANTI-proton is, because the exclusions are the positive codes only");
    cmp_int(b_pauf, bic::is_shortlived_for_fermi(-2112) ? 1 : 0, 1, "and so is an anti-neutron");
    cmp_int(b_pauf, bic::is_shortlived_for_fermi(3122) ? 1 : 0, 1, "a lambda is");
  }

  // -------------------------------------------------------------------------------------------
  // 3s. G4BinaryCascade::Capture's decision and G4Absorber::WillBeAbsorbed. Both are private, so
  //     the dump re-expresses them from their public ingredients; see write_imr_capture.
  //
  //     The FIELD each nucleon contributes is dumped per entry and fed back through a stub, not
  //     recomputed: `GetField` and `GetBarrier` are P9's and `bic_field.csv` already compares
  //     them. What is new and what this block checks is the sum over the list, the count, and
  //     the gate on the MEAN.
  // -------------------------------------------------------------------------------------------
  const int b_cap = new_bucket("CaptureDecision", 1e-14);
  const int b_capv = new_bucket("CaptureVerdict", 0.0);
  const int b_abs = new_bucket("AbsorbPredicate", 0.0);
  {
    // A propagator that answers from the dumped field, keyed by the track's position. The
    // capture decision asks `GetField(pdg,pos) - GetBarrier(pdg)` once per track and nothing
    // else, so this is the whole of what it needs.
    struct StubProp {
      const double* fields = nullptr;
      const double* radii = nullptr;
      const int* pdgs = nullptr;
      int n = 0;
      double field(int pdg, const deex::Vec3d& pos) const {
        for (int i = 0; i < n; ++i) {
          if (pdgs[i] == pdg && std::fabs(radii[i] * 1.e-12 - pos.x) < 1e-20) {
            return fields[i];
          }
        }
        return 0.0;
      }
      double barrier(int) const { return 0.0; }  // already folded into `fields`
    };
    const auto frows = read_csv("bic_imr_capturefield.csv");
    const auto rows = read_csv("bic_imr_capture.csv");
    for (const auto& r : rows) {
      const int a = iv(r, 0);
      const int l = iv(r, 2);
      double fields[6];
      double radii[6];
      int pdgs[6];
      double pmags[6];
      int n = 0;
      for (const auto& fr : frows) {
        if (iv(fr, 0) != a || iv(fr, 2) != l) { continue; }
        if (n >= 6) { break; }
        pdgs[n] = iv(fr, 4);
        radii[n] = dv(fr, 5);
        pmags[n] = dv(fr, 6);
        fields[n] = dv(fr, 7);
        ++n;
      }
      StubProp prop;
      prop.fields = fields;
      prop.radii = radii;
      prop.pdgs = pdgs;
      prop.n = n;
      bic::CascadeTrack list[6];
      for (int k = 0; k < n; ++k) {
        const double m = (pdgs[k] == 2212) ? mp : mn;
        list[k] = bic::CascadeTrack{};
        list[k].pdg = pdgs[k];
        list[k].state = bic::kInside;
        list[k].position = deex::Vec3d{radii[k] * 1.e-12, 0.0, 0.0};
        list[k].momentum = imr::LorentzVector(deex::Vec3d{0.0, 0.0, pmags[k]},
                                              std::sqrt(pmags[k] * pmags[k] + m * m));
      }
      const bic::CaptureDecision d = bic::capture_decision(list, n, dv(r, 6), prop);
      const std::string where = "A=" + sv(r, 0) + " list " + sv(r, 2);
      cmp_int(b_capv, d.particles_below_cut, iv(r, 3), where + " n_inside");
      cmp_int(b_capv, d.particles_above_cut, 0,
              where + " particlesAboveCut is always zero - the increment is commented out");
      cmp_scaled(b_cap, d.captured_energy, dv(r, 4), 1e-6, where + " captured energy");
      cmp_int(b_capv, d.capture ? 1 : 0, iv(r, 7), where + " capture");
    }
  }
  {
    const auto rows = read_csv("bic_imr_absorbcut.csv");
    for (const auto& r : rows) {
      const int pdg = iv(r, 0);
      const double ekin = dv(r, 1);
      const double cut = dv(r, 2);
      const double m = (pdg == 2212) ? mp
                                     : ((pdg == 2112) ? mn : ((pdg == 111) ? 134.9766 : 139.5701));
      const bool got = bic::will_be_absorbed(pdg, ekin + m, m, cut);
      cmp_int(b_abs, got ? 1 : 0, iv(r, 3),
              "pdg " + sv(r, 0) + " T=" + sv(r, 1) + " cut=" + sv(r, 2));
    }
    // theCutOnPAbsorb is ZERO, so nothing is ever absorbed - the whole of G4Absorber is dead
    // code in 11.1.1 and is refused by name. This is the assertion that says so, and the
    // absorbcut rows above are what show the predicate is live when the constant is not zero.
    for (double ekin = 0.0; ekin <= 2000.0; ekin += 5.0) {
      for (int pdg : {211, -211, 111}) {
        const double m = (pdg == 111) ? 134.9766 : 139.5701;
        cmp_int(b_abs, bic::will_be_absorbed(pdg, ekin + m, m, bic::cut_on_p_absorb()) ? 1 : 0, 0,
                "no pion is absorbed at the threshold Geant4 sets");
      }
    }
  }

  // 4. Structural assertions on the extracted tables. These are not oracle comparisons - they
  //    are the invariants the sampling depends on, checked on the port's own copy so that a
  //    mis-sliced table fails here rather than as a wrong angle 4,000 rows later.
  // -------------------------------------------------------------------------------------------
  const int b_tab = new_bucket("TableInvariants", 0.0);
  {
    // The channel census the width machinery branches on, asserted rather than assumed. Over the
    // 638 channels of the UNION closure - the binary cascade's species and FTFP's 27, see
    // docs/HADRONIC_PLAN.md section 9.3 - no two-body channel has TWO short-lived daughters, so
    // `IntegrateCMMomentum2` with its Simpson-inside-a-Simpson is unreachable; two three-body
    // channels DO have one, so that branch is live, and it was not when the closure held only
    // the cascade's own species. docs/RISK.md V149. A release that adds a
    // resonance-to-two-resonances channel fails here rather than running code nothing has checked.
    int two_body[3] = {0, 0, 0};
    int three_body[4] = {0, 0, 0, 0};
    int four_body[5] = {0, 0, 0, 0, 0};
    int one_body = 0;
    for (int sidx = 0; sidx < imr::kDecaySpeciesCount; ++sidx) {
      const int first = imr::decay_species_first_channel()[sidx];
      for (int c = 0; c < imr::decay_species_n_channels()[sidx]; ++c) {
        const int ch = first + c;
        const int nd = imr::decay_channel_n_daughters()[ch];
        int n_short = 0;
        for (int j = 0; j < nd; ++j) {
          const int d = imr::decay_channel_daughters()[ch * imr::kDecayMaxDaughters + j];
          const int di = imr::decay_species_index(d);
          if (di >= 0 && imr::decay_species_shortlived()[di] != 0) { ++n_short; }
        }
        if (nd == 1) { ++one_body; }
        if (nd == 2) { ++two_body[n_short]; }
        if (nd == 3) { ++three_body[n_short]; }
        if (nd == 4) { ++four_body[n_short]; }
      }
    }
    cmp_int(b_tab, one_body, 4, "one-daughter channels");
    cmp_int(b_tab, two_body[0], 220, "two-body channels with no short-lived daughter");
    cmp_int(b_tab, two_body[1], 335, "two-body channels with one short-lived daughter");
    cmp_int(b_tab, two_body[2], 0, "two-body channels with TWO - the dead branch");
    cmp_int(b_tab, three_body[0], 75, "three-body channels with no short-lived daughter");
    cmp_int(b_tab, three_body[1], 2, "three-body channels with one - a2(1320)0's two");
    cmp_int(b_tab, three_body[2] + three_body[3], 0, "three-body channels with two or three");
    cmp_int(b_tab, four_body[0], 2, "four-body channels - f2(1270) to 4 pi");
    cmp_int(b_tab, four_body[1] + four_body[2] + four_body[3] + four_body[4], 0,
            "four-body channels with a short-lived daughter");
    cmp_int(b_tab, one_body + two_body[0] + two_body[1] + three_body[0] + three_body[1] +
                       four_body[0],
            imr::kDecayChannelCount, "the census accounts for every channel");
  }
  {

    for (int j = 0; j < imr::kAngularNpEnergies; ++j) {
      const float last = imr::angular_np_sig()[j * imr::kAngularAngles + imr::kAngularAngles - 1];
      cmp_int(b_tab, (std::fabs(static_cast<double>(last) - 1.0) < 2e-5) ? 1 : 0, 1,
              "NP row " + std::to_string(j) + " cumulative ends at 1");
      if (j > 0) {
        cmp_int(b_tab, (imr::angular_np_elab()[j] > imr::angular_np_elab()[j - 1]) ? 1 : 0, 1,
                "NP elab increasing at " + std::to_string(j));
      }
    }
    for (int j = 0; j < imr::kAngularPpEnergies; ++j) {
      const float last = imr::angular_pp_sig()[j * imr::kAngularAngles + imr::kAngularAngles - 1];
      cmp_int(b_tab, (std::fabs(static_cast<double>(last) - 1.0) < 2e-5) ? 1 : 0, 1,
              "PP row " + std::to_string(j) + " cumulative ends at 1");
      if (j > 0) {
        cmp_int(b_tab, (imr::angular_pp_elab()[j] > imr::angular_pp_elab()[j - 1]) ? 1 : 0, 1,
                "PP elab increasing at " + std::to_string(j));
      }
    }
    // The 29th slot of G4XNNTotalLowE::ss is an uninitialised zero (see xsec_nn.cuh). Asserted
    // here rather than assumed, because if a release ever fills it the extractor's count check
    // fires first - and if someone "fixes" the extractor by relaxing the count, this fires.
    cmp_int(b_tab, (imr::nn_total_lowe_ss()[imr::kNNTotalLowESize - 1] == 0.0) ? 1 : 0, 1,
            "G4XNNTotalLowE::ss[28] is the zero-filled 29th slot");
    cmp_int(b_tab,
            (imr::nn_total_lowe_ss()[imr::kNNTotalLowESize - 2] == 3002.71) ? 1 : 0, 1,
            "G4XNNTotalLowE::ss[27] is the last real energy");
    // The stretched np grid: the pp vector's top filled node is at exactly one log unit above
    // its edgeMin and the np vector's is 1% past where its table means it to be. Both measured
    // from the port's own LogVec101, which is what reads them.
    const imr::LogVec101 vpp = imr::LogVec101::make(
        imr::nn_elastic_lowe_pp(), imr::lowe_emin_pp(), imr::lowe_emax());
    const imr::LogVec101 vnp = imr::LogVec101::make(
        imr::nn_elastic_lowe_np(), imr::lowe_emin_np(), imr::lowe_emax());
    const double log_span_pp = std::log(vpp.node_energy(100) / vpp.node_energy(0));
    const double log_span_np = std::log(vnp.node_energy(100) / vnp.node_energy(0));
    cmp_int(b_tab, (std::fabs(log_span_pp - 1.00) < 1e-12) ? 1 : 0, 1,
            "pp grid spans exactly 1.00 in log over its 101 filled nodes");
    cmp_int(b_tab, (std::fabs(log_span_np - 1.0099009900990099) < 1e-12) ? 1 : 0, 1,
            "np grid spans 1.00990 where its table means 0.99");
  }

  // -------------------------------------------------------------------------------------------
  std::printf("\n%-38s %10s %14s  %s\n", "bucket", "points", "worst", "where");
  for (const Bucket& b : buckets) {
    const bool bad = b.worst > b.tol;
    if (bad) { ++fails; }
    std::printf("%-38s %10lld %14.4g  %s%s\n", b.name, b.n, b.worst, bad ? "FAIL " : "",
                b.where.c_str());
  }
  std::printf("\ntest_bic_imr: %s\n", (fails == 0) ? "PASS" : "FAIL");
  return (fails == 0) ? 0 : 1;
}
