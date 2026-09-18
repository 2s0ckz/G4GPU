// The Bertini cascade's deterministic half against ref/oracle/bertini_*.csv.
//
// Everything here is EXACT. Nothing in this file is a distribution: G4CascadeParameters,
// G4InuclElementaryParticle's type table, all 34 channel tables and the nineteen angular and
// momentum distributions are either constants or samplers driven under the eight-value uniform
// cycle `ref/dump/dump_bertini.cc` installs, which makes them deterministic functions of
// (inputs, phase). The statistical half - ApplyYourself on a nucleus - is
// tests/test_bertini_cascade.cu.
//
//   bertini_params.csv        the twenty G4CascadeParameters values, the three transition
//                             energies, and the energy window and de-excitation choice of each
//                             of QBBC's three Bertini instances.
//   bertini_particles.csv     mass, charge, strangeness, baryon number and the nine type
//                             predicates for 43 type codes.
//   bertini_quasideuteron.csv G4NucleiModel::useQuasiDeuteron's whole truth table: 43 projectile
//                             types x 4 dibaryon arguments.
//   bertini_chbins.csv        the three energy scales, 30 + 30 + 31 bin edges.
//   bertini_chtables.csv      every value in every channel's data object: index[], tot[], sum[],
//                             inelastic[], multiplicities[][] and crossSections[][]. 220,536
//                             numbers, which is the whole of the INUCL tree's data.
//   bertini_chfinalstates.csv the 12,036 final-state lists behind them, as type codes.
//   bertini_channels.csv      getCrossSection and getCrossSectionSum on a 128-point grid per
//                             channel, which is what exercises the interpolator's extrapolation
//                             and the NN/NP/PP Stepanov override.
//   bertini_chsample.csv      getMultiplicity and getOutgoingParticleTypes under the cycle, with
//                             the draw counts.
//   bertini_angchoice.csv     which of the fifteen angular distributions ChooseDist returns for
//                             every (initial state, final state, kw) a collider can ask for.
//   bertini_momchoice.csv     the same for the four momentum distributions.
//   bertini_angdst.csv        GetCosTheta under the cycle for the thirteen two-body
//                             distributions, with the draw count.
//   bertini_3bodydst.csv      GetMomentum and the three-body GetCosTheta under the cycle.
//
// **Why the draw counts are compared and not just the values.** A transcription can produce a
// plausible angle from the wrong number of deviates - G4ParamExpTwoBodyAngDst draws one for the
// small/large angle choice and then a second for the inverted exponential, and returning early
// on either of its two guards skips the second. Under a cycle engine the count is observable and
// the value alone is not: with `kSeq` of period 8 a one-deviate error shifts every subsequent
// draw and would show up eventually, but at the point of the error it often does not.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "physics/hadronic/bertini/angular_dist.cuh"
#include "physics/hadronic/bertini/cascade_params.cuh"
#include "physics/hadronic/bertini/channel_tables.cuh"
#include "physics/hadronic/bertini/inucl_particle.cuh"
#include "physics/hadronic/bertini/nuclei_model.cuh"
#include "physics/hadronic/bertini/workspace.cuh"

using namespace g4gpu;
using namespace g4gpu::physics::hadronic;
using bert::BertiniWorkspace;
using bert::LV;
using bert::NucleiModel;
using bert::Vec3d;

/// The deliverable is device code, and a `__host__ __device__` template instantiated only from
/// the host is never compiled for the device at all. These kernels are never launched - this
/// package is validated on the host and the machine's GPU belongs to the integration builds -
/// but instantiating them is what proves the module compiles for the device: no host-only
/// header, no `std::` call without a device overload, no namespace-scope table a device function
/// cannot read. `-Xptxas -v` on this translation unit is where the register and stack numbers in
/// the package report come from.
__global__ void bertini_tables_probe(double ke, int is, int mult, double* out, int* kinds,
                                    bert::BertiniWorkspace* ws) {
  Philox<double> rng(1u, 2u, 3u);
  const bert::ChannelTable t = bert::channel_table(is);
  bool fell = false;
  out[0] = bert::channel_cross_section(t, ke);
  out[1] = bert::channel_cross_section_sum(t, ke);
  out[2] = double(bert::channel_multiplicity(t, ke, rng, fell));
  int chan = -1;
  bert::outgoing_particle_types(t, mult, ke, rng, kinds, chan, fell, ws->sigma_buf);
  out[3] = double(chan);
}

/// The nucleus model, built on the device from (A, Z) and then read the way a propagating
/// particle reads it. `NucleiModel` is 700-odd bytes and lives behind a pointer for the reason
/// workspace.cuh gives; building it into a LOCAL here would be the mistake that file exists to
/// prevent, so the probe takes it by pointer too.
__global__ void bertini_nucleus_probe(int a, int z, NucleiModel* m, BertiniWorkspace* ws,
                                      double* out) {
  Philox<double> rng(7u, 8u, 9u);
  const bert::CascadeParams par = bert::default_cascade_params();
  const bert::NucleiModelParams nmp = bert::nuclei_model_params(par);
  out[0] = double(static_cast<int>(bert::nm_generate_model(*m, a, z, nmp)));
  out[1] = m->nuclei_radius;
  out[2] = m->nuclei_volume;
  out[3] = bert::nm_density(*m, bert::kProton, 0);
  out[4] = bert::nm_fermi_momentum(*m, bert::kNeutron, 0);
  out[5] = bert::nm_potential(*m, bert::kLambda, 0);
  out[6] = bert::nm_fermi_kinetic(*m, bert::kProton, 0);
  out[7] = double(bert::nm_zone(*m, m->nuclei_radius * 0.5));
  out[8] = bert::nm_current_density(*m, bert::kUnboundPN, 0);
  bool refused = false;
  out[9] = bert::nm_total_cross_section(1.5, bert::kProton * bert::kProton, nmp, refused);
  out[10] = bert::nm_absorption_cross_section(0.2, bert::kPionPlus, nmp, refused);
  const LV nucleon = bert::nm_generate_nucleon_momentum(*m, bert::kProton, 0, rng);
  out[11] = nucleon.e;
  int dtype = 0;
  const LV qd = bert::nm_generate_quasideuteron(*m, bert::kProton, bert::kNeutron, 0, dtype, rng);
  out[12] = qd.e + double(dtype);
  const LV bullet = bert::lv_set_vect_m(Vec3d{0.0, 0.0, 2.0}, bert::inucl_particle_mass(1));
  out[13] = bert::nm_inverse_mean_free_path(*m, bert::kProton, bullet, bert::kNeutron, nucleon,
                                            0, nmp, refused);
  out[14] = bert::nm_generate_interaction_length(1.0, out[13], false, 0.0, rng);
  bert::CascadeParticle cp;
  cp.type = bert::kProton;
  cp.momentum = bullet;
  cp.position = Vec3d{0.0, 0.0, m->nuclei_radius * 0.9};
  cp.current_zone = m->number_of_zones - 1;
  bool in_zero = false;
  // `reflected_now` is the eighth argument and it is not optional. docs/RISK.md V126: Geant4's
  // `resetReflection()` clears the FLAG and not the counter, and the two are read by different
  // files, so the port carries them separately and `boundaryTransition` has to say which one it
  // just set. This probe passes `cp.reflected` so that the device kernel it instantiates is the
  // same call the cascade makes - a probe that called an older signature would not compile, and
  // for a week this one did not, because it was never rebuilt after the signature changed.
  bert::nm_boundary_transition(*m, cp.type, cp.position, cp.momentum, cp.current_zone, false,
                              cp.reflection_counter, cp.reflected, in_zero);
  out[15] = cp.momentum.e + double(cp.current_zone);
  out[16] = bert::nm_worth_to_propagate(*m, true, cp.type, 0, 0.1) ? 1.0 : 0.0;
  bool moving_in = false;
  out[17] = bert::cp_path_to_next_zone(cp.position, cp.momentum, cp.current_zone,
                                       bert::nm_radius(*m, 0), m->nuclei_radius, moving_in);
  out[18] = double(ws->n_cascade);
  bert::ws_push_cascade(*ws, cp);
  out[19] = double(static_cast<int>(ws->overflow));
}

__global__ void bertini_angdst_probe(double ekin, double pcm, int is, int fs, int kw,
                                     double* out) {
  Philox<double> rng(4u, 5u, 6u);
  const bert::AngDstChoice c = bert::choose_two_body_angdst(is, fs, kw);
  double v = 0.0;
  bool flat = false;
  if (c.kind == bert::AngDstKind::kNumInt) {
    v = bert::numint_cos_theta(data::bertini_numint_angdst()[c.index], ekin, pcm, rng);
  } else if (c.kind == bert::AngDstKind::kParamExp) {
    v = bert::paramexp_cos_theta(data::bertini_paramexp_angdst()[c.index], ekin, pcm, rng);
  } else if (c.kind == bert::AngDstKind::kParamAng3Body) {
    v = bert::paramang_cos_theta(data::bertini_paramang_angdst()[c.index], bert::kProton, ekin,
                                 rng, flat);
  }
  out[0] = v;
  const bert::CascadeParams par = bert::default_cascade_params();
  const int mi = bert::choose_multibody_momdst(is, 3, par);
  out[1] = bert::parammom_momentum(data::bertini_parammom_momdst()[mi], bert::kProton, ekin, rng);
  out[2] = bert::inucl_particle_mass(bert::kProton);
  out[3] = bert::inucl_get_al(56) + bert::inucl_cs_nn(50.0) + bert::inucl_cs_pn(20.0) +
           bert::inucl_fermi_energy(56, 26, 0) + bert::inucl_binding_energy(56, 26) +
           bert::inucl_binding_energy_asymptotic(56, 26);
  bool oor = false;
  out[4] = bert::inucl_nuclei_level_density(56, oor);
  const bert::ParaMakerParams pm = bert::para_maker_get_params(26.0);
  out[5] = pm.ak[1] + pm.cpa[5];
}

namespace {

int fails = 0;

/// The eight uniforms dump_bertini.cc's CycleEngine cycles through, in its order. Written out on
/// both sides rather than derived, because if the two lists diverge every exact comparison in
/// this file turns into noise.
__host__ __device__ inline const double* cycle_seq() {
  static const double v[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
  return v;
}

struct CycleRng {
  int phase = 0;
  int n = 0;
  __host__ __device__ void reset(int p) { phase = p; n = 0; }
  __host__ __device__ double uniform() {
    const double v = cycle_seq()[(n + phase) % 8];
    ++n;
    return v;
  }
};

// ---------------------------------------------------------------------------------------------
// Comparison bookkeeping, as tests/test_precompound.cu defines it
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

void cmp_str(int bi, const std::string& got, const std::string& want, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  if (got != want) {
    b.worst = 1.0;
    if (b.where.empty()) { b.where = where + " got '" + got + "' want '" + want + "'"; }
  }
}

// ---------------------------------------------------------------------------------------------
// CSV reading
// ---------------------------------------------------------------------------------------------

std::string oracle_dir() {
  const char* e = std::getenv("G4GPU_ORACLE");
  return (e != nullptr) ? std::string(e) : std::string("ref/oracle");
}

std::vector<std::string> read_lines(const std::string& name) {
  std::vector<std::string> out;
  const std::string path = oracle_dir() + "/" + name;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) {
    std::printf("FAIL: cannot open %s\n", path.c_str());
    ++fails;
    return out;
  }
  char line[65536];
  while (std::fgets(line, sizeof line, f) != nullptr) { out.push_back(line); }
  std::fclose(f);
  return out;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (const char c : s) {
    if (c == ',') { out.push_back(cur); cur.clear(); }
    else if (c != '\n' && c != '\r') { cur.push_back(c); }
  }
  out.push_back(cur);
  return out;
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

/// Splits a space-separated list of integers, as the two final-state columns carry them.
std::vector<int> split_codes(const std::string& s) {
  std::vector<int> out;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && s[i] == ' ') { ++i; }
    std::size_t j = i;
    while (j < s.size() && s[j] != ' ') { ++j; }
    if (j > i) { out.push_back(std::atoi(s.substr(i, j - i).c_str())); }
    i = j;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// 1. bertini_params.csv
// ---------------------------------------------------------------------------------------------
void check_params() {
  const int b = new_bucket("CascadeParameters", 0.0);
  const bert::CascadeParams p = bert::default_cascade_params();
  const bert::InterfaceLimits il = bert::default_interface_limits();

  std::map<std::string, double> want;
  const std::vector<std::string> lines = read_lines("bertini_params.csv");
  for (std::size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split(lines[i]);
    if (f.size() < 2) { continue; }
    want[f[0]] = std::atof(f[1].c_str());
  }
  if (want.empty()) { std::printf("FAIL: bertini_params.csv empty\n"); ++fails; return; }

  auto chk = [&](const char* k, double got) {
    if (want.find(k) == want.end()) {
      std::printf("FAIL: bertini_params.csv has no row '%s'\n", k);
      ++fails;
      return;
    }
    cmp(b, got, want[k], k);
  };

  chk("verbose", p.verbose);
  chk("checkConservation", p.check_conservation ? 1 : 0);
  chk("usePreCompound", p.use_precompound ? 1 : 0);
  chk("doCoalescence", p.do_coalescence ? 1 : 0);
  chk("showHistory", p.show_history ? 1 : 0);
  chk("use3BodyMom", p.use_3body_mom ? 1 : 0);
  chk("usePhaseSpace", p.use_phase_space ? 1 : 0);
  chk("piNAbsorption", p.pin_absorption);
  chk("useTwoParam", p.use_two_param ? 1 : 0);
  chk("radiusScale", p.radius_scale);
  chk("radiusSmall", p.radius_small);
  chk("radiusAlpha", p.radius_alpha);
  chk("radiusTrailing", p.radius_trailing);
  chk("fermiScale", p.fermi_scale);
  chk("xsecScale", p.xsec_scale);
  chk("gammaQDScale", p.gamma_qd_scale);
  chk("dpMaxDoublet", p.dp_max_doublet);
  chk("dpMaxTriplet", p.dp_max_triplet);
  chk("dpMaxAlpha", p.dp_max_alpha);
  chk("epsilon_energy_rel", il.ep_relative);
  chk("epsilon_energy_abs_MeV", il.ep_absolute_MeV);
  chk("balance_rel", il.balance_relative);
  chk("balance_abs_GeV", il.balance_absolute_GeV);
  chk("maximumTries", il.maximum_tries);

  // The three QBBC configurations. The de-excitation choice is the column that is not a number
  // in the oracle - it is which builder called usePreCompoundDeexcitation - so it is checked
  // against the source of truth the dump encodes: nucleons and pions get PreCompound, kaons and
  // hyperons do not.
  const int bd = new_bucket("QbbcBertiniRange", 0.0);
  struct Case { int pdg; const char* lo; const char* hi; bert::DeexciteChoice dx; };
  const Case cases[] = {
      {2212, "qbbc_nucleon_minE_MeV", "qbbc_nucleon_maxE_MeV", bert::DeexciteChoice::kPreCompound},
      {2112, "qbbc_nucleon_minE_MeV", "qbbc_nucleon_maxE_MeV", bert::DeexciteChoice::kPreCompound},
      {211, "qbbc_pion_minE_MeV", "qbbc_pion_maxE_MeV", bert::DeexciteChoice::kPreCompound},
      {-211, "qbbc_pion_minE_MeV", "qbbc_pion_maxE_MeV", bert::DeexciteChoice::kPreCompound},
      {321, "qbbc_kaon_hyperon_minE_MeV", "qbbc_kaon_hyperon_maxE_MeV",
       bert::DeexciteChoice::kCascade},
      {3122, "qbbc_kaon_hyperon_minE_MeV", "qbbc_kaon_hyperon_maxE_MeV",
       bert::DeexciteChoice::kCascade}};
  for (const Case& c : cases) {
    const bert::QbbcBertiniLookup r = bert::qbbc_bertini_range(c.pdg);
    cmp_int(bd, r.ok ? 1 : 0, 1, "pdg " + std::to_string(c.pdg) + " registered");
    cmp(bd, r.range.emin_MeV, want[c.lo], std::string("emin pdg ") + std::to_string(c.pdg));
    cmp(bd, r.range.emax_MeV, want[c.hi], std::string("emax pdg ") + std::to_string(c.pdg));
    cmp_int(bd, static_cast<int>(r.range.deexcite), static_cast<int>(c.dx),
            std::string("deexcite pdg ") + std::to_string(c.pdg));
  }
  // Sigma0 is the species with a Bertini channel table and no Bertini process, because
  // G4HadParticles::sHyperons omits it.
  cmp_int(bd, bert::qbbc_bertini_range(3212).ok ? 1 : 0, 0, "sigma0 not registered");
  cmp_int(bd, bert::qbbc_bertini_range(1000020040).ok ? 1 : 0, 0, "alpha not registered");
}

// ---------------------------------------------------------------------------------------------
// 2. bertini_particles.csv and bertini_quasideuteron.csv
// ---------------------------------------------------------------------------------------------
void check_particles() {
  const int bm = new_bucket("InuclParticleMass", 0.0);
  const int bi = new_bucket("InuclParticleFlags", 0.0);
  const std::vector<std::string> lines = read_lines("bertini_particles.csv");
  for (std::size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split(lines[i]);
    if (f.size() < 14) { continue; }
    const int code = iv(f, 0);
    const std::string w = "code " + std::to_string(code);
    const bert::InuclTypeRow r = bert::inucl_type_row(code);
    // The nuclei code 0 has no definition and Geant4's getParticleMass returns 0.0 for it; the
    // port returns -1 so that an unhandled type announces itself. That one row is checked as the
    // sentinel rather than against the oracle's zero.
    if (code == bert::kNuclei) {
      cmp_int(bi, r.mass_GeV < 0.0 ? 1 : 0, 1, w + " sentinel mass");
    } else {
      cmp(bm, r.mass_GeV, dv(f, 2), w + " mass");
      cmp_int(bi, r.strangeness, iv(f, 3), w + " strangeness");
    }
    cmp_int(bi, bert::inucl_names_baryon(code), iv(f, 4), w + " names::baryon");
    cmp_int(bi, bert::inucl_is_photon(code) ? 1 : 0, iv(f, 5), w + " isPhoton");
    cmp_int(bi, bert::inucl_is_muon(code) ? 1 : 0, iv(f, 6), w + " isMuon");
    cmp_int(bi, bert::inucl_is_electron(code) ? 1 : 0, iv(f, 7), w + " isElectron");
    cmp_int(bi, bert::inucl_is_neutrino(code) ? 1 : 0, iv(f, 8), w + " isNeutrino");
    cmp_int(bi, bert::inucl_is_pion(code) ? 1 : 0, iv(f, 9), w + " pion");
    cmp_int(bi, bert::inucl_is_nucleon(code) ? 1 : 0, iv(f, 10), w + " nucleon");
    cmp_int(bi, bert::inucl_is_antinucleon(code) ? 1 : 0, iv(f, 11), w + " antinucleon");
    cmp_int(bi, bert::inucl_is_quasideuteron(code) ? 1 : 0, iv(f, 12), w + " quasi_deutron");
    cmp_int(bi, bert::inucl_names_hyperon(code) ? 1 : 0, iv(f, 13), w + " hyperon");
    // The PDG round trip, for the codes that have one: type(makeDefinition(t)) == t.
    if (r.pdg != 0 && code != bert::kNuclei) {
      cmp_int(bi, bert::inucl_type_from_pdg(r.pdg), code, w + " pdg round trip");
    }
  }
}

// ---------------------------------------------------------------------------------------------
// 3. bertini_chbins.csv - the three energy scales
// ---------------------------------------------------------------------------------------------
void check_bins() {
  const int b = new_bucket("ChannelBinEdges", 0.0);
  const std::vector<std::string> lines = read_lines("bertini_chbins.csv");
  for (std::size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split(lines[i]);
    if (f.size() < 5) { continue; }
    const int is = iv(f, 0);
    const int nbins = iv(f, 2);
    const int bin = iv(f, 3);
    if (bin < 0) { continue; }   // the header row the dump writes per channel
    const bert::ChannelTable t = bert::channel_table(is);
    if (!t.valid()) { cmp_int(b, 0, 1, "no table for state " + std::to_string(is)); continue; }
    cmp_int(b, t.ne(), nbins, "state " + std::to_string(is) + " nbins");
    cmp(b, t.bins[bin], dv(f, 4),
        "state " + std::to_string(is) + " bin " + std::to_string(bin));
  }
}

// ---------------------------------------------------------------------------------------------
// 4. bertini_chtables.csv - every value in every channel's data object
// ---------------------------------------------------------------------------------------------
void check_chtables() {
  const int bidx = new_bucket("ChannelIndexArray", 0.0);
  const int btot = new_bucket("ChannelTot", 0.0);
  const int bsum = new_bucket("ChannelSum", 0.0);
  const int binel = new_bucket("ChannelInelastic", 0.0);
  const int bmult = new_bucket("ChannelMultiplicities", 0.0);
  const int bchan = new_bucket("ChannelCrossSections", 0.0);
  const int bname = new_bucket("ChannelName", 0.0);

  const std::vector<std::string> lines = read_lines("bertini_chtables.csv");
  long long parse_errors = 0;
  for (std::size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split(lines[i]);
    if (f.size() < 7) { continue; }
    const int is = iv(f, 0);
    const std::string block = sv(f, 2);
    const int mult = iv(f, 3);
    const int ebin = iv(f, 4);
    const int chan = iv(f, 5);
    const double v = dv(f, 6);
    if (block == "PARSE_ERROR") { ++parse_errors; continue; }

    const bert::ChannelTable t = bert::channel_table(is);
    if (!t.valid()) {
      cmp_int(bidx, 0, 1, "no table for state " + std::to_string(is));
      continue;
    }
    const std::string w = sv(f, 1) + " " + block + " m" + std::to_string(mult) + " e" +
                          std::to_string(ebin) + " c" + std::to_string(chan);
    if (block == "index") {
      cmp_int(bidx, t.def->index[mult - 2], static_cast<long long>(v), w);
    } else if (block == "index_last") {
      // G4CascadeData::print writes `(indices start to stop-1)`, so the oracle's second number
      // is the LAST index of the block and not the exclusive end. Comparing index[mult-1]
      // against it fails by exactly one on every channel, which is what it did first run.
      cmp_int(bidx, t.def->index[mult - 1] - 1, static_cast<long long>(v), w);
    } else if (block == "tot") {
      cmp(btot, t.tot[ebin], v, w);
    } else if (block == "sum") {
      cmp(bsum, t.sum[ebin], v, w);
    } else if (block == "inelastic") {
      cmp(binel, t.inelastic[ebin], v, w);
    } else if (block == "mult") {
      cmp(bmult, t.mult[(mult - 2) * t.ne() + ebin], v, w);
    } else if (block == "chan") {
      cmp(bchan, t.xs[chan * t.ne() + ebin], v, w);
    }
    cmp_str(bname, std::string(data::kBertiniChannelNames[t.slot]), sv(f, 1),
            "state " + std::to_string(is));
  }
  if (parse_errors != 0) {
    std::printf("FAIL: bertini_chtables.csv has %lld PARSE_ERROR rows\n", parse_errors);
    ++fails;
  }
}

void check_finalstates() {
  const int b = new_bucket("ChannelFinalStates", 0.0);
  const std::vector<std::string> lines = read_lines("bertini_chfinalstates.csv");
  for (std::size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split(lines[i]);
    if (f.size() < 6) { continue; }
    const int is = iv(f, 0);
    const int mult = iv(f, 2);
    const int idx = iv(f, 3);
    const std::vector<int> want = split_codes(sv(f, 5));
    const bert::ChannelTable t = bert::channel_table(is);
    if (!t.valid()) { cmp_int(b, 0, 1, "no table " + std::to_string(is)); continue; }
    const signed char* block = t.fs[mult - 2];
    if (block == nullptr) {
      cmp_int(b, 0, 1, "no fs block for mult " + std::to_string(mult));
      continue;
    }
    cmp_int(b, static_cast<long long>(want.size()), mult,
            sv(f, 1) + " m" + std::to_string(mult) + " size");
    for (std::size_t j = 0; j < want.size(); ++j) {
      cmp_int(b, block[idx * mult + static_cast<int>(j)], want[j],
              sv(f, 1) + " m" + std::to_string(mult) + "[" + std::to_string(idx) + "][" +
                  std::to_string(j) + "]");
    }
  }
}

// ---------------------------------------------------------------------------------------------
// 5. bertini_channels.csv - getCrossSection / getCrossSectionSum on the grid
// ---------------------------------------------------------------------------------------------
void check_channel_xsec() {
  const int b = new_bucket("ChannelGetCrossSection", 0.0);
  const int bs = new_bucket("ChannelGetCrossSectionSum", 0.0);
  const std::vector<std::string> lines = read_lines("bertini_channels.csv");
  for (std::size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split(lines[i]);
    if (f.size() < 4) { continue; }
    const int is = iv(f, 0);
    const double ke = dv(f, 1);
    const bert::ChannelTable t = bert::channel_table(is);
    if (!t.valid()) { cmp_int(b, 0, 1, "no table " + std::to_string(is)); continue; }
    char w[96];
    std::snprintf(w, sizeof w, "state %d ke %.6g", is, ke);
    cmp(b, bert::channel_cross_section(t, ke), dv(f, 2), w);
    cmp(bs, bert::channel_cross_section_sum(t, ke), dv(f, 3), w);
  }
}

// ---------------------------------------------------------------------------------------------
// 6. bertini_chsample.csv - the two samplers under the cycle
// ---------------------------------------------------------------------------------------------
void check_chsample() {
  // One workspace for the whole check, as a cascade would have: the sigma buffer is a data
  // member in Geant4 and a workspace field here, not a local (see workspace.cuh).
  static bert::BertiniWorkspace ws;
  const int bm = new_bucket("ChannelMultiplicitySample", 0.0);
  const int bmd = new_bucket("ChannelMultiplicityDraws", 0.0);
  const int bf = new_bucket("ChannelFinalStateSample", 0.0);
  const int bfd = new_bucket("ChannelFinalStateDraws", 0.0);
  const std::vector<std::string> lines = read_lines("bertini_chsample.csv");
  for (std::size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split(lines[i]);
    if (f.size() < 8) { continue; }
    const int is = iv(f, 0);
    const double ke = dv(f, 1);
    const int phase = iv(f, 2);
    const int want_mult = iv(f, 3);
    const int want_mdraws = iv(f, 4);
    const int fs_mult = iv(f, 5);
    const int want_fdraws = iv(f, 6);
    const std::vector<int> want_codes = split_codes(sv(f, 7));

    const bert::ChannelTable t = bert::channel_table(is);
    if (!t.valid()) { cmp_int(bm, 0, 1, "no table " + std::to_string(is)); continue; }

    char w[128];
    std::snprintf(w, sizeof w, "state %d ke %.6g phase %d", is, ke, phase);

    CycleRng rng;
    rng.reset(phase);
    bool fell = false;
    const int mult = bert::channel_multiplicity(t, ke, rng, fell);
    cmp_int(bm, mult, want_mult, std::string(w) + " mult");
    cmp_int(bmd, rng.n, want_mdraws, std::string(w) + " mult draws");

    rng.reset(phase);
    int kinds[bert::kMaxFinalStateSize];
    int chan = -1;
    const bert::ChannelRefusal r =
        bert::outgoing_particle_types(t, fs_mult, ke, rng, kinds, chan, fell, ws.sigma_buf);
    if (r != bert::ChannelRefusal::kNone) {
      cmp_int(bf, static_cast<int>(r), 0,
              std::string(w) + " m" + std::to_string(fs_mult) + " refused");
      continue;
    }
    cmp_int(bfd, rng.n, want_fdraws, std::string(w) + " m" + std::to_string(fs_mult) + " draws");
    // getOutgoingParticleTypes CLAMPS a multiplicity above the table's maximum rather than
    // failing - it prints " Illegal multiplicity" and sets `mult = maxMult` - so a row asking
    // for 8 from a six-multiplicity channel comes back with seven codes. The expected size is
    // therefore the clamped multiplicity, which is where the port's clamp is checked.
    const int clamped = (fs_mult > t.max_multiplicity()) ? t.max_multiplicity() : fs_mult;
    cmp_int(bf, static_cast<long long>(want_codes.size()), clamped,
            std::string(w) + " m" + std::to_string(fs_mult) + " size");
    for (std::size_t j = 0; j < want_codes.size() && j < bert::kMaxFinalStateSize; ++j) {
      cmp_int(bf, kinds[j], want_codes[j],
              std::string(w) + " m" + std::to_string(fs_mult) + "[" + std::to_string(j) + "]");
    }
  }

  // The one channel Geant4 reads out of bounds for, refused by name rather than reproduced.
  const bert::ChannelTable mu = bert::channel_table(bert::kMuonMinus * bert::kProton);
  const int bref = new_bucket("MuMinusPRefusal", 0.0);
  cmp_int(bref, mu.valid() ? 1 : 0, 1, "mu- p table exists");
  if (mu.valid()) {
    CycleRng rng;
    int kinds[bert::kMaxFinalStateSize];
    int chan = -1;
    bool fell = false;
    for (int m = 3; m <= 9; ++m) {
      rng.reset(0);
      const bert::ChannelRefusal r =
          bert::outgoing_particle_types(mu, m, 1.0, rng, kinds, chan, fell, ws.sigma_buf);
      cmp_int(bref, static_cast<int>(r),
              static_cast<int>(bert::ChannelRefusal::kSingleChannelAboveMult2),
              "mu- p mult " + std::to_string(m));
    }
    // Multiplicity 2 is the one that is in bounds, and it must still work.
    rng.reset(0);
    const bert::ChannelRefusal r2 =
        bert::outgoing_particle_types(mu, 2, 1.0, rng, kinds, chan, fell, ws.sigma_buf);
    cmp_int(bref, static_cast<int>(r2), 0, "mu- p mult 2 allowed");
  }

  // kMaxChannelsPerMult, from the data rather than from a reading.
  const int bcap = new_bucket("ChannelBufferCapacity", 0.0);
  int worst = 0;
  for (int i = 0; i < data::kBertiniChannels; ++i) {
    const data::BertiniChannel& d = data::bertini_channels()[i];
    for (int m = 0; m < d.n_mult; ++m) {
      const int n = d.index[m + 1] - d.index[m];
      if (n > worst) { worst = n; }
    }
  }
  cmp_int(bcap, worst, bert::kMaxChannelsPerMult, "widest multiplicity block");

  // The tail of index[] for the twelve six-multiplicity channels. It is checked HERE, by
  // construction, and not against the oracle: G4CascadeData::print only emits an
  // " (indices A to B)" line for multiplicities 2..NM+1, so index[7] and index[8] have no
  // dumped counterpart and no reader in the port either. An anti-vacuity perturbation that
  // padded them with zero instead of the last offset therefore went uncaught - which makes the
  // padding a question rather than a hole, and this is the pin that settles it.
  // G4CascadeData::initialize sets index[7] = N28 = N27+N8 and index[8] = N29 = N28+N9, and for
  // a channel with no eighth or ninth multiplicity N8 = N9 = 0, so both equal index[6].
  const int btail = new_bucket("ChannelIndexTail", 0.0);
  for (int i = 0; i < data::kBertiniChannels; ++i) {
    const data::BertiniChannel& d = data::bertini_channels()[i];
    if (d.n_mult >= 8) { continue; }
    const std::string w = std::string(data::kBertiniChannelNames[i]) + " index tail";
    cmp_int(btail, d.index[7], d.index[d.n_mult], w + "[7]");
    cmp_int(btail, d.index[8], d.index[d.n_mult], w + "[8]");
  }

  // `elastic_chan`, recomputed here from the final-state codes rather than trusted from the
  // extractor - the same pin, and for the same reason: it has no dumped counterpart (Geant4
  // keeps the index in a local inside initialize()) and only one reader in the port, the
  // NN/NP/PP Stepanov branch, which three channels reach. So an extractor perturbation that
  // simply failed to find the row went uncaught. G4CascadeData::initialize's rule is: the
  // first entry of the MULTIPLICITY-2 block whose two type codes multiply to the initial
  // state, or -1 (its `i2b == index[1]` case) if there is none.
  const int belas = new_bucket("ChannelElasticRow", 0.0);
  for (int i = 0; i < data::kBertiniChannels; ++i) {
    const data::BertiniChannel& d = data::bertini_channels()[i];
    const signed char* two = data::bertini_channel_fs() + d.fs_off[0];
    int want = -1;
    for (int j = 0; j < d.index[1]; ++j) {
      if (int(two[j * 2]) * int(two[j * 2 + 1]) == d.initial_state) { want = j; break; }
    }
    cmp_int(belas, d.elastic_chan, want,
            std::string(data::kBertiniChannelNames[i]) + " elastic row");
  }
  // Exactly ONE of the 34 tables has no elastic two-body row, and it is mu- p: its only
  // multiplicity-2 final state is {neutron, nu_mu}, whose product is 2 * -3 = -6, while its
  // initial state is mum * pro = -23. That is the case G4CascadeData's FIXME ("No elastic
  // channel in table!") is about, and it is the one channel whose `inelastic[]` is its whole
  // `tot[]`. The count is printed so that a change in the data cannot quietly move a channel
  // from one arm of that branch to the other.
  int no_elastic = 0;
  for (int i = 0; i < data::kBertiniChannels; ++i) {
    if (data::bertini_channels()[i].elastic_chan < 0) { ++no_elastic; }
  }
  std::printf("note: %d of %d channel tables carry no elastic two-body row\n", no_elastic,
              data::kBertiniChannels);
}

// ---------------------------------------------------------------------------------------------
// 7. bertini_angchoice.csv and bertini_momchoice.csv - the two dispatchers
// ---------------------------------------------------------------------------------------------
std::string angdst_name(const bert::AngDstChoice& c) {
  switch (c.kind) {
    case bert::AngDstKind::kNumInt:
      return data::kBertiniNumIntNames[c.index];
    case bert::AngDstKind::kParamExp:
      return data::kBertiniParamExpNames[c.index];
    case bert::AngDstKind::kParamAng3Body:
      return data::kBertiniParamAngNames[c.index];
    default:
      return "none";
  }
}

void check_dispatchers() {
  const int b = new_bucket("ChooseTwoBodyAngDst", 0.0);
  const std::vector<std::string> lines = read_lines("bertini_angchoice.csv");
  for (std::size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split(lines[i]);
    if (f.size() < 4) { continue; }
    const int is = iv(f, 0);
    const int fs = iv(f, 1);
    const int kw = iv(f, 2);
    char w[96];
    std::snprintf(w, sizeof w, "is %d fs %d kw %d", is, fs, kw);
    cmp_str(b, angdst_name(bert::choose_two_body_angdst(is, fs, kw)), sv(f, 3), w);
  }

  const int bm = new_bucket("ChooseMultiBodyMomDst", 0.0);
  const bert::CascadeParams par = bert::default_cascade_params();
  const std::vector<std::string> mlines = read_lines("bertini_momchoice.csv");
  for (std::size_t i = 1; i < mlines.size(); ++i) {
    const std::vector<std::string> f = split(mlines[i]);
    if (f.size() < 3) { continue; }
    const int is = iv(f, 0);
    const int mult = iv(f, 1);
    char w[96];
    std::snprintf(w, sizeof w, "is %d mult %d", is, mult);
    const int idx = bert::choose_multibody_momdst(is, mult, par);
    cmp_str(bm, std::string(data::kBertiniParamMomNames[idx]), sv(f, 2), w);
  }
}

// ---------------------------------------------------------------------------------------------
// 8. bertini_angdst.csv - the thirteen two-body distributions under the cycle
// ---------------------------------------------------------------------------------------------
void check_angdst() {
  const int b = new_bucket("AngDstCosTheta", 0.0);
  const int bd = new_bucket("AngDstDraws", 0.0);

  // Map the oracle's name string to (kind, index), which is how a row identifies its object.
  std::map<std::string, std::pair<bert::AngDstKind, int>> by_name;
  for (int i = 0; i < data::kBertiniNumIntAngDsts; ++i) {
    by_name[data::kBertiniNumIntNames[i]] = {bert::AngDstKind::kNumInt, i};
  }
  for (int i = 0; i < data::kBertiniParamExpAngDsts; ++i) {
    by_name[data::kBertiniParamExpNames[i]] = {bert::AngDstKind::kParamExp, i};
  }

  const std::vector<std::string> lines = read_lines("bertini_angdst.csv");
  for (std::size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split(lines[i]);
    if (f.size() < 6) { continue; }
    const std::string name = sv(f, 0);
    const double ekin = dv(f, 1);
    const double pcm = dv(f, 2);
    const int phase = iv(f, 3);
    const int want_draws = iv(f, 4);
    const double want_ct = dv(f, 5);
    const auto it = by_name.find(name);
    if (it == by_name.end()) {
      std::printf("FAIL: bertini_angdst.csv names an unknown distribution '%s'\n", name.c_str());
      ++fails;
      return;
    }
    char w[160];
    std::snprintf(w, sizeof w, "%s ekin %.6g pcm %.6g phase %d", name.c_str(), ekin, pcm, phase);
    CycleRng rng;
    rng.reset(phase);
    double ct = 0.0;
    if (it->second.first == bert::AngDstKind::kNumInt) {
      ct = bert::numint_cos_theta(data::bertini_numint_angdst()[it->second.second], ekin, pcm,
                                  rng);
    } else {
      ct = bert::paramexp_cos_theta(data::bertini_paramexp_angdst()[it->second.second], ekin,
                                    pcm, rng);
    }
    cmp(b, ct, want_ct, w);
    cmp_int(bd, rng.n, want_draws, std::string(w) + " draws");
  }
}

// ---------------------------------------------------------------------------------------------
// 9. bertini_3bodydst.csv - GetMomentum and the three-body GetCosTheta
// ---------------------------------------------------------------------------------------------
void check_3body() {
  const int bm = new_bucket("ParamMomGetMomentum", 0.0);
  const int bmd = new_bucket("ParamMomDraws", 0.0);
  const int ba = new_bucket("ParamAngCosTheta", 0.0);
  const int bad = new_bucket("ParamAngDraws", 0.0);

  std::map<std::string, int> mom_by_name, ang_by_name;
  for (int i = 0; i < data::kBertiniParamMomDsts; ++i) {
    mom_by_name[data::kBertiniParamMomNames[i]] = i;
  }
  for (int i = 0; i < data::kBertiniParamAngDsts; ++i) {
    ang_by_name[data::kBertiniParamAngNames[i]] = i;
  }

  const std::vector<std::string> lines = read_lines("bertini_3bodydst.csv");
  for (std::size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split(lines[i]);
    if (f.size() < 7) { continue; }
    const std::string kind = sv(f, 0);
    const std::string name = sv(f, 1);
    const int ptype = iv(f, 2);
    const double ekin = dv(f, 3);
    const int phase = iv(f, 4);
    const int want_draws = iv(f, 5);
    const double want_v = dv(f, 6);
    char w[160];
    std::snprintf(w, sizeof w, "%s %s ptype %d ekin %.6g phase %d", kind.c_str(), name.c_str(),
                  ptype, ekin, phase);
    CycleRng rng;
    rng.reset(phase);
    if (kind == "mom") {
      const auto it = mom_by_name.find(name);
      if (it == mom_by_name.end()) { ++fails; return; }
      const double v =
          bert::parammom_momentum(data::bertini_parammom_momdst()[it->second], ptype, ekin, rng);
      cmp(bm, v, want_v, w);
      cmp_int(bmd, rng.n, want_draws, std::string(w) + " draws");
    } else {
      const auto it = ang_by_name.find(name);
      if (it == ang_by_name.end()) { ++fails; return; }
      bool flat = false;
      const double v = bert::paramang_cos_theta(data::bertini_paramang_angdst()[it->second],
                                                ptype, ekin, rng, flat);
      cmp(ba, v, want_v, w);
      cmp_int(bad, rng.n, want_draws, std::string(w) + " draws");
    }
  }
}

// ---------------------------------------------------------------------------------------------
// 10. bertini_nuclei.csv, bertini_zonelookup.csv, bertini_potfold.csv, bertini_quasideuteron.csv
//     and bertini_nucxsec.csv - G4NucleiModel's whole deterministic half
// ---------------------------------------------------------------------------------------------
void check_nuclei_model() {
  const bert::CascadeParams par = bert::default_cascade_params();
  const bert::NucleiModelParams nmp = bert::nuclei_model_params(par);

  const int bz = new_bucket("NucleiModelZones", 0.0);
  const int br = new_bucket("NucleiModelRadii", 0.0);
  const int bv = new_bucket("NucleiModelVolumes", 0.0);
  const int bd = new_bucket("NucleiModelDensities", 0.0);
  const int bf = new_bucket("NucleiModelFermiMomenta", 0.0);
  const int bp = new_bucket("NucleiModelPotentials", 0.0);
  const int bk = new_bucket("NucleiModelFermiKinetic", 0.0);
  const int bb = new_bucket("NucleiModelBindingEnergies", 0.0);

  const std::vector<std::string> lines = read_lines("bertini_nuclei.csv");
  for (std::size_t i = 1; i < lines.size(); ++i) {
    const std::vector<std::string> f = split(lines[i]);
    if (f.size() < 21) { continue; }
    const int a = iv(f, 0);
    const int z = iv(f, 1);
    bert::NucleiModel m;
    const bert::NucleiModelRefusal r = bert::nm_generate_model(m, a, z, nmp);
    if (r != bert::NucleiModelRefusal::kNone) {
      cmp_int(bz, static_cast<int>(r), 0, "A=" + std::to_string(a));
      continue;
    }
    char w[96];
    const int iz = iv(f, 7);
    std::snprintf(w, sizeof w, "A=%d Z=%d zone %d", a, z, iz);
    cmp_int(bz, m.number_of_zones, iv(f, 2), w);
    cmp(br, m.nuclei_radius, dv(f, 3), std::string(w) + " nuclei_radius");
    cmp(bv, m.nuclei_volume, dv(f, 4), std::string(w) + " nuclei_volume");
    cmp(bb, m.binding_energies[0], dv(f, 5), std::string(w) + " be proton");
    cmp(bb, m.binding_energies[1], dv(f, 6), std::string(w) + " be neutron");
    cmp(br, bert::nm_radius(m, iz), dv(f, 8), std::string(w) + " radius");
    cmp(bv, bert::nm_volume(m, iz), dv(f, 9), std::string(w) + " volume");
    cmp(bd, bert::nm_density(m, bert::kProton, iz), dv(f, 10), std::string(w) + " dens p");
    cmp(bd, bert::nm_density(m, bert::kNeutron, iz), dv(f, 11), std::string(w) + " dens n");
    cmp(bf, bert::nm_fermi_momentum(m, bert::kProton, iz), dv(f, 12), std::string(w) + " pf p");
    cmp(bf, bert::nm_fermi_momentum(m, bert::kNeutron, iz), dv(f, 13), std::string(w) + " pf n");
    cmp(bp, bert::nm_potential(m, bert::kProton, iz), dv(f, 14), std::string(w) + " vp p");
    cmp(bp, bert::nm_potential(m, bert::kNeutron, iz), dv(f, 15), std::string(w) + " vp n");
    cmp(bk, bert::nm_fermi_kinetic(m, bert::kProton, iz), dv(f, 16), std::string(w) + " ek p");
    cmp(bk, bert::nm_fermi_kinetic(m, bert::kNeutron, iz), dv(f, 17), std::string(w) + " ek n");
    cmp(bp, bert::nm_potential(m, bert::kPionPlus, iz), dv(f, 18), std::string(w) + " vp pi");
    cmp(bp, bert::nm_potential(m, bert::kKaonPlus, iz), dv(f, 19), std::string(w) + " vp K");
    cmp(bp, bert::nm_potential(m, bert::kLambda, iz), dv(f, 20), std::string(w) + " vp Y");
  }

  // getZone(r), including the boundary radii themselves - `r < zone_radii[iz]` is strict, so a
  // particle exactly on a boundary is in the OUTER zone, and that is what the probes at
  // r == zone_radii[iz] check.
  const int bzl = new_bucket("NucleiModelGetZone", 0.0);
  const std::vector<std::string> zl = read_lines("bertini_zonelookup.csv");
  for (std::size_t i = 1; i < zl.size(); ++i) {
    const std::vector<std::string> f = split(zl[i]);
    if (f.size() < 4) { continue; }
    bert::NucleiModel m;
    bert::nm_generate_model(m, iv(f, 0), iv(f, 1), nmp);
    char w[96];
    std::snprintf(w, sizeof w, "A=%d Z=%d r=%.17g", iv(f, 0), iv(f, 1), dv(f, 2));
    cmp_int(bzl, bert::nm_zone(m, dv(f, 2)), iv(f, 3), w);
  }

  // getPotential's five-slot fold over all 44 type codes and every zone, including the
  // out-of-range zone that returns 0.
  const int bpf = new_bucket("NucleiModelPotentialFold", 0.0);
  const std::vector<std::string> pf = read_lines("bertini_potfold.csv");
  bert::NucleiModel fe;
  bert::nm_generate_model(fe, 56, 26, nmp);
  for (std::size_t i = 1; i < pf.size(); ++i) {
    const std::vector<std::string> f = split(pf[i]);
    if (f.size() < 5) { continue; }
    char w[96];
    std::snprintf(w, sizeof w, "type %d zone %d", iv(f, 2), iv(f, 3));
    cmp(bpf, bert::nm_potential(fe, iv(f, 2), iv(f, 3)), dv(f, 4), w);
  }

  // useQuasiDeuteron's whole truth table.
  const int bq = new_bucket("UseQuasiDeuteron", 0.0);
  const std::vector<std::string> qd = read_lines("bertini_quasideuteron.csv");
  for (std::size_t i = 1; i < qd.size(); ++i) {
    const std::vector<std::string> f = split(qd[i]);
    if (f.size() < 3) { continue; }
    char w[96];
    std::snprintf(w, sizeof w, "ptype %d qdtype %d", iv(f, 0), iv(f, 1));
    cmp_int(bq, bert::nm_use_quasideuteron(iv(f, 0), iv(f, 1)) ? 1 : 0, iv(f, 2), w);
  }

  // totalCrossSection and absorptionCrossSection.
  const int bt = new_bucket("NucleiModelTotalXsec", 0.0);
  const int ba = new_bucket("NucleiModelAbsorptionXsec", 0.0);
  const std::vector<std::string> nx = read_lines("bertini_nucxsec.csv");
  for (std::size_t i = 1; i < nx.size(); ++i) {
    const std::vector<std::string> f = split(nx[i]);
    if (f.size() < 5) { continue; }
    const std::string kind = sv(f, 0);
    const int bullet = iv(f, 1);
    const int target = iv(f, 2);
    const double ke = dv(f, 3);
    char w[128];
    std::snprintf(w, sizeof w, "%s bullet %d target %d ke %.6g", kind.c_str(), bullet, target,
                  ke);
    bool refused = false;
    if (kind == "total") {
      const double v = bert::nm_total_cross_section(ke, bullet * target, nmp, refused);
      cmp_int(bt, refused ? 1 : 0, 0, std::string(w) + " refused");
      cmp(bt, v, dv(f, 4), w);
    } else {
      const double v = bert::nm_absorption_cross_section(ke, bullet, nmp, refused);
      cmp_int(ba, refused ? 1 : 0, 0, std::string(w) + " refused");
      cmp(ba, v, dv(f, 4), w);
    }
  }

  // The trailing effect is OFF with the dumped parameters, and the reason is a strict `<`
  // against a zero radius. Pinned by construction: a repeat hit at exactly the same point must
  // still pass. If radiusTrailing ever becomes non-zero this assertion is where it shows.
  const int btr = new_bucket("PassTrailingIsOff", 0.0);
  const bert::Vec3d hits[2] = {bert::Vec3d{1.0, 2.0, 3.0}, bert::Vec3d{0.0, 0.0, 0.0}};
  cmp_int(btr, bert::nm_pass_trailing(hits, 2, hits[0], nmp.r_nucleon) ? 1 : 0, 1,
          "repeat hit at the same point passes with r_nucleon = 0");
  cmp(btr, nmp.r_nucleon, 0.0, "r_nucleon");
  // ... and that it CAN reject, so the loop above is not vacuous.
  cmp_int(btr, bert::nm_pass_trailing(hits, 2, hits[0], 1.0) ? 1 : 0, 0,
          "repeat hit rejected at r_nucleon = 1");

  // Two things the oracle grid cannot reach, pinned by construction instead. Both were
  // perturbations that went uncaught, and both are V52's "is the input read at all?" question
  // rather than a hole.
  //
  // `crossSectionUnits` multiplies both cross sections and is 1.0 in the install, so dropping
  // the multiplication changes nothing on any dumped row. It is checked by CHANGING the
  // parameter: xsecScale is settable from G4NUCMODEL_XSEC_SCALE, so a run that sets it has to
  // be reproducible, and the linearity is the whole of what the parameter does.
  const int bxs = new_bucket("CrossSectionUnitsScaling", 0.0);
  bert::NucleiModelParams scaled = nmp;
  scaled.cross_section_units = 2.0;
  bool ref1 = false, ref2 = false;
  const int state = bert::kProton * bert::kProton;
  for (double ke : {0.05, 0.5, 2.0, 6.0}) {
    const double one = bert::nm_total_cross_section(ke, state, nmp, ref1);
    const double two = bert::nm_total_cross_section(ke, state, scaled, ref2);
    cmp(bxs, two, 2.0 * one, "total xsec scaling at ke " + std::to_string(ke));
  }
  bert::NucleiModelParams qd_scaled = nmp;
  qd_scaled.gamma_qd_scale = 3.0;
  for (double ke : {0.005, 0.05, 0.5}) {
    const double one = bert::nm_absorption_cross_section(ke, bert::kPhoton, nmp, ref1);
    const double two = bert::nm_absorption_cross_section(ke, bert::kPhoton, qd_scaled, ref2);
    cmp(bxs, two, 3.0 * one, "gammaQD scaling at ke " + std::to_string(ke));
  }

  // The `csec < 0` floor in absorptionCrossSection. On the dumped grid it never fires: the pion
  // parametrisation is positive everywhere on (0, 0.3) - its resonance term alone is at least
  // 2 millibarn there - and above 1 GeV neither branch assigns anything, so csec stays exactly
  // 0. The only way to a negative value is the PHOTON table extrapolated BELOW its first bin,
  // where bins[0] = 0 and xsec[1] = 0.7 make the linear extrapolation negative. No caller can
  // get there - `getKinEnergyInTheTRS` returns `e - m`, which is non-negative - so the guard is
  // dead under physical input. It is pinned here rather than left as a reading: at ke = -0.001
  // the unfloored extrapolation is about -0.29 mb and the function must return exactly 0.
  const int bfl = new_bucket("AbsorptionNegativeFloor", 0.0);
  bool refn = false;
  cmp(bfl, bert::nm_absorption_cross_section(-0.001, bert::kPhoton, nmp, refn), 0.0,
      "photon absorption floored at ke = -0.001");
  cmp_int(bfl, refn ? 1 : 0, 0, "photon is an absorptive projectile");
  // The unfloored value, computed here from the same table, has to be negative for the line
  // above to be a test of the floor and not of a coincidence.
  const double unfloored =
      bert::interp_value(-0.001, bert::nm_gamma_qd_bins(), bert::nm_gamma_qd_xsec(), 30);
  cmp_int(bfl, (unfloored < 0.0) ? 1 : 0, 1, "unfloored extrapolation is negative");
  // And a non-absorptive projectile is refused by name rather than given a zero.
  bool refp = false;
  bert::nm_absorption_cross_section(0.1, bert::kProton, nmp, refp);
  cmp_int(bfl, refp ? 1 : 0, 1, "proton refused by absorptionCrossSection");
}

}  // namespace

int main() {
  check_params();
  check_particles();
  check_bins();
  check_chtables();
  check_finalstates();
  check_channel_xsec();
  check_chsample();
  check_dispatchers();
  check_angdst();
  check_3body();
  check_nuclei_model();

  std::printf("%-34s %12s %14s\n", "bucket", "points", "worst rel");
  long long total = 0;
  for (const Bucket& b : buckets) {
    total += b.n;
    const bool bad = (b.n == 0) || (b.worst > b.tol);
    std::printf("%-34s %12lld %14.3g %s%s\n", b.name, b.n, b.worst, bad ? "FAIL " : "ok   ",
                b.where.c_str());
    if (bad) { ++fails; }
  }
  std::printf("%lld comparisons, %d failures\n", total, fails);
  return (fails == 0) ? 0 : 1;
}
