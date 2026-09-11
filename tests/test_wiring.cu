// P8's wiring decisions, against the process managers Geant4 actually built.
//
// Nothing here is a model. What is checked is the four questions a stepper asks before it calls
// one, and every one of them is answered against `ref/oracle/decay_applicable.csv` and
// `ref/oracle/decay_atrest.csv` - which are dumps of the CONSTRUCTED QBBC's own process
// managers, not of the source. docs/RISK.md V43 is why: three questions about that physics list
// were answered by reading G4EmBuilder and all three were wrong, and the fix each time was to
// ask the object.
//
//   1. WHICH SPECIES DECAY IN FLIGHT. `decays_in_flight` must equal `is_applicable && !stable`
//      for every one of the 507 species the oracle dumps, not just the ones this port
//      transports - a predicate that enumerates its members is a snapshot of a rule (V43
//      again), and the rule is checkable much further than it is used.
//
//   2. WHICH STOPPED SPECIES DECAY, PER STAGE. The oracle walks each particle's
//      `GetAtRestProcessVector()` and asks every process for the length it would offer a
//      stopped track. A species whose vector holds a NON-Decay process at exactly 0.0 is one
//      whose decay is pre-empted: `G4HadronStoppingProcess::AtRestGetPhysicalInteractionLength`
//      is `return 0.0` and a zero cannot be beaten. Those are exactly the species that must not
//      decay at rest in the final stage and must decay at rest in stage 1, where Geant4's own
//      at-rest capture is inactivated. Derived from the CSV rather than listed here.
//
//   3. THE PDG ROUND TRIP, both directions, including the nuclear encoding.
//
//   4. THE IN-FLIGHT LENGTH IS AN EXPONENTIAL WITH THE RIGHT MEAN. `decay_in_flight_length`
//      re-draws every step where G4Decay carries a remaining count, and the claim in its
//      header is that the two are the same distribution. That claim is checked here against the
//      closed form rather than against a second sampler: the mean of N draws against
//      `beta*gamma*c*tau` from the oracle's own lifetime column, and the fraction below one
//      mean free path against `1 - 1/e`.
//
//   5. THE NEUTRON SUB-PROCESS CHOICE, AGAINST P2's TABLE AND NOT AGAINST A SECOND COPY OF IT.
//      P1's socket and P2's builder were two transcriptions of G4NeutronGeneralProcess's grid;
//      P8 made the socket a view of P2's tables. What is checked here is that the two now give
//      the same double for every node and every bin midpoint of both zones, and - separately,
//      and this is the measurement rather than the assertion - what the socket's own formula
//      used to give on the same rows.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "core/track_buffer.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/neutron_general_xs.cuh"
#include "physics/hadronic/wiring.cuh"
#include "physics/hadronic/xs/particlexs.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

int g_fails = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++g_fails;
  }
}

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
    if (first) { first = false; continue; }
    if (line[0] == '\0' || line[0] == '\n') { continue; }
    rows.push_back(split(line));
  }
  std::fclose(f);
  return rows;
}

int inum(const std::string& s) { return std::atoi(s.c_str()); }
double num(const std::string& s) { return std::atof(s.c_str()); }

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";

  // ---------------------------------------------------------------- 1. decays in flight
  //
  // Over every species the oracle dumps, and the port's answer for a species it has no
  // ParticleType for is "no": `pdg_code` of a type it does not know is 0, whose lifetime is -1,
  // so `decay_is_applicable` declines. That is checked as part of the same loop rather than
  // skipped - it is the whole reason a code with no row must not fall through to a nearby one.
  {
    bool ok = false;
    const auto rows = read_csv(dir + "/decay_applicable.csv", ok);
    if (!ok) { return 1; }
    int n = 0, mismatch = 0, unknown = 0, refused_unstable = 0;
    for (const auto& r : rows) {
      if (r.size() < 8) { continue; }
      const std::string name = r[0];
      const int pdg = inum(r[1]);
      const bool stable = inum(r[5]) != 0;
      const bool applicable = inum(r[6]) != 0;
      const ParticleType t = particle_type_of_pdg(pdg);
      if (t == ParticleType::kNumTypes) {
        // A species this port has no row for. `decays_in_flight` cannot be asked about it, and
        // that is the point: a code with no row is not silently a nearby species.
        ++unknown;
        continue;
      }
      // The round trip has to close, or the answer below is about a different particle. This
      // is what caught the hypernuclei: five codes in the oracle carry a non-zero lambda count
      // and decoding only (Z, A) turned a hypertriton into a triton.
      if (pdg_code(t) != pdg) {
        std::printf("  FAIL: %s pdg %d round-trips to %d\n", name.c_str(), pdg, pdg_code(t));
        ++g_fails;
        continue;
      }
      // A SPECIES WITH NO KERNEL IS A DIFFERENT QUESTION, and folding it in would have made
      // this loop assert something false. K0L, K0S, lambda, sigma+-, xi- have ParticleType
      // rows - P1 added them so a refusal can name what it refused - and Geant4 gives all six
      // a decay, but P4's tables refuse them by PDG code, so `decay_is_applicable` declines
      // and `decays_in_flight` is false where Geant4's is true. That is not a wrong answer to
      // a question anyone asks: no stepper ever sees one, because `species_disposition` refuses
      // them before a track exists. What has to hold is that the two refusals AGREE - a species
      // whose decay P4 refuses must be one this transport will not step - and that is the
      // assertion, because the alternative is a particle that gets a kernel and then silently
      // never decays.
      if (species_disposition(t) != SpeciesDisposition::kStepped) {
        if (applicable && !stable) { ++refused_unstable; }
        check(!had::decays_in_flight(t),
              name + ": refused by P4's tables and refused as a species");
        continue;
      }
      ++n;
      const bool want = applicable && !stable;
      if (had::decays_in_flight(t) != want) {
        std::printf("  FAIL: %s: decays_in_flight %d, Geant4 applicable %d stable %d\n",
                    name.c_str(), int(had::decays_in_flight(t)), int(applicable), int(stable));
        ++mismatch;
        ++g_fails;
      }
    }
    std::printf("1. decays_in_flight: %d transported species checked against the constructed "
                "QBBC, %d disagree\n"
                "   %d species have a ParticleType, no kernel and an unstable Geant4 "
                "definition - refused on both sides\n"
                "   %d species have no ParticleType at all and are declined by code\n",
                n, mismatch, refused_unstable, unknown);
    check(n >= 14, "at least the fourteen transported species were in the oracle");
  }

  // ---------------------------------------------------------------- 2. the at-rest stage switch
  {
    bool ok = false;
    const auto rows = read_csv(dir + "/decay_atrest.csv", ok);
    if (!ok) { return 1; }
    // Per species: does its at-rest vector hold a process that is NOT Decay and offers exactly
    // 0.0? Derived from the file, so the set cannot drift from Geant4's.
    std::map<int, bool> preempted;
    std::map<int, bool> has_decay;
    std::map<int, std::string> preempted_by;
    std::map<int, std::string> named;
    for (const auto& r : rows) {
      if (r.size() < 9) { continue; }
      const int pdg = inum(r[1]);
      named[pdg] = r[0];
      const std::string proc = r[5];
      if (proc == "Decay") {
        has_decay[pdg] = true;
        continue;
      }
      if (proc.empty()) { continue; }
      if (num(r[8]) == 0.0) {
        preempted[pdg] = true;
        preempted_by[pdg] = proc;
      }
    }
    int n = 0, skipped = 0;
    for (const auto& kv : named) {
      const int pdg = kv.first;
      const ParticleType t = particle_type_of_pdg(pdg);
      if (t == ParticleType::kNumTypes) { continue; }
      if (pdg_code(t) != pdg) { continue; }
      // Only species with a kernel. See section 1: a refused species never reaches a stepper,
      // so what its at-rest queue would have held is not a question this wiring answers.
      if (species_disposition(t) != SpeciesDisposition::kStepped) { ++skipped; continue; }
      // THE POSITRON IS THE ONE PRE-EMPTION THIS PORT ALREADY HAS. `decay_atrest.csv` gives it
      // `annihil` (subtype 5) at exactly 0.0 - a zero-length at-rest process, the same shape as
      // the three hadronic captures - and `step_lepton`'s dying branch does it:
      // `sample_annihilation_at_rest` emits the two 511 keV photons. So it is pre-empted and
      // NOT refused, which is why the derivation below cannot be "pre-empted implies a hole".
      // Named rather than filtered by charge or by lepton number, because what makes it
      // different is that the port has the model.
      if (t == ParticleType::kPositron) {
        check(had::stopped_refusal(t) == had::HadronicRefusal::kNumHadronicRefusals,
              "a stopped e+ refuses nothing: G4eplusAnnihilation at rest IS ported");
        ++n;
        continue;
      }
      ++n;
      const bool pre = preempted.count(pdg) != 0;
      const bool dec = has_decay.count(pdg) != 0;
      // Stage 1 inactivates the pre-empting process, so a species with G4Decay in its at-rest
      // vector decays; without one it does nothing. Note `dec` is false for a STABLE species
      // even when G4Decay is registered - `decay_atrest.csv` only lists processes on the
      // at-rest vector, and the triton's is there, so the stable test is the port's own.
      const bool want_stage1 = dec;
      const bool want_final = dec && !pre;
      const bool got_stage1 = had::decay_at_rest_allowed(t, had::HadronicStage::kStage1);
      const bool got_final = had::decay_at_rest_allowed(t, had::HadronicStage::kFinal);
      // The triton is the one row where the port says no and Geant4's vector says Decay: it is
      // flagged stable, so DecayIt is a no-op. Handled by name in the expectation rather than
      // by widening the rule, because it IS a difference and it has a reason.
      const bool stable_noop = (t == ParticleType::kTriton);
      if (got_stage1 != (want_stage1 && !stable_noop)) {
        std::printf("  FAIL: %s stage1 at-rest decay %d, expected %d (has Decay %d)\n",
                    kv.second.c_str(), int(got_stage1), int(want_stage1 && !stable_noop),
                    int(dec));
        ++g_fails;
      }
      if (got_final != (want_final && !stable_noop)) {
        std::printf("  FAIL: %s final at-rest decay %d, expected %d (pre-empted by %s)\n",
                    kv.second.c_str(), int(got_final), int(want_final && !stable_noop),
                    pre ? preempted_by[pdg].c_str() : "nothing");
        ++g_fails;
      }
      // And the refusal that replaces it has to exist for exactly the pre-empted ones, plus
      // the antiproton - which is pre-empted AND stable, so it has no decay to lose and its
      // stopping process is still a hole.
      const had::HadronicRefusal r = had::stopped_refusal(t);
      const bool has_refusal = (r != had::HadronicRefusal::kNumHadronicRefusals);
      if (has_refusal != pre) {
        std::printf("  FAIL: %s stopped_refusal %d, pre-empted %d\n", kv.second.c_str(),
                    int(has_refusal), int(pre));
        ++g_fails;
      }
    }
    std::printf("2. at-rest stage switch: %d transported species, both stages, against each "
                "one's own at-rest process vector (%d refused species skipped)\n", n, skipped);
    // The three the plan names, spelled out as well as derived, because "the derivation agrees
    // with itself" is not the claim - the claim is that these three change with the stage.
    check(had::decay_at_rest_allowed(ParticleType::kPionMinus, had::HadronicStage::kStage1),
          "a stopped pi- decays in stage 1");
    check(!had::decay_at_rest_allowed(ParticleType::kPionMinus, had::HadronicStage::kFinal),
          "a stopped pi- does not decay in the final stage");
    check(had::decay_at_rest_allowed(ParticleType::kKaonMinus, had::HadronicStage::kStage1),
          "a stopped K- decays in stage 1");
    check(!had::decay_at_rest_allowed(ParticleType::kKaonMinus, had::HadronicStage::kFinal),
          "a stopped K- does not decay in the final stage");
    check(had::decay_at_rest_allowed(ParticleType::kMuonMinus, had::HadronicStage::kStage1),
          "a stopped mu- decays in stage 1");
    check(!had::decay_at_rest_allowed(ParticleType::kMuonMinus, had::HadronicStage::kFinal),
          "a stopped mu- does not decay in the final stage");
    // And the positives, in BOTH stages, because a switch that changed them would be wrong in
    // the same file.
    for (ParticleType t : {ParticleType::kPionPlus, ParticleType::kKaonPlus,
                           ParticleType::kMuonPlus}) {
      check(had::decay_at_rest_allowed(t, had::HadronicStage::kStage1),
            std::string("a stopped ") + particle_name(t) + " decays in stage 1");
      check(had::decay_at_rest_allowed(t, had::HadronicStage::kFinal),
            std::string("a stopped ") + particle_name(t) + " decays in the final stage too");
    }
    // The antiproton decays in neither: it is stable.
    check(!had::decay_at_rest_allowed(ParticleType::kAntiProton, had::HadronicStage::kStage1),
          "a stopped antiproton does not decay in stage 1");
    check(had::stopped_refusal(ParticleType::kAntiProton)
              == had::HadronicRefusal::kStoppedAntiProton,
          "a stopped antiproton is a refused annihilation");
    // The proton has no at-rest process at all and therefore nothing to refuse.
    check(had::stopped_refusal(ParticleType::kProton)
              == had::HadronicRefusal::kNumHadronicRefusals,
          "a stopped proton refuses nothing");
  }

  // ---------------------------------------------------------------- 3. the nuclear encoding
  {
    struct Nuc { int z, a; ParticleType t; };
    const Nuc nucs[] = {
        {0, 1, ParticleType::kNeutron},  {1, 1, ParticleType::kProton},
        {1, 2, ParticleType::kDeuteron}, {1, 3, ParticleType::kTriton},
        {2, 3, ParticleType::kHe3},      {2, 4, ParticleType::kAlpha},
        {6, 12, ParticleType::kGenericIon}, {8, 16, ParticleType::kGenericIon},
        {82, 208, ParticleType::kGenericIon},
    };
    for (const Nuc& nn : nucs) {
      char buf[64];
      std::snprintf(buf, sizeof buf, "(Z=%d, A=%d)", nn.z, nn.a);
      check(particle_type_of_nucleus(nn.z, nn.a) == nn.t,
            std::string("nucleus ") + buf + " -> " + particle_name(nn.t));
      // And through the encoding, which is the path a de-excitation product takes.
      const int code = 1000000000 + nn.z * 10000 + nn.a * 10;
      check(particle_type_of_pdg(code) == nn.t,
            std::string("pdg code for ") + buf + " -> " + particle_name(nn.t));
      // The isomer digit must not change the species: an excited Fe57 is still Fe57.
      check(particle_type_of_pdg(code + 2) == nn.t,
            std::string("isomer digit ignored for ") + buf);
    }
    // A heavy recoil is a TRANSPORTED species as of P8c, and this assertion is the one it
    // inverted. It used to read `== kRefused`, with "that is what makes the energy ledger in
    // EmitterBooks necessary" over it - and the ledger showed 929 GenericIons carrying
    // 515.136 MeV out of `build_all.bat`'s 6000-proton depth-dose gate, which is what the
    // package was written for. It is kept rather than deleted because the DIRECTION is the
    // claim: `species_disposition` is the one function that decides whether a recoil becomes a
    // track or a counter, and it is the only thing standing between `push_nucleus` and a
    // number in a report.
    check(species_disposition(ParticleType::kGenericIon) == SpeciesDisposition::kStepped,
          "a heavy recoil maps to a transported species");
    check(species_disposition(ParticleType::kHe3) == SpeciesDisposition::kStepped,
          "He3 maps to a transported species");
    std::printf("3. the nuclear encoding: %d nuclides, both directions, isomer digit ignored\n",
                int(sizeof(nucs) / sizeof(nucs[0])));
  }

  // ---------------------------------------------------------------- 4. the length distribution
  //
  // The mean of N exponential draws has a standard error of mean/sqrt(N), so the tolerance is
  // arithmetic and not a guess: at N = 200,000 that is 0.22%, and the gate is 4 of them.
  {
    bool ok = false;
    const auto rows = read_csv(dir + "/decay_applicable.csv", ok);
    if (!ok) { return 1; }
    std::map<std::string, double> lifetime;
    for (const auto& r : rows) {
      if (r.size() < 8) { continue; }
      lifetime[r[0]] = num(r[4]);
    }
    const int kN = 200000;
    const real_t kEkin = 200;  // MeV, the energy every ref/b1hadron macro uses
    struct Case { ParticleType t; const char* name; };
    const Case cases[] = {
        {ParticleType::kPionPlus, "pi+"},   {ParticleType::kPionMinus, "pi-"},
        {ParticleType::kKaonPlus, "kaon+"}, {ParticleType::kKaonMinus, "kaon-"},
        {ParticleType::kMuonPlus, "mu+"},   {ParticleType::kMuonMinus, "mu-"},
        {ParticleType::kPiZero, "pi0"},
    };
    std::printf("4. the in-flight length is an exponential with mean beta*gamma*c*tau\n");
    for (const Case& c : cases) {
      const real_t mass = particle_def<real_t>(c.t).mass;
      const real_t want_mfp =
          decay::in_flight_mean_free_path<real_t>(pdg_code(c.t), mass, kEkin);
      // The mean free path itself, from the oracle's lifetime rather than from the port's own
      // table: beta*gamma*c*tau with beta*gamma = p/m.
      const real_t tau = static_cast<real_t>(lifetime[c.name]);
      const real_t p = std::sqrt((kEkin + 2 * mass) * kEkin);
      const real_t expect = p / mass * units::c_light<real_t>() * tau;
      const double dev = std::fabs(want_mfp - expect) / expect;
      if (!(dev <= 1e-14)) {
        std::printf("  FAIL: %s mean free path %.17g, Geant4's lifetime implies %.17g\n",
                    c.name, want_mfp, expect);
        ++g_fails;
      }
      double sum = 0;
      int below = 0;
      for (int i = 0; i < kN; ++i) {
        Philox<real_t> rng(static_cast<unsigned int>(pdg_code(c.t) + 1000000),
                           static_cast<unsigned int>(i), 0xD3CAu);
        const real_t d = had::decay_in_flight_length<real_t>(c.t, mass, kEkin, rng);
        sum += static_cast<double>(d);
        if (d < want_mfp) { ++below; }
      }
      const double mean = sum / kN;
      const double se = static_cast<double>(want_mfp) / std::sqrt(double(kN));
      const double zmean = std::fabs(mean - static_cast<double>(want_mfp)) / se;
      // 1 - 1/e = 0.632120558..., with a binomial standard error of sqrt(p(1-p)/N).
      const double pfrac = double(below) / double(kN);
      const double pwant = 1.0 - std::exp(-1.0);
      const double pse = std::sqrt(pwant * (1.0 - pwant) / double(kN));
      const double zfrac = std::fabs(pfrac - pwant) / pse;
      std::printf("   %-6s mfp %12.6g mm   mean %12.6g (%.2f sigma)   P(<mfp) %.5f "
                  "(%.2f sigma)\n",
                  c.name, double(want_mfp), mean, zmean, pfrac, zfrac);
      check(zmean <= 4.0, std::string(c.name) + ": the sampled mean is the mean free path");
      check(zfrac <= 4.0, std::string(c.name) + ": the fraction inside one mfp is 1 - 1/e");
    }
    // A stable species draws NOTHING, and that is what keeps the proton's B1 dose where it was.
    for (ParticleType t : {ParticleType::kProton, ParticleType::kAlpha, ParticleType::kHe3,
                           ParticleType::kDeuteron, ParticleType::kTriton,
                           ParticleType::kAntiProton}) {
      check(!had::decays_in_flight(t),
            std::string(particle_name(t)) + " draws no decay length");
    }
  }

  // ------------------------------------------------------- 5. the neutron table, one lookup
  //
  // On a REAL table: G4NeutronElasticXS, G4NeutronInelasticXS and G4NeutronCaptureXS out of
  // G4PARTICLEXS4.0, summed onto G4NeutronGeneralProcess's own two grids by P2's
  // `ngp_build_table`, for water and for lead. Synthetic rows would check that two copies of
  // an interpolation agree; what has to be checked is that they agree on the curve the
  // transport will read, at the nodes that curve is defined on.
  {
    namespace pxs = g4gpu::hadronic::xs;
    const std::string dir = host::g4particlexs_subdir("neutron");
    if (dir.empty()) {
      std::printf("5. SKIPPED: no G4PARTICLEXS dataset (set G4PARTICLEXSDATA or "
                  "G4GPU_DATA_DIR)\n");
      check(false, "the G4PARTICLEXS dataset was found");
    } else {
      data::ParticleXsTable<real_t> tab[3];
      pxs::PxsDataSet<real_t> ds[3];
      const pxs::PxsKind kinds[3] = {pxs::PxsKind::kNeutronElastic,
                                     pxs::PxsKind::kNeutronInelastic,
                                     pxs::PxsKind::kNeutronCapture};
      bool loaded = true;
      for (int i = 0; i < 3; ++i) {
        if (!pxs::pxs_load<real_t>(kinds[i], pxs::neutron<real_t>(), dir, tab[i], ds[i])) {
          std::printf("5. FAIL: could not load dataset %d from %s\n", i, dir.c_str());
          ++g_fails;
          loaded = false;
        }
      }
      if (loaded) {
        // Two materials, built here rather than read from an oracle CSV: the atom densities
        // are nominal (water at 1 g/cm^3, lead at 11.35) and only scale table 0, while the
        // three PARTITIONS - tables 1, 2 and 4 - are ratios and do not depend on them at all.
        // What matters is that the two have different element counts, because a one-element
        // material and a two-element one take different branches everywhere downstream.
        std::vector<data::Material<real_t>> mats(2);
        mats[0].n_elements = 2;                      // water
        mats[0].z[0] = 1;  mats[0].n_atoms[0] = 6.6866e19;
        mats[0].z[1] = 8;  mats[0].n_atoms[1] = 3.3433e19;
        mats[1].n_elements = 1;                      // lead
        mats[1].z[0] = 82; mats[1].n_atoms[0] = 3.2991e19;

        pxs::NeutronGeneralTable<real_t> ngt;
        pxs::ngp_build_table<real_t>(ds[0], ds[1], ds[2], mats.data(), 2, ngt);
        const had::NeutronGeneralXs<real_t> sock = had::neutron_general_view<real_t>(ngt);

        // (a) The constants. P1's socket declares the grid and P2's builder declares it again;
        // they are not shared, so they are asserted equal instead. A silent divergence here
        // would put the two zones' boundary in different places in the socket and the table.
        check(had::kNeutronXsEMin<real_t>() == pxs::ngp_min_energy<real_t>(),
              "kNeutronXsEMin == ngp_min_energy");
        check(had::kNeutronXsEMiddle<real_t>() == pxs::ngp_middle_energy<real_t>(),
              "kNeutronXsEMiddle == ngp_middle_energy");
        check(had::kNeutronXsEMax<real_t>() == pxs::ngp_max_energy<real_t>(),
              "kNeutronXsEMax == ngp_max_energy");
        check(had::kNeutronXsLowBins == pxs::ngp_n_low_bins(), "400 low bins, both files");
        check(had::kNeutronXsHighBins == pxs::ngp_n_high_bins(), "70 high bins, both files");
        check(had::kNeutronXsLowNodes == ngt.n_low, "401 low nodes");
        check(had::kNeutronXsHighNodes == ngt.n_high, "71 high nodes");
        check(had::kNeutronTimeLimit<real_t>() == pxs::ngp_time_limit<real_t>(),
              "the 10 us time limit, both files");

        // (b) The socket against P2's own evaluation, BITWISE, at every node and every bin
        // midpoint of both zones and both materials. Bitwise is the right tolerance: after the
        // unification they are the same call, so anything but equality is a wiring mistake.
        int n_pts = 0, n_diff = 0;
        double worst_sock = 0;
        // (c) And the formula the socket used BEFORE - the anti-vacuity measurement. Same
        // rows, same energies, the difference reported rather than asserted, because it is the
        // size of what was being read instead of the validated table.
        double worst_old_low = 0, worst_old_high = 0;
        for (int m = 0; m < 2; ++m) {
          for (int zone = 0; zone < 2; ++zone) {
            const bool low = (zone == 0);
            const int n = low ? ngt.n_low : ngt.n_high;
            const std::vector<real_t>& grid = low ? ngt.e_low : ngt.e_high;
            const std::vector<real_t>& row = low ? ngt.t0 : ngt.t3;
            const real_t e_min = low ? pxs::ngp_min_energy<real_t>()
                                     : pxs::ngp_middle_energy<real_t>();
            const real_t e_max = low ? pxs::ngp_middle_energy<real_t>()
                                     : pxs::ngp_max_energy<real_t>();
            const real_t* raw = row.data() + static_cast<std::size_t>(m) * n;
            for (int j = 0; j < 2 * n - 1; ++j) {
              // Even j is node j/2; odd j is the midpoint IN ENERGY of bin j/2, which is where
              // an interpolation done in the wrong variable is furthest from the right answer.
              real_t e;
              if ((j % 2) == 0) {
                e = grid[static_cast<std::size_t>(j / 2)];
              } else {
                const int b = j / 2;
                e = real_t(0.5) * (grid[static_cast<std::size_t>(b)]
                                   + grid[static_cast<std::size_t>(b + 1)]);
              }
              const real_t loge = std::log(e);
              const real_t got = sock.total(m, e, loge);
              const real_t want = pxs::ngp_lambda<real_t>(ngt, m, e, loge);
              ++n_pts;
              if (got != want) {
                ++n_diff;
                if (want != 0) {
                  worst_sock = std::fmax(worst_sock, std::fabs(got - want) / std::fabs(want));
                }
              }
              const real_t old = had::NeutronGeneralXs<real_t>::log_vector_value(
                  raw, n, e_min, e_max, e);
              if (want != 0) {
                const double d = std::fabs(old - want) / std::fabs(want);
                if (low) { worst_old_low = std::fmax(worst_old_low, d); }
                else     { worst_old_high = std::fmax(worst_old_high, d); }
              }
            }
          }
        }
        std::printf("5. the neutron table: %d points over 2 materials and both zones\n"
                    "   socket vs P2's ngp_lambda: %d differ, worst %.3g relative\n"
                    "   the formula the socket used BEFORE the unification (recomputed nodes,\n"
                    "   log(a)-log(b) for log(a/b)): worst %.3g low zone, %.3g high zone\n",
                    n_pts, n_diff, worst_sock, worst_old_low, worst_old_high);
        check(n_diff == 0, "the socket and P2's table are one lookup, bitwise");
        // The old formula must actually have DIFFERED, or the measurement above is vacuous and
        // the unification fixed nothing. It is a claim about the transcription, so it is
        // asserted in the direction that can fail if someone "tidies" one of the two.
        check(worst_old_low > 0 || worst_old_high > 0,
              "the pre-unification formula really did give a different double");

        // (d) The sub-process choice, against `ngp_select_subprocess` - the same structure
        // with the port's enum on the front. Every node, both zones, and the q values that
        // straddle each partition rather than a uniform sweep: q exactly AT a partition must
        // choose the lower branch, because PostStepDoIt tests `q <= p`.
        int n_sel = 0, n_sel_diff = 0;
        for (int m = 0; m < 2; ++m) {
          for (int zone = 0; zone < 2; ++zone) {
            const bool low = (zone == 0);
            const int n = low ? ngt.n_low : ngt.n_high;
            const std::vector<real_t>& grid = low ? ngt.e_low : ngt.e_high;
            for (int j = 0; j < n; ++j) {
              const real_t e = grid[static_cast<std::size_t>(j)];
              const real_t loge = std::log(e);
              const real_t p1 = low ? pxs::phys_vec_log_value(ngt.p1[m], e, loge)
                                    : pxs::phys_vec_log_value(ngt.p4[m], e, loge);
              const real_t p2 = low ? pxs::phys_vec_log_value(ngt.p2[m], e, loge) : p1;
              const real_t qs[6] = {real_t(0), p1, std::nextafter(p1, real_t(1)),
                                    p2, std::nextafter(p2, real_t(1)),
                                    std::nextafter(real_t(1), real_t(0))};
              for (real_t q : qs) {
                const int want = pxs::ngp_select_subprocess<real_t>(ngt, m, e, loge, q);
                const int got = static_cast<int>(sock.select(m, e, loge, q));
                ++n_sel;
                if (got != want) {
                  ++n_sel_diff;
                  if (n_sel_diff <= 3) {
                    std::printf("  FAIL: mat %d, %s zone, e %.17g, q %.17g: socket %d, "
                                "P2 %d\n", m, low ? "low" : "high", double(e), double(q), got,
                                want);
                  }
                  ++g_fails;
                }
              }
            }
          }
        }
        std::printf("   sub-process choice: %d (energy, q) points, %d disagree\n", n_sel,
                    n_sel_diff);
        // And the ORDER SWAP between the zones, which is the one thing about this selection
        // that is not symmetric and the one a rewrite gets wrong. Below the middle energy the
        // first partial is ELASTIC; above it the first partial is INELASTIC. Asserted with a
        // q just below each first partial, in each zone, on lead - where inelastic and elastic
        // are far enough apart that a swap could not pass by luck.
        {
          const real_t e_low = real_t(1);                       // 1 MeV, in the low zone
          const real_t e_high = real_t(100);                    // 100 MeV, in the high zone
          const real_t q_tiny = real_t(1e-12);
          check(sock.select(1, e_low, std::log(e_low), q_tiny)
                    == had::NeutronSubProcess::kElastic,
                "below 20 MeV the first partial is elastic");
          check(sock.select(1, e_high, std::log(e_high), q_tiny)
                    == had::NeutronSubProcess::kInelastic,
                "above 20 MeV the first partial is inelastic");
          // Capture exists only below the middle energy: table 3 is elastic + inelastic, so no
          // q can select capture in the high zone.
          bool capture_high = false;
          for (int i = 0; i <= 100; ++i) {
            const real_t q = real_t(i) / real_t(101);
            if (sock.select(1, e_high, std::log(e_high), q)
                == had::NeutronSubProcess::kCapture) {
              capture_high = true;
            }
          }
          check(!capture_high, "no q selects capture above 20 MeV");
        }

        // (e) SELECTION FREQUENCIES against the partials, with a deterministic seed.
        //
        // The checks above are about the boundaries; this one is about the measure. Draw one
        // uniform per trial from Philox, select, and compare the counted fraction of each
        // sub-process against the partial the table says: p_el, p_ei - p_el, 1 - p_ei below
        // the middle energy and p_inel, 1 - p_inel above it. The standard error of a counted
        // fraction is sqrt(p(1-p)/N), so the tolerance is arithmetic rather than chosen.
        //
        // What this catches that a boundary check cannot: a selection that draws TWO uniforms
        // (which would still respect every boundary and would desynchronise the stream), and
        // a cumulative partial read as a differential one - `p_ei` used where `p_ei - p_el`
        // was meant puts inelastic 30-40 points high at 1 MeV in lead and passes every
        // `q <= p` test above.
        {
          const int kN = 200000;
          const real_t energies[4] = {real_t(1e-3), real_t(1), real_t(19), real_t(100)};
          std::printf("   selection frequencies, %d draws per point:\n", kN);
          double worst_z = 0;
          for (int m = 0; m < 2; ++m) {
            for (real_t e : energies) {
              const real_t loge = std::log(e);
              const bool low = (e <= pxs::ngp_middle_energy<real_t>());
              int n[3] = {0, 0, 0};
              for (int i = 0; i < kN; ++i) {
                Philox<real_t> rng(static_cast<unsigned int>(0xB0B0u + m),
                                   static_cast<unsigned int>(i), 0x5EEDu);
                ++n[static_cast<int>(sock.select(m, e, loge, rng.uniform()))];
              }
              // The expected fractions, DIFFERENTIAL, from the cumulative table.
              real_t want[3] = {real_t(0), real_t(0), real_t(0)};
              if (low) {
                const real_t p_el = pxs::phys_vec_log_value(ngt.p1[m], e, loge);
                const real_t p_ei = pxs::phys_vec_log_value(ngt.p2[m], e, loge);
                want[0] = p_el;
                want[1] = p_ei - p_el;
                want[2] = real_t(1) - p_ei;
              } else {
                const real_t p_in = pxs::phys_vec_log_value(ngt.p4[m], e, loge);
                want[0] = real_t(1) - p_in;
                want[1] = p_in;
                want[2] = real_t(0);
              }
              double zmax = 0;
              for (int k = 0; k < 3; ++k) {
                const double p = static_cast<double>(want[k]);
                const double f = double(n[k]) / double(kN);
                if (p <= 0.0) {
                  // A sub-process with zero probability must be selected exactly never; there
                  // is no standard error to divide by and "almost never" is not the claim.
                  if (n[k] != 0) {
                    std::printf("  FAIL: mat %d at %.6g MeV: sub-process %d has p = 0 and was "
                                "selected %d times\n", m, double(e), k, n[k]);
                    ++g_fails;
                  }
                  continue;
                }
                const double se = std::sqrt(p * (1.0 - p) / double(kN));
                zmax = std::fmax(zmax, std::fabs(f - p) / se);
              }
              worst_z = std::fmax(worst_z, zmax);
              std::printf("     mat %d, %8.4g MeV: el %.5f in %.5f cap %.5f   want %.5f "
                          "%.5f %.5f   worst %.2f sigma\n",
                          m, double(e), double(n[0]) / kN, double(n[1]) / kN,
                          double(n[2]) / kN, double(want[0]), double(want[1]),
                          double(want[2]), zmax);
              check(zmax <= 4.0, "the selection frequencies are the table's partials");
            }
          }
          std::printf("     worst over 8 points and 3 sub-processes: %.2f sigma\n", worst_z);
        }
      }
    }
  }

  std::printf("\n%s (%d failures)\n", (g_fails == 0) ? "PASSED" : "FAILED", g_fails);
  return (g_fails == 0) ? 0 : 1;
}
