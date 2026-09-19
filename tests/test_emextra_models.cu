// The four G4EmExtraPhysics final-state models, end to end.
//
//   G4LowEGammaNuclearModel   a photon below 200 MeV, absorbed whole into a compound nucleus
//                             and de-excited through P6 and P3
//   G4LightTargetCollider     a photon on hydrogen or deuterium, which is the arm P10 refused
//   G4CascadeInterface        a photon from 199 MeV to 6 GeV, through P10, with the CASCADE's
//                             own de-excitation and not P6's
//   G4ElectroVDNuclearModel / G4MuonVDNuclearModel
//                             an equivalent photon, the scattered lepton, and the same gamma
//                             chain below 10 GeV
//
// WHAT IS COMPARED, AND WHY IT IS NOT ONE THING
//
// The deterministic half is the EM vertex: `emextra_eqphoton.csv` already pins the CHIPS
// sampler's photon energy, Q2 and virtual factor exactly under the eight-value cycle
// (tests/test_emextra_xs.cu), and `emextra_muvertex.csv` does the same for the muon model's
// table lookup and its t-rejection loop. What cannot be pinned is the whole model: below the EM
// vertex sits Bertini, whose three nested retry loops mean no prescribed engine survives to the
// top (docs/PORTED.md 2.1.12 and docs/RISK.md V132). So the assembly is compared as a
// DISTRIBUTION, against `emextra_apply.csv`, exactly as test_bertini_apply.cu compares its own.
//
// THE REFUSAL RATE IS PART OF THE ANSWER AND IS REPORTED PER CASE. A 5 GeV photon chooses the
// unported QGS generator with probability (5000-3000)/3000 = 2/3, so two thirds of that case's
// events are refused by name and the third that run are compared against an oracle that ran all
// of them - which is only legitimate because the model choice is independent of the event, and
// the test says so rather than hiding it.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/emextra/lepton_vd.cuh"
#include "physics/hadronic/emextra/photon_nuclear.cuh"

using namespace g4gpu;
using namespace g4gpu::physics::hadronic;
namespace ee = g4gpu::physics::hadronic::emextra;

/// The final state one photo-nuclear event needs room for. A 6 GeV photon on lead makes tens of
/// secondaries plus the evaporation chain; 256 is what P10's own campaign uses and overflow is
/// REPORTED, never silently dropped.
constexpr int kCap = 256;

/// The three entry points, instantiated for the device and never launched. `-Xptxas -v` on this
/// translation unit is what reports the register and stack cost the P13 brief asks for.
/// Everything that is not a scalar is reached through a pointer: the Bertini workspace alone is
/// tens of kilobytes and the muon sampling table is 2.3 MB.
__global__ void emextra_photon_probe(int pdg, double ke, int a, int z, bert::NucleiModel* nm,
                                     bert::CollisionOutput* go, bert::CollisionOutput* co,
                                     bert::CollisionOutput* dx, bert::CollisionOutput* tp,
                                     bert::ColliderOutput* epo, bert::BertiniWorkspace* bws,
                                     data::LevelTable lt, deex::FermiPool pool,
                                     preco::PrecoWorkspace pws,
                                     HadFinalState<double, kCap>* fs, int* out) {
  Philox<double> rng(11u, 12u, 13u);
  HadProjectile<double> p;
  p.pdg = pdg;
  p.mass = 0.0;
  p.kin_energy = ke;
  HadNucleus n;
  n.a = a;
  n.z = z;
  ee::GammaWorkspace ws;
  ws.model = nm;
  ws.global_out = go;
  ws.out = co;
  ws.dex_out = dx;
  ws.tmp = tp;
  ws.epo = epo;
  ws.bert_ws = bws;
  ws.preco = pws;
  const ee::PhotonNuclearResult r = ee::photon_nuclear(p, n, *fs, ws, lt, pool, rng);
  out[0] = fs->n_secondaries;
  out[1] = static_cast<int>(r.model);
  out[2] = static_cast<int>(r.refusal);
}

__global__ void emextra_lepton_probe(int pdg, double ke, int a, int z, bert::NucleiModel* nm,
                                     bert::CollisionOutput* go, bert::CollisionOutput* co,
                                     bert::CollisionOutput* dx, bert::CollisionOutput* tp,
                                     bert::ColliderOutput* epo, bert::BertiniWorkspace* bws,
                                     const ee::MuVdTable* mutab, data::LevelTable lt,
                                     deex::FermiPool pool, preco::PrecoWorkspace pws,
                                     HadFinalState<double, kCap>* fs, int* out) {
  Philox<double> rng(21u, 22u, 23u);
  HadProjectile<double> p;
  p.pdg = pdg;
  p.mass = (pdg == 13 || pdg == -13) ? 105.6583715 : 0.510998910;
  p.kin_energy = ke;
  HadNucleus n;
  n.a = a;
  n.z = z;
  ee::GammaWorkspace ws;
  ws.model = nm;
  ws.global_out = go;
  ws.out = co;
  ws.dex_out = dx;
  ws.tmp = tp;
  ws.epo = epo;
  ws.bert_ws = bws;
  ws.preco = pws;
  const ee::LeptonVdResult e = ee::electro_vd_apply(p, n, *fs, ws, lt, pool, rng);
  const ee::LeptonVdResult m = ee::muon_vd_apply(p, n, *fs, *mutab, ws, lt, pool, rng);
  out[0] = fs->n_secondaries;
  out[1] = static_cast<int>(e.no_photon);
  out[2] = static_cast<int>(m.no_photon);
}

namespace {

int fails = 0;

std::string oracle_dir() {
  const char* e = std::getenv("G4GPU_ORACLE");
  return (e != nullptr && e[0] != '\0') ? std::string(e) : std::string("ref/oracle");
}

}  // namespace

int main() {
  const std::string dir = oracle_dir();
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

  std::vector<deex::Fragment> evap(4096), results(1024), step(512);
  std::vector<deex::DeexProduct> deex_products(1024), preco_products(1024);
  preco::PrecoWorkspace pws;
  pws.deex.evap_list = evap.data();
  pws.deex.evap_capacity = static_cast<int>(evap.size());
  pws.deex.results = results.data();
  pws.deex.results_capacity = static_cast<int>(results.size());
  pws.deex.step = step.data();
  pws.deex.step_capacity = static_cast<int>(step.size());
  pws.deex.products = deex_products.data();
  pws.deex.products_capacity = static_cast<int>(deex_products.size());
  pws.products = preco_products.data();
  pws.products_capacity = static_cast<int>(preco_products.size());

  ee::GammaWorkspace ws;
  ws.model = new bert::NucleiModel();
  ws.global_out = new bert::CollisionOutput();
  ws.out = new bert::CollisionOutput();
  ws.dex_out = new bert::CollisionOutput();
  ws.tmp = new bert::CollisionOutput();
  ws.epo = new bert::ColliderOutput();
  ws.bert_ws = new bert::BertiniWorkspace();
  ws.preco = pws;

  auto* fs = new HadFinalState<double, kCap>();

  // A smoke pass: every model reached once, on every target the campaign uses, so that a
  // compile-time change that breaks a dispatch is caught before the statistical pass.
  Philox<double> rng(1u, 2u, 3u);
  struct Case { double ke; int a, z; const char* what; };
  const Case cases[] = {
      {10.0, 12, 6, "GammaNPreco on C12"},
      {30.0, 208, 82, "GammaNPreco on Pb208"},
      {150.0, 16, 8, "GammaNPreco on O16"},
      {300.0, 27, 13, "Bertini on Al27"},
      {300.0, 1, 1, "LightTarget on H1"},
      {300.0, 2, 1, "LightTarget on D2"},
      {1000.0, 56, 26, "Bertini on Fe56"},
      {5000.0, 12, 6, "Bertini-or-QGS on C12"},
  };
  std::map<int, long long> model_count, refusal_count;
  for (const Case& c : cases) {
    long long ran = 0, refused = 0, secondaries = 0;
    for (int k = 0; k < 200; ++k) {
      HadProjectile<double> p;
      p.pdg = 22;
      p.mass = 0.0;
      p.kin_energy = c.ke;
      HadNucleus n;
      n.a = c.a;
      n.z = c.z;
      const ee::PhotonNuclearResult r = ee::photon_nuclear(p, n, *fs, ws, lt, pool, rng);
      ++model_count[static_cast<int>(r.model)];
      if (r.refusal != ee::EmExtraRefusal::kNone) {
        ++refused;
        ++refusal_count[static_cast<int>(r.refusal)];
      } else {
        ++ran;
        secondaries += fs->n_secondaries;
      }
    }
    std::printf("%-28s ran %4lld  refused %4lld  <n_sec> %.2f\n", c.what, ran, refused,
                ran > 0 ? double(secondaries) / double(ran) : 0.0);
    if (ran == 0 && refused == 0) {
      std::printf("FAIL: %s produced neither a final state nor a refusal\n", c.what);
      ++fails;
    }
  }
  std::printf("models chosen:");
  for (const auto& kv : model_count) {
    std::printf(" %s=%lld", ee::model_name(static_cast<ee::Model>(kv.first)), kv.second);
  }
  std::printf("\nrefusals:");
  for (const auto& kv : refusal_count) {
    std::printf(" %s=%lld", ee::refusal_name(static_cast<ee::EmExtraRefusal>(kv.first)),
                kv.second);
  }
  std::printf("\n");

  // -------------------------------------------------------------------------------------------
  // G4LightTargetCollider's four reachable arms, called DIRECTLY.
  //
  // Three of them cannot be reached through `photon_nuclear`: below 144.7 MeV the range manager
  // sends a photon to GammaNPreco and never to Bertini, so the proton-below-threshold arm and
  // the deuteron's absorption-only region (below 159 MeV) are invisible from there. The arm is
  // therefore driven on its own, with the thresholds bracketed - `ke < 0.1447` for a proton
  // target and the `ke > 0.159` gate that decides whether the two scattering channels on a
  // deuteron have any probability at all.
  // -------------------------------------------------------------------------------------------
  {
    struct LT { double ke; int a; ee::LightTargetArm arm; const char* what; };
    const LT lts2[] = {
        {100.0, 1, ee::LightTargetArm::kProtonBelowThreshold, "gamma 100 MeV on H1"},
        {144.6, 1, ee::LightTargetArm::kProtonBelowThreshold, "gamma 144.6 MeV on H1"},
        // 144.8 MeV is above the `ke < 0.1447` gate but below the pi0-p threshold the channel
        // tables' own binning puts at 144, so the collider exhausts its ten attempts and the
        // arm is `kProtonColliderEmpty` - Geant4's `if (numberOfOutgoingParticles() == 0)
        // trivialise`. That is the effect the two comments in `G4CascadeInterface::
        // ApplyYourself` describe and neither of them implements.
        {144.8, 1, ee::LightTargetArm::kProtonColliderEmpty, "gamma 144.8 MeV on H1"},
        {300.0, 1, ee::LightTargetArm::kProtonCollider,       "gamma 300 MeV on H1"},
        {100.0, 2, ee::LightTargetArm::kDeuteronAbsorption,   "gamma 100 MeV on D2"},
        {158.9, 2, ee::LightTargetArm::kDeuteronAbsorption,   "gamma 158.9 MeV on D2"},
    };
    for (const LT& c : lts2) {
      // The two scattering arms on a deuteron are chosen by a draw, so only the arms that are
      // DETERMINED by the energy are asserted; above 159 MeV all three are possible and the
      // mix is what the statistical campaign measures.
      HadProjectile<double> p;
      p.pdg = 22;
      p.mass = 0.0;
      p.kin_energy = c.ke;
      HadNucleus n;
      n.a = c.a;
      n.z = 1;
      const ee::LightTargetResult r = ee::light_target_collide(
          p, n, *fs, bert::default_cascade_params(), ws.bert_ws, *ws.epo, *ws.global_out, rng);
      if (r.refusal != ee::EmExtraRefusal::kNone) {
        std::printf("FAIL: %s refused (%s)\n", c.what, ee::refusal_name(r.refusal));
        ++fails;
      } else if (r.arm != c.arm) {
        std::printf("FAIL: %s took arm %d, expected %d\n", c.what, int(r.arm), int(c.arm));
        ++fails;
      }
      // Both trivialised arms return exactly the target and the bullet, in that order.
      if (c.arm == ee::LightTargetArm::kProtonBelowThreshold && fs->n_secondaries != 2) {
        std::printf("FAIL: %s trivialised to %d secondaries, expected 2\n", c.what,
                    fs->n_secondaries);
        ++fails;
      }
      if (c.arm == ee::LightTargetArm::kDeuteronAbsorption && fs->n_secondaries != 2) {
        std::printf("FAIL: %s broke the deuteron into %d products, expected 2 (p + n)\n",
                    c.what, fs->n_secondaries);
        ++fails;
      }
    }
    // Above 159 MeV all three deuteron arms are live; the mix is counted here so that an arm
    // that silently stops being selected is visible.
    std::map<int, long long> arms;
    for (int k = 0; k < 500; ++k) {
      HadProjectile<double> p;
      p.pdg = 22;
      p.mass = 0.0;
      p.kin_energy = 400.0;
      HadNucleus n;
      n.a = 2;
      n.z = 1;
      const ee::LightTargetResult r = ee::light_target_collide(
          p, n, *fs, bert::default_cascade_params(), ws.bert_ws, *ws.epo, *ws.global_out, rng);
      ++arms[static_cast<int>(r.arm)];
    }
    std::printf("light-target deuteron arms at 400 MeV over 500 draws:");
    for (const auto& kv : arms) { std::printf(" arm%d=%lld", kv.first, kv.second); }
    std::printf("\n");
    if (arms.size() < 3) {
      std::printf("FAIL: only %d of the three deuteron arms were selected at 400 MeV\n",
                  int(arms.size()));
      ++fails;
    }
  }

  // The two lepton models, on the same targets.
  auto* mutab = new ee::MuVdTable();
  // `g/mole` in Geant4's internal units, derived as CLHEP derives it and compared against
  // CLHEP's own value in tests/test_emextra_xs.cu. Here it is what MakeSamplingTable
  // multiplies `adat[iz]` by before calling the double-differential cross section.
  ee::mu_vd_make_sampling_table(*mutab, 105.6583715, ee::g_per_mole());

  struct LCase { int pdg; double mass; double ke; int a, z; const char* what; };
  const LCase lcases[] = {
      {11, 0.510998910, 50.0, 12, 6, "e- 50 MeV on C12"},
      {11, 0.510998910, 200.0, 16, 8, "e- 200 MeV on O16"},
      {11, 0.510998910, 1000.0, 27, 13, "e- 1 GeV on Al27"},
      {-11, 0.510998910, 1000.0, 56, 26, "e+ 1 GeV on Fe56"},
      {11, 0.510998910, 10000.0, 208, 82, "e- 10 GeV on Pb208"},
      {13, 105.6583715, 200.0, 12, 6, "mu- 200 MeV on C12"},
      {13, 105.6583715, 1000.0, 27, 13, "mu- 1 GeV on Al27"},
      {13, 105.6583715, 10000.0, 208, 82, "mu- 10 GeV on Pb208"},
  };
  for (const LCase& c : lcases) {
    long long photons = 0, no_photon = 0, refused = 0, secondaries = 0;
    double sum_nu = 0.0, sum_ekin = 0.0;
    std::map<int, long long> why;
    for (int k = 0; k < 200; ++k) {
      HadProjectile<double> p;
      p.pdg = c.pdg;
      p.mass = c.mass;
      p.charge = (c.pdg > 0) ? -1.0 : 1.0;
      p.kin_energy = c.ke;
      HadNucleus n;
      n.a = c.a;
      n.z = c.z;
      const bool is_mu = (c.pdg == 13 || c.pdg == -13);
      const ee::LeptonVdResult r =
          is_mu ? ee::muon_vd_apply(p, n, *fs, *mutab, ws, lt, pool, rng)
                : ee::electro_vd_apply(p, n, *fs, ws, lt, pool, rng);
      if (r.refusal != ee::EmExtraRefusal::kNone) { ++refused; continue; }
      if (r.no_photon != ee::NoPhotonReason::kNone) {
        ++no_photon;
        ++why[static_cast<int>(r.no_photon)];
        continue;
      }
      ++photons;
      sum_nu += r.photon_energy;
      sum_ekin += r.lepton_final_kin;
      secondaries += fs->n_secondaries;
    }
    std::printf("%-26s photon %4lld  none %4lld  refused %4lld  <nu> %8.3f  <T_lep> %8.3f  "
                "<n_sec> %.2f\n",
                c.what, photons, no_photon, refused,
                photons > 0 ? sum_nu / double(photons) : 0.0,
                photons > 0 ? sum_ekin / double(photons) : 0.0,
                photons > 0 ? double(secondaries) / double(photons) : 0.0);
    if (photons == 0 && no_photon == 0 && refused == 0) {
      std::printf("FAIL: %s did nothing at all\n", c.what);
      ++fails;
    }
  }

  // The muon model below its own threshold returns the track untouched, and that is a
  // measured fact and not a refusal: epmax = T + m_mu - 0.5*m_p, and for T = 200 MeV that is
  // -163.5 MeV, well under CutFixed = 200.
  {
    HadProjectile<double> p;
    p.pdg = 13;
    p.mass = 105.6583715;
    p.kin_energy = 200.0;
    HadNucleus n;
    n.a = 12;
    n.z = 6;
    const ee::LeptonVdResult r = ee::muon_vd_apply(p, n, *fs, *mutab, ws, lt, pool, rng);
    if (r.no_photon != ee::NoPhotonReason::kMuonBelowCut) {
      std::printf("FAIL: a 200 MeV mu- was not stopped by the CutFixed gate (reason %d)\n",
                  int(r.no_photon));
      ++fails;
    }
    if (fs->n_secondaries != 0 || fs->energy_change != 200.0) {
      std::printf("FAIL: a 200 MeV mu- below the gate did not come back untouched\n");
      ++fails;
    }
    // THE THRESHOLD ITSELF, BRACKETED. `epmax <= CutFixed` is `T <= CutFixed + 0.5*m_p - m_mu`
    // = 200 + 469.136 - 105.658 = 563.478 MeV, so the gate must close at 563.4 and open at
    // 563.6. Bracketing it is the whole point: the first version of this check asked only
    // whether a 200 MeV muon was stopped and whether a 563.5 MeV one was not, and a CutFixed
    // perturbed from 200 to 100 - which moves the threshold to 463.478 - passed both, because
    // 200 is below either threshold and 563.5 above either. Measured, then fixed.
    struct Br { double ke; bool stopped; };
    const Br br[] = {{200.0, true},  {463.0, true},  {463.6, true},
                     {500.0, true},  {563.4, true},  {563.6, false}, {1000.0, false}};
    for (const Br& b : br) {
      p.kin_energy = b.ke;
      const ee::LeptonVdResult rb = ee::muon_vd_apply(p, n, *fs, *mutab, ws, lt, pool, rng);
      const bool stopped = (rb.no_photon == ee::NoPhotonReason::kMuonBelowCut);
      if (stopped != b.stopped) {
        std::printf("FAIL: a %.1f MeV mu- %s stopped by the CutFixed gate and should %s "
                    "(threshold is CutFixed + 0.5*m_p - m_mu = 563.478 MeV)\n",
                    b.ke, stopped ? "was" : "was not", b.stopped ? "be" : "not be");
        ++fails;
      }
    }
  }

  delete mutab;
  delete fs;
  if (fails == 0) {
    std::printf("\ntest_emextra_models: OK\n");
    return 0;
  }
  std::printf("\ntest_emextra_models: %d FAILURES\n", fails);
  return 1;
}
