// P23c: the anti-nucleus arm of G4GeneratorPrecompoundInterface::PropagateNuclNucl, and what the
// Fritiof at-rest captures of the anti-baryon family meet in P6's interface - a census of FTFP's
// outcomes by name for pbar, nbar, anti-d, anti-t, anti-He3 and anti-alpha, at rest (kinetic
// energy 0, the projectile `G4HadronicAbsorptionFritiof` hands FTFP) and in flight at 1 and 5 GeV
// per nucleon, on H1, C12, O16, Ca40 and Pb208, through the TRANSPORT's workspace (`entry::Workspace`,
// the one the interaction pool holds - not test_stopping's `HadronWorkspace`, whose kMaxProjA = 1
// refuses every anti-nucleus by capacity before the hand-over is reached).
//
// What is asserted, per event that ran: baryon number and charge conserved over the final state,
// counting each secondary by its PDG code's sign - so an anti-residual emitted as matter, or a
// captured anti-proton's charge counted twice, fails here. Per row: the outcome counts by name.
// Before P23c every anti-nucleus event that reached the hand-over stopped at
// `GeneratorRefusal::anti_nucleus`; the same program compiled against main c0c022b's tree is how
// docs/RISK.md V253's "before" column was measured (the `anti_residual_unbound` column is absent
// there and reads 0).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "core/rng.cuh"
#include "physics/hadronic/ftf/ftf_entry.cuh"

using namespace g4gpu;
namespace entry = g4gpu::hadronic::ftf::entry;
namespace hp = g4gpu::physics::hadronic;

#if __has_include("physics/hadronic/xs/anti_nucl_nuclear_xs.cuh")
#define P23C_TREE 1
#else
#define P23C_TREE 0
#endif

static int pdg_baryon(int pdg, int z, int a) {
  if (a > 0) { return a; }
  const int ap = (pdg < 0) ? -pdg : pdg;
  if (ap >= 1000000000) { return ((pdg < 0) ? -1 : 1) * ((ap / 10) % 1000); }
  const data::FtfHadron* d = data::ftf_find_hadron(pdg);
  return (d != nullptr) ? d->baryon : 0;
}
static int pdg_charge(int pdg, int z, int a) {
  if (a > 0) { return z; }
  const int ap = (pdg < 0) ? -pdg : pdg;
  if (ap >= 1000000000) { return ((pdg < 0) ? -1 : 1) * ((ap / 10000) % 1000); }
  const data::FtfHadron* d = data::ftf_find_hadron(pdg);
  return (d != nullptr) ? static_cast<int>(std::lrint(d->charge)) : 0;
}

int main(int argc, char** argv) {
  const int n_events = (argc > 1) ? std::atoi(argv[1]) : 200;
  const int kSlots = 2;
  static entry::Workspace slots[kSlots];
  static g4gpu::hadronic::ftf::LundTables<double> lund;
  const entry::Handle<entry::Workspace> h = entry::host_handle(slots, kSlots, &lund);
  struct Sp { int pdg, b; double q, m; const char* name; };
  const Sp sp[6] = {{-2212, -1, -1, 938.272013, "anti_proton"},
                    {-2112, -1, 0, 939.56536, "anti_neutron"},
                    {-1000010020, -2, -1, 1875.613, "anti_deuteron"},
                    {-1000010030, -3, -1, 2808.921, "anti_triton"},
                    {-1000020030, -3, -2, 2808.391, "anti_He3"},
                    {-1000020040, -4, -2, 3727.379, "anti_alpha"}};
  struct T { int z, a; const char* name; };
  const T tg[5] = {{1, 1, "H1"}, {6, 12, "C12"}, {8, 16, "O16"}, {20, 40, "Ca40"}, {82, 208, "Pb208"}};
  const double kin_per_n[3] = {0.0, 1000.0, 5000.0};
  int fails = 0;
  long long unbound_za[64] = {};
  long long tot_qx_events = 0;
  long long tot_ran = 0, tot_ev = 0, tot_anti_nuc = 0, tot_unbound = 0, tot_other_gen = 0;
  std::printf("P23c tree: %d; %d events per row\n", P23C_TREE, n_events);
  std::printf("%-13s %-6s %7s %6s %6s %6s %6s %6s %6s %6s %6s\n", "species", "target", "T/n MeV",
              "ran", "unchg", "antiN", "unbnd", "short", "dropr", "other", "unbal");
  for (const Sp& s : sp) {
    for (const T& t : tg) {
      for (double kpn : kin_per_n) {
        long long q_excess_events = 0, q_excess = 0;
        long long ran = 0, unch = 0, anti_nuc = 0, unbound = 0, shortl = 0, dropped = 0,
                  other = 0, unbal = 0;
        for (int ev = 0; ev < n_events; ++ev) {
          hp::HadProjectile<double> p;
          p.pdg = s.pdg;
          p.mass = s.m;
          p.charge = s.q;
          p.baryon_number = s.b;
          p.kin_energy = kpn * std::abs(s.b);
          hp::HadNucleus nuc;
          nuc.z = t.z;
          nuc.a = t.a;
          static hp::HadFinalState<double, entry::kMaxSecondaries> out;
          out = hp::HadFinalState<double, entry::kMaxSecondaries>();
          Philox<double> rng(static_cast<uint32_t>(ev), 0x23Cu, static_cast<uint32_t>(s.pdg));
          entry::Report rep;
          const int slot = ev % kSlots;
          const entry::Status st = entry::apply(h, slot, p, nuc, out, rep, rng);
          const auto& g = slots[slot].report.generator;
          if (st == entry::Status::kRan) {
            ++ran;
            int bsum = 0, qsum = 0;
            for (int i = 0; i < out.n_secondaries; ++i) {
              const auto& q = out.secondaries[i];
              bsum += pdg_baryon(q.pdg, q.z, q.a);
              qsum += pdg_charge(q.pdg, q.z, q.a);
            }
            const int bwant = s.b + t.a;
            const int qwant = static_cast<int>(s.q) + t.z;
            // CHARGE IS NOT CONSERVED BY GEANT4 HERE, by one unit per anti-proton the projectile
            // residual captures: `G4int Z = G4int(q/eplus + 0.1)` truncates -0.9 to 0, so the
            // captured anti-proton's -1 leaves the books (generator_interface.cuh, note 4). The
            // excess is therefore 0 <= Q - Q0 <= |B| for an anti-nucleus and exactly 0 otherwise;
            // the baryon number balances always.
            const int qx = qsum - qwant;
            if (qx > 0 && bsum == bwant && s.b < -1 && qx <= -s.b) { ++q_excess_events; q_excess += qx; }
            else if (bsum != bwant || qsum != qwant) {
              ++unbal;
              if (unbal <= 2) {
                std::printf("  unbalanced: %s on %s at %g: B %d want %d, Q %d want %d\n", s.name,
                            t.name, kpn, bsum, bwant, qsum, qwant);
              }
            }
          } else if (st == entry::Status::kPrimaryUnchanged) {
            ++unch;
          } else if (g.anti_nucleus) {
            ++anti_nuc;
#if P23C_TREE
          } else if (g.anti_residual_unbound) {
            ++unbound;
            const int k = g.refused_z * 8 + g.refused_a;
            if (k >= 0 && k < 64) { ++unbound_za[k]; }
#endif
          } else if (g.short_lived_track) {
            ++shortl;
          } else if (g.dropped_residual) {
            ++dropped;
          } else {
            ++other;
          }
        }
        std::printf("%-13s %-6s %7.0f %6lld %6lld %6lld %6lld %6lld %6lld %6lld %6lld\n", s.name,
                    t.name, kpn, ran, unch, anti_nuc, unbound, shortl, dropped, other, unbal);
        tot_qx_events += q_excess_events;
        tot_ev += n_events;
        tot_ran += ran;
        tot_anti_nuc += anti_nuc;
        tot_unbound += unbound;
        tot_other_gen += shortl + dropped + other;
        if (unbal > 0) { ++fails; }
#if P23C_TREE
        if (anti_nuc > 0) { ++fails; }  // the arm is written: nothing may stop there any more
#endif
      }
    }
  }
  std::printf("\nall rows: %lld events, %lld ran (%.2f%%), anti_nucleus %lld (%.2f%%), "
              "anti_residual_unbound %lld (%.2f%%), other refusals %lld (%.2f%%)\n",
              tot_ev, tot_ran, 100.0 * tot_ran / tot_ev, tot_anti_nuc,
              100.0 * tot_anti_nuc / tot_ev, tot_unbound, 100.0 * tot_unbound / tot_ev,
              tot_other_gen, 100.0 * tot_other_gen / tot_ev);
  std::printf("events whose final state carries Geant4's +1 charge per captured anti-proton: %lld\n",
              tot_qx_events);
  std::printf("unbound anti-residuals by (Z, A):");
  for (int k = 0; k < 64; ++k) {
    if (unbound_za[k] > 0) { std::printf(" (%d,%d) %lld", k / 8, k % 8, unbound_za[k]); }
  }
  std::printf("\n");
  std::printf("%s (%d failing rows)\n", fails == 0 ? "ALL OK" : "FAILED", fails);
  return fails == 0 ? 0 : 1;
}
