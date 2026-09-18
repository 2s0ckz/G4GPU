// FTFP through its CONTRACT HEADER only, the way P12 and P13 will call it.
//
// `tests/test_ftf_model.cu` includes the model and knows how to build its workspaces; that is
// what made it useless as evidence that anyone ELSE could. This file includes exactly one header
// of the package -
//
//     physics/hadronic/ftf/ftf_entry.cuh
//
// - and nothing else from `ftf/`. If the contract is incomplete, this file does not compile. If
// the workspace builder is wrong, this file produces no secondaries. Both are failures here.
//
// WHAT IT CHECKS, AND WHY THERE IS NO ORACLE TABLE
//
// Four projectiles x two targets x two energies, 2,000 events each, and the three conservation
// laws `G4GeneratorPrecompoundInterface` checks in its own debug block: baryon number exactly,
// charge exactly, and the energy balance to a stated bound. Those need no oracle row, and they
// are the checks that an entry point wired up wrongly cannot pass - a workspace whose scratch
// pointers were not re-wired, or a Lund table that was memset instead of constructed, changes
// the final state's books immediately. The DISTRIBUTIONS are `test_ftf_model.cu`'s job and are
// compared against Geant4 there, over 30,501 comparisons.
//
// The energy bound is loose ON PURPOSE and the reason is in docs/RISK.md V115: Geant4 does not
// conserve energy exactly across this hand-over either, because the residual carries a table
// mass and an excitation that are not the sum of what went in. 10 MeV of mean imbalance is the
// gate; the measured worst over these sixteen points is well inside it and is printed.
//
// AT REST IS A REAL CASE AND IT IS P12'S. An anti-proton at 1 MeV of kinetic energy is what
// `G4HadronicAbsorptionFritiof` hands FTFP, and it is the one row here that exercises
// `AdjustNucleons` (below 1 GeV/c per nucleon) and the annihilation channels together. It runs:
// 2,000 of 2,000 on carbon.
//
// A proton, a pion or an alpha at 1 MeV is NOT an FTFP event in QBBC - the builders hand those to
// the cascades - and running one anyway is what found the third outcome this contract has to
// carry. Every such call spends all 1,000 `Scatter` attempts, none of which can succeed, and
// Geant4 then returns THE PRIMARY UNCHANGED with a JustWarning. That is a final state, so
// `Status::kPrimaryUnchanged` is its own answer and not `kRefused`; a caller that read it as
// "nothing happened" would double-count the primary.
//
// THOSE FOUR ROWS RUN 50 EVENTS AND NOT 2,000, and the reason is measured rather than assumed:
// at 185-242 ms per call on lead - 1,002 attempts, each rebuilding both nuclei - 2,000 events is
// eight minutes a row and twenty for the four. Fifty is enough to establish that the outcome is
// always the same named one, which is all there is to establish; the twelve rows inside FTFP's
// window, the ones where the physics is, run the full 2,000.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "core/rng.cuh"
#include "physics/hadronic/ftf/ftf_entry.cuh"

using namespace g4gpu;
namespace entry = g4gpu::hadronic::ftf::entry;

namespace {

int fails = 0;

/// The eight (projectile, energy) pairs and the two targets. `a`/`z` are zero for a hadron; an
/// ion carries them and its mass comes from P3's table, which is what `G4IonTable::GetIonMass`
/// is (docs/PORTED.md 2.1.11b's note on why writing that number twice is the mistake).
struct Beam {
  const char* name;
  int pdg;
  int a, z;  ///< 0, 0 for a hadron
};

struct Target {
  const char* name;
  int a, z;
};

/// Baryon number and charge of one secondary, from the same table the model reads. A nuclear
/// fragment carries (Z, A) itself; a deuteron out of `MakeCoalescence` is a PDG nuclear code
/// with no row in the hadron table and is named here.
bool baryon_and_charge(const g4gpu::physics::hadronic::HadSecondary<double>& s, int& b, int& q) {
  if (s.a > 0) {
    b = s.a;
    q = s.z;
    return true;
  }
  const data::FtfHadron* d = data::ftf_find_hadron(s.pdg);
  if (d != nullptr) {
    b = d->baryon;
    q = static_cast<int>(d->charge);
    return true;
  }
  if (s.pdg == 1000010020) {  // deuteron
    b = 2;
    q = 1;
    return true;
  }
  b = 0;
  q = 0;
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = (argc > 1 && std::string(argv[1]) == "--quick");
  const int n_events = quick ? 200 : 2000;

  // ------------------------------------------------------------------------------------------
  // The builder, used exactly as a caller outside the package would - except that this test
  // runs on the HOST, so it takes `host_handle` rather than `build`. Both produce the same
  // `Handle`, and `apply` cannot tell them apart; `build`'s device path is exercised by the
  // byte arithmetic below and by whichever run first wires FTFP into a launch.
  // ------------------------------------------------------------------------------------------
  const int kSlots = 4;
  static entry::Workspace slots[kSlots];
  static hadronic::ftf::LundTables<double> lund;
  const entry::Handle<entry::Workspace> h = entry::host_handle(slots, kSlots, &lund);
  if (!h.ok()) {
    std::printf("FAIL: the handle the builder returned is not usable\n");
    return 1;
  }

  std::printf("FTFP entry contract\n");
  std::printf("  entry::Workspace       %9zu B/slot   %d slots = %.2f MB\n",
              sizeof(entry::Workspace), kSlots,
              double(entry::bytes_for<entry::Workspace>(kSlots)) / 1048576.0);
  std::printf("  entry::HadronWorkspace %9zu B/slot\n", sizeof(entry::HadronWorkspace));
  std::printf("  LundTables<double>     %9zu B shared\n",
              sizeof(hadronic::ftf::LundTables<double>));
  for (int n : {64, 256, 1024, 65536}) {
    std::printf("  %6d slots: Workspace %9.2f MB   HadronWorkspace %9.2f MB\n", n,
                double(entry::bytes_for<entry::Workspace>(n)) / 1048576.0,
                double(entry::bytes_for<entry::HadronWorkspace>(n)) / 1048576.0);
  }
  // A budget question a caller can ask before it allocates.
  {
    const std::size_t gb = 1073741824u;
    std::printf("  a 1 GB budget buys %d Workspace slots or %d HadronWorkspace slots\n",
                entry::slots_for_bytes<entry::Workspace>(gb),
                entry::slots_for_bytes<entry::HadronWorkspace>(gb));
  }

  // ------------------------------------------------------------------------------------------
  // The grid.
  // ------------------------------------------------------------------------------------------
  const Beam beams[] = {
      {"proton", 2212, 0, 0},
      {"pi+", 211, 0, 0},
      {"alpha", 0, 4, 2},
      {"anti_proton", -2212, 0, 0},
  };
  const Target targets[] = {{"C12", 12, 6}, {"Pb207", 207, 82}};
  // "At rest" is 1 MeV of kinetic energy and not zero: `G4HadronicProcess` never hands a model a
  // track of exactly zero energy - the at-rest processes give it the particle's rest mass plus
  // the binding energy it was captured from - and a literal zero would make plab zero and the
  // boost to the c.m.s. a division by the mass alone. 1 MeV is what P12's arm sees.
  const double kins[] = {1.0, 10000.0};

  static g4gpu::physics::hadronic::HadFinalState<double, 128> out;
  long long points = 0, ran = 0, refused = 0, silent = 0, unchanged = 0;
  long long bad_b = 0, bad_q = 0, unknown = 0, no_slot = 0;
  double worst_de = 0.0;
  std::string worst_where;

  std::printf("\n%-12s %-6s %8s %6s %8s %8s %8s %10s  %s\n", "beam", "target", "T_MeV", "N",
              "ran", "unchngd", "sec/ev", "<dE> MeV", "outcomes");
  for (const Beam& b : beams) {
    const bool is_ion = (b.a > 0);
    // An ion's mass is `G4IonTable::GetIonMass(Z, A)`, which is P3's `deex::nuclear_mass`; a
    // hadron's is the PDG mass out of the same table the model reads.
    double mass = 0.0;
    int baryon = 0, charge = 0, pdg = b.pdg;
    if (is_ion) {
      mass = deex::nuclear_mass(b.a, b.z);
      baryon = b.a;
      charge = b.z;
      pdg = 1000000000 + b.z * 10000 + b.a * 10;
    } else {
      const data::FtfHadron* d = data::ftf_find_hadron(b.pdg);
      if (d == nullptr) {
        std::printf("FAIL: no hadron-table row for %s\n", b.name);
        ++fails;
        continue;
      }
      mass = d->mass;
      baryon = d->baryon;
      charge = static_cast<int>(d->charge);
    }
    for (const Target& t : targets) {
      for (double kin : kins) {
        ++points;
        g4gpu::physics::hadronic::HadProjectile<double> hp;
        hp.pdg = pdg;
        hp.mass = mass;
        hp.charge = charge;
        hp.baryon_number = baryon;
        hp.kin_energy = kin;
        g4gpu::physics::hadronic::HadNucleus nuc;
        nuc.a = t.a;
        nuc.z = t.z;

        // The four out-of-window rows cost 7-242 ms a call and cannot succeed; 50 establishes
        // the outcome. See the file header.
        const bool out_of_window = (kin < 3000.0 && b.pdg >= 0);
        const int n_here = out_of_window ? (quick ? 10 : 50) : n_events;
        long long n_ran = 0, n_ref = 0, n_sec = 0, n_unchanged = 0;
        double sum_de = 0.0;
        std::string why;
        int seen[64] = {};
        for (int ev = 0; ev < n_here; ++ev) {
          out = g4gpu::physics::hadronic::HadFinalState<double, 128>();
          Philox<double> rng(static_cast<uint32_t>(ev), 101u);
          entry::Report rep;
          // The slot a kernel thread would take. Rotated over the four so that reuse of a
          // workspace between calls is exercised rather than assumed - a workspace that only
          // works the first time is the failure mode a single-slot test would miss.
          const int slot = ev % kSlots;
          const entry::Status st = entry::apply(h, slot, hp, nuc, out, rep, rng);
          if (st == entry::Status::kNoWorkspaceSlot) {
            ++no_slot;
            continue;
          }
          if (st == entry::Status::kPrimaryUnchanged) {
            ++n_unchanged;
            if (seen[61]++ == 0) { why += "primary unchanged (1000 attempts) "; }
            // Geant4's fallback hands the primary to `Propagate`, which lets a non-nucleon
            // escape, and the untouched nucleus comes back beside it as a ground-state residual:
            // TWO secondaries for a hadron beam and THREE for an ion, which has a projectile
            // residual as well. The books must still balance, so they are checked here and not
            // skipped - it was this row that found the primary going missing for an ion
            // (docs/RISK.md V146).
            const int want = is_ion ? 3 : 2;
            if (out.n_secondaries != want) {
              std::printf("FAIL: %s on %s at %g MeV returned the unchanged primary as %d "
                          "secondaries, wanted %d\n", b.name, t.name, kin, out.n_secondaries,
                          want);
              ++fails;
            }
            // THE BOOKS ON THIS PATH BALANCE FOR A HADRON AND NOT FOR AN ION, in Geant4 as
            // here, and the test asserts that rather than the law. `PropagateNuclNucl` sets the
            // projectile residual to the WHOLE primary when no projectile nucleon was hit
            // (`anAb == projectile.a && exEnergyB <= 0` -> `projectile4 = primary`), and the
            // primary ALSO escapes as a track - so an ion event carries the projectile twice.
            // `Propagate`, the hadron arm, has no such line and comes out exact. Reproduced
            // rather than corrected: it is reachable only through the 1000-attempt fallback,
            // which Geant4 itself raises a JustWarning for, and only outside FTFP's energy
            // window. docs/RISK.md V147.
            int bsum = 0, qsum = 0;
            for (int k = 0; k < out.n_secondaries; ++k) {
              int bb = 0, qq = 0;
              if (!baryon_and_charge(out.secondaries[k], bb, qq)) { ++unknown; }
              bsum += bb;
              qsum += qq;
            }
            const int want_b = baryon + t.a + (is_ion ? baryon : 0);
            const int want_q = charge + t.z + (is_ion ? charge : 0);
            if (bsum != want_b) { ++bad_b; }
            if (qsum != want_q) { ++bad_q; }
            continue;
          }
          if (st == entry::Status::kRefused) {
            ++n_ref;
            if (rep.refused != hadronic::ftf::FtfRefusal::kNone) {
              const int i = static_cast<int>(rep.refused);
              if (i >= 0 && i < 64 && seen[i]++ == 0) {
                why += std::string(entry::refusal_name(rep.refused)) + " ";
              }
            } else if (rep.generator_refused) {
              if (seen[63]++ == 0) { why += "P6 Propagate "; }
            } else if (rep.capacity) {
              if (seen[62]++ == 0) { why += "capacity "; }
            } else {
              ++silent;
              if (silent <= 3) {
                std::printf("FAIL: %s on %s at %g MeV: neither a final state nor a name\n",
                            b.name, t.name, kin);
              }
            }
            continue;
          }
          ++n_ran;
          n_sec += out.n_secondaries;
          int bsum = 0, qsum = 0;
          double e_out = 0.0;
          for (int k = 0; k < out.n_secondaries; ++k) {
            const auto& s = out.secondaries[k];
            e_out += s.kin_energy + s.mass;
            int bb = 0, qq = 0;
            if (!baryon_and_charge(s, bb, qq)) { ++unknown; }
            bsum += bb;
            qsum += qq;
          }
          if (bsum != baryon + t.a) { ++bad_b; }
          if (qsum != charge + t.z) { ++bad_q; }
          sum_de += e_out - (kin + mass + deex::nuclear_mass(t.a, t.z));
        }
        ran += n_ran;
        refused += n_ref;
        unchanged += n_unchanged;
        const double mde = n_ran ? sum_de / static_cast<double>(n_ran) : 0.0;
        if (std::fabs(mde) > std::fabs(worst_de)) {
          worst_de = mde;
          worst_where = std::string(b.name) + " on " + t.name + " at " +
                        std::to_string(static_cast<int>(kin)) + " MeV";
        }
        std::printf("%-12s %-6s %8.0f %6d %8lld %8lld %8.2f %10.3f  %s\n", b.name, t.name, kin,
                    n_here, n_ran, n_unchanged, n_ran ? double(n_sec) / n_ran : 0.0, mde,
                    why.c_str());
      }
    }
  }

  // ------------------------------------------------------------------------------------------
  // The slot contract itself: a thread past the end must be REFUSED and not aliased.
  // ------------------------------------------------------------------------------------------
  {
    g4gpu::physics::hadronic::HadProjectile<double> hp;
    const data::FtfHadron* d = data::ftf_find_hadron(2212);
    hp.pdg = 2212;
    hp.mass = d->mass;
    hp.charge = 1;
    hp.baryon_number = 1;
    hp.kin_energy = 10000.0;
    g4gpu::physics::hadronic::HadNucleus nuc;
    nuc.a = 12;
    nuc.z = 6;
    out = g4gpu::physics::hadronic::HadFinalState<double, 128>();
    Philox<double> rng(1u, 2u);
    entry::Report rep;
    const entry::Status st = entry::apply(h, kSlots, hp, nuc, out, rep, rng);
    if (st != entry::Status::kNoWorkspaceSlot) {
      std::printf("FAIL: slot %d of %d was served instead of refused\n", kSlots, kSlots);
      ++fails;
    }
    if (out.n_secondaries != 0) {
      std::printf("FAIL: a refused slot produced %d secondaries\n", out.n_secondaries);
      ++fails;
    }
    const entry::Handle<entry::Workspace> empty;
    if (empty.ok() || entry::apply(empty, 0, hp, nuc, out, rep, rng) !=
                          entry::Status::kNoWorkspaceSlot) {
      std::printf("FAIL: a default-constructed handle was usable\n");
      ++fails;
    }
  }

  std::printf("\n%lld points: %lld ran, %lld primary-unchanged, %lld refused by name, "
              "%lld silent, %lld no-slot\n", points, ran, unchanged, refused, silent, no_slot);
  std::printf("baryon number wrong in %lld, charge in %lld, unknown PDG %lld\n", bad_b, bad_q,
              unknown);
  std::printf("worst mean energy imbalance %.3f MeV at %s\n", worst_de, worst_where.c_str());

  if (ran == 0) {
    std::printf("FAIL: the entry point produced no final state at any of the %lld points\n",
                points);
    ++fails;
  }
  if (bad_b != 0 || bad_q != 0 || unknown != 0 || silent != 0 || no_slot != 0) {
    std::printf("FAIL: %lld baryon, %lld charge, %lld unknown, %lld silent, %lld no-slot\n",
                bad_b, bad_q, unknown, silent, no_slot);
    ++fails;
  }
  if (std::fabs(worst_de) > 10.0) {
    std::printf("FAIL: energy imbalance %.3f MeV at %s exceeds 10 MeV\n", worst_de,
                worst_where.c_str());
    ++fails;
  }

  std::printf("%s\n", fails == 0 ? "PASS test_ftf_entry" : "FAIL test_ftf_entry");
  return fails == 0 ? 0 : 1;
}
