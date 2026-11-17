// P26b: a real nucleus's spin and magnetic moment, and its delta rays, against Geant4 11.1.1.
//
//   1. The table against ENSDFSTATE.dat itself: every ground-state row the file holds is in
//      `data/ensdf_ground_states.hh` with the same 2J and the same double for mu, and no other
//      (Z, A) is.
//   2. The definition against G4IonTable (ref/oracle/ion_spin.csv): for every nuclide on the
//      dump's grid, `em::stepped_ion`'s `spin_known`, spin and magMoment2 against `GetIon(Z,A,0)`
//      and G4BetheBlochModel::SetupParameters' own expression.
//   3. The rate (ion_delta.csv): `em::ion_delta_xs` against GenericIon's model at the scaled
//      energy times the model's own charge-square ratio, as PostStepGetPhysicalInteractionLength
//      composes it; the model choice; and, reported, how far Geant4's lambda TABLE sits from it.
//   4. The sampler deviate for deviate (ion_delta_tape.csv): Geant4's models driven by a scripted
//      engine, the port's `em::sample_ion_delta` by the same script - the transfer, the inner
//      rejection's threshold (f/fmax, the spin term inside), the form-factor rejection's threshold
//      (grej, spin and moment inside), both found by the same 64-step bisection, and the final
//      state.
//   5. The sampler by Monte Carlo (ion_delta.csv's mc columns): the fraction of calls that make an
//      electron and the mean transfer, a million calls a point on each side.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "core/particle.cuh"
#include "core/rng.cuh"
#include "data/materials.cuh"
#include "host/g4data.cuh"
#include "physics/em/hadron_delta.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

int material_of(const std::string& n) {
  if (n == "G4_AIR") { return data::kAir; }
  if (n == "G4_WATER") { return data::kWater; }
  if (n == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (n == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;
}

std::vector<std::vector<std::string>> read_csv(const std::string& path) {
  std::vector<std::vector<std::string>> rows;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return rows; }
  char line[4096];
  bool header = true;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    if (header) {
      header = false;
      continue;
    }
    std::vector<std::string> cells;
    std::string cur;
    for (const char* c = line; *c != '\0' && *c != '\n' && *c != '\r'; ++c) {
      if (*c == ',') {
        cells.push_back(cur);
        cur.clear();
      } else {
        cur += *c;
      }
    }
    cells.push_back(cur);
    rows.push_back(cells);
  }
  std::fclose(f);
  return rows;
}

/// The dump's ScriptEngine: the scripted deviates in order, then 0.5 for ever.
struct ScriptRng {
  std::vector<double> seq;
  std::size_t n = 0;
  double uniform() { return (n < seq.size()) ? seq[n++] : (++n, 0.5); }
};

em::DeltaRay<real_t> shoot(const em::SteppedHadron<real_t>& h, double e, double cut,
                           const std::vector<double>& seq) {
  ScriptRng r{seq, 0};
  return em::sample_ion_delta<real_t>(h, e, cut, Vec3<real_t>{0, 0, 1}, r);
}

double rel(double a, double b) {
  if (a == b) { return 0; }
  return std::fabs(a - b) / std::fmax(std::fabs(b), 1e-300);
}

}  // namespace

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  int fails = 0;

  // ------------------------------------------------------------ 1. the table against the file
  {
    const std::string path = host::g4ensdfstate_dir() + "/ENSDFSTATE.dat";
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s\n", path.c_str());
      return 1;
    }
    int z, a, j;
    double e, life, mu;
    char fl[16];
    int rows = 0, ground = 0, bad = 0;
    while (std::fscanf(f, "%d %d %lf %15s %lf %d %lf", &z, &a, &e, fl, &life, &j, &mu) == 7) {
      ++rows;
      if (!(e == 0.0 && std::string(fl) == "-")) { continue; }
      ++ground;
      const int i = data::ensdf_ground_index(z, a);
      if (i < 0 || data::ensdf_ground_two_j()[i] != j || data::ensdf_ground_mu()[i] != mu) {
        if (bad++ < 5) { std::printf("  FAIL: (%d, %d) 2J %d mu %.9g not in the table\n", z, a, j, mu); }
      }
    }
    std::fclose(f);
    std::printf("== 1. data/ensdf_ground_states.hh against %s ==\n", path.c_str());
    std::printf("  %d rows, %d ground states; table holds %d; %d mismatched\n", rows, ground,
                data::kEnsdfGroundStates, bad);
    if (bad != 0 || ground != data::kEnsdfGroundStates || ground == 0) {
      std::printf("  FAIL\n");
      ++fails;
    }
    // Two named rows, by hand from the file: N14 2J = 2, mu = 2.03930941e-27 J/T; C12 2J = 0.
    const int n14 = data::ensdf_ground_index(7, 14);
    const int c12 = data::ensdf_ground_index(6, 12);
    if (n14 < 0 || data::ensdf_ground_two_j()[n14] != 2 ||
        data::ensdf_ground_mu()[n14] != 2.03930941e-27 || c12 < 0 ||
        data::ensdf_ground_two_j()[c12] != 0 || data::ensdf_ground_index(1, 4) != -1) {
      std::printf("  FAIL: N14 / C12 / H4 rows\n");
      ++fails;
    }
  }

  // ------------------------------------------------------------ 2. the definition, against Geant4
  {
    const auto rows = read_csv(dir + "/ion_spin.csv");
    int n = 0, bad = 0, nspin = 0, nfound = 0;
    double worst = 0;
    std::string where;
    for (const auto& r : rows) {
      if (r.size() < 7) { continue; }
      const int z = std::atoi(r[0].c_str()), a = std::atoi(r[1].c_str());
      const bool found = r[2] == "1";
      const int two_j = std::atoi(r[3].c_str());
      const double spin = std::atof(r[4].c_str()), mass = std::atof(r[5].c_str());
      const double mm2 = std::atof(r[6].c_str());
      const em::SteppedHadron<real_t> h = em::stepped_ion<real_t>(z, a);
      const ParticleDef<real_t> pd = em::ion_particle_def<real_t>(z, a);
      ++n;
      nspin += (two_j != 0);
      nfound += found;
      const double d = rel(h.def.mag_moment2, mm2);
      if (d > worst) {
        worst = d;
        char b[128];
        std::snprintf(b, sizeof b, "(%d, %d): %.17g vs %.17g", z, a, h.def.mag_moment2, mm2);
        where = b;
      }
      const bool ok = h.spin_known == found && h.def.spin == spin && 0.5 * two_j == spin &&
                      h.def.mass == mass && d <= 1e-14 && pd.spin == h.def.spin &&
                      pd.mag_moment2 == h.def.mag_moment2 && pd.mass == h.def.mass;
      if (!ok && bad++ < 5) {
        std::printf("  FAIL: (%d, %d) found %d/%d spin %g/%g mass %.17g/%.17g mm2 %.17g/%.17g\n",
                    z, a, int(h.spin_known), int(found), h.def.spin, spin, h.def.mass, mass,
                    h.def.mag_moment2, mm2);
      }
    }
    std::printf("\n== 2. em::stepped_ion against G4IonTable::GetIon (ion_spin.csv) ==\n");
    std::printf("  %d nuclides (%d with a ground row, %d with spin), %d mismatched; magMoment2 worst"
                " %.3g relative %s\n", n, nfound, nspin, bad, worst, where.c_str());
    if (n < 1000 || bad != 0) {
      std::printf("  FAIL%s\n", n < 1000 ? ": too few rows - regenerate ref/oracle" : "");
      ++fails;
    }
  }

  // ------------------------------------------------------------ 3 and 5. the rate and the MC
  double cut_of[data::kNumMaterials] = {};
  double worst_cut = 0;
  {
    const auto rows = read_csv(dir + "/ion_delta.csv");
    int n = 0, bad = 0, nmc = 0, badmc = 0, nopen = 0;
    double worst = 0, worst_q2 = 0, worst_table = 0, chi2 = 0, worst_z = 0;
    std::string where, where_table, where_z;
    for (const auto& r : rows) {
      if (r.size() < 12) { continue; }
      const int mi = material_of(r[0]);
      if (mi < 0) { continue; }
      const int z = std::atoi(r[1].c_str()), a = std::atoi(r[2].c_str());
      const double e = std::atof(r[3].c_str()), cut = std::atof(r[4].c_str());
      const std::string model = r[5];
      const double q2 = std::atof(r[6].c_str()), xs = std::atof(r[7].c_str());
      const double xs_table = std::atof(r[8].c_str());
      const long mc_n = std::atol(r[9].c_str()), mc_prod = std::atol(r[10].c_str());
      const double mc_sum = std::atof(r[11].c_str());
      const em::SteppedHadron<real_t> h = em::stepped_ion<real_t>(z, a);
      ++n;
      // The cut the dump read off the production-cuts table is the material's - to the ten digits
      // `build_b1_materials` writes it with (water's 0.277632595 MeV against Geant4's
      // 0.27763259496488374, 1.3e-10 relative; bone 1.2e-9; air 990 eV exactly). Everything below takes
      // Geant4's own cut, so that the comparison is of the ion's physics and not of that rounding.
      cut_of[mi] = cut;
      worst_cut = std::fmax(worst_cut, rel(mats[mi].cut_electron, cut));
      if (rel(mats[mi].cut_electron, cut) > 2e-9) {
        if (bad++ < 5) { std::printf("  FAIL: cut %.17g vs %.17g in %s\n", cut, mats[mi].cut_electron, r[0].c_str()); }
      }
      const ParticleDef<real_t> gi = particle_def<real_t>(ParticleType::kGenericIon);
      const em::HadronIoniModel pm =
          em::hadron_ioni_model(ParticleType::kGenericIon, gi, e * em::hadron_mass_ratio(h));
      const bool bb = (pm == em::HadronIoniModel::kBetheBloch);
      if (bb != (model == "BetheBloch")) {
        if (bad++ < 5) { std::printf("  FAIL: model %s at %.6g MeV (%d,%d)\n", model.c_str(), e, z, a); }
      }
      const double q2p = em::hadron_charge_sq_ratio(mats[mi], h, e);
      worst_q2 = std::fmax(worst_q2, rel(q2p, q2));
      const double ours = em::ion_delta_xs(mats[mi], h, real_t(e), real_t(cut), real_t(1e30));
      const double d = (xs == 0) ? std::fabs(ours) : rel(ours, xs);
      if (xs > 0) { ++nopen; }
      if (d > worst) {
        worst = d;
        char b[160];
        std::snprintf(b, sizeof b, "(%d,%d) %.4g MeV in %s: %.10g vs %.10g /mm", z, a, e,
                      r[0].c_str(), ours, xs);
        where = b;
      }
      if (xs > 0 && xs_table > 0 && rel(xs_table, xs) > worst_table) {
        worst_table = rel(xs_table, xs);
        char b[160];
        std::snprintf(b, sizeof b, "(%d,%d) %.4g MeV in %s", z, a, e, r[0].c_str());
        where_table = b;
      }
      // The Monte Carlo: the same million calls on the port's sampler.
      if (mc_n > 0) {
        ++nmc;
        Philox<real_t> rng(static_cast<uint32_t>(mi * 1009 + z * 31 + a),
                           static_cast<uint32_t>(e * 16), 26026u);
        long prod = 0;
        double sum = 0;
        for (long k = 0; k < mc_n; ++k) {
          const auto dr = em::sample_ion_delta<real_t>(h, real_t(e), real_t(cut),
                                                       Vec3<real_t>{0, 0, 1}, rng);
          if (dr.produced) {
            ++prod;
            sum += dr.delta_ekin;
          }
        }
        // The produced fraction is 1 - O(1e-5) at these energies, so a binomial on the FAILURES.
        const double fg = 1.0 - double(mc_prod) / mc_n, fp = 1.0 - double(prod) / mc_n;
        const double var = (fg * (1 - fg) + fp * (1 - fp)) / mc_n;
        const double zf = (var > 0) ? (fp - fg) / std::sqrt(var) : (fp == fg ? 0.0 : 99.0);
        // The mean transfer: the 1/T^2 spectrum's variance, estimated from the port's draws is
        // not kept, so a conservative 1/sqrt(N) of the mean relative is used as the scale.
        const double mg = (mc_prod > 0) ? mc_sum / mc_prod : 0, mp = (prod > 0) ? sum / prod : 0;
        chi2 += zf * zf;
        if (std::fabs(zf) > std::fabs(worst_z)) {
          worst_z = zf;
          char b[200];
          std::snprintf(b, sizeof b, "(%d,%d) %.4g MeV in %s: G4 %ld/%ld, port %ld; mean %.5g/%.5g",
                        z, a, e, r[0].c_str(), mc_prod, mc_n, prod, mg, mp);
          where_z = b;
        }
        if (std::fabs(zf) > 5 || (mg > 0 && rel(mp, mg) > 0.05)) {
          if (badmc++ < 5) { std::printf("  FAIL: %s\n", where_z.c_str()); }
        }
      }
    }
    std::printf("\n== 3. em::ion_delta_xs against GenericIon's model x q2 (ion_delta.csv) ==\n");
    std::printf("  %d points (%d with the window open), worst %.3g relative %s\n", n, nopen, worst,
                where.c_str());
    std::printf("  the materials' electron cuts against Geant4's: worst %.3g relative\n", worst_cut);
    std::printf("  q2 worst %.3g relative; Geant4's own lambda TABLE against the model: worst %.3g"
                " relative %s (reported, not asserted)\n", worst_q2, worst_table,
                where_table.c_str());
    if (n == 0 || nopen == 0 || bad != 0 || worst > 1e-12 || worst_q2 > 1e-13) {
      std::printf("  FAIL\n");
      ++fails;
    }
    std::printf("\n== 5. the sampler by Monte Carlo, a million calls a point ==\n");
    std::printf("  %d points, chi2 %.1f for %d (failures to produce, binomial), worst z %+.2f %s\n",
                nmc, chi2, nmc, worst_z, where_z.c_str());
    if (nmc == 0 || badmc != 0) {
      std::printf("  FAIL\n");
      ++fails;
    }
  }

  // ------------------------------------------------------------ 4. the tape
  {
    const auto rows = read_csv(dir + "/ion_delta_tape.csv");
    int n = 0, bad = 0, nf = 0, ng = 0;
    double wd = 0, wf = 0, wg = 0, ws = 0;
    std::string where;
    for (const auto& r : rows) {
      if (r.size() < 15) { continue; }
      const int mi = material_of(r[0]);
      if (mi < 0) { continue; }
      const int z = std::atoi(r[1].c_str()), a = std::atoi(r[2].c_str());
      const double e = std::atof(r[3].c_str()), r0 = std::atof(r[4].c_str());
      const double gd = std::atof(r[5].c_str()), gf = std::atof(r[6].c_str());
      const double gg = std::atof(r[7].c_str());
      const double cut = cut_of[mi];
      const em::SteppedHadron<real_t> h = em::stepped_ion<real_t>(z, a);
      ++n;
      const auto s0 = shoot(h, e, cut, {r0, 0.0, 0.0, 0.25});
      if (gd < 0) {
        if (s0.produced && bad++ < 5) { std::printf("  FAIL: (%d,%d) %.6g MeV: port produced, G4 window shut\n", z, a, e); }
        continue;
      }
      if (!s0.produced) {
        if (bad++ < 5) { std::printf("  FAIL: (%d,%d) %.6g MeV r0 %g: G4 produced, port did not\n", z, a, e, r0); }
        continue;
      }
      double lo = 0, hi = 1;
      if (shoot(h, e, cut, {r0, 1.0, 0.5, 0.0, 0.0, 0.25}).delta_ekin == s0.delta_ekin) { lo = 1; }
      if (lo < 1) {
        for (int it = 0; it < 64; ++it) {
          const double mid = 0.5 * (lo + hi);
          if (shoot(h, e, cut, {r0, mid, 0.5, 0.0, 0.0, 0.25}).delta_ekin == s0.delta_ekin) {
            lo = mid;
          } else {
            hi = mid;
          }
        }
      }
      const double pf = lo;
      lo = 0;
      hi = 1;
      if (shoot(h, e, cut, {r0, 0.0, 1.0, 0.25}).produced) { lo = 1; }
      if (lo < 1) {
        for (int it = 0; it < 64; ++it) {
          const double mid = 0.5 * (lo + hi);
          if (shoot(h, e, cut, {r0, 0.0, mid, 0.25}).produced) { lo = mid; } else { hi = mid; }
        }
      }
      const double pg = lo;
      nf += (gf < 1);
      ng += (gg < 1);
      const double dd = rel(s0.delta_ekin, gd);
      const double df = std::fabs(pf - gf), dg = std::fabs(pg - gg);
      double dstate = rel(s0.primary_ekin, std::atof(r[11].c_str()));
      const double gdv[3] = {std::atof(r[8].c_str()), std::atof(r[9].c_str()), std::atof(r[10].c_str())};
      const double gpv[3] = {std::atof(r[12].c_str()), std::atof(r[13].c_str()), std::atof(r[14].c_str())};
      const double pdv[3] = {s0.delta_dir.x, s0.delta_dir.y, s0.delta_dir.z};
      const double ppv[3] = {s0.primary_dir.x, s0.primary_dir.y, s0.primary_dir.z};
      for (int k = 0; k < 3; ++k) {
        dstate = std::fmax(dstate, std::fabs(pdv[k] - gdv[k]));
        dstate = std::fmax(dstate, std::fabs(ppv[k] - gpv[k]));
      }
      wd = std::fmax(wd, dd);
      ws = std::fmax(ws, dstate);
      if (df > wf) { wf = df; }
      if (dg > wg) {
        wg = dg;
        char b[200];
        std::snprintf(b, sizeof b, "(%d,%d) %.4g MeV in %s r0 %g: grej %.17g vs %.17g", z, a, e,
                      r[0].c_str(), r0, pg, gg);
        where = b;
      }
      if (dd > 1e-13 || df > 2e-15 || dg > 2e-15 || dstate > 1e-12) {
        if (bad++ < 8) {
          std::printf("  FAIL: (%d,%d) %.6g MeV in %s r0 %g: delta %.17g/%.17g f %.17g/%.17g "
                      "grej %.17g/%.17g state %.3g\n", z, a, e, r[0].c_str(), r0, s0.delta_ekin,
                      gd, pf, gf, pg, gg, dstate);
        }
      }
    }
    std::printf("\n== 4. em::sample_ion_delta against Geant4's models, deviate for deviate "
                "(ion_delta_tape.csv) ==\n");
    std::printf("  %d draws (%d with an inner threshold below 1, %d with a form-factor threshold"
                " below 1)\n", n, nf, ng);
    std::printf("  worst: transfer %.3g rel, f threshold %.3g abs, grej threshold %.3g abs %s, "
                "final state %.3g\n", wd, wf, wg, where.c_str(), ws);
    if (n == 0 || ng == 0 || bad != 0) {
      std::printf("  FAIL\n");
      ++fails;
    }
  }

  std::printf("\n%s (%d failures)\n", fails == 0 ? "PASS" : "FAIL", fails);
  return fails == 0 ? 0 : 1;
}
