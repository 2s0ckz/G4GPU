// G4IonFluctuations against Geant4's own: the alpha's energy-loss straggling.
//
// This test exists because an audit of Geant4's process tree found the port using the wrong
// fluctuation model for alphas. G4EmBuilder::ConstructIonEmPhysics registers G4ionIonisation
// for alpha and He3 - not G4hIonisation, which deuteron and triton get - and G4ionIonisation
// asks for its fluctuation model with
//
//     SetFluctModel(G4EmStandUtil::ModelOfFluctuations(true));       G4ionIonisation.cc:147
//
// which returns G4IonFluctuations, not G4UniversalFluctuation. (G4hIonisation reaches the same
// answer for an alpha through a name test, but it is not the process an alpha is given.)
// The two have the same mean by construction, so the total dose in example B1 was right with
// either and the error lived entirely in the width of the straggling distribution: exactly
// the quantity a single scoring volume cannot see. Nothing in the pipeline would have caught
// it, and nothing did.
//
// What is compared, and why it is the dispersion rather than samples:
//
//   G4IonFluctuations::Dispersion is deterministic. It carries both empirical corrections the
//   model exists for - Yang's charge-state straggling and Geissel's Fermi-gas factor - and the
//   cut correction that dilutes them, so comparing it exercises everything about the model
//   except the three-way choice of sampling distribution. Comparing samples instead would put
//   a Monte Carlo error bar between the port and Geant4 and turn a 3% transcription error into
//   something a few thousand samples cannot resolve.
//
//   The sampling itself is then checked separately, on the one property it must have: the
//   sampled mean must equal the mean handed in, in all three branches.
//
// Split four ways, because a single worst-case would say "wrong" without saying where:
//
//   alpha vs proton         - different branch of Factor (charge >= 1.5 takes an A13
//                             prefactor and rescales the reduced energy; charge < 1.5 does
//                             neither and uses a different row of Yang's b table). No stock
//                             list gives a proton this model, but G4MuIonisation gives it to
//                             muons below 200 keV and G4ionIonisation to every ion, so the
//                             branch is live and would otherwise be untested.
//   gas vs condensed        - picks the b row and the reduced-energy divisor.
//   Yang active vs not      - the tabulated term is used only below
//                             beta^2 < 3 * theBohrBeta2 * Zeff, and above it the relativistic
//                             factor stands alone.
//   below vs above Vavilov  - above 10 MeV * charge * mass/m_p the model is G4UniversalFluctuation
//                             and this file's job is only to route to it.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/particle.cuh"
#include "core/rng.cuh"
#include "data/materials.cuh"
#include "data/nist_excitation.hh"
#include "physics/em/hadron_ionisation.cuh"
#include "physics/em/ion_fluctuation.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

struct Row {
  std::string material, particle;
  int is_gas = 0;
  double vavilov = 0, e = 0, tcut = 0, tmax = 0, length = 0, dispersion = 0;
};

std::vector<Row> load(const std::string& path) {
  std::vector<Row> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) {
    std::fclose(f);
    return out;
  }
  while (std::fgets(line, sizeof line, f) != nullptr) {
    Row r;
    char mat[64], part[32];
    const int n = std::sscanf(line, "%63[^,],%31[^,],%d,%lf,%lf,%lf,%lf,%lf,%lf", mat, part,
                              &r.is_gas, &r.vavilov, &r.e, &r.tcut, &r.tmax, &r.length,
                              &r.dispersion);
    if (n != 9) { continue; }
    r.material = mat;
    r.particle = part;
    out.push_back(r);
  }
  std::fclose(f);
  return out;
}

struct Cell {
  int n = 0;
  double worst = 0;
  std::string where;
};

void note(Cell& c, double dev, const std::string& what) {
  ++c.n;
  if (dev > c.worst) {
    c.worst = dev;
    c.where = what;
  }
}

real_t a_of_z(int z) { return data::atomic_mass<real_t>(z); }

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  const auto rows = load(dir + "/ion_fluctuation.csv");
  if (rows.empty()) {
    std::printf("cannot read %s/ion_fluctuation.csv - run ref/oracle/run.bat first\n",
                dir.c_str());
    return 1;
  }

  int fails = 0;

  // ---------------------------------------------------------------- the materials
  // The same seven ref/dump/g4dump.cc builds. The three custom ones matter here for a reason
  // beyond coverage: they are constructed with `new G4Material(name, density, n)` and so carry
  // kStateUndefined, which G4IonFluctuations sends down its condensed-matter branch. The port
  // has no state for them either and falls back to a density test, and the is_gas column below
  // is what checks the two agree rather than assuming it.
  constexpr int kNMat = 7;
  const char* names[kNMat] = {"G4_AIR",     "G4_WATER",           "G4_A-150_TISSUE",
                              "G4_BONE_COMPACT_ICRU", "CustomSiGe", "CustomMolecularWater",
                              "CustomDerivedI"};
  data::Material<real_t> mats[kNMat];
  {
    data::Material<real_t> b1[data::kNumMaterials];
    data::build_b1_materials<real_t>(b1);
    for (int i = 0; i < 4; ++i) { mats[i] = b1[i]; }

    data::MaterialTable<real_t> table{};
    table.count = 0;
    {  // CustomSiGe: 4.2 g/cm3, Si 0.6 / Ge 0.4
      const int zs[2] = {14, 32};
      const real_t w[2] = {real_t(0.6), real_t(0.4)};
      const int idx = data::add_material<real_t>(table, real_t(4.2), 2, zs, w,
                                                 real_t(224.66048772837619));
      if (idx < 0) { std::printf("cannot build CustomSiGe\n"); return 1; }
      mats[4] = table.m[idx];
    }
    {  // CustomMolecularWater: H2O by atom count, 1 g/cm3, no SetMeanExcitationEnergy, so
       // Geant4 derives one - and so must this, rather than pasting a rounded copy of the
       // answer. A literal 68.9984175 eV against Geant4's 68.998417467952655 is 5e-10
       // relative, which the dispersion's log(4*eF/I) turns into 3e-11 and which then reads
       // as a transcription error in a model that is in fact exact.
      const real_t aH = a_of_z(1), aO = a_of_z(8);
      const real_t total = real_t(2) * aH + aO;
      const int zs[2] = {1, 8};
      const double w[2] = {static_cast<double>(real_t(2) * aH / total),
                           static_cast<double>(aO / total)};
      const real_t wr[2] = {real_t(w[0]), real_t(w[1])};
      const double iev = data::derive_mean_excitation_eV(2, zs, w, a_of_z);
      const int idx = data::add_material<real_t>(table, real_t(1.0), 2, zs, wr, real_t(iev));
      if (idx < 0) { std::printf("cannot build CustomMolecularWater\n"); return 1; }
      mats[5] = table.m[idx];
    }
    {  // CustomDerivedI: 2.5 g/cm3, H 0.1 / C 0.6 / Pb 0.3, excitation derived not given
      const int zs[3] = {1, 6, 82};
      const double w[3] = {0.1, 0.6, 0.3};
      const real_t wr[3] = {real_t(0.1), real_t(0.6), real_t(0.3)};
      const double iev = data::derive_mean_excitation_eV(3, zs, w, a_of_z);
      const int idx =
          data::add_material<real_t>(table, real_t(2.5), 3, zs, wr, real_t(iev));
      if (idx < 0) { std::printf("cannot build CustomDerivedI\n"); return 1; }
      mats[6] = table.m[idx];
    }
  }
  auto index_of = [&](const std::string& n) {
    for (int i = 0; i < kNMat; ++i) {
      if (n == names[i]) { return i; }
    }
    return -1;
  };

  // ---------------------------------------------------------------- 0. the gas flag
  // Before any number is compared: does the port put each material on the same side of
  // Geant4's `kStateGas == GetState()` test? A mismatch here changes which row of Yang's b
  // table is read and whether the reduced energy is divided by sqrt(q) or sqrt(qZ), which
  // moves the dispersion by percent and would be reported below as an arithmetic error.
  {
    std::printf("== matter state ==\n");
    int bad = 0;
    for (int i = 0; i < kNMat; ++i) {
      int want = -1;
      for (const Row& r : rows) {
        if (r.material == names[i]) { want = r.is_gas; break; }
      }
      if (want < 0) { continue; }
      const bool ours = data::material_is_gas<real_t>(mats[i]);
      if (ours != (want == 1)) {
        std::printf("  FAIL: %s - Geant4 says %s, this port says %s\n", names[i],
                    want ? "gas" : "not gas", ours ? "gas" : "not gas");
        ++bad;
      }
    }
    std::printf("  %d materials, %d disagreements\n\n", kNMat, bad);
    fails += bad;
  }

  // ---------------------------------------------------------------- 1. the Vavilov boundary
  // Where the model stops being itself and defers to G4UniversalFluctuation. Getting this
  // wrong by a factor - using the proton mass for the alpha, say - moves the handover from
  // 79.45 MeV to 20 MeV and swaps the model over a range where alphas are actually
  // transported, while leaving every dispersion below it correct.
  {
    std::printf("== Vavilov handover ==\n");
    struct Want { const char* name; ParticleType t; };
    const Want w[2] = {{"alpha", ParticleType::kAlpha}, {"proton", ParticleType::kProton}};
    for (const Want& x : w) {
      double want = -1;
      for (const Row& r : rows) {
        if (r.particle == x.name) { want = r.vavilov; break; }
      }
      if (want < 0) { continue; }
      const ParticleDef<real_t> pd = particle_def<real_t>(x.t);
      const real_t ours =
          em::kIonFlucParameter<real_t>() * std::fabs(pd.charge) * pd.mass;
      const double dev = std::fabs(ours / want - 1);
      std::printf("  %-8s ours %.10g MeV, G4 %.10g MeV, dev %.3e\n", x.name,
                  static_cast<double>(ours), want, dev);
      if (dev > 1e-12) {
        std::printf("  FAIL: %s handover is off by %.3e\n", x.name, dev);
        ++fails;
      }
    }
    std::printf("\n");
  }

  // ---------------------------------------------------------------- 2. the dispersion
  Cell cells[2][2][2];  // [alpha?][gas?][Yang active?]
  Cell per_mat[kNMat];  // reported alongside, because a material property that is off by a
                        // rounding step shows up in every branch at once and would otherwise
                        // read as four independent arithmetic errors.
  int compared = 0, skipped_above_vavilov = 0;

  for (const Row& r : rows) {
    const int mi = index_of(r.material);
    if (mi < 0) { continue; }
    const bool is_alpha = (r.particle == "alpha");
    const ParticleDef<real_t> pd =
        particle_def<real_t>(is_alpha ? ParticleType::kAlpha : ParticleType::kProton);

    // Dispersion is defined at every energy; the model only stops *using* it above the
    // handover. Compare it everywhere - a bug in Factor that only shows above 79 MeV is still
    // a bug for GenericIon, whose handover is elsewhere.
    if (r.e > r.vavilov) { ++skipped_above_vavilov; }

    const real_t ours = em::ion_dispersion<real_t>(
        mats[mi], pd, real_t(r.e), real_t(r.tcut), real_t(r.tmax), real_t(r.length),
        pd.charge * pd.charge);

    // Which branch of Factor this point took, so the report localises a failure.
    const real_t beta2 = em::ion_beta2<real_t>(real_t(r.e), pd.mass);
    const bool yang = beta2 < real_t(3) * em::kBohrBeta2<real_t>() * mats[mi].z_eff;

    const double scale = (r.dispersion > 1e-300) ? r.dispersion : 1e-300;
    const double dev = (r.dispersion > 1e-300)
                           ? std::fabs(static_cast<double>(ours) - r.dispersion) / scale
                           : (std::fabs(static_cast<double>(ours)) > 1e-300 ? 1.0 : 0.0);
    char buf[220];
    std::snprintf(buf, sizeof buf, "%s in %s at %.5g MeV over %.3g mm (ours %.9g, G4 %.9g)",
                  r.particle.c_str(), r.material.c_str(), r.e, r.length,
                  static_cast<double>(ours), r.dispersion);
    note(cells[is_alpha ? 1 : 0][r.is_gas ? 1 : 0][yang ? 1 : 0], dev, buf);
    note(per_mat[mi], dev, buf);
    ++compared;
  }

  std::printf("== G4IonFluctuations::Dispersion ==\n");
  std::printf("  %d points (%d of them above the Vavilov handover, compared anyway)\n\n",
              compared, skipped_above_vavilov);
  const char* pn[2] = {"proton", "alpha"};
  const char* gn[2] = {"condensed", "gas"};
  const char* yn[2] = {"Geissel only", "Geissel+Yang"};
  // 1e-12, and it is a rounding limit. Every number on both sides comes from the same table
  // and the same closed forms in double precision - including G4Pow's own powA, transcribed
  // in data/g4pow.hh rather than replaced by std::pow, which by itself is a 1e-7 difference.
  // What this limit leaves room for is a compiler reassociating a sum; what it leaves no room
  // for is a wrong coefficient row, a wrong b-table index, or a missing correction.
  constexpr double kLimit = 1e-12;
  for (int a = 0; a < 2; ++a) {
    for (int g = 0; g < 2; ++g) {
      for (int y = 0; y < 2; ++y) {
        const Cell& c = cells[a][g][y];
        if (c.n == 0) { continue; }
        std::printf("  %-7s %-10s %-13s %5d points  worst %10.3e  %s\n", pn[a], gn[g], yn[y],
                    c.n, c.worst, c.where.c_str());
        if (c.worst > kLimit) {
          std::printf("  FAIL: %s/%s/%s off by %.3e, limit %.0e\n", pn[a], gn[g], yn[y],
                      c.worst, kLimit);
          ++fails;
        }
      }
    }
  }
  std::printf("\n  per material:\n");
  for (int i = 0; i < kNMat; ++i) {
    if (per_mat[i].n == 0) { continue; }
    std::printf("    %-24s %5d points  worst %10.3e\n", names[i], per_mat[i].n,
                per_mat[i].worst);
  }

  // Every branch must actually have been reached. A dump that happened to cover only
  // condensed matter, or only energies above the Yang cut, would leave half of Factor
  // untested while every check above passed.
  for (int a = 0; a < 2; ++a) {
    for (int y = 0; y < 2; ++y) {
      if (cells[a][0][y].n == 0 && cells[a][1][y].n == 0) {
        std::printf("  FAIL: no %s points with %s\n", pn[a], yn[y]);
        ++fails;
      }
    }
  }
  if (cells[1][1][0].n == 0 && cells[1][1][1].n == 0) {
    std::printf("  FAIL: no alpha-in-a-gas points at all\n");
    ++fails;
  }
  std::printf("\n");

  // ---------------------------------------------------------------- 3. the sampler's mean
  // A fluctuation that does not preserve the mean is a systematic energy-loss error, and it
  // shows up as a Bragg peak in the wrong place rather than as a wrong width. All three
  // branches are exercised: sn >= 2 (Gaussian), 0.1 < sn < 2 (Gamma), sn <= 0.1 (uniform).
  {
    std::printf("== E[sampled loss] against the mean handed in ==\n");
    std::printf("  %-22s %10s %9s %11s %11s %10s  %s\n", "material", "E(MeV)", "step(mm)",
                "mean", "sampled", "dev", "branch");
    const ParticleDef<real_t> pd = particle_def<real_t>(ParticleType::kAlpha);
    // Energies either side of the 79.45 MeV handover, and step lengths from a tenth of the
    // range to a fraction of it, so sn spans all three branches.
    const double energies[] = {0.5, 5.0, 40.0, 200.0};
    const double steps[] = {0.001, 0.05, 0.5};
    const int mi_list[] = {0, 1, 4};  // air, water, SiGe
    const int kSamples = 400000;
    for (int k = 0; k < 3; ++k) {
      const int mi = mi_list[k];
      for (double e : energies) {
        for (double len : steps) {
          Philox<real_t> rng(static_cast<uint32_t>(mi * 131 + int(e)),
                             static_cast<uint32_t>(len * 1000), 23u);
          const real_t tmax = em::hadron_max_secondary_energy<real_t>(pd, real_t(e));
          const real_t tcut = fmin(mats[mi].cut_electron, tmax);
          // A physically plausible mean loss: a fixed fraction of the kinetic energy, small
          // enough that the model's own large-loss widening is off for the short steps and on
          // for the long ones.
          const real_t mean = real_t(e) * real_t(len) / real_t(0.5 + len) * real_t(0.05);
          if (!(mean > real_t(0))) { continue; }
          double sum = 0;
          double sn_seen = 0;
          for (int i = 0; i < kSamples; ++i) {
            sum += em::sample_ion_fluctuation<real_t>(mats[mi], pd, real_t(e), tcut, tmax,
                                                      real_t(len), mean, pd.charge * pd.charge,
                                                      rng);
          }
          {
            const real_t siga = std::sqrt(em::ion_dispersion<real_t>(
                mats[mi], pd, real_t(e), tcut, tmax, real_t(len), pd.charge * pd.charge));
            sn_seen = (siga > 0) ? static_cast<double>(mean / siga) : 0.0;
          }
          const double got = sum / kSamples;
          const double dev = std::fabs(got / static_cast<double>(mean) - 1);
          const char* branch = (e > 79.45199) ? "universal"
                               : (sn_seen >= 2.0) ? "gauss"
                               : (sn_seen > 0.1)  ? "gamma"
                                                  : "uniform";
          std::printf("  %-22s %10.4g %9.4g %11.5g %11.5g %10.2e  %s\n", names[mi], e, len,
                      static_cast<double>(mean), got, dev, branch);
          // 1% of the mean at 400,000 samples. The uniform branch has a standard deviation of
          // meanLoss/sqrt(3), so its standard error alone is 0.09% - the limit is an order
          // above that and still an order below any bias worth having.
          if (dev > 0.01) {
            std::printf("  FAIL: mean not preserved (%s, %.4g MeV, %.4g mm): off by %.3e\n",
                        names[mi], e, len, dev);
            ++fails;
          }
        }
      }
    }
    std::printf("\n");
  }

  std::printf("%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
