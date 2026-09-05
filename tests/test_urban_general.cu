// G4UrbanMscModel::ComputeCrossSectionPerAtom for every mass and charge, against Geant4's own.
//
// WHY THIS EXISTS
//
// Urban is not the electron's multiple-scattering model, which is how this port came to
// transcribe only the electron half of it. `G4hMultipleScattering::InitialiseProcess` defaults
// to `new G4UrbanMscModel()`, and `G4EmBuilder::ConstructIonEmPhysics` hands alpha, He3,
// deuteron, triton and GenericIon a *fresh* `G4hMultipleScattering` with no model set:
//
//     part = G4Alpha::Alpha();
//     ph->RegisterProcess(new G4hMultipleScattering(), part);      // -> Urban
//
// Only the light hadrons and the muons get WentzelVI, and only because
// `ConstructLightHadrons` calls `SetEmModel(new G4WentzelVIModel())` on theirs. So an alpha -
// a species this port transports today - scatters by Urban in Geant4 and was scattering by
// WentzelVI here.
//
// WHAT IS COMPARED
//
// `ref/oracle/urban_msc.csv` is `ComputeCrossSectionPerAtom` for eight particles, every Z from
// 1 to 92, and 1 keV to 10 GeV at eight points per decade. It is deterministic, so this is an
// exact comparison rather than a statistical one, and it pins the whole function: the
// heavy-particle velocity map, the eps branches, the Z interpolation, both coefficient tables,
// the 10 MeV handover and the high-energy branch.
//
// Split by particle, because the failure this is guarding against is mass-specific and a
// single worst case would report whichever mass happened to be worst:
//
//   * the velocity map. A heavy particle is mapped onto the electron of the same velocity and
//     the electron formula is run on *that* energy. Omit the map and an alpha's cross section
//     is still a plausible number with the wrong energy dependence.
//   * the 10 MeV boundary is in the mapped energy, so for an alpha it sits near 8 GeV of
//     kinetic energy. A test that only ran to 10 MeV would never reach the branch.
//   * the coefficient table is chosen by the *sign* of the charge, so a proton and an alpha
//     read the positron table. That is Geant4's rule as written; it looks like a bug and is
//     not one.
//   * chargeSquare scales the whole thing, so an alpha is four protons unless it is not.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/particle.cuh"
#include "data/materials.cuh"
#include "physics/em/urban_msc.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

struct Row {
  std::string part;
  int z = 0;
  double e = 0, xs = 0;
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
    char name[32];
    Row r;
    if (std::sscanf(line, "%31[^,],%d,%lf,%lf", name, &r.z, &r.e, &r.xs) != 4) { continue; }
    r.part = name;
    out.push_back(r);
  }
  std::fclose(f);
  return out;
}

struct Species {
  const char* name;
  ParticleType type;
};

const Species kSpecies[] = {
    {"e-", ParticleType::kElectron},   {"e+", ParticleType::kPositron},
    {"proton", ParticleType::kProton}, {"anti_proton", ParticleType::kAntiProton},
    {"alpha", ParticleType::kAlpha},   {"He3", ParticleType::kHe3},
    {"mu-", ParticleType::kMuonMinus}, {"pi+", ParticleType::kPionPlus},
};
constexpr int kNPart = static_cast<int>(sizeof kSpecies / sizeof kSpecies[0]);

struct Cell {
  int n = 0;
  double worst = 0;
  std::string where;
};

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  const auto rows = load(dir + "/urban_msc.csv");
  if (rows.empty()) {
    std::printf("cannot read %s/urban_msc.csv - run ref/oracle/run.bat first\n", dir.c_str());
    return 1;
  }

  int fails = 0;
  Cell cells[kNPart];
  int zmin = 999, zmax = 0;
  double emin = 1e30, emax = 0;

  for (const Row& r : rows) {
    int pi = -1;
    for (int i = 0; i < kNPart; ++i) {
      if (r.part == kSpecies[i].name) { pi = i; break; }
    }
    if (pi < 0) { continue; }
    const ParticleDef<real_t> pd = particle_def<real_t>(kSpecies[pi].type);
    const real_t ours =
        em::urban_xs_per_atom<real_t>(real_t(r.z), real_t(r.e), pd.mass, pd.charge);

    if (r.z < zmin) { zmin = r.z; }
    if (r.z > zmax) { zmax = r.z; }
    if (r.e < emin) { emin = r.e; }
    if (r.e > emax) { emax = r.e; }

    Cell& c = cells[pi];
    ++c.n;
    // Relative where the reference is non-zero. It is genuinely zero nowhere on this grid,
    // but a zero would otherwise divide and report a silent NaN as a pass.
    const double dev = (r.xs > 1e-300) ? std::fabs(ours / r.xs - 1)
                                       : (std::fabs(static_cast<double>(ours)) > 1e-300 ? 1.0
                                                                                        : 0.0);
    if (dev > c.worst) {
      c.worst = dev;
      char buf[200];
      std::snprintf(buf, sizeof buf, "Z=%d at %.5g MeV (ours %.9g, G4 %.9g mm^2)", r.z, r.e,
                    static_cast<double>(ours), r.xs);
      c.where = buf;
    }
  }

  std::printf("== G4UrbanMscModel::ComputeCrossSectionPerAtom ==\n");
  std::printf("  Z %d..%d, %.4g..%.4g MeV\n\n", zmin, zmax, emin, emax);

  // 1e-12. There is nothing between this port's answer and Geant4's but double-precision
  // arithmetic over the same two coefficient tables, so what this leaves room for is a
  // compiler reassociating a sum, and none at all for a missing velocity map, a table read
  // with the wrong sign convention, or chargeSquare left at one.
  constexpr double kLimit = 1e-12;
  for (int i = 0; i < kNPart; ++i) {
    const Cell& c = cells[i];
    std::printf("  %-12s %7d points  worst %10.3e  %s\n", kSpecies[i].name, c.n, c.worst,
                c.where.c_str());
    if (c.n == 0) {
      std::printf("  FAIL: no oracle rows for %s\n", kSpecies[i].name);
      ++fails;
    } else if (c.worst > kLimit) {
      std::printf("  FAIL: %s is off by %.3e, limit %.0e\n", kSpecies[i].name, c.worst,
                  kLimit);
      ++fails;
    }
  }

  // The heavy-particle branch has to have been reached above the mapped 10 MeV boundary, which
  // for an alpha is about 8 GeV of kinetic energy. Without a point up there the high-energy
  // branch is untested and every check above still passes.
  if (emax < 5e3) {
    std::printf("  FAIL: the oracle stops at %.4g MeV, below the alpha's mapped 10 MeV\n",
                emax);
    ++fails;
  }
  if (zmin > 1 || zmax < 92) {
    std::printf("  FAIL: Z runs %d..%d; the branches below Zdat[0] and above Zdat[14] need"
                " Z=1 and Z=92\n", zmin, zmax);
    ++fails;
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
