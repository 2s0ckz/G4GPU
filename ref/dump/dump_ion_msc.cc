// G4UrbanMscModel's STEPPING half for an ion, for src/physics/em/urban_msc.cuh (P14b).
//
// Two files. `ion_msc_step.csv` is deterministic - the step limit and both directions of the
// true<->geometric path conversion. `ion_msc_sample.csv` is the angular distribution,
// N draws per cell at a fixed step.
//
// WHY THIS IS NOT `urban_msc.csv`. That file is `ComputeCrossSectionPerAtom` for eight species
// over Z 1..92, and the port has been exact against it for a long time. What was missing is
// everything the model does with that cross section once a TRACK has it: the step limit type an
// ion gets, facrange, the true/geometric detour, and the angle. None of it appears in a
// per-atom cross section.
//
// HOW A MODEL WITH NO CALLABLE SURFACE IS DRIVEN. `ComputeTruePathLengthLimit`,
// `ComputeGeomPathLength` and `ComputeTrueStepLength` mutate `G4UrbanMscModel`'s own members
// across calls and read a `G4Track` - the same problem docs/PORTED.md records for WentzelVI,
// which is why that row is checked as properties. The way out here is not to call the model at
// all but to drive the REAL PROCESS off the particle's own process manager:
//
//     g = proc->AlongStepGetPhysicalInteractionLength(track, 0, t_in, safety, &sel)
//     t = model->ComputeTrueStepLength(g)        // g == zPathLength, so this RETURNS tPathLength
//     t'= model->ComputeTrueStepLength(f*g)      // the inverse conversion, after a fresh set-up
//
// The second line is the trick that makes the whole chain readable: `ComputeTrueStepLength`
// starts with `if(geomStepLength == zPathLength) { return tPathLength; }`
// (G4UrbanMscModel.cc:739), so handing it back the geometric length the model just computed
// reports the true length that produced it without touching any state. Everything the process
// was configured with - the model, the step limit type, facrange, the lateral-displacement
// flag, the ionisation process the range comes from - is whatever QBBC set, read out rather
// than constructed here.
//
// EVERY ROW IS AT A GEOMETRY BOUNDARY, and that is a limitation worth stating rather than
// hiding. `ComputeTruePathLengthLimit` takes `presafety` from `sp->GetSafety()` when the
// pre-step status is `fGeomBoundary` and from `ComputeSafety(position, tPathLength)` otherwise
// - and the second goes through `G4Navigator::ComputeSafety`, which calls
// `LocateGlobalPointWithinVolume` and therefore assumes the navigator is already located in
// the volume the point is in. After `BeamOn(1)` it is located wherever the last track died, so
// a non-boundary row would compute a safety in the wrong volume. Setting the status to
// `fGeomBoundary` and calling `SetSafety` instead makes the safety an exact COLUMN and calls no
// navigator at all. What that costs is the two fMinimal states that are not reachable from a
// boundary - StartTracking's `tlimit = geombig` on the first step of a track, and the tlimit
// frozen from an earlier boundary - which are asserted as properties in tests/test_ion_msc.cu
// against the six lines of source quoted there, and are NOT oracled. Named, not approximated.
//
// THE RANGE AND THE MEAN FREE PATH ARE COLUMNS, from the same functions the model calls:
// `G4VMscModel::GetRange` is `ionisation->GetRange(E, couple)` on the eloss process
// `G4LossTableManager` holds for the particle, and `GetTransportMeanFreePath` is
// `1/(pFactor*CrossSectionPerVolume(material, part, E, 0, DBL_MAX))` with pFactor 1. Being
// columns, a disagreement in the conversion cannot be blamed on the port's range table (which
// `hadron_tables.csv` already checks) and vice versa.
//
// AND THERE IS NO TABLE FOR THE MEAN FREE PATH, which is the finding this file was written
// around. `G4VMscModel::GetParticleChangeForMSC` builds `xSectionTable` only for a particle
// that is not named "GenericIon" and is under 1 GeV (G4VMscModel.cc:94); `SetForceBuildTable`
// is called nowhere in 11.1.1. Alpha (3727 MeV), He3, triton, deuteron and GenericIon all fail
// that test, so `GetTransportMeanFreePath` evaluates the parameterisation at the energy asked
// for. docs/PORTED.md 4.3's rule - the reference is a table, not a model - is inverted here,
// and a port that built a log-grid lambda table for an ion would be wrong by its interpolation
// error rather than right by its physics.
//
// FIVE MATERIALS AND NOT LEAD. The dumper's material list is shared and append-only
// (docs/HADRONIC_PLAN.md section 6) and adding a material to it renumbers the index every
// other oracle CSV is keyed by, so this uses what is there: the four B1 materials, Zeff 7.22 to
// 13.8, plus `CustomDerivedI` (30% lead by mass, Zeff ~ 50) for the high-Z end. Z dependence of
// the cross section itself is covered to Z = 92 by `urban_msc.csv`; what this file adds over Z
// is the material-level Zeff coefficients - doverrb, coeffth1/2, coeffc1..4 - and five values
// of Zeff from 7 to 50 exercise all of them.
//
// SEEDS. One seed per sampler cell, from the cell's own indices, so a cell's numbers do not
// depend on how many cells ran before it.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <vector>

#include "G4Alpha.hh"
#include "G4DynamicParticle.hh"
#include "G4He3.hh"
#include "G4IonTable.hh"
#include "G4IonisParamMat.hh"
#include "G4LossTableManager.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleChangeForMSC.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4UrbanMscModel.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4VMultipleScattering.hh"
#include "G4VProcess.hh"
#include "Randomize.hh"

namespace {

struct Sp {
  const G4ParticleDefinition* def;
  const char* name;
  int z, a;
};

/// Alpha and He3 because QBBC gives each its own model-less `G4hMultipleScattering`, and three
/// real nuclides because since P8c an elastic recoil is transported as one: carbon and oxygen
/// are what a proton beam in water and tissue makes, and calcium is the heaviest thing bone
/// produces. Each concrete ion is a different mass AND a different bare charge, which is what
/// `SetParticle` reads, so the pair is varied rather than one at a time.
std::vector<Sp> species() {
  auto* it = G4IonTable::GetIonTable();
  return {
      {G4Alpha::Alpha(), "alpha", 2, 4},
      {G4He3::He3(), "He3", 2, 3},
      {it->GetIon(6, 12, 0.0), "C12", 6, 12},
      {it->GetIon(8, 16, 0.0), "O16", 8, 16},
      {it->GetIon(20, 40, 0.0), "Ca40", 20, 40},
  };
}

/// The multiple-scattering process on this particle's own manager, whatever QBBC put there.
/// Found by SUBTYPE and not by name: `fMultipleScattering` is 10 for every one of them, where
/// the name is "msc" for the light ions' own processes and for the shared ion one, so a name
/// match would also have to know which.
G4VMultipleScattering* msc_process(const G4ParticleDefinition* p) {
  auto* pm = p->GetProcessManager();
  if (nullptr == pm) { return nullptr; }
  auto* v = pm->GetProcessList();
  for (std::size_t i = 0; i < v->size(); ++i) {
    G4VProcess* pr = (*v)[static_cast<G4int>(i)];
    if (nullptr != pr && pr->GetProcessSubType() == 10) {
      return dynamic_cast<G4VMultipleScattering*>(pr);
    }
  }
  return nullptr;
}

const G4MaterialCutsCouple* couple_of(const G4Material* m) {
  auto* pct = G4ProductionCutsTable::GetProductionCutsTable();
  for (std::size_t i = 0; i < pct->GetTableSize(); ++i) {
    const G4MaterialCutsCouple* c = pct->GetMaterialCutsCouple(static_cast<G4int>(i));
    if (c->GetMaterial() == m) { return c; }
  }
  return nullptr;
}

/// `G4VMscModel::GetTransportMeanFreePath`'s no-table branch, which is the only branch an ion
/// takes. pFactor is 1 whenever the run has no base materials, which B1's four and the
/// dumper's three custom ones do not.
double transport_mfp(G4VEmModel* model, const G4Material* m, const G4ParticleDefinition* p,
                     double ekin) {
  const double x = model->CrossSectionPerVolume(m, p, ekin, 0.0, DBL_MAX);
  return (x > 0.0) ? 1.0 / x : DBL_MAX;
}

/// One fresh (track, step) pair in a given material at a given energy, at a boundary with a
/// chosen safety. Returned by pointer because G4Track owns the G4DynamicParticle and the
/// caller keeps both alive across the process calls.
struct Setup {
  G4DynamicParticle* dp;
  G4Track* track;
  G4Step* step;
};

Setup make_setup(const Sp& s, const G4Material* m, const G4MaterialCutsCouple* c, double ekin,
                 double safety) {
  auto* dp = new G4DynamicParticle(s.def, G4ThreeVector(0, 0, 1), ekin);
  auto* track = new G4Track(dp, 0.0, G4ThreeVector(0, 0, 0));
  auto* step = new G4Step();
  track->SetStep(step);
  step->SetTrack(track);
  G4StepPoint* pre = step->GetPreStepPoint();
  G4StepPoint* post = step->GetPostStepPoint();
  pre->SetMaterial(const_cast<G4Material*>(m));
  pre->SetMaterialCutsCouple(c);
  pre->SetPosition(G4ThreeVector(0, 0, 0));
  pre->SetMomentumDirection(G4ThreeVector(0, 0, 1));
  pre->SetKineticEnergy(ekin);
  pre->SetSafety(safety);
  pre->SetStepStatus(fGeomBoundary);  // see the header: this is what makes safety a column
  post->SetMaterial(const_cast<G4Material*>(m));
  post->SetMaterialCutsCouple(c);
  post->SetPosition(G4ThreeVector(0, 0, 0));
  post->SetMomentumDirection(G4ThreeVector(0, 0, 1));
  post->SetKineticEnergy(ekin);
  post->SetSafety(safety);
  post->SetStepStatus(fAlongStepDoItProc);
  track->SetKineticEnergy(ekin);
  track->SetMomentumDirection(G4ThreeVector(0, 0, 1));
  return {dp, track, step};
}

void free_setup(Setup& su) {
  delete su.step;
  delete su.track;  // deletes the dynamic particle
  su.step = nullptr;
  su.track = nullptr;
  su.dp = nullptr;
}

/// The five energies, per nucleon, and why these five. 0.05 and 0.5 MeV/u are the band an
/// elastic recoil lives in (a 200 MeV proton's oxygen recoils are tens of keV to a few MeV,
/// and their whole range is microns); 5 and 50 bracket the ions a shield sees; 210 MeV/u is
/// B1's 840 MeV alpha, which is the one number this package has to not move.
const double kEperA[6] = {0.05, 0.15, 0.5, 5.0, 50.0, 210.0};
constexpr int kNE = 6;

/// Proposed step lengths, as fractions of the range. 0.001 is below `tlimitminfix2` for a short
/// recoil and below `dtrl` for everything; 0.03 straddles nothing; 0.06 and 0.2 are above
/// `dtrl = 0.05`, which is the branch boundary in ComputeGeomPathLength; 1.0 is
/// `tPathLength == currentRange`, its own branch.
const double kTfrac[5] = {0.001, 0.01, 0.06, 0.2, 1.0};

constexpr int kN = 400000;
constexpr int kBins = 24;

}  // namespace

static void dump_ion_msc(const DumpContext& ctx) {
  const std::vector<Sp> sp = species();
  auto* ltm = G4LossTableManager::Instance();

  // Four B1 materials plus the lead-bearing custom one; see the header.
  std::vector<const G4Material*> mats;
  for (const G4Material* m : ctx.materials) {
    const G4String& n = m->GetName();
    if (n == "G4_AIR" || n == "G4_WATER" || n == "G4_A-150_TISSUE"
        || n == "G4_BONE_COMPACT_ICRU" || n == "CustomDerivedI") {
      mats.push_back(m);
    }
  }

  // ---------------------------------------------------------------- deterministic
  FILE* f = std::fopen("ion_msc_step.csv", "w");
  std::fprintf(f, "particle,Z,A,mass_MeV,charge,material,zeff,radlen_mm,doverra,doverrb,"
                  "step_limit_type,facrange,lat_disp,energy_MeV,range_mm,lambda0_mm,"
                  "e_rfin_MeV,lambda_rfin_mm,safety_mm,t_in_mm,t_out_mm,g_out_mm,g_frac,"
                  "t_back_mm\n");
  for (const Sp& s : sp) {
    if (nullptr == s.def) { continue; }
    G4VMultipleScattering* proc = msc_process(s.def);
    if (nullptr == proc) { continue; }
    G4VMscModel* model = proc->GetModelByIndex(0);
    if (nullptr == model) { continue; }
    G4VEnergyLossProcess* eloss = ltm->GetEnergyLossProcess(s.def);
    if (nullptr == eloss) { continue; }
    for (const G4Material* m : mats) {
      const G4MaterialCutsCouple* c = couple_of(m);
      if (nullptr == c) { continue; }
      const double zeff = m->GetIonisation()->GetZeffective();
      // The two Zeff coefficients that choose the distance estimate, recomputed here from the
      // one formula in InitialiseModelCache rather than read back (the cache has no accessor).
      // They are columns so the port is compared against the same two numbers, and the pair is
      // dumped rather than the one that is used, because which one is used is the finding:
      // `mass < masslimite` takes doverra and every species here takes doverrb.
      const double sqrtZ = std::sqrt(zeff);
      const double doverra = 9.6280e-1 - 8.4848e-2 * sqrtZ + 4.3769e-3 * zeff;
      const double doverrb = 1.15 - 9.76e-4 * zeff;
      for (int ie = 0; ie < kNE; ++ie) {
        const double e = kEperA[ie] * s.a * MeV;
        // GetRange and the mfp BEFORE StartTracking: CrossSectionPerVolume goes through
        // ComputeCrossSectionPerAtom, which calls SetParticle and would otherwise leave the
        // model's mass and charge set by whichever species was asked last.
        const double range = eloss->GetRange(e, c);
        const double lam0 = transport_mfp(model, m, s.def, e);
        for (int it = 0; it < 5; ++it) {
          const double t_in = kTfrac[it] * range;
          // Safety twice the doverrb distance, so `distance < presafety` fires and the step is
          // returned unlimited: t_out is then min(t_in, range) exactly and no random number is
          // drawn, which is what makes this comparison exact rather than statistical. The
          // branch that randomises is measured in the second table below.
          const double safety = 2.0 * range * doverrb;
          const double rfin = std::max(range - std::min(t_in, range), 0.01 * range);
          const double e_rfin = eloss->GetKineticEnergy(rfin, c);
          const double lam_rfin = transport_mfp(model, m, s.def, e_rfin);

          Setup su = make_setup(s, m, c, e, safety);
          proc->StartTracking(su.track);
          G4double dummy = 0.0;
          G4GPILSelection sel = NotCandidateForSelection;
          double t_req = t_in;
          const double g_out =
              proc->AlongStepGetPhysicalInteractionLength(*su.track, 0.0, t_req, dummy, &sel);
          const double t_out = model->ComputeTrueStepLength(g_out);
          free_setup(su);

          // The inverse conversion, on a fresh set-up so par1..par3 are the ones the forward
          // conversion above left rather than the ones a mutated zPathLength would give.
          const double g_frac = 0.5;
          Setup s2 = make_setup(s, m, c, e, safety);
          proc->StartTracking(s2.track);
          t_req = t_in;
          const double g2 =
              proc->AlongStepGetPhysicalInteractionLength(*s2.track, 0.0, t_req, dummy, &sel);
          const double t_back = model->ComputeTrueStepLength(g_frac * g2);
          free_setup(s2);

          std::fprintf(f,
                       "%s,%d,%d,%.17g,%.17g,%s,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%d,"
                       "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                       s.name, s.z, s.a, s.def->GetPDGMass() / MeV,
                       s.def->GetPDGCharge() / CLHEP::eplus, m->GetName().c_str(), zeff,
                       m->GetRadlen() / mm, doverra, doverrb, static_cast<int>(proc->StepLimitType()),
                       proc->RangeFactor(), proc->LateralDisplasmentFlag() ? 1 : 0, e / MeV,
                       range / mm, lam0 / mm, e_rfin / MeV, lam_rfin / mm, safety / mm,
                       t_in / mm, t_out / mm, g_out / mm, g_frac, t_back / mm);
        }
      }
    }
  }
  std::fclose(f);

  // ---------------------------------------------------------------- the randomised step limit
  //
  // Randomizetlimit is `max(G4RandGauss::shoot(tlimit, 0.1*(tlimit - tlimitmin)), tlimitmin)`,
  // so N draws of the returned true length recover BOTH of the two numbers the fMinimal branch
  // computes: the mean is tlimit and the standard deviation is a tenth of (tlimit - tlimitmin).
  // At N = 400,000 the standard error on the mean is 1.6e-4 of the sigma, i.e. 1.6e-5 of
  // tlimit, so facrange (0.2 against the lepton's 0.04) and the `max(tlimit, tlimitmin)` floor
  // are pinned far tighter than either could be wrong by.
  //
  // THREE SAFETIES PER CELL, AND THE OTHER TWO ARE WHAT MEASURES doverrb. With safety zero the
  // `distance < presafety` early return cannot fire and the fMinimal branch is always reached.
  // The other two sit either side of the threshold itself, at 0.99 and 1.01 of
  // `currentRange*doverrb`, and the test is strict (`if(distance < presafety)`): the 1.01 row
  // must return the step unlimited and the 0.99 row must not. Without them doverrb would be
  // compared as the dumper's transcription of one formula against the port's transcription of
  // the same formula, which docs/RISK.md V37 says is not an oracle. With them it is pinned to
  // 1% by whichever cells the limit actually bites in - and it bites in seven of 150, the ones
  // where the transport mfp is under five times the range.
  FILE* g = std::fopen("ion_msc_limit.csv", "w");
  std::fprintf(g, "particle,Z,A,material,energy_MeV,range_mm,lambda0_mm,lambda0_over_range,"
                  "doverrb,safety_over_distance,safety_mm,facrange,n,n_limited,mean_t_mm,"
                  "sd_t_mm,min_t_mm,max_t_mm\n");
  const double kSafetyFrac[3] = {0.0, 0.99, 1.01};
  int si = 0;
  for (const Sp& s : sp) {
    ++si;
    if (nullptr == s.def) { continue; }
    G4VMultipleScattering* proc = msc_process(s.def);
    G4VMscModel* model = (nullptr != proc) ? proc->GetModelByIndex(0) : nullptr;
    G4VEnergyLossProcess* eloss = ltm->GetEnergyLossProcess(s.def);
    if (nullptr == proc || nullptr == model || nullptr == eloss) { continue; }
    int mi = 0;
    for (const G4Material* m : mats) {
      ++mi;
      const G4MaterialCutsCouple* c = couple_of(m);
      if (nullptr == c) { continue; }
      const double zeff = m->GetIonisation()->GetZeffective();
      const double doverrb = 1.15 - 9.76e-4 * zeff;
      for (int ie = 0; ie < kNE; ++ie) {
        const double e = kEperA[ie] * s.a * MeV;
        const double range = eloss->GetRange(e, c);
        const double lam0 = transport_mfp(model, m, s.def, e);
        for (int isf = 0; isf < 3; ++isf) {
          const double safety = kSafetyFrac[isf] * range * doverrb;
          CLHEP::HepRandom::setTheSeed(770000UL + 100000UL * (unsigned long)si
                                       + 1000UL * (unsigned long)mi + 10UL * (unsigned long)ie
                                       + (unsigned long)isf);
          // Welford, not sum-of-squares. Most cells do not randomise at all - see the
          // lambda0_over_range column - and there `s2/n - mean^2` cancels down to 1e-14 of a
          // number of order 100, which prints as a spurious 1e-6 standard deviation. Welford
          // gives an exact zero, so `n_limited == 0` and `sd == 0` say the same thing twice
          // rather than one of them saying it wrongly.
          double mean = 0, m2 = 0, lo = DBL_MAX, hi = -DBL_MAX;
          long n_limited = 0;
          // 20,000 where the moments are the measurement; 200 on the threshold rows, where
          // what is being measured is which BRANCH ran and one draw would say it.
          const int n = (isf == 0) ? 20000 : 200;
          for (int k = 0; k < n; ++k) {
            Setup su = make_setup(s, m, c, e, safety);
            proc->StartTracking(su.track);
            G4double dummy = 0.0;
            G4GPILSelection sel = NotCandidateForSelection;
            double t_req = range;
            const double gg =
                proc->AlongStepGetPhysicalInteractionLength(*su.track, 0.0, t_req, dummy, &sel);
            const double t = model->ComputeTrueStepLength(gg);
            free_setup(su);
            if (t != range) { ++n_limited; }
            const double d = t - mean;
            mean += d / (k + 1);
            m2 += d * (t - mean);
            if (t < lo) { lo = t; }
            if (t > hi) { hi = t; }
          }
          std::fprintf(g, "%s,%d,%d,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%ld,"
                          "%.17g,%.17g,%.17g,%.17g\n",
                       s.name, s.z, s.a, m->GetName().c_str(), e / MeV, range / mm, lam0 / mm,
                       (range > 0.0 ? lam0 / range : 0.0), doverrb, kSafetyFrac[isf],
                       safety / mm, proc->RangeFactor(), n, n_limited, mean / mm,
                       std::sqrt(m2 / n) / mm, lo / mm, hi / mm);
        }
      }
    }
  }
  std::fclose(g);

  // ---------------------------------------------------------------- the angular distribution
  //
  // One set-up per cell, then N draws of `AlongStepDoIt` at the SAME geometric length the
  // set-up produced - which is free of state mutation for the reason the header gives:
  // `ComputeTrueStepLength(g)` with g == zPathLength returns tPathLength and changes nothing.
  // So every draw samples at one fixed (step, energy) and only SampleScattering consumes
  // random numbers.
  //
  // The displacement is a column and it must be exactly zero for every row. That is the whole
  // of `MuHadLateralDisplacement = false` as a measurement rather than a claim: a port that
  // left the lepton's displacement branch switched on would move a track sideways AND consume
  // three extra uniforms per step, which shifts everything downstream of it.
  // THE ENERGY THE SAMPLER USES IS A COLUMN, AND IT IS A TRANSCRIPTION RATHER THAN A READING.
  // `SampleScattering` starts by replacing the pre-step energy with a post-step one, in three
  // bands of tPathLength/currentRange (0.05 = dtrl, and 0.01), and G4UrbanMscModel exposes no
  // accessor for the result. `scatter_energy_MeV` below is those three lines evaluated here
  // with the same `GetEnergy`/`GetDEDX` the model calls, so the angular comparison is at the
  // same energy on both sides rather than at two energies that differ by the port's own guess.
  // It is NOT an independent oracle for those three lines - docs/RISK.md V37 - which is why the
  // step fractions include 0.005: below 0.01 of the range the bands collapse and
  // scatter_energy_MeV is the pre-step energy exactly, so those rows test the sampler with
  // nothing transcribed at all, and the other two extend it to the branch where
  // `currentKinEnergy != kinEnergy` and the tau re-derivation and the invbetacp sqrt turn on.
  FILE* h = std::fopen("ion_msc_sample.csv", "w");
  // `sd_one_minus_cost` IS A COLUMN AND mean_cost2 IS NOT ENOUGH. An ion's steps sit at
  // tau ~ 1e-6, where <cos> is 1 - 1e-6 and <cos^2> is 1 - 2e-6: a test that reconstructs
  // var(1 - cos) from those two loses every significant digit to cancellation and then reports
  // a two-digit variance as if it were exact, which is how a perfectly good distribution
  // comparison comes out at 145 sigma. Welford on (1 - cos) directly, so the standard error the
  // port compares against is the real one.
  std::fprintf(h, "particle,Z,A,material,energy_MeV,range_mm,lambda0_mm,t_frac,t_mm,g_mm,tau,"
                  "scatter_energy_MeV,lambda_scat_mm,n,n_scattered,mean_cost,mean_cost2,"
                  "mean_one_minus_cost,sd_one_minus_cost,mean_disp_mm,hist_lo,hist_hi");
  for (int b = 0; b < kBins; ++b) { std::fprintf(h, ",h%d", b); }
  std::fprintf(h, "\n");
  si = 0;
  for (const Sp& s : sp) {
    ++si;
    if (nullptr == s.def) { continue; }
    G4VMultipleScattering* proc = msc_process(s.def);
    G4VMscModel* model = (nullptr != proc) ? proc->GetModelByIndex(0) : nullptr;
    G4VEnergyLossProcess* eloss = ltm->GetEnergyLossProcess(s.def);
    if (nullptr == proc || nullptr == model || nullptr == eloss) { continue; }
    int mi = 0;
    for (const G4Material* m : mats) {
      ++mi;
      const G4MaterialCutsCouple* c = couple_of(m);
      if (nullptr == c) { continue; }
      for (int ie = 2; ie < kNE; ++ie) {  // 0.05 MeV/u has a range of nanometres; skip it here
        const double e = kEperA[ie] * s.a * MeV;
        const double range = eloss->GetRange(e, c);
        const double lam0 = transport_mfp(model, m, s.def, e);
        const double tfrac[3] = {0.005, 0.05, 0.3};
        for (int it = 0; it < 3; ++it) {
          const double t_in = tfrac[it] * range;
          // Safety above the doverrb distance for every Zeff (doverrb <= 1.15 - 9.76e-4*Zeff
          // is at most 1.15), so the early return fires and the step limit draws nothing: the
          // sampler's random numbers are the only ones this cell consumes.
          Setup su = make_setup(s, m, c, e, 3.0 * range);
          proc->StartTracking(su.track);
          G4double dummy = 0.0;
          G4GPILSelection sel = NotCandidateForSelection;
          double t_req = t_in;
          const double gg =
              proc->AlongStepGetPhysicalInteractionLength(*su.track, 0.0, t_req, dummy, &sel);
          const double t_true = model->ComputeTrueStepLength(gg);
          su.step->SetStepLength(gg);
          // G4UrbanMscModel::SampleScattering lines 786-792, with the model's own accessors.
          double e_scat = e;
          if (t_true > range * 0.05) {
            e_scat = eloss->GetKineticEnergy(range - t_true, c);
          } else if (t_true > range * 0.01) {
            e_scat = e - t_true * eloss->GetDEDX(e, c);
          }
          const double lam_scat = transport_mfp(model, m, s.def, e_scat);
          CLHEP::HepRandom::setTheSeed(880000UL + 10000UL * (unsigned long)si
                                       + 1000UL * (unsigned long)mi + 10UL * (unsigned long)ie
                                       + (unsigned long)it);
          long hh[kBins] = {0};
          double a1 = 0, a2 = 0, a3 = 0, ad = 0;
          double om = 0, om2 = 0;  // Welford on (1 - cos); see the header note
          long nsc = 0;
          const double lo = -12.0, hi = 0.31;  // log10(1 - cos), 1e-12 to 2
          for (int k = 0; k < kN; ++k) {
            auto* pc = static_cast<G4ParticleChangeForMSC*>(proc->AlongStepDoIt(*su.track,
                                                                                *su.step));
            const G4ThreeVector* nd = pc->GetProposedMomentumDirection();
            const double cost = (nullptr != nd) ? nd->z() : 1.0;
            const G4ThreeVector& np = pc->GetProposedPosition();
            ad += np.mag();
            a1 += cost;
            a2 += cost * cost;
            a3 += 1.0 - cost;
            const double omd = (1.0 - cost) - om;
            om += omd / (k + 1);
            om2 += omd * ((1.0 - cost) - om);
            if (cost < 1.0) {
              ++nsc;
              const double x = std::log10(1.0 - cost);
              int b = int(kBins * (x - lo) / (hi - lo));
              if (b < 0) { b = 0; }
              if (b >= kBins) { b = kBins - 1; }
              ++hh[b];
            }
          }
          std::fprintf(h, "%s,%d,%d,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                          "%d,%ld,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
                       s.name, s.z, s.a, m->GetName().c_str(), e / MeV, range / mm, lam0 / mm,
                       tfrac[it], t_true / mm, gg / mm, t_true / lam0, e_scat / MeV,
                       lam_scat / mm, kN, nsc, a1 / kN, a2 / kN, om,
                       std::sqrt(om2 / kN), ad / kN, lo, hi);
          for (int b = 0; b < kBins; ++b) { std::fprintf(h, ",%ld", hh[b]); }
          std::fprintf(h, "\n");
          free_setup(su);
        }
      }
    }
  }
  std::fclose(h);
  std::printf("wrote ion_msc_step.csv ion_msc_limit.csv ion_msc_sample.csv\n");
}

G4GPU_REGISTER_DUMP("ion_msc", "ion_msc_step.csv ion_msc_limit.csv ion_msc_sample.csv",
                    dump_ion_msc);
