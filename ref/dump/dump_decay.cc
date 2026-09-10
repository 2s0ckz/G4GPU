// The oracle for P4 (decay): Geant4 11.1.1's own decay tables, lifetimes, interaction
// lengths, boost and sampled final states.
//
// Eleven files, and the split is along the line between what has an exact answer and what has
// only a distribution:
//
//   decay_applicable.csv  every particle in the table: lifetime, stable flag, whether
//                         G4Decay::IsApplicable accepts it, how many channels it has, and
//                         the two pre-assigned-decay fields of a freshly built
//                         G4DynamicParticle. The last two are the point: the plan requires
//                         the pre-assigned path be shown unused rather than assumed unused.
//   decay_tables.csv      one row per channel: kinematics name, branching ratio, daughters
//                         in G4VDecayChannel's own order, IsOKWithParentMass, and the
//                         two-body momentum where the channel has one. Exact.
//   decay_process.csv     G4Decay::GetMeanFreePath and ::GetMeanLifeTime over a grid of
//                         (species, kinetic energy) that crosses the gamma = 20 branch.
//                         Exact.
//   decay_select.csv      G4DecayTable::SelectADecayChannel sampled, at the PDG mass and at
//                         masses that CLOSE channels - which is where the algorithm stops
//                         agreeing with a clean rewrite. See the comment on dump_select.
//   decay_dalitz.csv      G4KL3DecayChannel::DalitzDensity on a grid, which is what pins the
//                         KL3 form factors. The sampled spectrum cannot: see dump_dalitz.
//   decay_atrest.csv      every AT-REST process QBBC registers, per species, with the at-rest
//                         interaction length each offers a stopped track. The plan says a
//                         stopped pi- is captured rather than decayed; this is the
//                         measurement, and it says something stronger than the plan does -
//                         see the comment on dump_atrest.
//   decay_boost.csv       G4DecayProducts::Boost applied to a FIXED rest-frame product set,
//                         so the boost is separated from the sampling. Exact.
//   decay_moments.csv     per channel and per product SLOT: the first two moments of the
//                         kinetic energy, its range, and the first two moments of cos(theta)
//                         against a fixed axis. Statistical.
//   decay_pairs.csv       per channel and per pair of slots: the first two moments of the
//                         cosine of the opening angle. Statistical, and it is the part of the
//                         final state that isotropy does not already determine.
//   decay_hist.csv        the kinetic energy spectrum of every product slot, 40 bins.
//                         Statistical, and the only thing that can see the SHAPE - a
//                         mis-transcribed Dalitz density or Michel spectrum moves a histogram
//                         long before it moves a mean.
//   decay_closure.csv     how well each channel's own four-momenta close, worst over the
//                         sample. Not a physics quantity - a measurement of the channel's
//                         arithmetic, so that the port can be required to lose precision
//                         exactly where Geant4 loses it instead of against a chosen bound.
//                         See the comment on sample_channel; two of these channels do not
//                         close anywhere near double precision and both have a reason.
//
// WHY BOTH MOMENTS AND HISTOGRAMS. A mean alone cannot see a redistribution that keeps the
// centre: two of these channels are rejection samplers against a shape (the muon's Michel
// spectrum, KL3's Dalitz density) and getting the shape's exponent wrong while keeping its
// support would pass on the mean. And a histogram alone is noisy per bin; the moments are
// what give a tight number. docs/RISK.md has the same argument for the Rayleigh angular dump.
//
// WHY THE PAIR ANGLES. The products of every channel here are isotropic in the parent rest
// frame, so <cos theta> against a fixed axis is zero and <cos^2> is 1/3 for every slot of
// every channel - which makes them a check that the overall rotation is uniform and nothing
// else. What carries the channel's actual angular content is the angle BETWEEN two products:
// pi0 -> gamma gamma is exactly back to back, K -> pi pi0 is exactly back to back, a
// three-body channel is not, and the muon's electron-neutrino correlation is fixed by the
// energy sharing through cos = 1 - 2/Ee - 2/Ene + 2/(Ee Ene). A transposed pair of daughters
// or a wrong rotation composition shows up here and nowhere else.
//
// SEEDS. Every sampled block sets CLHEP::HepRandom::setTheSeed to a fixed value derived from
// what is being sampled, so the file is reproducible run to run - the same rule the existing
// rayleigh_angular dump follows.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "G4DalitzDecayChannel.hh"
#include "G4Decay.hh"
#include "G4DecayProducts.hh"
#include "G4DecayTable.hh"
#include "G4DynamicParticle.hh"
#include "G4KL3DecayChannel.hh"
#include "G4MuonDecayChannel.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4PhaseSpaceDecayChannel.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4VDecayChannel.hh"
#include "G4VProcess.hh"
#include "Randomize.hh"

namespace {

/// The species this package transcribes: the unstable ones QBBC transports, plus the neutron
/// (unstable, transported, and killed by the tracking cut long before it decays). Named
/// rather than discovered so that the port's table and the oracle cover the same set and a
/// species added to one without the other is a missing row rather than a silent gap.
const char* const kSpecies[] = {"mu+", "mu-", "pi+", "pi-", "pi0", "kaon+", "kaon-", "neutron"};
constexpr int kNumSpecies = 8;

G4ParticleDefinition* find(const char* name) {
  return G4ParticleTable::GetParticleTable()->FindParticle(G4String(name));
}

/// G4Decay::GetMeanFreePath and ::GetMeanLifeTime are `protected` - they are the process
/// framework's interface to the stepper, not a public query. Re-exported here rather than
/// reached through PostStepGetPhysicalInteractionLength, which would multiply them by a random
/// number and turn an exact oracle into a sampled one.
class DecayProbe : public G4Decay {
 public:
  using G4Decay::GetMeanFreePath;
  using G4Decay::GetMeanLifeTime;
};

/// The kinematic limit on daughter `slot`'s kinetic energy: it is largest when every other
/// daughter moves together as one system of mass equal to the sum of their masses, so the
/// two-body formula with (m_slot, sum of the rest) is exact for any number of daughters.
/// Zero is the lower limit for a three-or-more-body channel and equals the upper one for a
/// two-body channel - which the min/max columns of decay_moments.csv show directly.
double slot_max_kinetic_energy(double parent_mass, G4VDecayChannel* ch, int slot) {
  const G4int nd = ch->GetNumberOfDaughters();
  const double m = ch->GetDaughter(slot)->GetPDGMass();
  double rest = 0.0;
  for (G4int d = 0; d < nd; ++d) {
    if (d != slot) { rest += ch->GetDaughter(d)->GetPDGMass(); }
  }
  const double p = G4PhaseSpaceDecayChannel::Pmx(parent_mass, m, rest);
  if (p <= 0.0) { return 0.0; }
  return std::sqrt(p * p + m * m) - m;
}

/// G4KL3DecayChannel::DalitzDensity is `protected`; re-exported the same way DecayProbe
/// re-exports G4Decay's lengths. The two form factor parameters do NOT need this - the data
/// members pLambda and pXi0 are private but `GetDalitzParameterLambda()` and
/// `GetDalitzParameterXi()` are public inline accessors (G4KL3DecayChannel.hh:52), which is
/// worth writing down because this package spent a while believing they were unreachable and
/// building a spectrum-shape argument around it.
///
/// A probe has to be CONSTRUCTED rather than cast from the table's channel: the table holds a
/// G4KL3DecayChannel* and there is no downcast to a class it does not know. Constructing one
/// with the same (parent, pion, lepton, neutrino) names runs the same constructor and so
/// selects the same parameters - which is checked, not assumed, against the accessors on the
/// channel in the real table.
class KL3Probe : public G4KL3DecayChannel {
 public:
  KL3Probe(const G4String& parent, G4double br, const G4String& pion, const G4String& lepton,
           const G4String& nu)
      : G4KL3DecayChannel(parent, br, pion, lepton, nu) {}
  using G4KL3DecayChannel::DalitzDensity;
};

/// G4PhaseSpaceDecayChannel::Pmx is public and static, so the two-body kinematic limit can be
/// asked of Geant4 itself rather than recomputed here.
double two_body_pmax(double parent_mass, const G4VDecayChannel* ch) {
  if (ch->GetNumberOfDaughters() != 2) { return -1.0; }
  G4VDecayChannel* c = const_cast<G4VDecayChannel*>(ch);
  const double m0 = c->GetDaughter(0)->GetPDGMass();
  const double m1 = c->GetDaughter(1)->GetPDGMass();
  return G4PhaseSpaceDecayChannel::Pmx(parent_mass, m0, m1);
}

// -------------------------------------------------------------------------------------
void dump_applicable(FILE* f) {
  std::fprintf(f,
               "name,pdg,mass_MeV,width_MeV,lifetime_ns,stable,is_applicable,has_table,"
               "n_channels,preassigned_proper_time_ns,has_preassigned_products\n");
  G4Decay decay;
  G4ParticleTable* table = G4ParticleTable::GetParticleTable();
  auto* it = table->GetIterator();
  it->reset();
  while ((*it)()) {
    G4ParticleDefinition* p = it->value();
    // Real nuclei are generated on demand and there can be thousands of them in a table
    // after a hadronic event; the decay question for a nucleus is G4RadioactiveDecay's, which
    // is not in QBBC's chain (HADRONIC_PLAN section 2). The LIGHT ions are kept - deuteron,
    // triton, He3, alpha - because the triton is the counter-example that this file exists to
    // record: it is flagged STABLE and carries a 17.774 year lifetime, so G4Decay::IsApplicable
    // (which reads the lifetime) accepts it while GetMeanFreePath and DecayIt (which read the
    // flag) refuse to do anything with it.
    if (p->GetParticleType() == "nucleus" && p->GetBaryonNumber() > 4) { continue; }
    G4DecayTable* dt = p->GetDecayTable();
    // A fresh G4DynamicParticle, exactly as a primary generator or a secondary constructor
    // would make one, so that its pre-assigned-decay fields are the DEFAULTS. This is the
    // measurement that says G4Decay's pre-assigned branch is unreachable in this port.
    G4DynamicParticle dp(p, G4ThreeVector(0, 0, 1), 1.0 * MeV);
    std::fprintf(f, "%s,%d,%.17g,%.17g,%.17g,%d,%d,%d,%d,%.17g,%d\n",
                 p->GetParticleName().c_str(), p->GetPDGEncoding(), p->GetPDGMass() / MeV,
                 p->GetPDGWidth() / MeV, p->GetPDGLifeTime() / ns, p->GetPDGStable() ? 1 : 0,
                 decay.IsApplicable(*p) ? 1 : 0, (dt != nullptr) ? 1 : 0,
                 (dt != nullptr) ? dt->entries() : 0,
                 dp.GetPreAssignedDecayProperTime() / ns,
                 (dp.GetPreAssignedDecayProducts() != nullptr) ? 1 : 0);
  }
}

// -------------------------------------------------------------------------------------
void dump_tables(FILE* f) {
  std::fprintf(f,
               "parent,parent_pdg,parent_mass_MeV,parent_width_MeV,parent_lifetime_ns,"
               "parent_stable,n_channels,channel,kinematics,br,n_daughters,"
               "d0,d0_pdg,d0_mass_MeV,d0_width_MeV,d1,d1_pdg,d1_mass_MeV,d1_width_MeV,"
               "d2,d2_pdg,d2_mass_MeV,d2_width_MeV,sum_daughter_mass_MeV,"
               "is_ok_with_pdg_mass,two_body_pmax_MeV,kl3_lambda,kl3_xi0\n");
  for (int s = 0; s < kNumSpecies; ++s) {
    G4ParticleDefinition* p = find(kSpecies[s]);
    G4DecayTable* dt = p->GetDecayTable();
    const double pm = p->GetPDGMass();
    for (G4int i = 0; i < dt->entries(); ++i) {
      G4VDecayChannel* ch = dt->GetDecayChannel(i);
      const G4int nd = ch->GetNumberOfDaughters();
      double sum = 0.0;
      std::string dn[3] = {"", "", ""};
      int dpdg[3] = {0, 0, 0};
      double dm[3] = {0, 0, 0};
      double dw[3] = {0, 0, 0};
      for (G4int d = 0; d < nd && d < 3; ++d) {
        G4ParticleDefinition* dd = ch->GetDaughter(d);
        dn[d] = dd->GetParticleName();
        dpdg[d] = dd->GetPDGEncoding();
        dm[d] = dd->GetPDGMass() / MeV;
        dw[d] = dd->GetPDGWidth() / MeV;
      }
      for (G4int d = 0; d < nd; ++d) { sum += ch->GetDaughter(d)->GetPDGMass() / MeV; }
      // The KL3 form factors, straight off the channel in the real table. Zero for every
      // other kind, which is also what the port's ChannelRow stores.
      double lambda = 0.0, xi0 = 0.0;
      if (auto* kl3 = dynamic_cast<G4KL3DecayChannel*>(ch)) {
        lambda = kl3->GetDalitzParameterLambda();
        xi0 = kl3->GetDalitzParameterXi();
      }
      std::fprintf(f,
                   "%s,%d,%.17g,%.17g,%.17g,%d,%d,%d,%s,%.17g,%d,"
                   "%s,%d,%.17g,%.17g,%s,%d,%.17g,%.17g,%s,%d,%.17g,%.17g,"
                   "%.17g,%d,%.17g,%.17g,%.17g\n",
                   p->GetParticleName().c_str(), p->GetPDGEncoding(), pm / MeV,
                   p->GetPDGWidth() / MeV, p->GetPDGLifeTime() / ns,
                   p->GetPDGStable() ? 1 : 0, dt->entries(), i,
                   ch->GetKinematicsName().c_str(), ch->GetBR(), nd, dn[0].c_str(), dpdg[0],
                   dm[0], dw[0], dn[1].c_str(), dpdg[1], dm[1], dw[1], dn[2].c_str(), dpdg[2],
                   dm[2], dw[2], sum, ch->IsOKWithParentMass(pm) ? 1 : 0,
                   two_body_pmax(pm, ch) / MeV, lambda, xi0);
    }
  }
}

// -------------------------------------------------------------------------------------
/// G4KL3DecayChannel::DalitzDensity on a grid, for all four KL3 channels of K+ and K-.
///
/// THIS FILE EXISTS BECAUSE THE SPECTRUM CANNOT SEE THE FORM FACTORS. The argument was that
/// pLambda and pXi0 are unreachable and that the only check on them is the shape of the
/// sampled pion spectrum. Both halves were wrong. The parameters have public accessors, and
/// the spectrum is nowhere near sharp enough: substituting K0L's pLambda = 0.0300 for K+'s
/// 0.0286 changes F = 1 + lambda*q2/m_pi^2 by 1.6% at the edge of the Dalitz region and less
/// inside it, which moves the sampled pion spectrum by a few parts in a thousand - a third of
/// a sigma per bin at 400,000 samples. It was measured: the perturbation passed every
/// assertion in tests/test_decay.cu. And pXi0 for Ke3 multiplies m_e^2 = 0.261 MeV^2 against
/// a coefficient of order m_K^3, so it is invisible in the spectrum at any sample size.
///
/// Dumped as a deterministic grid, the density pins both parameters and the whole Chounet
/// formula to machine precision. The grid is in KINETIC energies, because that is what
/// DalitzDensity takes (its first three lines add the masses back); it sweeps the pion and
/// lepton energies over the released energy and takes the neutrino's from what is left, and
/// keeps the points where that is non-negative. Rho/RhoMax is negative over part of that
/// region, which is fine and is why the acceptance test in DecayIt is `r <= w` and not `|w|`.
void dump_dalitz(FILE* f) {
  std::fprintf(f, "parent,lepton,lambda,xi0,mass_k_MeV,mass_pi_MeV,mass_l_MeV,mass_nu_MeV,"
                  "epi_MeV,el_MeV,enu_MeV,density\n");
  struct Mode { const char* parent; const char* pion; const char* lepton; const char* nu; };
  const Mode modes[] = {{"kaon+", "pi0", "e+", "nu_e"},
                        {"kaon+", "pi0", "mu+", "nu_mu"},
                        {"kaon-", "pi0", "e-", "anti_nu_e"},
                        {"kaon-", "pi0", "mu-", "anti_nu_mu"}};
  constexpr int kSteps = 16;
  for (const Mode& mode : modes) {
    KL3Probe probe(mode.parent, 1.0, mode.pion, mode.lepton, mode.nu);
    const double mk = find(mode.parent)->GetPDGMass();
    const double mpi = find(mode.pion)->GetPDGMass();
    const double ml = find(mode.lepton)->GetPDGMass();
    const double mnu = find(mode.nu)->GetPDGMass();
    const double q = mk - mpi - ml - mnu;
    for (int i = 0; i <= kSteps; ++i) {
      for (int j = 0; j <= kSteps; ++j) {
        const double epi = q * i / double(kSteps);
        const double el = q * j / double(kSteps);
        const double enu = q - epi - el;
        if (enu < 0.0) { continue; }
        const double d = probe.DalitzDensity(mk, epi, el, enu, mpi, ml, mnu);
        std::fprintf(f, "%s,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     mode.parent, mode.lepton, probe.GetDalitzParameterLambda(),
                     probe.GetDalitzParameterXi(), mk / MeV, mpi / MeV, ml / MeV, mnu / MeV,
                     epi / MeV, el / MeV, enu / MeV, d);
      }
    }
  }
}

// -------------------------------------------------------------------------------------
/// G4DecayTable::SelectADecayChannel, sampled - the frequency with which each channel index
/// is chosen, at the PDG mass and at masses that CLOSE some of the channels.
///
/// The frequencies at the PDG mass are just BR/sumBR and could have been asserted from the
/// table. The reduced masses are why this dump exists, because the algorithm's answer there
/// is not what a clean rewrite produces. `br = sumBR * rand` is drawn against the sum of the
/// OPEN channels, but the walk accumulates `sum += GetBR()` over EVERY channel and only then
/// tests the mass. Two consequences, and both are visible in this CSV:
///
///   A closed channel's slice of the cumulative axis is absorbed by the next OPEN channel
///   after it in table order - not redistributed in proportion, and not skipped.
///
///   Any channel whose cumulative bound lies beyond sumBR is NEVER selected, because
///   `br < sumBR` always. For kaon+ at 0.82 of its mass, where pi+pi+pi- and pi+pi0pi0 are
///   both shut, Kmu3 has a selection probability of exactly zero.
///
/// A table with every channel open cannot see either rule, which is why the fractions below
/// go low enough to shut some. For kaon+ 0.84 shuts pi+pi+pi- and 0.82 also shuts pi+pi0pi0;
/// measured, Kmu3 is selected 0 times out of 200,000 at both, with a branching ratio of
/// 0.0335 and IsOKWithParentMass true. Where every channel of a species shuts - pi+/pi- at
/// 0.70, the neutron at anything below 0.999 - SelectADecayChannel returns nullptr
/// (sumBR == 0, G4DecayTable.cc:93), which is the port's kNoChannel path with an oracle.
///
/// Those all-shut cases are sampled 100 times and not 200,000. G4DecayTable prints "no
/// possible DecayChannel" to G4cout on every one of them, unconditionally and with no
/// verbosity gate, and at 200,000 draws per species that was 87 MB on the console of a shared
/// oracle run. A hundred draws prove the answer is nullptr every time just as well.
void dump_select(FILE* f) {
  std::fprintf(f, "parent,parent_pdg,mass_fraction,parent_mass_MeV,n_channels,samples,"
                  "n_null,channel,br,is_ok,count\n");
  constexpr int kDraws = 200000;
  constexpr int kClosedDraws = 100;
  const double fractions[] = {1.0, 0.86, 0.84, 0.82, 0.70};
  for (int s = 0; s < kNumSpecies; ++s) {
    G4ParticleDefinition* p = find(kSpecies[s]);
    G4DecayTable* dt = p->GetDecayTable();
    const G4int nch = dt->entries();
    for (double frac : fractions) {
      const double pm = p->GetPDGMass() * frac;
      bool any_open = false;
      for (G4int c = 0; c < nch; ++c) {
        any_open = any_open || dt->GetDecayChannel(c)->IsOKWithParentMass(pm);
      }
      const int draws = any_open ? kDraws : kClosedDraws;
      std::vector<int> count(static_cast<std::size_t>(nch), 0);
      int n_null = 0;
      CLHEP::HepRandom::setTheSeed(505000UL + 100UL * (unsigned long)(s + 1) +
                                   (unsigned long)(frac * 100.0));
      for (int i = 0; i < draws; ++i) {
        G4VDecayChannel* ch = dt->SelectADecayChannel(pm);
        if (ch == nullptr) {
          ++n_null;
          continue;
        }
        for (G4int c = 0; c < nch; ++c) {
          if (dt->GetDecayChannel(c) == ch) {
            ++count[static_cast<std::size_t>(c)];
            break;
          }
        }
      }
      for (G4int c = 0; c < nch; ++c) {
        G4VDecayChannel* ch = dt->GetDecayChannel(c);
        std::fprintf(f, "%s,%d,%.17g,%.17g,%d,%d,%d,%d,%.17g,%d,%d\n",
                     p->GetParticleName().c_str(), p->GetPDGEncoding(), frac, pm / MeV, nch,
                     draws, n_null, c, ch->GetBR(), ch->IsOKWithParentMass(pm) ? 1 : 0,
                     count[static_cast<std::size_t>(c)]);
      }
    }
  }
}

// -------------------------------------------------------------------------------------
/// Every AT-REST process QBBC registers, per species, with the at-rest interaction length
/// each one offers a stopped track. This is the oracle for the plan's "a stopped pi- is
/// captured, not decayed" - measured, not argued.
///
/// `AtRestGetPhysicalInteractionLength` is public virtual on G4VProcess, so the number the
/// stepper would compare can be asked of each process directly - but only after
/// `StartTracking`, because G4Decay's answer is
/// `theNumberOfInteractionLengthLeft * GetMeanLifeTime` (G4Decay.cc:490) and that member is
/// -1.0 from G4VProcess's constructor until StartTracking calls
/// ResetNumberOfInteractionLengthLeft. Called without it, every unstable species reported
/// exactly MINUS its own lifetime, which is what first said the call was being made outside
/// the state the number belongs to.
///
/// So one of the two columns is sampled and one is not, deliberately. G4Decay's length is
/// `-log(rand) * tau` under the fixed seed below and no port can reproduce the draw (Philox
/// against MixMax); every G4HadronStoppingProcess answers a hard 0.0 whatever the seed and
/// whatever the track (G4HadronStoppingProcess.cc:115 ignores both). What the port is
/// required to reproduce is therefore the ZERO, the ORDER and the SIGN, not the sample.
///
/// WHAT THIS FILE IS FOR. A process whose at-rest length is exactly zero is not a competitor
/// in a smallest-wins race, it is a pre-emption: no sample of `-log(rand) * tau` can beat it.
/// So for pi-, kaon- AND mu- the at-rest branch of G4Decay never fires, and the port has to
/// say so rather than describe a race. Rows with n_atrest == 0 are written with index -1, so
/// "this species has no at-rest process" is a row and not an absence.
void dump_atrest(FILE* f) {
  std::fprintf(f,
               "name,pdg,stable,n_atrest,index,process_name,process_type,process_subtype,"
               "at_rest_length_ns\n");
  auto* it = G4ParticleTable::GetParticleTable()->GetIterator();
  it->reset();
  while ((*it)()) {
    G4ParticleDefinition* p = it->value();
    G4ProcessManager* pm = p->GetProcessManager();
    // Only the particles QBBC actually built a process manager for. That is the set whose
    // at-rest queue is a fact about this physics list rather than about the particle table.
    if (pm == nullptr) { continue; }
    G4ProcessVector* v = pm->GetAtRestProcessVector();
    const int n = (v != nullptr) ? static_cast<int>(v->size()) : 0;
    if (n == 0) {
      std::fprintf(f, "%s,%d,%d,0,-1,,,,\n", p->GetParticleName().c_str(),
                   p->GetPDGEncoding(), p->GetPDGStable() ? 1 : 0);
      continue;
    }
    for (int i = 0; i < n; ++i) {
      G4VProcess* proc = (*v)[i];
      // A stopped track: zero kinetic energy, no touchable. Neither G4Decay's nor
      // G4HadronStoppingProcess's at-rest length reads the material or the volume.
      CLHEP::HepRandom::setTheSeed(4242UL + static_cast<unsigned long>(i));
      auto* dp = new G4DynamicParticle(p, G4ThreeVector(0, 0, 1), 0.0);
      G4Track track(dp, 0.0, G4ThreeVector(0, 0, 0));
      track.SetTrackStatus(fStopButAlive);
      // StartTracking is what draws theNumberOfInteractionLengthLeft. Without it G4Decay
      // multiplies G4VProcess's constructor value of -1.0 by the mean life and reports MINUS
      // the lifetime - which is what this dump did on its first run, for every unstable
      // species, and is how the missing call was found.
      proc->StartTracking(&track);
      G4ForceCondition cond = NotForced;
      const double len = proc->AtRestGetPhysicalInteractionLength(track, &cond);
      std::fprintf(f, "%s,%d,%d,%d,%d,%s,%d,%d,%.17g\n", p->GetParticleName().c_str(),
                   p->GetPDGEncoding(), p->GetPDGStable() ? 1 : 0, n, i,
                   proc->GetProcessName().c_str(),
                   static_cast<int>(proc->GetProcessType()),
                   proc->GetProcessSubType(), len / ns);
    }
  }
}

// -------------------------------------------------------------------------------------
/// G4Decay::GetMeanFreePath and ::GetMeanLifeTime, over a grid that crosses the one branch
/// they have: `rKineticEnergy = Ekin/mass > HighestValue (20)`, where the mean free path
/// switches from p/m*c*tau to (Ekin/m + 1)*c*tau. For a pion the crossing is at 2.93 GeV, for
/// a muon at 2.11 GeV, for a kaon at 9.87 GeV; the grid below straddles all three.
///
/// Also the zero-kinetic-energy point, which returns DBL_MIN - the "too slow particle" branch
/// that hands a stopped hadron to the at-rest competition. It is written to the CSV as the
/// literal DBL_MIN so a port that returned zero instead would differ.
void dump_process(FILE* f) {
  std::fprintf(f, "name,pdg,mass_MeV,ekin_MeV,ekin_over_mass,mean_free_path_mm,mean_life_ns\n");
  DecayProbe decay;
  G4ForceCondition cond = NotForced;
  const double energies[] = {0.0,     1.0e-6, 1.0e-3, 0.1,    1.0,     10.0,
                             100.0,   1000.0, 2000.0, 2110.0, 2200.0,  2930.0,
                             3000.0,  9870.0, 1.0e4,  1.0e5,  1.0e6};
  for (int s = 0; s < kNumSpecies; ++s) {
    G4ParticleDefinition* p = find(kSpecies[s]);
    for (double e : energies) {
      // G4Track takes ownership of the G4DynamicParticle. Neither GetMeanFreePath nor
      // GetMeanLifeTime reads the material or the volume, so a track with no touchable is
      // enough - and the verbose branches that would read the material are behind
      // verboseLevel > 1, which G4Decay's constructor sets to 1.
      auto* dp = new G4DynamicParticle(p, G4ThreeVector(0, 0, 1), e * MeV);
      G4Track track(dp, 0.0, G4ThreeVector(0, 0, 0));
      const double mfp = decay.GetMeanFreePath(track, 0.0, &cond);
      const double mlt = decay.GetMeanLifeTime(track, &cond);
      std::fprintf(f, "%s,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n", p->GetParticleName().c_str(),
                   p->GetPDGEncoding(), p->GetPDGMass() / MeV, e, e / (p->GetPDGMass() / MeV),
                   mfp / mm, mlt / ns);
    }
  }
}

// -------------------------------------------------------------------------------------
/// G4DecayProducts::Boost(totalEnergy, momentumDirection) on a FIXED product set.
///
/// The sampling and the boost are separated deliberately. A boost is a deterministic function
/// of (parent mass, parent energy, parent direction, the rest-frame four-momenta), so it can
/// be compared to machine precision - and if it were only ever exercised through a sampled
/// decay, an error in it would arrive mixed with Monte Carlo noise.
///
/// The rest-frame set is a two-body pi+ -> mu+ nu_mu configuration with the muon along a
/// deliberately oblique axis, so that every component of the boost matrix is exercised and a
/// transposed row would show. The parent directions include one along +z (where a wrong
/// cross term cancels) and three oblique ones (where it does not), and the parent energies
/// include the rest case E == m, where G4DecayProducts::Boost's `totalEnergy > mass` test
/// makes the momentum exactly zero rather than a rounding-sized nonzero.
void dump_boost(FILE* f) {
  std::fprintf(f,
               "parent,parent_mass_MeV,parent_ekin_MeV,dirx,diry,dirz,slot,daughter,"
               "daughter_pdg,px_MeV,py_MeV,pz_MeV,e_MeV,mass_MeV,ekin_MeV\n");
  G4ParticleDefinition* pip = find("pi+");
  G4ParticleDefinition* mup = find("mu+");
  G4ParticleDefinition* numu = find("nu_mu");
  const double pm = pip->GetPDGMass();
  const double m0 = mup->GetPDGMass();
  const double pmom = G4PhaseSpaceDecayChannel::Pmx(pm, m0, 0.0);
  // An oblique rest-frame axis: (2, -3, 6)/7 is a unit vector in exact rational arithmetic,
  // so the rest-frame input carries no rounding of its own.
  const G4ThreeVector rest_dir(2.0 / 7.0, -3.0 / 7.0, 6.0 / 7.0);

  const G4ThreeVector dirs[] = {G4ThreeVector(0, 0, 1), G4ThreeVector(1, 0, 0),
                                G4ThreeVector(2.0 / 7.0, -3.0 / 7.0, 6.0 / 7.0),
                                G4ThreeVector(-6.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0)};
  const double ekins[] = {0.0, 1.0e-6, 1.0, 100.0, 1000.0, 1.0e5};
  for (const G4ThreeVector& dir : dirs) {
    for (double ekin : ekins) {
      G4DynamicParticle parent(pip, G4ThreeVector(0, 0, 0), 0.0, pm);
      G4DecayProducts products(parent);
      products.PushProducts(
          new G4DynamicParticle(mup, rest_dir, std::sqrt(pmom * pmom + m0 * m0) - m0, m0));
      products.PushProducts(new G4DynamicParticle(numu, -1.0 * rest_dir, pmom, 0.0));
      products.Boost(ekin * MeV + pm, dir);
      for (G4int i = 0; i < products.entries(); ++i) {
        const G4DynamicParticle* q = products[i];
        const G4LorentzVector p4 = q->Get4Momentum();
        std::fprintf(f, "pi+,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,"
                        "%.17g,%.17g\n",
                     pm / MeV, ekin, dir.x(), dir.y(), dir.z(), i, i,
                     q->GetDefinition()->GetPDGEncoding(), p4.x() / MeV, p4.y() / MeV,
                     p4.z() / MeV, p4.t() / MeV, q->GetMass() / MeV,
                     q->GetKineticEnergy() / MeV);
      }
    }
  }
}

// -------------------------------------------------------------------------------------
constexpr int kHistBins = 40;
constexpr int kSamples = 400000;

struct SlotStats {
  int pdg = 0;
  int daughter = -1;  ///< which daughter of the channel this slot holds; see sample_channel
  double n = 0;
  double sum_e = 0, sum_e2 = 0;
  double min_e = 1e300, max_e = -1e300;
  double sum_cz = 0, sum_cz2 = 0;
  double emax = 0;  ///< the kinematic limit, and the histogram's top edge
  double hist[kHistBins] = {0};
};

/// Sample one channel N times and accumulate per-slot moments, pair angular moments and
/// per-slot histograms. `label` names the parent in the CSV; `pdg` keys it.
///
/// Each slot's histogram runs from zero to that slot's own KINEMATIC LIMIT, which is exact
/// (slot_max_kinetic_energy above) rather than a padded bound. That matters for a channel
/// whose daughters have wildly different limits: the proton of a neutron beta decay tops out
/// at 7.5e-4 MeV against the electron's 0.78 MeV, and a shared top edge would put every
/// proton in bin zero and check nothing about its spectrum. The bin index is clamped into
/// range so a sample AT the limit lands in the last bin - a spectrum that reached beyond the
/// limit would show as an over-full last bin, which is why it is clamped rather than dropped.
///
/// THE PRODUCT SLOT IS NOT THE DAUGHTER INDEX, and the limits have to be attached to the
/// right one. `G4PhaseSpaceDecayChannel::ThreeBodyDecayIt` and `G4KL3DecayChannel::DecayIt`
/// both push daughter 0, then daughter 2, then daughter 1 - because they derive daughter 1's
/// momentum from the other two - so slot 1 of a Kmu3 decay holds the NEUTRINO and slot 2 the
/// muon. Indexing the limits by slot got this wrong first time round and it was visible in
/// the data: slot 1's sampled maximum was 188.17 MeV against a stated limit of 134.03, i.e.
/// the neutrino was being measured against the muon's bound.
///
/// The mapping is not hard-coded from the kinematics name; it is READ OUT OF GEANT4 by
/// sampling one decay and matching each product's definition pointer against the channel's
/// daughter list. That way the port's push order is compared against what Geant4 actually did
/// rather than against a second copy of the same assumption. The pre-pass consumes random
/// numbers, so the seed is set again before the real loop.
void sample_channel(FILE* fm, FILE* fp, FILE* fh, FILE* fc, const char* label, int pdg,
                    int channel_index, G4VDecayChannel* ch, double parent_mass,
                    unsigned long seed) {
  const int nd = ch->GetNumberOfDaughters();

  std::vector<SlotStats> slots(nd);
  CLHEP::HepRandom::setTheSeed(seed);
  {
    G4DecayProducts* pr = ch->DecayIt(parent_mass);
    std::vector<bool> claimed(nd, false);
    for (int s = 0; s < nd; ++s) {
      int found = -1;
      if (pr != nullptr && pr->entries() == nd) {
        const G4ParticleDefinition* def = (*pr)[s]->GetDefinition();
        for (int d = 0; d < nd; ++d) {
          if (!claimed[d] && ch->GetDaughter(d) == def) {
            claimed[d] = true;
            found = d;
            break;
          }
        }
      }
      slots[s].daughter = (found >= 0) ? found : s;
      const double lim = slot_max_kinetic_energy(parent_mass, ch, slots[s].daughter);
      slots[s].emax = (lim > 0.0) ? lim : 1.0;
    }
    delete pr;
  }
  std::vector<double> pair_cos(static_cast<std::size_t>(nd) * nd, 0.0);
  std::vector<double> pair_cos2(static_cast<std::size_t>(nd) * nd, 0.0);

  // How well the channel's own four-momenta close, in the parent rest frame. This is dumped
  // rather than asserted in the port against a chosen tolerance, because the answer is a
  // property of the CHANNEL and not of either implementation, and it is not always small:
  //
  //   G4MuonDecayChannel neglects the electron mass in its angles, so its energies miss by
  //   exactly m_e and its momenta by up to m_e - 4.8e-3 of the muon mass.
  //   G4PhaseSpaceDecayChannel::ManyBodyDecayIt boosts each nested subsystem by
  //   beta = p/sqrt(p^2 + m_sub^2), and when a subsystem is nearly massless that beta is
  //   within 1e-11 of one, so CLHEP's `1/sqrt(1-b2)` and `(gamma-1)/b2` lose most of their
  //   significant digits. The residual is then parts in 1e6, not parts in 1e15.
  //
  // Comparing the port's residual against Geant4's own turns both of those from "what
  // tolerance should I pick" into "does the port lose precision where Geant4 loses it".
  double worst_p = 0, worst_e = 0;
  CLHEP::HepRandom::setTheSeed(seed);
  int accepted = 0;
  for (int i = 0; i < kSamples; ++i) {
    G4DecayProducts* products = ch->DecayIt(parent_mass);
    if (products == nullptr) { continue; }
    if (products->entries() != nd) {
      // An empty product list is G4PhaseSpaceDecayChannel's PART112 answer; counted, not
      // silently skipped, so a channel that never produces anything cannot look like a
      // channel with a narrow spectrum.
      delete products;
      continue;
    }
    ++accepted;
    {
      G4LorentzVector sum(0, 0, 0, 0);
      for (int s = 0; s < nd; ++s) { sum += (*products)[s]->Get4Momentum(); }
      const double pr = sum.vect().mag() / parent_mass;
      const double er = std::fabs(sum.t() - parent_mass) / parent_mass;
      if (pr > worst_p) { worst_p = pr; }
      if (er > worst_e) { worst_e = er; }
    }
    for (int s = 0; s < nd; ++s) {
      const G4DynamicParticle* q = (*products)[s];
      const double e = q->GetKineticEnergy() / MeV;
      const G4ThreeVector d = q->GetMomentumDirection();
      SlotStats& st = slots[s];
      st.pdg = q->GetDefinition()->GetPDGEncoding();
      st.n += 1;
      st.sum_e += e;
      st.sum_e2 += e * e;
      if (e < st.min_e) { st.min_e = e; }
      if (e > st.max_e) { st.max_e = e; }
      st.sum_cz += d.z();
      st.sum_cz2 += d.z() * d.z();
      int b = static_cast<int>(e / st.emax * kHistBins);
      if (b < 0) { b = 0; }
      if (b >= kHistBins) { b = kHistBins - 1; }
      st.hist[b] += 1;
    }
    for (int a = 0; a < nd; ++a) {
      for (int b = a + 1; b < nd; ++b) {
        const G4ThreeVector da = (*products)[a]->GetMomentumDirection();
        const G4ThreeVector db = (*products)[b]->GetMomentumDirection();
        const double c = da.dot(db);
        pair_cos[static_cast<std::size_t>(a) * nd + b] += c;
        pair_cos2[static_cast<std::size_t>(a) * nd + b] += c * c;
      }
    }
    delete products;
  }

  for (int s = 0; s < nd; ++s) {
    const SlotStats& st = slots[s];
    const double n = (st.n > 0) ? st.n : 1.0;
    std::fprintf(fm, "%s,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                 label, pdg, channel_index, s, st.daughter, st.pdg, accepted, st.emax,
                 st.sum_e / n, st.sum_e2 / n, st.min_e, st.max_e, st.sum_cz / n,
                 st.sum_cz2 / n);
    for (int b = 0; b < kHistBins; ++b) {
      std::fprintf(fh, "%s,%d,%d,%d,%d,%d,%d,%.17g,%d,%.17g\n", label, pdg, channel_index, s,
                   st.daughter, st.pdg, kHistBins, st.emax, b, st.hist[b]);
    }
  }
  for (int a = 0; a < nd; ++a) {
    for (int b = a + 1; b < nd; ++b) {
      const double n = (accepted > 0) ? double(accepted) : 1.0;
      std::fprintf(fp, "%s,%d,%d,%d,%d,%d,%.17g,%.17g\n", label, pdg, channel_index, a, b,
                   accepted, pair_cos[static_cast<std::size_t>(a) * nd + b] / n,
                   pair_cos2[static_cast<std::size_t>(a) * nd + b] / n);
    }
  }
  std::fprintf(fc, "%s,%d,%d,%s,%d,%d,%.17g,%.17g\n", label, pdg, channel_index,
               ch->GetKinematicsName().c_str(), nd, accepted, worst_p, worst_e);
}

void dump_spectra(FILE* fm, FILE* fp, FILE* fh, FILE* fc) {
  std::fprintf(fc,
               "parent,parent_pdg,channel,kinematics,n_daughters,samples,worst_p_residual,"
               "worst_e_residual\n");
  std::fprintf(fm,
               "parent,parent_pdg,channel,slot,daughter,daughter_pdg,samples,ekin_limit,"
               "mean_ekin,mean_ekin2,min_ekin,max_ekin,mean_cosz,mean_cos2z\n");
  std::fprintf(fp, "parent,parent_pdg,channel,slot_i,slot_j,samples,mean_cos,mean_cos2\n");
  std::fprintf(fh,
               "parent,parent_pdg,channel,slot,daughter,daughter_pdg,nbins,emax_MeV,bin,"
               "count\n");

  for (int s = 0; s < kNumSpecies; ++s) {
    G4ParticleDefinition* p = find(kSpecies[s]);
    G4DecayTable* dt = p->GetDecayTable();
    for (G4int i = 0; i < dt->entries(); ++i) {
      // A seed per (species, channel) so that adding a channel does not move an existing
      // channel's numbers.
      const unsigned long seed = 90210UL + 1000UL * (unsigned long)(s + 1) + (unsigned long)i;
      sample_channel(fm, fp, fh, fc, p->GetParticleName().c_str(), p->GetPDGEncoding(), i,
                     dt->GetDecayChannel(i), p->GetPDGMass(), seed);
    }
  }

  // The general N-body arm of G4PhaseSpaceDecayChannel::DecayIt, which no table of any
  // transported species reaches - every channel above has two or three daughters. It is
  // sampled here from a SYNTHETIC four-body channel so that the port's transcription of
  // ManyBodyDecayIt has an oracle at all. kaon+ -> e+ nu_e gamma gamma is chosen because it
  // is kinematically wide open (0.511 MeV of daughter mass against 493.677) and because its
  // daughters all exist in the table; it is not a physical decay mode and it is not in any
  // decay table. Channel index 99 marks it as synthetic.
  {
    G4PhaseSpaceDecayChannel four("kaon+", 1.0, 4, "e+", "nu_e", "gamma", "gamma");
    G4ParticleDefinition* kp = find("kaon+");
    sample_channel(fm, fp, fh, fc, "kaon+_4body", kp->GetPDGEncoding(), 99, &four,
                   kp->GetPDGMass(), 777001UL);
  }
}

// -------------------------------------------------------------------------------------
void dump_decay(const DumpContext&) {
  {
    FILE* f = std::fopen("decay_applicable.csv", "w");
    dump_applicable(f);
    std::fclose(f);
  }
  {
    FILE* f = std::fopen("decay_tables.csv", "w");
    dump_tables(f);
    std::fclose(f);
  }
  {
    FILE* f = std::fopen("decay_process.csv", "w");
    dump_process(f);
    std::fclose(f);
  }
  {
    FILE* f = std::fopen("decay_select.csv", "w");
    dump_select(f);
    std::fclose(f);
  }
  {
    FILE* f = std::fopen("decay_dalitz.csv", "w");
    dump_dalitz(f);
    std::fclose(f);
  }
  {
    FILE* f = std::fopen("decay_atrest.csv", "w");
    dump_atrest(f);
    std::fclose(f);
  }
  {
    FILE* f = std::fopen("decay_boost.csv", "w");
    dump_boost(f);
    std::fclose(f);
  }
  {
    FILE* fm = std::fopen("decay_moments.csv", "w");
    FILE* fp = std::fopen("decay_pairs.csv", "w");
    FILE* fh = std::fopen("decay_hist.csv", "w");
    FILE* fc = std::fopen("decay_closure.csv", "w");
    dump_spectra(fm, fp, fh, fc);
    std::fclose(fm);
    std::fclose(fp);
    std::fclose(fh);
    std::fclose(fc);
  }
}

}  // namespace

G4GPU_REGISTER_DUMP("decay",
                    "decay_applicable.csv decay_tables.csv decay_process.csv "
                    "decay_select.csv decay_dalitz.csv decay_atrest.csv decay_boost.csv "
                    "decay_moments.csv decay_pairs.csv decay_hist.csv decay_closure.csv",
                    dump_decay);
