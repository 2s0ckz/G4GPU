// The pre-equilibrium module's oracle: what Geant4 11.1.1 answers for every piece of
// G4PreCompoundModel, so the port can be diffed against it rather than against a reading.
//
// Nine files:
//
//   preco_params.csv        the thirteen G4DeexPrecoParameters values this package is
//                           dispatched by, read from the install. NOT the three
//                           G4GeneratorPrecompoundInterface constants - see below.
//   preco_channels.csv      per (Z, A, E*, P, Pc, H) and per channel: the residual, the
//                           binding energy, the energy threshold, both masses, IsItPossible,
//                           and CalcEmissionProbability for **all five OPTxs branches**.
//                           SetOPTxs is public and is not locked by the run state, so the
//                           Dostrovsky (0), Chatterjee (1, 2), Kalbach (3) and hybrid (4)
//                           inverse cross sections are all reachable from one run - which is
//                           what makes GetAlpha and GetBeta checked numbers instead of
//                           unvalidated transcription.
//   preco_sample.csv        SampleKineticEnergy per channel under a prescribed uniform cycle,
//                           so the rejection sampler is a deterministic function of its
//                           inputs and the phase.
//   preco_transitions.csv   CalculateProbability and the three GetTransitionProbN getters for
//                           **five transition configurations** - CEM, CEM+NGB, Gupta,
//                           Gupta+NGB, GNASH - and the exciton configuration
//                           PerformTransition leaves. UseCEMtr and UseNGB are public inline
//                           setters on G4VPreCompoundTransitions with no state lock, and
//                           G4GNASHTransitions has a public constructor, so all five are
//                           reachable. The GNASH rows are the point of the file: they show
//                           P1 = P2 = P3 = 0.
//   preco_equilibrium.csv   the equilibrium predicate's inputs on the same grid: the level
//                           density, G4lrint(sqrt(12/pi^2 U g)) and both gates' verdicts;
//                           preco_gates.csv adds the four nuclides and the four excitation
//                           energies at which the two gates' four differing tokens are
//                           decidable at all.
//   preco_emission.csv      G4PreCompoundEmission::PerformEmission under the cycle: which
//                           channel was chosen, the ejectile's four-momentum, and the
//                           residual's (Z, A, E*, P, Pc, H). Exact, and it checks
//                           ChooseFragment's cumulative walk, the sampler, the isotropic
//                           direction and the boost in one row.
//   preco_deexcite.csv      G4PreCompoundModel::DeExcite, N = 20,000 under a fixed seed on
//                           fifteen fragments, tagged pre-equilibrium against equilibrium by
//                           G4ReactionProduct::GetCreatorModelID.
//   preco_apply.csv         G4PreCompoundModel::ApplyYourself, N = 20,000 on nine
//                           (projectile, target, energy) cases - four matched neutron/proton
//                           pairs and one below the entry gate. The initial fragment
//                           ApplyYourself builds is never handed back, so this is the only way
//                           to check it: through its consequences, with Geant4 supplying the
//                           target mass and the projectile four-momentum.
//   preco_propagate.csv     G4GeneratorPrecompoundInterface::Propagate on a real
//                           G4Fancy3DNucleus with a hand-built track list, under the cycle -
//                           plus the nucleon list itself, so the port replays the same
//                           wounded nucleus rather than inventing one.
//
// **The one number here the oracle cannot reach.** G4GeneratorPrecompoundInterface's
// CaptureThreshold (70 MeV), DeltaM (5 MeV) and DeltaR (0) are private members with public
// SETTERS and no getters, so no run can be asked what they are. preco_propagate.csv checks
// them indirectly and completely: CaptureThreshold decides which tracks escape, and the
// escaped set is dumped track by track.
//
// **How pre-equilibrium and equilibrium products are told apart.** They can be, exactly:
// G4PreCompoundEmission::PerformEmission calls SetCreatorModelID(model_PRECO) on the ejectile,
// while every de-excitation channel stamps its own id - model_G4EvaporationChannel,
// model_G4PhotonEvaporation, model_G4FermiBreakUpVI, model_G4GEMChannel and so on. So
// preco_deexcite.csv carries the creator id per product and preco_modelids.csv the id-to-name
// map. One caveat, stated because it is a real off-by-one: PerformEmission also stamps PRECO
// on the RESIDUAL fragment, so if the handler later releases that fragment untouched (the
// A <= 1 or cold-natural-isotope arms of BreakItUp) the released product carries PRECO too.
// The initial fragment is therefore given creator id -1 here, which makes an untouched
// release of the ORIGINAL fragment distinguishable, and the count of leading PRECO products
// whose PDG is one of the six ejectiles is dumped separately from the count of leading PRECO
// products of any kind.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "G4Alpha.hh"
#include "G4DeexPrecoParameters.hh"
#include "G4Deuteron.hh"
#include "G4DynamicParticle.hh"
#include "G4ExcitationHandler.hh"
#include "G4Fancy3DNucleus.hh"
#include "G4Fragment.hh"
#include "G4GNASHTransitions.hh"
#include "G4GeneratorPrecompoundInterface.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadSecondary.hh"
#include "G4He3.hh"
#include "G4KineticTrack.hh"
#include "G4KineticTrackVector.hh"
#include "G4Neutron.hh"
#include "G4Nucleon.hh"
#include "G4Nucleus.hh"
#include "G4NuclearLevelData.hh"
#include "G4NucleiProperties.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicsModelCatalog.hh"
#include "G4Pow.hh"
#include "G4PreCompoundAlpha.hh"
#include "G4PreCompoundDeuteron.hh"
#include "G4PreCompoundEmission.hh"
#include "G4PreCompoundHe3.hh"
#include "G4PreCompoundModel.hh"
#include "G4PreCompoundNeutron.hh"
#include "G4PreCompoundProton.hh"
#include "G4PreCompoundTransitions.hh"
#include "G4PreCompoundTriton.hh"
#include "G4Proton.hh"
#include "G4ReactionProduct.hh"
#include "G4ReactionProductVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4Triton.hh"
#include "G4VPreCompoundFragment.hh"
#include "Randomize.hh"

#include "CLHEP/Random/RandomEngine.h"

namespace {

// ---------------------------------------------------------------------------------------------
// A CLHEP engine that returns a prescribed cycle and counts the draws.
//
// The same eight values and the same reasoning as dump_elastic.cc's: spread over (0,1), away
// from both endpoints, and shared with the port so that every sampler in this package becomes
// a deterministic function of (inputs, phase). The draw COUNT is dumped too, because a
// transcription that gets the right answer from the wrong number of deviates is wrong in a way
// only the count reveals - and G4PreCompoundTransitions' CEM branch consumes exactly one, in
// the middle of the function, which is easy to omit.
// ---------------------------------------------------------------------------------------------

const double kSeq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};

class CycleEngine : public CLHEP::HepRandomEngine {
 public:
  void reset(int phase) { phase_ = phase; n_ = 0; }
  int draws() const { return n_; }
  double flat() override {
    const double v = kSeq[(n_ + phase_) % 8];
    ++n_;
    return v;
  }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "CycleEngine"; }

 private:
  int phase_ = 0;
  int n_ = 0;
};

// ---------------------------------------------------------------------------------------------
// The grid
// ---------------------------------------------------------------------------------------------

struct Nuclide { int z, a; const char* name; };

/// Light to heavy, and chosen so that both (Z, A) gates are exercised: Li6 sits just above
/// fMinZForPreco = 3 and fMinAForPreco = 5, and He4 below both (it fails the entry gate, which
/// is what preco_equilibrium.csv records).
const Nuclide kNuclides[] = {
  {2, 4, "He4"},   {3, 6, "Li6"},    {6, 12, "C12"},   {13, 27, "Al27"},
  {20, 40, "Ca40"}, {26, 56, "Fe56"}, {47, 107, "Ag107"}, {79, 197, "Au197"},
  {82, 208, "Pb208"}, {92, 238, "U238"},
};

/// E* in MeV. 0.5 is below fPrecoLowEnergy*A for everything above A = 5 and above it for He4,
/// so the low gate is crossed inside the grid; 200 is above fPrecoHighEnergy*A = 30A only for
/// A <= 6, so the high gate is crossed too.
const double kExc[] = {0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0};

/// (P, Pc, H). The last three close the ion channels one at a time: (2,0,1) has no charged
/// particle exciton so the proton, deuteron, triton, He3 and alpha GetRj all vanish; (4,4,2)
/// has no neutral one so the neutron's, the deuteron's and the triton's do; (2,1,1) is
/// ApplyYourself's own initial configuration.
struct ExcitonConfig { int p, pc, h; };
const ExcitonConfig kExcitons[] = {
  {2, 1, 1}, {3, 1, 2}, {4, 2, 2}, {5, 2, 3}, {6, 3, 3}, {8, 4, 4}, {2, 0, 1}, {4, 4, 2},
};

/// A fragment of (Z, A) with excitation E* and momentum pz along z, plus an exciton
/// configuration. E* is set through the four-momentum, which is the only way G4Fragment
/// accepts one.
G4Fragment make_fragment(int Z, int A, double eexc, double pz, const ExcitonConfig& ex) {
  const double m = G4NucleiProperties::GetNuclearMass(A, Z) + eexc;
  const G4LorentzVector lv(0.0, 0.0, pz, std::sqrt(m * m + pz * pz));
  G4Fragment f(A, Z, lv);
  f.SetNumberOfExcitedParticle(ex.p, ex.pc);
  f.SetNumberOfHoles(ex.h, 0);
  return f;
}

/// The six channels in G4PreCompoundEmissionFactory's order: n, p, d, ALPHA, t, He3. That
/// order is what ChooseFragment walks, so it is the order the port and this file both use.
std::vector<G4VPreCompoundFragment*> make_channels() {
  std::vector<G4VPreCompoundFragment*> v;
  v.push_back(new G4PreCompoundNeutron());
  v.push_back(new G4PreCompoundProton());
  v.push_back(new G4PreCompoundDeuteron());
  v.push_back(new G4PreCompoundAlpha());
  v.push_back(new G4PreCompoundTriton());
  v.push_back(new G4PreCompoundHe3());
  return v;
}

// ---------------------------------------------------------------------------------------------

void dump_params() {
  G4NuclearLevelData* nd = G4NuclearLevelData::GetInstance();
  const G4DeexPrecoParameters* p = nd->GetParameters();
  FILE* f = std::fopen("preco_params.csv", "w");
  std::fprintf(f, "name,value\n");
  std::fprintf(f, "fPrecoLowEnergy,%.17g\n", p->GetPrecoLowEnergy() / MeV);
  std::fprintf(f, "fPrecoHighEnergy,%.17g\n", p->GetPrecoHighEnergy() / MeV);
  std::fprintf(f, "fPhenoFactor,%.17g\n", p->GetPhenoFactor());
  std::fprintf(f, "fMinZForPreco,%d\n", p->GetMinZForPreco());
  std::fprintf(f, "fMinAForPreco,%d\n", p->GetMinAForPreco());
  std::fprintf(f, "fPrecoType,%d\n", p->GetPrecoModelType());
  std::fprintf(f, "fNeverGoBack,%d\n", p->NeverGoBack() ? 1 : 0);
  std::fprintf(f, "fUseSoftCutoff,%d\n", p->UseSoftCutoff() ? 1 : 0);
  std::fprintf(f, "fUseCEM,%d\n", p->UseCEM() ? 1 : 0);
  std::fprintf(f, "fUseGNASH,%d\n", p->UseGNASH() ? 1 : 0);
  std::fprintf(f, "fUseHETC,%d\n", p->UseHETC() ? 1 : 0);
  std::fprintf(f, "fUseAngularGen,%d\n", p->UseAngularGen() ? 1 : 0);
  std::fprintf(f, "fPrecoDummy,%d\n", p->PrecoDummy() ? 1 : 0);
  // The two the transitions and the emission read from the same block, so that a test which
  // perturbs one of them knows which number moved.
  std::fprintf(f, "fFermiEnergy,%.17g\n", p->GetFermiEnergy() / MeV);
  std::fprintf(f, "fTransitionsR0_fm,%.17g\n", p->GetTransitionsR0() / fermi);
  std::fprintf(f, "fR0_fm,%.17g\n", p->GetR0() / fermi);
  // G4Pow::logfactorial, the one table G4PreCompoundEmission::rho depends on. rho itself is
  // private and is only reached when fUseAngularGen is set, which cannot be done after
  // initialisation - so this is the checkable part of that path and the arithmetic around it
  // is transcription. Dumped to 40, well past any exciton number this model reaches.
  for (int i = 0; i <= 40; ++i) {
    std::fprintf(f, "logfactorial_%d,%.17g\n", i, G4Pow::GetInstance()->logfactorial(i));
  }
  std::fclose(f);
}

/// The model-id to name map, so preco_deexcite.csv's creator ids mean something.
void dump_model_ids() {
  FILE* f = std::fopen("preco_modelids.csv", "w");
  std::fprintf(f, "id,name\n");
  const G4int n = G4PhysicsModelCatalog::Entries();
  for (G4int i = 0; i < n; ++i) {
    const G4int id = G4PhysicsModelCatalog::GetModelID(i);
    std::fprintf(f, "%d,%s\n", id,
                 G4PhysicsModelCatalog::GetModelNameFromIndex(i).c_str());
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------

/// preco_channels.csv - the deterministic core of this package.
///
/// `Initialize` has to be called before anything else is read; it is what fills theResA,
/// theBindingEnergy, theMinKinEnergy and theMaxKinEnergy. `SetOPTxs` is called BEFORE
/// Initialize for each option because Initialize reads OPTxs to choose `elim`, so the
/// kinetic-energy window itself depends on the option and not only the cross section.
void dump_channels() {
  auto ch = make_channels();
  FILE* f = std::fopen("preco_channels.csv", "w");
  std::fprintf(f,
               "Z,A,Eexc_MeV,P,Pc,H,chan,ejZ,ejA,resZ,resA,possible,binding_MeV,"
               "threshold_MeV,mass_MeV,resmass_MeV,prob_opt0,prob_opt1,prob_opt2,prob_opt3,"
               "prob_opt4\n");
  for (const Nuclide& nu : kNuclides) {
    for (const double e : kExc) {
      for (const ExcitonConfig& ex : kExcitons) {
        for (std::size_t c = 0; c < ch.size(); ++c) {
          double prob[5];
          int possible = 0;
          int resz = 0, resa = 0;
          double binding = 0.0, threshold = 0.0, mass = 0.0, resmass = 0.0;
          for (int opt = 0; opt < 5; ++opt) {
            ch[c]->SetOPTxs(opt);
            G4Fragment frag = make_fragment(nu.z, nu.a, e * MeV, 0.0, ex);
            ch[c]->Initialize(frag);
            const G4bool ok = ch[c]->IsItPossible(frag);
            prob[opt] = ok ? ch[c]->CalcEmissionProbability(frag) : 0.0;
            if (opt == 3) {
              possible = ok ? 1 : 0;
              resz = ch[c]->GetRestZ();
              resa = ch[c]->GetRestA();
              binding = ch[c]->GetBindingEnergy() / MeV;
              threshold = ch[c]->GetEnergyThreshold() / MeV;
              mass = ch[c]->GetNuclearMass() / MeV;
              resmass = ch[c]->GetRestNuclearMass() / MeV;
            }
          }
          ch[c]->SetOPTxs(3);
          std::fprintf(f,
                       "%d,%d,%.17g,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,"
                       "%.17g,%.17g,%.17g,%.17g,%.17g\n",
                       nu.z, nu.a, e, ex.p, ex.pc, ex.h, static_cast<int>(c), ch[c]->GetZ(),
                       ch[c]->GetA(), resz, resa, possible, binding, threshold, mass, resmass,
                       prob[0], prob[1], prob[2], prob[3], prob[4]);
        }
      }
    }
  }
  std::fclose(f);
  for (auto* c : ch) { delete c; }
}

/// preco_sample.csv - SampleKineticEnergy under the cycle.
///
/// CalcEmissionProbability must run first: it is what leaves `probmax`, and SampleKineticEnergy
/// multiplies that member by 1.25 and rejects against it. So the pair is dumped together and
/// the draw count says how many rejections it took.
void dump_sample() {
  auto ch = make_channels();
  auto* eng = new CycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);

  FILE* f = std::fopen("preco_sample.csv", "w");
  std::fprintf(f, "Z,A,Eexc_MeV,P,Pc,H,chan,phase,prob,ekin_MeV,draws\n");
  for (const Nuclide& nu : kNuclides) {
    for (const double e : kExc) {
      for (const ExcitonConfig& ex : kExcitons) {
        for (std::size_t c = 0; c < ch.size(); ++c) {
          for (int phase = 0; phase < 8; ++phase) {
            G4Fragment frag = make_fragment(nu.z, nu.a, e * MeV, 0.0, ex);
            ch[c]->Initialize(frag);
            if (!ch[c]->IsItPossible(frag)) { continue; }
            const double prob = ch[c]->CalcEmissionProbability(frag);
            if (!(prob > 0.0)) { continue; }
            eng->reset(phase);
            const double ekin = ch[c]->SampleKineticEnergy(frag);
            std::fprintf(f, "%d,%d,%.17g,%d,%d,%d,%d,%d,%.17g,%.17g,%d\n", nu.z, nu.a, e, ex.p,
                         ex.pc, ex.h, static_cast<int>(c), phase, prob, ekin / MeV,
                         eng->draws());
          }
        }
      }
    }
  }
  std::fclose(f);
  CLHEP::HepRandom::setTheEngine(saved);
  delete eng;
  for (auto* c : ch) { delete c; }
}

/// preco_transitions.csv - all five transition configurations.
///
/// Configurations 0..3 are one G4PreCompoundTransitions with UseCEMtr and UseNGB toggled;
/// configuration 4 is a G4GNASHTransitions, whose rows exist to show that P1, P2 and P3 come
/// back as exactly zero however the probability itself comes out.
void dump_transitions() {
  G4PreCompoundTransitions pct;
  G4GNASHTransitions gnash;
  auto* eng = new CycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);

  FILE* f = std::fopen("preco_transitions.csv", "w");
  std::fprintf(f,
               "Z,A,Eexc_MeV,P,Pc,H,config,phase,total,p1,p2,p3,draws_prob,"
               "newP,newPc,newH,draws_perform\n");
  // config: 0 = CEM, 1 = CEM+NGB, 2 = Gupta, 3 = Gupta+NGB, 4 = GNASH
  for (const Nuclide& nu : kNuclides) {
    for (const double e : kExc) {
      for (const ExcitonConfig& ex : kExcitons) {
        for (int config = 0; config < 5; ++config) {
          for (int phase = 0; phase < 8; ++phase) {
            G4VPreCompoundTransitions* tr = nullptr;
            if (config < 4) {
              pct.UseCEMtr(config == 0 || config == 1);
              pct.UseNGB(config == 1 || config == 3);
              tr = &pct;
            } else {
              tr = &gnash;
            }
            G4Fragment frag = make_fragment(nu.z, nu.a, e * MeV, 0.0, ex);
            eng->reset(phase);
            const double total = tr->CalculateProbability(frag);
            const double p1 = tr->GetTransitionProb1();
            const double p2 = tr->GetTransitionProb2();
            const double p3 = tr->GetTransitionProb3();
            const int dp = eng->draws();
            eng->reset(phase + 3);
            tr->PerformTransition(frag);
            std::fprintf(f, "%d,%d,%.17g,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%d,%d\n",
                         nu.z, nu.a, e, ex.p, ex.pc, ex.h, config, phase, total, p1, p2, p3, dp,
                         frag.GetNumberOfParticles(), frag.GetNumberOfCharged(),
                         frag.GetNumberOfHoles(), eng->draws());
          }
        }
      }
    }
  }
  std::fclose(f);
  CLHEP::HepRandom::setTheEngine(saved);
  delete eng;
}

/// preco_equilibrium.csv - the inputs of the equilibrium decision.
///
/// The level density and G4lrint(sqrt((12/pi^2) U g)) are what the model computes at the top of
/// each outer iteration, and the entry gate's verdict is the six-clause test at the top of
/// DeExcite. Both are dumped so the two forms of the (Z, A) clause - AND at the entry, OR in
/// the loop - can be compared separately.
void dump_equilibrium() {
  G4NuclearLevelData* nd = G4NuclearLevelData::GetInstance();
  const G4DeexPrecoParameters* p = nd->GetParameters();
  const double low = p->GetPrecoLowEnergy();
  const double high = p->GetPrecoHighEnergy();
  const int minz = p->GetMinZForPreco();
  const int mina = p->GetMinAForPreco();
  const double ldfact = 12.0 / CLHEP::pi2;

  FILE* f = std::fopen("preco_equilibrium.csv", "w");
  std::fprintf(f, "Z,A,Eexc_MeV,P,Pc,H,level_density,eq_exciton_number,entry_gate,loop_gate\n");
  for (const Nuclide& nu : kNuclides) {
    for (const double e : kExc) {
      for (const ExcitonConfig& ex : kExcitons) {
        const double U = e * MeV;
        const double ld = nd->GetLevelDensity(nu.z, nu.a, U);
        const int neq = G4lrint(std::sqrt(ldfact * U * ld));
        // The entry gate, as G4PreCompoundModel::DeExcite writes it: note the AND.
        const int entry = ((nu.z < minz && nu.a < mina) || U < low * nu.a ||
                           U > nu.a * high) ? 1 : 0;
        // The (Z, A) and excitation clauses of the loop's test, which are an OR and an
        // inclusive `<=`. preco_gates.csv is where the two forms are separated at the boundary.
        const int loop = (nu.z < minz || nu.a < mina || U <= low * nu.a ||
                          U > nu.a * high) ? 1 : 0;
        std::fprintf(f, "%d,%d,%.17g,%d,%d,%d,%.17g,%d,%d,%d\n", nu.z, nu.a, e, ex.p, ex.pc,
                     ex.h, ld * MeV, neq, entry, loop);
      }
    }
  }
  std::fclose(f);
}

/// preco_gates.csv - both gates as predicates, at the excitation energies where the four tokens
/// that differ between them are decidable.
///
/// The entry gate and the loop gate are not the same test: AND against OR in the (Z, A) clause,
/// and `<` against `<=` on the low limit. **Neither difference is observable in a product
/// distribution**, which is why this file exists rather than another campaign:
///
///   * a fragment the (Z, A) clause disagrees about - `Z < minZ, A >= minA` or the reverse -
///     passes the entry gate, fails the loop's OR on the very same iteration, and reaches the
///     same G4ExcitationHandler having consumed exactly ONE extra uniform. Same products, same
///     distribution; only the random stream moves.
///   * `U <= fLowLimitExc*A` against `U <` differ on a set of measure zero, and U has been
///     through a sqrt round trip in G4Fragment's constructor, so no decimal excitation energy
///     lands on it.
///
/// So the tokens are pinned by construction instead. `U` is written here at exactly
/// `fPrecoLowEnergy*A` and `fPrecoHighEnergy*A` and at the adjacent representable doubles either
/// side, and `%.17g` round-trips a double, so the port evaluates its comparison on the same bits
/// this file evaluated its own on and `<` and `<=` give different answers. The nuclide list adds
/// H3, He6, Li4 and Be7 to the main grid's - the four that sit on the two sides of the (Z, A)
/// disagreement, which the main grid has none of, every one of its entries having either both
/// of Z and A small (He4) or neither.
///
/// The gates are pure arithmetic on (Z, A, U) and touch no mass table, so an unbound nuclide
/// like Li4 is a legitimate row here.
void dump_gates() {
  G4NuclearLevelData* nd = G4NuclearLevelData::GetInstance();
  const G4DeexPrecoParameters* p = nd->GetParameters();
  const double low = p->GetPrecoLowEnergy();
  const double high = p->GetPrecoHighEnergy();
  const int minz = p->GetMinZForPreco();
  const int mina = p->GetMinAForPreco();

  const Nuclide gate_nuc[] = {
    {1, 3, "H3"},   {2, 4, "He4"},  {2, 6, "He6"},   {3, 4, "Li4"},    {3, 6, "Li6"},
    {4, 7, "Be7"},  {6, 12, "C12"}, {26, 56, "Fe56"}, {82, 208, "Pb208"},
  };

  FILE* f = std::fopen("preco_gates.csv", "w");
  std::fprintf(f, "Z,A,U_MeV,entry_gate,loop_gate\n");
  for (const Nuclide& nu : gate_nuc) {
    const double lo = low * nu.a;
    const double hi = nu.a * high;
    const double us[] = {
      0.0, std::nextafter(lo, 0.0), lo, std::nextafter(lo, 1.0e300),
      0.5 * (lo + hi), std::nextafter(hi, 0.0), hi, std::nextafter(hi, 1.0e300),
    };
    for (const double U : us) {
      // Both written exactly as their sources write them: G4PreCompoundModel.cc's entry test
      // (the AND, the strict `<`) and the (Z, A)/excitation clauses of its loop test (the OR,
      // the inclusive `<=`).
      const int entry = ((nu.z < minz && nu.a < mina) || U < low * nu.a ||
                         U > nu.a * high) ? 1 : 0;
      const int loop = (nu.z < minz || nu.a < mina || U <= low * nu.a ||
                        U > nu.a * high) ? 1 : 0;
      std::fprintf(f, "%d,%d,%.17g,%d,%d\n", nu.z, nu.a, U / MeV, entry, loop);
    }
  }
  std::fclose(f);
}

/// preco_emission.csv - G4PreCompoundEmission::PerformEmission, exact under the cycle.
///
/// GetTotalProbability has to be called first: it is CalculateProbabilities, which fills the
/// cumulative array ChooseFragment reads and the per-channel state SampleKineticEnergy reads.
/// The row therefore checks, in order: the six emission-probability integrals, the cumulative
/// walk, the rejection sampler, the isotropic direction (G4RandomDirection's Marsaglia
/// rejection pair, so a variable number of draws), the boost into the fragment's frame, and the
/// residual bookkeeping. `draws` is what makes the number of deviates part of the comparison.
///
/// The emitted particle's identity comes back as a PDG code; the chosen channel index is
/// recovered from it, since the six are distinct species.
void dump_emission() {
  G4PreCompoundEmission emission;
  emission.SetOPTxs(3);   // what G4PreCompoundModel::InitialiseModel does, from fPrecoType
  auto* eng = new CycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(eng);

  FILE* f = std::fopen("preco_emission.csv", "w");
  std::fprintf(f,
               "Z,A,Eexc_MeV,pz_MeV,P,Pc,H,phase,total_prob,pdg,ekin_MeV,px,py,pz_out,e_out,"
               "resZ,resA,resEexc_MeV,resP,resPc,resH,draws\n");
  const double pzs[] = {0.0, 500.0};
  for (const Nuclide& nu : kNuclides) {
    for (const double e : kExc) {
      for (const ExcitonConfig& ex : kExcitons) {
        for (const double pz : pzs) {
          for (int phase = 0; phase < 8; ++phase) {
            G4Fragment frag = make_fragment(nu.z, nu.a, e * MeV, pz * MeV, ex);
            eng->reset(phase);
            const double total = emission.GetTotalProbability(frag);
            if (!(total > 0.0)) { continue; }
            G4ReactionProduct* rp = emission.PerformEmission(frag);
            if (rp == nullptr) { continue; }
            const G4ThreeVector mom = rp->GetMomentum();
            std::fprintf(f,
                         "%d,%d,%.17g,%.17g,%d,%d,%d,%d,%.17g,%d,%.17g,%.17g,%.17g,%.17g,"
                         "%.17g,%d,%d,%.17g,%d,%d,%d,%d\n",
                         nu.z, nu.a, e, pz, ex.p, ex.pc, ex.h, phase, total,
                         rp->GetDefinition()->GetPDGEncoding(), rp->GetKineticEnergy() / MeV,
                         mom.x() / MeV, mom.y() / MeV, mom.z() / MeV,
                         rp->GetTotalEnergy() / MeV, frag.GetZ_asInt(), frag.GetA_asInt(),
                         frag.GetExcitationEnergy() / MeV, frag.GetNumberOfParticles(),
                         frag.GetNumberOfCharged(), frag.GetNumberOfHoles(), eng->draws());
            delete rp;
          }
        }
      }
    }
  }
  std::fclose(f);
  CLHEP::HepRandom::setTheEngine(saved);
  delete eng;
}

// ---------------------------------------------------------------------------------------------

struct Campaign {
  int Z, A;
  double eexc;    ///< MeV
  double pz;      ///< MeV/c along z
  int p, pc, h;
};

/// preco_deexcite.csv - the whole model, statistically.
///
/// Fifteen fragments with chosen exciton configurations, 20,000 calls each under one seed.
///
/// Products are keyed by (PDG, is_preco), and `is_preco` is the LEADING RUN of products whose
/// creator model id is PRECO **and** whose PDG is one of the six pre-equilibrium ejectiles.
/// The raw creator id alone is not enough, and the failure is not hypothetical: Al27 at
/// E* = 30 MeV produced 2,347 Mg25 residuals carrying the PRECO id, because
/// G4PreCompoundEmission::PerformEmission stamps PRECO on the RESIDUAL fragment as well as on
/// the ejectile, and G4ExcitationHandler then released that residual untouched (Mg25 is a
/// natural isotope at zero excitation) and copied the fragment's id onto the product. So the
/// id says "this product's fragment was last touched by PRECO", not "this product was emitted
/// pre-equilibrium".
///
/// The leading-run rule is exact, because G4PreCompoundModel::DeExcite pushes each ejectile as
/// it is emitted and PerformEquilibriumEmission then INSERTS the handler's whole output at the
/// end - so the pre-equilibrium products are precisely the first `n` entries. It is also the
/// rule the port applies to its own list, and the boundary itself is checked independently by
/// preco_deexcite_mult.csv, which histograms `n` per event.
void dump_deexcite() {
  G4PreCompoundModel model;
  model.InitialiseModel();
  const G4int preco_id = G4PhysicsModelCatalog::GetModelID("model_PRECO");

  const Campaign cs[] = {
    // light: Li6 is the first nuclide above both entry gates
    {3, 6, 5.0, 0.0, 2, 1, 1},     {3, 6, 20.0, 0.0, 3, 1, 2},
    {6, 12, 10.0, 0.0, 2, 1, 1},   {6, 12, 50.0, 0.0, 4, 2, 2},
    // medium
    {13, 27, 30.0, 0.0, 2, 1, 1},  {20, 40, 80.0, 0.0, 4, 2, 2},
    {26, 56, 5.0, 0.0, 2, 1, 1},   {26, 56, 50.0, 0.0, 3, 1, 2},
    {26, 56, 100.0, 0.0, 6, 3, 3}, {26, 56, 50.0, 500.0, 4, 2, 2},
    // heavy, into the fission-competition region of the equilibrium tail
    {47, 107, 100.0, 0.0, 5, 2, 3}, {79, 197, 50.0, 0.0, 2, 1, 1},
    {82, 208, 100.0, 0.0, 4, 2, 2}, {82, 208, 200.0, 0.0, 8, 4, 4},
    {92, 238, 60.0, 0.0, 3, 1, 2},
  };
  const int kN = 20000;

  FILE* f = std::fopen("preco_deexcite.csv", "w");
  std::fprintf(f, "Z,A,Eexc_MeV,pz_MeV,P,Pc,H,N,pdg,is_preco,count,mean_ekin_MeV,"
                  "mean_ekin2_MeV2\n");
  // preco_deexcite_species.csv carries the SAME tally folded onto (Z, A, is_preco), which is
  // the key the port can be compared on at all - the last digit of an ion's PDG code is an
  // isomer index G4IonTable assigns per run, so the port emits pdg = 0 for a heavy product.
  //
  // It exists for one column. `count`, `sum_ekin` and `sum_ekin2` are additive, so a test can
  // fold the file above itself and 877 of the rows in this campaign do merge that way. The
  // second moment of the per-event MULTIPLICITY is not additive: E[(k_a+k_b)^2] carries a cross
  // term, so folding E[k^2] over isomers understates the variance of the sum and makes the
  // comparison stricter than the data supports. It is therefore accumulated here on the folded
  // key. `count` is repeated so the test can assert its own folding against this one.
  //
  // The variance matters because the port's alone will not do: the port emits exactly one proton
  // in every 20 MeV p + C12 event, so its measured variance is zero, and Geant4 emits one in
  // 19,996 of 20,000 - a 2.0 sigma difference that a port-side-only variance reports as 1e9.
  FILE* s = std::fopen("preco_deexcite_species.csv", "w");
  std::fprintf(s, "Z,A,Eexc_MeV,pz_MeV,P,Pc,H,N,spZ,spA,is_preco,count,mean_mult2\n");
  FILE* g = std::fopen("preco_deexcite_residual.csv", "w");
  std::fprintf(g, "Z,A,Eexc_MeV,pz_MeV,P,Pc,H,N,resZ,resA,count\n");
  FILE* h = std::fopen("preco_deexcite_mult.csv", "w");
  std::fprintf(h, "Z,A,Eexc_MeV,pz_MeV,P,Pc,H,N,n_preco_ejectiles,count\n");

  for (const Campaign& c : cs) {
    CLHEP::HepRandom::setTheSeed(20260913 + c.Z * 1000 + c.A * 7 + int(c.eexc));
    std::map<int, long long> count;              // key = pdg*2 + is_preco
    std::map<int, double> sum_e, sum_e2;
    std::map<int, long long> sp_count;            // key = ((Z+500)*1000 + A)*2 + is_preco
    std::map<int, double> sp_sum_k2;
    std::map<int, int> per_event;                 // the same key -> multiplicity this event
    std::map<int, long long> residual;           // key = 1000*Z + A of the heaviest product
    std::map<int, long long> mult;               // n_preco_ejectiles -> events
    for (int n = 0; n < kN; ++n) {
      G4Fragment frag = make_fragment(c.Z, c.A, c.eexc * MeV, c.pz * MeV,
                                      ExcitonConfig{c.p, c.pc, c.h});
      // Left at G4Fragment's default -1 rather than PRECO, so that a fragment released
      // untouched by the handler is distinguishable from a pre-equilibrium ejectile.
      frag.SetCreatorModelID(-1);
      G4ReactionProductVector* out = model.DeExcite(frag);
      if (out == nullptr) { continue; }
      int heaviestA = -1, hz = 0, ha = 0;
      int n_preco_ej = 0;
      bool still_leading = true;
      per_event.clear();
      for (G4ReactionProduct* rp : *out) {
        const int pdg = rp->GetDefinition()->GetPDGEncoding();
        const double ekin = rp->GetKineticEnergy() / MeV;
        // The leading run of PRECO-tagged products that are one of the six ejectiles - see
        // this function's header for why the raw id is not the tag.
        const bool is_ejectile = (pdg == 2112 || pdg == 2212 || pdg == 1000010020 ||
                                  pdg == 1000010030 || pdg == 1000020030 ||
                                  pdg == 1000020040);
        int is_preco = 0;
        if (still_leading && rp->GetCreatorModelID() == preco_id && is_ejectile) {
          is_preco = 1;
          ++n_preco_ej;
        } else {
          still_leading = false;
        }
        const int key = pdg * 2 + is_preco;
        ++count[key];
        sum_e[key] += ekin;
        sum_e2[key] += ekin * ekin;
        int z = 0, a = 0;
        if (pdg > 1000000000) {
          a = (pdg / 10) % 1000;
          z = (pdg / 10000) % 1000;
        } else if (pdg == 2112) { a = 1; z = 0; }
        else if (pdg == 2212) { a = 1; z = 1; }
        // The species key, which is (Z, A) except that a conversion electron is given Z = -1 so
        // that it is not merged with the gammas. `z`/`a` above are left alone because the
        // residual histogram below is keyed on them and must not move.
        const int spz = (pdg == 11) ? -1 : z;
        const int spkey = ((spz + 500) * 1000 + a) * 2 + is_preco;
        ++sp_count[spkey];
        ++per_event[spkey];
        if (a > heaviestA) { heaviestA = a; hz = z; ha = a; }
        delete rp;
      }
      delete out;
      for (const auto& kv : per_event) {
        sp_sum_k2[kv.first] += double(kv.second) * kv.second;
      }
      ++residual[1000 * hz + ha];
      ++mult[n_preco_ej];
    }
    for (const auto& kv : count) {
      const long long m = kv.second;
      std::fprintf(f, "%d,%d,%.17g,%.17g,%d,%d,%d,%d,%d,%d,%lld,%.17g,%.17g\n", c.Z, c.A,
                   c.eexc, c.pz, c.p, c.pc, c.h, kN, kv.first / 2, kv.first % 2, m,
                   sum_e[kv.first] / double(m), sum_e2[kv.first] / double(m));
    }
    for (const auto& kv : sp_count) {
      const int za = kv.first / 2;
      std::fprintf(s, "%d,%d,%.17g,%.17g,%d,%d,%d,%d,%d,%d,%d,%lld,%.17g\n", c.Z, c.A, c.eexc,
                   c.pz, c.p, c.pc, c.h, kN, za / 1000 - 500, za % 1000, kv.first % 2,
                   kv.second, sp_sum_k2[kv.first] / double(kN));
    }
    for (const auto& kv : residual) {
      std::fprintf(g, "%d,%d,%.17g,%.17g,%d,%d,%d,%d,%d,%d,%lld\n", c.Z, c.A, c.eexc, c.pz,
                   c.p, c.pc, c.h, kN, kv.first / 1000, kv.first % 1000, kv.second);
    }
    for (const auto& kv : mult) {
      std::fprintf(h, "%d,%d,%.17g,%.17g,%d,%d,%d,%d,%d,%lld\n", c.Z, c.A, c.eexc, c.pz, c.p,
                   c.pc, c.h, kN, kv.first, kv.second);
    }
  }
  std::fclose(f);
  std::fclose(s);
  std::fclose(g);
  std::fclose(h);
}

// ---------------------------------------------------------------------------------------------

/// preco_apply.csv - G4PreCompoundModel::ApplyYourself, the model as a hadronic interaction.
///
/// **Why this is statistical and not exact.** The initial fragment ApplyYourself builds -
/// `G4Fragment(A + Ap, Z + Zp, p)` with `p = thePrimary.Get4Momentum() + (0,0,0,M(A,Z))`,
/// `SetNumberOfExcitedParticle(2, 1)` and `SetNumberOfHoles(1, 0)` - is a local and is never
/// handed back; DeExcite is called on it and only the final state comes out. Rebuilding that
/// fragment here and dumping its fields would compare the port against a second copy of the
/// same two lines, and the two things most worth checking - WHICH mass Geant4 adds
/// (`G4NucleiProperties::GetNuclearMass(A, Z)`, the target's, not the compound's) and WHICH
/// four-momentum it adds it to - would then be assumed on both sides instead of measured. So
/// the whole final state is dumped instead: a wrong target mass moves E* and therefore every
/// spectrum, a wrong (A + Ap, Z + Zp) moves the residual, and a wrong (2, 1, 1) exciton
/// configuration moves the pre-equilibrium multiplicity.
///
/// The nine cases are four matched (neutron, proton) pairs plus one that fails the entry gate.
/// `U` for a nucleon on a target is the projectile's kinetic energy plus its separation energy
/// from the compound, and the gate needs `fPrecoLowEnergy*(A+1) <= U <= fPrecoHighEnergy*(A+1)`;
/// the last case, a 1 MeV proton on Pb208, gives U ~ 5 MeV against a low limit of 20.9 MeV, so
/// ApplyYourself skips pre-equilibrium entirely and the row is a check that the port skips it
/// too.
///
/// `is_preco` is the same leading-run rule dump_deexcite uses, and it is exact for the same
/// reason: ApplyYourself copies `*result` in order, so the pre-equilibrium ejectiles are the
/// first entries of the secondary list.
void dump_applyyourself() {
  G4PreCompoundModel model;
  model.InitialiseModel();
  const G4int preco_id = G4PhysicsModelCatalog::GetModelID("model_PRECO");

  // Four MATCHED pairs, a neutron and a proton at the same energy on the same target, because
  // the one thing ApplyYourself does that a reading would not predict is to give the neutron
  // projectile a CHARGED particle exciton - `SetNumberOfExcitedParticle(2, 1)` is unconditional
  // - and the only way to see what that costs is to have both projectiles on one target.
  // docs/RISK.md V51.
  struct ACase { int pdg; int z, a; double ekin; };
  const ACase cs[] = {
    {2212, 6, 12, 20.0},    {2112, 6, 12, 20.0},
    {2212, 13, 27, 100.0},  {2112, 13, 27, 100.0},
    {2212, 26, 56, 100.0},  {2112, 26, 56, 100.0},
    {2212, 79, 197, 100.0}, {2112, 79, 197, 100.0},
    // Below the entry gate: U ~ 5 MeV against fPrecoLowEnergy*209 = 20.9 MeV, so ApplyYourself
    // skips pre-equilibrium entirely and every event must land in the zero bin.
    {2212, 82, 208, 1.0},
  };
  const int kN = 20000;

  FILE* f = std::fopen("preco_apply.csv", "w");
  std::fprintf(f, "proj_pdg,Z,A,ekin_MeV,N,pdg,is_preco,count,mean_ekin_MeV,"
                  "mean_ekin2_MeV2\n");
  // Folded onto (Z, A, is_preco), for the per-event multiplicity's second moment. Same reason
  // as preco_deexcite_species.csv - see its header.
  FILE* s = std::fopen("preco_apply_species.csv", "w");
  std::fprintf(s, "proj_pdg,Z,A,ekin_MeV,N,spZ,spA,is_preco,count,mean_mult2\n");
  FILE* h = std::fopen("preco_apply_mult.csv", "w");
  std::fprintf(h, "proj_pdg,Z,A,ekin_MeV,N,n_preco_ejectiles,count\n");

  for (const ACase& c : cs) {
    const G4ParticleDefinition* part =
        (c.pdg == 2212) ? static_cast<const G4ParticleDefinition*>(G4Proton::Proton())
                        : static_cast<const G4ParticleDefinition*>(G4Neutron::Neutron());
    CLHEP::HepRandom::setTheSeed(20260914 + c.pdg + c.z * 1000 + int(c.ekin));
    std::map<int, long long> count;
    std::map<int, double> sum_e, sum_e2;
    std::map<int, long long> sp_count;
    std::map<int, double> sp_sum_k2;
    std::map<int, int> per_event;
    std::map<int, long long> mult;
    for (int n = 0; n < kN; ++n) {
      G4DynamicParticle dp(part, G4ThreeVector(0, 0, 1), c.ekin * MeV);
      G4HadProjectile proj(dp);
      G4Nucleus nucleus(c.a, c.z);
      G4HadFinalState* r = model.ApplyYourself(proj, nucleus);
      if (r == nullptr) { continue; }
      int n_preco_ej = 0;
      bool still_leading = true;
      per_event.clear();
      const std::size_t ns = r->GetNumberOfSecondaries();
      for (std::size_t i = 0; i < ns; ++i) {
        const G4HadSecondary* s = r->GetSecondary(i);
        const int pdg = s->GetParticle()->GetDefinition()->GetPDGEncoding();
        const double ekin = s->GetParticle()->GetKineticEnergy() / MeV;
        const bool is_ejectile = (pdg == 2112 || pdg == 2212 || pdg == 1000010020 ||
                                  pdg == 1000010030 || pdg == 1000020030 ||
                                  pdg == 1000020040);
        int is_preco = 0;
        if (still_leading && s->GetCreatorModelID() == preco_id && is_ejectile) {
          is_preco = 1;
          ++n_preco_ej;
        } else {
          still_leading = false;
        }
        const int key = pdg * 2 + is_preco;
        ++count[key];
        sum_e[key] += ekin;
        sum_e2[key] += ekin * ekin;
        int z = 0, a = 0;
        if (pdg > 1000000000) {
          a = (pdg / 10) % 1000;
          z = (pdg / 10000) % 1000;
        } else if (pdg == 2112) { a = 1; z = 0; }
        else if (pdg == 2212) { a = 1; z = 1; }
        else if (pdg == 11) { z = -1; }
        const int spkey = ((z + 500) * 1000 + a) * 2 + is_preco;
        ++sp_count[spkey];
        ++per_event[spkey];
      }
      for (const auto& kv : per_event) {
        sp_sum_k2[kv.first] += double(kv.second) * kv.second;
      }
      ++mult[n_preco_ej];
      // theResult is a member of G4HadronicInteraction and is Clear()ed at the top of the next
      // ApplyYourself, which deletes the G4HadSecondary objects. Nothing to free here, and
      // freeing it would double-delete on the next call.
    }
    for (const auto& kv : count) {
      std::fprintf(f, "%d,%d,%d,%.17g,%d,%d,%d,%lld,%.17g,%.17g\n", c.pdg, c.z, c.a,
                   c.ekin, kN, kv.first / 2, kv.first % 2, kv.second,
                   sum_e[kv.first] / double(kv.second), sum_e2[kv.first] / double(kv.second));
    }
    for (const auto& kv : sp_count) {
      const int za = kv.first / 2;
      std::fprintf(s, "%d,%d,%d,%.17g,%d,%d,%d,%d,%lld,%.17g\n", c.pdg, c.z, c.a, c.ekin, kN,
                   za / 1000 - 500, za % 1000, kv.first % 2, kv.second,
                   sp_sum_k2[kv.first] / double(kN));
    }
    for (const auto& kv : mult) {
      std::fprintf(h, "%d,%d,%d,%.17g,%d,%d,%lld\n", c.pdg, c.z, c.a, c.ekin, kN, kv.first,
                   kv.second);
    }
  }
  std::fclose(f);
  std::fclose(s);
  std::fclose(h);
}

// ---------------------------------------------------------------------------------------------

/// preco_propagate.csv and preco_propagate_nucleons.csv -
/// G4GeneratorPrecompoundInterface::Propagate on a real wounded nucleus.
///
/// The nucleus is a G4Fancy3DNucleus, initialised and then wounded by calling SetHit on the
/// first `nhit` nucleons of its own loop order. Its nucleon positions and Fermi momenta are
/// SAMPLED at Init, so they are an input the port cannot invent: the whole nucleon list is
/// dumped (charge, four-momentum, PDG mass, binding energy, hit flag) into
/// preco_propagate_nucleons.csv, and the port replays it.
///
/// The tracks are hand-built protons and neutrons at prescribed positions - some inside the
/// nuclear radius, some outside - and the CycleEngine makes each capture test deterministic.
/// Each track carries creator model id `50000 + i`, which
/// `G4GeneratorPrecompoundInterface::Propagate` copies onto the G4ReactionProduct it makes for
/// an escaping track, so the escaped SET is identified exactly rather than matched by momentum.
///
/// That set is the whole capture logic: it fixes anA, aZ, numberOfEx, numberOfCh and
/// captured4Momentum, hence the fragment DeExcite receives. The de-excitation products that
/// follow are summed rather than compared one by one, because they are P3's and P6's samplers
/// under a fixed eight-value cycle and their entry-by-entry comparison belongs in
/// preco_deexcite.csv.
///
/// **Two findings this dump had to be built around, both worth the reader's minute.**
///
/// 1. `G4Fancy3DNucleus::ChooseFermiMomenta` gives every nucleon
///    `E = PDGMass - BindingEnergy/A` on top of a Fermi three-momentum, so
///    `Get4Momentum().mag()` is BELOW the PDG mass for every one of them. Propagate's
///    "Check that we use QGS model" loop is exactly that test on the HIT nucleons - so a
///    wounded nucleus whose hit nucleons were left as Init made them selects the QGS branch,
///    always. G4FTFModel does not hit this because it re-sets each hit nucleon's energy to the
///    on-shell `sqrt(mt2 + pz^2)` form before handing over; a bare Fancy3DNucleus does. The
///    `on_shell` cases below put the hit nucleons back on their mass shell so that the FTF
///    branch is reached at all, and case 3 leaves them off shell so the QGS branch is too.
/// 2. The QGS branch dereferences `GetPrimaryProjectile()` with no null check, and that
///    pointer is null until a cascade calls `SetPrimaryProjectile`. So an interface used
///    without one crashes there rather than reporting. One is set here for every case.
void dump_propagate() {
  auto* eng = new CycleEngine();
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();

  FILE* fn = std::fopen("preco_propagate_nucleons.csv", "w");
  std::fprintf(fn, "case,Z,A,radius_fm,idx,charge,hit,px,py,pz,e,pdgmass,binding\n");
  FILE* f = std::fopen("preco_propagate.csv", "w");
  std::fprintf(f, "case,on_shell,qgsm,phase,prim_pz,prim_e,n_escaped,escaped_ids,sum_px,"
                  "sum_py,sum_pz,sum_e,n_deex,deex_px,deex_py,deex_pz,deex_e,draws\n");

  struct PCase { int z, a, nhit; int on_shell; };
  const PCase cases[] = {{6, 12, 2, 1}, {13, 27, 3, 1}, {26, 56, 4, 1}, {26, 56, 4, 0}};

  // The primary projectile every case is given: a 1 GeV proton along +z. Only the QGS branch
  // reads it, and it reads Get4Momentum() alone.
  G4DynamicParticle prim_dp(G4Proton::Proton(), G4ThreeVector(0., 0., 1.), 1000.0 * MeV);
  G4HadProjectile primary(prim_dp);

  // The hand-built track list: protons and neutrons at three radii and three kinetic energies,
  // plus one pion so that the "not a nucleon" arm is exercised. Positions in fermi.
  struct TSpec { int pdg; double r_fm; double ekin_MeV; };
  const TSpec tspecs[] = {
    {2212, 0.5, 5.0},  {2112, 0.5, 20.0}, {2212, 1.5, 60.0}, {2112, 1.5, 150.0},
    {2212, 8.0, 30.0}, {2112, 2.5, 10.0}, {211, 0.5, 80.0},  {2212, 0.5, 200.0},
  };

  int case_index = 0;
  for (const PCase& pc : cases) {
    // The nucleus is built once per case under a fixed JamesRandom seed so that its nucleon
    // sample is reproducible, and only then is the engine swapped for the cycle.
    CLHEP::HepRandom::setTheEngine(saved);
    CLHEP::HepRandom::setTheSeed(4242 + case_index);
    G4Fancy3DNucleus nucleus;
    nucleus.Init(pc.a, pc.z);
    const double radius = nucleus.GetNuclearRadius();

    int idx = 0;
    G4Nucleon* nuc = nucleus.StartLoop() ? nucleus.GetNextNucleon() : nullptr;
    while (nuc != nullptr) {
      // G4Nucleon::Hit(G4int) is how a cascade marks a nucleon wounded without owning a
      // G4VSplitableHadron: it writes the sentinel pointer 1111 into theSplitableHadron, which
      // is what AreYouHit() tests. The G4int overload exists for exactly this.
      if (idx < pc.nhit) {
        nuc->Hit(1);
        if (pc.on_shell != 0) {
          // Put the hit nucleon back on its mass shell, which is what G4FTFModel leaves
          // behind and what Init does NOT - see finding 1 above. Nothing else about the
          // nucleon is touched, and the whole list is dumped either way.
          const G4LorentzVector p = nuc->Get4Momentum();
          const double m = nuc->GetDefinition()->GetPDGMass();
          nuc->SetMomentum(G4LorentzVector(p.vect(), std::sqrt(p.vect().mag2() + m * m)));
        }
      }
      ++idx;
      nuc = nucleus.GetNextNucleon();
    }
    // Propagate's "Check that we use QGS model" loop, verbatim and on the same object, so that
    // the branch it selects is a dumped number and not an inference from `on_shell`. It is not
    // a second implementation of anything: the whole test is one accessor against another.
    //
    // It matters that this is measured rather than assumed. Setting a nucleon's energy to
    // `sqrt(p^2 + m^2)` does NOT guarantee `Get4Momentum().mag() >= m` afterwards - the round
    // trip through a square root can land a half-ulp low - so the predicate is ulp-sensitive
    // and the `on_shell` intent above does not decide the branch by itself.
    G4bool qgsm = false;
    nuc = nucleus.StartLoop() ? nucleus.GetNextNucleon() : nullptr;
    while (nuc != nullptr) {
      if (nuc->AreYouHit() &&
          nuc->Get4Momentum().mag() < nuc->GetDefinition()->GetPDGMass()) {
        qgsm = true;
      }
      nuc = nucleus.GetNextNucleon();
    }

    idx = 0;
    nuc = nucleus.StartLoop() ? nucleus.GetNextNucleon() : nullptr;
    while (nuc != nullptr) {
      const G4LorentzVector p = nuc->Get4Momentum();
      std::fprintf(fn, "%d,%d,%d,%.17g,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                   case_index, pc.z, pc.a, radius / fermi, idx,
                   G4int(nuc->GetDefinition()->GetPDGCharge() / eplus + 0.1),
                   nuc->AreYouHit() ? 1 : 0, p.x() / MeV, p.y() / MeV, p.z() / MeV,
                   p.e() / MeV, nuc->GetDefinition()->GetPDGMass() / MeV,
                   nuc->GetBindingEnergy() / MeV);
      ++idx;
      nuc = nucleus.GetNextNucleon();
    }

    for (int phase = 0; phase < 4; ++phase) {
      // Propagate deletes the tracks and the vector, and SetHit is idempotent, so the nucleus
      // is reused across phases - which is what makes the nucleon dump above valid for all of
      // them.
      G4KineticTrackVector* tracks = new G4KineticTrackVector;
      int i = 0;
      for (const TSpec& ts : tspecs) {
        const G4ParticleDefinition* def = nullptr;
        if (ts.pdg == 2212) { def = G4Proton::Proton(); }
        else if (ts.pdg == 2112) { def = G4Neutron::Neutron(); }
        else { def = G4ParticleTable::GetParticleTable()->FindParticle(ts.pdg); }
        const double m = def->GetPDGMass();
        const double pmag = std::sqrt(ts.ekin_MeV * MeV * (ts.ekin_MeV * MeV + 2.0 * m));
        // A direction that is not along an axis, so the boost and the radius test are both
        // non-degenerate; fixed, not sampled.
        const G4ThreeVector dir = G4ThreeVector(0.3, -0.5, 0.81).unit();
        const G4LorentzVector p4(pmag * dir, ts.ekin_MeV * MeV + m);
        const G4ThreeVector pos = ts.r_fm * fermi * G4ThreeVector(0.6, 0.48, -0.64).unit();
        auto* t = new G4KineticTrack(def, 0.0, pos, p4);
        t->SetCreatorModelID(50000 + i);
        tracks->push_back(t);
        ++i;
      }

      G4PreCompoundModel* preco = new G4PreCompoundModel();
      preco->InitialiseModel();
      G4GeneratorPrecompoundInterface iface(preco);
      iface.SetPrimaryProjectile(primary);   // finding 2: the QGS branch has no null check

      CLHEP::HepRandom::setTheEngine(eng);
      eng->reset(phase);
      G4ReactionProductVector* out = iface.Propagate(tracks, &nucleus);
      const int draws = eng->draws();
      CLHEP::HepRandom::setTheEngine(saved);

      std::string ids;
      int n_escaped = 0, n_deex = 0;
      G4LorentzVector esc(0., 0., 0., 0.), dx(0., 0., 0., 0.);
      if (out != nullptr) {
        for (G4ReactionProduct* rp : *out) {
          const G4LorentzVector p(rp->GetMomentum(), rp->GetTotalEnergy());
          const int cid = rp->GetCreatorModelID();
          if (cid >= 50000 && cid < 50000 + int(sizeof tspecs / sizeof tspecs[0])) {
            ++n_escaped;
            esc += p;
            if (!ids.empty()) { ids += "|"; }
            ids += std::to_string(cid - 50000);
          } else {
            ++n_deex;
            dx += p;
          }
          delete rp;
        }
        delete out;
      }
      const G4LorentzVector pp = primary.Get4Momentum();
      std::fprintf(f,
                   "%d,%d,%d,%d,%.17g,%.17g,%d,%s,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,"
                   "%.17g,%.17g,%d\n",
                   case_index, pc.on_shell, qgsm ? 1 : 0, phase, pp.z() / MeV, pp.e() / MeV,
                   n_escaped, ids.c_str(), esc.x() / MeV, esc.y() / MeV, esc.z() / MeV,
                   esc.e() / MeV, n_deex, dx.x() / MeV, dx.y() / MeV, dx.z() / MeV,
                   dx.e() / MeV, draws);
      std::fflush(f);
      // `iface` owns `preco` through SetDeExcitation and deletes nothing; the model is
      // registered in G4HadronicInteractionRegistry and outlives this scope, which is what the
      // registry is for. No delete here, deliberately - deleting it would leave a dangling
      // entry in the registry that the next case's FindModel("PRECO") would return.
    }
    ++case_index;
  }
  std::fclose(f);
  std::fclose(fn);
  CLHEP::HepRandom::setTheEngine(saved);
  delete eng;
}

// ---------------------------------------------------------------------------------------------

void dump_precompound(const DumpContext&) {
  dump_params();
  dump_model_ids();
  dump_channels();
  dump_sample();
  dump_transitions();
  dump_equilibrium();
  dump_gates();
  dump_emission();
  dump_deexcite();
  dump_applyyourself();
  dump_propagate();
}

}  // namespace

G4GPU_REGISTER_DUMP("precompound",
                    "preco_params.csv preco_modelids.csv preco_channels.csv preco_sample.csv "
                    "preco_transitions.csv preco_equilibrium.csv preco_gates.csv "
                    "preco_emission.csv "
                    "preco_deexcite.csv preco_deexcite_species.csv preco_deexcite_residual.csv "
                    "preco_deexcite_mult.csv "
                    "preco_apply.csv preco_apply_species.csv preco_apply_mult.csv "
                    "preco_propagate.csv preco_propagate_nucleons.csv",
                    dump_precompound);
