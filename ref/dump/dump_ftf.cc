// FTFP's oracle: what Geant4 11.1.1 answers for G4FTFParameters, for the Lund fragmentation
// tables, and for the samplers under them - so the port can be diffed against the install
// rather than against a reading of it.
//
// Files:
//
//   ftf_params.csv       G4FTFParameters::InitForInteraction on a (projectile, target,
//                        Plab-per-particle) grid: every getter, plus the whole public
//                        ProcParams[5][7]. Deterministic - InitForInteraction draws no
//                        random numbers - so this is an exact oracle with no phase column.
//   ftf_procprob.csv     GetProcProb(ProcN, y) over a rapidity grid on a few of those rows.
//                        The table is dumped as well; this is the *function* over it, which
//                        is what the model actually calls, including the y < Ymin branch.
//   ftf_lund_params.csv  the scalar parameters of G4VLongitudinalStringDecay /
//                        G4LundStringFragmentation as the QBBC chain constructs them.
//   ftf_lund_tables.csv  SetMinMasses()'s output: minMassQQbarStr[5][5],
//                        minMassQDiQStr[5][5][5], Meson[5][5][7] + MesonWeight,
//                        Baryon[5][5][5][4] + BaryonWeight, Qcharge[5], Prob_QQbar[5].
//   ftf_hadrons.csv      every entry of the initialised G4ParticleTable: PDG code, name,
//                        sub-type, mass, width, charge, baryon number, short-lived flag and
//                        G4SampleResonance's minimum mass. This is the table the port has to
//                        carry, because FindParticle(code) is how every one of the classes
//                        above turns a quark pair into a hadron. tools/ftf_hadrons.pl turns
//                        it into src/data/ftf_hadrons.hh with the counts asserted.
//   ftf_minmass.csv      SetMinimalStringMass over parton pairs - the tables above read
//                        through the function that reads them, including the DiQuark -
//                        AntiDiQuark re-arrangement arm that no single table entry shows.
//   ftf_build.csv        G4HadronBuilder::Build / BuildLowSpin / BuildHighSpin for every
//                        parton pair under a prescribed uniform cycle, with the draw count.
//                        This is what makes the meson and baryon mixing tables *measured*.
//   ftf_samplers.csv     SampleQuarkFlavor, SampleQuarkPt and GetLightConeZ under the cycle.
//   ftf_fragment.csv     G4LundStringFragmentation::FragmentString on constructed strings
//                        under the cycle: every hadron, its four-momentum and formation
//                        time, in order.
//   ftf_fragstat.csv     the same strings, N = 20,000 under a fixed seed: multiplicity by
//                        species and the z / pT moments. The statistical half.
//
// WHY THE EXACT HALF CAN BE EXACT. Geant4 lets a random engine be installed, so every
// sampler here is driven under an eight-value uniform cycle and becomes a deterministic
// function of (inputs, phase). Each row carries the number of deviates the call consumed as
// well as its answer, because a transcription that produces a plausible hadron from the wrong
// number of draws is wrong in a way only the count reveals - G4HadronBuilder::Meson spends
// one draw on the spin and a second on the scalar/vector mixing only for a neutral
// light-quark pair, and that second draw is the easy one to omit.
//
// THE THREE NUMBERS NO RUN CAN BE ASKED FOR. G4LundStringFragmentation::Tmt (190 MeV) is a
// private member with no accessor and no setter; G4FTFParameters' EnableDiffDissociation...
// flag is public but its two consumers are inside InitForInteraction; and
// G4FTFTunings::fApplicabilityOfTunes has a getter, so the tune index IS dumped. Tmt is
// checked only through the Pt spectrum in ftf_fragment.csv, and its header says so.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "G4AntiNeutron.hh"
#include "G4AntiProton.hh"
#include "G4Deuteron.hh"
#include "G4DynamicParticle.hh"
#include "G4DiffractiveExcitation.hh"
#include "G4DiffractiveSplitableHadron.hh"
#include "G4ElasticHNScattering.hh"
#include "G4ExcitedString.hh"
#include "G4ExcitedStringDecay.hh"
#include "G4FTFModel.hh"
#include "G4FTFParameters.hh"
#include "G4FTFParticipants.hh"
#include "G4Fancy3DNucleus.hh"
#include "G4InteractionContent.hh"
#include "G4Lambda.hh"
#include "G4Nucleon.hh"
#include "G4Nucleus.hh"
#include "G4ReactionProduct.hh"
#include "G4V3DNucleus.hh"
#include "G4FTFTunings.hh"
#include "G4FragmentingString.hh"
#include "G4HadronBuilder.hh"
#include "G4HadronicParameters.hh"
#include "G4IonTable.hh"
#include "G4Alpha.hh"
#include "G4AntiLambda.hh"
#include "G4GenericIon.hh"
#include "G4HadronicInteraction.hh"
#include "G4HadronicProcess.hh"
#include "G4He3.hh"
#include "G4KaonMinus.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4Triton.hh"
#include "G4VProcess.hh"
#include "G4KaonPlus.hh"
#include "G4KineticTrack.hh"
#include "G4KineticTrackVector.hh"
#include "G4LundStringFragmentation.hh"
#include "G4Neutron.hh"
#include "G4Parton.hh"
#include "G4ParticleTable.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4PionZero.hh"
#include "G4Proton.hh"
#include "G4SampleResonance.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/JamesRandom.h"

namespace {

// ---------------------------------------------------------------------------------------------
// A CLHEP engine that returns a prescribed cycle and counts the draws.
//
// The same eight values and the same reasoning as dump_elastic.cc's and dump_precompound.cc's,
// and the port's tests read the same eight, so every sampler below is a deterministic function
// of (inputs, phase).
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
// Reaching the protected state of the string-decay classes without editing the install.
//
// MassCut, SigmaQT, DiquarkSuppress, DiquarkBreakProb, StrangeSuppress, the two loop limits,
// the four mixing vectors, the four heavy-quark probabilities, MaxMass, Kappa and `hadronizer`
// are all `protected` in G4VLongitudinalStringDecay, and PossibleHadronMass, ProduceOneHadron,
// FindParticle and GetLightConeZ are protected members too. A derived class can read them, so
// the oracle derives one. Nothing in the install is modified - which matters, because editing
// the install would invalidate the oracle it is the oracle of (docs/PORTED.md 2.4 makes the
// same point about G4ElasticHadrNucleusHE).
// ---------------------------------------------------------------------------------------------

class LundProbe : public G4LundStringFragmentation {
 public:
  using G4LundStringFragmentation::ClusterLoopInterrupt;
  using G4LundStringFragmentation::DiquarkBreakProb;
  using G4LundStringFragmentation::DiquarkSuppress;
  using G4LundStringFragmentation::Kappa;
  using G4LundStringFragmentation::MassCut;
  using G4LundStringFragmentation::MaxMass;
  using G4LundStringFragmentation::ProbBBbar;
  using G4LundStringFragmentation::ProbCB;
  using G4LundStringFragmentation::ProbCCbar;
  using G4LundStringFragmentation::ProbEta_b;
  using G4LundStringFragmentation::ProbEta_c;
  using G4LundStringFragmentation::SigmaQT;
  using G4LundStringFragmentation::StrangeSuppress;
  using G4LundStringFragmentation::StringLoopInterrupt;
  using G4LundStringFragmentation::hadronizer;
  using G4LundStringFragmentation::pspin_barion;
  using G4LundStringFragmentation::pspin_meson;
  using G4LundStringFragmentation::scalarMesonMix;
  using G4LundStringFragmentation::vectorMesonMix;

  using G4LundStringFragmentation::FindParticle;
  using G4LundStringFragmentation::PossibleHadronMass;
  using G4LundStringFragmentation::ProduceOneHadron;
};

// ---------------------------------------------------------------------------------------------
// Reaching the four samplers G4LundStringFragmentation declares PRIVATE.
//
// GetLightConeZ, Sample4Momentum, StopFragmenting and IsItFragmentable are protected virtuals
// in G4VLongitudinalStringDecay and PRIVATE overrides in G4LundStringFragmentation, so a
// derived class cannot name them and `using` them is C2876. They are still the four decisions
// the fragmentation is made of, and validating them only through FragmentString would make the
// whole chain one all-or-nothing comparison - the shape docs/RISK.md V52 warns about from the
// other side: a check that cannot say WHICH line is wrong.
//
// So they are reached with the standard explicit-instantiation access route: access control is
// not applied to the template-arguments of an explicit instantiation ([temp.spec]), and the
// injected friend carries the pointer-to-member out. Nothing in the install is modified and no
// cast is undefined - the pointer has exactly the declared type.
// ---------------------------------------------------------------------------------------------

template <typename Tag, typename Tag::type M> struct PrivateBridge {
  friend typename Tag::type bridge(Tag) { return M; }
};

struct TagLcz {
  using type = G4double (G4LundStringFragmentation::*)(G4double, G4double, G4int,
                                                       G4ParticleDefinition*, G4double,
                                                       G4double);
  friend type bridge(TagLcz);
};
template struct PrivateBridge<TagLcz, &G4LundStringFragmentation::GetLightConeZ>;

struct TagS4M {
  using type = void (G4LundStringFragmentation::*)(G4LorentzVector*, G4double,
                                                   G4LorentzVector*, G4double, G4double);
  friend type bridge(TagS4M);
};
template struct PrivateBridge<TagS4M, &G4LundStringFragmentation::Sample4Momentum>;

struct TagStop {
  using type = G4bool (G4LundStringFragmentation::*)(const G4FragmentingString* const);
  friend type bridge(TagStop);
};
template struct PrivateBridge<TagStop, &G4LundStringFragmentation::StopFragmenting>;

struct TagFragmentable {
  using type = G4bool (G4LundStringFragmentation::*)(const G4FragmentingString* const);
  friend type bridge(TagFragmentable);
};
template struct PrivateBridge<TagFragmentable, &G4LundStringFragmentation::IsItFragmentable>;

// The three functions that ENUMERATE SplitLast's final states. They are the one part of the
// fragmentation the eight-value cycle cannot resolve: they fill FS_LeftHadron/FS_RightHadron/
// FS_Weight with up to 350 entries and SampleState then collapses all of that to one index, so
// a weight change of a few per cent moves no index and is invisible in ftf_fragment.csv - and,
// measured, is still invisible at 20,000 events per case (about 0.7 sigma). Dumping the list
// itself makes the weights exact. FS_* and NumberOf_FS are public members of
// G4VLongitudinalStringDecay; only these three methods are private.
using LastSplitFn = G4bool (G4LundStringFragmentation::*)(G4FragmentingString*&,
                                                          G4ParticleDefinition*&,
                                                          G4ParticleDefinition*&);
struct TagQQbarLast {
  using type = LastSplitFn;
  friend type bridge(TagQQbarLast);
};
template struct PrivateBridge<TagQQbarLast,
                              &G4LundStringFragmentation::Quark_AntiQuark_lastSplitting>;

struct TagQDiQLast {
  using type = LastSplitFn;
  friend type bridge(TagQDiQLast);
};
template struct PrivateBridge<TagQDiQLast,
                              &G4LundStringFragmentation::Quark_Diquark_lastSplitting>;

struct TagDiQADiQAbove {
  using type = LastSplitFn;
  friend type bridge(TagDiQADiQAbove);
};
template struct PrivateBridge<
    TagDiQADiQAbove,
    &G4LundStringFragmentation::Diquark_AntiDiquark_aboveThreshold_lastSplitting>;

struct TagDiQADiQBelow {
  using type = LastSplitFn;
  friend type bridge(TagDiQADiQBelow);
};
template struct PrivateBridge<
    TagDiQADiQBelow,
    &G4LundStringFragmentation::Diquark_AntiDiquark_belowThreshold_lastSplitting>;

// ---------------------------------------------------------------------------------------------
// The grid
// ---------------------------------------------------------------------------------------------

struct Target { int z, a; const char* name; };

/// Hydrogen because `NumberOfTargetNeutrons = 0` divides the h+N average by a zero weight;
/// C and O because water and tissue are made of them; Al and Fe because the destruction
/// parameters' `NumberOfTargetNucleons > 10` and `> 26` gates fall between them (Al27 has 27
/// nucleons, so `> 26` is true - Fe56 and Pb208 too, and C12/O16 are below); Pb because it is
/// the heaviest thing the oracle grids elsewhere use.
const Target kTargets[] = {
  {1, 1, "H1"}, {6, 12, "C12"}, {8, 16, "O16"}, {13, 27, "Al27"}, {26, 56, "Fe56"},
  {82, 208, "Pb208"},
};

/// Plab per particle, MeV/c. 20 MeV/c is below the anti-baryon branch's 40 MeV low-energy
/// limit (which is a table of six constants, not a formula) and 100 is above it; 2000 is the
/// anti-baryon nuclear-destruction branch's `Plab < 2 GeV/c` boundary exactly, and 1999 and
/// 2001 straddle it; 3000 is QBBC's FTF threshold.
const double kPlab[] = {
  20.0, 100.0, 500.0, 1000.0, 1999.0, 2000.0, 2001.0, 3000.0, 6000.0, 1.0e4, 5.0e4, 1.0e5,
  1.0e6,
};

/// The projectiles QBBC can put into FTFP, plus the ones that select a different arm of
/// InitForInteraction: pi0 and K0L for the meson arms, the anti-baryons for the Arkhipov
/// parameterisation and its nine annihilation weight sets, a hyperon and an anti-hyperon for
/// the `else` arm that calls GetMinMass, and d / alpha / C12 for the nucleus arm.
std::vector<const G4ParticleDefinition*> projectiles() {
  G4ParticleTable* pt = G4ParticleTable::GetParticleTable();
  std::vector<const G4ParticleDefinition*> v;
  const int codes[] = {2212, 2112, 211, -211, 111, 321, -321, 130, -2212, -2112,
                       3122, -3122, 3112, 3222, 3312, 3334, -3334};
  for (int c : codes) {
    const G4ParticleDefinition* p = pt->FindParticle(c);
    if (p) { v.push_back(p); }
  }
  G4IonTable* it = pt->GetIonTable();
  v.push_back(it->GetIon(1, 2, 0.0));   // deuteron
  v.push_back(it->GetIon(1, 3, 0.0));   // triton
  v.push_back(it->GetIon(2, 4, 0.0));   // alpha
  v.push_back(it->GetIon(6, 12, 0.0));  // C12
  v.push_back(it->GetIon(26, 56, 0.0));  // Fe56
  return v;
}

// ---------------------------------------------------------------------------------------------
// ftf_params.csv and ftf_procprob.csv
// ---------------------------------------------------------------------------------------------

void dump_params() {
  FILE* f = std::fopen("ftf_params.csv", "w");
  std::fprintf(f,
               "proj_pdg,proj_name,tgt_z,tgt_a,plab_mev,index_tune,xtotal,xelastic,xinelastic,"
               "prob_elastic,radius2,slope,gamma0,avpt2_elastic,prob_annih,"
               "delta_prob_qexchg,prob_same_qexchg,proj_min_diff_m,proj_min_nondiff_m,"
               "tar_min_diff_m,tar_min_nondiff_m,average_pt2,prob_log_distr_prd,"
               "prob_log_distr,pt2_kink,qprob0,qprob1,qprob2,max_n_collisions,"
               "prob_of_interaction,cof_nd_proj,cof_nd_tgt,r2_nd,exci_e_per_wounded,"
               "d_nd,pt2_nd,max_pt2_nd");
  for (int p = 0; p < 5; ++p) {
    for (int q = 0; q < 7; ++q) { std::fprintf(f, ",pp_%d_%d", p, q); }
  }
  std::fprintf(f, "\n");

  G4FTFParameters par;
  const std::vector<const G4ParticleDefinition*> projs = projectiles();
  for (const G4ParticleDefinition* proj : projs) {
    for (const Target& t : kTargets) {
      for (double plab : kPlab) {
        par.InitForInteraction(proj, t.a, t.z, plab);
        const std::vector<G4double> qp = par.GetQuarkProbabilitiesAtGluonSplitUp();
        const G4int itune = G4FTFTunings::Instance()->GetIndexTune(
            proj, std::sqrt(plab * plab + proj->GetPDGMass() * proj->GetPDGMass()) -
                      proj->GetPDGMass());
        std::fprintf(f, "%d,%s,%d,%d,%.17g,%d", proj->GetPDGEncoding(),
                     proj->GetParticleName().c_str(), t.z, t.a, plab, itune);
        const double vals[] = {
            par.GetTotalCrossSection(), par.GetElasticCrossSection(),
            par.GetInelasticCrossSection(), par.GetProbabilityOfElasticScatt(),
            par.RadiusOfHNinteractions2, par.GetSlope(), par.FTFGamma0,
            par.GetAvaragePt2ofElasticScattering(), par.GetProbabilityOfAnnihilation(),
            par.GetDeltaProbAtQuarkExchange(), par.GetProbOfSameQuarkExchange(),
            par.GetProjMinDiffMass(), par.GetProjMinNonDiffMass(), par.GetTarMinDiffMass(),
            par.GetTarMinNonDiffMass(), par.GetAveragePt2(), par.GetProbLogDistrPrD(),
            par.GetProbLogDistr(), par.GetPt2Kink(),
            qp.size() > 0 ? qp[0] : -1.0, qp.size() > 1 ? qp[1] : -1.0,
            qp.size() > 2 ? qp[2] : -1.0,
            par.GetMaxNumberOfCollisions(), par.GetProbOfInteraction(),
            par.GetCofNuclearDestructionPr(), par.GetCofNuclearDestruction(),
            par.GetR2ofNuclearDestruction(), par.GetExcitationEnergyPerWoundedNucleon(),
            par.GetDofNuclearDestruction(), par.GetPt2ofNuclearDestruction(),
            par.GetMaxPt2ofNuclearDestruction()};
        for (double v : vals) { std::fprintf(f, ",%.17g", v); }
        for (int i = 0; i < 5; ++i) {
          for (int j = 0; j < 7; ++j) { std::fprintf(f, ",%.17g", par.ProcParams[i][j]); }
        }
        std::fprintf(f, "\n");
      }
    }
  }
  std::fclose(f);

  // GetProcProb over y, on one target and a few energies per projectile. `y` runs below the
  // smallest Ymin in any tune (0.0) and above the largest (2.3 for the pion's process 0), and
  // -100 and 1000 are Ymin sentinels the A > 10 branches write, so both sides of the
  // `y < Ymin` test are crossed for every process.
  f = std::fopen("ftf_procprob.csv", "w");
  std::fprintf(f, "proj_pdg,tgt_z,tgt_a,plab_mev,proc,y,prob\n");
  const double kY[] = {-1.0, 0.0, 0.5, 0.93, 1.0, 1.4, 2.0, 2.3, 3.0, 5.0, 8.0};
  for (const G4ParticleDefinition* proj : projs) {
    for (const Target& t : kTargets) {
      for (double plab : {1000.0, 6000.0, 1.0e5}) {
        par.InitForInteraction(proj, t.a, t.z, plab);
        for (int proc = 0; proc < 5; ++proc) {
          for (double y : kY) {
            std::fprintf(f, "%d,%d,%d,%.17g,%d,%.17g,%.17g\n", proj->GetPDGEncoding(), t.z,
                         t.a, plab, proc, y, par.GetProcProb(proc, y));
          }
        }
      }
    }
  }
  std::fclose(f);

  // GammaElastic / GetInelasticProbability / GetProbabilityOfInteraction over b^2, which are
  // the three functions G4FTFParticipants' impact-parameter sampling calls.
  f = std::fopen("ftf_geom.csv", "w");
  std::fprintf(f, "proj_pdg,tgt_z,tgt_a,plab_mev,b2_fm2,gamma_elastic,prob_inelastic,prob_int\n");
  const double kB2[] = {0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0};
  for (const G4ParticleDefinition* proj : projs) {
    for (const Target& t : kTargets) {
      for (double plab : {1000.0, 6000.0, 1.0e5}) {
        par.InitForInteraction(proj, t.a, t.z, plab);
        for (double b2 : kB2) {
          std::fprintf(f, "%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n", proj->GetPDGEncoding(),
                       t.z, t.a, plab, b2, par.GammaElastic(b2),
                       par.GetInelasticProbability(b2), par.GetProbabilityOfInteraction(b2));
        }
      }
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// ftf_lund_params.csv and ftf_lund_tables.csv
// ---------------------------------------------------------------------------------------------

void dump_lund_tables() {
  LundProbe* L = new LundProbe();

  FILE* f = std::fopen("ftf_lund_params.csv", "w");
  std::fprintf(f, "name,value\n");
  std::fprintf(f, "MassCut,%.17g\n", L->MassCut);
  std::fprintf(f, "SigmaQT,%.17g\n", L->SigmaQT);
  std::fprintf(f, "StrangeSuppress,%.17g\n", L->StrangeSuppress);
  std::fprintf(f, "DiquarkSuppress,%.17g\n", L->DiquarkSuppress);
  std::fprintf(f, "DiquarkBreakProb,%.17g\n", L->DiquarkBreakProb);
  std::fprintf(f, "StringLoopInterrupt,%d\n", L->StringLoopInterrupt);
  std::fprintf(f, "ClusterLoopInterrupt,%d\n", L->ClusterLoopInterrupt);
  std::fprintf(f, "pspin_barion,%.17g\n", L->pspin_barion);
  for (size_t i = 0; i < L->pspin_meson.size(); ++i) {
    std::fprintf(f, "pspin_meson_%d,%.17g\n", (int)i, L->pspin_meson[i]);
  }
  for (size_t i = 0; i < L->scalarMesonMix.size(); ++i) {
    std::fprintf(f, "scalarMesonMix_%d,%.17g\n", (int)i, L->scalarMesonMix[i]);
  }
  for (size_t i = 0; i < L->vectorMesonMix.size(); ++i) {
    std::fprintf(f, "vectorMesonMix_%d,%.17g\n", (int)i, L->vectorMesonMix[i]);
  }
  std::fprintf(f, "ProbCCbar,%.17g\n", L->ProbCCbar);
  std::fprintf(f, "ProbBBbar,%.17g\n", L->ProbBBbar);
  std::fprintf(f, "ProbCB,%.17g\n", L->ProbCB);
  std::fprintf(f, "ProbEta_c,%.17g\n", L->ProbEta_c);
  std::fprintf(f, "ProbEta_b,%.17g\n", L->ProbEta_b);
  std::fprintf(f, "Kappa,%.17g\n", L->Kappa);
  std::fprintf(f, "MaxMass,%.17g\n", L->MaxMass);
  std::fprintf(f, "Mass_of_light_quark,%.17g\n", L->Mass_of_light_quark);
  std::fprintf(f, "Mass_of_s_quark,%.17g\n", L->Mass_of_s_quark);
  std::fprintf(f, "Mass_of_c_quark,%.17g\n", L->Mass_of_c_quark);
  std::fprintf(f, "Mass_of_b_quark,%.17g\n", L->Mass_of_b_quark);
  std::fprintf(f, "Mass_of_string_junction,%.17g\n", L->Mass_of_string_junction);
  std::fprintf(f, "MinimalStringMass,%.17g\n", L->MinimalStringMass);
  std::fprintf(f, "MinimalStringMass2,%.17g\n", L->MinimalStringMass2);
  std::fprintf(f, "NumberOf_FS,%d\n", L->NumberOf_FS);
  // The two G4HadronicParameters flags this package's dispatch reads.
  std::fprintf(f, "EnableBCParticles,%d\n",
               G4HadronicParameters::Instance()->EnableBCParticles() ? 1 : 0);
  std::fprintf(f, "EnableDiffDissociationForBGreater10,%d\n",
               G4HadronicParameters::Instance()->EnableDiffDissociationForBGreater10() ? 1 : 0);
  std::fprintf(f, "EnableCRCoalescence,%d\n",
               G4HadronicParameters::Instance()->EnableCRCoalescence() ? 1 : 0);
  std::fprintf(f, "FTF_numberOfTunes,%d\n", G4FTFTunings::sNumberOfTunes);
  for (int i = 0; i < G4FTFTunings::sNumberOfTunes; ++i) {
    std::fprintf(f, "FTF_tuneState_%d,%d\n", i,
                 G4FTFTunings::Instance()->GetTuneApplicabilityState(i));
  }
  std::fclose(f);

  f = std::fopen("ftf_lund_tables.csv", "w");
  std::fprintf(f, "table,i,j,k,l,value\n");
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      std::fprintf(f, "minMassQQbarStr,%d,%d,-1,-1,%.17g\n", i, j, L->minMassQQbarStr[i][j]);
    }
  }
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        std::fprintf(f, "minMassQDiQStr,%d,%d,%d,-1,%.17g\n", i, j, k,
                     L->minMassQDiQStr[i][j][k]);
      }
    }
  }
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        std::fprintf(f, "Meson,%d,%d,%d,-1,%d\n", i, j, k, L->Meson[i][j][k]);
        std::fprintf(f, "MesonWeight,%d,%d,%d,-1,%.17g\n", i, j, k, L->MesonWeight[i][j][k]);
      }
    }
  }
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 4; ++l) {
          std::fprintf(f, "Baryon,%d,%d,%d,%d,%d\n", i, j, k, l, L->Baryon[i][j][k][l]);
          std::fprintf(f, "BaryonWeight,%d,%d,%d,%d,%.17g\n", i, j, k, l,
                       L->BaryonWeight[i][j][k][l]);
        }
      }
    }
  }
  for (int i = 0; i < 5; ++i) {
    std::fprintf(f, "Qcharge,%d,-1,-1,-1,%d\n", i, L->Qcharge[i]);
    std::fprintf(f, "Prob_QQbar,%d,-1,-1,-1,%.17g\n", i, L->Prob_QQbar[i]);
  }
  std::fclose(f);
  delete L;
}

// ---------------------------------------------------------------------------------------------
// ftf_hadrons.csv - the particle table, which is what FindParticle(code) reads
// ---------------------------------------------------------------------------------------------

void dump_hadrons() {
  G4ParticleTable* pt = G4ParticleTable::GetParticleTable();
  G4SampleResonance BrW;
  FILE* f = std::fopen("ftf_hadrons.csv", "w");
  std::fprintf(f, "pdg,name,subtype,mass,width,charge,baryon,shortlived,minmass,"
                  "nq4,naq4,nq5,naq5,iisospin,iispin,ptype\n");
  const G4int n = (G4int)pt->size();
  for (G4int i = 0; i < n; ++i) {
    const G4ParticleDefinition* d = pt->GetParticle(i);
    if (!d) { continue; }
    // G4SampleResonance::GetMinimumMass dereferences GetDecayTable() with no null check, and
    // a DIQUARK is short-lived with no decay table - so asking it about `anti_ud1` is an
    // access violation. Found by the first run of this dump, which died there. The guard is
    // the reason the column can be -1 for a short-lived particle, and it is also a statement
    // about G4ExcitedStringDecay::FragmentStrings, which calls GetMinimumMass on every
    // short-lived product without that check: it is safe there only because a diquark is
    // never a fragmentation product.
    double minmass = -1.0;
    if (d->IsShortLived() && d->GetDecayTable() != nullptr) {
      minmass = BrW.GetMinimumMass(d);
    }
    // Three columns P11b needs and P11 did not:
    //   `iisospin` is 2I. G4DiffractiveExcitation::ExciteParticipants_doChargeExchange asks
    //     `GetPDGiIsospin() == 3` - "was this hadron a Delta" - in four places.
    //   `iispin` is 2J, and it is a DRAW COUNT: G4Parton's constructor spends one uniform on
    //     the colour and a second on the spin projection ONLY when `GetPDGiSpin() != 0`, so a
    //     spin-0 diquark costs one deviate and a quark costs two. Every SplitUp builds two
    //     partons, so getting this wrong shifts the whole remaining stream of an event.
    //   `ptype` is GetParticleType(), which is what G4Parton's constructor branches on
    //     ("quarks" / "diquarks" / "gluons"); GetParticleSubType() ("quark" / "di_quark") is
    //     the one the rest of FTF uses, and the two are dumped so that nothing has to assume
    //     they agree.
    // None of the three is derivable from the PDG code without a rule this port would be
    // guessing at.
    std::fprintf(f, "%d,%s,%s,%.17g,%.17g,%.17g,%d,%d,%.17g,%d,%d,%d,%d,%d,%d,%s\n",
                 d->GetPDGEncoding(), d->GetParticleName().c_str(),
                 d->GetParticleSubType().c_str(), d->GetPDGMass(), d->GetPDGWidth(),
                 d->GetPDGCharge(), d->GetBaryonNumber(), d->IsShortLived() ? 1 : 0, minmass,
                 d->GetQuarkContent(4), d->GetAntiQuarkContent(4), d->GetQuarkContent(5),
                 d->GetAntiQuarkContent(5), d->GetPDGiIsospin(), d->GetPDGiSpin(),
                 d->GetParticleType().c_str());
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// The parton pairs every string-decay function is driven with
// ---------------------------------------------------------------------------------------------

/// Quarks d u s c b and their antiquarks, and the diquarks the tables can reach. The diquark
/// codes are the (max*1000 + min*100 + spin) form CreatePartonPair builds.
std::vector<int> parton_codes() {
  std::vector<int> v;
  for (int q = 1; q <= 5; ++q) { v.push_back(q); v.push_back(-q); }
  for (int q1 = 1; q1 <= 5; ++q1) {
    for (int q2 = 1; q2 <= q1; ++q2) {
      for (int spin : {1, 3}) {
        if (q1 == q2 && spin == 1) { continue; }  // no spin-0 diquark of identical quarks
        const int c = q1 * 1000 + q2 * 100 + spin;
        v.push_back(c);
        v.push_back(-c);
      }
    }
  }
  return v;
}

/// A G4ExcitedString of two partons carrying light-cone momenta that give the string `mass`.
/// The partons are massless and back to back along z, which is what G4FTFModel hands over
/// after PutOnMassShell and what TransformToAlignedCms produces in any case.
G4ExcitedString* make_string(int left_code, int right_code, double mass, int direction) {
  G4Parton* l = new G4Parton(left_code);
  G4Parton* r = new G4Parton(right_code);
  const double half = 0.5 * mass;
  l->Set4Momentum(G4LorentzVector(0.0, 0.0, half, half));
  r->Set4Momentum(G4LorentzVector(0.0, 0.0, -half, half));
  return new G4ExcitedString(l, r, direction);
}

// ---------------------------------------------------------------------------------------------
// ftf_minmass.csv
// ---------------------------------------------------------------------------------------------

void dump_minmass() {
  LundProbe* L = new LundProbe();
  FILE* f = std::fopen("ftf_minmass.csv", "w");
  std::fprintf(f, "left,right,string_mass,minimal_mass,minimal_mass2,is4q,fragmentable,"
                  "possible_mass,had1,had2\n");
  const std::vector<int> codes = parton_codes();
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (int lc : codes) {
    for (int rc : codes) {
      // SetMinimalStringMass throws on an illegal parton combination (same sub-type with the
      // same sign, or different sub-type with opposite signs), so the oracle only asks about
      // the pairs the model can build - and the port's own predicate is checked against the
      // `legal` column rather than against a reading of the two conditions.
      const bool l_is_di = std::abs(lc) > 1000;
      const bool r_is_di = std::abs(rc) > 1000;
      const bool same_subtype = (l_is_di == r_is_di);
      const bool legal = same_subtype ? (lc * (long long)rc < 0) : (lc * (long long)rc > 0);
      if (!legal) { continue; }
      // G4FragmentingString's constructor stores GetDefinition() without a null check and
      // SetMinimalStringMass dereferences it, so a parton code with no G4DiQuarks /
      // G4Quarks definition is an access violation rather than an answer. Skipped, and the
      // port refuses the same code by name.
      if (L->FindParticle(lc) == nullptr || L->FindParticle(rc) == nullptr) { continue; }
      for (double m : {400.0, 1000.0, 2500.0, 6000.0, 20000.0}) {
        G4ExcitedString* s = make_string(lc, rc, m, 1);
        G4FragmentingString fs(*s);
        L->SetMinimalStringMass(&fs);
        const double mm = L->MinimalStringMass;
        const double mm2 = L->MinimalStringMass2;
        eng.reset(0);
        std::pair<G4ParticleDefinition*, G4ParticleDefinition*> hadrons(nullptr, nullptr);
        const double pm = L->PossibleHadronMass(&fs, 0, &hadrons);
        std::fprintf(f, "%d,%d,%.17g,%.17g,%.17g,%d,%d,%.17g,%d,%d\n", lc, rc, m, mm, mm2,
                     fs.IsAFourQuarkString() ? 1 : 0,
                     (std::abs(mm) < fs.Get4Momentum().mag()) ? 1 : 0, pm,
                     hadrons.first ? hadrons.first->GetPDGEncoding() : 0,
                     hadrons.second ? hadrons.second->GetPDGEncoding() : 0);
        delete s;
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
  delete L;
}

// ---------------------------------------------------------------------------------------------
// ftf_build.csv - G4HadronBuilder under the cycle
// ---------------------------------------------------------------------------------------------

void dump_build() {
  LundProbe* L = new LundProbe();
  G4HadronBuilder* hb = L->hadronizer;
  FILE* f = std::fopen("ftf_build.csv", "w");
  std::fprintf(f, "black,white,phase,build,build_draws,lowspin,lowspin_draws,highspin,"
                  "highspin_draws\n");
  const std::vector<int> codes = parton_codes();
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (int bc : codes) {
    for (int wc : codes) {
      // Meson() and Barion() throw on illegal content: a meson needs |id| <= 5 on both ends,
      // a baryon a diquark on one end and a quark <= 5 on the other. Build() dispatches on
      // the sub-type, so the reachable pairs are quark+quark and diquark+quark - and
      // diquark+diquark, which Build sends to Barion() and which throws. Skipped here and
      // refused by name in the port.
      const bool b_di = std::abs(bc) > 1000;
      const bool w_di = std::abs(wc) > 1000;
      if (b_di && w_di) { continue; }
      G4ParticleDefinition* b = L->FindParticle(bc);
      G4ParticleDefinition* w = L->FindParticle(wc);
      if (!b || !w) { continue; }
      for (int phase = 0; phase < 8; ++phase) {
        eng.reset(phase);
        G4ParticleDefinition* r1 = hb->Build(b, w);
        const int d1 = eng.draws();
        eng.reset(phase);
        G4ParticleDefinition* r2 = hb->BuildLowSpin(b, w);
        const int d2 = eng.draws();
        eng.reset(phase);
        G4ParticleDefinition* r3 = hb->BuildHighSpin(b, w);
        const int d3 = eng.draws();
        std::fprintf(f, "%d,%d,%d,%d,%d,%d,%d,%d,%d\n", bc, wc, phase,
                     r1 ? r1->GetPDGEncoding() : 0, d1, r2 ? r2->GetPDGEncoding() : 0, d2,
                     r3 ? r3->GetPDGEncoding() : 0, d3);
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
  delete L;
}

// ---------------------------------------------------------------------------------------------
// ftf_samplers.csv
// ---------------------------------------------------------------------------------------------

void dump_samplers() {
  LundProbe* L = new LundProbe();
  FILE* f = std::fopen("ftf_samplers.csv", "w");
  std::fprintf(f, "what,arg1,arg2,arg3,arg4,phase,value,draws\n");
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);

  // SampleQuarkFlavor at the three StrangeSuppress values Splitup can set: the constructed
  // (1-0.12)/2, and the values a 1250 MeV and a 20 GeV string produce through the
  // `1 - (Mth/StringMass)^2.5` correction.
  for (double ss : {0.44, (1.0 - 0.12) / 2.0, 0.5, 0.3}) {
    L->SetStrangenessSuppression(ss);
    for (int phase = 0; phase < 8; ++phase) {
      eng.reset(phase);
      const int q = L->SampleQuarkFlavor();
      std::fprintf(f, "SampleQuarkFlavor,%.17g,0,0,0,%d,%d,%d\n", ss, phase, q, eng.draws());
    }
  }
  L->SetStrangenessSuppression((1.0 - 0.12) / 2.0);

  // The HEAVY branch of SampleQuarkFlavor. ProbCB is 2.5e-4 in a real run and the cycle's
  // smallest deviate is 0.05, so the branch is unreachable at the physical value and the first
  // version of this dump never entered it. SetProbCCbar and SetProbBBbar are public and are
  // not locked by the run state, so the branch can be reached by raising them - the same
  // technique dump_precompound.cc uses on SetOPTxs to reach all five inverse cross sections.
  // Three settings: c only, b only, and both, so that the `ksi < ProbCCbar` split inside the
  // branch is decided both ways.
  {
    const double saved_c = L->ProbCCbar;
    const double saved_b = L->ProbBBbar;
    const double heavy[3][2] = {{0.5, 0.0}, {0.0, 0.5}, {0.2, 0.6}};
    for (const auto& hb : heavy) {
      L->SetProbCCbar(hb[0]);
      L->SetProbBBbar(hb[1]);
      for (int phase = 0; phase < 8; ++phase) {
        eng.reset(phase);
        const int q = L->SampleQuarkFlavor();
        std::fprintf(f, "SampleQuarkFlavorHeavy,%.17g,%.17g,0,0,%d,%d,%d\n", hb[0], hb[1],
                     phase, q, eng.draws());
      }
    }
    L->SetProbCCbar(saved_c);
    L->SetProbBBbar(saved_b);
  }

  for (double ptmax : {-1.0, 100.0, 500.0, 2000.0, 1.0e5}) {
    for (int phase = 0; phase < 8; ++phase) {
      eng.reset(phase);
      const G4ThreeVector pt = L->SampleQuarkPt(ptmax);
      std::fprintf(f, "SampleQuarkPt_x,%.17g,0,0,0,%d,%.17g,%d\n", ptmax, phase, pt.x(),
                   eng.draws());
      std::fprintf(f, "SampleQuarkPt_y,%.17g,0,0,0,%d,%.17g,%d\n", ptmax, phase, pt.y(),
                   eng.draws());
    }
  }

  // GetLightConeZ: the quark arm (Alund = 1, Blund = 0.7/GeV^2 with its analytic zOfMaxyf)
  // and the diquark arm (the an = 2.5 + pt^2 power law, with the > 3000 reflection).
  G4ParticleTable* pt = G4ParticleTable::GetParticleTable();
  const int hadrons[] = {211, 111, 321, 2212, 3122, 2224};
  const int partons[] = {1, 2, 3, -1, 2101, 2103, 3201, 3203};
  for (int h : hadrons) {
    const G4ParticleDefinition* hd = pt->FindParticle(h);
    if (!hd) { continue; }
    for (int pc : partons) {
      for (double px : {0.0, 200.0, 800.0}) {
        for (int phase = 0; phase < 8; ++phase) {
          eng.reset(phase);
          const double z = (L->*bridge(TagLcz()))(
              0.05, 0.95, pc, const_cast<G4ParticleDefinition*>(hd), px, 0.0);
          std::fprintf(f, "GetLightConeZ,%d,%d,%.17g,0,%d,%.17g,%d\n", h, pc, px, phase, z,
                       eng.draws());
        }
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
  delete L;
}

// ---------------------------------------------------------------------------------------------
// ftf_fragment.csv and ftf_fragstat.csv
// ---------------------------------------------------------------------------------------------

/// The string cases: a q-qbar string, a q-qq string (both diquark spins), and a qq-qqbar
/// string, each at several masses spanning the three regimes - below the minimal mass (one
/// hadron), just above it (SplitLast only) and far above it (the fragmentation loop runs).
struct StringCase { int left, right; double mass; const char* name; };
/// The first two and the ninth are BELOW the string's minimal mass, so `IsItFragmentable` is
/// false for them and FragmentString takes the ProduceOneHadron branch instead of the loop -
/// the one path the first version of this dump did not reach, found by the port's test having
/// no coverage for it. minMassQQbarStr[0][0] is 339.95 MeV and minMassQDiQStr[0][1][1] is
/// 1144.54, so 300 and 1000 are below and 400 and 2000 above.
///
/// `u-ubar-1.5` and `s-sbar-1.5` are there to measure what docs/RISK.md V85 costs: at 1.5 GeV
/// a quarter of the events are a bare SplitLast, which is the only place the Meson/MesonWeight
/// tables are read, and the d-dbar row of those tables is the broken one.
const StringCase kStrings[] = {
  {1, -1, 300.0, "d-dbar-0.3"},       {1, -1, 400.0, "d-dbar-0.4"},
  {1, -1, 1500.0, "d-dbar-1.5"},      {1, -1, 5000.0, "d-dbar-5"},
  {1, -1, 20000.0, "d-dbar-20"},      {2, -2, 1500.0, "u-ubar-1.5"},
  {2, -2, 5000.0, "u-ubar-5"},        {3, -3, 1500.0, "s-sbar-1.5"},
  {3, -3, 5000.0, "s-sbar-5"},        {2, -1, 5000.0, "u-dbar-5"},
  {1, 2103, 1000.0, "d-ud1-1"},       {1, 2103, 2000.0, "d-ud1-2"},
  {1, 2103, 5000.0, "d-ud1-5"},       {1, 2103, 20000.0, "d-ud1-20"},
  {2, 2101, 5000.0, "u-ud0-5"},       {3, 2103, 5000.0, "s-ud1-5"},
  {2101, -2101, 5000.0, "ud0-ud0bar-5"}, {2103, -2103, 20000.0, "ud1-ud1bar-20"},
  {2101, -2101, 2000.0, "ud0-ud0bar-2"},
};

/// IsItFragmentable, StopFragmenting and Sample4Momentum on their own, which is the point of
/// the PrivateBridge above: the three decisions that make FragmentString's control flow, each
/// with a verdict a test can name. StopFragmenting is a rejection on a Gaussian in
/// (M^2 - Mmin^2), so it is dumped per phase; IsItFragmentable draws nothing.
void dump_decisions() {
  LundProbe* L = new LundProbe();
  FILE* f = std::fopen("ftf_decisions.csv", "w");
  std::fprintf(f, "left,right,mass,minimal_mass,is4q,fragmentable,phase,stop,stop_draws,"
                  "m1,m2,s4m_px,s4m_py,s4m_pz,s4m_e,s4m_apx,s4m_apy,s4m_apz,s4m_ae,"
                  "s4m_draws\n");
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (const StringCase& sc : kStrings) {
    G4ExcitedString* s = make_string(sc.left, sc.right, sc.mass, 1);
    G4FragmentingString fs(*s);
    fs.SetLeftPartonStable();
    L->SetMinimalStringMass(&fs);
    const double mm = L->MinimalStringMass;
    eng.reset(0);
    const bool frag = (L->*bridge(TagFragmentable()))(&fs);
    for (int phase = 0; phase < 8; ++phase) {
      eng.reset(phase);
      const bool stop = (L->*bridge(TagStop()))(&fs);
      const int sd = eng.draws();
      // Two mass pairs: below and above the 930 MeV branch Sample4Momentum tests, so both the
      // single and the DOUBLE application of its `1 - 0.55*((m1+m2)/M)^2` factor are dumped.
      for (int pair = 0; pair < 3; ++pair) {
        const double m1 = (pair == 0) ? 139.57018 : 938.272013;
        const double m2 = (pair == 2) ? 938.272013 : ((pair == 0) ? 139.57018 : 139.57018);
        if (m1 + m2 >= sc.mass) { continue; }
        G4LorentzVector p, ap;
        eng.reset(phase);
        (L->*bridge(TagS4M()))(&p, m1, &ap, m2, sc.mass);
        std::fprintf(f,
                     "%d,%d,%.17g,%.17g,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                     "%.17g,%.17g,%.17g,%.17g,%d\n",
                     sc.left, sc.right, sc.mass, mm, fs.IsAFourQuarkString() ? 1 : 0,
                     frag ? 1 : 0, phase, stop ? 1 : 0, sd, m1, m2, p.px(), p.py(), p.pz(),
                     p.e(), ap.px(), ap.py(), ap.pz(), ap.e(), eng.draws());
      }
    }
    delete s;
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
  delete L;
}

/// The final-state enumeration itself: which (left, right) pairs each of the three
/// last-splitting functions lists, in order, and with what weight. This is what makes the
/// |p|^3 in the diquark-antidiquark weight and the zero c and b entries of Prob_QQbar exact
/// rather than merely consistent with a sampled index.
void dump_last_states() {
  LundProbe* L = new LundProbe();
  FILE* f = std::fopen("ftf_laststates.csv", "w");
  std::fprintf(f, "which,left,right,mass,ok,number_of_fs,index,fs_left,fs_right,fs_weight\n");
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (const StringCase& sc : kStrings) {
    G4ExcitedString* s = make_string(sc.left, sc.right, sc.mass, 1);
    G4FragmentingString* fs = new G4FragmentingString(*s);
    fs->SetLeftPartonStable();  // SplitLast does this before it dispatches
    L->SetMinimalStringMass(fs);
    G4ParticleDefinition* lh = nullptr;
    G4ParticleDefinition* rh = nullptr;
    L->NumberOf_FS = 0;
    for (G4int i = 0; i < 350; ++i) { L->FS_Weight[i] = 0.; }
    eng.reset(0);
    const char* which = nullptr;
    G4bool ok = false;
    if (fs->IsAFourQuarkString()) {
      which = "DiQ-ADiQ-above";
      ok = (L->*bridge(TagDiQADiQAbove()))(fs, lh, rh);
    } else if (fs->DecayIsQuark() && fs->StableIsQuark()) {
      which = "Q-Qbar";
      ok = (L->*bridge(TagQQbarLast()))(fs, lh, rh);
    } else {
      which = "Q-DiQ";
      ok = (L->*bridge(TagQDiQLast()))(fs, lh, rh);
    }
    if (L->NumberOf_FS == 0) {
      std::fprintf(f, "%s,%d,%d,%.17g,%d,%d,-1,0,0,0\n", which, sc.left, sc.right, sc.mass,
                   ok ? 1 : 0, L->NumberOf_FS);
    }
    for (G4int i = 0; i < L->NumberOf_FS; ++i) {
      std::fprintf(f, "%s,%d,%d,%.17g,%d,%d,%d,%d,%d,%.17g\n", which, sc.left, sc.right,
                   sc.mass, ok ? 1 : 0, L->NumberOf_FS, i,
                   L->FS_LeftHadron[i] ? L->FS_LeftHadron[i]->GetPDGEncoding() : 0,
                   L->FS_RightHadron[i] ? L->FS_RightHadron[i]->GetPDGEncoding() : 0,
                   L->FS_Weight[i]);
    }
    delete fs;
    delete s;
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
  delete L;
}

void dump_fragment() {
  LundProbe* L = new LundProbe();
  FILE* f = std::fopen("ftf_fragment.csv", "w");
  std::fprintf(f, "case,left,right,mass,direction,phase,draws,nhadrons,index,pdg,px,py,pz,e,"
                  "formation_time\n");
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (const StringCase& sc : kStrings) {
    for (int direction : {1, -1}) {
      for (int phase = 0; phase < 8; ++phase) {
        G4ExcitedString* s = make_string(sc.left, sc.right, sc.mass, direction);
        eng.reset(phase);
        G4KineticTrackVector* v = L->FragmentString(*s);
        const int draws = eng.draws();
        const int n = v ? (int)v->size() : -1;
        if (!v || v->empty()) {
          std::fprintf(f, "%s,%d,%d,%.17g,%d,%d,%d,%d,-1,0,0,0,0,0,0\n", sc.name, sc.left,
                       sc.right, sc.mass, direction, phase, draws, n);
        } else {
          for (int i = 0; i < (int)v->size(); ++i) {
            const G4LorentzVector p = (*v)[i]->Get4Momentum();
            std::fprintf(f, "%s,%d,%d,%.17g,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                         sc.name, sc.left, sc.right, sc.mass, direction, phase, draws, n, i,
                         (*v)[i]->GetDefinition()->GetPDGEncoding(), p.px(), p.py(), p.pz(),
                         p.e(), (*v)[i]->GetFormationTime());
          }
        }
        if (v) {
          for (auto* t : *v) { delete t; }
          delete v;
        }
        delete s;
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
  delete L;
}

void dump_fragstat() {
  LundProbe* L = new LundProbe();
  CLHEP::HepJamesRandom eng(20260911);
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);

  FILE* f = std::fopen("ftf_fragstat.csv", "w");
  std::fprintf(f, "case,mass,n_events,pdg,count,sum_e,sum_pz,sum_pt2\n");
  FILE* g = std::fopen("ftf_fragstat_mult.csv", "w");
  std::fprintf(g, "case,mass,n_events,multiplicity,count\n");
  // 20,000 events a case, and the number is written into every row so that the test runs the
  // port at whatever the oracle ran at rather than at a constant of its own.
  //
  // WHAT 20,000 COSTS, MEASURED. At 20,000 the port's mean multiplicity came out ABOVE the
  // oracle's in 12 of the 15 fragmenting cases, combining to +3.3 sigma, and it stayed above
  // for four different port seeds (+3.3, +1.1, +2.7, +1.8) - which looks like a real excess
  // and is not one. The four port samples share one oracle sample, so a single low fluctuation
  // of THAT sample biases all four comparisons the same way. Re-running this block at
  // N = 200,000, against the port at the same 200,000, moved the combination to -1.8 with 8 of
  // 15 cases negative and every case within 0.16%. The 20,000-event oracle is itself the
  // limit, not the port. Raising N here to 200,000 takes this dump from about one minute to
  // seven, which is why it is not the committed value; the test's tolerance of 5 sigma per
  // species is what absorbs the difference.
  const int N = 20000;
  for (const StringCase& sc : kStrings) {
    std::map<int, long long> count;
    std::map<int, double> sum_e, sum_pz, sum_pt2;
    std::map<int, long long> mult;
    for (int ev = 0; ev < N; ++ev) {
      G4ExcitedString* s = make_string(sc.left, sc.right, sc.mass, 1);
      G4KineticTrackVector* v = L->FragmentString(*s);
      const int n = v ? (int)v->size() : 0;
      mult[n]++;
      if (v) {
        for (auto* t : *v) {
          const int pdg = t->GetDefinition()->GetPDGEncoding();
          const G4LorentzVector p = t->Get4Momentum();
          count[pdg]++;
          sum_e[pdg] += p.e();
          sum_pz[pdg] += p.pz();
          sum_pt2[pdg] += p.px() * p.px() + p.py() * p.py();
          delete t;
        }
        delete v;
      }
      delete s;
    }
    for (const auto& kv : count) {
      std::fprintf(f, "%s,%.17g,%d,%d,%lld,%.17g,%.17g,%.17g\n", sc.name, sc.mass, N,
                   kv.first, kv.second, sum_e[kv.first], sum_pz[kv.first],
                   sum_pt2[kv.first]);
    }
    for (const auto& kv : mult) {
      std::fprintf(g, "%s,%.17g,%d,%d,%lld\n", sc.name, sc.mass, N, kv.first, kv.second);
    }
  }
  std::fclose(f);
  std::fclose(g);
  CLHEP::HepRandom::setTheEngine(saved);
  delete L;
}

// ---------------------------------------------------------------------------------------------
// ftf_resonance.csv - G4SampleResonance::SampleMass, the Breit-Wigner every short-lived product
// of the fragmentation has its mass redrawn from
// ---------------------------------------------------------------------------------------------

void dump_resonance() {
  FILE* f = std::fopen("ftf_resonance.csv", "w");
  std::fprintf(f, "pdg,pole,gamma,min,max,phase,mass,draws\n");
  G4SampleResonance BrW;
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);

  // The arms first, by hand: zero width (no draw at all), minMass > maxMass (A.R.'s 2017
  // protection, which replaces the minimum with the maximum rather than throwing), and a
  // pole outside [min, max] on both sides so that the zero-width max/min clamp is decided
  // both ways.
  const double kArms[][4] = {
    {770.0, 0.0, 300.0, 1000.0},    // zero width, pole inside
    {770.0, 0.0, 900.0, 1000.0},    // zero width, pole below the minimum
    {770.0, 0.0, 300.0, 500.0},     // zero width, pole above the maximum
    {770.0, 150.0, 1200.0, 900.0},  // minMass > maxMass
    {770.0, 150.0, 290.0, 1520.0},  // the ordinary rho
    {1232.0, 117.0, 1088.0, 1817.0},// the Delta
    {892.0, 51.0, 644.0, 1147.0},   // the K*
  };
  for (const auto& a : kArms) {
    for (int phase = 0; phase < 8; ++phase) {
      eng.reset(phase);
      const double m = BrW.SampleMass(a[0], a[1], a[2], a[3]);
      std::fprintf(f, "0,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%d\n", a[0], a[1], a[2], a[3], phase,
                   m, eng.draws());
    }
  }

  // Then every short-lived particle the fragmentation can produce, with the arguments
  // FragmentStrings actually passes: (PDGMass, PDGWidth, GetMinimumMass + 10 MeV,
  // PDGMass + 5*PDGWidth). This is the row that makes data/ftf_hadrons.hh's `minmass` column
  // load-bearing rather than decorative.
  G4ParticleTable* pt = G4ParticleTable::GetParticleTable();
  const G4int n = (G4int)pt->size();
  for (G4int i = 0; i < n; ++i) {
    const G4ParticleDefinition* d = pt->GetParticle(i);
    if (!d || !d->IsShortLived() || d->GetDecayTable() == nullptr) { continue; }
    const G4int code = d->GetPDGEncoding();
    if (code == 0 || code >= 1000000000 || code <= -1000000000) { continue; }
    const double pole = d->GetPDGMass();
    const double gamma = d->GetPDGWidth();
    const double lo = BrW.GetMinimumMass(d) + 10.0 * MeV;
    const double hi = pole + 5.0 * gamma;
    for (int phase = 0; phase < 8; ++phase) {
      eng.reset(phase);
      const double m = BrW.SampleMass(pole, gamma, lo, hi);
      std::fprintf(f, "%d,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%d\n", code, pole, gamma, lo, hi,
                   phase, m, eng.draws());
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// ftf_corrector.csv - EnergyAndMomentumCorrector on its own
//
// It draws no random number, so this is an exact oracle with no phase axis, and it is dumped
// SEPARATELY from FragmentStrings for the reason docs/RISK.md V52 gives: the corrector is a
// 500-iteration fixed point, and a transcription that converges to the same answer by a
// different route is right, while one that returns the same answer because it never ran is not.
// The four early returns are enumerated by hand, because each of them is a different meaning of
// "false" - and one of them, the empty list, returns TRUE.
// ---------------------------------------------------------------------------------------------

struct CorrectorHadron { int pdg; double px, py, pz; };
struct CorrectorCase {
  const char* name;
  double cms_px, cms_py, cms_pz, cms_e;
  int n;
  CorrectorHadron h[6];
};

const CorrectorCase kCorrectorCases[] = {
  {"empty", 0, 0, 5000, 6000, 0, {}},
  {"single", 0, 0, 5000, 6000, 1, {{211, 100, 0, 2000}}},
  {"too-heavy", 0, 0, 0, 500, 2, {{2212, 10, 0, 100}, {-2212, -10, 0, -100}}},
  {"two-pions", 0, 0, 0, 2000, 2, {{211, 100, 50, 300}, {-211, -80, -40, -280}}},
  {"boosted", 0, 0, 5000, 6000, 3,
   {{211, 100, 50, 1500}, {-211, -80, -40, 1400}, {111, 20, 10, 1600}}},
  {"off-axis", 300, -200, 5000, 6200, 4,
   {{2212, 150, -100, 1200}, {-211, -80, -40, 900}, {211, 60, 30, 1100}, {111, 10, 5, 700}}},
  {"far-off", 0, 0, 0, 8000, 5,
   {{2212, 900, 0, 900}, {-2212, -900, 0, -900}, {211, 200, 100, 50},
    {-211, -100, -50, -100}, {111, 30, 20, 10}}},
  // Two back-to-back momenta, which is as close as a hadron list gets to the corrector's
  // `SumMass = SumMom.m2(); if (SumMass < 0) return FALSE` guard - and does not reach it. The
  // guard is DEAD: every term of the sum is timelike and future-pointing, and that survives
  // addition, so m2 cannot come out negative. Established by removing the guard from the port
  // and finding no oracle row that moves, which is the sentinel probe docs/RISK.md V52 asks
  // for rather than an argument about it.
  {"back-to-back", 0, 0, 0, 20000, 2, {{111, 3000, 0, 0}, {111, -3000, 0, 0}}},
};

struct TagCorrector {
  using type = G4bool (G4ExcitedStringDecay::*)(G4KineticTrackVector*, G4LorentzVector&);
  friend type bridge(TagCorrector);
};
template struct PrivateBridge<TagCorrector, &G4ExcitedStringDecay::EnergyAndMomentumCorrector>;

void dump_corrector() {
  FILE* f = std::fopen("ftf_corrector.csv", "w");
  // The INPUT momenta are columns of the same row, so that the test builds the corrector's
  // input from the oracle rather than from a second copy of the table above. The corrector
  // never changes the length or the order of the list, so out[i] is in[i] corrected.
  std::fprintf(f, "case,cms_px,cms_py,cms_pz,cms_e,n,success,index,pdg,in_px,in_py,in_pz,"
                  "px,py,pz,e\n");
  G4ExcitedStringDecay* dec = new G4ExcitedStringDecay(new G4LundStringFragmentation());
  for (const CorrectorCase& c : kCorrectorCases) {
    G4KineticTrackVector* v = new G4KineticTrackVector;
    for (int i = 0; i < c.n; ++i) {
      const G4ParticleDefinition* d =
          G4ParticleTable::GetParticleTable()->FindParticle(c.h[i].pdg);
      const double m = d->GetPDGMass();
      const G4ThreeVector p3(c.h[i].px, c.h[i].py, c.h[i].pz);
      const G4LorentzVector mom(p3, std::sqrt(p3.mag2() + m * m));
      v->push_back(new G4KineticTrack(const_cast<G4ParticleDefinition*>(d), 0.0,
                                      G4ThreeVector(0, 0, 0), mom));
    }
    G4LorentzVector cms(c.cms_px, c.cms_py, c.cms_pz, c.cms_e);
    const G4bool ok = (dec->*bridge(TagCorrector()))(v, cms);
    if (v->empty()) {
      std::fprintf(f, "%s,%.17g,%.17g,%.17g,%.17g,%d,%d,-1,0,0,0,0,0,0,0,0\n", c.name,
                   c.cms_px, c.cms_py, c.cms_pz, c.cms_e, c.n, ok ? 1 : 0);
    }
    for (int i = 0; i < (int)v->size(); ++i) {
      const G4LorentzVector p = (*v)[i]->Get4Momentum();
      std::fprintf(f,
                   "%s,%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%d,%.17g,%.17g,%.17g,"
                   "%.17g,%.17g,%.17g,%.17g\n",
                   c.name, c.cms_px, c.cms_py, c.cms_pz, c.cms_e, c.n, ok ? 1 : 0, i,
                   (*v)[i]->GetDefinition()->GetPDGEncoding(), c.h[i].px, c.h[i].py, c.h[i].pz,
                   p.px(), p.py(), p.pz(), p.e());
    }
    for (auto* t : *v) { delete t; }
    delete v;
  }
  delete dec;
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// ftf_strings.csv / ftf_stringstat*.csv - G4ExcitedStringDecay::FragmentStrings
//
// The whole layer: several strings at once, in the c.m.s. of their sum, with every short-lived
// product's mass redrawn from a Breit-Wigner and the energy-momentum correction loop putting
// the sum back. `track` is a NOT-EXCITED string - G4ExcitedString built on a G4KineticTrack
// rather than on two partons - which FragmentStrings copies through the G4KineticTrack
// constructor, kaon0 coin toss and all.
// ---------------------------------------------------------------------------------------------

struct StringSpec { int left, right; double mass; int direction; double bz, bx; int track_pdg; };
struct StringVecCase { const char* name; int n; StringSpec s[4]; };

const StringVecCase kStringVecs[] = {
  {"two-strings", 2, {{1, -1, 5000, +1, 0.30, 0.00, 0}, {2, 2103, 4000, -1, -0.20, 0.05, 0}}},
  {"three-strings", 3, {{1, -1, 5000, +1, 0.30, 0.00, 0}, {2, 2103, 4000, -1, -0.20, 0.05, 0},
                        {3, -3, 3000, +1, 0.10, -0.05, 0}}},
  {"with-track", 3, {{1, -1, 5000, +1, 0.30, 0.00, 0}, {0, 0, 0, +1, 0.00, 0.00, 2212},
                     {2, 2103, 4000, -1, -0.20, 0.05, 0}}},
  // A K0S and not a kaon0, deliberately. G4KineticTrack's constructor substitutes K0S or K0L
  // for a kaon0 ON A COIN TOSS, so a track built here with code 311 would be substituted while
  // THIS vector is being built - out of the counted region, from whatever the engine's state
  // was - and FragmentStrings' own copy of it would then find a K0S and spend no deviate. The
  // asymmetry is not a property of the model: any G4KineticTrack that exists has already been
  // through that constructor, so a not-excited string can never carry a kaon0 in a real run.
  // The port transcribes the substitution in its copy anyway, because FragmentStrings calls
  // the constructor and a caller that hands it a kaon0 must get Geant4's answer.
  {"with-k0s-track", 2, {{1, -1, 5000, +1, 0.30, 0.00, 0}, {0, 0, 0, +1, 0.20, 0.02, 310}}},
  {"one-string", 1, {{1, 2103, 6000, +1, 0.25, 0.03, 0}}},
  {"light-pair", 2, {{1, -1, 1500, +1, 0.10, 0.00, 0}, {2, 2101, 2000, -1, -0.10, 0.02, 0}}},
  {"below-threshold", 2, {{1, -1, 300, +1, 0.05, 0.00, 0}, {1, 2103, 1000, -1, -0.05, 0.01, 0}}},
  {"heavy-pair", 2, {{2101, -2101, 8000, +1, 0.20, 0.00, 0},
                     {2103, -2103, 12000, -1, -0.15, 0.04, 0}}},
};

/// One case's strings, freshly built. FragmentStrings MUTATES them - it transforms the partons
/// into the c.m.s. and, only on failure, back - so every call needs its own copy.
G4ExcitedStringVector* make_string_vector(const StringVecCase& c) {
  G4ExcitedStringVector* v = new G4ExcitedStringVector;
  for (int i = 0; i < c.n; ++i) {
    const StringSpec& s = c.s[i];
    if (s.track_pdg != 0) {
      const G4ParticleDefinition* d =
          G4ParticleTable::GetParticleTable()->FindParticle(s.track_pdg);
      const double m = d->GetPDGMass();
      G4LorentzVector mom(0.0, 0.0, 0.0, m);
      mom.boost(G4ThreeVector(s.bx, 0.0, s.bz));
      G4KineticTrack* kt = new G4KineticTrack(const_cast<G4ParticleDefinition*>(d), 0.0,
                                              G4ThreeVector(0, 0, 0), mom);
      v->push_back(new G4ExcitedString(kt));
      continue;
    }
    G4Parton* l = new G4Parton(s.left);
    G4Parton* r = new G4Parton(s.right);
    const double half = 0.5 * s.mass;
    G4LorentzVector pl(0.0, 0.0, half, half);
    G4LorentzVector pr(0.0, 0.0, -half, half);
    pl.boost(G4ThreeVector(s.bx, 0.0, s.bz));
    pr.boost(G4ThreeVector(s.bx, 0.0, s.bz));
    l->Set4Momentum(pl);
    r->Set4Momentum(pr);
    v->push_back(new G4ExcitedString(l, r, s.direction));
  }
  return v;
}

void delete_string_vector(G4ExcitedStringVector* v) {
  for (auto* s : *v) { delete s; }
  delete v;
}

/// The cases themselves, so that the port builds its input from the oracle and not from a
/// second copy of the table that can drift from this one.
void dump_string_specs() {
  FILE* f = std::fopen("ftf_stringspec.csv", "w");
  std::fprintf(f, "case,n,index,left,right,mass,direction,bz,bx,track_pdg,"
                  "lpx,lpy,lpz,le,rpx,rpy,rpz,re\n");
  for (const StringVecCase& c : kStringVecs) {
    G4ExcitedStringVector* v = make_string_vector(c);
    for (int i = 0; i < c.n; ++i) {
      const StringSpec& s = c.s[i];
      G4LorentzVector pl(0, 0, 0, 0), pr(0, 0, 0, 0);
      int track_pdg = s.track_pdg;
      if (s.track_pdg != 0) {
        pl = (*v)[i]->GetKineticTrack()->Get4Momentum();
        // The code the track ACTUALLY carries, which is not necessarily the one asked for -
        // see the K0S comment on the case table. The port builds its input from this column.
        track_pdg = (*v)[i]->GetKineticTrack()->GetDefinition()->GetPDGEncoding();
      } else {
        pl = (*v)[i]->GetLeftParton()->Get4Momentum();
        pr = (*v)[i]->GetRightParton()->Get4Momentum();
      }
      std::fprintf(f,
                   "%s,%d,%d,%d,%d,%.17g,%d,%.17g,%.17g,%d,"
                   "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                   c.name, c.n, i, s.left, s.right, s.mass, s.direction, s.bz, s.bx,
                   track_pdg, pl.px(), pl.py(), pl.pz(), pl.e(), pr.px(), pr.py(), pr.pz(),
                   pr.e());
    }
    delete_string_vector(v);
  }
  std::fclose(f);
}

void dump_strings() {
  FILE* f = std::fopen("ftf_strings.csv", "w");
  std::fprintf(f, "case,phase,draws,nhadrons,index,pdg,px,py,pz,e,mass,formation_time\n");
  G4ExcitedStringDecay* dec = new G4ExcitedStringDecay(new G4LundStringFragmentation());
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (const StringVecCase& c : kStringVecs) {
    for (int phase = 0; phase < 8; ++phase) {
      G4ExcitedStringVector* v = make_string_vector(c);
      eng.reset(phase);
      G4KineticTrackVector* out = dec->FragmentStrings(v);
      const int draws = eng.draws();
      const int n = (out == nullptr) ? -1 : (int)out->size();
      if (out == nullptr || out->empty()) {
        std::fprintf(f, "%s,%d,%d,%d,-1,0,0,0,0,0,0,0\n", c.name, phase, draws, n);
      } else {
        for (int i = 0; i < (int)out->size(); ++i) {
          const G4LorentzVector p = (*out)[i]->Get4Momentum();
          std::fprintf(f, "%s,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", c.name,
                       phase, draws, n, i, (*out)[i]->GetDefinition()->GetPDGEncoding(), p.px(),
                       p.py(), p.pz(), p.e(), p.mag(), (*out)[i]->GetFormationTime());
        }
      }
      if (out != nullptr) {
        for (auto* t : *out) { delete t; }
        delete out;
      }
      delete_string_vector(v);
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
  delete dec;
}

/// The statistical half of the same cases: species, multiplicity, and the ENERGY BALANCE, which
/// is what the corrector exists for. `nhadrons = 0` counts the events FragmentStrings gave up
/// on after 100 attempts - it returns a null vector then, and that is a physical outcome rather
/// than an error, so it is a bin like any other.
void dump_stringstat() {
  G4ExcitedStringDecay* dec = new G4ExcitedStringDecay(new G4LundStringFragmentation());
  CLHEP::HepJamesRandom eng(20260912);
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);

  FILE* f = std::fopen("ftf_stringstat.csv", "w");
  std::fprintf(f, "case,n_events,pdg,count,sum_e,sum_pz,sum_pt2\n");
  FILE* g = std::fopen("ftf_stringstat_mult.csv", "w");
  std::fprintf(g, "case,n_events,multiplicity,count\n");
  FILE* h = std::fopen("ftf_stringstat_balance.csv", "w");
  std::fprintf(h, "case,n_events,n_ok,rel_e_bin,count\n");
  const int N = 20000;
  for (const StringVecCase& c : kStringVecs) {
    std::map<int, long long> count, mult, balance;
    std::map<int, double> sum_e, sum_pz, sum_pt2;
    long long n_ok = 0;
    for (int ev = 0; ev < N; ++ev) {
      G4ExcitedStringVector* v = make_string_vector(c);
      G4LorentzVector total(0, 0, 0, 0);
      for (auto* s : *v) { total += s->Get4Momentum(); }
      G4KineticTrackVector* out = dec->FragmentStrings(v);
      const int n = (out == nullptr) ? 0 : (int)out->size();
      mult[n]++;
      if (out != nullptr) {
        G4LorentzVector sum(0, 0, 0, 0);
        for (auto* t : *out) {
          const int pdg = t->GetDefinition()->GetPDGEncoding();
          const G4LorentzVector p = t->Get4Momentum();
          count[pdg]++;
          sum_e[pdg] += p.e();
          sum_pz[pdg] += p.pz();
          sum_pt2[pdg] += p.px() * p.px() + p.py() * p.py();
          sum += p;
          delete t;
        }
        delete out;
        if (n > 0) {
          ++n_ok;
          // log10 of the relative energy error, floored - the corrector's whole job in one
          // number. Geant4's own threshold for running it is 1e-6 and its convergence limit is
          // 1e-5, so a port that never ran it would sit in the -2 and -3 bins.
          const double rel = std::fabs((sum.e() - total.e()) / total.e());
          int bin = -12;
          if (rel > 0.0) { bin = (int)std::floor(std::log10(rel)); }
          if (bin < -12) { bin = -12; }
          if (bin > 1) { bin = 1; }
          balance[bin]++;
        }
      }
      delete_string_vector(v);
    }
    for (const auto& kv : count) {
      std::fprintf(f, "%s,%d,%d,%lld,%.17g,%.17g,%.17g\n", c.name, N, kv.first, kv.second,
                   sum_e[kv.first], sum_pz[kv.first], sum_pt2[kv.first]);
    }
    for (const auto& kv : mult) {
      std::fprintf(g, "%s,%d,%d,%lld\n", c.name, N, kv.first, kv.second);
    }
    for (const auto& kv : balance) {
      std::fprintf(h, "%s,%d,%lld,%d,%lld\n", c.name, N, n_ok, kv.first, kv.second);
    }
  }
  std::fclose(f);
  std::fclose(g);
  std::fclose(h);
  CLHEP::HepRandom::setTheEngine(saved);
  delete dec;
}

// =============================================================================================
// P11b - the model above the fragmentation
//
// Five exact tables under the cycle engine and two statistical ones under a fixed seed. The
// split is not arbitrary: everything G4FTFModel does BEFORE it needs a sampled nucleus is a
// deterministic function of (inputs, phase) and is compared exactly, and everything that starts
// from a G4Fancy3DNucleus is compared statistically, because the nucleus is sampled from a
// rejection loop whose stream no two engines share.
//
// The one exception is `ftf_getlist.csv`, which is exact BECAUSE the nucleus is replayed:
// `ftf_nucleus.csv` dumps a configuration nucleon by nucleon and the port rebuilds exactly it
// before calling its own GetList. That is P9's pattern (docs/PORTED.md 2.1.10) and it is what
// makes the impact-parameter sampler and the participant list checkable at all.
// =============================================================================================

const G4ParticleDefinition* def_of(int pdg) {
  return G4ParticleTable::GetParticleTable()->FindParticle(pdg);
}

/// `Init` and `GetStrings` are PROTECTED in G4FTFModel - only G4VPartonStringModel::Scatter is
/// supposed to call them - so the string-level table is dumped through a derived class, exactly
/// as LundProbe reaches G4VLongitudinalStringDecay's protected state. Nothing in the install is
/// modified.
class FtfModelProbe : public G4FTFModel {
 public:
  explicit FtfModelProbe(const G4String& n) : G4FTFModel(n) {}
  using G4FTFModel::GetStrings;
  using G4FTFModel::Init;
};

/// A splitable hadron standing in for the projectile, with a chosen four-momentum.
G4DiffractiveSplitableHadron* make_projectile_splitable(int pdg, double px, double py,
                                                        double pz) {
  const G4ParticleDefinition* d = def_of(pdg);
  G4ReactionProduct rp(d);
  rp.SetMomentum(px, py, pz);
  rp.SetTotalEnergy(std::sqrt(px * px + py * py + pz * pz + d->GetPDGMass() * d->GetPDGMass()));
  return new G4DiffractiveSplitableHadron(rp);
}

/// A splitable hadron standing in for a target nucleon, built through the G4Nucleon
/// constructor - which is the one G4FTFParticipants uses and the one that copies the POSITION.
/// The energy is set OFF the mass shell, downwards, exactly as G4Fancy3DNucleus leaves it
/// (bic/nucleus/nucleus_model.cuh's second "not guaranteed"), because that is what the
/// excitation's `toBePutOnMassShell` block has to repair.
G4DiffractiveSplitableHadron* make_target_splitable(int pdg, double px, double py, double pz,
                                                    double e, double x, double y, double z) {
  G4Nucleon n;
  if (pdg == 2212) {
    n.SetParticleType(G4Proton::Proton());
  } else {
    n.SetParticleType(G4Neutron::Neutron());
  }
  G4LorentzVector p(px, py, pz, e);
  n.SetMomentum(p);
  n.SetPosition(G4ThreeVector(x, y, z));
  G4DiffractiveSplitableHadron* h = new G4DiffractiveSplitableHadron(n);
  // `G4Nucleon::SetParticleType` accepts only proton, neutron, lambda and their anti-species,
  // so a DELTA target - which a quark exchange produces and G4FTFAnnihilation's weight table
  // has arms for - cannot be built through the nucleon constructor. Setting the definition
  // afterwards is what G4DiffractiveExcitation itself does when it turns a nucleon into a
  // Delta. Without this the `pbar_d_1G` row asked Geant4 about a NEUTRON and the port about a
  // Delta+, and the disagreement looked like a port bug.
  if (pdg != 2212 && pdg != 2112) {
    h->SetDefinition(G4ParticleTable::GetParticleTable()->FindParticle(pdg));
  }
  return h;
}

// ---------------------------------------------------------------------------------------------
// ftf_splitup.csv - G4DiffractiveSplitableHadron::SplitUp and the two GetNextParton calls
// ---------------------------------------------------------------------------------------------

/// Every hadron code FTF can be asked to split: the QBBC beams, the nucleons and their Delta
/// isobars, the neutral mesons whose pi0/eta/eta' arm spends an extra deviate, the hyperons
/// whose baryon arm has the SuppresUUDDSS = 1 branch (Omega-, Delta++ and Delta-), and the
/// charmed and bottom hadrons whose `absPDGcode > 4000` arm is a different formula altogether.
const int kSplitUpCodes[] = {2212, 2112, -2212, -2112, 211,  -211, 321,  -321, 311,  -311,
                             111,  221,  331,    113,   223,  333,  2214, 2114, 1114, 2224,
                             3122, 3222, 3112,   3212,  3322, 3312, 3334, 411,  421,  431,
                             511,  521,  4122,   5122,  4232, 130,  310};

void dump_splitup() {
  FILE* f = std::fopen("ftf_splitup.csv", "w");
  std::fprintf(f, "pdg,phase,parton0,parton1,draws\n");
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (int code : kSplitUpCodes) {
    const G4ParticleDefinition* d = def_of(code);
    if (d == nullptr) { continue; }
    for (int ph = 0; ph < 8; ++ph) {
      eng.reset(ph);
      G4DiffractiveSplitableHadron* h = make_projectile_splitable(code, 0.0, 0.0, 5000.0);
      h->SplitUp();
      G4Parton* p0 = h->GetNextParton();
      G4Parton* p1 = h->GetNextParton();
      std::fprintf(f, "%d,%d,%d,%d,%d\n", code, ph, p0 ? p0->GetPDGcode() : 0,
                   p1 ? p1->GetPDGcode() : 0, eng.draws());
      delete h;
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// The collision cases the excitation and the elastic channel are driven with
// ---------------------------------------------------------------------------------------------

struct CollisionCase {
  const char* name;
  int proj_pdg;
  int targ_pdg;
  double plab;        ///< projectile pz, MeV/c, in the frame the two hadrons are given in
  double targ_px, targ_py, targ_pz, targ_e;
  int targ_a, targ_z;  ///< the nucleus G4FTFParameters is initialised for
  int proj_status, targ_status;
  int proj_ncol, targ_ncol;
};

/// Twelve collisions covering the four arms of ExciteParticipants and both signs of the
/// projectile. The target four-momentum is deliberately off shell downwards (E below
/// sqrt(p^2+m^2)) for half of them, because that is the state a G4Fancy3DNucleus nucleon
/// arrives in and the `toBePutOnMassShell` block is what repairs it; the other half are on
/// shell, which is what PutOnMassShell leaves behind.
const CollisionCase kCollisions[] = {
    {"p_C_4GeV",     2212,  2212, 4700.0,  0.0,   0.0,  0.0,  920.0, 12,  6, 1, 1, 0, 0},
    {"p_C_4GeV_n",   2212,  2112, 4700.0,  30.0, -20.0, 15.0, 930.0, 12,  6, 1, 1, 0, 0},
    {"p_Pb_10GeV",   2212,  2212, 10800.0, 60.0,  40.0, -25.0, 900.0, 207, 82, 1, 1, 0, 0},
    {"n_O_10GeV",    2112,  2212, 10800.0, 0.0,   0.0,  0.0,  938.272013, 16, 8, 1, 1, 0, 0},
    {"pip_C_4GeV",   211,   2212, 4130.0,  0.0,   0.0,  0.0,  938.272013, 12, 6, 1, 1, 0, 0},
    {"pim_Fe_50GeV", -211,  2112, 50000.0, -10.0, 25.0, 5.0,  925.0, 56, 26, 1, 1, 0, 0},
    {"kp_C_10GeV",   321,   2212, 10500.0, 0.0,   0.0,  0.0,  938.272013, 12, 6, 1, 1, 0, 0},
    {"km_Al_50GeV",  -321,  2112, 50000.0, 12.0, -8.0,  30.0, 935.0, 27, 13, 1, 1, 0, 0},
    {"pbar_C_10GeV", -2212, 2212, 10800.0, 0.0,   0.0,  0.0,  938.272013, 12, 6, 1, 1, 0, 0},
    {"p_C_50GeV",    2212,  2112, 50000.0, 0.0,   0.0,  0.0,  939.56536, 12, 6, 1, 1, 0, 0},
    {"p_C_2ndcol",   2212,  2212, 4700.0,  0.0,   0.0,  0.0,  938.272013, 12, 6, 0, 0, 1, 1},
    {"delta_C_4GeV", 2214,  2112, 4700.0,  0.0,   0.0,  0.0,  939.56536, 12, 6, 2, 2, 1, 0},
};

G4FTFParameters* params_for(const CollisionCase& c) {
  G4FTFParameters* p = new G4FTFParameters();
  const G4ParticleDefinition* d = def_of(c.proj_pdg);
  p->InitForInteraction(d, c.targ_a, c.targ_z, c.plab);
  return p;
}

// ---------------------------------------------------------------------------------------------
// ftf_hnelastic.csv - G4ElasticHNScattering::ElasticScattering
// ---------------------------------------------------------------------------------------------

void dump_hnelastic() {
  FILE* f = std::fopen("ftf_hnelastic.csv", "w");
  std::fprintf(f, "case,phase,result,ppx,ppy,ppz,pe,tpx,tpy,tpz,te,pncol,tncol,ptime,"
                  "px_pos,py_pos,pz_pos,draws\n");
  G4ElasticHNScattering el;
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (const CollisionCase& c : kCollisions) {
    G4FTFParameters* par = params_for(c);
    for (int ph = 0; ph < 8; ++ph) {
      eng.reset(ph);
      const G4ParticleDefinition* pd = def_of(c.proj_pdg);
      G4DiffractiveSplitableHadron* pr =
          make_projectile_splitable(c.proj_pdg, 0.0, 0.0, c.plab);
      (void)pd;
      G4DiffractiveSplitableHadron* tr = make_target_splitable(
          c.targ_pdg, c.targ_px, c.targ_py, c.targ_pz, c.targ_e, 1.0 * fermi, -2.0 * fermi,
          0.5 * fermi);
      pr->SetStatus(c.proj_status);
      tr->SetStatus(c.targ_status);
      pr->SetCollisionCount(c.proj_ncol);
      tr->SetCollisionCount(c.targ_ncol);
      tr->SetTimeOfCreation(3.25);
      const G4bool r = el.ElasticScattering(pr, tr, par);
      const G4LorentzVector pp = pr->Get4Momentum();
      const G4LorentzVector tp = tr->Get4Momentum();
      const G4ThreeVector ps = pr->GetPosition();
      std::fprintf(f,
                   "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%.17g,"
                   "%.17g,%.17g,%.17g,%d\n",
                   c.name, ph, r ? 1 : 0, pp.px(), pp.py(), pp.pz(), pp.e(), tp.px(), tp.py(),
                   tp.pz(), tp.e(), pr->GetSoftCollisionCount(), tr->GetSoftCollisionCount(),
                   pr->GetTimeOfCreation(), ps.x(), ps.y(), ps.z(), eng.draws());
      delete pr;
      delete tr;
    }
    delete par;
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// ftf_excite.csv - G4DiffractiveExcitation::ExciteParticipants
// ---------------------------------------------------------------------------------------------

void dump_excite() {
  FILE* f = std::fopen("ftf_excite.csv", "w");
  std::fprintf(f, "case,phase,result,ppdg,tpdg,ppx,ppy,ppz,pe,tpx,tpy,tpz,te,"
                  "pstatus,tstatus,pncol,tncol,draws\n");
  G4DiffractiveExcitation exc;
  G4ElasticHNScattering el;
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (const CollisionCase& c : kCollisions) {
    G4FTFParameters* par = params_for(c);
    for (int ph = 0; ph < 8; ++ph) {
      eng.reset(ph);
      G4DiffractiveSplitableHadron* pr =
          make_projectile_splitable(c.proj_pdg, 0.0, 0.0, c.plab);
      G4DiffractiveSplitableHadron* tr = make_target_splitable(
          c.targ_pdg, c.targ_px, c.targ_py, c.targ_pz, c.targ_e, 1.0 * fermi, -2.0 * fermi,
          0.5 * fermi);
      pr->SetStatus(c.proj_status);
      tr->SetStatus(c.targ_status);
      pr->SetCollisionCount(c.proj_ncol);
      tr->SetCollisionCount(c.targ_ncol);
      tr->SetTimeOfCreation(3.25);
      const G4bool r = exc.ExciteParticipants(pr, tr, par, &el);
      const G4LorentzVector pp = pr->Get4Momentum();
      const G4LorentzVector tp = tr->Get4Momentum();
      std::fprintf(f,
                   "%s,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                   "%d,%d,%d,%d,%d\n",
                   c.name, ph, r ? 1 : 0, pr->GetDefinition()->GetPDGEncoding(),
                   tr->GetDefinition()->GetPDGEncoding(), pp.px(), pp.py(), pp.pz(), pp.e(),
                   tp.px(), tp.py(), tp.pz(), tp.e(), pr->GetStatus(), tr->GetStatus(),
                   pr->GetSoftCollisionCount(), tr->GetSoftCollisionCount(), eng.draws());
      delete pr;
      delete tr;
    }
    delete par;
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// ftf_annih.csv - G4FTFAnnihilation::Annihilate, every channel
// ---------------------------------------------------------------------------------------------

struct AnnihCase {
  const char* name;
  int proj_pdg;
  int targ_pdg;
  double plab;        ///< anti-baryon pz, MeV/c
  double targ_px, targ_py, targ_pz, targ_e;
  int targ_a, targ_z;
};

/// Nine collisions chosen so that the FOUR channels are all reached across the phase grid.
/// `Ksi` selects among X_a : X_b : X_c : X_d, and the four cross sections move by orders of
/// magnitude between "at rest" (Prel2 <= 0, the fixed 625.1 / 0 / 49.989 / 6.614 mb) and 10
/// GeV/c, so the low-momentum rows reach the 3-shirt diagram and the high ones the single
/// quark-antiquark string. The anti-neutron and the anti-Lambda rows exercise two more of the
/// nine-by-two weight table's arms, and the Delta+ target row the `2214` entry.
const AnnihCase kAnnihCases[] = {
    {"pbar_p_rest",  -2212, 2212, 1.0,     0.0,  0.0,  0.0,  938.272013, 1,  1},
    {"pbar_p_100",   -2212, 2212, 100.0,   0.0,  0.0,  0.0,  938.272013, 12, 6},
    {"pbar_p_1G",    -2212, 2212, 1000.0,  0.0,  0.0,  0.0,  938.272013, 12, 6},
    {"pbar_p_10G",   -2212, 2212, 10000.0, 0.0,  0.0,  0.0,  938.272013, 12, 6},
    {"pbar_n_1G",    -2212, 2112, 1000.0,  20.0, -15.0, 10.0, 930.0,     12, 6},
    {"nbar_p_1G",    -2112, 2212, 1000.0,  0.0,  0.0,  0.0,  938.272013, 12, 6},
    {"nbar_n_10G",   -2112, 2112, 10000.0, 0.0,  0.0,  0.0,  939.56536,  56, 26},
    {"lbar_p_1G",    -3122, 2212, 1000.0,  0.0,  0.0,  0.0,  938.272013, 12, 6},
    {"pbar_d_1G",    -2212, 2214, 1000.0,  0.0,  0.0,  0.0,  1232.0,     12, 6},
};

void dump_annih() {
  FILE* f = std::fopen("ftf_annih.csv", "w");
  std::fprintf(f, "case,phase,result,ppdg,tpdg,ppx,ppy,ppz,pe,tpx,tpy,tpz,te,"
                  "pstatus,tstatus,pncol,tncol,pq0,pq1,pq0px,pq0py,pq0pz,pq0e,"
                  "pq1px,pq1py,pq1pz,pq1e,ptime,pposx,pposy,pposz,"
                  "nadd,apdg,aq0,aq1,apx,apy,apz,ae,"
                  "aq0px,aq0py,aq0pz,aq0e,aq1px,aq1py,aq1pz,aq1e,draws\n");
  G4FTFAnnihilation ann;
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (const AnnihCase& c : kAnnihCases) {
    G4FTFParameters* par = new G4FTFParameters();
    par->InitForInteraction(def_of(c.proj_pdg), c.targ_a, c.targ_z, c.plab);
    for (int ph = 0; ph < 8; ++ph) {
      eng.reset(ph);
      G4DiffractiveSplitableHadron* pr =
          make_projectile_splitable(c.proj_pdg, 0.0, 0.0, c.plab);
      G4DiffractiveSplitableHadron* tr = make_target_splitable(
          c.targ_pdg, c.targ_px, c.targ_py, c.targ_pz, c.targ_e, 1.0 * fermi, -2.0 * fermi,
          0.5 * fermi);
      pr->SetStatus(1);
      tr->SetStatus(1);
      tr->SetTimeOfCreation(3.25);
      G4VSplitableHadron* add = nullptr;
      const G4bool r = ann.Annihilate(pr, tr, add, par);
      const G4LorentzVector pp = pr->Get4Momentum();
      const G4LorentzVector tp = tr->Get4Momentum();
      const G4ThreeVector ps = pr->GetPosition();
      // The projectile's two partons, which the annihilation channels set AND give momenta to;
      // they survive into the string because BuildStrings finds the hadron already split.
      G4Parton* p0 = pr->GetNextParton();
      G4Parton* p1 = pr->GetNextParton();
      const G4LorentzVector p0m = p0 ? p0->Get4Momentum() : G4LorentzVector(0, 0, 0, 0);
      const G4LorentzVector p1m = p1 ? p1->Get4Momentum() : G4LorentzVector(0, 0, 0, 0);
      G4LorentzVector am(0, 0, 0, 0), a0m(0, 0, 0, 0), a1m(0, 0, 0, 0);
      int apdg = 0, aq0 = 0, aq1 = 0;
      if (add != nullptr) {
        am = add->Get4Momentum();
        apdg = add->GetDefinition() ? add->GetDefinition()->GetPDGEncoding() : 0;
        G4Parton* a0 = add->GetNextParton();
        G4Parton* a1 = add->GetNextParton();
        if (a0) { aq0 = a0->GetPDGcode(); a0m = a0->Get4Momentum(); }
        if (a1) { aq1 = a1->GetPDGcode(); a1m = a1->Get4Momentum(); }
      }
      std::fprintf(f,
                   "%s,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                   "%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                   "%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,"
                   "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n",
                   c.name, ph, r ? 1 : 0, pr->GetDefinition()->GetPDGEncoding(),
                   tr->GetDefinition()->GetPDGEncoding(), pp.px(), pp.py(), pp.pz(), pp.e(),
                   tp.px(), tp.py(), tp.pz(), tp.e(), pr->GetStatus(), tr->GetStatus(),
                   pr->GetSoftCollisionCount(), tr->GetSoftCollisionCount(),
                   p0 ? p0->GetPDGcode() : 0, p1 ? p1->GetPDGcode() : 0, p0m.px(), p0m.py(),
                   p0m.pz(), p0m.e(), p1m.px(), p1m.py(), p1m.pz(), p1m.e(),
                   pr->GetTimeOfCreation(), ps.x(), ps.y(), ps.z(), add ? 1 : 0, apdg, aq0, aq1,
                   am.px(), am.py(), am.pz(), am.e(), a0m.px(), a0m.py(), a0m.pz(), a0m.e(),
                   a1m.px(), a1m.py(), a1m.pz(), a1m.e(), eng.draws());
      delete pr;
      delete tr;
      delete add;
    }
    delete par;
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// ftf_cstrings.csv - G4DiffractiveExcitation::CreateStrings
// ---------------------------------------------------------------------------------------------

void dump_create_strings() {
  FILE* f = std::fopen("ftf_cstrings.csv", "w");
  std::fprintf(f, "case,phase,isproj,nstrings,left,right,lpx,lpy,lpz,le,rpx,rpy,rpz,re,"
                  "direction,time,posx,posy,posz,draws\n");
  G4DiffractiveExcitation exc;
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (const CollisionCase& c : kCollisions) {
    G4FTFParameters* par = params_for(c);
    for (int side = 0; side < 2; ++side) {
      for (int ph = 0; ph < 8; ++ph) {
        eng.reset(ph);
        G4DiffractiveSplitableHadron* h = nullptr;
        if (side == 0) {
          h = make_projectile_splitable(c.proj_pdg, 120.0, -80.0, c.plab);
        } else {
          h = make_target_splitable(c.targ_pdg, c.targ_px, c.targ_py, c.targ_pz,
                                    c.targ_e + 400.0, 1.0 * fermi, -2.0 * fermi, 0.5 * fermi);
        }
        h->SetStatus(side == 0 ? c.proj_status : c.targ_status);
        h->SetTimeOfCreation(3.25);
        G4ExcitedString* first = nullptr;
        G4ExcitedString* second = nullptr;
        exc.CreateStrings(h, side == 0, first, second, par);
        const int n = (first ? 1 : 0) + (second ? 1 : 0);
        if (first != nullptr) {
          const G4LorentzVector pl = first->GetLeftParton()->Get4Momentum();
          const G4LorentzVector pr = first->GetRightParton()->Get4Momentum();
          const G4ThreeVector pos = first->GetPosition();
          std::fprintf(f,
                       "%s,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                       "%d,%.17g,%.17g,%.17g,%.17g,%d\n",
                       c.name, ph, side, n, first->GetLeftParton()->GetPDGcode(),
                       first->GetRightParton()->GetPDGcode(), pl.px(), pl.py(), pl.pz(), pl.e(),
                       pr.px(), pr.py(), pr.pz(), pr.e(), first->GetDirection(),
                       first->GetTimeOfCreation(), pos.x(), pos.y(), pos.z(), eng.draws());
        }
        delete first;
        delete second;
        delete h;
      }
    }
    delete par;
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// ftf_nucleus.csv and ftf_getlist.csv - the replayed nucleus and the participant list
// ---------------------------------------------------------------------------------------------

struct NucCase {
  const char* name;
  int a, z;
};

const NucCase kNucCases[] = {{"C12", 12, 6},  {"O16", 16, 8},   {"Al27", 27, 13},
                             {"Fe56", 56, 26}, {"Pb207", 207, 82}};

/// The projectile the participant list is sampled for. `plab` is the z momentum after
/// G4HadProjectile has already put the beam on +z, so it is the whole of the momentum.
struct ListCase {
  const char* name;
  int pdg;
  double plab;
};

const ListCase kListCases[] = {{"p4", 2212, 4700.0},
                               {"p10", 2212, 10800.0},
                               {"pip4", 211, 4130.0},
                               {"kp10", 321, 10500.0},
                               {"pbar10", -2212, 10800.0}};

void dump_getlist() {
  FILE* fn = std::fopen("ftf_nucleus.csv", "w");
  // `posx,posy,posz` and not `x,y,z`: a column named `z` already carries the nucleus's CHARGE,
  // and a reader that maps names to indices takes the LAST duplicate - so the port replayed
  // Al27 with Z = 3 (the first nucleon's z coordinate in fm, truncated) and got a hadron-nucleon
  // interaction radius for the wrong isotope. Every other column of ftf_getlist.csv still
  // agreed, including the impact parameter and the draw count, because none of them depends on
  // Z; only the per-nucleon `RadiusOfHNinteractions2 > b^2` verdict does, and it moved by one
  // nucleon in one of the 200 (nucleus, projectile, phase) groups.
  std::fprintf(fn, "nucleus,a,z,index,type,posx,posy,posz,px,py,pz,e,binding\n");
  FILE* fl = std::fopen("ftf_getlist.csv", "w");
  std::fprintf(fl, "nucleus,proj,phase,b,ninter,index,targ_index,time,status,"
                   "prx,pry,prz,draws\n");

  // The nucleus is sampled with an ORDINARY engine and a fixed seed, then dumped; the cycle
  // engine is installed only for GetList, so the participant list is exact while the
  // configuration it runs on is reproducible by replay rather than by sharing a stream.
  CLHEP::HepJamesRandom nuceng(20260912);
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();

  for (const NucCase& nc : kNucCases) {
    CLHEP::HepRandom::setTheEngine(&nuceng);
    // The participants OWN the nucleus: `G4VParticipants::Init(A, Z)` builds a
    // G4Fancy3DNucleus and calls SortNucleonsIncZ, which is what G4FTFModel::Init does. It is
    // built here rather than handed in with SetNucleus because SetNucleus DELETES whatever is
    // already there - so `SetNucleus(nullptr)` at the end of a phase would destroy the very
    // configuration the next phase is supposed to replay. (Note also that the DECLARATION of
    // G4VParticipants::Init names its parameters `(theZ, theA)` and the definition names them
    // `(theA, theZ)`; the definition is what runs, and G4FTFModel passes A first.)
    G4FTFParticipants parts;
    parts.Init(nc.a, nc.z);
    G4V3DNucleus* nuc = parts.GetWoundedNucleus();
    nuc->StartLoop();
    G4Nucleon* n = nullptr;
    int idx = 0;
    while ((n = nuc->GetNextNucleon())) {
      const int type = (n->GetDefinition() == G4Proton::Proton())    ? 1
                       : (n->GetDefinition() == G4Neutron::Neutron()) ? 2
                                                                      : 3;
      const G4ThreeVector p = n->GetPosition();
      const G4LorentzVector m = n->Get4Momentum();
      std::fprintf(fn, "%s,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                   nc.name, nc.a, nc.z, idx, type, p.x(), p.y(), p.z(), m.px(), m.py(), m.pz(),
                   m.e(), n->GetBindingEnergy());
      ++idx;
    }

    for (const ListCase& lc : kListCases) {
      const G4ParticleDefinition* d = def_of(lc.pdg);
      G4FTFParameters* par = new G4FTFParameters();
      par->InitForInteraction(d, nc.a, nc.z, lc.plab);
      for (int ph = 0; ph < 8; ++ph) {
        // Erase the marks the previous phase left, so every phase starts from the SAME
        // configuration - which is the whole point of replaying one.
        nuc->StartLoop();
        while ((n = nuc->GetNextNucleon())) {
          if (n->AreYouHit()) {
            delete n->GetSplitableHadron();
            n->Hit(nullptr);
          }
        }
        CLHEP::HepRandom::setTheEngine(&eng);
        eng.reset(ph);
        G4ReactionProduct rp(d);
        rp.SetMomentum(0.0, 0.0, lc.plab);
        rp.SetTotalEnergy(
            std::sqrt(lc.plab * lc.plab + d->GetPDGMass() * d->GetPDGMass()));
        parts.GetList(rp, par);
        const int draws = eng.draws();
        const double b = parts.GetImpactParameter();
        int ninter = 0;
        parts.StartLoop();
        while (parts.Next()) { ++ninter; }
        parts.StartLoop();
        int k = 0;
        while (parts.Next()) {
          const G4InteractionContent& in = parts.GetInteraction();
          // Which nucleon of the replayed configuration this interaction names, by identity.
          int targ_index = -1;
          nuc->StartLoop();
          int j = 0;
          while ((n = nuc->GetNextNucleon())) {
            if (n == in.GetTargetNucleon()) { targ_index = j; }
            ++j;
          }
          const G4ThreeVector pr = in.GetProjectile()->GetPosition();
          std::fprintf(fl, "%s,%s,%d,%.17g,%d,%d,%d,%.17g,%d,%.17g,%.17g,%.17g,%d\n", nc.name,
                       lc.name, ph, b, ninter, k, targ_index, in.GetInteractionTime(),
                       in.GetStatus(), pr.x(), pr.y(), pr.z(), draws);
          ++k;
        }
        if (ninter == 0) {
          std::fprintf(fl, "%s,%s,%d,%.17g,0,-1,-1,0,0,0,0,0,%d\n", nc.name, lc.name, ph, b,
                       draws);
        }
      }
      delete par;
    }
    // Erase the marks before `parts` goes out of scope and takes the nucleus with it.
    nuc->StartLoop();
    while ((n = nuc->GetNextNucleon())) {
      if (n->AreYouHit()) {
        delete n->GetSplitableHadron();
        n->Hit(nullptr);
      }
    }
    parts.Clean();
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(fn);
  std::fclose(fl);
}

// ---------------------------------------------------------------------------------------------
// ftf_modelstat*.csv - the statistical half: the whole model, N events under a fixed seed
// ---------------------------------------------------------------------------------------------

struct ModelCase {
  const char* name;
  int pdg;
  double kin;   ///< kinetic energy, MeV - TOTAL, not per nucleon
  int a, z;
  int proj_a = 0;  ///< a projectile ION's mass number; 0 for a hadron
  int proj_z = 0;
};

/// The projectile definition, which for an ion is `G4IonTable::GetIon(Z, A, 0)` and not
/// anything `FindParticle` can be asked for.
const G4ParticleDefinition* projectile_def(const ModelCase& c) {
  if (c.proj_a > 0) {
    return G4ParticleTable::GetParticleTable()->GetIonTable()->GetIon(c.proj_z, c.proj_a, 0.0);
  }
  return def_of(c.pdg);
}

/// The beams and targets docs/HADRONIC_PLAN.md's P11 brief names, less the ion and anti-nucleon
/// arms that this port refuses: {p, n, pi+, pi-, K+} at {4, 10, 50} GeV on {C, O, Al, Fe, Pb}
/// would be 75 cases at 20,000 events, which is more than the oracle's runtime allows in one
/// dump; the fifteen below sample the corners - the lightest and heaviest target at every
/// energy for the proton, and one target per species.
const ModelCase kModelCases[] = {
    {"p_C_4",    2212, 4000.0,  12,  6},  {"p_C_10",   2212, 10000.0, 12,  6},
    {"p_C_50",   2212, 50000.0, 12,  6},  {"p_Pb_4",   2212, 4000.0,  207, 82},
    {"p_Pb_10",  2212, 10000.0, 207, 82}, {"p_Pb_50",  2212, 50000.0, 207, 82},
    {"n_O_10",   2112, 10000.0, 16,  8},  {"n_Fe_10",  2112, 10000.0, 56,  26},
    {"pip_C_4",  211,  4000.0,  12,  6},  {"pip_Al_10", 211, 10000.0, 27,  13},
    {"pim_C_10", -211, 10000.0, 12,  6},  {"pim_Pb_50", -211, 50000.0, 207, 82},
    {"kp_C_10",  321,  10000.0, 12,  6},  {"kp_Fe_50", 321,  50000.0, 56,  26},
    {"p_O_10",   2212, 10000.0, 16,  8},
    // The ION arm, at the brief's {alpha, C12} x {8, 20} GeV per nucleon. The kinetic energy
    // below is the TOTAL, which is what G4HadProjectile carries: 4 x 8 GeV, 12 x 20 GeV and so
    // on. These exercise the nucleus-nucleus branches of GetList, ReggeonCascade,
    // PutOnMassShell and BuildStrings, which share almost all of their code with the
    // hadron-nucleus ones and none of their test coverage without these rows.
    {"a_C_8",    0, 32000.0,  12,  6,  4,  2},
    {"a_Pb_8",   0, 32000.0,  207, 82, 4,  2},
    {"a_C_20",   0, 80000.0,  12,  6,  4,  2},
    {"C12_C_8",  0, 96000.0,  12,  6,  12, 6},
    {"C12_Pb_8", 0, 96000.0,  207, 82, 12, 6},
    {"C12_C_20", 0, 240000.0, 12,  6,  12, 6},
    // The ANTI-NUCLEON arm. `G4HadronicBuilder::BuildFTFP_BERT(..., bert=false)` gives FTFP
    // every energy for anti-nucleons, so this is the only beam that reaches G4FTFAnnihilation
    // at all - and the only one for which `theAdditionalString` is ever non-empty, which is
    // what makes `CreateStrings`' HadronIsString arm reachable. Both are above
    // `LowEnergyLimit` = 1 GeV/c (5 GeV of kinetic energy is 5.86 GeV/c for a proton mass), so
    // `HighEnergyInter` is true and `AdjustNucleons` - which this port refuses - is not on the
    // path; a slower anti-proton is a REFUSAL and not a statistical row.
    {"pbar_C_5",  -2212, 5000.0,  12,  6},
    {"pbar_Pb_5", -2212, 5000.0,  207, 82},
    {"nbar_C_5",  -2112, 5000.0,  12,  6},
};

// ---------------------------------------------------------------------------------------------
// ftf_aanucleus.csv and ftf_getlist_aa.csv - GetList's NUCLEUS-NUCLEUS arm, exactly
//
// The hadron arm is checked exactly by replaying one nucleus (ftf_nucleus.csv); the AA arm needs
// TWO, and the projectile one has been boosted and Lorentz-contracted by `G4FTFModel::Init`
// before GetList sees it. Both are dumped after those operations, so the port replays exactly
// the configuration the participant search ran on and the comparison is exact rather than
// statistical - which matters because a 3% difference in an AA participant count is invisible
// against the event-to-event spread and obvious here.
// ---------------------------------------------------------------------------------------------

struct AaCase {
  const char* name;
  int proj_a, proj_z;
  int targ_a, targ_z;
  double kin;  ///< TOTAL kinetic energy of the ion, MeV
};

const AaCase kAaCases[] = {
    {"He4_C12", 4, 2, 12, 6, 32000.0},
    {"C12_C12", 12, 6, 12, 6, 96000.0},
    {"C12_Pb207", 12, 6, 207, 82, 96000.0},
};

void dump_getlist_aa() {
  FILE* fn = std::fopen("ftf_aanucleus.csv", "w");
  std::fprintf(fn, "case,side,a,z,index,type,posx,posy,posz,px,py,pz,e,binding\n");
  FILE* fl = std::fopen("ftf_getlist_aa.csv", "w");
  std::fprintf(fl, "case,phase,b,ninter,index,proj_index,targ_index,time,status,draws\n");

  CLHEP::HepJamesRandom nuceng(20260915);
  CycleEngine eng;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();

  for (const AaCase& c : kAaCases) {
    CLHEP::HepRandom::setTheEngine(&nuceng);
    G4FTFParticipants parts;
    parts.InitProjectileNucleus(c.proj_a, c.proj_z, 0);
    parts.Init(c.targ_a, c.targ_z);
    G4V3DNucleus* pnuc = parts.GetProjectileNucleus();
    G4V3DNucleus* tnuc = parts.GetWoundedNucleus();

    // The projectile's boost and contraction, exactly as G4FTFModel::Init applies them.
    const G4ParticleDefinition* d =
        G4ParticleTable::GetParticleTable()->GetIonTable()->GetIon(c.proj_z, c.proj_a, 0.0);
    const double m = d->GetPDGMass();
    const double p = std::sqrt(c.kin * (c.kin + 2.0 * m));
    const double e = c.kin + m;
    const G4ThreeVector boost(0.0, 0.0, p / e);
    pnuc->DoLorentzBoost(boost);
    pnuc->DoLorentzContraction(boost);

    G4Nucleon* n = nullptr;
    for (int side = 0; side < 2; ++side) {
      G4V3DNucleus* nuc = (side == 0) ? pnuc : tnuc;
      const int a = (side == 0) ? c.proj_a : c.targ_a;
      const int z = (side == 0) ? c.proj_z : c.targ_z;
      nuc->StartLoop();
      int idx = 0;
      while ((n = nuc->GetNextNucleon())) {
        const int type = (n->GetDefinition() == G4Proton::Proton())     ? 1
                         : (n->GetDefinition() == G4Neutron::Neutron()) ? 2
                                                                        : 3;
        const G4ThreeVector pos = n->GetPosition();
        const G4LorentzVector mom = n->Get4Momentum();
        std::fprintf(fn, "%s,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     c.name, side, a, z, idx, type, pos.x(), pos.y(), pos.z(), mom.px(),
                     mom.py(), mom.pz(), mom.e(), n->GetBindingEnergy());
        ++idx;
      }
    }

    G4FTFParameters* par = new G4FTFParameters();
    par->InitForInteraction(d, c.targ_a, c.targ_z, p / c.proj_a);

    // The projectile nucleus is TRANSLATED by GetList when it finds an interaction, so every
    // phase has to start from the same positions: they are saved and restored around each call.
    std::vector<G4ThreeVector> saved_pos;
    pnuc->StartLoop();
    while ((n = pnuc->GetNextNucleon())) { saved_pos.push_back(n->GetPosition()); }

    for (int ph = 0; ph < 8; ++ph) {
      for (int side = 0; side < 2; ++side) {
        G4V3DNucleus* nuc = (side == 0) ? pnuc : tnuc;
        nuc->StartLoop();
        while ((n = nuc->GetNextNucleon())) {
          if (n->AreYouHit()) {
            delete n->GetSplitableHadron();
            n->Hit(nullptr);
          }
        }
      }
      {
        pnuc->StartLoop();
        size_t k = 0;
        while ((n = pnuc->GetNextNucleon())) { n->SetPosition(saved_pos[k++]); }
      }
      CLHEP::HepRandom::setTheEngine(&eng);
      eng.reset(ph);
      G4ReactionProduct rp(d);
      rp.SetMomentum(0.0, 0.0, p);
      rp.SetTotalEnergy(e);
      parts.GetList(rp, par);
      const int draws = eng.draws();
      const double b = parts.GetImpactParameter();
      int ninter = 0;
      parts.StartLoop();
      while (parts.Next()) { ++ninter; }
      parts.StartLoop();
      int kk = 0;
      while (parts.Next()) {
        const G4InteractionContent& in = parts.GetInteraction();
        int pi = -1, ti = -1, j = 0;
        pnuc->StartLoop();
        j = 0;
        while ((n = pnuc->GetNextNucleon())) {
          if (n == in.GetProjectileNucleon()) { pi = j; }
          ++j;
        }
        tnuc->StartLoop();
        j = 0;
        while ((n = tnuc->GetNextNucleon())) {
          if (n == in.GetTargetNucleon()) { ti = j; }
          ++j;
        }
        std::fprintf(fl, "%s,%d,%.17g,%d,%d,%d,%d,%.17g,%d,%d\n", c.name, ph, b, ninter, kk, pi,
                     ti, in.GetInteractionTime(), in.GetStatus(), draws);
        ++kk;
      }
      if (ninter == 0) {
        std::fprintf(fl, "%s,%d,%.17g,0,-1,-1,-1,0,0,%d\n", c.name, ph, b, draws);
      }
    }
    // Erase the marks so the splitable hadrons go with `parts`.
    for (int side = 0; side < 2; ++side) {
      G4V3DNucleus* nuc = (side == 0) ? pnuc : tnuc;
      nuc->StartLoop();
      while ((n = nuc->GetNextNucleon())) {
        if (n->AreYouHit()) {
          delete n->GetSplitableHadron();
          n->Hit(nullptr);
        }
      }
    }
    parts.Clean();
    delete par;
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(fn);
  std::fclose(fl);
}

// ---------------------------------------------------------------------------------------------
// ftf_nucstat.csv - the nucleus's own geometry, which is what the FTF geometry is built on
// ---------------------------------------------------------------------------------------------

/// `GetOuterRadius()` and the transverse RMS of one sampled nucleus, N times per nuclide.
///
/// This is P9's `G4Fancy3DNucleus` and not this package's code, and it is dumped here because
/// FTF is the first consumer that is SENSITIVE to it in a way a radial histogram is not.
/// `G4FTFParticipants::GetList` samples an impact parameter in a disc of radius
/// `GetOuterRadius() + 2 fm` and then asks, nucleon by nucleon, whether `b^2` in the TRANSVERSE
/// plane is inside `RadiusOfHNinteractions2`. So the two numbers that decide a participant count
/// are an EXTREME-value statistic (the furthest nucleon) and a transverse second moment -
/// neither of which a histogram of |r| constrains tightly.
///
/// He4 and C12 are here because they are the two projectile ions the brief names, and C12
/// because it is the one nuclide `G4Fancy3DNucleus::ChoosePositions` builds from three alpha
/// clusters rather than from the density (docs/RISK.md V68).
void dump_nucstat() {
  // `C12big` is the same nuclide as `C12` at ten times the statistics, and it is here for
  // docs/RISK.md V88: C12's outer radius sits at 2.8 sigma over 20,000 configurations, C12 on
  // carbon is the one model case that fails, and the same nuclide is that case's projectile AND
  // its target - so the question "is the C12 configuration itself biased" has to be asked at a
  // sample size where the answer is not a fluctuation. It samples from its own engine seed, so
  // the two rows are independent draws and not the first 20,000 of the 200,000.
  struct NucStatCase { const char* name; int a, z; int n; };
  const NucStatCase kCases[] = {{"He4", 4, 2, 20000},    {"C12", 12, 6, 20000},
                                {"O16", 16, 8, 20000},   {"Al27", 27, 13, 20000},
                                {"Fe56", 56, 26, 20000}, {"Pb207", 207, 82, 20000},
                                {"C12big", 12, 6, 200000}};
  FILE* f = std::fopen("ftf_nucstat.csv", "w");
  std::fprintf(f, "nucleus,a,z,n_events,quantity,bin,count\n");
  CLHEP::HepJamesRandom eng(20260914);
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  CLHEP::HepRandom::setTheEngine(&eng);
  for (const NucStatCase& c : kCases) {
    const int N = c.n;
    if (N > 20000) { eng.setSeed(20260916, 0); }
    std::map<int, long long> outer, rms_t, rms_r;
    G4Fancy3DNucleus* nuc = new G4Fancy3DNucleus();
    for (int ev = 0; ev < N; ++ev) {
      nuc->Init(c.a, c.z);
      nuc->SortNucleonsIncZ();
      // `GetOuterRadius()` in quarter-fermi bins.
      int b = (G4int)(4.0 * nuc->GetOuterRadius() / fermi);
      if (b < 0) { b = 0; }
      if (b > 199) { b = 199; }
      ++outer[b];
      double sx = 0.0, sr = 0.0;
      nuc->StartLoop();
      G4Nucleon* n = nullptr;
      while ((n = nuc->GetNextNucleon())) {
        const G4ThreeVector p = n->GetPosition();
        sx += (p.x()*p.x() + p.y()*p.y()) / (fermi*fermi);
        sr += p.mag2() / (fermi*fermi);
      }
      int bt = (G4int)(10.0 * std::sqrt(sx / c.a));
      if (bt < 0) { bt = 0; }
      if (bt > 199) { bt = 199; }
      ++rms_t[bt];
      int br = (G4int)(10.0 * std::sqrt(sr / c.a));
      if (br < 0) { br = 0; }
      if (br > 199) { br = 199; }
      ++rms_r[br];
    }
    delete nuc;
    for (const auto& kv : outer) {
      std::fprintf(f, "%s,%d,%d,%d,outer_qfm,%d,%lld\n", c.name, c.a, c.z, N, kv.first,
                   kv.second);
    }
    for (const auto& kv : rms_t) {
      std::fprintf(f, "%s,%d,%d,%d,rms_t_dfm,%d,%lld\n", c.name, c.a, c.z, N, kv.first,
                   kv.second);
    }
    for (const auto& kv : rms_r) {
      std::fprintf(f, "%s,%d,%d,%d,rms_r_dfm,%d,%lld\n", c.name, c.a, c.z, N, kv.first,
                   kv.second);
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

// ---------------------------------------------------------------------------------------------
// ftf_aaradius.csv - the ONE geometric number a nucleus-nucleus collision has that a
// hadron-nucleus one does not
//
// `G4FTFParticipants::GetList`'s AA arm samples the impact parameter in a disc of radius
//
//     xyradius = theProjectileNucleus->GetOuterRadius() + theNucleus->GetOuterRadius() + 2 fm
//
// and the projectile's outer radius is measured AFTER `G4FTFModel::Init` has boosted the
// projectile nucleus and Lorentz-CONTRACTED it, so it is not the radius `ftf_nucstat.csv`
// measures. That makes it the one input to the AA geometry that no other table constrains:
// ftf_getlist_aa.csv replays both nuclei and therefore tests the arithmetic on a GIVEN
// configuration, ftf_nucstat.csv measures the nucleus before the boost, and nothing measured
// the contracted one. This file does, as a distribution, for every ion case.
//
// It exists because C12 on carbon disagreed - 3.7% fewer participants and 2.3% larger impact
// parameter - while the nucleus, the FTF parameters and the replayed GetList all agreed
// exactly. That is the same shape of question docs/RISK.md V99 answered by adding three
// counters that localise rather than detect.
void dump_aaradius() {
  FILE* f = std::fopen("ftf_aaradius.csv", "w");
  std::fprintf(f, "case,n_events,quantity,bin,count\n");
  const int N = 20000;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  for (const ModelCase& c : kModelCases) {
    if (c.proj_a <= 0) { continue; }  // The projectile is a hadron: there is no second radius
    CLHEP::HepJamesRandom eng(20260917);
    CLHEP::HepRandom::setTheEngine(&eng);
    FtfModelProbe* model = new FtfModelProbe("FTFP");
    const G4ParticleDefinition* d = projectile_def(c);
    std::map<int, long long> projr, targr, xyr;
    for (int ev = 0; ev < N; ++ev) {
      G4Nucleus nucleus(c.a, c.z);
      const double p = std::sqrt(c.kin * (c.kin + 2.0 * d->GetPDGMass()));
      G4DynamicParticle dp(d, G4ThreeVector(0.0, 0.0, p));
      model->Init(nucleus, dp);
      const double pr = model->GetProjectileNucleus()->GetOuterRadius() / fermi;
      const double tr = model->GetWoundedNucleus()->GetOuterRadius() / fermi;
      const double xy = pr + tr + 2.0;
      int b = (G4int)(20.0 * pr);  // twentieth-fermi bins: the contracted radius is small
      if (b < 0) { b = 0; }
      if (b > 499) { b = 499; }
      ++projr[b];
      b = (G4int)(4.0 * tr);
      if (b < 0) { b = 0; }
      if (b > 199) { b = 199; }
      ++targr[b];
      b = (G4int)(4.0 * xy);
      if (b < 0) { b = 0; }
      if (b > 199) { b = 199; }
      ++xyr[b];
    }
    delete model;
    for (const auto& kv : projr) {
      std::fprintf(f, "%s,%d,proj_outer_twfm,%d,%lld\n", c.name, N, kv.first, kv.second);
    }
    for (const auto& kv : targr) {
      std::fprintf(f, "%s,%d,targ_outer_qfm,%d,%lld\n", c.name, N, kv.first, kv.second);
    }
    for (const auto& kv : xyr) {
      std::fprintf(f, "%s,%d,xyradius_qfm,%d,%lld\n", c.name, N, kv.first, kv.second);
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}


// ---------------------------------------------------------------------------------------------
// ftf_prescatter.csv - what G4FTFModel produces BEFORE G4VPartonStringModel::Scatter throws any
// of it away
//
// `Scatter` runs `Init` + `GetStrings` and then rejects the whole attempt if the nuclear
// residual is unphysical, re-sampling everything. For a C12 beam on carbon that rejection fires
// on about one attempt in thirteen, and it is a CONDITIONING: it removes the events with the
// most hit nucleons, so every distribution measured after `Scatter` - `ftf_modelstat_*.csv` -
// is a conditional one. Two implementations that agree perfectly on the model can then
// disagree on those tables if they disagree on the rejection, and the tables cannot tell the
// two apart.
//
// This file is the same quantities measured with the rejection NOT applied: one `Init` plus one
// `GetStrings` per event, and the hit counts read straight off the two nuclei. The residual
// clauses are pure integer logic on those counts, so a port that reproduces these histograms
// reproduces the rejection rate as well, and one that does not has a difference in the model
// rather than in the loop around it.
void dump_prescatter() {
  FILE* f = std::fopen("ftf_prescatter.csv", "w");
  std::fprintf(f, "case,n_events,quantity,bin,count\n");
  const int N = 20000;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  for (const ModelCase& c : kModelCases) {
    // The six ion cases and two hadron controls. The rest are covered by ftf_modelstat_*.csv,
    // where the rejection almost never fires.
    const bool wanted = c.proj_a > 0 || std::strcmp(c.name, "p_C_10") == 0 ||
                        std::strcmp(c.name, "p_Pb_10") == 0;
    if (!wanted) { continue; }
    CLHEP::HepJamesRandom eng(20260918);
    CLHEP::HepRandom::setTheEngine(&eng);
    FtfModelProbe* model = new FtfModelProbe("FTFP");
    const G4ParticleDefinition* d = projectile_def(c);
    std::map<int, long long> tp, tn, pp, pn, bimp, nstr, nncoll, tused, rejected;
    for (int ev = 0; ev < N; ++ev) {
      G4Nucleus nucleus(c.a, c.z);
      const double p = std::sqrt(c.kin * (c.kin + 2.0 * d->GetPDGMass()));
      G4DynamicParticle dp(d, G4ThreeVector(0.0, 0.0, p));
      model->Init(nucleus, dp);
      G4ExcitedStringVector* v = model->GetStrings();
      int ntp = 0, ntn = 0, npp = 0, npn = 0;
      const std::vector<G4Nucleon>& tt = model->GetWoundedNucleus()->GetNucleons();
      for (const auto& nuc : tt) {
        if (!nuc.AreYouHit()) { continue; }
        if (nuc.GetDefinition() == G4Proton::Proton()) { ++ntp; }
        if (nuc.GetDefinition() == G4Neutron::Neutron()) { ++ntn; }
      }
      if (model->GetProjectileNucleus() != nullptr) {
        const std::vector<G4Nucleon>& ppn = model->GetProjectileNucleus()->GetNucleons();
        for (const auto& nuc : ppn) {
          if (!nuc.AreYouHit()) { continue; }
          if (nuc.GetDefinition() == G4Proton::Proton() ||
              nuc.GetDefinition() == G4AntiProton::AntiProton()) { ++npp; }
          if (nuc.GetDefinition() == G4Neutron::Neutron() ||
              nuc.GetDefinition() == G4AntiNeutron::AntiNeutron()) { ++npn; }
        }
      }
      ++tp[ntp]; ++tn[ntn]; ++pp[npp]; ++pn[npn];
      const double b = model->GetImpactParameter() / fermi;
      int bb = (G4int)(2.0 * b);
      if (bb < 0) { bb = 0; }
      if (bb > 79) { bb = 79; }
      ++bimp[bb];
      ++nstr[v ? (G4int)v->size() : 0];
      ++nncoll[model->GetNumberOfNNcollisions()];
      ++tused[c.a - model->GetNumberOfTargetSpectatorNucleons()];

      // G4VPartonStringModel::Scatter's residual clauses, literally, on the counts above.
      const G4int zT = c.z - ntp;
      const G4int nT = c.a - c.z - ntn;
      const G4int zP = (c.proj_a > 0) ? c.proj_z - npp : 0;
      const G4int nP = (c.proj_a > 0) ? c.proj_a - c.proj_z - npn : 0;
      G4int why = 0;
      if ( ( zT > 3 && nT == 0 ) || ( zT == 0 && nT > 1 ) ) why |= 1;
      if ( ( zP > 3 && nP == 0 ) || ( zP == 0 && nP > 1 ) ) why |= 2;
      ++rejected[why];

      if (v) {
        for (size_t i = 0; i < v->size(); ++i) { delete (*v)[i]; }
        delete v;
      }
    }
    delete model;
    struct Out { const char* q; std::map<int, long long>* m; };
    const Out outs[] = {{"targ_p_hits", &tp}, {"targ_n_hits", &tn},  {"proj_p_hits", &pp},
                        {"proj_n_hits", &pn}, {"b_halffm", &bimp},   {"nstrings", &nstr},
                        {"nncoll", &nncoll},  {"participants", &tused}, {"rejected", &rejected}};
    for (const Out& o : outs) {
      for (const auto& kv : *o.m) {
        std::fprintf(f, "%s,%d,%s,%d,%lld\n", c.name, N, o.q, kv.first, kv.second);
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
}

/// ftf_modelcases.csv - the INPUTS of the statistical cases, so that the port does not have to
/// copy them.
///
/// `pmass` matters and is the reason this file exists rather than a table duplicated in the
/// test: for an ION the projectile's PDG mass comes from `G4IonTable`, which is
/// `G4NucleiProperties::GetNuclearMass(A, Z)` and NOT the sum of nucleon masses minus the
/// binding energy that `G4Fancy3DNucleus::GetMass()` gives (P9's nucleus_model.cuh lists all
/// three answers). Getting it from the wrong one moves sqrt(s) and every distribution with it,
/// by an amount too small to look like a bug and too large to be rounding.
void dump_modelcases() {
  FILE* f = std::fopen("ftf_modelcases.csv", "w");
  // The G4FTFParameters columns are here because P11's `ftf_params.csv` grid is 22 named
  // PROJECTILE PARTICLES and an ion is not one of them - `InitForInteraction`'s
  // `ProjectileIsNucleus` arm is reached only through a beam whose baryon number exceeds 1, and
  // nothing before P11b ever handed it one. These eleven numbers are every parameter the model
  // above reads, for exactly the cases the statistical half runs, and they are compared
  // EXACTLY: an ion arm that is 1% out in `RadiusOfHNinteractions2` produces a participant
  // count that is 3% out and looks like a fluctuation.
  // `RadiusOfHNinteractions2` has no getter - it is `Xtotal/pi/10` and nothing else, so
  // comparing `xtotal` compares it.
  std::fprintf(f, "case,pdg,kin,a,z,proj_a,proj_z,pmass,plab,plab_per_n,"
                  "xtotal,xelastic,xinel,prob_el,prob_annih,cof_nd,cof_nd_pr,r2_nd,dof_nd,"
                  "pt2_nd,maxpt2_nd,exc_per_wn,max_ncoll\n");
  for (const ModelCase& c : kModelCases) {
    const G4ParticleDefinition* d = projectile_def(c);
    const double m = d->GetPDGMass();
    const double p = std::sqrt(c.kin * (c.kin + 2.0 * m));
    const double plab_per_n = (c.proj_a > 0) ? p / c.proj_a : p;
    G4FTFParameters* par = new G4FTFParameters();
    par->InitForInteraction(d, c.a, c.z, plab_per_n);
    std::fprintf(f,
                 "%s,%d,%.17g,%d,%d,%d,%d,%.17g,%.17g,%.17g,"
                 "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                 c.name, d->GetPDGEncoding(), c.kin, c.a, c.z, c.proj_a, c.proj_z, m, p,
                 plab_per_n, par->GetTotalCrossSection(), par->GetElasticCrossSection(),
                 par->GetInelasticCrossSection(),
                 par->GetProbabilityOfElasticScatt(), par->GetProbabilityOfAnnihilation(),
                 par->GetCofNuclearDestruction(), par->GetCofNuclearDestructionPr(),
                 par->GetR2ofNuclearDestruction(), par->GetDofNuclearDestruction(),
                 par->GetPt2ofNuclearDestruction(), par->GetMaxPt2ofNuclearDestruction(),
                 par->GetExcitationEnergyPerWoundedNucleon(), par->GetMaxNumberOfCollisions());
    delete par;
  }
  std::fclose(f);
}

/// `G4VPartonStringModel::Scatter` N times, plus a separate `Init` + `GetStrings` pass for the
/// string-level numbers, which `Scatter` consumes and does not return.
///
/// THE TWO PASSES USE DIFFERENT EVENTS ON PURPOSE. Sharing them would mean calling GetStrings
/// and then Scatter on the same model instance, and Scatter calls Init again - so the strings
/// dumped would belong to a nucleus the hadrons never saw. Each pass is its own N events under
/// its own seed, and both are statistical, so nothing is lost.
void dump_modelstat() {
  FILE* fs = std::fopen("ftf_modelstat_strings.csv", "w");
  std::fprintf(fs, "case,n_events,quantity,bin,count\n");
  FILE* fh = std::fopen("ftf_modelstat_species.csv", "w");
  std::fprintf(fh, "case,n_events,pdg,count,sum_e,sum_pz,sum_pt2\n");
  FILE* fm = std::fopen("ftf_modelstat_mult.csv", "w");
  std::fprintf(fm, "case,n_events,multiplicity,count\n");
  FILE* fw = std::fopen("ftf_modelstat_wounded.csv", "w");
  std::fprintf(fw, "case,n_events,quantity,bin,count\n");

  const int N = 20000;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();

  for (const ModelCase& c : kModelCases) {
    const G4ParticleDefinition* d = projectile_def(c);

    // ---- pass 1: the strings at GetStrings exit ----
    {
      CLHEP::HepJamesRandom eng(20260912);
      CLHEP::HepRandom::setTheEngine(&eng);
      FtfModelProbe* model = new FtfModelProbe("FTFP");
      std::map<int, long long> nstrings, mass_bin, excited, bimp, nncoll, tused;
      for (int ev = 0; ev < N; ++ev) {
        G4Nucleus nucleus(c.a, c.z);
        const double p = std::sqrt(c.kin * (c.kin + 2.0 * d->GetPDGMass()));
        G4DynamicParticle dp(d, G4ThreeVector(0.0, 0.0, p));
        model->Init(nucleus, dp);
        G4ExcitedStringVector* v = model->GetStrings();
        // The three PUBLIC counters, which is what says WHERE a statistical disagreement is
        // rather than only that there is one:
        //   `b`      the impact parameter in half-fermi bins - the nucleus's geometry and the
        //            sampler, with none of the physics above them;
        //   `nncoll` NumberOfNNcollisions, incremented once per SUCCESSFUL
        //            G4DiffractiveExcitation::ExciteParticipants, so it separates "how many
        //            nucleons were hit" from "how many of them were actually excited";
        //   `tused`  A - NumberOfTargetSpectatorNucleons. BuildStrings decrements that counter
        //            once per target nucleon it turns into a string OR into a quark-exchange
        //            kinetic track, and NOT for a reggeon-cascade nucleon - so it is the
        //            Glauber participant count, separately from the reggeon cascade's.
        {
          const double b = model->GetImpactParameter() / fermi;
          int bb = (G4int)(2.0 * b);
          if (bb < 0) { bb = 0; }
          if (bb > 79) { bb = 79; }
          ++bimp[bb];
          ++nncoll[model->GetNumberOfNNcollisions()];
          ++tused[c.a - model->GetNumberOfTargetSpectatorNucleons()];
        }
        const int n = v ? (int)v->size() : 0;
        ++nstrings[n];
        for (int i = 0; i < n; ++i) {
          G4ExcitedString* s = (*v)[i];
          ++excited[s->IsExcited() ? 1 : 0];
          const double m = s->Get4Momentum().mag();
          // log10(M/MeV) in tenths of a decade, clamped into [0, 60).
          int b = (m > 0.0) ? (int)(10.0 * std::log10(m)) : -1;
          if (b < -1) { b = -1; }
          if (b > 59) { b = 59; }
          ++mass_bin[b];
        }
        if (v) {
          for (int i = 0; i < n; ++i) { delete (*v)[i]; }
          delete v;
        }
      }
      for (const auto& kv : nstrings) {
        std::fprintf(fs, "%s,%d,nstrings,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      for (const auto& kv : mass_bin) {
        std::fprintf(fs, "%s,%d,log10mass,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      for (const auto& kv : excited) {
        std::fprintf(fs, "%s,%d,excited,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      for (const auto& kv : bimp) {
        std::fprintf(fs, "%s,%d,b_halffm,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      for (const auto& kv : nncoll) {
        std::fprintf(fs, "%s,%d,nncoll,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      for (const auto& kv : tused) {
        std::fprintf(fs, "%s,%d,participants,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      delete model;
    }

    // ---- pass 2: the hadrons and the wounded nucleus at the hand-over ----
    {
      CLHEP::HepJamesRandom eng(20260913);
      CLHEP::HepRandom::setTheEngine(&eng);
      G4FTFModel* model = new G4FTFModel("FTFP");
      model->SetFragmentationModel(new G4ExcitedStringDecay(new G4LundStringFragmentation()));
      std::map<int, long long> count, mult, holes, charged_holes;
      std::map<int, double> sum_e, sum_pz, sum_pt2;
      for (int ev = 0; ev < N; ++ev) {
        G4Nucleus nucleus(c.a, c.z);
        const double p = std::sqrt(c.kin * (c.kin + 2.0 * d->GetPDGMass()));
        G4DynamicParticle dp(d, G4ThreeVector(0.0, 0.0, p));
        G4KineticTrackVector* r = model->Scatter(nucleus, dp);
        const int n = r ? (int)r->size() : 0;
        ++mult[n];
        for (int i = 0; i < n; ++i) {
          G4KineticTrack* t = (*r)[i];
          const int pdg = t->GetDefinition()->GetPDGEncoding();
          ++count[pdg];
          const G4LorentzVector m = t->Get4Momentum();
          sum_e[pdg] += m.e();
          sum_pz[pdg] += m.pz();
          sum_pt2[pdg] += m.perp2();
          delete t;
        }
        delete r;
        int nh = 0, nch = 0;
        const std::vector<G4Nucleon>& they = model->GetWoundedNucleus()->GetNucleons();
        for (const auto& nuc : they) {
          if (nuc.AreYouHit()) {
            ++nh;
            if (nuc.GetDefinition() == G4Proton::Proton()) { ++nch; }
          }
        }
        ++holes[nh];
        ++charged_holes[nch];
      }
      for (const auto& kv : count) {
        std::fprintf(fh, "%s,%d,%d,%lld,%.17g,%.17g,%.17g\n", c.name, N, kv.first, kv.second,
                     sum_e[kv.first], sum_pz[kv.first], sum_pt2[kv.first]);
      }
      for (const auto& kv : mult) {
        std::fprintf(fm, "%s,%d,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      for (const auto& kv : holes) {
        std::fprintf(fw, "%s,%d,holes,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      for (const auto& kv : charged_holes) {
        std::fprintf(fw, "%s,%d,charged_holes,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      delete model;
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(fs);
  std::fclose(fh);
  std::fclose(fm);
  std::fclose(fw);
}

// ---------------------------------------------------------------------------------------------
// ftf_modelbig*.csv - docs/RISK.md V88's rule, applied
// ---------------------------------------------------------------------------------------------

/// The three cases that carried the worst z-scores at 20,000 events, re-run at 200,000.
///
/// V88: "20,000 events is the oracle's limit and four port seeds agreeing is not independence -
/// when a statistical row sits at 3 sigma, re-run BOTH sides at 200,000 before calling it."
/// The worst rows at 20,000 were n_Fe_10's NN-collision count (3.49), pip_Al_10's multiplicity
/// (3.43) and pim_C_10's string count (3.31), and a 3-sigma row at 20,000 is either a
/// fluctuation - in which case it shrinks in SIGMA terms at 200,000 only if it was one - or a
/// real 0.5% difference, which at ten times the statistics comes back at 10 sigma. The point of
/// the re-run is that those two outcomes are distinguishable and a single 20,000-event number
/// is not.
///
/// Only three cases, and only the light targets: this runs inside the same table dump as
/// everything else and 200,000 events of a 50 GeV pion on lead would dominate its runtime.
const ModelCase kBigCases[] = {
    {"n_Fe_10", 2112, 10000.0, 56, 26},
    {"pip_Al_10", 211, 10000.0, 27, 13},
    {"pim_C_10", -211, 10000.0, 12, 6},
    // docs/RISK.md V88: a row that sits above 3 sigma at 20,000 events is re-run at 200,000 on
    // BOTH sides before it is called a difference. C12 on carbon at 8 GeV/nucleon is the only
    // case that does - participants at 5.34 sigma, excited/not-excited at 6.32 - and it is the
    // one case in the table where the same nuclide is the projectile AND the target, so a bias
    // in one nuclide's sampled radius would enter it twice.
    {"C12_C_8", 0, 96000.0, 12, 6, 12, 6},
};

void dump_modelbig() {
  FILE* fs = std::fopen("ftf_modelbig_strings.csv", "w");
  std::fprintf(fs, "case,n_events,quantity,bin,count\n");
  FILE* fm = std::fopen("ftf_modelbig_mult.csv", "w");
  std::fprintf(fm, "case,n_events,multiplicity,count\n");

  const int N = 200000;
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();

  for (const ModelCase& c : kBigCases) {
    const G4ParticleDefinition* d = projectile_def(c);
    {
      CLHEP::HepJamesRandom eng(20260912);
      CLHEP::HepRandom::setTheEngine(&eng);
      FtfModelProbe* model = new FtfModelProbe("FTFP");
      std::map<int, long long> nstrings, nncoll, tused;
      for (int ev = 0; ev < N; ++ev) {
        G4Nucleus nucleus(c.a, c.z);
        const double p = std::sqrt(c.kin * (c.kin + 2.0 * d->GetPDGMass()));
        G4DynamicParticle dp(d, G4ThreeVector(0.0, 0.0, p));
        model->Init(nucleus, dp);
        G4ExcitedStringVector* v = model->GetStrings();
        ++nstrings[v ? (int)v->size() : 0];
        ++nncoll[model->GetNumberOfNNcollisions()];
        ++tused[c.a - model->GetNumberOfTargetSpectatorNucleons()];
        if (v) {
          for (size_t i = 0; i < v->size(); ++i) { delete (*v)[i]; }
          delete v;
        }
      }
      for (const auto& kv : nstrings) {
        std::fprintf(fs, "%s,%d,nstrings,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      for (const auto& kv : nncoll) {
        std::fprintf(fs, "%s,%d,nncoll,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      for (const auto& kv : tused) {
        std::fprintf(fs, "%s,%d,participants,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      delete model;
    }
    {
      CLHEP::HepJamesRandom eng(20260913);
      CLHEP::HepRandom::setTheEngine(&eng);
      G4FTFModel* model = new G4FTFModel("FTFP");
      model->SetFragmentationModel(new G4ExcitedStringDecay(new G4LundStringFragmentation()));
      std::map<int, long long> mult;
      for (int ev = 0; ev < N; ++ev) {
        G4Nucleus nucleus(c.a, c.z);
        const double p = std::sqrt(c.kin * (c.kin + 2.0 * d->GetPDGMass()));
        G4DynamicParticle dp(d, G4ThreeVector(0.0, 0.0, p));
        G4KineticTrackVector* r = model->Scatter(nucleus, dp);
        ++mult[r ? (int)r->size() : 0];
        if (r) {
          for (size_t i = 0; i < r->size(); ++i) { delete (*r)[i]; }
          delete r;
        }
      }
      for (const auto& kv : mult) {
        std::fprintf(fm, "%s,%d,%d,%lld\n", c.name, N, kv.first, kv.second);
      }
      delete model;
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(fs);
  std::fclose(fm);
}

// ---------------------------------------------------------------------------------------------
// ftf_windows.csv - which particles QBBC actually gives FTFP, and between which energies
//
// docs/PORTED.md 2.1.11b states the energy windows by reading the builders - 3 GeV upwards for
// p, n, pi+- through `GetMinEnergyTransitionFTF_Cascade()`, all energies for anti-nucleons
// through `BuildFTFP_BERT(..., bert=false)`, 3 GeV/nucleon upwards for ions through
// `G4IonPhysicsXS`. Reading a builder is not measuring one: the number that decides is what the
// CONSTRUCTED process holds, after every SetMinEnergy/SetMaxEnergy any of the three builders
// made, and G4HadronicProcess keeps it where it can be read.
//
// The physics list is already built by the time any dump runs, so this costs nothing: it walks
// the process manager of each species, finds the G4HadronicProcess objects, and writes one row
// per registered model with the energy range that model is asked for. A row whose model name is
// FTFP means this package is on that species' path in that window; there is no row for a
// species QBBC does not give it, and the absence is the point.
void dump_ftfwindows() {
  FILE* f = std::fopen("ftf_windows.csv", "w");
  std::fprintf(f, "particle,pdg,baryon,charge,mass_MeV,process,subtype,model,emin_MeV,emax_MeV\n");
  const G4ParticleDefinition* all[] = {
      G4Proton::Proton(),        G4Neutron::Neutron(),      G4PionPlus::PionPlus(),
      G4PionMinus::PionMinus(),  G4KaonPlus::KaonPlus(),    G4KaonMinus::KaonMinus(),
      G4AntiProton::AntiProton(), G4AntiNeutron::AntiNeutron(),
      G4Lambda::Lambda(),        G4AntiLambda::AntiLambda(),
      G4Deuteron::Deuteron(),    G4Triton::Triton(),        G4He3::He3(),
      G4Alpha::Alpha(),          G4GenericIon::GenericIon()};
  for (const G4ParticleDefinition* pn : all) {
    G4ProcessManager* pm = pn->GetProcessManager();
    if (pm == nullptr) { continue; }
    G4ProcessVector* pv = pm->GetProcessList();
    for (G4int i = 0; i < (G4int)pv->size(); ++i) {
      G4VProcess* pr = (*pv)[i];
      G4HadronicProcess* hp = dynamic_cast<G4HadronicProcess*>(pr);
      if (hp == nullptr) { continue; }
      const std::vector<G4HadronicInteraction*>& models = hp->GetHadronicInteractionList();
      if (models.empty()) {
        std::fprintf(f, "%s,%d,%d,%d,%.17g,%s,%d,(none),0,0\n", pn->GetParticleName().c_str(),
                     pn->GetPDGEncoding(), pn->GetBaryonNumber(),
                     (G4int)pn->GetPDGCharge(), pn->GetPDGMass() / MeV,
                     pr->GetProcessName().c_str(), pr->GetProcessSubType());
        continue;
      }
      for (const G4HadronicInteraction* m : models) {
        std::fprintf(f, "%s,%d,%d,%d,%.17g,%s,%d,%s,%.17g,%.17g\n",
                     pn->GetParticleName().c_str(), pn->GetPDGEncoding(),
                     pn->GetBaryonNumber(), (G4int)pn->GetPDGCharge(), pn->GetPDGMass() / MeV,
                     pr->GetProcessName().c_str(), pr->GetProcessSubType(),
                     m->GetModelName().c_str(), m->GetMinEnergy() / MeV,
                     m->GetMaxEnergy() / MeV);
      }
    }
  }
  std::fclose(f);
}

void dump_ftf(const DumpContext&) {
  dump_params();
  dump_lund_tables();
  dump_hadrons();
  dump_minmass();
  dump_build();
  dump_samplers();
  dump_decisions();
  dump_last_states();
  dump_fragment();
  dump_fragstat();
  dump_resonance();
  dump_corrector();
  dump_string_specs();
  dump_strings();
  dump_stringstat();
  dump_splitup();
  dump_hnelastic();
  dump_excite();
  dump_annih();
  dump_create_strings();
  dump_getlist();
  dump_getlist_aa();
  dump_nucstat();
  dump_aaradius();
  dump_prescatter();
  dump_ftfwindows();
  dump_modelcases();
  dump_modelstat();
  dump_modelbig();
}

}  // namespace

G4GPU_REGISTER_DUMP("ftf",
                    "ftf_params.csv ftf_procprob.csv ftf_geom.csv ftf_lund_params.csv "
                    "ftf_lund_tables.csv ftf_hadrons.csv ftf_minmass.csv ftf_build.csv "
                    "ftf_samplers.csv ftf_decisions.csv ftf_laststates.csv ftf_fragment.csv "
                    "ftf_fragstat.csv ftf_fragstat_mult.csv ftf_resonance.csv "
                    "ftf_corrector.csv ftf_stringspec.csv ftf_strings.csv ftf_stringstat.csv "
                    "ftf_stringstat_mult.csv ftf_stringstat_balance.csv ftf_splitup.csv "
                    "ftf_hnelastic.csv ftf_excite.csv ftf_cstrings.csv ftf_nucleus.csv "
                    "ftf_getlist.csv ftf_getlist_aa.csv ftf_aanucleus.csv ftf_annih.csv "
                    "ftf_modelstat_strings.csv ftf_modelstat_species.csv "
                    "ftf_modelstat_mult.csv ftf_modelstat_wounded.csv ftf_modelcases.csv "
                    "ftf_nucstat.csv ftf_aaradius.csv ftf_prescatter.csv ftf_windows.csv "
                    "ftf_modelbig_strings.csv "
                    "ftf_modelbig_mult.csv",
                    dump_ftf);
